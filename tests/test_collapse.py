# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# tests/test_collapse.py
# =============================================================================
#
# Project      : MoEWatch
# Description  : Tests for moewatch.analyzer.collapse (expert health
#                 state machine).
#
#                 Coverage targets:
#
#                 ExpertStatus / ExpertState / LayerCollapseReport
#                   - dataclass defaults and field types
#
#                 CollapseDetector.analyze()
#                   - empty collector -> empty dict (no crash)
#                   - uniform routing -> all experts HEALTHY, severity HEALTHY
#                   - injected synthetic expert utilization with one expert
#                     starved of tokens -> COLD then DEAD after
#                     cold_steps_limit consecutive steps
#                   - COLD -> HEALTHY recovery resets consecutive_cold_steps
#                     and clears cold_onset_step
#                   - DEAD is a terminal / one-way state (no recovery even
#                     after utilization recovers)
#                   - load_imbalance_ratio passthrough from LayerStats
#                   - severity escalates HEALTHY -> DEGRADED -> CRITICAL
#                   - multiple layers analysed independently
#                   - reset() clears state for one or all layers
#                   - get_expert_state() returns None for unknown layer/expert
#                   - repeated analyze() calls are idempotent for stable input
#
# Author       : Abinesh N (@Abineshabee)
# Repository   : https://github.com/Abineshabee/MoEWatch
# License      : Apache 2.0
#
# =============================================================================

from __future__ import annotations

import time
from typing import Dict

import pytest
import torch

from moewatch.analyzer.collapse import (
    CollapseDetector,
    ExpertState,
    ExpertStatus,
    LayerCollapseReport,
)
from moewatch.collector.stat_collector import StatCollector
from moewatch.config import OutputMode, WatchConfig
from moewatch.hooks.router_hook import RoutingEvent


# ===========================================================================
# ── Local config fixture ─────────────────────────────────────────────────────
# ===========================================================================
#
# CollapseDetector derives its utilization-space "cold" threshold from the
# ratio cold_threshold / dead_threshold (see collapse.py _analyze_layer:
# cold_factor = cold_threshold / dead_threshold, capped at 10.0, then
# cold_util_threshold = (1/n_experts) * cold_factor * 0.5).
#
# For uniform routing over n_experts to classify as HEALTHY, we need
# cold_util_threshold < 1/n_experts, i.e. cold_factor < 2.0, i.e.
# cold_threshold / dead_threshold < 2.0. ``strict_config`` (from conftest)
# uses a 4x ratio (0.02/0.005), which makes *all* experts COLD even under
# perfectly uniform routing -- by design, since that fixture targets
# entropy/gradient analyzers, not CollapseDetector.
#
# This module therefore defines its own collapse-tuned config with a
# cold_steps_limit small enough to reach DEAD quickly, but a
# cold_threshold/dead_threshold ratio of 1.5 so uniform routing is HEALTHY.
# ===========================================================================


@pytest.fixture
def collapse_config() -> WatchConfig:
    """WatchConfig tuned so CollapseDetector classifies uniform routing as
    HEALTHY while still promoting a fully-starved expert to DEAD quickly."""
    return WatchConfig(
        output=OutputMode.SILENT,
        dead_threshold=0.02,
        cold_threshold=0.03,
        cold_steps_limit=3,
        entropy_warn=0.3,
        entropy_critical=0.15,
        entropy_drop_warn=0.1,
        load_imbalance_warn=2.0,
        load_imbalance_error=3.5,
        log_every=1,
        sample_every=1,
    )


# ===========================================================================
# ── Helpers ──────────────────────────────────────────────────────────────────
# ===========================================================================


def _write_utilization_step(
    collector: StatCollector,
    layer_name: str,
    n_experts: int,
    step: int,
    dead_experts: set,
    cold_experts: set,
) -> None:
    """Write one RoutingEvent producing a chosen per-expert utilization.

    Healthy experts each receive an equal share of the remaining tokens.
    ``cold_experts`` receive a single token (low utilization, but > 0).
    ``dead_experts`` receive zero tokens.

    All experts appear in ``selected_experts`` with weight proportional to
    their target token count, achieved by repeating rows.
    """
    healthy_experts = [
        e for e in range(n_experts) if e not in dead_experts and e not in cold_experts
    ]

    rows = []
    # Each healthy expert gets 10 tokens, each cold expert gets 1 token,
    # dead experts get 0 tokens.
    for e in healthy_experts:
        rows.extend([e] * 10)
    for e in cold_experts:
        rows.extend([e] * 1)

    if not rows:
        # Degenerate: everyone dead -> route a single token to expert 0
        # so the buffer isn't completely empty (still all near-zero util).
        rows = [0]

    batch_size = len(rows)
    selected = torch.tensor(rows, dtype=torch.long).unsqueeze(-1)  # [B, 1]
    logits = torch.zeros(batch_size, n_experts)

    event = RoutingEvent(
        timestamp=time.time(),
        global_step=step,
        layer_name=layer_name,
        routing_logits=logits.detach(),
        selected_experts=selected,
        expert_count=n_experts,
        batch_size=batch_size,
    )
    collector.write_routing_event(event)


# ===========================================================================
# ── Dataclass defaults ───────────────────────────────────────────────────────
# ===========================================================================


class TestExpertStatusEnum:
    def test_enum_members(self) -> None:
        assert ExpertStatus.UNKNOWN.value == "UNKNOWN"
        assert ExpertStatus.HEALTHY.value == "HEALTHY"
        assert ExpertStatus.COLD.value == "COLD"
        assert ExpertStatus.DEAD.value == "DEAD"


class TestExpertStateDefaults:
    def test_default_construction(self) -> None:
        state = ExpertState(expert_id=2)
        assert state.expert_id == 2
        assert state.status == ExpertStatus.UNKNOWN
        assert state.consecutive_cold_steps == 0
        assert state.token_count == 0
        assert state.utilization == 0.0
        assert state.cold_onset_step is None

    def test_custom_construction(self) -> None:
        state = ExpertState(
            expert_id=0,
            status=ExpertStatus.DEAD,
            consecutive_cold_steps=50,
            token_count=0,
            utilization=0.0,
            cold_onset_step=10,
        )
        assert state.status is ExpertStatus.DEAD
        assert state.consecutive_cold_steps == 50
        assert state.cold_onset_step == 10


class TestLayerCollapseReportDefaults:
    def test_default_construction(self) -> None:
        report = LayerCollapseReport(layer_name="layers.0.gate")
        assert report.layer_name == "layers.0.gate"
        assert report.expert_states == {}
        assert report.num_dead_experts == 0
        assert report.num_cold_experts == 0
        assert report.num_healthy_experts == 0
        assert report.load_imbalance_ratio == 1.0
        assert report.severity == "UNKNOWN"
        assert report.step == 0

    def test_independent_default_dicts(self) -> None:
        """Mutating one report's expert_states must not affect another."""
        r1 = LayerCollapseReport(layer_name="a")
        r2 = LayerCollapseReport(layer_name="b")
        r1.expert_states[0] = ExpertState(expert_id=0)
        assert r2.expert_states == {}


# ===========================================================================
# ── CollapseDetector: empty / guard paths ───────────────────────────────────
# ===========================================================================


class TestCollapseDetectorEmpty:
    def test_empty_collector_returns_empty_dict(
        self, default_config: WatchConfig
    ) -> None:
        detector = CollapseDetector(default_config)
        collector = StatCollector(default_config)
        reports = detector.analyze(collector)
        assert reports == {}

    def test_registered_layer_with_no_events(
        self, default_config: WatchConfig
    ) -> None:
        """A registered layer with zero routing events still produces a
        report (LayerStats default is uniform utilization with step=0)."""
        detector = CollapseDetector(default_config)
        collector = StatCollector(default_config)
        collector.register_layer("layers.0.gate", n_experts=4)

        reports = detector.analyze(collector)
        assert "layers.0.gate" in reports
        report = reports["layers.0.gate"]
        assert isinstance(report, LayerCollapseReport)
        assert report.step == 0


# ===========================================================================
# ── CollapseDetector: uniform routing -> HEALTHY ────────────────────────────
# ===========================================================================


class TestCollapseDetectorUniformHealthy:
    def test_uniform_routing_all_healthy(self, collapse_config: WatchConfig) -> None:
        detector = CollapseDetector(collapse_config)
        collector = StatCollector(collapse_config)
        n_experts = 4

        collector.register_layer("layers.0.gate", n_experts)
        for step in range(10):
            _write_utilization_step(
                collector,
                "layers.0.gate",
                n_experts=n_experts,
                step=step,
                dead_experts=set(),
                cold_experts=set(),
            )

        reports = detector.analyze(collector)
        report = reports["layers.0.gate"]

        assert report.num_dead_experts == 0
        assert report.num_cold_experts == 0
        assert report.num_healthy_experts == n_experts
        assert report.severity == "HEALTHY"
        for expert_id, state in report.expert_states.items():
            assert state.status == ExpertStatus.HEALTHY
            assert state.consecutive_cold_steps == 0
            assert state.cold_onset_step is None

    def test_load_imbalance_ratio_passthrough(
        self, default_config: WatchConfig
    ) -> None:
        """LayerCollapseReport.load_imbalance_ratio mirrors LayerStats value."""
        detector = CollapseDetector(default_config)
        collector = StatCollector(default_config)
        n_experts = 4

        collector.register_layer("layers.0.gate", n_experts)
        for step in range(5):
            _write_utilization_step(
                collector,
                "layers.0.gate",
                n_experts=n_experts,
                step=step,
                dead_experts=set(),
                cold_experts=set(),
            )

        reports = detector.analyze(collector)
        stats = collector.get_layer_stats("layers.0.gate")
        assert reports["layers.0.gate"].load_imbalance_ratio == pytest.approx(
            stats.load_imbalance_ratio
        )


# ===========================================================================
# ── CollapseDetector: COLD -> DEAD promotion ─────────────────────────────────
# ===========================================================================


class TestCollapseDetectorColdToDead:
    def test_cold_expert_promoted_to_dead_after_limit(
        self, collapse_config: WatchConfig
    ) -> None:
        """Expert 3 receives zero tokens every step; after
        ``cold_steps_limit`` consecutive analyze() calls it must be DEAD."""
        detector = CollapseDetector(collapse_config)
        collector = StatCollector(collapse_config)
        n_experts = 4
        starved_expert = 3
        limit = collapse_config.cold_steps_limit

        collector.register_layer("layers.0.gate", n_experts)

        last_report: LayerCollapseReport
        for step in range(limit + 2):
            _write_utilization_step(
                collector,
                "layers.0.gate",
                n_experts=n_experts,
                step=step,
                dead_experts={starved_expert},
                cold_experts=set(),
            )
            reports = detector.analyze(collector)
            last_report = reports["layers.0.gate"]

        starved_state = last_report.expert_states[starved_expert]
        assert starved_state.status == ExpertStatus.DEAD
        assert starved_state.consecutive_cold_steps >= limit
        assert last_report.num_dead_experts == 1
        assert last_report.severity == "CRITICAL"

    def test_cold_step_counter_increments_monotonically(
        self, collapse_config: WatchConfig
    ) -> None:
        """Before crossing cold_steps_limit, consecutive_cold_steps grows by
        1 each analyze() call for a continuously-starved expert."""
        detector = CollapseDetector(collapse_config)
        collector = StatCollector(collapse_config)
        n_experts = 4
        starved_expert = 3
        limit = collapse_config.cold_steps_limit

        collector.register_layer("layers.0.gate", n_experts)

        counters = []
        for step in range(limit - 1):
            _write_utilization_step(
                collector,
                "layers.0.gate",
                n_experts=n_experts,
                step=step,
                dead_experts={starved_expert},
                cold_experts=set(),
            )
            reports = detector.analyze(collector)
            state = reports["layers.0.gate"].expert_states[starved_expert]
            counters.append(state.consecutive_cold_steps)
            assert state.status in (ExpertStatus.COLD, ExpertStatus.UNKNOWN)
            assert state.status != ExpertStatus.DEAD

        # Counter should be strictly increasing 1, 2, 3, ...
        assert counters == sorted(counters)
        assert counters == list(range(1, limit))

    def test_cold_onset_step_recorded_once(
        self, collapse_config: WatchConfig
    ) -> None:
        """cold_onset_step is set on first COLD step and not overwritten on
        subsequent COLD steps."""
        detector = CollapseDetector(collapse_config)
        collector = StatCollector(collapse_config)
        n_experts = 4
        starved_expert = 3

        collector.register_layer("layers.0.gate", n_experts)

        onset_steps = []
        for step in range(3):
            _write_utilization_step(
                collector,
                "layers.0.gate",
                n_experts=n_experts,
                step=step,
                dead_experts={starved_expert},
                cold_experts=set(),
            )
            reports = detector.analyze(collector)
            state = reports["layers.0.gate"].expert_states[starved_expert]
            onset_steps.append(state.cold_onset_step)

        # Onset step recorded at the first analysis call and unchanged after.
        assert onset_steps[0] == onset_steps[1] == onset_steps[2]
        assert onset_steps[0] is not None


# ===========================================================================
# ── CollapseDetector: COLD -> HEALTHY recovery ───────────────────────────────
# ===========================================================================


class TestCollapseDetectorRecovery:
    def test_cold_expert_recovers_to_healthy(self) -> None:
        """An expert that goes COLD and then receives a healthy token share
        over enough subsequent steps eventually crosses back over the cold
        utilization threshold and returns to HEALTHY, with its cold counter
        and onset step reset.

        Note: LayerStats.expert_utilization is computed over a rolling
        window of *all* buffered routing events (oldest-evicted), so
        recovery is gradual rather than instantaneous -- a single
        "recovered" step does not immediately overwrite the accumulated
        history from the starved steps. A generous ``cold_steps_limit``
        is used here so the gradual recovery completes before DEAD
        promotion would otherwise occur.
        """
        config = WatchConfig(
            output=OutputMode.SILENT,
            dead_threshold=0.02,
            cold_threshold=0.03,
            cold_steps_limit=10,
            entropy_warn=0.3,
            entropy_critical=0.15,
            entropy_drop_warn=0.1,
            load_imbalance_warn=2.0,
            load_imbalance_error=3.5,
            log_every=1,
            sample_every=1,
        )
        detector = CollapseDetector(config)
        collector = StatCollector(config)
        n_experts = 4
        recovering_expert = 3

        collector.register_layer("layers.0.gate", n_experts)

        # Phase 1: expert 3 goes cold for 2 steps (below limit).
        for step in range(2):
            _write_utilization_step(
                collector,
                "layers.0.gate",
                n_experts=n_experts,
                step=step,
                dead_experts={recovering_expert},
                cold_experts=set(),
            )
            reports = detector.analyze(collector)

        cold_state = reports["layers.0.gate"].expert_states[recovering_expert]
        assert cold_state.status == ExpertStatus.COLD
        assert cold_state.consecutive_cold_steps == 2
        assert cold_state.cold_onset_step is not None

        # Phase 2: expert 3 receives a full healthy share every step.
        # Rolling utilization gradually rises until it crosses the cold
        # threshold, at which point the state machine transitions to
        # HEALTHY and resets its counters.
        recovered = False
        for step in range(2, 2 + 6):
            _write_utilization_step(
                collector,
                "layers.0.gate",
                n_experts=n_experts,
                step=step,
                dead_experts=set(),
                cold_experts=set(),
            )
            reports = detector.analyze(collector)
            state = reports["layers.0.gate"].expert_states[recovering_expert]
            if state.status == ExpertStatus.HEALTHY:
                recovered = True
                break

        assert recovered, "Expert 3 never recovered to HEALTHY"
        assert state.consecutive_cold_steps == 0
        assert state.cold_onset_step is None


# ===========================================================================
# ── CollapseDetector: DEAD is terminal ───────────────────────────────────────
# ===========================================================================


class TestCollapseDetectorDeadIsTerminal:
    def test_dead_expert_does_not_recover(
        self, collapse_config: WatchConfig
    ) -> None:
        """Once an expert is promoted to DEAD, restoring its utilization to
        a healthy share must NOT transition it back to HEALTHY (one-way
        state machine progression)."""
        detector = CollapseDetector(collapse_config)
        collector = StatCollector(collapse_config)
        n_experts = 4
        dying_expert = 3
        limit = collapse_config.cold_steps_limit

        collector.register_layer("layers.0.gate", n_experts)

        # Drive expert 3 to DEAD.
        for step in range(limit + 1):
            _write_utilization_step(
                collector,
                "layers.0.gate",
                n_experts=n_experts,
                step=step,
                dead_experts={dying_expert},
                cold_experts=set(),
            )
            reports = detector.analyze(collector)

        assert reports["layers.0.gate"].expert_states[dying_expert].status == (
            ExpertStatus.DEAD
        )

        # Now "restore" expert 3 to a healthy share of tokens.
        for step in range(limit + 1, limit + 4):
            _write_utilization_step(
                collector,
                "layers.0.gate",
                n_experts=n_experts,
                step=step,
                dead_experts=set(),
                cold_experts=set(),
            )
            reports = detector.analyze(collector)

        # Still DEAD — terminal state, no recovery.
        final_state = reports["layers.0.gate"].expert_states[dying_expert]
        assert final_state.status == ExpertStatus.DEAD


# ===========================================================================
# ── CollapseDetector: severity classification ────────────────────────────────
# ===========================================================================


class TestCollapseDetectorSeverity:
    def test_severity_healthy_when_all_balanced(
        self, collapse_config: WatchConfig
    ) -> None:
        detector = CollapseDetector(collapse_config)
        collector = StatCollector(collapse_config)
        n_experts = 4

        collector.register_layer("layers.0.gate", n_experts)
        for step in range(5):
            _write_utilization_step(
                collector,
                "layers.0.gate",
                n_experts=n_experts,
                step=step,
                dead_experts=set(),
                cold_experts=set(),
            )

        reports = detector.analyze(collector)
        assert reports["layers.0.gate"].severity == "HEALTHY"

    def test_severity_critical_with_dead_expert(
        self, collapse_config: WatchConfig
    ) -> None:
        detector = CollapseDetector(collapse_config)
        collector = StatCollector(collapse_config)
        n_experts = 4
        limit = collapse_config.cold_steps_limit

        collector.register_layer("layers.0.gate", n_experts)
        for step in range(limit + 1):
            _write_utilization_step(
                collector,
                "layers.0.gate",
                n_experts=n_experts,
                step=step,
                dead_experts={3},
                cold_experts=set(),
            )
            reports = detector.analyze(collector)

        assert reports["layers.0.gate"].severity == "CRITICAL"

    def test_severity_degraded_with_cold_but_no_dead(
        self, collapse_config: WatchConfig
    ) -> None:
        """A single cold (but not yet dead) expert with low cold_fraction
        should classify as DEGRADED, not CRITICAL."""
        detector = CollapseDetector(collapse_config)
        collector = StatCollector(collapse_config)
        n_experts = 8  # cold_fraction = 1/8 = 0.125, below the 0.5 critical
        # cutoff but > 0.1 degraded cutoff -> DEGRADED via num_cold > 0.

        collector.register_layer("layers.0.gate", n_experts)
        for step in range(2):
            _write_utilization_step(
                collector,
                "layers.0.gate",
                n_experts=n_experts,
                step=step,
                dead_experts=set(),
                cold_experts={7},
            )
            reports = detector.analyze(collector)

        report = reports["layers.0.gate"]
        assert report.num_cold_experts >= 1
        assert report.num_dead_experts == 0
        assert report.severity == "DEGRADED"


# ===========================================================================
# ── CollapseDetector: multi-layer independence ───────────────────────────────
# ===========================================================================


class TestCollapseDetectorMultiLayer:
    def test_layers_analysed_independently(
        self, collapse_config: WatchConfig
    ) -> None:
        detector = CollapseDetector(collapse_config)
        collector = StatCollector(collapse_config)
        n_experts = 4
        limit = collapse_config.cold_steps_limit

        collector.register_layer("layers.0.gate", n_experts)
        collector.register_layer("layers.1.gate", n_experts)

        for step in range(limit + 1):
            # Layer 0: fully healthy.
            _write_utilization_step(
                collector,
                "layers.0.gate",
                n_experts=n_experts,
                step=step,
                dead_experts=set(),
                cold_experts=set(),
            )
            # Layer 1: expert 2 starved -> eventually DEAD.
            _write_utilization_step(
                collector,
                "layers.1.gate",
                n_experts=n_experts,
                step=step,
                dead_experts={2},
                cold_experts=set(),
            )
            reports = detector.analyze(collector)

        assert reports["layers.0.gate"].severity == "HEALTHY"
        assert reports["layers.1.gate"].severity == "CRITICAL"
        assert reports["layers.0.gate"].num_dead_experts == 0
        assert reports["layers.1.gate"].num_dead_experts == 1


# ===========================================================================
# ── CollapseDetector: reset() and get_expert_state() ─────────────────────────
# ===========================================================================


class TestCollapseDetectorMaintenance:
    def test_reset_single_layer(self, collapse_config: WatchConfig) -> None:
        detector = CollapseDetector(collapse_config)
        collector = StatCollector(collapse_config)
        n_experts = 4

        collector.register_layer("layers.0.gate", n_experts)
        collector.register_layer("layers.1.gate", n_experts)
        for step in range(3):
            _write_utilization_step(
                collector, "layers.0.gate", n_experts, step, set(), set()
            )
            _write_utilization_step(
                collector, "layers.1.gate", n_experts, step, set(), set()
            )
        detector.analyze(collector)

        detector.reset("layers.0.gate")

        assert detector.get_expert_state("layers.0.gate", 0) is None
        assert detector.get_expert_state("layers.1.gate", 0) is not None

    def test_reset_all_layers(self, collapse_config: WatchConfig) -> None:
        detector = CollapseDetector(collapse_config)
        collector = StatCollector(collapse_config)
        n_experts = 4

        collector.register_layer("layers.0.gate", n_experts)
        for step in range(3):
            _write_utilization_step(
                collector, "layers.0.gate", n_experts, step, set(), set()
            )
        detector.analyze(collector)

        detector.reset()

        assert detector.get_expert_state("layers.0.gate", 0) is None

    def test_get_expert_state_unknown_layer(
        self, default_config: WatchConfig
    ) -> None:
        detector = CollapseDetector(default_config)
        assert detector.get_expert_state("nonexistent.layer", 0) is None

    def test_get_expert_state_unknown_expert(
        self, collapse_config: WatchConfig
    ) -> None:
        detector = CollapseDetector(collapse_config)
        collector = StatCollector(collapse_config)
        n_experts = 4

        collector.register_layer("layers.0.gate", n_experts)
        _write_utilization_step(
            collector, "layers.0.gate", n_experts, 0, set(), set()
        )
        detector.analyze(collector)

        assert detector.get_expert_state("layers.0.gate", 999) is None


# ===========================================================================
# ── CollapseDetector: idempotence on stable input ────────────────────────────
# ===========================================================================


class TestCollapseDetectorIdempotence:
    def test_repeated_analyze_on_stable_input_is_stable(
        self, collapse_config: WatchConfig
    ) -> None:
        """Calling analyze() repeatedly with the same stable healthy
        utilization should keep producing HEALTHY severity and zero
        consecutive_cold_steps for all experts."""
        detector = CollapseDetector(collapse_config)
        collector = StatCollector(collapse_config)
        n_experts = 4

        collector.register_layer("layers.0.gate", n_experts)
        for step in range(10):
            _write_utilization_step(
                collector, "layers.0.gate", n_experts, step, set(), set()
            )
            reports = detector.analyze(collector)
            report = reports["layers.0.gate"]
            assert report.severity == "HEALTHY"
            for state in report.expert_states.values():
                assert state.consecutive_cold_steps == 0


# ===========================================================================
# ── CollapseDetector: __repr__ ───────────────────────────────────────────────
# ===========================================================================


class TestCollapseDetectorRepr:
    def test_repr_contains_key_info(self, default_config: WatchConfig) -> None:
        detector = CollapseDetector(default_config)
        text = repr(detector)
        assert "CollapseDetector" in text
        assert "cold_steps_limit" in text
