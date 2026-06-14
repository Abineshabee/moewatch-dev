# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# tests/test_gradient_starvation.py
# =============================================================================
#
# Project      : MoEWatch
# Description  : Tests for moewatch.analyzer.gradient_starvation
#                 (Tier 1 signal: GradientStarvationAnalyzer).
#
#                 Coverage targets:
#
#                 GradientStarvationReport
#                   - dataclass defaults
#
#                 GradientStarvationAnalyzer.analyze()
#                   - empty collector -> empty dict (no crash)
#                   - layer with no gradient events -> empty list
#                   - insufficient samples (< _MIN_SAMPLES_FOR_DETECTION)
#                     -> default report, n_samples reflects actual count
#                   - healthy gradient norms (>= cold_threshold) ->
#                     starvation_score == 0, starvation_detected == False
#                   - starved gradient norms (< cold_threshold) ->
#                     starvation_score in (0, 1], counters increment
#                   - starvation_detected becomes True only after
#                     cold_steps_limit consecutive starved analyze() calls
#                   - starvation_onset_step recorded once and persisted
#                     across calls while starving
#                   - recovery (norm rises above cold_threshold) resets
#                     counter and clears onset step
#                   - gradient_norm_mean / gradient_norm_std match the
#                     manually-computed population statistics
#                   - NaN / Inf gradient norms handled without crashing
#                   - multiple experts within one layer tracked
#                     independently, ordered by ascending expert_id
#                   - multiple layers analysed independently
#                   - reset() clears state for one or all layers
#                   - get_starvation_count() / is_layer_registered()
#                   - repr() contains key configuration info
#
# Author       : Abinesh N (@Abineshabee)
# Repository   : https://github.com/Abineshabee/MoEWatch
# License      : Apache 2.0
#
# =============================================================================

from __future__ import annotations

import math

import pytest

from moewatch.analyzer.gradient_starvation import (
    GradientStarvationAnalyzer,
    GradientStarvationReport,
)
from moewatch.collector.stat_collector import StatCollector
from moewatch.config import OutputMode, WatchConfig

from conftest import make_gradient_event, make_routing_event


# ===========================================================================
# ── Helper ──────────────────────────────────────────────────────────────────
# ===========================================================================


def _feed_gradients(
    collector: StatCollector,
    layer_name: str,
    expert_id: int,
    norms: list,
    start_step: int = 0,
) -> None:
    """Write a sequence of GradientEvents for one (layer, expert) pair."""
    for i, norm in enumerate(norms):
        collector.write_gradient_event(
            make_gradient_event(
                layer_name=layer_name,
                expert_id=expert_id,
                gradient_norm=norm,
                global_step=start_step + i,
            )
        )


# ===========================================================================
# ── Dataclass defaults ───────────────────────────────────────────────────────
# ===========================================================================


class TestGradientStarvationReportDefaults:
    def test_default_construction(self) -> None:
        report = GradientStarvationReport(layer_name="layers.0.experts", expert_id=2)
        assert report.layer_name == "layers.0.experts"
        assert report.expert_id == 2
        assert report.gradient_norm_mean == 0.0
        assert report.gradient_norm_std == 0.0
        assert report.starvation_score == 0.0
        assert report.starvation_detected is False
        assert report.starvation_onset_step is None
        assert report.step == 0
        assert report.n_samples == 0


# ===========================================================================
# ── analyze(): empty / guard paths ───────────────────────────────────────────
# ===========================================================================


class TestGradientStarvationAnalyzerEmpty:
    def test_empty_collector_returns_empty_dict(
        self, default_config: WatchConfig
    ) -> None:
        analyzer = GradientStarvationAnalyzer(default_config)
        collector = StatCollector(default_config)
        reports = analyzer.analyze(collector)
        assert reports == {}

    def test_layer_with_routing_only_no_gradients(
        self, default_config: WatchConfig
    ) -> None:
        """A layer registered only via routing events (no gradient events
        written) produces an empty per-expert gradient stats dict, and
        therefore an empty (or absent) entry in the gradient report."""
        analyzer = GradientStarvationAnalyzer(default_config)
        collector = StatCollector(default_config)
        collector.register_layer("layers.0.gate", n_experts=4)
        collector.write_routing_event(
            make_routing_event(layer_name="layers.0.gate", n_experts=4, global_step=0)
        )

        reports = analyzer.analyze(collector)

        # register_layer also creates empty gradient buffers for each
        # expert index, so "layers.0.gate" appears with default reports
        # (n_samples == 0) rather than being entirely absent.
        if "layers.0.gate" in reports:
            for r in reports["layers.0.gate"]:
                assert r.n_samples == 0
                assert r.starvation_detected is False


# ===========================================================================
# ── Insufficient samples ─────────────────────────────────────────────────────
# ===========================================================================


class TestGradientStarvationInsufficientSamples:
    def test_below_min_samples_returns_default_report(
        self, default_config: WatchConfig
    ) -> None:
        analyzer = GradientStarvationAnalyzer(default_config)
        collector = StatCollector(default_config)

        # Only 2 samples -- below _MIN_SAMPLES_FOR_DETECTION (3).
        _feed_gradients(collector, "layers.0.experts", 0, [0.15, 0.15])

        reports = analyzer.analyze(collector)
        layer_reports = reports["layers.0.experts"]
        report = next(r for r in layer_reports if r.expert_id == 0)

        assert report.n_samples == 2
        assert report.gradient_norm_mean == 0.0
        assert report.starvation_score == 0.0
        assert report.starvation_detected is False
        assert report.starvation_onset_step is None


# ===========================================================================
# ── Healthy gradients ────────────────────────────────────────────────────────
# ===========================================================================


class TestGradientStarvationHealthy:
    def test_healthy_norms_zero_starvation_score(
        self, default_config: WatchConfig
    ) -> None:
        """Gradient norms well above cold_threshold (0.05 default) produce
        starvation_score == 0.0 and starvation_detected == False."""
        analyzer = GradientStarvationAnalyzer(default_config)
        collector = StatCollector(default_config)

        norms = [0.5] * 10
        _feed_gradients(collector, "layers.0.experts", 0, norms)

        reports = analyzer.analyze(collector)
        report = next(
            r for r in reports["layers.0.experts"] if r.expert_id == 0
        )

        assert report.starvation_score == 0.0
        assert report.starvation_detected is False
        assert report.starvation_onset_step is None
        assert report.gradient_norm_mean == pytest.approx(0.5)
        assert report.gradient_norm_std == pytest.approx(0.0, abs=1e-9)
        assert report.n_samples == 10


# ===========================================================================
# ── Starved gradients ────────────────────────────────────────────────────────
# ===========================================================================


class TestGradientStarvationStarved:
    def test_starved_norms_positive_score(self, default_config: WatchConfig) -> None:
        """Gradient norms below cold_threshold produce starvation_score > 0."""
        analyzer = GradientStarvationAnalyzer(default_config)
        collector = StatCollector(default_config)

        # cold_threshold default 0.05; use a much smaller norm.
        norms = [0.001] * 5
        _feed_gradients(collector, "layers.0.experts", 0, norms)

        reports = analyzer.analyze(collector)
        report = next(r for r in reports["layers.0.experts"] if r.expert_id == 0)

        expected_score = max(
            0.0, 1.0 - report.gradient_norm_mean / default_config.cold_threshold
        )
        assert report.starvation_score == pytest.approx(expected_score)
        assert 0.0 < report.starvation_score <= 1.0

    def test_zero_gradient_norm_gives_score_one(
        self, default_config: WatchConfig
    ) -> None:
        analyzer = GradientStarvationAnalyzer(default_config)
        collector = StatCollector(default_config)

        norms = [0.0] * 5
        _feed_gradients(collector, "layers.0.experts", 0, norms)

        reports = analyzer.analyze(collector)
        report = next(r for r in reports["layers.0.experts"] if r.expert_id == 0)

        assert report.starvation_score == pytest.approx(1.0)


# ===========================================================================
# ── starvation_detected: cold_steps_limit promotion ──────────────────────────
# ===========================================================================


class TestGradientStarvationDetection:
    def test_starvation_detected_after_cold_steps_limit(
        self, strict_config: WatchConfig
    ) -> None:
        """starvation_detected becomes True only once
        consecutive_cold_steps >= config.cold_steps_limit, where the
        counter increments once per analyze() call -- but only once the
        expert has accumulated >= _MIN_SAMPLES_FOR_DETECTION (3) gradient
        samples (below that, a default report with no counter update is
        returned)."""
        analyzer = GradientStarvationAnalyzer(strict_config)
        collector = StatCollector(strict_config)
        limit = strict_config.cold_steps_limit
        min_samples = 3  # GradientStarvationAnalyzer._MIN_SAMPLES_FOR_DETECTION

        # Norms well below strict_config.cold_threshold (0.02).
        starved_norm = 0.001

        total_calls = (min_samples - 1) + limit
        last_report = None
        for call_idx in range(total_calls):
            collector.write_gradient_event(
                make_gradient_event(
                    layer_name="layers.0.experts",
                    expert_id=0,
                    gradient_norm=starved_norm,
                    global_step=call_idx,
                )
            )
            reports = analyzer.analyze(collector)
            last_report = next(
                r for r in reports["layers.0.experts"] if r.expert_id == 0
            )

            n_below_limit_calls = (min_samples - 1) + (limit - 1)
            if call_idx + 1 <= n_below_limit_calls:
                assert last_report.starvation_detected is False

        assert last_report.starvation_detected is True
        assert last_report.starvation_score > 0.0

    def test_starvation_onset_step_recorded_once(
        self, strict_config: WatchConfig
    ) -> None:
        analyzer = GradientStarvationAnalyzer(strict_config)
        collector = StatCollector(strict_config)

        onset_steps = []
        # First 2 calls have n_samples < _MIN_SAMPLES_FOR_DETECTION (3) and
        # return default reports (onset_step=None). From call index 2
        # onward (n_samples=3), the onset step is recorded.
        for step in range(5):
            collector.write_gradient_event(
                make_gradient_event(
                    layer_name="layers.0.experts",
                    expert_id=0,
                    gradient_norm=0.001,
                    global_step=step,
                )
            )
            reports = analyzer.analyze(collector)
            report = next(
                r for r in reports["layers.0.experts"] if r.expert_id == 0
            )
            onset_steps.append(report.starvation_onset_step)

        # The first two reports (n_samples < 3) have no onset recorded.
        assert onset_steps[0] is None
        assert onset_steps[1] is None

        # From the third report onward, onset is recorded and stable.
        recorded = onset_steps[2:]
        assert all(s is not None for s in recorded)
        assert all(s == recorded[0] for s in recorded)


# ===========================================================================
# ── Recovery resets counters ─────────────────────────────────────────────────
# ===========================================================================


class TestGradientStarvationRecovery:
    def test_recovery_resets_counter_and_onset(
        self, strict_config: WatchConfig
    ) -> None:
        analyzer = GradientStarvationAnalyzer(strict_config)
        collector = StatCollector(strict_config)

        # Phase 1: starve for 2 calls (below limit, but enough to set onset
        # once the rolling mean drops below threshold).
        for step in range(2):
            collector.write_gradient_event(
                make_gradient_event(
                    layer_name="layers.0.experts",
                    expert_id=0,
                    gradient_norm=0.001,
                    global_step=step,
                )
            )
            reports = analyzer.analyze(collector)

        report = next(r for r in reports["layers.0.experts"] if r.expert_id == 0)
        assert report.starvation_onset_step is not None
        cold_count_before = analyzer.get_starvation_count("layers.0.experts", 0)
        assert cold_count_before > 0

        # Phase 2: recover with healthy gradient norms. Because
        # GradientStats.gradient_norm_mean is computed over a rolling
        # window of *all* buffered samples, enough healthy samples must be
        # written for the mean to rise back above cold_threshold.
        recovered = False
        for step in range(2, 2 + 50):
            collector.write_gradient_event(
                make_gradient_event(
                    layer_name="layers.0.experts",
                    expert_id=0,
                    gradient_norm=0.5,
                    global_step=step,
                )
            )
            reports = analyzer.analyze(collector)
            report = next(r for r in reports["layers.0.experts"] if r.expert_id == 0)
            if report.gradient_norm_mean >= strict_config.cold_threshold:
                recovered = True
                break

        assert recovered, "Mean gradient norm never recovered above cold_threshold"
        assert report.starvation_score == 0.0
        assert report.starvation_onset_step is None
        assert analyzer.get_starvation_count("layers.0.experts", 0) == 0


# ===========================================================================
# ── Statistics correctness ───────────────────────────────────────────────────
# ===========================================================================


class TestGradientStarvationStatistics:
    def test_mean_and_std_match_manual_computation(
        self, default_config: WatchConfig
    ) -> None:
        analyzer = GradientStarvationAnalyzer(default_config)
        collector = StatCollector(default_config)

        norms = [0.1, 0.2, 0.3, 0.4, 0.5]
        _feed_gradients(collector, "layers.0.experts", 0, norms)

        reports = analyzer.analyze(collector)
        report = next(r for r in reports["layers.0.experts"] if r.expert_id == 0)

        expected_mean = sum(norms) / len(norms)
        expected_var = sum((x - expected_mean) ** 2 for x in norms) / len(norms)
        expected_std = math.sqrt(expected_var)

        assert report.gradient_norm_mean == pytest.approx(expected_mean)
        assert report.gradient_norm_std == pytest.approx(expected_std)

    def test_step_reflects_most_recent_event(
        self, default_config: WatchConfig
    ) -> None:
        analyzer = GradientStarvationAnalyzer(default_config)
        collector = StatCollector(default_config)

        _feed_gradients(
            collector, "layers.0.experts", 0, [0.1, 0.2, 0.3], start_step=10
        )

        reports = analyzer.analyze(collector)
        report = next(r for r in reports["layers.0.experts"] if r.expert_id == 0)

        assert report.step == 12  # start_step=10 + 2 (0-indexed last)


# ===========================================================================
# ── Non-finite gradient norms ────────────────────────────────────────────────
# ===========================================================================


class TestGradientStarvationNonFinite:
    def test_nan_and_inf_norms_do_not_crash(
        self, default_config: WatchConfig
    ) -> None:
        analyzer = GradientStarvationAnalyzer(default_config)
        collector = StatCollector(default_config)

        norms = [0.1, float("nan"), 0.2, float("inf"), 0.3]
        _feed_gradients(collector, "layers.0.experts", 0, norms)

        reports = analyzer.analyze(collector)
        report = next(r for r in reports["layers.0.experts"] if r.expert_id == 0)

        # Should not raise, and resulting stats should be finite.
        assert math.isfinite(report.gradient_norm_mean)
        assert math.isfinite(report.gradient_norm_std)
        assert math.isfinite(report.starvation_score)

    def test_all_non_finite_norms_return_default(
        self, default_config: WatchConfig
    ) -> None:
        analyzer = GradientStarvationAnalyzer(default_config)
        collector = StatCollector(default_config)

        norms = [float("nan"), float("inf"), float("-inf")]
        _feed_gradients(collector, "layers.0.experts", 0, norms)

        reports = analyzer.analyze(collector)
        report = next(r for r in reports["layers.0.experts"] if r.expert_id == 0)

        assert report.gradient_norm_mean == 0.0
        assert report.starvation_score == 0.0
        assert report.starvation_detected is False


# ===========================================================================
# ── Multiple experts / multiple layers ───────────────────────────────────────
# ===========================================================================


class TestGradientStarvationMultiExpertMultiLayer:
    def test_multiple_experts_tracked_independently_ordered(
        self, default_config: WatchConfig
    ) -> None:
        analyzer = GradientStarvationAnalyzer(default_config)
        collector = StatCollector(default_config)

        _feed_gradients(collector, "layers.0.experts", 0, [0.5] * 5)
        _feed_gradients(collector, "layers.0.experts", 1, [0.001] * 5)
        _feed_gradients(collector, "layers.0.experts", 2, [0.5] * 5)

        reports = analyzer.analyze(collector)
        layer_reports = reports["layers.0.experts"]

        # Ordered by ascending expert_id.
        expert_ids = [r.expert_id for r in layer_reports]
        assert expert_ids == sorted(expert_ids)

        by_id = {r.expert_id: r for r in layer_reports}
        assert by_id[0].starvation_score == 0.0
        assert by_id[1].starvation_score > 0.0
        assert by_id[2].starvation_score == 0.0

    def test_multiple_layers_independent(self, default_config: WatchConfig) -> None:
        analyzer = GradientStarvationAnalyzer(default_config)
        collector = StatCollector(default_config)

        _feed_gradients(collector, "layers.0.experts", 0, [0.5] * 5)
        _feed_gradients(collector, "layers.1.experts", 0, [0.001] * 5)

        reports = analyzer.analyze(collector)

        r0 = next(r for r in reports["layers.0.experts"] if r.expert_id == 0)
        r1 = next(r for r in reports["layers.1.experts"] if r.expert_id == 0)

        assert r0.starvation_score == 0.0
        assert r1.starvation_score > 0.0


# ===========================================================================
# ── Maintenance: reset(), get_starvation_count(), is_layer_registered() ─────
# ===========================================================================


class TestGradientStarvationMaintenance:
    def test_reset_single_layer(self, strict_config: WatchConfig) -> None:
        analyzer = GradientStarvationAnalyzer(strict_config)
        collector = StatCollector(strict_config)

        _feed_gradients(collector, "layers.0.experts", 0, [0.001] * 5)
        _feed_gradients(collector, "layers.1.experts", 0, [0.001] * 5)
        analyzer.analyze(collector)

        assert analyzer.is_layer_registered("layers.0.experts")
        assert analyzer.is_layer_registered("layers.1.experts")

        analyzer.reset("layers.0.experts")

        assert not analyzer.is_layer_registered("layers.0.experts")
        assert analyzer.is_layer_registered("layers.1.experts")

    def test_reset_all_layers(self, strict_config: WatchConfig) -> None:
        analyzer = GradientStarvationAnalyzer(strict_config)
        collector = StatCollector(strict_config)

        _feed_gradients(collector, "layers.0.experts", 0, [0.001] * 5)
        analyzer.analyze(collector)
        assert analyzer.is_layer_registered("layers.0.experts")

        analyzer.reset()

        assert not analyzer.is_layer_registered("layers.0.experts")

    def test_get_starvation_count_unknown_returns_zero(
        self, default_config: WatchConfig
    ) -> None:
        analyzer = GradientStarvationAnalyzer(default_config)
        assert analyzer.get_starvation_count("nonexistent", 0) == 0

    def test_is_layer_registered_false_before_analyze(
        self, default_config: WatchConfig
    ) -> None:
        analyzer = GradientStarvationAnalyzer(default_config)
        assert analyzer.is_layer_registered("layers.0.experts") is False


# ===========================================================================
# ── repr() ───────────────────────────────────────────────────────────────────
# ===========================================================================


class TestGradientStarvationRepr:
    def test_repr_contains_key_info(self, default_config: WatchConfig) -> None:
        analyzer = GradientStarvationAnalyzer(default_config)
        text = repr(analyzer)
        assert "GradientStarvationAnalyzer" in text
        assert "cold_threshold" in text
        assert "cold_steps_limit" in text
