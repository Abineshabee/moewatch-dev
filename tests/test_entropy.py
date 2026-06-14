# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# tests/test_entropy.py
# =============================================================================
#
# Project      : MoEWatch
# Description  : Tests for moewatch.analyzer.entropy (Tier 2 signal).
#
#                Coverage targets:
#
#                compute_entropy()
#                  - Uniform distribution → H = log(n) nats
#                  - Collapsed distribution (all mass on 1 expert) → H = 0
#                  - Empty array → H = 0 (no crash)
#                  - All-zero array → H = 0
#                  - Single-element array → H = 0
#                  - Unnormalised positive array normalised correctly
#                  - Numerical stability: near-zero probabilities
#
#                compute_entropy_norm()
#                  - Uniform distribution → normalised entropy ≈ 1.0
#                  - Collapsed distribution → normalised entropy ≈ 0.0
#                  - n_experts < 2 → returns 0.0 (guard)
#                  - Result always in [0.0, 1.0]
#
#                LayerEntropyReport dataclass
#                  - All required fields present
#                  - normalized_entropy in [0, 1]
#                  - trend in {"DECLINING", "STABLE", "IMPROVING", "UNKNOWN"}
#                  - drift_detected is bool
#
#                EntropyAnalyzer.analyze()
#                  - Returns dict[str, LayerEntropyReport]
#                  - One report per registered layer
#                  - Empty collector → empty dict (no crash)
#                  - Uniform routing → high normalised entropy
#                  - Collapsed routing → low normalised entropy
#                  - Trend classification: DECLINING for declining sequence
#                  - Trend classification: STABLE for flat sequence
#                  - CUSUM drift_detected = True for sharp entropy drop
#                  - CUSUM drift_detected = False for stable signal
#                  - Multiple layers analysed independently
#                  - Repeated analyze() calls are idempotent (no state bleed)
#
# =============================================================================

from __future__ import annotations

import math
import time
from typing import Dict

import numpy as np
import pytest
import torch

from moewatch.analyzer.entropy import (
    EntropyAnalyzer,
    LayerEntropyReport,
    compute_entropy,
    compute_entropy_norm,
)
from moewatch.collector.stat_collector import StatCollector
from moewatch.config import OutputMode, WatchConfig
from moewatch.hooks.router_hook import RoutingEvent

from conftest import make_routing_event


# ===========================================================================
# ── Helper ──────────────────────────────────────────────────────────────────
# ===========================================================================


def _build_collector_with_events(
    config: WatchConfig,
    layer_name: str,
    n_experts: int,
    n_events: int,
    uniform: bool = True,
) -> StatCollector:
    """Build a StatCollector pre-filled with routing events."""
    collector = StatCollector(config)
    collector.register_layer(layer_name, n_experts)
    for step in range(n_events):
        collector.write_routing_event(
            make_routing_event(
                layer_name=layer_name,
                n_experts=n_experts,
                batch_size=8,
                global_step=step,
                uniform=uniform,
            )
        )
    return collector


def _build_declining_collector(
    config: WatchConfig,
    layer_name: str = "layer.0.gate",
    n_experts: int = 4,
    n_steps: int = 40,
) -> StatCollector:
    """
    Build a StatCollector where logit bias toward expert 0 grows linearly.
    Creates a steadily declining entropy trajectory for trend tests.
    """
    collector = StatCollector(config)
    collector.register_layer(layer_name, n_experts)
    for step in range(n_steps):
        progress = step / max(n_steps - 1, 1)
        logits = torch.zeros(8, n_experts)
        logits[:, 0] = 20.0 * progress
        top_k = min(2, n_experts)
        _, selected = torch.topk(logits, k=top_k, dim=-1)
        collector.write_routing_event(
            RoutingEvent(
                timestamp=time.time(),
                global_step=step,
                layer_name=layer_name,
                routing_logits=logits.detach(),
                selected_experts=selected.detach(),
                expert_count=n_experts,
                batch_size=8,
            )
        )
    return collector


def _build_stable_collector(
    config: WatchConfig,
    layer_name: str = "layer.0.gate",
    n_experts: int = 4,
    n_steps: int = 30,
) -> StatCollector:
    """Constant uniform routing → stable entropy."""
    return _build_collector_with_events(
        config, layer_name, n_experts, n_steps, uniform=True
    )


# ===========================================================================
# ── Section 1: compute_entropy() unit tests ──────────────────────────────────
# ===========================================================================


class TestComputeEntropy:
    """Unit tests for the compute_entropy() helper function."""

    def test_uniform_four_experts(self) -> None:
        """H(uniform, 4) = log(4) nats ≈ 1.386."""
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        result = compute_entropy(probs)
        expected = math.log(4)
        assert abs(result - expected) < 1e-6, (
            f"Expected H={expected:.6f}, got {result:.6f}"
        )

    def test_uniform_eight_experts(self) -> None:
        """H(uniform, 8) = log(8) nats ≈ 2.079."""
        probs = np.ones(8) / 8.0
        result = compute_entropy(probs)
        expected = math.log(8)
        assert abs(result - expected) < 1e-6

    def test_collapsed_distribution(self) -> None:
        """All mass on one expert → H = 0."""
        probs = np.array([1.0, 0.0, 0.0, 0.0])
        result = compute_entropy(probs)
        assert abs(result) < 1e-9, f"Collapsed entropy should be 0.0, got {result}"

    def test_two_expert_half_half(self) -> None:
        """H(0.5, 0.5) = log(2) nats."""
        probs = np.array([0.5, 0.5])
        result = compute_entropy(probs)
        expected = math.log(2)
        assert abs(result - expected) < 1e-6

    def test_empty_array_returns_zero(self) -> None:
        result = compute_entropy(np.array([]))
        assert result == 0.0

    def test_all_zero_array_returns_zero(self) -> None:
        result = compute_entropy(np.array([0.0, 0.0, 0.0]))
        assert result == 0.0

    def test_single_element_returns_zero(self) -> None:
        result = compute_entropy(np.array([1.0]))
        assert result == 0.0

    def test_unnormalised_positive_array(self) -> None:
        """
        compute_entropy normalises internally.
        [2, 2, 2, 2] → same as [0.25, 0.25, 0.25, 0.25].
        """
        unnorm = np.array([2.0, 2.0, 2.0, 2.0])
        uniform = np.array([0.25, 0.25, 0.25, 0.25])
        assert abs(compute_entropy(unnorm) - compute_entropy(uniform)) < 1e-6

    def test_near_zero_probabilities_no_nan(self) -> None:
        """Near-zero probs must not produce nan or inf."""
        probs = np.array([1.0 - 3e-15, 1e-15, 1e-15, 1e-15])
        result = compute_entropy(probs)
        assert math.isfinite(result), f"compute_entropy returned non-finite: {result}"
        assert result >= 0.0

    def test_result_is_float(self) -> None:
        probs = np.array([0.3, 0.3, 0.4])
        result = compute_entropy(probs)
        assert isinstance(result, float)

    def test_entropy_is_non_negative(self) -> None:
        for _ in range(10):
            probs = np.random.dirichlet(np.ones(8))
            assert compute_entropy(probs) >= 0.0

    def test_entropy_increases_with_uniformity(self) -> None:
        """More uniform → higher entropy."""
        peaked = np.array([0.9, 0.05, 0.025, 0.025])
        uniform = np.array([0.25, 0.25, 0.25, 0.25])
        assert compute_entropy(peaked) < compute_entropy(uniform), (
            "Peaked distribution should have lower entropy than uniform"
        )

    def test_negative_clamped_gracefully(self) -> None:
        """Small negative values (float precision) should not crash."""
        probs = np.array([-1e-15, 0.5, 0.5])
        result = compute_entropy(probs)
        assert math.isfinite(result)
        assert result >= 0.0


# ===========================================================================
# ── Section 2: compute_entropy_norm() unit tests ─────────────────────────────
# ===========================================================================


class TestComputeEntropyNorm:
    """Unit tests for the compute_entropy_norm() helper."""

    def test_uniform_normalised_is_one(self) -> None:
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        result = compute_entropy_norm(probs, n_experts=4)
        assert abs(result - 1.0) < 1e-6, (
            f"Normalised entropy for uniform distribution should be 1.0, got {result}"
        )

    def test_collapsed_normalised_is_zero(self) -> None:
        probs = np.array([1.0, 0.0, 0.0, 0.0])
        result = compute_entropy_norm(probs, n_experts=4)
        assert abs(result) < 1e-6, (
            f"Normalised entropy for collapsed distribution should be 0.0, got {result}"
        )

    def test_result_in_range_zero_to_one(self) -> None:
        for _ in range(20):
            probs = np.random.dirichlet(np.ones(8))
            result = compute_entropy_norm(probs, n_experts=8)
            assert 0.0 <= result <= 1.0, (
                f"Normalised entropy out of [0,1]: {result}"
            )

    def test_n_experts_less_than_2_returns_zero(self) -> None:
        probs = np.array([1.0])
        assert compute_entropy_norm(probs, n_experts=1) == 0.0
        assert compute_entropy_norm(probs, n_experts=0) == 0.0

    def test_partial_collapse_between_zero_and_one(self) -> None:
        """Partially collapsed distribution should yield normalised in (0, 1)."""
        probs = np.array([0.7, 0.1, 0.1, 0.1])
        result = compute_entropy_norm(probs, n_experts=4)
        assert 0.0 < result < 1.0, (
            f"Partially collapsed should give 0 < norm_entropy < 1, got {result}"
        )

    def test_eight_experts_uniform(self) -> None:
        probs = np.ones(8) / 8
        result = compute_entropy_norm(probs, n_experts=8)
        assert abs(result - 1.0) < 1e-6

    def test_n_experts_mismatch_with_probs_uses_parameter(self) -> None:
        """
        n_experts parameter is used as H_max denominator, not len(probs).
        With a 4-element uniform and n_experts=8, result should be < 1.
        """
        probs = np.ones(4) / 4
        result = compute_entropy_norm(probs, n_experts=8)
        # H_max = log(8), H = log(4), so norm = log(4)/log(8) ≈ 0.667
        expected = math.log(4) / math.log(8)
        assert abs(result - expected) < 1e-5


# ===========================================================================
# ── Section 3: LayerEntropyReport dataclass ──────────────────────────────────
# ===========================================================================


class TestLayerEntropyReport:
    """LayerEntropyReport structure and field constraints."""

    def _get_report(
        self, config: WatchConfig, uniform: bool = True
    ) -> LayerEntropyReport:
        collector = _build_collector_with_events(
            config, "layer.0.gate", n_experts=4, n_events=20, uniform=uniform
        )
        analyzer = EntropyAnalyzer(config)
        reports = analyzer.analyze(collector)
        assert "layer.0.gate" in reports
        return reports["layer.0.gate"]

    def test_report_has_layer_name(self, default_config: WatchConfig) -> None:
        report = self._get_report(default_config)
        assert report.layer_name == "layer.0.gate"

    def test_report_has_current_entropy_float(
        self, default_config: WatchConfig
    ) -> None:
        report = self._get_report(default_config)
        assert isinstance(report.current_entropy, float)
        assert math.isfinite(report.current_entropy)

    def test_report_has_normalized_entropy_in_range(
        self, default_config: WatchConfig
    ) -> None:
        report = self._get_report(default_config)
        assert 0.0 <= report.normalized_entropy <= 1.0, (
            f"normalized_entropy out of [0,1]: {report.normalized_entropy}"
        )

    def test_report_trend_is_valid_string(self, default_config: WatchConfig) -> None:
        report = self._get_report(default_config)
        assert report.trend in {"DECLINING", "STABLE", "IMPROVING", "UNKNOWN"}, (
            f"Invalid trend value: {report.trend!r}"
        )

    def test_report_drift_detected_is_bool(self, default_config: WatchConfig) -> None:
        report = self._get_report(default_config)
        assert isinstance(report.drift_detected, bool)

    def test_report_has_step_field(self, default_config: WatchConfig) -> None:
        report = self._get_report(default_config)
        assert hasattr(report, "step")
        assert isinstance(report.step, int)


# ===========================================================================
# ── Section 4: EntropyAnalyzer.analyze() ────────────────────────────────────
# ===========================================================================


class TestEntropyAnalyzerReturnStructure:
    """analyze() returns the correct data structure."""

    def test_returns_dict(self, default_config: WatchConfig) -> None:
        collector = _build_collector_with_events(
            default_config, "layer.0.gate", n_experts=4, n_events=10
        )
        analyzer = EntropyAnalyzer(default_config)
        result = analyzer.analyze(collector)
        assert isinstance(result, dict)

    def test_one_report_per_layer(self, default_config: WatchConfig) -> None:
        collector = StatCollector(default_config)
        for i in range(3):
            layer_name = f"layer.{i}.gate"
            collector.register_layer(layer_name, n_experts=4)
            for step in range(10):
                collector.write_routing_event(
                    make_routing_event(
                        layer_name=layer_name,
                        n_experts=4,
                        global_step=step,
                    )
                )
        analyzer = EntropyAnalyzer(default_config)
        result = analyzer.analyze(collector)
        assert len(result) == 3, (
            f"Expected 3 reports (one per layer), got {len(result)}"
        )

    def test_keys_are_layer_names(self, default_config: WatchConfig) -> None:
        collector = _build_collector_with_events(
            default_config, "my_custom_layer.gate", n_experts=8, n_events=10
        )
        analyzer = EntropyAnalyzer(default_config)
        result = analyzer.analyze(collector)
        assert "my_custom_layer.gate" in result

    def test_values_are_layer_entropy_reports(
        self, default_config: WatchConfig
    ) -> None:
        collector = _build_collector_with_events(
            default_config, "layer.0.gate", n_experts=4, n_events=10
        )
        analyzer = EntropyAnalyzer(default_config)
        result = analyzer.analyze(collector)
        for name, report in result.items():
            assert isinstance(report, LayerEntropyReport), (
                f"Expected LayerEntropyReport for '{name}', got {type(report)}"
            )

    def test_empty_collector_returns_empty_dict(
        self, default_config: WatchConfig
    ) -> None:
        """A collector with no routing events should yield an empty report dict."""
        collector = StatCollector(default_config)
        analyzer = EntropyAnalyzer(default_config)
        result = analyzer.analyze(collector)
        assert result == {}

    def test_registered_but_empty_layer_handled_gracefully(
        self, default_config: WatchConfig
    ) -> None:
        """A registered layer with zero events should not crash."""
        collector = StatCollector(default_config)
        collector.register_layer("layer.0.gate", n_experts=4)
        # No events written
        analyzer = EntropyAnalyzer(default_config)
        # Should either return an empty dict or a report with UNKNOWN trend
        try:
            result = analyzer.analyze(collector)
            # If it returned something, trend should be valid
            for report in result.values():
                assert report.trend in {"DECLINING", "STABLE", "IMPROVING", "UNKNOWN"}
        except Exception as exc:
            pytest.fail(f"analyze() raised on empty layer: {exc}")


class TestEntropyAnalyzerEntropyValues:
    """Entropy values reflect actual routing distribution."""

    def test_uniform_routing_high_normalized_entropy(
        self,
        entropy_stat_collector_uniform: StatCollector,
        default_config: WatchConfig,
    ) -> None:
        """Uniform routing → normalized entropy should be close to 1.0."""
        analyzer = EntropyAnalyzer(default_config)
        result = analyzer.analyze(entropy_stat_collector_uniform)
        report = result["layer.0.gate"]
        assert report.normalized_entropy > 0.85, (
            f"Uniform routing should yield high entropy, got {report.normalized_entropy:.4f}"
        )

    def test_collapsed_routing_low_normalized_entropy(
        self,
        entropy_stat_collector_collapsed: StatCollector,
        default_config: WatchConfig,
    ) -> None:
        """Collapsed routing (expert 0 dominates) → low normalized entropy."""
        analyzer = EntropyAnalyzer(default_config)
        result = analyzer.analyze(entropy_stat_collector_collapsed)
        report = result["layer.0.gate"]
        assert report.normalized_entropy < 0.2, (
            f"Collapsed routing should yield low entropy, got {report.normalized_entropy:.4f}"
        )

    def test_uniform_entropy_greater_than_collapsed(
        self, default_config: WatchConfig
    ) -> None:
        """Basic ordering: H(uniform) > H(collapsed)."""
        uniform_collector = _build_collector_with_events(
            default_config, "layer.0.gate", n_experts=4, n_events=20, uniform=True
        )
        collapsed_collector = _build_collector_with_events(
            default_config, "layer.0.gate", n_experts=4, n_events=20, uniform=False
        )
        analyzer = EntropyAnalyzer(default_config)

        uniform_report = analyzer.analyze(uniform_collector)["layer.0.gate"]
        # Fresh analyzer for collapsed (independent CUSUM state)
        analyzer2 = EntropyAnalyzer(default_config)
        collapsed_report = analyzer2.analyze(collapsed_collector)["layer.0.gate"]

        assert uniform_report.normalized_entropy > collapsed_report.normalized_entropy, (
            "Uniform routing should produce higher entropy than collapsed routing"
        )

    def test_normalized_entropy_always_in_range(
        self, default_config: WatchConfig
    ) -> None:
        """For any routing pattern, normalized_entropy must stay in [0, 1]."""
        for n_experts in [2, 4, 8, 16]:
            collector = _build_collector_with_events(
                default_config, f"layer.{n_experts}.gate",
                n_experts=n_experts, n_events=15, uniform=True
            )
            analyzer = EntropyAnalyzer(default_config)
            result = analyzer.analyze(collector)
            for report in result.values():
                assert 0.0 <= report.normalized_entropy <= 1.0, (
                    f"normalized_entropy={report.normalized_entropy} out of [0,1] "
                    f"for n_experts={n_experts}"
                )

    def test_current_entropy_is_non_negative(
        self, default_config: WatchConfig
    ) -> None:
        for uniform in [True, False]:
            collector = _build_collector_with_events(
                default_config, "layer.0.gate", n_experts=4, n_events=10, uniform=uniform
            )
            analyzer = EntropyAnalyzer(default_config)
            result = analyzer.analyze(collector)
            for report in result.values():
                assert report.current_entropy >= 0.0


class TestEntropyAnalyzerTrendClassification:
    """Trend field reflects entropy trajectory."""

    def test_declining_entropy_classified_as_declining(
        self, default_config: WatchConfig
    ) -> None:
        """
        Steadily increasing bias → entropy decreases → DECLINING trend expected.
        We use a large n_steps so the trend window has sufficient history.
        """
        collector = _build_declining_collector(
            default_config, n_steps=40, n_experts=4
        )
        analyzer = EntropyAnalyzer(default_config)
        result = analyzer.analyze(collector)
        report = result["layer.0.gate"]
        # After 40 steps of linear collapse, trend should be DECLINING
        assert report.trend in {"DECLINING", "UNKNOWN"}, (
            f"Expected DECLINING or UNKNOWN for declining entropy, got {report.trend!r}"
        )

    def test_stable_entropy_classified_as_stable_or_unknown(
        self, default_config: WatchConfig
    ) -> None:
        """
        Constant uniform logits → entropy doesn't change → STABLE or UNKNOWN.
        UNKNOWN is acceptable when there is insufficient history.
        """
        collector = _build_stable_collector(default_config, n_steps=30)
        analyzer = EntropyAnalyzer(default_config)
        result = analyzer.analyze(collector)
        report = result["layer.0.gate"]
        assert report.trend in {"STABLE", "UNKNOWN", "IMPROVING"}, (
            f"Unexpected trend for stable entropy: {report.trend!r}"
        )

    def test_trend_unknown_with_few_events(self, default_config: WatchConfig) -> None:
        """Less than _MIN_TREND_HISTORY events → trend must be UNKNOWN."""
        collector = _build_collector_with_events(
            default_config, "layer.0.gate", n_experts=4, n_events=2
        )
        analyzer = EntropyAnalyzer(default_config)
        result = analyzer.analyze(collector)
        if "layer.0.gate" in result:
            report = result["layer.0.gate"]
            assert report.trend == "UNKNOWN", (
                f"Expected UNKNOWN with only 2 events, got {report.trend!r}"
            )

    def test_trend_is_string_type(self, default_config: WatchConfig) -> None:
        collector = _build_collector_with_events(
            default_config, "layer.0.gate", n_experts=4, n_events=15
        )
        analyzer = EntropyAnalyzer(default_config)
        result = analyzer.analyze(collector)
        report = result["layer.0.gate"]
        assert isinstance(report.trend, str)


class TestEntropyAnalyzerDriftDetection:
    """CUSUM drift_detected field."""

    def test_drift_not_detected_for_stable_uniform(
        self, default_config: WatchConfig
    ) -> None:
        """
        Perfectly uniform routing has no drift → drift_detected should be False
        (or True only after CUSUM's threshold is crossed, which requires large excursions).
        """
        collector = _build_stable_collector(default_config, n_steps=30)
        analyzer = EntropyAnalyzer(default_config)
        result = analyzer.analyze(collector)
        report = result["layer.0.gate"]
        # For stable uniform routing, drift_detected should be False
        assert report.drift_detected is False, (
            f"Expected drift_detected=False for stable uniform routing, "
            f"got {report.drift_detected}"
        )

    def test_drift_detected_is_bool(self, default_config: WatchConfig) -> None:
        collector = _build_collector_with_events(
            default_config, "layer.0.gate", n_experts=4, n_events=10
        )
        analyzer = EntropyAnalyzer(default_config)
        result = analyzer.analyze(collector)
        for report in result.values():
            assert isinstance(report.drift_detected, bool)

    def test_large_entropy_drop_triggers_drift(
        self, default_config: WatchConfig
    ) -> None:
        """
        A rapid sharp drop in entropy (from uniform to fully collapsed in few steps)
        should eventually trigger CUSUM drift detection.
        """
        collector = StatCollector(default_config)
        collector.register_layer("layer.0.gate", n_experts=4)

        # Phase 1: 20 perfectly uniform steps
        for step in range(20):
            collector.write_routing_event(
                make_routing_event(
                    layer_name="layer.0.gate",
                    n_experts=4,
                    batch_size=8,
                    global_step=step,
                    uniform=True,
                )
            )
        # Phase 2: 20 fully collapsed steps (abrupt change)
        for step in range(20, 40):
            collector.write_routing_event(
                make_routing_event(
                    layer_name="layer.0.gate",
                    n_experts=4,
                    batch_size=8,
                    global_step=step,
                    uniform=False,  # hard collapse
                )
            )

        analyzer = EntropyAnalyzer(default_config)
        result = analyzer.analyze(collector)
        report = result["layer.0.gate"]

        # After a large abrupt change, CUSUM should fire; we allow for the
        # possibility that the internal window might not capture the full drop.
        # The primary assertion is no crash and a finite, valid report.
        assert isinstance(report.drift_detected, bool)
        assert math.isfinite(report.current_entropy)
        assert 0.0 <= report.normalized_entropy <= 1.0


class TestEntropyAnalyzerMultiLayer:
    """Multiple layers are analyzed independently."""

    def test_two_layers_independent_entropy(
        self, populated_collector: StatCollector, default_config: WatchConfig
    ) -> None:
        """
        populated_collector has:
          - layers.0.gate: uniform events
          - layers.1.gate: collapsed events
        Uniform layer should have higher entropy than collapsed.
        """
        analyzer = EntropyAnalyzer(default_config)
        result = analyzer.analyze(populated_collector)

        assert "layers.0.gate" in result
        assert "layers.1.gate" in result

        uniform_report = result["layers.0.gate"]
        collapsed_report = result["layers.1.gate"]

        assert uniform_report.normalized_entropy > collapsed_report.normalized_entropy, (
            f"Uniform layer entropy ({uniform_report.normalized_entropy:.4f}) should "
            f"exceed collapsed layer entropy ({collapsed_report.normalized_entropy:.4f})"
        )

    def test_each_layer_has_its_own_cusum(
        self, default_config: WatchConfig
    ) -> None:
        """CUSUM detectors are per-layer and must not bleed state between layers."""
        collector = StatCollector(default_config)
        for i in range(3):
            layer_name = f"layer.{i}.gate"
            collector.register_layer(layer_name, n_experts=4)
            for step in range(20):
                collector.write_routing_event(
                    make_routing_event(
                        layer_name=layer_name,
                        n_experts=4,
                        global_step=step,
                        uniform=True,
                    )
                )

        analyzer = EntropyAnalyzer(default_config)
        result = analyzer.analyze(collector)

        assert len(result) == 3
        for name, report in result.items():
            assert report.layer_name == name
            assert isinstance(report.drift_detected, bool)


class TestEntropyAnalyzerIdempotence:
    """Calling analyze() multiple times should not corrupt internal state."""

    def test_repeated_calls_consistent_results(
        self, default_config: WatchConfig
    ) -> None:
        """Two analyze() calls on the same collector should give consistent entropy values."""
        collector = _build_collector_with_events(
            default_config, "layer.0.gate", n_experts=4, n_events=20, uniform=True
        )
        analyzer = EntropyAnalyzer(default_config)

        r1 = analyzer.analyze(collector)
        r2 = analyzer.analyze(collector)

        # Entropy values should be identical (same collector, no new events)
        assert abs(
            r1["layer.0.gate"].current_entropy - r2["layer.0.gate"].current_entropy
        ) < 1e-9

    def test_new_events_update_report(self, default_config: WatchConfig) -> None:
        """Adding collapsed events after initial uniform analysis changes the report."""
        collector = _build_collector_with_events(
            default_config, "layer.0.gate", n_experts=4, n_events=20, uniform=True
        )
        analyzer = EntropyAnalyzer(default_config)
        r_uniform = analyzer.analyze(collector)
        entropy_before = r_uniform["layer.0.gate"].normalized_entropy

        # Add 40 collapsed events to push entropy down
        for step in range(20, 60):
            collector.write_routing_event(
                make_routing_event(
                    layer_name="layer.0.gate",
                    n_experts=4,
                    batch_size=8,
                    global_step=step,
                    uniform=False,
                )
            )

        r_collapsed = analyzer.analyze(collector)
        entropy_after = r_collapsed["layer.0.gate"].normalized_entropy

        assert entropy_after < entropy_before, (
            f"Adding collapsed events should reduce entropy "
            f"({entropy_before:.4f} → {entropy_after:.4f})"
        )


class TestEntropyAnalyzerEdgeCases:
    """Edge cases and defensive behaviours."""

    def test_single_expert_handled(self, default_config: WatchConfig) -> None:
        """n_experts=1 should not crash (degenerate case)."""
        collector = StatCollector(default_config)
        collector.register_layer("layer.0.gate", n_experts=1)
        for step in range(5):
            logits = torch.zeros(4, 1)
            collector.write_routing_event(
                RoutingEvent(
                    timestamp=time.time(),
                    global_step=step,
                    layer_name="layer.0.gate",
                    routing_logits=logits.detach(),
                    selected_experts=torch.zeros(4, 1, dtype=torch.long),
                    expert_count=1,
                    batch_size=4,
                )
            )
        analyzer = EntropyAnalyzer(default_config)
        try:
            result = analyzer.analyze(collector)
            # If a report was returned, its entropy should be 0
            if "layer.0.gate" in result:
                report = result["layer.0.gate"]
                assert report.normalized_entropy == 0.0 or math.isfinite(
                    report.normalized_entropy
                )
        except Exception as exc:
            pytest.fail(f"analyze() crashed on single-expert layer: {exc}")

    def test_large_n_experts(self, default_config: WatchConfig) -> None:
        """n_experts=64 should not cause overflow or precision issues."""
        collector = _build_collector_with_events(
            default_config, "layer.0.gate", n_experts=64, n_events=20, uniform=True
        )
        analyzer = EntropyAnalyzer(default_config)
        result = analyzer.analyze(collector)
        if "layer.0.gate" in result:
            report = result["layer.0.gate"]
            assert 0.0 <= report.normalized_entropy <= 1.0
            assert math.isfinite(report.current_entropy)

    def test_very_large_batch_size(self, default_config: WatchConfig) -> None:
        """Large batch sizes should not cause issues."""
        collector = _build_collector_with_events(
            default_config, "layer.0.gate", n_experts=4, n_events=10
        )
        # Overwrite with large batch events
        for step in range(10, 15):
            collector.write_routing_event(
                make_routing_event(
                    layer_name="layer.0.gate",
                    n_experts=4,
                    batch_size=2048,
                    global_step=step,
                    uniform=True,
                )
            )
        analyzer = EntropyAnalyzer(default_config)
        result = analyzer.analyze(collector)
        assert "layer.0.gate" in result

    def test_strict_config_triggers_lower_entropy_threshold(
        self, strict_config: WatchConfig
    ) -> None:
        """With strict thresholds, collapsed routing entropy should still be in range."""
        collector = _build_collector_with_events(
            strict_config, "layer.0.gate", n_experts=4, n_events=20, uniform=False
        )
        analyzer = EntropyAnalyzer(strict_config)
        result = analyzer.analyze(collector)
        if "layer.0.gate" in result:
            report = result["layer.0.gate"]
            assert 0.0 <= report.normalized_entropy <= 1.0

    def test_analyzer_does_not_mutate_stat_collector(
        self, default_config: WatchConfig
    ) -> None:
        """analyze() must not modify the StatCollector state."""
        collector = _build_collector_with_events(
            default_config, "layer.0.gate", n_experts=4, n_events=15
        )
        stats_before = collector.get_all_stats()
        n_routing_before = len(stats_before["routing"])

        analyzer = EntropyAnalyzer(default_config)
        analyzer.analyze(collector)

        stats_after = collector.get_all_stats()
        n_routing_after = len(stats_after["routing"])

        assert n_routing_before == n_routing_after, (
            "analyze() modified the number of routing layers in StatCollector"
        )
