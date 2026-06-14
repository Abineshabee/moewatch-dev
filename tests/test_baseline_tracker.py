# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# tests/test_baseline_tracker.py
# =============================================================================
#
# Project      : MoEWatch
# Description  : Tests for moewatch.collector.baseline_tracker.BaselineTracker.
#
#                Coverage targets (>= 80%):
#                  - register_layer() initialises clean history
#                  - register_layer() is idempotent
#                  - update_signal() adds clean points
#                  - update_signal() excludes intervention-influenced steps
#                  - mark_intervention() creates exclusion window
#                  - is_baseline_valid() False before min_clean_steps
#                  - is_baseline_valid() True after sufficient clean steps
#                  - is_baseline_valid() False for unregistered layer
#                  - compute_counterfactual_delta() raises on insufficient history
#                  - compute_counterfactual_delta() raises on unregistered layer
#                  - compute_counterfactual_delta() returns float for valid baseline
#                  - compute_counterfactual_delta() correct sign (improvement > 0)
#                  - get_baseline() returns 0.0 when invalid
#                  - get_baseline() returns projected value for valid baseline
#                  - intervention window steps excluded from clean history
#                  - steps after window end are accepted again
#                  - multiple interventions tracked independently
#                  - auto-register on update_signal / mark_intervention
#                  - thread-safe concurrent updates
#                  - __repr__ contains expected fields
#
# =============================================================================

from __future__ import annotations

import math
import threading
from typing import List

import pytest

from moewatch.collector.baseline_tracker import BaselineTracker
from moewatch.config import OutputMode, WatchConfig


# ===========================================================================
# ── Helpers ──────────────────────────────────────────────────────────────────
# ===========================================================================


def _make_config(min_clean_steps: int = 5, exclusion_window: int = 10) -> WatchConfig:
    return WatchConfig(
        output=OutputMode.SILENT,
        baseline_min_clean_steps=min_clean_steps,
        baseline_exclusion_window=exclusion_window,
    )


def _fill_clean_history(
    tracker: BaselineTracker,
    layer: str,
    n: int,
    start_step: int = 0,
    value: float = 0.5,
) -> None:
    """Push ``n`` clean signal updates for ``layer``."""
    for i in range(n):
        tracker.update_signal(layer, value, step=start_step + i)


# ===========================================================================
# ── 1. register_layer ────────────────────────────────────────────────────────
# ===========================================================================


class TestRegisterLayer:
    def test_register_creates_state(self) -> None:
        config = _make_config()
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        assert "layer.0.gate" in tracker._clean_history

    def test_register_initialises_empty_history(self) -> None:
        config = _make_config()
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        assert tracker._clean_history["layer.0.gate"] == []

    def test_register_idempotent_preserves_history(self) -> None:
        config = _make_config()
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        tracker.update_signal("layer.0.gate", 0.8, step=0)
        tracker.register_layer("layer.0.gate")  # second call — idempotent
        assert len(tracker._clean_history["layer.0.gate"]) == 1

    def test_register_multiple_layers(self) -> None:
        config = _make_config()
        tracker = BaselineTracker(config)
        for i in range(4):
            tracker.register_layer(f"layer.{i}.gate")
        assert len(tracker._clean_history) == 4

    def test_repr_after_registration(self) -> None:
        config = _make_config()
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        r = repr(tracker)
        assert "BaselineTracker" in r


# ===========================================================================
# ── 2. update_signal ─────────────────────────────────────────────────────────
# ===========================================================================


class TestUpdateSignal:
    def test_update_adds_clean_point(self) -> None:
        config = _make_config()
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        tracker.update_signal("layer.0.gate", 0.75, step=1)
        assert len(tracker._clean_history["layer.0.gate"]) == 1

    def test_update_multiple_points(self) -> None:
        config = _make_config()
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        for step in range(10):
            tracker.update_signal("layer.0.gate", float(step) * 0.1, step=step)
        assert len(tracker._clean_history["layer.0.gate"]) == 10

    def test_update_auto_registers_unknown_layer(self) -> None:
        config = _make_config()
        tracker = BaselineTracker(config)
        tracker.update_signal("new.layer", 0.5, step=0)
        assert "new.layer" in tracker._clean_history

    def test_update_records_step_and_value(self) -> None:
        config = _make_config()
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        tracker.update_signal("layer.0.gate", 0.42, step=7)
        entry = tracker._clean_history["layer.0.gate"][0]
        assert entry == (7, 0.42)

    def test_update_advances_current_step(self) -> None:
        config = _make_config()
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        tracker.update_signal("layer.0.gate", 0.5, step=42)
        assert tracker._current_step["layer.0.gate"] == 42


# ===========================================================================
# ── 3. mark_intervention ─────────────────────────────────────────────────────
# ===========================================================================


class TestMarkIntervention:
    def test_mark_creates_exclusion_window(self) -> None:
        config = _make_config(exclusion_window=10)
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        tracker.mark_intervention("layer.0.gate", start_step=50)
        windows = tracker._intervention_windows["layer.0.gate"]
        assert len(windows) == 1
        assert windows[0] == (50, 60)

    def test_mark_auto_registers_unknown_layer(self) -> None:
        config = _make_config()
        tracker = BaselineTracker(config)
        tracker.mark_intervention("auto.layer", start_step=0)
        assert "auto.layer" in tracker._intervention_windows

    def test_mark_multiple_windows_accumulated(self) -> None:
        config = _make_config(exclusion_window=5)
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        tracker.mark_intervention("layer.0.gate", start_step=10)
        tracker.mark_intervention("layer.0.gate", start_step=30)
        assert len(tracker._intervention_windows["layer.0.gate"]) == 2

    def test_exclusion_window_end_step_correct(self) -> None:
        config = _make_config(exclusion_window=50)
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        tracker.mark_intervention("layer.0.gate", start_step=100)
        _, end = tracker._intervention_windows["layer.0.gate"][0]
        assert end == 150

    def test_mark_at_step_zero(self) -> None:
        config = _make_config(exclusion_window=20)
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        tracker.mark_intervention("layer.0.gate", start_step=0)
        start, end = tracker._intervention_windows["layer.0.gate"][0]
        assert start == 0 and end == 20


# ===========================================================================
# ── 4. is_baseline_valid ─────────────────────────────────────────────────────
# ===========================================================================


class TestIsBaselineValid:
    def test_false_before_min_clean_steps(self) -> None:
        config = _make_config(min_clean_steps=5)
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        _fill_clean_history(tracker, "layer.0.gate", n=4)
        assert tracker.is_baseline_valid("layer.0.gate") is False

    def test_true_at_exactly_min_clean_steps(self) -> None:
        config = _make_config(min_clean_steps=5)
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        _fill_clean_history(tracker, "layer.0.gate", n=5)
        assert tracker.is_baseline_valid("layer.0.gate") is True

    def test_true_above_min_clean_steps(self) -> None:
        config = _make_config(min_clean_steps=5)
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        _fill_clean_history(tracker, "layer.0.gate", n=20)
        assert tracker.is_baseline_valid("layer.0.gate") is True

    def test_false_for_unregistered_layer(self) -> None:
        config = _make_config()
        tracker = BaselineTracker(config)
        assert tracker.is_baseline_valid("does.not.exist") is False

    def test_false_when_all_steps_excluded(self) -> None:
        """If every update falls inside exclusion window, clean history stays empty."""
        config = _make_config(min_clean_steps=5, exclusion_window=100)
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        tracker.mark_intervention("layer.0.gate", start_step=0)
        # Steps 0–99 are excluded
        for step in range(10):
            tracker.update_signal("layer.0.gate", 0.5, step=step)
        assert tracker.is_baseline_valid("layer.0.gate") is False

    def test_returns_bool(self) -> None:
        config = _make_config()
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        result = tracker.is_baseline_valid("layer.0.gate")
        assert isinstance(result, bool)


# ===========================================================================
# ── 5. Exclusion window effect on update_signal ──────────────────────────────
# ===========================================================================


class TestExclusionWindowEffect:
    def test_steps_inside_window_not_added_to_clean_history(self) -> None:
        config = _make_config(min_clean_steps=3, exclusion_window=10)
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        tracker.mark_intervention("layer.0.gate", start_step=5)

        # Steps 5–14 are excluded
        for step in range(15):
            tracker.update_signal("layer.0.gate", 0.5, step=step)

        history = tracker._clean_history["layer.0.gate"]
        clean_steps = [s for s, _ in history]
        for s in range(5, 15):
            assert s not in clean_steps, (
                f"Step {s} inside exclusion window [5,15) was added to clean history"
            )

    def test_steps_before_window_accepted(self) -> None:
        config = _make_config(exclusion_window=10)
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        tracker.mark_intervention("layer.0.gate", start_step=10)
        for step in range(10):
            tracker.update_signal("layer.0.gate", 0.5, step=step)
        history = tracker._clean_history["layer.0.gate"]
        assert len(history) == 10

    def test_steps_after_window_end_accepted(self) -> None:
        config = _make_config(exclusion_window=10)
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        tracker.mark_intervention("layer.0.gate", start_step=0)
        # Steps 0–9 excluded; step 10 onward is clean
        for step in range(15):
            tracker.update_signal("layer.0.gate", 0.5, step=step)
        history = tracker._clean_history["layer.0.gate"]
        clean_steps = [s for s, _ in history]
        assert all(s >= 10 for s in clean_steps), (
            f"Some excluded steps leaked into clean history: {clean_steps}"
        )

    def test_boundary_step_equal_to_end_is_clean(self) -> None:
        """exclusion window is [start, end) — step == end_step is CLEAN."""
        config = _make_config(exclusion_window=5)
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        tracker.mark_intervention("layer.0.gate", start_step=0)
        # end_step = 5; step=5 should be clean
        tracker.update_signal("layer.0.gate", 0.5, step=5)
        history = tracker._clean_history["layer.0.gate"]
        assert len(history) == 1 and history[0][0] == 5

    def test_multiple_windows_exclude_all_covered_steps(self) -> None:
        config = _make_config(exclusion_window=5)
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        tracker.mark_intervention("layer.0.gate", start_step=0)   # [0, 5)
        tracker.mark_intervention("layer.0.gate", start_step=10)  # [10, 15)

        for step in range(20):
            tracker.update_signal("layer.0.gate", 0.5, step=step)

        history = tracker._clean_history["layer.0.gate"]
        clean_steps = {s for s, _ in history}
        excluded = set(range(0, 5)) | set(range(10, 15))
        assert clean_steps.isdisjoint(excluded), (
            f"Excluded steps leaked into clean history: {clean_steps & excluded}"
        )


# ===========================================================================
# ── 6. compute_counterfactual_delta ──────────────────────────────────────────
# ===========================================================================


class TestComputeCounterfactualDelta:
    def test_raises_on_unregistered_layer(self) -> None:
        config = _make_config()
        tracker = BaselineTracker(config)
        with pytest.raises(KeyError):
            tracker.compute_counterfactual_delta("not.registered", 0.5)

    def test_raises_on_insufficient_clean_history(self) -> None:
        config = _make_config(min_clean_steps=10)
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        _fill_clean_history(tracker, "layer.0.gate", n=5)
        with pytest.raises(ValueError, match="insufficient clean history"):
            tracker.compute_counterfactual_delta("layer.0.gate", 0.5)

    def test_returns_float_for_valid_baseline(self) -> None:
        config = _make_config(min_clean_steps=5)
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        _fill_clean_history(tracker, "layer.0.gate", n=10, value=0.6)
        tracker.update_signal("layer.0.gate", 0.6, step=10)
        result = tracker.compute_counterfactual_delta("layer.0.gate", 0.6)
        assert isinstance(result, float)
        assert math.isfinite(result)

    def test_positive_delta_when_actual_above_baseline(self) -> None:
        """
        Flat baseline at 0.5; actual = 0.8 → delta should be positive (improvement).
        """
        config = _make_config(min_clean_steps=5)
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        for step in range(10):
            tracker.update_signal("layer.0.gate", 0.5, step=step)
        delta = tracker.compute_counterfactual_delta("layer.0.gate", 0.8)
        assert delta > 0, f"Expected positive delta (improvement), got {delta}"

    def test_negative_delta_when_actual_below_baseline(self) -> None:
        config = _make_config(min_clean_steps=5)
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        for step in range(10):
            tracker.update_signal("layer.0.gate", 0.8, step=step)
        delta = tracker.compute_counterfactual_delta("layer.0.gate", 0.3)
        assert delta < 0, f"Expected negative delta (regression), got {delta}"

    def test_near_zero_delta_when_actual_equals_baseline(self) -> None:
        config = _make_config(min_clean_steps=5)
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        for step in range(20):
            tracker.update_signal("layer.0.gate", 0.6, step=step)
        # Current step is 19; baseline at 19 should be ≈ 0.6 for flat history
        delta = tracker.compute_counterfactual_delta("layer.0.gate", 0.6)
        assert abs(delta) < 0.1, (
            f"Expected delta ≈ 0 for actual ≈ baseline, got {delta}"
        )

    def test_linear_trend_projected_correctly(self) -> None:
        """
        history: y = 0.01 * step over steps 0–29.
        Baseline at step 29 should be ≈ 0.29.
        actual = 0.35 → delta ≈ 0.06 (positive improvement).
        """
        config = _make_config(min_clean_steps=10)
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        for step in range(30):
            tracker.update_signal("layer.0.gate", 0.01 * step, step=step)
        delta = tracker.compute_counterfactual_delta("layer.0.gate", 0.35)
        # Projected baseline at step 29 ≈ 0.29 → delta ≈ 0.06
        assert delta > 0, f"Expected positive delta, got {delta}"


# ===========================================================================
# ── 7. get_baseline ──────────────────────────────────────────────────────────
# ===========================================================================


class TestGetBaseline:
    def test_returns_zero_when_invalid(self) -> None:
        config = _make_config(min_clean_steps=20)
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        result = tracker.get_baseline("layer.0.gate", step=100)
        assert result == 0.0

    def test_returns_float_when_valid(self) -> None:
        config = _make_config(min_clean_steps=5)
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        _fill_clean_history(tracker, "layer.0.gate", n=10, value=0.5)
        result = tracker.get_baseline("layer.0.gate", step=10)
        assert isinstance(result, float)
        assert math.isfinite(result)

    def test_flat_history_baseline_is_constant(self) -> None:
        config = _make_config(min_clean_steps=5)
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        _fill_clean_history(tracker, "layer.0.gate", n=10, value=0.7)
        # Flat history → baseline should be ≈ 0.7 at any step
        for step in [5, 20, 100]:
            b = tracker.get_baseline("layer.0.gate", step=step)
            assert abs(b - 0.7) < 0.05, (
                f"Flat baseline should be ≈ 0.7 at step {step}, got {b}"
            )

    def test_returns_zero_for_unregistered_layer(self) -> None:
        config = _make_config()
        tracker = BaselineTracker(config)
        # get_baseline checks is_baseline_valid which returns False for unknown
        result = tracker.get_baseline("no.such.layer", step=0)
        assert result == 0.0


# ===========================================================================
# ── 8. _project_baseline static method ───────────────────────────────────────
# ===========================================================================


class TestProjectBaseline:
    def test_flat_history_same_value(self) -> None:
        history = [(i, 0.5) for i in range(10)]
        result = BaselineTracker._project_baseline(history, step=20)
        assert abs(result - 0.5) < 1e-6

    def test_linear_history_correct_projection(self) -> None:
        # y = 0.1 * step → at step 20 projection should be ≈ 2.0
        history = [(i, 0.1 * i) for i in range(10)]
        result = BaselineTracker._project_baseline(history, step=20)
        assert abs(result - 2.0) < 0.05

    def test_degenerate_same_step_returns_mean(self) -> None:
        # All steps identical → zero variance → should return mean value
        history = [(5, 0.3), (5, 0.5), (5, 0.7)]
        result = BaselineTracker._project_baseline(history, step=10)
        assert abs(result - 0.5) < 1e-6

    def test_returns_float(self) -> None:
        history = [(i, float(i)) for i in range(5)]
        result = BaselineTracker._project_baseline(history, step=5)
        assert isinstance(result, float)


# ===========================================================================
# ── 9. Multi-layer isolation ─────────────────────────────────────────────────
# ===========================================================================


class TestMultiLayerIsolation:
    def test_clean_histories_are_independent(self) -> None:
        config = _make_config(min_clean_steps=5)
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        tracker.register_layer("layer.1.gate")

        _fill_clean_history(tracker, "layer.0.gate", n=10, value=0.8)
        _fill_clean_history(tracker, "layer.1.gate", n=3, value=0.2)

        assert tracker.is_baseline_valid("layer.0.gate") is True
        assert tracker.is_baseline_valid("layer.1.gate") is False

    def test_exclusion_window_on_one_layer_does_not_affect_other(self) -> None:
        config = _make_config(exclusion_window=20)
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        tracker.register_layer("layer.1.gate")

        tracker.mark_intervention("layer.0.gate", start_step=0)

        # Both layers get same steps
        for step in range(10):
            tracker.update_signal("layer.0.gate", 0.5, step=step)
            tracker.update_signal("layer.1.gate", 0.5, step=step)

        # layer.0 should have 0 clean points (all excluded)
        assert len(tracker._clean_history["layer.0.gate"]) == 0
        # layer.1 should have 10 clean points (no exclusion on it)
        assert len(tracker._clean_history["layer.1.gate"]) == 10

    def test_multiple_layers_baselines_computable_independently(self) -> None:
        config = _make_config(min_clean_steps=5)
        tracker = BaselineTracker(config)
        for i in range(3):
            layer = f"layer.{i}.gate"
            tracker.register_layer(layer)
            for step in range(10):
                tracker.update_signal(layer, float(i) * 0.1, step=step)

        for i in range(3):
            layer = f"layer.{i}.gate"
            assert tracker.is_baseline_valid(layer)
            b = tracker.get_baseline(layer, step=10)
            assert abs(b - float(i) * 0.1) < 0.05


# ===========================================================================
# ── 10. Thread safety ────────────────────────────────────────────────────────
# ===========================================================================


class TestThreadSafety:
    def test_concurrent_updates_no_crash(self) -> None:
        config = _make_config(min_clean_steps=5)
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        errors: List[Exception] = []

        def worker(tid: int) -> None:
            try:
                for step in range(20):
                    tracker.update_signal("layer.0.gate", float(tid) * 0.01, step=tid * 100 + step)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent update errors: {errors}"

    def test_concurrent_mark_and_update(self) -> None:
        config = _make_config()
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        errors: List[Exception] = []

        def marker() -> None:
            try:
                for step in range(0, 100, 20):
                    tracker.mark_intervention("layer.0.gate", start_step=step)
            except Exception as exc:
                errors.append(exc)

        def updater() -> None:
            try:
                for step in range(50):
                    tracker.update_signal("layer.0.gate", 0.5, step=step)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=marker),
            threading.Thread(target=updater),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []


# ===========================================================================
# ── 11. Edge cases ────────────────────────────────────────────────────────────
# ===========================================================================


class TestEdgeCases:
    def test_zero_exclusion_window_no_steps_excluded(self) -> None:
        config = _make_config(exclusion_window=0)
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        tracker.mark_intervention("layer.0.gate", start_step=5)
        tracker.update_signal("layer.0.gate", 0.5, step=5)
        # Window is [5, 5) — empty → step 5 should be accepted
        assert len(tracker._clean_history["layer.0.gate"]) == 1

    def test_large_min_clean_steps_keeps_invalid(self) -> None:
        config = _make_config(min_clean_steps=1000)
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        _fill_clean_history(tracker, "layer.0.gate", n=50)
        assert tracker.is_baseline_valid("layer.0.gate") is False

    def test_history_bounded_by_max_history_length(self) -> None:
        """Clean history should not grow unboundedly."""
        config = _make_config(min_clean_steps=5)
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        n_writes = BaselineTracker._MAX_HISTORY_LENGTH + 500
        for step in range(n_writes):
            tracker.update_signal("layer.0.gate", 0.5, step=step)
        assert len(tracker._clean_history["layer.0.gate"]) <= BaselineTracker._MAX_HISTORY_LENGTH

    def test_update_signal_negative_value_accepted(self) -> None:
        """Negative signal values (e.g. reward deltas) should be accepted."""
        config = _make_config()
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0.gate")
        tracker.update_signal("layer.0.gate", -0.5, step=0)
        assert tracker._clean_history["layer.0.gate"][0] == (0, -0.5)

    def test_repr_contains_layer_count(self) -> None:
        config = _make_config()
        tracker = BaselineTracker(config)
        for i in range(3):
            tracker.register_layer(f"layer.{i}.gate")
        r = repr(tracker)
        assert "3" in r
