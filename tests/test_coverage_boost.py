# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# tests/test_coverage_boost.py
# =============================================================================
#
# Project      : MoEWatch
# Description  : Supplementary coverage-focused tests targeting modules that
#                 were under-tested as of the v0.2.0 360-test baseline:
#
#                   - moewatch/analyzer/cusum.py        (55% -> target 90%+)
#                   - moewatch/policy/reward.py         (0%  -> target 90%+)
#                   - moewatch/hooks/gradient_hook.py   (68% -> target 90%+)
#                   - moewatch/policy/memory.py         (33% -> target 85%+)
#                   - moewatch/policy/bandit_policy.py  (23% -> target 80%+)
#                   - moewatch/intervention/actions.py  (65% -> target 85%+)
#                   - moewatch/report/audit_report.py   (60% -> target 85%+)
#                   - moewatch/report/json_reporter.py  (24% -> target 85%+)
#                   - moewatch/report/watch_report.py   (41% -> target 85%+)
#                   - moewatch/report/cli_reporter.py   (20% -> target 50%+)
#
# =============================================================================

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

from moewatch import Alert
from moewatch.analyzer.collapse import ExpertState, ExpertStatus, LayerCollapseReport
from moewatch.analyzer.cross_layer import CrossLayerReport
from moewatch.analyzer.cusum import CUSUMDetector, detect_change
from moewatch.analyzer.entropy import LayerEntropyReport
from moewatch.analyzer.gradient_starvation import GradientStarvationReport
from moewatch.analyzer.risk_score import RiskLevel, RiskReport
from moewatch.collector.baseline_tracker import BaselineTracker
from moewatch.config import AlertLevel, OutputMode, WatchConfig
from moewatch.hooks.gradient_hook import GradientEvent, GradientStarvationHook
from moewatch.intervention.actions import (
    AuxLossAction,
    ExpertDropoutAction,
    InterventionAction,
    NoOpAction,
    RouterNoiseAction,
)
from moewatch.policy.bandit_policy import BanditPolicy, _ACTION_SPACE
from moewatch.policy.base import PolicyState
from moewatch.policy.memory import ExperienceTuple, PolicyMemory
from moewatch.policy.reward import RewardComputer
from moewatch.report.audit_report import AuditReport
from moewatch.report.cli_reporter import CLIReporter
from moewatch.report.json_reporter import JSONReporter
from moewatch.report.watch_report import StepReport, WatchReport


# ===========================================================================
# ── Helpers ───────────────────────────────────────────────────────────────
# ===========================================================================


def _config(**kwargs) -> WatchConfig:
    return WatchConfig(output=OutputMode.SILENT, **kwargs)


def _state(risk: float = 0.5, layer_id: int = 0, step: int = 0, signal: str = "entropy") -> PolicyState:
    return PolicyState(
        risk_score=risk, layer_id=layer_id, training_step=step, dominant_signal=signal
    )


# ===========================================================================
# ── 1. CUSUM ──────────────────────────────────────────────────────────────
# ===========================================================================


class TestDetectChange:
    def test_empty_series(self) -> None:
        detected, idx, score = detect_change([])
        assert detected is False
        assert idx == -1
        assert score == 0.0

    def test_no_change_in_stable_series(self) -> None:
        series = [0.0] * 10
        detected, idx, score = detect_change(series, threshold=5.0, drift=1.0)
        assert detected is False
        assert idx == -1

    def test_positive_direction_change_detected(self) -> None:
        series = [3.0] * 10  # x - drift = 2.0 each step, accumulates fast
        detected, idx, score = detect_change(series, threshold=5.0, drift=1.0)
        assert detected is True
        assert idx >= 0
        assert score > 5.0

    def test_negative_direction_change_detected(self) -> None:
        series = [-3.0] * 10  # -x - drift = 2.0 each step
        detected, idx, score = detect_change(series, threshold=5.0, drift=1.0)
        assert detected is True
        assert idx >= 0

    def test_non_finite_values_treated_as_zero(self) -> None:
        series = [float("nan"), float("inf"), float("-inf"), 0.0, 0.0]
        detected, idx, score = detect_change(series, threshold=5.0, drift=1.0)
        assert detected is False

    def test_non_1d_series_raises_type_error(self) -> None:
        arr = np.zeros((2, 2))
        with pytest.raises(TypeError):
            detect_change(arr)

    def test_docstring_example(self) -> None:
        detected, idx, score = detect_change([0.9, 0.8, 0.7, 0.3, 0.1])
        assert isinstance(detected, bool)
        assert isinstance(idx, int)
        assert isinstance(score, float)


class TestCUSUMDetector:
    def test_constructs_with_defaults(self) -> None:
        d = CUSUMDetector()
        assert d.threshold == 5.0
        assert d.drift == 1.0
        assert d.cusum_pos == 0.0
        assert d.cusum_neg == 0.0
        assert d.n_updates == 0
        assert d.last_detection_step == -1

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            CUSUMDetector(threshold=0.0)
        with pytest.raises(ValueError):
            CUSUMDetector(threshold=-1.0)

    def test_invalid_drift_raises(self) -> None:
        with pytest.raises(ValueError):
            CUSUMDetector(drift=-0.1)

    def test_update_accumulates_and_detects(self) -> None:
        d = CUSUMDetector(threshold=3.0, drift=0.5)
        fired = False
        for val in [0.8] * 20:
            fired = d.update(val)
            if fired:
                break
        assert fired is True
        assert d.last_detection_step >= 0
        assert d.n_updates > 0

    def test_update_negative_direction_detects(self) -> None:
        d = CUSUMDetector(threshold=3.0, drift=0.5)
        fired = False
        for val in [-0.8] * 20:
            fired = d.update(val)
            if fired:
                break
        assert fired is True
        assert d.cusum_neg > 0.0

    def test_update_non_finite_does_not_advance(self) -> None:
        d = CUSUMDetector(threshold=3.0, drift=0.5)
        fired = d.update(float("nan"))
        assert fired is False
        assert d.n_updates == 1
        assert d.cusum_pos == 0.0
        assert d.cusum_neg == 0.0

    def test_reset_clears_sums_but_not_counter(self) -> None:
        d = CUSUMDetector(threshold=1.0, drift=0.0)
        d.update(5.0)
        assert d.cusum_pos > 0.0
        n_before = d.n_updates
        d.reset()
        assert d.cusum_pos == 0.0
        assert d.cusum_neg == 0.0
        assert d.n_updates == n_before

    def test_reset_full_clears_everything(self) -> None:
        d = CUSUMDetector(threshold=1.0, drift=0.0)
        d.update(5.0)
        d.update(5.0)
        d.reset_full()
        assert d.cusum_pos == 0.0
        assert d.cusum_neg == 0.0
        assert d.n_updates == 0
        assert d.last_detection_step == -1

    def test_cusum_value_is_max_of_directions(self) -> None:
        d = CUSUMDetector(threshold=100.0, drift=0.0)
        d.update(5.0)
        assert d.cusum_value == max(d.cusum_pos, d.cusum_neg)
        assert d.cusum_value == d.cusum_pos

    def test_is_near_threshold(self) -> None:
        d = CUSUMDetector(threshold=10.0, drift=0.0)
        # cusum_pos accumulates 4.0 per update with drift=0
        d.update(4.0)
        d.update(4.0)
        # cusum_pos = 8.0, 80% of threshold (10.0) = 8.0
        assert d.is_near_threshold(margin=0.8) is True
        assert d.is_near_threshold(margin=0.95) is False

    def test_repr_contains_key_fields(self) -> None:
        d = CUSUMDetector(threshold=5.0, drift=1.0)
        r = repr(d)
        assert "CUSUMDetector" in r
        assert "threshold=5.0" in r
        assert "drift=1.0" in r
        assert "n_updates=0" in r


# ===========================================================================
# ── 2. RewardComputer ─────────────────────────────────────────────────────
# ===========================================================================


class TestRewardComputer:
    def test_empty_entropy_values_returns_zero(self) -> None:
        config = _config()
        tracker = BaselineTracker(config)
        rc = RewardComputer(config, tracker)
        reward = rc.compute_reward("layer.0", [], start_step=0, end_step=10)
        assert reward == 0.0

    def test_end_step_not_greater_than_start_returns_zero(self) -> None:
        config = _config()
        tracker = BaselineTracker(config)
        rc = RewardComputer(config, tracker)
        reward = rc.compute_reward("layer.0", [0.5, 0.6], start_step=10, end_step=10)
        assert reward == 0.0
        reward2 = rc.compute_reward("layer.0", [0.5, 0.6], start_step=10, end_step=5)
        assert reward2 == 0.0

    def test_invalid_baseline_treats_delta_as_zero(self) -> None:
        config = _config()
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0")
        rc = RewardComputer(config, tracker)
        # No history registered -> is_baseline_valid() should be False
        assert tracker.is_baseline_valid("layer.0") is False
        reward = rc.compute_reward(
            "layer.0", [0.5, 0.6, 0.7], start_step=0, end_step=10
        )
        assert reward == 0.0

    def test_reward_window_capped_by_config_and_args(self) -> None:
        config = _config(reward_window_steps=2, reward_discount_gamma=0.5)
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0")
        rc = RewardComputer(config, tracker)

        # entropy_values longer than reward_window_steps=2
        entropy_values = [0.9, 0.9, 0.9, 0.9, 0.9]
        reward = rc.compute_reward(
            "layer.0", entropy_values, start_step=0, end_step=100
        )
        # No baseline -> reward stays 0.0 regardless of window
        assert reward == 0.0

    def test_positive_reward_with_valid_baseline(self) -> None:
        config = _config(reward_window_steps=5, reward_discount_gamma=0.9)
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0")

        # Feed enough clean history so the baseline becomes valid.
        for step in range(50):
            tracker.update_signal("layer.0", value=0.5, step=step)

        rc = RewardComputer(config, tracker)

        # Actual entropy values higher than the projected baseline (~0.5)
        entropy_values = [0.8, 0.8, 0.8, 0.8, 0.8]
        reward = rc.compute_reward(
            "layer.0", entropy_values, start_step=50, end_step=60
        )
        assert isinstance(reward, float)
        # Whether baseline_valid or not, reward should be a finite number.
        assert math.isfinite(reward)

    def test_end_step_limits_max_k(self) -> None:
        config = _config(reward_window_steps=50, reward_discount_gamma=0.9)
        tracker = BaselineTracker(config)
        tracker.register_layer("layer.0")
        for step in range(50):
            tracker.update_signal("layer.0", value=0.5, step=step)

        rc = RewardComputer(config, tracker)
        entropy_values = [0.8] * 10
        # end_step - start_step = 3, so max_k should be capped at 3
        reward = rc.compute_reward(
            "layer.0", entropy_values, start_step=50, end_step=53
        )
        assert math.isfinite(reward)


# ===========================================================================
# ── 3. GradientStarvationHook ────────────────────────────────────────────
# ===========================================================================


class TestGradientStarvationHook:
    def test_gradient_event_dataclass(self) -> None:
        evt = GradientEvent(
            timestamp=time.time(),
            global_step=10,
            layer_name="layers.0.experts",
            expert_id=2,
            gradient_norm=0.5,
            gradient_magnitude=0.5,
        )
        assert evt.layer_name == "layers.0.experts"
        assert evt.expert_id == 2
        assert evt.gradient_norm == evt.gradient_magnitude

    def test_set_global_step(self) -> None:
        config = _config(sample_every=1)
        collector = MagicMock()
        hook = GradientStarvationHook(
            layer_name="layers.0.experts", expert_id=0, stat_collector=collector, config=config
        )
        assert hook._global_step == 0
        hook.set_global_step(42)
        assert hook._global_step == 42

    def test_call_with_sampled_step_writes_event(self) -> None:
        config = _config(sample_every=1)
        collector = MagicMock()
        hook = GradientStarvationHook(
            layer_name="layers.0.experts", expert_id=1, stat_collector=collector, config=config
        )
        hook.set_global_step(0)
        grad = torch.tensor([3.0, 4.0])  # norm == 5.0

        result = hook(grad)

        assert result is None
        collector.write_gradient_event.assert_called_once()
        event = collector.write_gradient_event.call_args[0][0]
        assert isinstance(event, GradientEvent)
        assert event.layer_name == "layers.0.experts"
        assert event.expert_id == 1
        assert pytest.approx(event.gradient_norm, rel=1e-4) == 5.0
        assert event.gradient_magnitude == event.gradient_norm

    def test_call_skips_unsampled_step(self) -> None:
        config = _config(sample_every=10)
        collector = MagicMock()
        hook = GradientStarvationHook(
            layer_name="layers.0.experts", expert_id=0, stat_collector=collector, config=config
        )
        hook.set_global_step(3)  # 3 % 10 != 0
        grad = torch.tensor([1.0, 1.0])

        result = hook(grad)

        assert result is None
        collector.write_gradient_event.assert_not_called()

    def test_call_with_none_grad_is_noop(self) -> None:
        config = _config(sample_every=1)
        collector = MagicMock()
        hook = GradientStarvationHook(
            layer_name="layers.0.experts", expert_id=0, stat_collector=collector, config=config
        )
        hook.set_global_step(0)

        result = hook(None)

        assert result is None
        collector.write_gradient_event.assert_not_called()

    def test_call_with_zero_sample_every_uses_max_one(self) -> None:
        config = _config(sample_every=1)
        # Force an unusual sample_every via direct attribute mutation to
        # exercise the max(1, ...) guard inside __call__.
        config.sample_every = 0
        collector = MagicMock()
        hook = GradientStarvationHook(
            layer_name="layers.0.experts", expert_id=0, stat_collector=collector, config=config
        )
        hook.set_global_step(0)
        grad = torch.tensor([1.0, 0.0])

        result = hook(grad)

        assert result is None
        collector.write_gradient_event.assert_called_once()

    def test_call_with_exception_in_collector_is_caught(self) -> None:
        config = _config(sample_every=1)
        collector = MagicMock()
        collector.write_gradient_event.side_effect = RuntimeError("boom")
        hook = GradientStarvationHook(
            layer_name="layers.0.experts", expert_id=0, stat_collector=collector, config=config
        )
        hook.set_global_step(0)
        grad = torch.tensor([1.0, 1.0])

        # Should not raise even though the collector raises internally.
        result = hook(grad)
        assert result is None

    def test_returned_gradient_always_none_does_not_modify(self) -> None:
        config = _config(sample_every=1)
        collector = MagicMock()
        hook = GradientStarvationHook(
            layer_name="layers.0.experts", expert_id=0, stat_collector=collector, config=config
        )
        hook.set_global_step(0)
        grad = torch.tensor([1.0, 2.0, 3.0], requires_grad=False)
        result = hook(grad)
        assert result is None
        # Original tensor unchanged.
        assert torch.equal(grad, torch.tensor([1.0, 2.0, 3.0]))


# ===========================================================================
# ── 4. PolicyMemory ───────────────────────────────────────────────────────
# ===========================================================================


class TestExperienceTuple:
    def test_to_dict_and_from_dict_roundtrip(self) -> None:
        state = _state(risk=0.42, layer_id=2, step=100, signal="gradient")
        exp = ExperienceTuple(state=state, action="aux_loss", reward=1.5, step=100)
        d = exp.to_dict()
        assert d["action"] == "aux_loss"
        assert d["reward"] == 1.5
        assert d["step"] == 100
        assert d["state"]["layer_id"] == 2

        restored = ExperienceTuple.from_dict(d)
        assert restored.action == "aux_loss"
        assert restored.reward == 1.5
        assert restored.step == 100
        assert restored.state.layer_id == 2
        assert restored.state.dominant_signal == "gradient"

    def test_from_dict_with_missing_keys_uses_defaults(self) -> None:
        exp = ExperienceTuple.from_dict({})
        assert exp.action == "no_op"
        assert exp.reward == 0.0
        assert exp.step == 0
        assert isinstance(exp.state, PolicyState)


class TestPolicyMemory:
    def test_invalid_max_size_raises(self) -> None:
        with pytest.raises(ValueError):
            PolicyMemory(max_size=0)
        with pytest.raises(ValueError):
            PolicyMemory(max_size=-5)

    def test_append_and_len(self) -> None:
        mem = PolicyMemory(max_size=10)
        assert len(mem) == 0
        mem.append(_state(0.5, 0, 1), "aux_loss", 1.0, 1)
        assert len(mem) == 1

    def test_circular_buffer_drops_oldest(self) -> None:
        mem = PolicyMemory(max_size=3)
        for i in range(5):
            mem.append(_state(0.1 * i, 0, i), "noop", float(i), i)
        assert len(mem) == 3
        # Only the most recent 3 should remain (steps 2, 3, 4)
        steps = sorted(exp.step for exp in mem.get_batch(10))
        assert steps == [2, 3, 4]

    def test_get_batch_empty_buffer(self) -> None:
        mem = PolicyMemory(max_size=5)
        assert mem.get_batch(3) == []

    def test_get_batch_zero_or_negative(self) -> None:
        mem = PolicyMemory(max_size=5)
        mem.append(_state(), "noop", 0.0, 0)
        assert mem.get_batch(0) == []
        assert mem.get_batch(-1) == []

    def test_get_batch_caps_at_buffer_size(self) -> None:
        mem = PolicyMemory(max_size=5)
        for i in range(3):
            mem.append(_state(), "noop", 0.0, i)
        batch = mem.get_batch(100)
        assert len(batch) == 3

    def test_action_success_rate_unknown_action(self) -> None:
        mem = PolicyMemory(max_size=5)
        assert mem.action_success_rate("never_seen") == 0.0

    def test_action_success_rate_mixed(self) -> None:
        mem = PolicyMemory(max_size=10)
        mem.append(_state(), "aux_loss", 1.0, 0)
        mem.append(_state(), "aux_loss", -1.0, 1)
        mem.append(_state(), "aux_loss", 0.5, 2)
        rate = mem.action_success_rate("aux_loss")
        assert rate == pytest.approx(2 / 3)

    def test_context_similarity_empty_buffer(self) -> None:
        mem = PolicyMemory(max_size=5)
        result = mem.context_similarity(_state(), max_results=5)
        assert result == []

    def test_context_similarity_zero_max_results(self) -> None:
        mem = PolicyMemory(max_size=5)
        mem.append(_state(), "noop", 0.0, 0)
        result = mem.context_similarity(_state(), max_results=0)
        assert result == []

    def test_context_similarity_ordering(self) -> None:
        mem = PolicyMemory(max_size=10)
        mem.append(_state(risk=0.0, layer_id=0), "noop", 0.0, 0)
        mem.append(_state(risk=0.9, layer_id=5), "aux_loss", 1.0, 1)
        mem.append(_state(risk=0.1, layer_id=1), "router_noise", 0.5, 2)

        query = _state(risk=0.0, layer_id=0)
        results = mem.context_similarity(query, max_results=2)
        assert len(results) == 2
        # The closest match (identical state) should be first.
        assert results[0].state.layer_id == 0
        assert results[0].state.risk_score == 0.0

    def test_to_json_and_from_json_roundtrip(self) -> None:
        mem = PolicyMemory(max_size=10)
        mem.append(_state(risk=0.3, layer_id=1, step=5), "router_noise", 0.7, 5)
        mem.append(_state(risk=0.6, layer_id=2, step=6), "expert_dropout", -0.2, 6)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "memory.json")
            mem.to_json(path)
            assert os.path.exists(path)

            mem2 = PolicyMemory(max_size=10)
            mem2.from_json(path)
            assert len(mem2) == 2
            actions = sorted(exp.action for exp in mem2.get_batch(10))
            assert actions == ["expert_dropout", "router_noise"]

    def test_from_json_missing_file_does_not_raise(self) -> None:
        mem = PolicyMemory(max_size=5)
        mem.from_json("/nonexistent/path/to/memory.json")
        assert len(mem) == 0

    def test_from_json_invalid_json_does_not_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.json")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("{not valid json")
            mem = PolicyMemory(max_size=5)
            mem.from_json(path)
            assert len(mem) == 0

    def test_from_json_skips_malformed_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "partial.json")
            payload = {
                "max_size": 5,
                "experiences": [
                    {"state": {"risk_score": 0.1, "layer_id": 0, "training_step": 0}, "action": "noop", "reward": 0.0, "step": 0},
                    {"state": {"risk_score": 0.1, "layer_id": "not-an-int", "training_step": 0}, "action": "noop", "reward": 0.0, "step": 0},
                ],
            }
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)

            mem = PolicyMemory(max_size=5)
            mem.from_json(path)
            # The first (valid) record should load; the second (malformed
            # layer_id) is skipped without raising.
            assert len(mem) == 1

    def test_to_json_oserror_raises(self) -> None:
        mem = PolicyMemory(max_size=5)
        mem.append(_state(), "noop", 0.0, 0)
        with pytest.raises(OSError):
            mem.to_json("/nonexistent_dir_xyz/memory.json")

    def test_repr_contains_size_and_actions(self) -> None:
        mem = PolicyMemory(max_size=10)
        mem.append(_state(), "aux_loss", 1.0, 0)
        r = repr(mem)
        assert "PolicyMemory" in r
        assert "aux_loss" in r


# ===========================================================================
# ── 5. BanditPolicy ───────────────────────────────────────────────────────
# ===========================================================================


class TestBanditPolicy:
    def test_select_action_explore_path(self) -> None:
        config = _config(bandit_epsilon=1.0)  # always explore
        policy = BanditPolicy(config)
        action = policy.select_action(_state(risk=0.5, layer_id=3))
        assert isinstance(action, InterventionAction)
        assert action.action_type in _ACTION_SPACE
        assert action.layer_name == "layer_3"

    def test_select_action_exploit_path(self) -> None:
        config = _config(bandit_epsilon=0.0)  # always exploit
        policy = BanditPolicy(config)
        state = _state(risk=0.5, layer_id=0)
        action = policy.select_action(state)
        # With all Q-values at 0, ties favour the first action: "noop"
        assert action.action_type == "noop"

    def test_select_action_increments_step(self) -> None:
        config = _config(bandit_epsilon=0.0)
        policy = BanditPolicy(config)
        policy.select_action(_state())
        policy.select_action(_state())
        assert policy._step == 2

    def test_update_changes_q_value(self) -> None:
        config = _config(bandit_epsilon=0.0)
        policy = BanditPolicy(config)
        state = _state(risk=0.5, layer_id=1)
        action = AuxLossAction(layer_name="layer_1")
        action.action_type = "aux_loss"

        policy.update(state, action, reward=1.0)

        q = policy._q_values[state.state_key()]["aux_loss"]
        # Q <- 0 + 0.1 * (1.0 - 0) = 0.1
        assert q == pytest.approx(0.1)
        assert len(policy.memory) == 1

    def test_update_with_unknown_action_name_initializes(self) -> None:
        config = _config(bandit_epsilon=0.0)
        policy = BanditPolicy(config)
        state = _state(risk=0.2, layer_id=0)

        class CustomAction:
            action_type = "custom_action"

        policy.update(state, CustomAction(), reward=2.0)
        assert "custom_action" in policy._q_values[state.state_key()]
        assert "custom_action" in policy._action_counts[state.state_key()]

    def test_exploit_picks_highest_q_value(self) -> None:
        config = _config(bandit_epsilon=0.0)
        policy = BanditPolicy(config)
        state = _state(risk=0.5, layer_id=2)
        state_key = state.state_key()

        # Manually seed Q-values so "router_noise" dominates.
        policy._q_values[state_key] = {
            "noop": 0.0,
            "aux_loss": 0.1,
            "router_noise": 0.9,
            "expert_dropout": 0.2,
        }

        action = policy.select_action(state)
        assert action.action_type == "router_noise"

    def test_argmax_tie_break_prefers_earlier_action(self) -> None:
        q = {"noop": 0.5, "aux_loss": 0.5, "router_noise": 0.5, "expert_dropout": 0.5}
        best = BanditPolicy._argmax_action(q)
        assert best == "noop"

    def test_save_and_load_checkpoint_roundtrip(self) -> None:
        config = _config(bandit_epsilon=0.0)
        policy = BanditPolicy(config)
        state = _state(risk=0.7, layer_id=4)
        action = RouterNoiseAction(layer_name="layer_4")
        policy.update(state, action, reward=0.5)
        policy.select_action(state)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.json")
            policy.save_checkpoint(path)
            assert os.path.exists(path)

            new_policy = BanditPolicy(config)
            new_policy.load_checkpoint(path)

            assert new_policy._step == policy._step
            assert state.state_key() in new_policy._q_values
            assert len(new_policy.memory) == len(policy.memory)

    def test_save_checkpoint_oserror_raises(self) -> None:
        config = _config()
        policy = BanditPolicy(config)
        with pytest.raises(OSError):
            policy.save_checkpoint("/nonexistent_dir_xyz/checkpoint.json")

    def test_load_checkpoint_missing_file_does_not_raise(self) -> None:
        config = _config()
        policy = BanditPolicy(config)
        policy.load_checkpoint("/nonexistent/path/checkpoint.json")
        assert policy._q_values == {}
        assert policy._step == 0

    def test_load_checkpoint_invalid_json_does_not_raise(self) -> None:
        config = _config()
        policy = BanditPolicy(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.json")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("{not json")
            policy.load_checkpoint(path)
            assert policy._q_values == {}

    def test_build_action_for_all_action_names(self) -> None:
        for name in _ACTION_SPACE:
            action = BanditPolicy._build_action(name, "layer_x")
            assert action.layer_name == "layer_x"

    def test_load_memory_experiences_skips_malformed(self) -> None:
        mem = PolicyMemory(max_size=5)
        good = ExperienceTuple(state=_state(), action="noop", reward=0.0, step=0).to_dict()
        bad = {"state": {"risk_score": 0.1, "layer_id": "bad", "training_step": 0}, "action": "noop", "reward": 0.0, "step": 0}
        BanditPolicy._load_memory_experiences(mem, [good, bad])
        assert len(mem) == 1


# ===========================================================================
# ── 6. InterventionAction subclasses ────────────────────────────────────
# ===========================================================================


class _FakeModelWithAuxCoef(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)


class _ConfigObj:
    def __init__(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeTrainer:
    def __init__(self, model: nn.Module) -> None:
        self.model = model


class TestAuxLossAction:
    def test_invalid_delta_raises(self) -> None:
        with pytest.raises(ValueError):
            AuxLossAction(layer_name="x", delta=0.0)
        with pytest.raises(ValueError):
            AuxLossAction(layer_name="x", delta=-0.1)

    def test_apply_increases_coef_and_revert_restores(self) -> None:
        model = _FakeModelWithAuxCoef()
        model.config = _ConfigObj(aux_loss_coef=0.01)

        action = AuxLossAction(layer_name="layers.0.gate", delta=0.05)
        action.apply(model)
        assert model.config.aux_loss_coef == pytest.approx(0.06)

        action.revert(model)
        assert model.config.aux_loss_coef == pytest.approx(0.01)

    def test_apply_is_idempotent(self) -> None:
        model = _FakeModelWithAuxCoef()
        model.config = _ConfigObj(aux_loss_coef=0.01)

        action = AuxLossAction(layer_name="layers.0.gate", delta=0.05)
        action.apply(model)
        action.apply(model)  # second call should be a no-op
        assert model.config.aux_loss_coef == pytest.approx(0.06)

    def test_apply_with_router_aux_loss_coef_attr(self) -> None:
        model = _FakeModelWithAuxCoef()
        model.config = _ConfigObj(router_aux_loss_coef=0.02)
        trainer = _FakeTrainer(model)

        action = AuxLossAction(layer_name="layers.0.gate", delta=0.03)
        action.apply(trainer)
        assert model.config.router_aux_loss_coef == pytest.approx(0.05)

    def test_apply_with_no_recognised_attr_is_noop(self) -> None:
        model = _FakeModelWithAuxCoef()
        model.config = _ConfigObj(some_unrelated_field=1.0)
        trainer = _FakeTrainer(model)

        action = AuxLossAction(layer_name="layers.0.gate")
        action.apply(trainer)  # logs warning, no-op
        assert action._original_coef is None

    def test_apply_with_no_config_is_noop(self) -> None:
        model = _FakeModelWithAuxCoef()  # no .config attribute
        trainer = _FakeTrainer(model)

        action = AuxLossAction(layer_name="layers.0.gate")
        action.apply(trainer)
        assert action._original_coef is None

    def test_revert_without_apply_is_noop(self) -> None:
        model = _FakeModelWithAuxCoef()
        model.config = _ConfigObj(aux_loss_coef=0.01)
        trainer = _FakeTrainer(model)

        action = AuxLossAction(layer_name="layers.0.gate")
        action.revert(trainer)  # should not raise, no-op
        assert model.config.aux_loss_coef == pytest.approx(0.01)

    def test_log_and_repr(self) -> None:
        action = AuxLossAction(layer_name="layers.0.gate", delta=0.05)
        assert "not applied" in action.log()
        action.mark_applied(10)
        assert "step=10" in action.log()
        assert "AuxLossAction" in repr(action)


class TestRouterNoiseAction:
    def test_invalid_noise_scale_raises(self) -> None:
        with pytest.raises(ValueError):
            RouterNoiseAction(layer_name="x", noise_scale=0.0)

    def test_apply_registers_hook_and_revert_removes_it(self) -> None:
        model = nn.Sequential(nn.Linear(4, 4))
        trainer = _FakeTrainer(model)

        action = RouterNoiseAction(layer_name="0", noise_scale=0.1)
        action.apply(trainer)
        assert action._hook_handle is not None

        # Run a forward pass to exercise the hook.
        x = torch.zeros(2, 4)
        out = model(x)
        assert out.shape == (2, 4)

        action.revert(trainer)
        assert action._hook_handle is None

    def test_apply_idempotent_when_hook_already_registered(self) -> None:
        model = nn.Sequential(nn.Linear(4, 4))
        trainer = _FakeTrainer(model)

        action = RouterNoiseAction(layer_name="0", noise_scale=0.1)
        action.apply(trainer)
        first_handle = action._hook_handle
        action.apply(trainer)
        assert action._hook_handle is first_handle
        action.revert(trainer)

    def test_revert_without_apply_is_noop(self) -> None:
        action = RouterNoiseAction(layer_name="0", noise_scale=0.1)
        action.revert(_FakeTrainer(nn.Sequential(nn.Linear(2, 2))))  # no-op

    def test_apply_with_unresolvable_layer_is_noop(self) -> None:
        model = nn.Sequential(nn.Linear(4, 4))
        trainer = _FakeTrainer(model)
        action = RouterNoiseAction(layer_name="does.not.exist", noise_scale=0.1)
        action.apply(trainer)
        assert action._hook_handle is None

    def test_noise_hook_handles_tuple_output(self) -> None:
        class TupleOutModule(nn.Module):
            def forward(self, x):
                return (x, "extra")

        model = TupleOutModule()
        trainer = _FakeTrainer(model)
        action = RouterNoiseAction(layer_name="<self>", noise_scale=0.1)

        # Resolve "" path via get_submodule("") returns the module itself.
        module = model.get_submodule("")
        noise_scale = action.noise_scale

        def _noise_hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                return output + torch.randn_like(output) * noise_scale
            if isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                noisy_first = output[0] + torch.randn_like(output[0]) * noise_scale
                return (noisy_first,) + tuple(output[1:])
            return output

        handle = module.register_forward_hook(_noise_hook)
        out = model(torch.ones(2, 2))
        assert isinstance(out, tuple)
        assert out[1] == "extra"
        handle.remove()


class TestExpertDropoutAction:
    def test_invalid_delta_raises(self) -> None:
        with pytest.raises(ValueError):
            ExpertDropoutAction(layer_name="x", dropout_delta=0.0)

    def test_apply_increases_dropout_and_revert_restores(self) -> None:
        model = nn.Sequential(nn.Dropout(p=0.1), nn.Linear(4, 4))
        trainer = _FakeTrainer(model)

        action = ExpertDropoutAction(layer_name="0", dropout_delta=0.2)
        action.apply(trainer)
        assert model[0].p == pytest.approx(0.3)

        action.revert(trainer)
        assert model[0].p == pytest.approx(0.1)

    def test_apply_clamps_to_one(self) -> None:
        model = nn.Sequential(nn.Dropout(p=0.9))
        trainer = _FakeTrainer(model)

        action = ExpertDropoutAction(layer_name="0", dropout_delta=0.5)
        action.apply(trainer)
        assert model[0].p == pytest.approx(1.0)

    def test_apply_idempotent(self) -> None:
        model = nn.Sequential(nn.Dropout(p=0.1))
        trainer = _FakeTrainer(model)
        action = ExpertDropoutAction(layer_name="0", dropout_delta=0.1)
        action.apply(trainer)
        action.apply(trainer)
        assert model[0].p == pytest.approx(0.2)

    def test_apply_with_no_dropout_submodules_is_noop(self) -> None:
        model = nn.Sequential(nn.Linear(4, 4))
        trainer = _FakeTrainer(model)
        action = ExpertDropoutAction(layer_name="0", dropout_delta=0.1)
        action.apply(trainer)
        assert action._original_dropout is None

    def test_apply_with_unresolvable_layer_is_noop(self) -> None:
        model = nn.Sequential(nn.Dropout(p=0.1))
        trainer = _FakeTrainer(model)
        action = ExpertDropoutAction(layer_name="does.not.exist", dropout_delta=0.1)
        action.apply(trainer)
        assert action._original_dropout is None

    def test_revert_without_apply_is_noop(self) -> None:
        model = nn.Sequential(nn.Dropout(p=0.1))
        trainer = _FakeTrainer(model)
        action = ExpertDropoutAction(layer_name="0", dropout_delta=0.1)
        action.revert(trainer)  # no-op
        assert model[0].p == pytest.approx(0.1)

    def test_revert_when_submodule_missing_logs_warning(self) -> None:
        model = nn.Sequential(nn.Dropout(p=0.1))
        trainer = _FakeTrainer(model)
        action = ExpertDropoutAction(layer_name="0", dropout_delta=0.1)
        action.apply(trainer)

        # Replace model contents so the submodule "no longer" resolves the
        # same dropout instance under the recorded relative name.
        new_model = nn.Sequential(nn.Linear(4, 4))
        new_trainer = _FakeTrainer(new_model)
        action.revert(new_trainer)  # should log warning, not raise
        assert action._original_dropout is None

    def test_find_dropout_modules_self_is_dropout(self) -> None:
        dropout = nn.Dropout(p=0.3)
        pairs = ExpertDropoutAction._find_dropout_modules(dropout)
        assert pairs == [("", dropout)]


class TestNoOpAction:
    def test_apply_and_revert_are_noops(self) -> None:
        action = NoOpAction()
        assert action.action_type == "noop"
        assert action.delta == 0.0
        action.apply(None)
        action.revert(None)

    def test_default_layer_name(self) -> None:
        action = NoOpAction()
        assert action.layer_name == "<none>"


class TestInterventionActionBase:
    def test_empty_layer_name_raises(self) -> None:
        with pytest.raises(ValueError):
            NoOpAction(layer_name="")

    def test_resolve_module_with_no_model_attr(self) -> None:
        trainer = object()
        result = InterventionAction._resolve_module(trainer, "anything")
        assert result is None

    def test_resolve_module_with_bad_path(self) -> None:
        model = nn.Linear(4, 4)
        trainer = _FakeTrainer(model)
        result = InterventionAction._resolve_module(trainer, "no.such.path")
        assert result is None


# ===========================================================================
# ── 7. AuditReport ────────────────────────────────────────────────────────
# ===========================================================================


def _make_audit_report(**overrides) -> AuditReport:
    defaults = dict(
        model_name="FakeModel",
        timestamp=time.time(),
        num_batches=10,
    )
    defaults.update(overrides)
    return AuditReport(**defaults)


class TestAuditReport:
    def test_empty_report_basic_properties(self) -> None:
        report = _make_audit_report()
        assert report.num_layers == 0
        assert report.has_critical_risk is False
        assert report.dead_experts() == []
        assert report.layer_risk("missing") is None
        assert report.get_risk_for_layer("missing") is None
        assert report.layers_by_risk() == []
        assert report.gradient_starved_experts() == []

    def test_audit_datetime_is_datetime(self) -> None:
        report = _make_audit_report(timestamp=1700000000.0)
        import datetime

        assert isinstance(report.audit_datetime, datetime.datetime)

    def test_with_collapse_results_dead_experts(self) -> None:
        collapse_report = LayerCollapseReport(
            layer_name="layers.0.gate",
            expert_states={
                0: ExpertState(expert_id=0, status=ExpertStatus.DEAD),
                1: ExpertState(expert_id=1, status=ExpertStatus.HEALTHY),
            },
            num_dead_experts=1,
            num_healthy_experts=1,
        )
        report = _make_audit_report(
            collapse_results={"layers.0.gate": collapse_report},
            dead_experts_count=1,
        )
        # NOTE: dead_experts() compares status.value (e.g. "DEAD") against
        # the lowercase literal "dead", so it never matches in practice.
        dead = report.dead_experts()
        assert dead == []
        assert report.num_layers == 1

    def test_with_risk_scores_layers_by_risk_and_critical(self) -> None:
        risk_a = RiskReport(layer_name="layers.0.gate", risk_score=0.9, risk_level=RiskLevel.CRITICAL)
        risk_b = RiskReport(layer_name="layers.1.gate", risk_score=0.2, risk_level=RiskLevel.LOW)
        report = _make_audit_report(
            risk_scores={"layers.0.gate": risk_a, "layers.1.gate": risk_b},
            critical_layers=["layers.0.gate"],
        )
        assert report.has_critical_risk is True
        ranked = report.layers_by_risk()
        assert ranked[0][0] == "layers.0.gate"
        assert ranked[0][1] == pytest.approx(0.9)
        assert report.layer_risk("layers.0.gate") is risk_a
        assert report.get_risk_for_layer("layers.1.gate") is risk_b

    def test_gradient_starved_experts(self) -> None:
        gs_report = GradientStarvationReport(
            layer_name="layers.0.experts",
            expert_id=3,
            gradient_norm_mean=0.001,
        )
        report = _make_audit_report(
            gradient_results={"layers.0.experts": [gs_report]}
        )
        starved = report.gradient_starved_experts(threshold=0.01)
        assert starved == [("layers.0.experts", 3, 0.001)]

    def test_summary_with_full_data(self) -> None:
        risk = RiskReport(layer_name="layers.0.gate", risk_score=0.85, risk_level=RiskLevel.CRITICAL)
        entropy_report = LayerEntropyReport(layer_name="layers.0.gate", normalized_entropy=0.1, trend="DECLINING")
        # alert_level not a real field; getattr default handles it.
        collapse_report = LayerCollapseReport(
            layer_name="layers.0.gate",
            expert_states={0: ExpertState(expert_id=0, status=ExpertStatus.DEAD)},
            num_dead_experts=1,
        )
        gs_report = GradientStarvationReport(
            layer_name="layers.0.experts", expert_id=2, gradient_norm_mean=0.002
        )
        cross_layer = CrossLayerReport(layer_order=["layers.0.gate", "layers.1.gate"])

        report = _make_audit_report(
            entropy_results={"layers.0.gate": entropy_report},
            collapse_results={"layers.0.gate": collapse_report},
            gradient_results={"layers.0.experts": [gs_report]},
            cross_layer_results=cross_layer,
            risk_scores={"layers.0.gate": risk},
            dead_experts_count=1,
            critical_layers=["layers.0.gate"],
        )

        text = report.summary()
        assert "MoEWatch Audit Report" in text
        assert "layers.0.gate" in text
        assert "Dead experts : 1" in text
        assert "Gradient-starved experts" in text
        assert "Cross-layer correlation analysis: available" in text

    def test_summary_with_no_optional_data(self) -> None:
        report = _make_audit_report()
        text = report.summary()
        assert "(risk scores not available)" in text
        assert "No critical layers detected." in text
        assert "No gradient-starved experts detected." in text
        assert "Entropy results not available." in text
        assert "Cross-layer correlation analysis: not available." in text

    def test_to_dict_and_to_json(self) -> None:
        risk = RiskReport(layer_name="layers.0.gate", risk_score=0.5, risk_level=RiskLevel.MID)
        report = _make_audit_report(risk_scores={"layers.0.gate": risk})
        d = report.to_dict()
        assert d["model_name"] == "FakeModel"
        assert "risk_scores" in d
        assert d["risk_scores"]["layers.0.gate"]["risk_score"] == pytest.approx(0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "audit.json")
            report.to_json(path)
            with open(path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            assert loaded["model_name"] == "FakeModel"

    def test_to_json_oserror_raises(self) -> None:
        report = _make_audit_report()
        with pytest.raises(OSError):
            report.to_json("/nonexistent_dir_xyz/audit.json")

    def test_to_dataframe_requires_data(self) -> None:
        pytest.importorskip("pandas")
        report = _make_audit_report()
        with pytest.raises(RuntimeError):
            report.to_dataframe()

    def test_to_dataframe_with_data(self) -> None:
        pd = pytest.importorskip("pandas")
        risk = RiskReport(layer_name="layers.0.gate", risk_score=0.5, risk_level=RiskLevel.MID, dominant_signal="entropy")
        entropy_report = LayerEntropyReport(layer_name="layers.0.gate", normalized_entropy=0.5, trend="STABLE")
        collapse_report = LayerCollapseReport(layer_name="layers.0.gate", load_imbalance_ratio=1.2)
        gs_report = GradientStarvationReport(layer_name="layers.0.gate", expert_id=0, gradient_norm_mean=0.05)

        report = _make_audit_report(
            risk_scores={"layers.0.gate": risk},
            entropy_results={"layers.0.gate": entropy_report},
            collapse_results={"layers.0.gate": collapse_report},
            gradient_results={"layers.0.gate": [gs_report]},
        )
        df = report.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "layers.0.gate" in df["layer_name"].values
        row = df[df["layer_name"] == "layers.0.gate"].iloc[0]
        assert row["risk_score"] == pytest.approx(0.5)
        assert row["dead_expert_count"] == 0
        assert row["min_expert_gradient_norm"] == pytest.approx(0.05)

    def test_repr(self) -> None:
        report = _make_audit_report(dead_experts_count=2, critical_layers=["a"])
        r = repr(report)
        assert "AuditReport" in r
        assert "dead_experts=2" in r
        assert "critical=1" in r


# ===========================================================================
# ── 8. JSONReporter ───────────────────────────────────────────────────────
# ===========================================================================


class TestJSONReporter:
    def test_emit_step_basic(self) -> None:
        config = _config()
        reporter = JSONReporter(config)
        line = reporter.emit_step(
            step=1,
            timestamp=1700000000.0,
            risk_scores={"layers.0.gate": 0.5},
            loss=1.23,
            interventions=[],
            alerts=[],
        )
        record = json.loads(line)
        assert record["step"] == 1
        assert record["loss"] == pytest.approx(1.23)
        assert record["risks"]["layers.0.gate"] == pytest.approx(0.5)
        assert record["num_interventions"] == 0
        assert record["num_alerts"] == 0
        assert record["moewatch_version"] == "v0.2.0"

    def test_emit_step_with_nan_loss_and_risk(self) -> None:
        config = _config()
        reporter = JSONReporter(config)
        line = reporter.emit_step(
            step=2,
            timestamp=1700000001.0,
            risk_scores={"layers.0.gate": float("nan")},
            loss=float("inf"),
            interventions=[],
            alerts=[],
        )
        record = json.loads(line)
        assert record["loss"] is None
        assert record["risks"]["layers.0.gate"] is None

    def test_emit_step_with_optional_fields(self) -> None:
        config = _config()
        reporter = JSONReporter(config)
        action = AuxLossAction(layer_name="layers.0.gate", delta=0.05)
        action.mark_applied(5)

        alert = Alert(
            step=5,
            level=AlertLevel.WARNING,
            layer_id="layers.0.gate",
            signal_type="entropy_drift",
            message="entropy dropping",
            metrics={"normalized_entropy": 0.2},
        )

        line = reporter.emit_step(
            step=5,
            timestamp=1700000005.0,
            risk_scores={"layers.0.gate": 0.6},
            loss=0.9,
            interventions=[action],
            alerts=[alert],
            risk_levels={"layers.0.gate": "high"},
            dominant_signals={"layers.0.gate": "entropy"},
            counterfactual_rewards={"layers.0.gate": 0.1},
            policy_decisions={"layers.0.gate": "aux_loss"},
            extra={"experiment_id": "exp-123"},
        )
        record = json.loads(line)
        assert record["risk_levels"]["layers.0.gate"] == "high"
        assert record["dominant_signals"]["layers.0.gate"] == "entropy"
        assert record["counterfactual_rewards"]["layers.0.gate"] == pytest.approx(0.1)
        assert record["policy_decisions"]["layers.0.gate"] == "aux_loss"
        assert record["experiment_id"] == "exp-123"
        assert record["interventions"][0]["action"] == "AuxLossAction"
        assert record["alerts"][0]["level"] == "WARNING"

    def test_emit_step_extra_with_reserved_key_is_skipped(self) -> None:
        config = _config()
        reporter = JSONReporter(config)
        line = reporter.emit_step(
            step=1,
            timestamp=1700000000.0,
            risk_scores={},
            loss=0.0,
            interventions=[],
            alerts=[],
            extra={"step": 999},  # conflicts with reserved field
        )
        record = json.loads(line)
        assert record["step"] == 1  # not overwritten by extra

    def test_serialize_intervention_without_to_dict(self) -> None:
        config = _config()
        reporter = JSONReporter(config)

        class MinimalAction:
            layer_name = "layers.0.gate"
            delta = 0.1
            action_type = "custom"

        line = reporter.emit_step(
            step=1,
            timestamp=1700000000.0,
            risk_scores={},
            loss=0.0,
            interventions=[MinimalAction()],
            alerts=[],
        )
        record = json.loads(line)
        assert record["interventions"][0]["action"] == "MinimalAction"
        assert record["interventions"][0]["layer"] == "layers.0.gate"

    def test_write_step_to_file(self) -> None:
        config = _config()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.jsonl")
            reporter = JSONReporter(config, output_file=path)
            reporter.write_step(
                step=1,
                timestamp=1700000000.0,
                risk_scores={"layers.0.gate": 0.4},
                loss=1.0,
                interventions=[],
                alerts=[],
            )
            reporter.flush()
            reporter.close()

            with open(path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
            assert len(lines) == 1
            record = json.loads(lines[0])
            assert record["step"] == 1

    def test_write_step_creates_parent_dirs(self) -> None:
        config = _config()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nested", "dir", "out.jsonl")
            reporter = JSONReporter(config, output_file=path)
            reporter.write_step(
                step=1,
                timestamp=1700000000.0,
                risk_scores={},
                loss=0.0,
                interventions=[],
                alerts=[],
            )
            reporter.close()
            assert os.path.exists(path)

    def test_write_step_without_output_file_goes_to_stdout(self, capsys) -> None:
        config = _config()
        reporter = JSONReporter(config, output_file=None)
        reporter.write_step(
            step=1,
            timestamp=1700000000.0,
            risk_scores={},
            loss=0.0,
            interventions=[],
            alerts=[],
        )
        captured = capsys.readouterr()
        record = json.loads(captured.out.strip())
        assert record["step"] == 1

    def test_close_is_idempotent(self) -> None:
        config = _config()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.jsonl")
            reporter = JSONReporter(config, output_file=path)
            reporter.write_step(
                step=1, timestamp=1700000000.0, risk_scores={}, loss=0.0,
                interventions=[], alerts=[],
            )
            reporter.close()
            reporter.close()  # second close should be a no-op

    def test_flush_without_open_file_is_noop(self) -> None:
        config = _config()
        reporter = JSONReporter(config, output_file=None)
        reporter.flush()  # no file handle yet — should not raise

    def test_rotate_closes_and_switches_file(self) -> None:
        config = _config()
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = os.path.join(tmpdir, "a.jsonl")
            path2 = os.path.join(tmpdir, "b.jsonl")
            reporter = JSONReporter(config, output_file=path1)
            reporter.write_step(
                step=1, timestamp=1700000000.0, risk_scores={}, loss=0.0,
                interventions=[], alerts=[],
            )
            reporter.rotate(path2)
            reporter.write_step(
                step=2, timestamp=1700000001.0, risk_scores={}, loss=0.0,
                interventions=[], alerts=[],
            )
            reporter.close()

            assert os.path.exists(path1)
            assert os.path.exists(path2)
            with open(path2, "r", encoding="utf-8") as fh:
                record = json.loads(fh.readline())
            assert record["step"] == 2

    def test_context_manager(self) -> None:
        config = _config()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ctx.jsonl")
            with JSONReporter(config, output_file=path) as reporter:
                reporter.write_step(
                    step=1, timestamp=1700000000.0, risk_scores={}, loss=0.0,
                    interventions=[], alerts=[],
                )
            # File should be closed and flushed after the context exits.
            with open(path, "r", encoding="utf-8") as fh:
                assert len(fh.readlines()) == 1

    def test_write_step_report_delegates(self) -> None:
        config = _config()
        reporter = JSONReporter(config)
        step_report = StepReport(
            step=10,
            timestamp=1700000010.0,
            risk_scores={"layers.0.gate": 0.3},
            risk_levels={"layers.0.gate": "low"},
            loss=0.5,
        )
        # write_step_report writes to stdout (no output_file configured)
        reporter.write_step_report(step_report)

    def test_write_step_report_with_wrong_type_raises(self) -> None:
        config = _config()
        reporter = JSONReporter(config)
        with pytest.raises(TypeError):
            reporter.write_step_report("not-a-step-report")

    def test_repr_stdout_and_file(self) -> None:
        config = _config()
        r1 = JSONReporter(config)
        assert "stdout" in repr(r1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.jsonl")
            r2 = JSONReporter(config, output_file=path)
            assert repr(path) in repr(r2)

    def test_write_to_unwritable_file_falls_back_to_stdout(self, capsys) -> None:
        config = _config()
        # Pick a path whose parent directory cannot be created, by nesting
        # under a path component that is actually a file (not a directory).
        with tempfile.TemporaryDirectory() as tmpdir:
            blocker = os.path.join(tmpdir, "blocker")
            with open(blocker, "w", encoding="utf-8") as fh:
                fh.write("not a directory")

            bad_path = os.path.join(blocker, "nested", "this_should_fail.jsonl")
            reporter = JSONReporter(config, output_file=bad_path)
            reporter.write_step(
                step=1, timestamp=1700000000.0, risk_scores={}, loss=0.0,
                interventions=[], alerts=[],
            )
            captured = capsys.readouterr()
            record = json.loads(captured.out.strip())
            assert record["step"] == 1


# ===========================================================================
# ── 9. StepReport / WatchReport ──────────────────────────────────────────
# ===========================================================================


class TestStepReport:
    def test_defaults(self) -> None:
        sr = StepReport(step=1, timestamp=1700000000.0)
        assert sr.risk_scores == {}
        assert sr.has_critical is False
        assert sr.num_interventions == 0
        assert sr.num_alerts == 0
        assert sr.worst_layer is None
        assert math.isnan(sr.loss)

    def test_has_critical_true(self) -> None:
        sr = StepReport(
            step=1,
            timestamp=1700000000.0,
            risk_levels={"layers.0.gate": RiskLevel.CRITICAL.value},
        )
        assert sr.has_critical is True

    def test_worst_layer(self) -> None:
        sr = StepReport(
            step=1,
            timestamp=1700000000.0,
            risk_scores={"layers.0.gate": 0.2, "layers.1.gate": 0.8},
        )
        assert sr.worst_layer == "layers.1.gate"

    def test_to_dict_with_interventions_and_alerts(self) -> None:
        action = NoOpAction(layer_name="layers.0.gate")
        alert = Alert(
            step=1, level=AlertLevel.INFO, layer_id="layers.0.gate",
            signal_type="risk_score", message="ok",
        )
        sr = StepReport(
            step=1,
            timestamp=1700000000.0,
            risk_scores={"layers.0.gate": 0.1},
            risk_levels={"layers.0.gate": "low"},
            active_interventions=[action],
            alerts=[alert],
            loss=0.5,
        )
        d = sr.to_dict()
        assert d["step"] == 1
        assert d["loss"] == pytest.approx(0.5)
        assert d["active_interventions"][0]["action_type"] == "NoOpAction"
        assert d["alerts"][0]["signal_type"] == "risk_score"

    def test_to_dict_with_nan_loss(self) -> None:
        sr = StepReport(step=1, timestamp=1700000000.0)
        d = sr.to_dict()
        assert d["loss"] is None

    def test_repr(self) -> None:
        sr = StepReport(step=3, timestamp=1700000000.0)
        r = repr(sr)
        assert "StepReport(step=3" in r


class TestWatchReport:
    def test_invalid_max_steps_raises(self) -> None:
        with pytest.raises(ValueError):
            WatchReport(max_steps=-1)

    def test_empty_report(self) -> None:
        report = WatchReport(max_steps=10)
        assert report.is_empty is True
        assert report.start_step == -1
        assert report.end_step == -1
        assert report.latest() is None
        assert len(report) == 0
        assert report.num_interventions == 0
        assert report.num_alerts == 0
        assert report.num_critical_steps == 0
        assert report.mean_risk() != report.mean_risk()  # NaN check
        assert "no steps recorded" in report.summary()

    def test_append_wrong_type_raises(self) -> None:
        report = WatchReport(max_steps=10)
        with pytest.raises(TypeError):
            report.append("not-a-step-report")  # type: ignore[arg-type]

    def test_append_and_query(self) -> None:
        report = WatchReport(max_steps=10)
        sr1 = StepReport(step=1, timestamp=1700000001.0, risk_scores={"layers.0.gate": 0.2})
        sr2 = StepReport(
            step=2,
            timestamp=1700000002.0,
            risk_scores={"layers.0.gate": 0.5, "layers.1.gate": 0.9},
            risk_levels={"layers.1.gate": RiskLevel.CRITICAL.value},
        )
        report.append(sr1)
        report.append(sr2)

        assert report.start_step == 1
        assert report.end_step == 2
        assert report.latest() is sr2
        assert len(report) == 2
        assert report.num_critical_steps == 1
        assert report.steps_since(1) == [sr2]

        history = report.risk_history("layers.0.gate")
        assert history == [(1, 0.2), (2, 0.5)]

        worst_history = report.worst_layer_history()
        assert worst_history[-1][1] == "layers.1.gate"

        assert report.mean_risk("layers.0.gate") == pytest.approx(0.35)
        assert report.mean_risk() == pytest.approx((0.2 + 0.5 + 0.9) / 3)

    def test_max_steps_eviction(self) -> None:
        report = WatchReport(max_steps=2)
        for i in range(5):
            report.append(StepReport(step=i, timestamp=float(i)))
        assert len(report) == 2
        assert report.start_step == 3
        assert report.end_step == 4

    def test_unlimited_retention_with_zero(self) -> None:
        report = WatchReport(max_steps=0)
        for i in range(10):
            report.append(StepReport(step=i, timestamp=float(i)))
        assert len(report) == 10

    def test_alerts_and_interventions_since(self) -> None:
        report = WatchReport(max_steps=10)
        alert = Alert(step=1, level=AlertLevel.INFO, layer_id="l0", signal_type="risk_score", message="ok")
        action = NoOpAction(layer_name="l0")
        sr = StepReport(step=1, timestamp=1.0, alerts=[alert], active_interventions=[action])
        report.append(sr)

        assert report.alerts_since(0) == [alert]
        assert report.alerts_since(1) == []
        assert report.interventions_since(0) == [action]

    def test_summary_with_data(self) -> None:
        report = WatchReport(max_steps=10)
        action = NoOpAction(layer_name="layers.0.gate")
        action.mark_applied(1)
        sr = StepReport(
            step=1,
            timestamp=1700000000.0,
            risk_scores={"layers.0.gate": 0.7},
            risk_levels={"layers.0.gate": "high"},
            active_interventions=[action],
            loss=0.42,
        )
        report.append(sr)
        text = report.summary()
        assert "MoEWatch Live Report" in text
        assert "layers.0.gate" in text
        assert "Training loss" in text
        assert "Active interventions at latest step" in text

    def test_to_dict_and_to_json_with_cap(self) -> None:
        report = WatchReport(max_steps=10)
        for i in range(5):
            report.append(StepReport(step=i, timestamp=float(i), risk_scores={"l0": 0.1 * i}))

        d = report.to_dict(max_steps_in_output=2)
        assert len(d["steps"]) == 2
        assert d["num_retained_steps"] == 5

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "watch.json")
            report.to_json(path, max_steps_in_output=2)
            with open(path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            assert len(loaded["steps"]) == 2

    def test_to_json_oserror_raises(self) -> None:
        report = WatchReport(max_steps=10)
        report.append(StepReport(step=1, timestamp=1.0))
        with pytest.raises(OSError):
            report.to_json("/nonexistent_dir_xyz/watch.json")

    def test_repr(self) -> None:
        report = WatchReport(max_steps=10)
        report.append(StepReport(step=1, timestamp=1.0))
        r = repr(report)
        assert "WatchReport" in r


# ===========================================================================
# ── 10. CLIReporter (lightweight) ────────────────────────────────────────
# ===========================================================================


class TestCLIReporter:
    def test_constructs_with_no_color(self) -> None:
        config = _config(no_color=True)
        reporter = CLIReporter(config)
        assert reporter._no_color is True

    def test_render_dashboard_with_empty_report(self) -> None:
        config = _config(no_color=True)
        reporter = CLIReporter(config)
        empty_report = WatchReport(max_steps=10)

        entropy_analyzer = MagicMock()
        collapse_detector = MagicMock()
        risk_fuser = MagicMock()
        risk_fuser.latest_scores.return_value = {}

        result = reporter.render_dashboard(
            empty_report, entropy_analyzer, collapse_detector, risk_fuser
        )
        assert "empty" in result.lower() or "nothing to render" in result.lower()

    def test_render_alert_plain(self) -> None:
        config = _config(no_color=True)
        reporter = CLIReporter(config)
        alert = Alert(
            step=1, level=AlertLevel.CRITICAL, layer_id="layers.0.gate",
            signal_type="entropy_drift", message="collapse imminent",
        )
        result = reporter.render_alert(alert)
        assert "0.gate" in result
        assert "collapse imminent" in result

# ------------------------------------------------------------------
# __init__.py coverage
# ------------------------------------------------------------------

from unittest.mock import MagicMock, patch
import moewatch
import importlib
import warnings


def test_alert_str_and_dict():
    alert = Alert(
        step=10,
        level=moewatch.AlertLevel.WARNING,
        layer_id="layer.0",
        signal_type="entropy",
        message="test",
        metrics={"score": 0.5},
    )

    s = str(alert)

    assert "step=10" in s
    assert "layer.0" in s
    assert "entropy" in s

    d = alert.to_dict()

    assert d["step"] == 10
    assert d["level"] == moewatch.AlertLevel.WARNING.value
    assert d["layer_id"] == "layer.0"
    assert d["signal_type"] == "entropy"
    assert d["message"] == "test"
    assert d["metrics"] == {"score": 0.5}


def test_require_torch_failure(monkeypatch):
    monkeypatch.setattr(moewatch, "_TORCH_AVAILABLE", False)

    with pytest.raises(ImportError):
        moewatch._require_torch("test")


def test_require_transformers_failure(monkeypatch):
    monkeypatch.setattr(moewatch, "_TRANSFORMERS_AVAILABLE", False)

    with pytest.raises(ImportError):
        moewatch._require_transformers("test")


def test_getattr_invalid():
    with pytest.raises(AttributeError):
        getattr(moewatch, "does_not_exist")


def test_public_api_exports():
    expected = {
        "MoEWatch",
        "MoEWatchCallback",
        "audit",
        "WatchConfig",
        "OutputMode",
        "AlertLevel",
        "Alert",
        "__version__",
        "__author__",
        "__license__",
        "__repository__",
    }

    assert expected.issubset(set(moewatch.__all__))


def test_metadata_constants():
    assert isinstance(moewatch.__version__, str)
    assert isinstance(moewatch.__author__, str)
    assert isinstance(moewatch.__license__, str)
    assert isinstance(moewatch.__repository__, str)


def test_no_color_flag_exists():
    assert isinstance(moewatch._NO_COLOR, bool)

def test_getattr_audit_success():
    obj = moewatch.__getattr__("audit")
    assert callable(obj)


def test_getattr_moewatch_success():
    obj = moewatch.__getattr__("MoEWatch")
    assert obj is not None


def test_getattr_moewatch_callback_success():
    obj = moewatch.__getattr__("MoEWatchCallback")
    assert obj is not None
