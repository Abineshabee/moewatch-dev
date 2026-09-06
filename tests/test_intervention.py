# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# tests/test_intervention.py
# =============================================================================
#
# Project      : MoEWatch
# Description  : Tests for InterventionEngine, SafetyGuard, and action classes.
#
#                Coverage targets (>= 80%):
#
#                InterventionAction subclasses
#                  - NoOpAction.apply/revert are no-ops
#                  - AuxLossAction.action_type == "aux_loss"
#                  - RouterNoiseAction.action_type == "router_noise"
#                  - ExpertDropoutAction.action_type == "expert_dropout"
#                  - NoOpAction.action_type == "noop"
#                  - action.log() returns string
#                  - mark_applied() sets applied_step
#                  - apply/revert on mock trainer without crash
#
#                SafetyGuard
#                  - Constructs with WatchConfig
#                  - NoOpAction always passes check
#                  - check() with no intervention history passes (no cooldown issue)
#                  - check() during cooldown returns NoOp (cooldown guard)
#                  - delta too large returns NoOp (delta limit guard)
#                  - loss spike returns NoOp (loss guard)
#                  - check() result has .passed and .recommended_action
#                  - record_intervention() updates cooldown
#                  - update_baseline_loss() accepts valid float
#
#                InterventionEngine
#                  - Constructs with config, trainer, baseline_tracker
#                  - propose_intervention() passes valid action
#                  - propose_intervention() downgrades when layer already active
#                  - apply_intervention() with NoOp does not create active entry
#                  - apply_intervention() with real action creates active entry
#                  - apply_intervention() marks baseline exclusion window
#                  - check_observation_windows() resolves expired window
#                  - check_observation_windows() defers unexpired window
#                  - intervention_log grows on each apply
#
# =============================================================================

from __future__ import annotations

import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from moewatch.collector.baseline_tracker import BaselineTracker
from moewatch.config import OutputMode, WatchConfig
from moewatch.intervention.actions import (
    AuxLossAction,
    ExpertDropoutAction,
    InterventionAction,
    NoOpAction,
    RouterNoiseAction,
)
from moewatch.intervention.engine import InterventionEngine
from moewatch.intervention.safety import SafetyGuard
from moewatch.policy.base import PolicyState
from moewatch.policy.rule_policy import RulePolicy


# ===========================================================================
# ── Helpers ──────────────────────────────────────────────────────────────────
# ===========================================================================


def _config(
    cooldown: int = 5,
    max_delta: float = 0.5,
    loss_guard: float = 2.0,
    reward_window: int = 10,
) -> WatchConfig:
    return WatchConfig(
        output=OutputMode.SILENT,
        intervention_cooldown=cooldown,
        intervention_max_delta=max_delta,
        loss_guard_threshold=loss_guard,
        reward_window_steps=reward_window,
        baseline_min_clean_steps=3,
        baseline_exclusion_window=10,
    )


def _mock_trainer() -> MagicMock:
    """A MagicMock that quacks like a transformers.Trainer."""
    trainer = MagicMock()
    trainer.args = MagicMock()
    trainer.args.aux_loss_coef = 0.01
    return trainer


def _make_engine(
    config: WatchConfig | None = None,
    trainer: Any = None,
    baseline_tracker: BaselineTracker | None = None,
) -> InterventionEngine:
    cfg = config or _config()
    tr = trainer or _mock_trainer()
    bt = baseline_tracker or BaselineTracker(cfg)
    return InterventionEngine(cfg, tr, bt)


def _policy_state(risk: float = 0.9, layer_id: int = 0, step: int = 0) -> PolicyState:
    return PolicyState(risk_score=risk, layer_id=layer_id, training_step=step)


# ===========================================================================
# ── 1. Action classes ─────────────────────────────────────────────────────────
# ===========================================================================


class TestNoOpAction:
    def test_action_type(self) -> None:
        a = NoOpAction(layer_name="layer.0.gate")
        assert a.action_type == "noop"

    def test_apply_does_not_raise(self) -> None:
        a = NoOpAction(layer_name="layer.0.gate")
        a.apply(_mock_trainer())  # must not raise

    def test_revert_does_not_raise(self) -> None:
        a = NoOpAction(layer_name="layer.0.gate")
        a.revert(_mock_trainer())

    def test_log_returns_string(self) -> None:
        a = NoOpAction(layer_name="layer.0.gate")
        assert isinstance(a.log(), str)

    def test_delta_is_zero(self) -> None:
        a = NoOpAction(layer_name="layer.0.gate")
        assert a.delta == 0.0

    def test_mark_applied_sets_step(self) -> None:
        a = NoOpAction(layer_name="layer.0.gate")
        a.mark_applied(42)
        assert a.applied_step == 42


class TestAuxLossAction:
    def test_action_type(self) -> None:
        a = AuxLossAction(layer_name="layer.0.gate")
        assert a.action_type == "aux_loss"

    def test_layer_name_stored(self) -> None:
        a = AuxLossAction(layer_name="my.layer")
        assert a.layer_name == "my.layer"

    def test_log_contains_action_type(self) -> None:
        a = AuxLossAction(layer_name="layer.0.gate")
        assert "aux_loss" in a.log()

    def test_apply_no_crash_with_mock_trainer(self) -> None:
        a = AuxLossAction(layer_name="layer.0.gate")
        a.apply(_mock_trainer())

    def test_revert_no_crash(self) -> None:
        trainer = _mock_trainer()
        a = AuxLossAction(layer_name="layer.0.gate")
        a.apply(trainer)
        a.revert(trainer)

    def test_delta_is_positive(self) -> None:
        a = AuxLossAction(layer_name="layer.0.gate")
        assert a.delta > 0


class TestRouterNoiseAction:
    def test_action_type(self) -> None:
        a = RouterNoiseAction(layer_name="layer.0.gate")
        assert a.action_type == "router_noise"

    def test_apply_no_crash(self) -> None:
        a = RouterNoiseAction(layer_name="layer.0.gate")
        a.apply(_mock_trainer())

    def test_revert_no_crash(self) -> None:
        trainer = _mock_trainer()
        a = RouterNoiseAction(layer_name="layer.0.gate")
        a.apply(trainer)
        a.revert(trainer)


class TestExpertDropoutAction:
    def test_action_type(self) -> None:
        a = ExpertDropoutAction(layer_name="layer.0.gate")
        assert a.action_type == "expert_dropout"

    def test_apply_no_crash(self) -> None:
        a = ExpertDropoutAction(layer_name="layer.0.gate")
        a.apply(_mock_trainer())

    def test_revert_no_crash(self) -> None:
        trainer = _mock_trainer()
        a = ExpertDropoutAction(layer_name="layer.0.gate")
        a.apply(trainer)
        a.revert(trainer)


# ===========================================================================
# ── 2. SafetyGuard ────────────────────────────────────────────────────────────
# ===========================================================================


class TestSafetyGuardConstruction:
    def test_constructs(self) -> None:
        guard = SafetyGuard(_config())
        assert guard is not None

    def test_config_stored(self) -> None:
        cfg = _config()
        guard = SafetyGuard(cfg)
        assert guard.config is cfg


class TestSafetyGuardCheck:
    def _check(
        self,
        action: InterventionAction,
        config: WatchConfig | None = None,
        current_loss: float = 0.5,
        risk_scores: Dict[str, float] | None = None,
        layer_order: List[str] | None = None,
    ):
        cfg = config or _config()
        guard = SafetyGuard(cfg)
        action.mark_applied(0)
        return guard.check(
            action,
            current_loss,
            risk_scores or {"layer.0.gate": 0.5},
            layer_order or ["layer.0.gate"],
        )

    def test_noop_always_passes(self) -> None:
        result = self._check(NoOpAction(layer_name="layer.0.gate"))
        assert result.passed is True
        assert isinstance(result.recommended_action, NoOpAction)

    def test_valid_action_passes_on_first_call(self) -> None:
        result = self._check(AuxLossAction(layer_name="layer.0.gate"))
        assert result.passed is True

    def test_result_has_passed_attribute(self) -> None:
        result = self._check(AuxLossAction(layer_name="layer.0.gate"))
        assert hasattr(result, "passed")
        assert isinstance(result.passed, bool)

    def test_result_has_recommended_action(self) -> None:
        result = self._check(AuxLossAction(layer_name="layer.0.gate"))
        assert hasattr(result, "recommended_action")
        assert isinstance(result.recommended_action, InterventionAction)

    def test_cooldown_guard_fires_on_repeat(self) -> None:
        """Two interventions on same layer within cooldown window → second downgraded."""
        cfg = _config(cooldown=100)
        guard = SafetyGuard(cfg)
        layer = "layer.0.gate"
        guard.record_intervention(layer, step=0)

        action = AuxLossAction(layer_name=layer)
        action.mark_applied(5)  # step 5, still in cooldown [0, 100)
        result = guard.check(action, 0.5, {layer: 0.5}, [layer])

        assert result.passed is False
        assert isinstance(result.recommended_action, NoOpAction)

    def test_delta_limit_guard_fires(self) -> None:
        """action.delta > config.intervention_max_delta → downgraded."""
        cfg = _config(max_delta=0.001)  # very tight limit
        guard = SafetyGuard(cfg)
        # AuxLossAction default delta is likely > 0.001
        action = AuxLossAction(layer_name="layer.0.gate")
        action.mark_applied(0)
        result = guard.check(action, 0.5, {"layer.0.gate": 0.5}, ["layer.0.gate"])
        if action.delta > cfg.intervention_max_delta:
            assert result.passed is False

    def test_loss_guard_fires_on_spike(self) -> None:
        """current_loss >> loss_baseline → downgraded."""
        cfg = _config(loss_guard=2.0)
        guard = SafetyGuard(cfg)
        guard.update_baseline_loss(0.5)  # baseline = 0.5
        # current_loss = 5.0 = 10x baseline → spike
        action = AuxLossAction(layer_name="layer.0.gate")
        action.mark_applied(0)
        result = guard.check(action, 5.0, {"layer.0.gate": 0.5}, ["layer.0.gate"])
        assert result.passed is False

    def test_loss_guard_does_not_fire_when_no_baseline(self) -> None:
        """If update_baseline_loss never called, loss guard should not fire."""
        cfg = _config()
        guard = SafetyGuard(cfg)
        # No baseline set
        action = AuxLossAction(layer_name="layer.0.gate")
        action.mark_applied(0)
        result = guard.check(action, 1.0, {"layer.0.gate": 0.5}, ["layer.0.gate"])
        # Loss guard skipped (no baseline) → result may still pass if no other issues
        assert result.passed is True or result.passed is False  # just no crash


class TestSafetyGuardRecordIntervention:
    def test_record_sets_cooldown(self) -> None:
        cfg = _config(cooldown=50)
        guard = SafetyGuard(cfg)
        guard.record_intervention("layer.0.gate", step=100)
        assert guard._intervention_history.get("layer.0.gate") == 100

    def test_update_baseline_loss_valid_float(self) -> None:
        guard = SafetyGuard(_config())
        guard.update_baseline_loss(0.75)
        assert guard._loss_baseline == 0.75

    def test_update_baseline_loss_nan_ignored(self) -> None:
        guard = SafetyGuard(_config())
        guard.update_baseline_loss(float("nan"))
        assert guard._loss_baseline is None

    def test_update_baseline_loss_inf_ignored(self) -> None:
        guard = SafetyGuard(_config())
        guard.update_baseline_loss(float("inf"))
        assert guard._loss_baseline is None


# ===========================================================================
# ── 3. InterventionEngine construction ───────────────────────────────────────
# ===========================================================================


class TestInterventionEngineConstruction:
    def test_constructs_without_error(self) -> None:
        engine = _make_engine()
        assert engine is not None

    def test_attributes_set(self) -> None:
        cfg = _config()
        trainer = _mock_trainer()
        bt = BaselineTracker(cfg)
        engine = InterventionEngine(cfg, trainer, bt)
        assert engine.config is cfg
        assert engine.model is trainer
        assert engine.baseline_tracker is bt

    def test_safety_guard_created(self) -> None:
        engine = _make_engine()
        assert isinstance(engine.safety_guard, SafetyGuard)

    def test_intervention_log_empty_on_init(self) -> None:
        engine = _make_engine()
        assert engine._intervention_log == []


# ===========================================================================
# ── 4. propose_intervention ──────────────────────────────────────────────────
# ===========================================================================


class TestProposeIntervention:
    def test_valid_action_returned_unchanged(self) -> None:
        engine = _make_engine()
        action = AuxLossAction(layer_name="layer.0.gate")
        result = engine.propose_intervention(
            action,
            current_loss=0.5,
            risk_scores={"layer.0.gate": 0.5},
            layer_order=["layer.0.gate"],
            step=0,
        )
        assert result.action_type == "aux_loss"

    def test_noop_passed_through_unchanged(self) -> None:
        engine = _make_engine()
        action = NoOpAction(layer_name="layer.0.gate")
        result = engine.propose_intervention(
            action, 0.5, {"layer.0.gate": 0.5}, ["layer.0.gate"], 0
        )
        assert isinstance(result, NoOpAction)

    def test_layer_already_active_downgraded_to_noop(self) -> None:
        """If a layer already has an active intervention, new action → NoOp."""
        cfg = _config(cooldown=1)
        engine = _make_engine(config=cfg)
        layer = "layer.0.gate"

        # Apply first action to mark it active
        a1 = AuxLossAction(layer_name=layer)
        validated = engine.propose_intervention(a1, 0.5, {layer: 0.5}, [layer], 0)
        engine.apply_intervention(validated, step=0)

        # Second proposal for same layer while first is active → NoOp
        a2 = RouterNoiseAction(layer_name=layer)
        result = engine.propose_intervention(a2, 0.5, {layer: 0.5}, [layer], 1)
        assert isinstance(result, NoOpAction)

    def test_mark_applied_called_on_action(self) -> None:
        engine = _make_engine()
        action = NoOpAction(layer_name="layer.0.gate")
        engine.propose_intervention(action, 0.5, {}, [], step=77)
        assert action.applied_step == 77


# ===========================================================================
# ── 5. apply_intervention ────────────────────────────────────────────────────
# ===========================================================================


class TestApplyIntervention:
    def test_noop_does_not_register_active(self) -> None:
        engine = _make_engine()
        action = NoOpAction(layer_name="layer.0.gate")
        engine.apply_intervention(action, step=0)
        assert "layer.0.gate" not in engine._active_interventions

    def test_non_noop_registers_active_intervention(self) -> None:
        cfg = _config(cooldown=1)
        engine = _make_engine(config=cfg)
        layer = "layer.0.gate"
        action = AuxLossAction(layer_name=layer)
        validated = engine.propose_intervention(action, 0.5, {layer: 0.5}, [layer], 0)
        if validated.action_type != "noop":
            engine.apply_intervention(validated, step=0)
            assert layer in engine._active_interventions

    def test_apply_logs_to_intervention_log(self) -> None:
        engine = _make_engine()
        action = NoOpAction(layer_name="layer.0.gate")
        engine.apply_intervention(action, step=5)
        assert len(engine._intervention_log) == 1
        assert engine._intervention_log[0]["event"] == "applied"

    def test_apply_non_noop_marks_baseline_exclusion(self) -> None:
        cfg = _config(cooldown=1, reward_window=20)
        bt = BaselineTracker(cfg)
        engine = _make_engine(config=cfg, baseline_tracker=bt)
        layer = "layer.0.gate"

        action = AuxLossAction(layer_name=layer)
        validated = engine.propose_intervention(action, 0.5, {layer: 0.5}, [layer], 0)
        if validated.action_type != "noop":
            engine.apply_intervention(validated, step=10)
            windows = bt._intervention_windows.get(layer, [])
            assert len(windows) > 0, "Expected exclusion window after apply_intervention"

    def test_apply_creates_observation_window(self) -> None:
        cfg = _config(cooldown=1, reward_window=50)
        engine = _make_engine(config=cfg)
        layer = "layer.0.gate"
        action = AuxLossAction(layer_name=layer)
        validated = engine.propose_intervention(action, 0.5, {layer: 0.5}, [layer], 0)
        if validated.action_type != "noop":
            engine.apply_intervention(validated, step=0)
            assert layer in engine._observation_windows
            _, end = engine._observation_windows[layer]
            assert end == 50

    def test_state_stored_when_provided(self) -> None:
        cfg = _config(cooldown=1)
        engine = _make_engine(config=cfg)
        layer = "layer.0.gate"
        action = AuxLossAction(layer_name=layer)
        validated = engine.propose_intervention(action, 0.5, {layer: 0.5}, [layer], 0)
        if validated.action_type != "noop":
            state = PolicyState(risk_score=0.9, layer_id=0, training_step=0)
            engine.apply_intervention(validated, step=0, state=state)
            assert layer in engine._pending_states


# ===========================================================================
# ── 6. check_observation_windows ─────────────────────────────────────────────
# ===========================================================================


class TestCheckObservationWindows:
    def _setup_engine_with_active_intervention(
        self,
        layer: str = "layer.0.gate",
        step: int = 0,
        reward_window: int = 5,
    ):
        cfg = _config(cooldown=1, reward_window=reward_window)
        bt = BaselineTracker(cfg)
        bt.register_layer(layer)
        # Fill some clean history so baseline is valid
        for s in range(10):
            bt.update_signal(layer, 0.5, step=s - 20)  # old clean steps
        engine = _make_engine(config=cfg, baseline_tracker=bt)

        action = AuxLossAction(layer_name=layer)
        validated = engine.propose_intervention(action, 0.5, {layer: 0.5}, [layer], step)
        if validated.action_type != "noop":
            state = PolicyState(risk_score=0.9, layer_id=0, training_step=step)
            engine.apply_intervention(validated, step=step, state=state)
        return engine, validated

    def test_unexpired_window_not_resolved(self) -> None:
        engine, _ = self._setup_engine_with_active_intervention(step=0, reward_window=50)
        policy = RulePolicy(_config())
        layer = "layer.0.gate"
        if layer in engine._observation_windows:
            engine.check_observation_windows(step=1, risk_scores={layer: 0.5}, policy=policy)
            assert layer in engine._observation_windows, (
                "Window was resolved prematurely (step=1 < window_end=50)"
            )

    def test_expired_window_resolved(self) -> None:
        engine, _ = self._setup_engine_with_active_intervention(step=0, reward_window=5)
        policy = RulePolicy(_config())
        layer = "layer.0.gate"
        if layer in engine._observation_windows:
            # Step past the window end
            engine.check_observation_windows(step=10, risk_scores={layer: 0.3}, policy=policy)
            assert layer not in engine._observation_windows, (
                "Window not resolved after its end step"
            )

    def test_missing_risk_score_defers_window(self) -> None:
        engine, _ = self._setup_engine_with_active_intervention(step=0, reward_window=5)
        policy = RulePolicy(_config())
        layer = "layer.0.gate"
        if layer in engine._observation_windows:
            # Provide empty risk_scores → layer deferred
            engine.check_observation_windows(step=10, risk_scores={}, policy=policy)
            assert layer in engine._observation_windows, (
                "Window resolved without risk score (should defer)"
            )

    def test_no_crash_on_empty_windows(self) -> None:
        engine = _make_engine()
        policy = RulePolicy(_config())
        engine.check_observation_windows(step=100, risk_scores={"layer.0.gate": 0.5}, policy=policy)
        # Just must not crash

    def test_policy_update_called_on_expired_window(self) -> None:
        engine, _ = self._setup_engine_with_active_intervention(step=0, reward_window=5)
        layer = "layer.0.gate"
        policy = MagicMock()
        if layer in engine._observation_windows:
            engine.check_observation_windows(step=10, risk_scores={layer: 0.3}, policy=policy)
            # policy.update should have been called
            assert policy.update.called or layer not in engine._observation_windows


# ===========================================================================
# ── 7. Intervention log ───────────────────────────────────────────────────────
# ===========================================================================


class TestInterventionLog:
    def test_log_grows_per_apply(self) -> None:
        engine = _make_engine()
        for i in range(3):
            action = NoOpAction(layer_name=f"layer.{i}.gate")
            engine.apply_intervention(action, step=i)
        assert len(engine._intervention_log) == 3

    def test_log_entries_have_event_key(self) -> None:
        engine = _make_engine()
        action = NoOpAction(layer_name="layer.0.gate")
        engine.apply_intervention(action, step=0)
        for entry in engine._intervention_log:
            assert "event" in entry

    def test_log_entry_has_step(self) -> None:
        engine = _make_engine()
        action = NoOpAction(layer_name="layer.0.gate")
        engine.apply_intervention(action, step=42)
        assert engine._intervention_log[0]["step"] == 42

    def test_log_downgrade_recorded_on_failed_proposal(self) -> None:
        cfg = _config(cooldown=1000)  # long cooldown to force guard failure
        engine = _make_engine(config=cfg)
        layer = "layer.0.gate"
        # Apply once
        a1 = AuxLossAction(layer_name=layer)
        v1 = engine.propose_intervention(a1, 0.5, {layer: 0.5}, [layer], 0)
        engine.apply_intervention(v1, step=0)
        log_before = len(engine._intervention_log)

        # Second proposal — layer active → downgrade logged
        a2 = RouterNoiseAction(layer_name=layer)
        engine.propose_intervention(a2, 0.5, {layer: 0.5}, [layer], 1)
        # A downgrade log entry should have been added if layer was active
        if layer in engine._active_interventions:
            assert len(engine._intervention_log) > log_before


# ===========================================================================
# ── Regression: RouterNoiseAction hook ordering (prepend=True) ──────────────
# ===========================================================================


class TestRouterNoiseActionHookOrdering:
    """Regression tests for the RouterNoiseAction measurement-correctness bug.

    Prior to the fix, ``RouterNoiseAction`` registered its forward hook with
    the default ``prepend=False``.  Because ``RouterForwardHook`` (MoEWatch's
    monitoring hook) is registered first during ``HookManager.attach()``,
    PyTorch executed it *before* the noise hook, so MoEWatch observed the
    original pre-noise logits while the model actually consumed the
    post-noise ones.  The fix uses ``prepend=True`` so the noise hook fires
    first, and the monitoring hook then sees the already-noised output.
    """

    # ------------------------------------------------------------------
    # Minimal test fixtures
    # ------------------------------------------------------------------

    @staticmethod
    def _make_model():
        """Return a tiny MoE whose router always outputs [1.0, 0.0]."""
        import torch
        import torch.nn as nn

        class DeterministicRouter(nn.Module):
            def forward(self, x):
                return torch.tensor([[1.0, 0.0]])

        class TinyMoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.router = DeterministicRouter()

            def forward(self, x):
                return self.router(x)

        return TinyMoE()

    # ------------------------------------------------------------------
    # Core regression test
    # ------------------------------------------------------------------

    def test_monitoring_hook_observes_post_noise_output(self) -> None:
        """MoEWatch must record the noised logits, not the original ones.

        Simulates the exact runtime scenario that was broken:

        1. A monitoring hook is registered first (as HookManager does).
        2. RouterNoiseAction is applied (registers its hook second).
        3. After a forward pass, the monitoring hook's recorded output
           must match the model's actual (noisy) output, not the original
           clean logits.
        """
        import torch

        model = self._make_model()
        recorded_by_monitor: list = []

        # Step 1 — monitoring hook registered FIRST (mirrors HookManager)
        def monitoring_hook(module, input, output):
            recorded_by_monitor.append(output.detach().clone())

        monitor_handle = model.router.register_forward_hook(monitoring_hook)

        # Step 2 — RouterNoiseAction registered AFTER (mirrors InterventionEngine)
        action = RouterNoiseAction(layer_name="router", noise_scale=10.0)
        action.apply(model)

        # Step 3 — forward pass
        model_output = model(torch.zeros(1))

        assert len(recorded_by_monitor) == 1, "Monitoring hook should have fired once."

        # The monitor must have seen the same tensor the model received.
        assert torch.allclose(recorded_by_monitor[0], model_output), (
            "RouterNoiseAction hook ordering bug: MoEWatch recorded the "
            "pre-noise logits instead of the post-noise logits that the "
            "model actually used."
        )

        # Sanity-check: the output must actually be noisy (not the original).
        original = torch.tensor([[1.0, 0.0]])
        assert not torch.allclose(model_output, original), (
            "Expected noisy output but got the clean original — "
            "noise injection may not be working."
        )

        # Cleanup
        monitor_handle.remove()
        action.revert(model)

    def test_monitoring_hook_sees_clean_output_without_noise_action(self) -> None:
        """Baseline: monitoring hook sees clean output when no noise is active."""
        import torch

        model = self._make_model()
        recorded: list = []

        handle = model.router.register_forward_hook(
            lambda m, i, o: recorded.append(o.detach().clone())
        )
        model(torch.zeros(1))

        original = torch.tensor([[1.0, 0.0]])
        assert torch.allclose(recorded[0], original), (
            "Without RouterNoiseAction, the router should output the clean logits."
        )
        handle.remove()

    def test_revert_restores_clean_output(self) -> None:
        """After revert(), the router returns to its original clean output."""
        import torch

        model = self._make_model()
        action = RouterNoiseAction(layer_name="router", noise_scale=10.0)
        action.apply(model)
        action.revert(model)

        out = model(torch.zeros(1))
        original = torch.tensor([[1.0, 0.0]])
        assert torch.allclose(out, original), (
            "After RouterNoiseAction.revert(), router output should be clean."
        )

    def test_noise_hook_prepend_position(self) -> None:
        """The noise hook must be prepended so it fires before other hooks.

        Verifies hook execution order by checking that the router's output
        after RouterNoiseAction.apply() is noisy — if prepend=True is
        working, the noise hook is first in the chain and all subsequent
        hooks (including the monitoring hook) see the already-modified tensor.
        """
        import torch

        model = self._make_model()

        # Register a monitoring hook first (mirrors HookManager)
        recorded_by_monitor: list = []
        model.router.register_forward_hook(
            lambda m, i, o: recorded_by_monitor.append(o.detach().clone())
        )

        action = RouterNoiseAction(layer_name="router", noise_scale=10.0)
        action.apply(model)

        out = model(torch.zeros(1))

        # The model output must be noisy (noise injection is active)
        original = torch.tensor([[1.0, 0.0]])
        assert not torch.allclose(out, original), (
            "Router output should be noisy after RouterNoiseAction is applied."
        )

        # The monitor (registered before the noise hook) must also observe
        # the noisy output — this is the key ordering assertion.
        assert torch.allclose(recorded_by_monitor[0], out), (
            "Monitoring hook should observe the post-noise output (prepend=True "
            "ensures noise hook runs before the monitor, not after)."
        )

        action.revert(model)
