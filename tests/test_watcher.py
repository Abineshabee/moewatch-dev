# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# tests/test_watcher.py
# =============================================================================
#
# Project      : MoEWatch
# Description  : End-to-end integration tests for MoEWatch (live monitor).
#
#                All tests run on CPU with FakeMoEModel; no real HuggingFace
#                Trainer or GPU is required.
#
#                Coverage targets (>= 80%):
#
#                  - MoEWatch constructs on FakeMoEModel
#                  - MoEWatch raises ValueError on non-MoE model
#                  - start() attaches hooks
#                  - stop() detaches hooks
#                  - Double start() raises RuntimeError
#                  - stop() without start() does not raise
#                  - Context manager (__enter__ / __exit__) attaches + detaches
#                  - step() returns WatchReport with risk_scores
#                  - step() increments internal step counter
#                  - Multiple step() calls accumulate state
#                  - get_alerts() returns list[Alert]
#                  - get_alerts(since_step=N) filters correctly
#                  - attach() returns MoEWatchCallback
#                  - MoEWatchCallback.on_step_end calls step()
#                  - SILENT output mode: no stdout output
#                  - Config defaults used when config=None
#                  - WatchReport has expected attributes
#
# =============================================================================

from __future__ import annotations

import io
import sys
import time
from typing import List
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from moewatch._watcher import MoEWatch, MoEWatchCallback
from moewatch.config import AlertLevel, OutputMode, WatchConfig

from conftest import FakeMoEModel, FakeNonMoEModel


# ===========================================================================
# ── Helpers ──────────────────────────────────────────────────────────────────
# ===========================================================================


def _silent_config(**kwargs) -> WatchConfig:
    return WatchConfig(output=OutputMode.SILENT, **kwargs)


def _run_steps(watch: MoEWatch, model: FakeMoEModel, n: int = 5, batch: int = 4) -> None:
    """Run ``n`` forward passes and call watch.step() for each."""
    for step in range(n):
        watch.hook_manager.set_global_step(step)
        x = torch.randn(batch, model.hidden)
        with torch.no_grad():
            model(x)
        watch.step(step)


def _mock_trainer() -> MagicMock:
    trainer = MagicMock()
    trainer.add_callback = MagicMock()
    trainer.args = MagicMock()
    trainer.args.aux_loss_coef = 0.01
    return trainer


def _mock_trainer_state(step: int = 0) -> MagicMock:
    state = MagicMock()
    state.global_step = step
    return state


# ===========================================================================
# ── 1. Construction ──────────────────────────────────────────────────────────
# ===========================================================================


class TestMoEWatchConstruction:
    def test_constructs_on_valid_moe_model(self, moe_model: FakeMoEModel) -> None:
        watch = MoEWatch(moe_model, _silent_config())
        assert watch is not None

    def test_model_stored(self, moe_model: FakeMoEModel) -> None:
        watch = MoEWatch(moe_model, _silent_config())
        assert watch.model is moe_model

    def test_config_stored(self, moe_model: FakeMoEModel) -> None:
        cfg = _silent_config()
        watch = MoEWatch(moe_model, cfg)
        assert watch.config is cfg

    def test_default_config_when_none(self, moe_model: FakeMoEModel) -> None:
        watch = MoEWatch(moe_model, config=None)
        assert watch.config is not None

    def test_raises_on_non_moe_model(self, non_moe_model: FakeNonMoEModel) -> None:
        with pytest.raises(ValueError):
            MoEWatch(non_moe_model, _silent_config())

    def test_not_running_at_construction(self, moe_model: FakeMoEModel) -> None:
        watch = MoEWatch(moe_model, _silent_config())
        assert watch._running is False

    def test_alerts_empty_at_construction(self, moe_model: FakeMoEModel) -> None:
        watch = MoEWatch(moe_model, _silent_config())
        assert watch.alerts == []

    def test_global_step_zero_at_construction(self, moe_model: FakeMoEModel) -> None:
        watch = MoEWatch(moe_model, _silent_config())
        assert watch._global_step == 0


# ===========================================================================
# ── 2. start() / stop() lifecycle ────────────────────────────────────────────
# ===========================================================================


class TestStartStop:
    def test_start_sets_running(self, moe_model: FakeMoEModel) -> None:
        watch = MoEWatch(moe_model, _silent_config())
        watch.start()
        assert watch._running is True
        watch.stop()

    def test_start_attaches_hooks(self, moe_model: FakeMoEModel) -> None:
        watch = MoEWatch(moe_model, _silent_config())
        watch.start()
        assert watch.hook_manager.is_attached() is True
        watch.stop()

    def test_start_initialises_sub_systems(self, moe_model: FakeMoEModel) -> None:
        watch = MoEWatch(moe_model, _silent_config())
        watch.start()
        assert watch.stat_collector is not None
        assert watch.entropy_analyzer is not None
        assert watch.collapse_detector is not None
        watch.stop()

    def test_stop_detaches_hooks(self, moe_model: FakeMoEModel) -> None:
        watch = MoEWatch(moe_model, _silent_config())
        watch.start()
        watch.stop()
        assert not watch.hook_manager.is_attached()

    def test_stop_sets_not_running(self, moe_model: FakeMoEModel) -> None:
        watch = MoEWatch(moe_model, _silent_config())
        watch.start()
        watch.stop()
        assert watch._running is False

    def test_double_start_raises(self, moe_model: FakeMoEModel) -> None:
        watch = MoEWatch(moe_model, _silent_config())
        watch.start()
        try:
            with pytest.raises(RuntimeError):
                watch.start()
        finally:
            watch.stop()

    def test_stop_without_start_no_crash(self, moe_model: FakeMoEModel) -> None:
        """stop() before start() must not raise."""
        watch = MoEWatch(moe_model, _silent_config())
        watch.stop()  # graceful no-op

    def test_stop_idempotent(self, moe_model: FakeMoEModel) -> None:
        watch = MoEWatch(moe_model, _silent_config())
        watch.start()
        watch.stop()
        watch.stop()  # second stop → no crash


# ===========================================================================
# ── 3. Context manager ───────────────────────────────────────────────────────
# ===========================================================================


class TestContextManager:
    def test_enter_sets_running(self, moe_model: FakeMoEModel) -> None:
        with MoEWatch(moe_model, _silent_config()) as watch:
            assert watch._running is True

    def test_exit_stops_monitoring(self, moe_model: FakeMoEModel) -> None:
        with MoEWatch(moe_model, _silent_config()) as watch:
            pass
        assert watch._running is False

    def test_exit_detaches_hooks_on_exception(self, moe_model: FakeMoEModel) -> None:
        """Hooks must be detached even if an exception is raised inside `with`."""
        watch_ref = None
        try:
            with MoEWatch(moe_model, _silent_config()) as watch:
                watch_ref = watch
                raise RuntimeError("simulated training error")
        except RuntimeError:
            pass
        assert watch_ref is not None
        assert not watch_ref.hook_manager.is_attached()

    def test_enter_returns_self(self, moe_model: FakeMoEModel) -> None:
        watch = MoEWatch(moe_model, _silent_config())
        result = watch.__enter__()
        assert result is watch
        watch.__exit__(None, None, None)


# ===========================================================================
# ── 4. step() ────────────────────────────────────────────────────────────────
# ===========================================================================


class TestStep:
    def test_step_returns_watch_report(self, moe_model: FakeMoEModel) -> None:
        with MoEWatch(moe_model, _silent_config()) as watch:
            x = torch.randn(4, moe_model.hidden)
            with torch.no_grad():
                moe_model(x)
            report = watch.step(0)
            assert report is not None

    def test_watch_report_has_risk_scores(self, moe_model: FakeMoEModel) -> None:
        with MoEWatch(moe_model, _silent_config()) as watch:
            x = torch.randn(4, moe_model.hidden)
            with torch.no_grad():
                moe_model(x)
            report = watch.step(0)
            assert hasattr(report, "risk_scores")

    def test_watch_report_has_step_field(self, moe_model: FakeMoEModel) -> None:
        with MoEWatch(moe_model, _silent_config()) as watch:
            x = torch.randn(4, moe_model.hidden)
            with torch.no_grad():
                moe_model(x)
            report = watch.step(10)
            assert hasattr(report, "step")

    def test_step_increments_global_step(self, moe_model: FakeMoEModel) -> None:
        with MoEWatch(moe_model, _silent_config()) as watch:
            for i in range(5):
                x = torch.randn(2, moe_model.hidden)
                with torch.no_grad():
                    moe_model(x)
                watch.step(i)
            assert watch._global_step == 4

    def test_multiple_steps_no_crash(self, moe_model: FakeMoEModel) -> None:
        with MoEWatch(moe_model, _silent_config()) as watch:
            _run_steps(watch, moe_model, n=10)
        # Just must not crash

    def test_step_populates_detected_layers(self, moe_model: FakeMoEModel) -> None:
        with MoEWatch(moe_model, _silent_config()) as watch:
            x = torch.randn(4, moe_model.hidden)
            with torch.no_grad():
                moe_model(x)
            watch.step(0)
            assert len(watch._detected_layers) > 0

    def test_step_reports_layer_count_matches_model(
        self, moe_model: FakeMoEModel
    ) -> None:
        with MoEWatch(moe_model, _silent_config()) as watch:
            x = torch.randn(4, moe_model.hidden)
            with torch.no_grad():
                moe_model(x)
            report = watch.step(0)
            # 2 layers in FakeMoEModel → 2 risk scores
            assert len(report.risk_scores) == 2


# ===========================================================================
# ── 5. get_alerts() ──────────────────────────────────────────────────────────
# ===========================================================================


class TestGetAlerts:
    def test_get_alerts_returns_list(self, moe_model: FakeMoEModel) -> None:
        with MoEWatch(moe_model, _silent_config()) as watch:
            _run_steps(watch, moe_model, n=3)
        assert isinstance(watch.get_alerts(), list)

    def test_get_alerts_since_step_filters(self, moe_model: FakeMoEModel) -> None:
        """get_alerts(since_step=N) must exclude alerts from step < N."""
        cfg = _silent_config(
            entropy_warn=0.99,  # very tight → triggers alerts on any routing
            entropy_critical=0.98,
        )
        with MoEWatch(moe_model, cfg) as watch:
            _run_steps(watch, moe_model, n=20)

        all_alerts = watch.get_alerts()
        if all_alerts:
            # Pick a midpoint step
            mid = all_alerts[len(all_alerts) // 2].step
            filtered = watch.get_alerts(since_step=mid)
            assert all(a.step >= mid for a in filtered), (
                "get_alerts(since_step) returned alerts before the filter step"
            )

    def test_get_alerts_since_step_zero_returns_all(
        self, moe_model: FakeMoEModel
    ) -> None:
        with MoEWatch(moe_model, _silent_config()) as watch:
            _run_steps(watch, moe_model, n=5)
        all_alerts = watch.get_alerts()
        filtered = watch.get_alerts(since_step=0)
        assert len(filtered) == len(all_alerts)

    def test_alerts_have_required_fields(self, moe_model: FakeMoEModel) -> None:
        cfg = _silent_config(entropy_warn=0.99)
        with MoEWatch(moe_model, cfg) as watch:
            _run_steps(watch, moe_model, n=10)
        for alert in watch.get_alerts():
            assert hasattr(alert, "step")
            assert hasattr(alert, "level")
            assert hasattr(alert, "layer_id")
            assert hasattr(alert, "signal_type")
            assert hasattr(alert, "message")


# ===========================================================================
# ── 6. attach() and MoEWatchCallback ─────────────────────────────────────────
# ===========================================================================


class TestAttachAndCallback:
    def test_attach_returns_callback(self, moe_model: FakeMoEModel) -> None:
        with MoEWatch(moe_model, _silent_config()) as watch:
            trainer = _mock_trainer()
            cb = watch.attach(trainer)
            assert isinstance(cb, MoEWatchCallback)

    def test_attach_calls_trainer_add_callback(self, moe_model: FakeMoEModel) -> None:
        with MoEWatch(moe_model, _silent_config()) as watch:
            trainer = _mock_trainer()
            watch.attach(trainer)
            trainer.add_callback.assert_called_once()

    def test_callback_on_step_end_calls_step(self, moe_model: FakeMoEModel) -> None:
        with MoEWatch(moe_model, _silent_config()) as watch:
            cb = MoEWatchCallback(watch)
            state = _mock_trainer_state(step=5)

            # Run a forward pass first so there are events to process
            x = torch.randn(4, moe_model.hidden)
            with torch.no_grad():
                moe_model(x)

            cb.on_step_end(args=MagicMock(), state=state, control=MagicMock())
            assert watch._global_step == 5

    def test_callback_stores_moewatch_reference(self, moe_model: FakeMoEModel) -> None:
        with MoEWatch(moe_model, _silent_config()) as watch:
            cb = MoEWatchCallback(watch)
            assert cb.moewatch is watch

    def test_callback_on_step_end_no_crash(self, moe_model: FakeMoEModel) -> None:
        with MoEWatch(moe_model, _silent_config()) as watch:
            cb = MoEWatchCallback(watch)
            state = _mock_trainer_state(step=0)
            x = torch.randn(4, moe_model.hidden)
            with torch.no_grad():
                moe_model(x)
            cb.on_step_end(MagicMock(), state, MagicMock())


# ===========================================================================
# ── 7. Output mode: SILENT ───────────────────────────────────────────────────
# ===========================================================================


class TestSilentOutput:
    def test_silent_mode_no_stdout(self, moe_model: FakeMoEModel) -> None:
        """SILENT mode must not print anything to stdout."""
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            with MoEWatch(moe_model, _silent_config()) as watch:
                _run_steps(watch, moe_model, n=5)
        output = buf.getvalue()
        assert output == "", (
            f"SILENT mode produced stdout output: {output[:200]!r}"
        )


# ===========================================================================
# ── 8. Large model integration ───────────────────────────────────────────────
# ===========================================================================


class TestLargeModelIntegration:
    def test_four_layer_eight_expert_model(self) -> None:
        from conftest import FakeMoEModel
        model = FakeMoEModel(n_layers=4, n_experts=8, hidden=64)
        with MoEWatch(model, _silent_config()) as watch:
            for step in range(10):
                watch.hook_manager.set_global_step(step)
                x = torch.randn(4, 64)
                with torch.no_grad():
                    model(x)
                report = watch.step(step)
            assert len(report.risk_scores) == 4

    def test_watch_model_params_unchanged(self, moe_model: FakeMoEModel) -> None:
        """Monitoring must never modify model weights."""
        before = {
            name: p.data.clone()
            for name, p in moe_model.named_parameters()
        }
        with MoEWatch(moe_model, _silent_config()) as watch:
            _run_steps(watch, moe_model, n=5)
        after = {
            name: p.data
            for name, p in moe_model.named_parameters()
        }
        for name in before:
            assert torch.allclose(before[name], after[name]), (
                f"Parameter '{name}' was modified by MoEWatch"
            )


# ===========================================================================
# ── 9. WatchReport structure ─────────────────────────────────────────────────
# ===========================================================================


class TestWatchReport:
    def test_watch_report_risk_scores_is_dict(self, moe_model: FakeMoEModel) -> None:
        with MoEWatch(moe_model, _silent_config()) as watch:
            x = torch.randn(4, moe_model.hidden)
            with torch.no_grad():
                moe_model(x)
            report = watch.step(0)
            assert isinstance(report.risk_scores, dict)

    def test_watch_report_alerts_is_list(self, moe_model: FakeMoEModel) -> None:
        with MoEWatch(moe_model, _silent_config()) as watch:
            x = torch.randn(4, moe_model.hidden)
            with torch.no_grad():
                moe_model(x)
            report = watch.step(0)
            assert isinstance(report.alerts, list)

    def test_watch_report_step_matches_call(self, moe_model: FakeMoEModel) -> None:
        with MoEWatch(moe_model, _silent_config()) as watch:
            x = torch.randn(4, moe_model.hidden)
            with torch.no_grad():
                moe_model(x)
            report = watch.step(77)
            assert report.step == 77
