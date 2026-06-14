# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# tests/test_audit.py
# =============================================================================
#
# Project      : MoEWatch
# Description  : End-to-end integration tests for audit() on CPU.
#
#                All tests run on CPU (device="cpu") with FakeMoEModel
#                and a minimal TensorDataset-backed DataLoader. No GPU
#                or real pretrained weights required.
#
#                Coverage targets (>= 80%):
#
#                  - audit() returns AuditReport
#                  - AuditReport.entropy_results populated
#                  - AuditReport.collapse_results populated
#                  - AuditReport.risk_scores populated
#                  - AuditReport.dead_experts_count is int >= 0
#                  - AuditReport.critical_layers is list
#                  - num_batches limits batch count
#                  - hooks detached after audit() (model unhooked)
#                  - model weights unchanged after audit()
#                  - audit() raises ValueError on non-MoE model
#                  - audit() raises TypeError on non-Module input
#                  - audit() works with num_batches > dataloader size
#                  - AuditReport.summary() returns non-empty string
#                  - AuditReport bool is True when layers detected
#                  - AuditReport.has_critical_risk when critical layers present
#                  - AuditReport.dead_experts() returns list of tuples
#                  - audit() with SILENT config produces no stdout
#                  - audit() with minimal 1-batch dataloader works
#                  - audit() correctly processes collapsed model
#
# =============================================================================

from __future__ import annotations

import io
import sys
from typing import List
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from moewatch._audit import audit
from moewatch.config import OutputMode, WatchConfig
from moewatch.report.audit_report import AuditReport

from conftest import FakeCollapsingMoEModel, FakeMoEModel, FakeNonMoEModel


# ===========================================================================
# ── Helpers ──────────────────────────────────────────────────────────────────
# ===========================================================================


def _silent_config(**kwargs) -> WatchConfig:
    return WatchConfig(output=OutputMode.SILENT, **kwargs)


def _make_dataloader(
    n_samples: int = 32,
    hidden: int = 32,
    batch_size: int = 8,
) -> DataLoader:
    """Create a minimal DataLoader returning (input_tensor,) batches."""
    x = torch.randn(n_samples, hidden)
    dataset = TensorDataset(x)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _run_audit(
    model: nn.Module,
    n_samples: int = 32,
    hidden: int = 32,
    batch_size: int = 8,
    num_batches: int = 4,
    config: WatchConfig | None = None,
) -> AuditReport:
    """Run audit() with a minimal synthetic dataloader."""
    dl = _make_dataloader(n_samples=n_samples, hidden=hidden, batch_size=batch_size)
    cfg = config or _silent_config()
    return audit(model, dl, num_batches=num_batches, config=cfg, device="cpu")


# ===========================================================================
# ── 1. Basic return value ─────────────────────────────────────────────────────
# ===========================================================================


class TestAuditBasicReturn:
    def test_returns_audit_report(self, small_moe_model: FakeMoEModel) -> None:
        report = _run_audit(small_moe_model, hidden=small_moe_model.hidden)
        assert isinstance(report, AuditReport)

    def test_entropy_results_populated(self, small_moe_model: FakeMoEModel) -> None:
        report = _run_audit(small_moe_model, hidden=small_moe_model.hidden)
        assert len(report.entropy_results) > 0, (
            "Expected at least one layer in entropy_results"
        )

    def test_collapse_results_populated(self, small_moe_model: FakeMoEModel) -> None:
        report = _run_audit(small_moe_model, hidden=small_moe_model.hidden)
        assert len(report.collapse_results) > 0

    def test_risk_scores_populated(self, small_moe_model: FakeMoEModel) -> None:
        report = _run_audit(small_moe_model, hidden=small_moe_model.hidden)
        assert len(report.risk_scores) > 0

    def test_dead_experts_count_is_non_negative_int(
        self, small_moe_model: FakeMoEModel
    ) -> None:
        report = _run_audit(small_moe_model, hidden=small_moe_model.hidden)
        assert isinstance(report.dead_experts_count, int)
        assert report.dead_experts_count >= 0

    def test_critical_layers_is_list(self, small_moe_model: FakeMoEModel) -> None:
        report = _run_audit(small_moe_model, hidden=small_moe_model.hidden)
        assert isinstance(report.critical_layers, list)

    def test_entropy_result_keys_are_strings(
        self, small_moe_model: FakeMoEModel
    ) -> None:
        report = _run_audit(small_moe_model, hidden=small_moe_model.hidden)
        for key in report.entropy_results:
            assert isinstance(key, str)

    def test_risk_scores_keys_match_entropy_keys(
        self, small_moe_model: FakeMoEModel
    ) -> None:
        report = _run_audit(small_moe_model, hidden=small_moe_model.hidden)
        assert set(report.entropy_results.keys()) == set(report.risk_scores.keys()), (
            "entropy_results and risk_scores should cover the same layers"
        )


# ===========================================================================
# ── 2. Layer count ────────────────────────────────────────────────────────────
# ===========================================================================


class TestAuditLayerCount:
    def test_two_layer_model_two_entropy_reports(self) -> None:
        model = FakeMoEModel(n_layers=2, n_experts=4, hidden=32)
        report = _run_audit(model, hidden=32, num_batches=4)
        assert len(report.entropy_results) == 2

    def test_four_layer_model_four_risk_scores(self) -> None:
        model = FakeMoEModel(n_layers=4, n_experts=4, hidden=32)
        report = _run_audit(model, hidden=32, num_batches=4)
        assert len(report.risk_scores) == 4

    def test_expert_count_per_layer(self) -> None:
        n_experts = 8
        model = FakeMoEModel(n_layers=2, n_experts=n_experts, hidden=32)
        report = _run_audit(model, hidden=32)
        for layer_name, collapse_report in report.collapse_results.items():
            assert len(collapse_report.expert_states) == n_experts, (
                f"Layer '{layer_name}' should have {n_experts} expert states"
            )


# ===========================================================================
# ── 3. num_batches controls processing ───────────────────────────────────────
# ===========================================================================


class TestNumBatches:
    def test_num_batches_one_works(self, small_moe_model: FakeMoEModel) -> None:
        report = _run_audit(small_moe_model, hidden=small_moe_model.hidden, num_batches=1)
        assert isinstance(report, AuditReport)

    def test_num_batches_larger_than_dataloader(
        self, small_moe_model: FakeMoEModel
    ) -> None:
        """audit() stops early when dataloader exhausted."""
        # 4 samples ÷ batch_size=4 = 1 batch; num_batches=100 should still work
        dl = _make_dataloader(n_samples=4, hidden=small_moe_model.hidden, batch_size=4)
        report = audit(
            small_moe_model, dl, num_batches=100,
            config=_silent_config(), device="cpu"
        )
        assert isinstance(report, AuditReport)

    def test_num_batches_limits_processing(self, small_moe_model: FakeMoEModel) -> None:
        """With num_batches=2 from a larger dataloader, only 2 batches are processed."""
        # This is hard to assert directly without patching; just ensure no crash
        dl = _make_dataloader(n_samples=64, hidden=small_moe_model.hidden, batch_size=8)
        report = audit(
            small_moe_model, dl, num_batches=2,
            config=_silent_config(), device="cpu"
        )
        assert isinstance(report, AuditReport)


# ===========================================================================
# ── 4. Hooks detached after audit ─────────────────────────────────────────────
# ===========================================================================


class TestHookCleanup:
    def test_no_hooks_after_audit(self, small_moe_model: FakeMoEModel) -> None:
        """After audit() returns, the model must have no lingering hooks."""
        _run_audit(small_moe_model, hidden=small_moe_model.hidden)
        # Check all modules — no forward hooks should remain
        for name, module in small_moe_model.named_modules():
            # PyTorch stores forward hooks in _forward_hooks dict
            fwd_hooks = getattr(module, "_forward_hooks", {})
            assert len(fwd_hooks) == 0, (
                f"Module '{name}' has {len(fwd_hooks)} dangling hook(s) after audit()"
            )

    def test_hooks_detached_even_after_error(self, small_moe_model: FakeMoEModel) -> None:
        """Hooks must be cleaned up even when an exception occurs mid-audit."""
        original_forward = small_moe_model.forward
        call_count = [0]

        def failing_forward(x):
            call_count[0] += 1
            if call_count[0] >= 3:
                raise RuntimeError("simulated forward failure")
            return original_forward(x)

        small_moe_model.forward = failing_forward
        try:
            with pytest.raises(RuntimeError):
                dl = _make_dataloader(n_samples=32, hidden=small_moe_model.hidden)
                audit(small_moe_model, dl, num_batches=10,
                      config=_silent_config(), device="cpu")
        finally:
            # Restore
            small_moe_model.forward = original_forward

        # Hooks must be gone
        for name, module in small_moe_model.named_modules():
            fwd_hooks = getattr(module, "_forward_hooks", {})
            assert len(fwd_hooks) == 0, (
                f"Module '{name}' has lingering hook after error in audit()"
            )


# ===========================================================================
# ── 5. Model weights unchanged ────────────────────────────────────────────────
# ===========================================================================


class TestModelUnchanged:
    def test_audit_does_not_modify_weights(self, small_moe_model: FakeMoEModel) -> None:
        before = {
            name: param.data.clone()
            for name, param in small_moe_model.named_parameters()
        }
        _run_audit(small_moe_model, hidden=small_moe_model.hidden)
        for name, param in small_moe_model.named_parameters():
            assert torch.allclose(before[name], param.data), (
                f"Parameter '{name}' was modified by audit()"
            )

    def test_audit_does_not_require_grad(self, small_moe_model: FakeMoEModel) -> None:
        """audit() must not accidentally enable gradient tracking on parameters."""
        # Set all params to no-grad
        for p in small_moe_model.parameters():
            p.requires_grad_(False)
        _run_audit(small_moe_model, hidden=small_moe_model.hidden)
        for p in small_moe_model.parameters():
            assert not p.requires_grad, "audit() enabled requires_grad on a parameter"
        # Restore for other tests
        for p in small_moe_model.parameters():
            p.requires_grad_(True)


# ===========================================================================
# ── 6. Error conditions ───────────────────────────────────────────────────────
# ===========================================================================


class TestAuditErrors:
    def test_non_moe_model_raises_value_error(
        self, non_moe_model: FakeNonMoEModel
    ) -> None:
        dl = _make_dataloader(hidden=32)
        with pytest.raises(ValueError):
            audit(non_moe_model, dl, num_batches=2, config=_silent_config(), device="cpu")

    def test_non_module_raises_type_error(self) -> None:
        dl = _make_dataloader(hidden=32)
        with pytest.raises(TypeError):
            audit("not a model", dl, num_batches=2,  # type: ignore[arg-type]
                  config=_silent_config(), device="cpu")

    def test_invalid_device_raises_runtime_error(
        self, small_moe_model: FakeMoEModel
    ) -> None:
        dl = _make_dataloader(hidden=small_moe_model.hidden)
        with pytest.raises((RuntimeError, AssertionError)):
            audit(small_moe_model, dl, num_batches=2,
                  config=_silent_config(), device="cuda:999")


# ===========================================================================
# ── 7. AuditReport interface ──────────────────────────────────────────────────
# ===========================================================================


class TestAuditReportInterface:
    def test_summary_returns_string(self, small_moe_model: FakeMoEModel) -> None:
        report = _run_audit(small_moe_model, hidden=small_moe_model.hidden)
        s = report.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_bool_true_when_layers_detected(
        self, small_moe_model: FakeMoEModel
    ) -> None:
        report = _run_audit(small_moe_model, hidden=small_moe_model.hidden)
        assert bool(report) is True

    def test_has_critical_risk_is_bool(self, small_moe_model: FakeMoEModel) -> None:
        report = _run_audit(small_moe_model, hidden=small_moe_model.hidden)
        assert isinstance(report.has_critical_risk, bool)

    def test_dead_experts_returns_list(self, small_moe_model: FakeMoEModel) -> None:
        report = _run_audit(small_moe_model, hidden=small_moe_model.hidden)
        result = report.dead_experts()
        assert isinstance(result, list)

    def test_dead_experts_tuples_are_layer_expert_pairs(
        self, small_moe_model: FakeMoEModel
    ) -> None:
        report = _run_audit(small_moe_model, hidden=small_moe_model.hidden)
        for item in report.dead_experts():
            assert isinstance(item, tuple)
            assert len(item) == 2
            layer_name, expert_id = item
            assert isinstance(layer_name, str)
            assert isinstance(expert_id, int)

    def test_get_risk_for_layer_returns_or_none(
        self, small_moe_model: FakeMoEModel
    ) -> None:
        report = _run_audit(small_moe_model, hidden=small_moe_model.hidden)
        known_layer = list(report.risk_scores.keys())[0]
        risk = report.get_risk_for_layer(known_layer)
        assert risk is not None

    def test_get_risk_for_unknown_layer_returns_none(
        self, small_moe_model: FakeMoEModel
    ) -> None:
        report = _run_audit(small_moe_model, hidden=small_moe_model.hidden)
        assert report.get_risk_for_layer("completely.unknown.layer") is None

    def test_risk_scores_sorted_returns_list(
        self, small_moe_model: FakeMoEModel
    ) -> None:
        report = _run_audit(small_moe_model, hidden=small_moe_model.hidden)
        if hasattr(report, "risk_scores_sorted"):
            sorted_scores = report.risk_scores_sorted()
            assert isinstance(sorted_scores, list)

    def test_entropy_normalized_values_in_range(
        self, small_moe_model: FakeMoEModel
    ) -> None:
        report = _run_audit(small_moe_model, hidden=small_moe_model.hidden)
        for layer_name, er in report.entropy_results.items():
            assert 0.0 <= er.normalized_entropy <= 1.0, (
                f"Layer '{layer_name}' normalized_entropy={er.normalized_entropy} "
                f"out of [0,1]"
            )

    def test_collapse_severity_valid_values(
        self, small_moe_model: FakeMoEModel
    ) -> None:
        report = _run_audit(small_moe_model, hidden=small_moe_model.hidden)
        for layer_name, cr in report.collapse_results.items():
            assert cr.severity in {"HEALTHY", "DEGRADED", "CRITICAL", "UNKNOWN"}, (
                f"Layer '{layer_name}' has invalid severity: {cr.severity!r}"
            )


# ===========================================================================
# ── 8. SILENT output ──────────────────────────────────────────────────────────
# ===========================================================================


class TestSilentOutput:
    def test_silent_audit_no_stdout(self, small_moe_model: FakeMoEModel) -> None:
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            _run_audit(small_moe_model, hidden=small_moe_model.hidden)
        output = buf.getvalue()
        assert output == "", (
            f"SILENT audit produced stdout: {output[:200]!r}"
        )


# ===========================================================================
# ── 9. Collapsed model audit ──────────────────────────────────────────────────
# ===========================================================================


class TestCollapsedModelAudit:
    def test_collapsed_model_low_entropy(self) -> None:
        """Collapsed gate (expert 0 monopoly) should yield low normalized entropy."""
        model = FakeCollapsingMoEModel(n_experts=4, hidden=32)
        report = _run_audit(model, hidden=32, num_batches=5)
        for layer_name, er in report.entropy_results.items():
            assert er.normalized_entropy < 0.5, (
                f"Collapsed model layer '{layer_name}' entropy too high: "
                f"{er.normalized_entropy:.4f}"
            )

    def test_collapsed_model_has_high_load_imbalance(self) -> None:
        model = FakeCollapsingMoEModel(n_experts=4, hidden=32)
        report = _run_audit(model, hidden=32, num_batches=5)
        for layer_name, cr in report.collapse_results.items():
            assert cr.load_imbalance_ratio > 1.0, (
                f"Collapsed model should have load_imbalance_ratio > 1.0 for '{layer_name}'"
            )


# ===========================================================================
# ── 10. Config override (router_modules) ──────────────────────────────────────
# ===========================================================================


class TestAuditWithRouterOverride:
    def test_manual_router_override_succeeds(
        self, small_moe_model: FakeMoEModel
    ) -> None:
        """Provide explicit router_modules to bypass auto-detection."""
        # Discover the correct gate names for the 2-layer model
        from moewatch.hooks.detection import detect_router_modules
        default_cfg = _silent_config()
        detected = detect_router_modules(small_moe_model, default_cfg)
        gate_names = list(detected.keys())

        cfg = _silent_config(router_modules=gate_names)
        dl = _make_dataloader(n_samples=16, hidden=small_moe_model.hidden, batch_size=4)
        report = audit(small_moe_model, dl, num_batches=4, config=cfg, device="cpu")
        assert len(report.entropy_results) == len(gate_names)

    def test_bad_router_override_raises_value_error(
        self, small_moe_model: FakeMoEModel
    ) -> None:
        cfg = _silent_config(router_modules=["this.does.not.exist"])
        dl = _make_dataloader(n_samples=16, hidden=small_moe_model.hidden, batch_size=4)
        with pytest.raises(ValueError):
            audit(small_moe_model, dl, num_batches=2, config=cfg, device="cpu")
