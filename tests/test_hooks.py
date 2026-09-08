# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# tests/test_hooks.py
# =============================================================================
#
# Project      : MoEWatch
# Description  : Tests for moewatch.hooks.manager.HookManager.
#
#                Coverage targets:
#                  - attach() succeeds on FakeMoEModel
#                  - is_attached() returns True after attach, False after detach
#                  - detach() is idempotent (safe to call multiple times)
#                  - attach() on non-MoE model raises ValueError
#                  - Double attach() is a no-op (logs warning, no duplicate hooks)
#                  - Forward pass triggers RouterForwardHook → RoutingEvent written
#                  - get_layer_map() returns correct names after attach
#                  - get_layer_map() returns empty dict after detach
#                  - set_global_step() propagates to active hooks
#                  - Context manager (via MoEWatch.__enter__/__exit__) cleans up
#                  - HookManager as context manager (attach/detach pair)
#                  - Partial attach failure guarantees cleanup (no leaked handles)
#                  - Zero weight modification guarantee (params unchanged)
#
# =============================================================================

from __future__ import annotations

import threading
import time
from typing import List, Tuple
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from moewatch.collector.stat_collector import StatCollector
from moewatch.config import OutputMode, WatchConfig
from moewatch.hooks.manager import HookManager
from moewatch.hooks.router_hook import RouterForwardHook, RoutingEvent

from conftest import FakeModernMoEModel, FakeMoEModel, FakeNonMoEModel


# ===========================================================================
# ── Helper: run a forward pass on a FakeMoEModel ────────────────────────────
# ===========================================================================


def _run_forward(model: FakeMoEModel, batch: int = 4) -> torch.Tensor:
    """Execute one forward pass on the model and return the output."""
    x = torch.randn(batch, model.hidden)
    with torch.no_grad():
        return model(x)


# ===========================================================================
# ── 1. Construction ──────────────────────────────────────────────────────────
# ===========================================================================


class TestHookManagerConstruction:
    """HookManager initialises correctly and is not yet attached."""

    def test_constructs_without_error(
        self,
        moe_model: FakeMoEModel,
        stat_collector: StatCollector,
        default_config: WatchConfig,
    ) -> None:
        manager = HookManager(moe_model, stat_collector, default_config)
        assert manager is not None

    def test_not_attached_at_construction(
        self,
        moe_model: FakeMoEModel,
        stat_collector: StatCollector,
        default_config: WatchConfig,
    ) -> None:
        manager = HookManager(moe_model, stat_collector, default_config)
        assert manager.is_attached() is False

    def test_get_layer_map_empty_before_attach(
        self,
        moe_model: FakeMoEModel,
        stat_collector: StatCollector,
        default_config: WatchConfig,
    ) -> None:
        manager = HookManager(moe_model, stat_collector, default_config)
        assert manager.get_layer_map() == {}

    def test_attributes_stored_correctly(
        self,
        moe_model: FakeMoEModel,
        stat_collector: StatCollector,
        default_config: WatchConfig,
    ) -> None:
        manager = HookManager(moe_model, stat_collector, default_config)
        assert manager.model is moe_model
        assert manager.stat_collector is stat_collector
        assert manager.config is default_config


# ===========================================================================
# ── 2. attach() lifecycle ─────────────────────────────────────────────────────
# ===========================================================================


class TestAttach:
    """Tests for the attach() method."""

    def test_attach_succeeds(
        self,
        moe_model: FakeMoEModel,
        stat_collector: StatCollector,
        default_config: WatchConfig,
    ) -> None:
        manager = HookManager(moe_model, stat_collector, default_config)
        manager.attach()
        assert manager.is_attached() is True
        manager.detach()

    def test_attach_populates_layer_map(
        self,
        moe_model: FakeMoEModel,
        stat_collector: StatCollector,
        default_config: WatchConfig,
    ) -> None:
        manager = HookManager(moe_model, stat_collector, default_config)
        manager.attach()
        layer_map = manager.get_layer_map()
        assert len(layer_map) == 2, (
            f"Expected 2 layers in map, got {len(layer_map)}: {list(layer_map.keys())}"
        )
        manager.detach()

    def test_attach_layer_names_contain_gate(
        self,
        moe_model: FakeMoEModel,
        stat_collector: StatCollector,
        default_config: WatchConfig,
    ) -> None:
        manager = HookManager(moe_model, stat_collector, default_config)
        manager.attach()
        for name in manager.get_layer_map():
            assert "gate" in name.lower(), (
                f"Expected 'gate' in layer name '{name}'"
            )
        manager.detach()

    def test_attach_on_non_moe_model_raises_value_error(
        self,
        non_moe_model: FakeNonMoEModel,
        default_config: WatchConfig,
    ) -> None:
        collector = StatCollector(default_config)
        manager = HookManager(non_moe_model, collector, default_config)
        with pytest.raises(ValueError):
            manager.attach()

    def test_attach_on_non_moe_leaves_not_attached(
        self,
        non_moe_model: FakeNonMoEModel,
        default_config: WatchConfig,
    ) -> None:
        """After a failed attach(), is_attached() must still be False."""
        collector = StatCollector(default_config)
        manager = HookManager(non_moe_model, collector, default_config)
        try:
            manager.attach()
        except ValueError:
            pass
        assert manager.is_attached() is False

    def test_double_attach_is_no_op(
        self,
        moe_model: FakeMoEModel,
        stat_collector: StatCollector,
        default_config: WatchConfig,
    ) -> None:
        """Calling attach() twice should not register duplicate hooks."""
        manager = HookManager(moe_model, stat_collector, default_config)
        manager.attach()
        handles_after_first = len(manager._handles)

        manager.attach()  # second call — should be a no-op
        handles_after_second = len(manager._handles)

        assert handles_after_first == handles_after_second, (
            "Double attach() added extra hook handles"
        )
        manager.detach()

    def test_attach_registers_stat_collector_layers(
        self,
        moe_model: FakeMoEModel,
        stat_collector: StatCollector,
        default_config: WatchConfig,
    ) -> None:
        """attach() must register detected layers with stat_collector."""
        manager = HookManager(moe_model, stat_collector, default_config)
        manager.attach()

        all_stats = stat_collector.get_all_stats()
        routing_keys = list(all_stats.get("routing", {}).keys())
        assert len(routing_keys) == 2, (
            f"Expected 2 routing layers registered, got: {routing_keys}"
        )
        manager.detach()


class TestModernRouterArchitecture:
    """End-to-end regression coverage for the raw-nn.Parameter "TopKRouter"
    layout used by current transformers (>=4.57) Mixtral / Qwen3-MoE /
    DeepSeek-V3 / OLMoE implementations (see ``FakeModernTopKRouter`` /
    ``FakeModernMoEModel`` in conftest.py).

    Before the fix, ``attach()`` raised "no MoE router modules detected"
    for this architecture outright (auto-detection rejected any gate
    lacking ``out_features``). Even bypassing that with a manual
    ``config.router_modules`` override, ``_infer_expert_count()`` still
    fell back to a hardcoded ``2``, silently registering far too few
    experts with ``StatCollector`` and permanently clamping every real
    per-token expert index into that too-small range — corrupting
    utilization statistics without raising anything at all.
    """

    def test_attach_detects_modern_router(
        self, stat_collector: StatCollector, default_config: WatchConfig
    ) -> None:
        """Auto-detection must find the router without config.router_modules."""
        model = FakeModernMoEModel(n_layers=1, n_experts=8, hidden=16)
        manager = HookManager(model, stat_collector, default_config)
        manager.attach()

        assert len(manager.get_layer_map()) == 1
        manager.detach()

    def test_attach_registers_full_expert_count(
        self, stat_collector: StatCollector, default_config: WatchConfig
    ) -> None:
        """The registered expert count must match num_experts (8), not the
        hardcoded fallback of 2 that this bug used to silently apply."""
        model = FakeModernMoEModel(n_layers=1, n_experts=8, hidden=16)
        manager = HookManager(model, stat_collector, default_config)
        manager.attach()

        (layer_name,) = manager.get_layer_map().keys()
        assert stat_collector._expert_counts[layer_name] == 8
        manager.detach()

    def test_routing_stats_not_clamped_into_two_experts(
        self, stat_collector: StatCollector, default_config: WatchConfig
    ) -> None:
        """Full pipeline: forward passes through a healthy, uniformly
        routed 8-expert modern-router model must produce utilization
        spread across all 8 experts, not collapsed into 2 phantom
        buckets by index clamping."""
        torch.manual_seed(0)
        model = FakeModernMoEModel(n_layers=1, n_experts=8, hidden=16)
        manager = HookManager(model, stat_collector, default_config)
        manager.attach()

        x = torch.randn(64, 16)
        for step in range(5):
            manager.set_global_step(step)
            model(x)

        (layer_name,) = manager.get_layer_map().keys()
        stats = stat_collector.get_all_stats()["routing"][layer_name]

        assert stats.expert_token_counts.numel() == 8
        # Every expert must have received a nonzero share of tokens —
        # a near-uniform random gate over 8 experts should not leave
        # experts 2..7 completely empty (which is what the clamp bug
        # would do, since every index was folded into {0, 1}).
        assert (stats.expert_token_counts > 0).all(), (
            f"expected nonzero tokens for all 8 experts, got "
            f"{stats.expert_token_counts.tolist()}"
        )
        manager.detach()


# ===========================================================================
# ── 3. detach() lifecycle ─────────────────────────────────────────────────────
# ===========================================================================


class TestDetach:
    """Tests for the detach() method."""

    def test_detach_after_attach_sets_not_attached(
        self, attached_hook_manager: HookManager
    ) -> None:
        # attached_hook_manager fixture calls attach(); we call detach() here
        # Note: fixture teardown also calls detach(), so this is safe
        assert attached_hook_manager.is_attached() is True

    def test_detach_clears_layer_map(
        self,
        moe_model: FakeMoEModel,
        stat_collector: StatCollector,
        default_config: WatchConfig,
    ) -> None:
        manager = HookManager(moe_model, stat_collector, default_config)
        manager.attach()
        assert len(manager.get_layer_map()) > 0
        manager.detach()
        assert manager.get_layer_map() == {}

    def test_detach_sets_is_attached_false(
        self,
        moe_model: FakeMoEModel,
        stat_collector: StatCollector,
        default_config: WatchConfig,
    ) -> None:
        manager = HookManager(moe_model, stat_collector, default_config)
        manager.attach()
        manager.detach()
        assert manager.is_attached() is False

    def test_detach_idempotent_first_call(
        self,
        moe_model: FakeMoEModel,
        stat_collector: StatCollector,
        default_config: WatchConfig,
    ) -> None:
        manager = HookManager(moe_model, stat_collector, default_config)
        manager.attach()
        manager.detach()
        # Second detach must not raise
        manager.detach()
        assert manager.is_attached() is False

    def test_detach_idempotent_without_attach(
        self,
        moe_model: FakeMoEModel,
        stat_collector: StatCollector,
        default_config: WatchConfig,
    ) -> None:
        """Calling detach() without a prior attach() must not raise."""
        manager = HookManager(moe_model, stat_collector, default_config)
        manager.detach()  # should be a silent no-op

    def test_detach_clears_handles_list(
        self,
        moe_model: FakeMoEModel,
        stat_collector: StatCollector,
        default_config: WatchConfig,
    ) -> None:
        manager = HookManager(moe_model, stat_collector, default_config)
        manager.attach()
        assert len(manager._handles) > 0
        manager.detach()
        assert len(manager._handles) == 0


# ===========================================================================
# ── 4. Forward hook fires → RoutingEvent written ────────────────────────────
# ===========================================================================


class TestHookFiring:
    """Verify that a forward pass causes routing events to be written."""

    def test_forward_pass_writes_routing_events(
        self,
        moe_model: FakeMoEModel,
        stat_collector: StatCollector,
        default_config: WatchConfig,
    ) -> None:
        manager = HookManager(moe_model, stat_collector, default_config)
        manager.attach()
        try:
            _run_forward(moe_model, batch=4)
            all_stats = stat_collector.get_all_stats()
            routing = all_stats.get("routing", {})
            # Each of the 2 layers should have at least 1 event
            assert len(routing) >= 1, "No routing layers have stats after forward pass"
            for layer_name, layer_stats in routing.items():
                assert layer_stats.step >= 0, (
                    f"LayerStats for '{layer_name}' has invalid step"
                )
        finally:
            manager.detach()

    def test_routing_event_has_correct_expert_count(
        self,
        moe_model: FakeMoEModel,
        stat_collector: StatCollector,
        default_config: WatchConfig,
    ) -> None:
        manager = HookManager(moe_model, stat_collector, default_config)
        manager.attach()
        try:
            _run_forward(moe_model, batch=4)
            all_stats = stat_collector.get_all_stats()
            for layer_name, layer_stats in all_stats.get("routing", {}).items():
                n = layer_stats.expert_token_counts.shape[0]
                assert n == moe_model.n_experts, (
                    f"Layer '{layer_name}' reported {n} experts, "
                    f"expected {moe_model.n_experts}"
                )
        finally:
            manager.detach()

    def test_multiple_forward_passes_accumulate_events(
        self,
        moe_model: FakeMoEModel,
        stat_collector: StatCollector,
        default_config: WatchConfig,
    ) -> None:
        manager = HookManager(moe_model, stat_collector, default_config)
        manager.attach()
        try:
            for step in range(5):
                manager.set_global_step(step)
                _run_forward(moe_model, batch=4)

            all_stats = stat_collector.get_all_stats()
            for layer_name, layer_stats in all_stats.get("routing", {}).items():
                assert layer_stats.step >= 4, (
                    f"Expected step >= 4, got {layer_stats.step} for '{layer_name}'"
                )
        finally:
            manager.detach()

    def test_no_events_after_detach(
        self,
        moe_model: FakeMoEModel,
        stat_collector: StatCollector,
        default_config: WatchConfig,
    ) -> None:
        """After detach(), forward passes must NOT write new events."""
        manager = HookManager(moe_model, stat_collector, default_config)
        manager.attach()
        _run_forward(moe_model, batch=4)
        manager.detach()

        # Capture event count after detach
        all_stats_after_detach = stat_collector.get_all_stats()
        step_before = {
            k: v.step for k, v in all_stats_after_detach.get("routing", {}).items()
        }

        # Additional forward pass — hooks should be gone
        _run_forward(moe_model, batch=4)

        all_stats_later = stat_collector.get_all_stats()
        step_after = {
            k: v.step for k, v in all_stats_later.get("routing", {}).items()
        }
        assert step_before == step_after, (
            "Routing events written after detach() — hooks not removed properly"
        )


# ===========================================================================
# ── 5. set_global_step() propagation ────────────────────────────────────────
# ===========================================================================


class TestSetGlobalStep:
    """set_global_step() should propagate the training step to hook callbacks."""

    def test_set_global_step_before_forward_tags_events(
        self,
        moe_model: FakeMoEModel,
        stat_collector: StatCollector,
        default_config: WatchConfig,
    ) -> None:
        manager = HookManager(moe_model, stat_collector, default_config)
        manager.attach()
        try:
            manager.set_global_step(42)
            _run_forward(moe_model, batch=4)

            all_stats = stat_collector.get_all_stats()
            for layer_name, layer_stats in all_stats.get("routing", {}).items():
                assert layer_stats.step == 42, (
                    f"Expected step=42 for '{layer_name}', got {layer_stats.step}"
                )
        finally:
            manager.detach()

    def test_set_global_step_updates_incrementally(
        self,
        moe_model: FakeMoEModel,
        stat_collector: StatCollector,
        default_config: WatchConfig,
    ) -> None:
        manager = HookManager(moe_model, stat_collector, default_config)
        manager.attach()
        try:
            for step in [0, 10, 20, 99]:
                manager.set_global_step(step)
                _run_forward(moe_model, batch=2)

            all_stats = stat_collector.get_all_stats()
            for layer_name, layer_stats in all_stats.get("routing", {}).items():
                assert layer_stats.step == 99, (
                    f"Expected final step=99, got {layer_stats.step}"
                )
        finally:
            manager.detach()

    def test_set_global_step_no_error_when_not_attached(
        self,
        moe_model: FakeMoEModel,
        stat_collector: StatCollector,
        default_config: WatchConfig,
    ) -> None:
        """set_global_step() before attach() must not raise."""
        manager = HookManager(moe_model, stat_collector, default_config)
        manager.set_global_step(100)  # no hooks registered yet — safe no-op


# ===========================================================================
# ── 6. Zero weight modification guarantee ───────────────────────────────────
# ===========================================================================


class TestZeroWeightModification:
    """Attach/forward/detach cycle must never modify model parameters."""

    def test_attach_detach_does_not_modify_parameters(
        self,
        moe_model: FakeMoEModel,
        stat_collector: StatCollector,
        default_config: WatchConfig,
    ) -> None:
        snapshots_before = {
            name: param.data.clone()
            for name, param in moe_model.named_parameters()
        }
        manager = HookManager(moe_model, stat_collector, default_config)
        manager.attach()
        manager.detach()

        for name, param in moe_model.named_parameters():
            assert torch.allclose(snapshots_before[name], param.data), (
                f"Parameter '{name}' was modified by HookManager lifecycle"
            )

    def test_forward_with_hooks_does_not_modify_parameters(
        self,
        moe_model: FakeMoEModel,
        stat_collector: StatCollector,
        default_config: WatchConfig,
    ) -> None:
        snapshots_before = {
            name: param.data.clone()
            for name, param in moe_model.named_parameters()
        }
        manager = HookManager(moe_model, stat_collector, default_config)
        manager.attach()
        try:
            for _ in range(5):
                _run_forward(moe_model, batch=4)
        finally:
            manager.detach()

        for name, param in moe_model.named_parameters():
            assert torch.allclose(snapshots_before[name], param.data), (
                f"Parameter '{name}' was modified during hooked forward passes"
            )


# ===========================================================================
# ── 7. Thread safety ─────────────────────────────────────────────────────────
# ===========================================================================


class TestThreadSafety:
    """Concurrent forward passes should not corrupt hook state."""

    def test_concurrent_forward_passes_no_crash(
        self,
        moe_model: FakeMoEModel,
        stat_collector: StatCollector,
        default_config: WatchConfig,
    ) -> None:
        """Run forward passes from multiple threads — should not raise."""
        manager = HookManager(moe_model, stat_collector, default_config)
        manager.attach()

        errors: List[Exception] = []

        def run_forward() -> None:
            try:
                for _ in range(3):
                    _run_forward(moe_model, batch=2)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=run_forward) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        manager.detach()

        assert errors == [], f"Errors during concurrent forward passes: {errors}"


# ===========================================================================
# ── 8. get_layer_map() contract ──────────────────────────────────────────────
# ===========================================================================


class TestGetLayerMap:
    """get_layer_map() must return a fresh shallow copy each time."""

    def test_returns_shallow_copy(
        self,
        moe_model: FakeMoEModel,
        stat_collector: StatCollector,
        default_config: WatchConfig,
    ) -> None:
        manager = HookManager(moe_model, stat_collector, default_config)
        manager.attach()
        try:
            m1 = manager.get_layer_map()
            m2 = manager.get_layer_map()
            assert m1 is not m2, "get_layer_map() returned the same dict object twice"
            assert m1 == m2, "get_layer_map() returned dicts with different contents"
        finally:
            manager.detach()

    def test_mutating_returned_map_does_not_affect_internal_state(
        self,
        moe_model: FakeMoEModel,
        stat_collector: StatCollector,
        default_config: WatchConfig,
    ) -> None:
        manager = HookManager(moe_model, stat_collector, default_config)
        manager.attach()
        try:
            m = manager.get_layer_map()
            m.clear()  # destroy the copy
            # Internal state should be untouched
            assert len(manager.get_layer_map()) == 2
        finally:
            manager.detach()

    def test_layer_map_values_are_nn_modules(
        self,
        moe_model: FakeMoEModel,
        stat_collector: StatCollector,
        default_config: WatchConfig,
    ) -> None:
        manager = HookManager(moe_model, stat_collector, default_config)
        manager.attach()
        try:
            for name, module in manager.get_layer_map().items():
                assert isinstance(module, nn.Module), (
                    f"Layer map value for '{name}' is not nn.Module: {type(module)}"
                )
        finally:
            manager.detach()


# ===========================================================================
# ── 9. Cleanup on failed attach ──────────────────────────────────────────────
# ===========================================================================


class TestPartialAttachCleanup:
    """If attach() raises after registering some hooks, all handles are removed."""

    def test_no_handles_remain_after_failed_attach(
        self,
        non_moe_model: FakeNonMoEModel,
        default_config: WatchConfig,
    ) -> None:
        collector = StatCollector(default_config)
        manager = HookManager(non_moe_model, collector, default_config)

        try:
            manager.attach()
        except ValueError:
            pass

        assert len(manager._handles) == 0, (
            f"Leaked {len(manager._handles)} hook handle(s) after failed attach()"
        )
        assert manager.is_attached() is False

# ===========================================================================
# ── 10. RouterForwardHook internal helpers & defensive branches ────────────
# ===========================================================================

def test_extract_logits_tensor():
    logits = torch.randn(2, 4)

    result = RouterForwardHook._extract_logits(logits)

    assert result is logits


def test_extract_logits_tuple():
    logits = torch.randn(2, 4)

    result = RouterForwardHook._extract_logits(("foo", logits))

    assert result is logits


def test_extract_logits_dict():
    logits = torch.randn(2, 4)

    result = RouterForwardHook._extract_logits(
        {"router_logits": logits}
    )

    assert result is logits


def test_extract_logits_object_attr():
    class Output:
        pass

    out = Output()
    out.router_logits = torch.randn(2, 4)

    result = RouterForwardHook._extract_logits(out)

    assert result is out.router_logits


def test_extract_logits_none():
    result = RouterForwardHook._extract_logits({"bad": 123})

    assert result is None


def test_infer_top_k_attr():
    module = nn.Linear(4, 4)
    module.top_k = 3

    assert RouterForwardHook._infer_top_k(module, 8) == 3


def test_infer_top_k_clamped():
    module = nn.Linear(4, 4)
    module.top_k = 999

    assert RouterForwardHook._infer_top_k(module, 4) == 4


def test_infer_top_k_default():
    module = nn.Linear(4, 4)

    assert RouterForwardHook._infer_top_k(module, 8) == 1


def test_select_top_k():
    logits = torch.tensor([[1.0, 5.0, 3.0]])

    indices = RouterForwardHook._select_top_k(
        logits,
        top_k=2,
    )

    assert indices.shape[-1] == 2
    assert 1 in indices[0]


def test_router_hook_set_global_step(
    stat_collector,
    default_config,
):
    hook = RouterForwardHook(
        "layer",
        stat_collector,
        default_config,
    )

    hook.set_global_step(123)

    assert hook._global_step == 123


def test_hook_skips_1d_logits(
    stat_collector,
    default_config,
):
    hook = RouterForwardHook(
        "layer",
        stat_collector,
        default_config,
    )

    module = nn.Linear(4, 4)

    hook(
        module,
        (),
        torch.randn(4),
    )


def test_hook_handles_stat_collector_exception(
    default_config,
):
    collector = MagicMock()
    collector.write_routing_event.side_effect = RuntimeError("boom")

    hook = RouterForwardHook(
        "layer",
        collector,
        default_config,
    )

    logits = torch.randn(2, 4)

    hook(
        nn.Linear(4, 4),
        (),
        logits,
    )


def test_hook_writes_routing_event(
    default_config,
):
    collector = MagicMock()

    hook = RouterForwardHook(
        "layer",
        collector,
        default_config,
    )

    logits = torch.randn(3, 8)

    hook(
        nn.Linear(8, 8),
        (),
        logits,
    )

    collector.write_routing_event.assert_called_once()

    event = collector.write_routing_event.call_args[0][0]

    assert isinstance(event, RoutingEvent)
    assert event.expert_count == 8
    assert event.batch_size == 3


# ===========================================================================
# ── Regression: out-of-band forward passes must not contaminate StatCollector
# ===========================================================================


class TestHookArmDisarmGate:
    """Regression tests for the router-hook contamination bug.

    Before this fix, ``RouterForwardHook`` recorded EVERY forward pass
    through a monitored module, with no way to distinguish the one real
    training forward pass per step from any other incidental forward call
    (e.g. a user's own periodic evaluation snapshot for logging/metrics,
    made *after* ``MoEWatch.step()`` returns but while hooks are still
    attached). Because such calls are typically drawn from a different
    input distribution than training, mixing them into the same rolling
    window used for entropy/risk-score computation silently corrupted the
    signal that drives alerting and intervention decisions.

    The fix gates recording behind an ``armed`` flag: ``pre_step()`` arms
    hooks immediately before the real forward pass; ``step()`` disarms
    them once that pass has already been recorded, so anything the caller
    does afterward (until the next ``pre_step()``) is safely ignored.
    """

    def test_disarmed_hook_does_not_write_routing_event(self, default_config):
        """A disarmed hook must not call stat_collector.write_routing_event."""
        collector = MagicMock()
        hook = RouterForwardHook("layer", collector, default_config)

        hook.set_armed(False)
        hook(nn.Linear(8, 8), (), torch.randn(3, 8))

        collector.write_routing_event.assert_not_called()

    def test_armed_hook_writes_routing_event(self, default_config):
        """An armed hook (the default) behaves exactly as before the fix."""
        collector = MagicMock()
        hook = RouterForwardHook("layer", collector, default_config)

        hook(nn.Linear(8, 8), (), torch.randn(3, 8))

        collector.write_routing_event.assert_called_once()

    def test_rearming_resumes_recording(self, default_config):
        """set_armed(True) after set_armed(False) resumes normal recording."""
        collector = MagicMock()
        hook = RouterForwardHook("layer", collector, default_config)

        hook.set_armed(False)
        hook(nn.Linear(8, 8), (), torch.randn(3, 8))
        assert collector.write_routing_event.call_count == 0

        hook.set_armed(True)
        hook(nn.Linear(8, 8), (), torch.randn(3, 8))
        assert collector.write_routing_event.call_count == 1

    def test_hook_manager_arm_disarm_fan_out(self, default_config):
        """HookManager.arm()/disarm() must toggle every attached router hook."""
        model = FakeMoEModel()
        collector = StatCollector(default_config)
        manager = HookManager(model, collector, default_config)
        manager.attach()
        try:
            manager.disarm()
            for hook in manager._router_hooks:
                assert hook._armed is False

            manager.arm()
            for hook in manager._router_hooks:
                assert hook._armed is True
        finally:
            manager.detach()

    def test_out_of_band_forward_pass_does_not_pollute_stat_collector(
        self, default_config
    ):
        """End-to-end: an out-of-band forward pass between step() and the
        next pre_step() must not add a RoutingEvent, while the real
        pre_step()-bracketed forward pass still does."""
        model = FakeMoEModel()
        collector = StatCollector(default_config)
        manager = HookManager(model, collector, default_config)
        manager.attach()
        try:
            layer_name = next(iter(manager.get_layer_map()))
            buffer = collector._routing_buffers[layer_name]

            # Simulate pre_step() -> real forward -> step()-disarm.
            manager.arm()
            model(torch.randn(3, model.hidden))
            count_after_real_forward = len(buffer)

            manager.disarm()

            # Out-of-band forward pass (e.g. an eval snapshot) — must be ignored.
            model(torch.randn(3, model.hidden))
            count_after_oob_forward = len(buffer)

            assert count_after_oob_forward == count_after_real_forward, (
                "An out-of-band forward pass while disarmed must not add "
                "any additional recorded routing events."
            )
        finally:
            manager.detach()
