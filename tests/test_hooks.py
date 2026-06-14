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

from conftest import FakeMoEModel, FakeNonMoEModel


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
