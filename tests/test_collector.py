# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# tests/test_collector.py
# =============================================================================
#
# Project      : MoEWatch
# Description  : Tests for RingBuffer (capacity, eviction, thread-safety) and
#                StatCollector (layer registration, event dispatch, stat
#                computation).
#
#                Coverage targets:
#
#                RingBuffer
#                  - write() / read_all() round-trip
#                  - capacity wrapping: oldest events evicted correctly
#                  - read_window() returns ≤ window_size most-recent events
#                  - read_window(0) returns []
#                  - read_window(n > length) returns all events
#                  - length() and __len__ consistent
#                  - clear() empties buffer; subsequent reads return []
#                  - invalid capacity raises ValueError
#                  - thread-safe concurrent writes produce no data corruption
#                  - returned lists are copies (mutation safe)
#
#                StatCollector
#                  - register_layer() creates buffers
#                  - write_routing_event() dispatches to correct layer buffer
#                  - write_gradient_event() dispatches correctly
#                  - get_layer_stats() returns LayerStats with correct fields
#                  - LayerStats.expert_token_counts sums correctly
#                  - LayerStats.expert_utilization sums to 1.0
#                  - LayerStats.load_imbalance_ratio == 1.0 for uniform routing
#                  - LayerStats.load_imbalance_ratio > 1.0 for skewed routing
#                  - get_all_stats() returns both routing and gradient keys
#                  - write to unregistered layer does not crash (auto-registers)
#                  - thread-safe concurrent writes
#
# =============================================================================

from __future__ import annotations

import threading
import time
from typing import List

import pytest
import torch

from moewatch.collector.ring_buffer import RingBuffer
from moewatch.collector.stat_collector import GradientStats, LayerStats, StatCollector
from moewatch.config import OutputMode, WatchConfig
from moewatch.hooks.gradient_hook import GradientEvent
from moewatch.hooks.router_hook import RoutingEvent

from conftest import make_gradient_event, make_routing_event


# ===========================================================================
# ── Section A: RingBuffer Tests ──────────────────────────────────────────────
# ===========================================================================


class TestRingBufferConstruction:
    """Construction and validation."""

    def test_default_capacity(self) -> None:
        buf: RingBuffer = RingBuffer()
        assert buf.capacity == 1000

    def test_custom_capacity(self) -> None:
        buf: RingBuffer = RingBuffer(capacity=42)
        assert buf.capacity == 42

    def test_empty_on_construction(self) -> None:
        buf: RingBuffer = RingBuffer(capacity=10)
        assert buf.length() == 0

    @pytest.mark.parametrize("bad_cap", [0, -1, -100])
    def test_zero_or_negative_capacity_raises(self, bad_cap: int) -> None:
        with pytest.raises(ValueError):
            RingBuffer(capacity=bad_cap)

    def test_non_integer_capacity_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            RingBuffer(capacity=3.5)  # type: ignore[arg-type]

    def test_capacity_one_is_valid(self) -> None:
        buf: RingBuffer = RingBuffer(capacity=1)
        assert buf.capacity == 1

    def test_repr_contains_capacity(self) -> None:
        buf: RingBuffer = RingBuffer(capacity=77)
        assert "77" in repr(buf)


class TestRingBufferWriteAndRead:
    """Core write / read semantics."""

    def test_write_increases_length(self, small_ring_buffer: RingBuffer) -> None:
        small_ring_buffer.write(make_routing_event(global_step=0))
        assert small_ring_buffer.length() == 1

    def test_write_two_events(self, small_ring_buffer: RingBuffer) -> None:
        small_ring_buffer.write(make_routing_event(global_step=0))
        small_ring_buffer.write(make_routing_event(global_step=1))
        assert small_ring_buffer.length() == 2

    def test_read_all_returns_all_events(self, small_ring_buffer: RingBuffer) -> None:
        events = [make_routing_event(global_step=i) for i in range(3)]
        for e in events:
            small_ring_buffer.write(e)
        result = small_ring_buffer.read_all()
        assert len(result) == 3

    def test_read_all_order_is_chronological(self, small_ring_buffer: RingBuffer) -> None:
        """Events should be returned oldest-first (insertion order)."""
        for step in range(3):
            small_ring_buffer.write(make_routing_event(global_step=step))
        result = small_ring_buffer.read_all()
        steps = [e.global_step for e in result]
        assert steps == sorted(steps), f"Events not in chronological order: {steps}"

    def test_read_all_empty_buffer_returns_empty_list(
        self, small_ring_buffer: RingBuffer
    ) -> None:
        assert small_ring_buffer.read_all() == []


class TestRingBufferCapacityWrapping:
    """
    Core requirement: once capacity is reached, the oldest event is evicted.
    The architecture doc specifies this explicitly.
    """

    def test_writing_beyond_capacity_evicts_oldest(self) -> None:
        """Write capacity+1 events; oldest (step=0) must be gone."""
        capacity = 5
        buf: RingBuffer = RingBuffer(capacity=capacity)
        for step in range(capacity + 1):
            buf.write(make_routing_event(global_step=step))

        result = buf.read_all()
        steps = [e.global_step for e in result]

        assert len(result) == capacity, (
            f"Buffer should contain exactly {capacity} events, got {len(result)}"
        )
        assert 0 not in steps, (
            f"Oldest event (step=0) should have been evicted, but steps={steps}"
        )
        assert 1 in steps, "step=1 should be the new oldest"
        assert capacity in steps, f"step={capacity} (newest) must be present"

    def test_writing_double_capacity_retains_only_latest_n(self) -> None:
        capacity = 10
        buf: RingBuffer = RingBuffer(capacity=capacity)
        for step in range(capacity * 2):
            buf.write(make_routing_event(global_step=step))

        result = buf.read_all()
        steps = [e.global_step for e in result]

        assert len(result) == capacity
        # The latest 'capacity' steps should be present: [10, 11, ..., 19]
        expected_steps = list(range(capacity, capacity * 2))
        assert steps == expected_steps, (
            f"Expected {expected_steps}, got {steps}"
        )

    def test_length_never_exceeds_capacity(self) -> None:
        capacity = 3
        buf: RingBuffer = RingBuffer(capacity=capacity)
        for step in range(20):
            buf.write(make_routing_event(global_step=step))
            assert buf.length() <= capacity, (
                f"Buffer length {buf.length()} exceeded capacity {capacity} at step {step}"
            )

    def test_capacity_exactly_one_always_has_newest(self) -> None:
        buf: RingBuffer = RingBuffer(capacity=1)
        for step in range(5):
            buf.write(make_routing_event(global_step=step))

        result = buf.read_all()
        assert len(result) == 1
        assert result[0].global_step == 4, (
            f"Expected step=4 (newest), got {result[0].global_step}"
        )

    def test_event_order_preserved_after_wrap(self) -> None:
        """After wrap, remaining events must still be in chronological order."""
        capacity = 5
        buf: RingBuffer = RingBuffer(capacity=capacity)
        n_writes = 12
        for step in range(n_writes):
            buf.write(make_routing_event(global_step=step))

        result = buf.read_all()
        steps = [e.global_step for e in result]
        assert steps == sorted(steps), (
            f"Events out of order after wrap: {steps}"
        )


class TestRingBufferReadWindow:
    """read_window() semantics."""

    def test_read_window_returns_correct_count(
        self, prefilled_ring_buffer: RingBuffer
    ) -> None:
        """prefilled_ring_buffer has 10 events (capacity=10)."""
        result = prefilled_ring_buffer.read_window(3)
        assert len(result) == 3

    def test_read_window_returns_most_recent(
        self, prefilled_ring_buffer: RingBuffer
    ) -> None:
        """read_window(3) should return the 3 newest events."""
        result = prefilled_ring_buffer.read_window(3)
        steps = [e.global_step for e in result]
        # Steps 0–9 were written; window(3) should give [7, 8, 9]
        assert steps == [7, 8, 9], f"Expected [7,8,9], got {steps}"

    def test_read_window_zero_returns_empty(
        self, prefilled_ring_buffer: RingBuffer
    ) -> None:
        result = prefilled_ring_buffer.read_window(0)
        assert result == []

    def test_read_window_larger_than_length_returns_all(
        self, small_ring_buffer: RingBuffer
    ) -> None:
        for step in range(3):
            small_ring_buffer.write(make_routing_event(global_step=step))
        result = small_ring_buffer.read_window(100)
        assert len(result) == 3

    def test_read_window_equal_to_length(
        self, prefilled_ring_buffer: RingBuffer
    ) -> None:
        result = prefilled_ring_buffer.read_window(10)
        assert len(result) == 10

    def test_read_window_one(self, prefilled_ring_buffer: RingBuffer) -> None:
        result = prefilled_ring_buffer.read_window(1)
        assert len(result) == 1
        assert result[0].global_step == 9, (
            f"read_window(1) should return newest event (step=9), got {result[0].global_step}"
        )

    def test_read_window_empty_buffer(self, small_ring_buffer: RingBuffer) -> None:
        result = small_ring_buffer.read_window(5)
        assert result == []

    def test_read_window_negative_returns_empty(
        self, prefilled_ring_buffer: RingBuffer
    ) -> None:
        result = prefilled_ring_buffer.read_window(-1)
        assert result == []


class TestRingBufferClear:
    """clear() semantics."""

    def test_clear_empties_buffer(self, prefilled_ring_buffer: RingBuffer) -> None:
        assert prefilled_ring_buffer.length() == 10
        prefilled_ring_buffer.clear()
        assert prefilled_ring_buffer.length() == 0

    def test_clear_read_all_returns_empty(
        self, prefilled_ring_buffer: RingBuffer
    ) -> None:
        prefilled_ring_buffer.clear()
        assert prefilled_ring_buffer.read_all() == []

    def test_clear_allows_new_writes(self, prefilled_ring_buffer: RingBuffer) -> None:
        prefilled_ring_buffer.clear()
        prefilled_ring_buffer.write(make_routing_event(global_step=99))
        assert prefilled_ring_buffer.length() == 1

    def test_clear_on_empty_buffer_no_error(
        self, small_ring_buffer: RingBuffer
    ) -> None:
        small_ring_buffer.clear()  # already empty
        assert small_ring_buffer.length() == 0


class TestRingBufferLenDunder:
    """__len__ delegates to length()."""

    def test_len_empty(self, small_ring_buffer: RingBuffer) -> None:
        assert len(small_ring_buffer) == 0

    def test_len_after_writes(self, small_ring_buffer: RingBuffer) -> None:
        for i in range(3):
            small_ring_buffer.write(make_routing_event(global_step=i))
        assert len(small_ring_buffer) == 3


class TestRingBufferReturnedListsAreCopies:
    """read_all() and read_window() must return copies, not the internal deque."""

    def test_read_all_mutation_does_not_affect_buffer(
        self, prefilled_ring_buffer: RingBuffer
    ) -> None:
        result = prefilled_ring_buffer.read_all()
        original_len = len(result)
        result.clear()
        assert prefilled_ring_buffer.length() == original_len, (
            "Mutating read_all() result corrupted internal buffer"
        )

    def test_read_window_mutation_does_not_affect_buffer(
        self, prefilled_ring_buffer: RingBuffer
    ) -> None:
        result = prefilled_ring_buffer.read_window(5)
        result.clear()
        assert prefilled_ring_buffer.length() == 10


class TestRingBufferThreadSafety:
    """Concurrent writes must not produce data corruption."""

    def test_concurrent_writes_do_not_corrupt(self) -> None:
        capacity = 100
        buf: RingBuffer = RingBuffer(capacity=capacity)
        n_threads = 8
        writes_per_thread = 50
        errors: List[Exception] = []

        def writer(thread_id: int) -> None:
            try:
                for i in range(writes_per_thread):
                    buf.write(make_routing_event(global_step=thread_id * 1000 + i))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors during concurrent writes: {errors}"
        # Buffer must not exceed capacity
        assert buf.length() <= capacity

    def test_concurrent_read_write_no_deadlock(self) -> None:
        buf: RingBuffer = RingBuffer(capacity=50)
        errors: List[Exception] = []
        stop_flag = threading.Event()

        def writer() -> None:
            step = 0
            while not stop_flag.is_set():
                buf.write(make_routing_event(global_step=step))
                step += 1

        def reader() -> None:
            for _ in range(20):
                try:
                    _ = buf.read_all()
                    _ = buf.read_window(10)
                except Exception as exc:
                    errors.append(exc)
                    break

        r = threading.Thread(target=reader)
        w = threading.Thread(target=writer)
        r.start()
        w.start()
        r.join(timeout=5)
        stop_flag.set()
        w.join(timeout=2)

        assert errors == [], f"Errors during concurrent read/write: {errors}"


# ===========================================================================
# ── Section B: StatCollector Tests ──────────────────────────────────────────
# ===========================================================================


class TestStatCollectorConstruction:
    """StatCollector initialises cleanly."""

    def test_constructs_without_error(self, default_config: WatchConfig) -> None:
        collector = StatCollector(default_config)
        assert collector is not None

    def test_no_layers_registered_initially(self, stat_collector: StatCollector) -> None:
        stats = stat_collector.get_all_stats()
        assert stats["routing"] == {}
        assert stats["gradient"] == {}


class TestStatCollectorRegisterLayer:
    """register_layer() creates routing and gradient buffers."""

    def test_register_creates_routing_buffer(
        self, stat_collector: StatCollector
    ) -> None:
        stat_collector.register_layer("layers.0.gate", n_experts=4)
        stats = stat_collector.get_all_stats()
        assert "layers.0.gate" in stats["routing"]

    def test_register_multiple_layers(self, stat_collector: StatCollector) -> None:
        stat_collector.register_layer("layers.0.gate", n_experts=4)
        stat_collector.register_layer("layers.1.gate", n_experts=8)
        stats = stat_collector.get_all_stats()
        assert "layers.0.gate" in stats["routing"]
        assert "layers.1.gate" in stats["routing"]

    def test_double_register_same_layer_no_crash(
        self, stat_collector: StatCollector
    ) -> None:
        stat_collector.register_layer("layers.0.gate", n_experts=4)
        stat_collector.register_layer("layers.0.gate", n_experts=4)  # idempotent
        stats = stat_collector.get_all_stats()
        assert "layers.0.gate" in stats["routing"]


class TestStatCollectorWriteRoutingEvent:
    """write_routing_event() dispatches to the correct buffer."""

    def test_write_routing_event_basic(self, stat_collector: StatCollector) -> None:
        stat_collector.register_layer("layers.0.gate", n_experts=4)
        event = make_routing_event(layer_name="layers.0.gate", n_experts=4)
        stat_collector.write_routing_event(event)
        stats = stat_collector.get_all_stats()
        layer_stats = stats["routing"]["layers.0.gate"]
        assert layer_stats.step >= 0

    def test_write_routing_event_increments_layer_stats(
        self, stat_collector: StatCollector
    ) -> None:
        stat_collector.register_layer("layers.0.gate", n_experts=4)
        for step in range(5):
            stat_collector.write_routing_event(
                make_routing_event(
                    layer_name="layers.0.gate",
                    n_experts=4,
                    global_step=step,
                )
            )
        stats = stat_collector.get_all_stats()
        assert stats["routing"]["layers.0.gate"].step == 4

    def test_write_routing_event_multiple_layers_independent(
        self, stat_collector: StatCollector
    ) -> None:
        stat_collector.register_layer("layers.0.gate", n_experts=4)
        stat_collector.register_layer("layers.1.gate", n_experts=4)

        stat_collector.write_routing_event(
            make_routing_event(layer_name="layers.0.gate", global_step=10)
        )
        stat_collector.write_routing_event(
            make_routing_event(layer_name="layers.1.gate", global_step=20)
        )

        stats = stat_collector.get_all_stats()
        assert stats["routing"]["layers.0.gate"].step == 10
        assert stats["routing"]["layers.1.gate"].step == 20


class TestStatCollectorLayerStats:
    """LayerStats field correctness."""

    def test_expert_token_counts_shape(self, stat_collector: StatCollector) -> None:
        n_experts = 4
        stat_collector.register_layer("layers.0.gate", n_experts)
        stat_collector.write_routing_event(
            make_routing_event(layer_name="layers.0.gate", n_experts=n_experts, batch_size=8)
        )
        layer_stats = stat_collector.get_layer_stats("layers.0.gate")
        assert layer_stats.expert_token_counts.shape == (n_experts,), (
            f"Expected shape ({n_experts},), got {layer_stats.expert_token_counts.shape}"
        )

    def test_expert_utilization_sums_to_one(self, stat_collector: StatCollector) -> None:
        n_experts = 4
        stat_collector.register_layer("layers.0.gate", n_experts)
        for step in range(10):
            stat_collector.write_routing_event(
                make_routing_event(layer_name="layers.0.gate", n_experts=n_experts, global_step=step)
            )
        layer_stats = stat_collector.get_layer_stats("layers.0.gate")
        total = layer_stats.expert_utilization.sum().item()
        assert abs(total - 1.0) < 1e-4, (
            f"expert_utilization should sum to 1.0, got {total}"
        )

    def test_uniform_routing_gives_balanced_load_imbalance(
        self, stat_collector: StatCollector
    ) -> None:
        """Uniform logits → all experts selected equally → load_imbalance_ratio ≈ 1.0."""
        n_experts = 4
        stat_collector.register_layer("layers.0.gate", n_experts)
        for step in range(20):
            stat_collector.write_routing_event(
                make_routing_event(
                    layer_name="layers.0.gate",
                    n_experts=n_experts,
                    batch_size=32,
                    global_step=step,
                    uniform=True,
                )
            )
        layer_stats = stat_collector.get_layer_stats("layers.0.gate")
        ratio = layer_stats.load_imbalance_ratio
        # For uniform routing the ratio should be close to 1.0
        assert ratio >= 1.0, f"load_imbalance_ratio must be >= 1.0, got {ratio}"
        assert ratio < 3.0, (
            f"For uniform routing load_imbalance_ratio should be close to 1.0, got {ratio}"
        )

    def test_collapsed_routing_gives_high_load_imbalance(
        self, stat_collector: StatCollector
    ) -> None:
        """Collapsed logits → expert 0 monopoly → load_imbalance_ratio >> 1."""
        n_experts = 4
        stat_collector.register_layer("layers.0.gate", n_experts)
        for step in range(10):
            stat_collector.write_routing_event(
                make_routing_event(
                    layer_name="layers.0.gate",
                    n_experts=n_experts,
                    batch_size=16,
                    global_step=step,
                    uniform=False,  # collapsed: expert 0 gets 20x bias
                )
            )
        layer_stats = stat_collector.get_layer_stats("layers.0.gate")
        assert layer_stats.load_imbalance_ratio > 1.5, (
            f"Collapsed routing should yield load_imbalance_ratio > 1.5, "
            f"got {layer_stats.load_imbalance_ratio}"
        )

    def test_raw_logits_window_shape(self, stat_collector: StatCollector) -> None:
        n_experts = 4
        stat_collector.register_layer("layers.0.gate", n_experts)
        for step in range(5):
            stat_collector.write_routing_event(
                make_routing_event(
                    layer_name="layers.0.gate",
                    n_experts=n_experts,
                    batch_size=8,
                    global_step=step,
                )
            )
        layer_stats = stat_collector.get_layer_stats("layers.0.gate")
        assert layer_stats.raw_logits_window is not None
        # Last dim must be n_experts
        assert layer_stats.raw_logits_window.shape[-1] == n_experts

    def test_layer_name_in_stats(self, stat_collector: StatCollector) -> None:
        stat_collector.register_layer("layers.0.gate", 4)
        stat_collector.write_routing_event(
            make_routing_event(layer_name="layers.0.gate", n_experts=4)
        )
        ls = stat_collector.get_layer_stats("layers.0.gate")
        assert ls.layer_name == "layers.0.gate"

    def test_get_layer_stats_raises_on_unknown_layer(
        self, stat_collector: StatCollector
    ) -> None:
        with pytest.raises(KeyError):
            stat_collector.get_layer_stats("this.does.not.exist")


class TestStatCollectorWriteGradientEvent:
    """write_gradient_event() dispatches correctly; GradientStats computed."""

    def test_write_gradient_event_basic(self, stat_collector: StatCollector) -> None:
        stat_collector.register_layer("layers.0.gate", n_experts=4)
        event = make_gradient_event(
            layer_name="layers.0.experts",
            expert_id=0,
            gradient_norm=0.1,
        )
        stat_collector.write_gradient_event(event)
        all_stats = stat_collector.get_all_stats()
        assert "layers.0.experts" in all_stats["gradient"]

    def test_gradient_stats_expert_id_indexed(
        self, stat_collector: StatCollector
    ) -> None:
        stat_collector.register_layer("layers.0.gate", n_experts=4)
        for eid in range(4):
            stat_collector.write_gradient_event(
                make_gradient_event(
                    layer_name="layers.0.experts",
                    expert_id=eid,
                    gradient_norm=float(eid) * 0.1,
                )
            )
        all_stats = stat_collector.get_all_stats()
        grad_layer = all_stats["gradient"].get("layers.0.experts", {})
        assert len(grad_layer) == 4, (
            f"Expected 4 expert entries, got {len(grad_layer)}"
        )

    def test_gradient_norm_history_captured(
        self, collector_with_gradients: StatCollector
    ) -> None:
        all_stats = collector_with_gradients.get_all_stats()
        grad = all_stats.get("gradient", {})
        assert "layers.0.experts" in grad
        # Expert 3 should have some history
        expert3 = grad["layers.0.experts"].get(3)
        assert expert3 is not None
        assert len(expert3.gradient_norm_history) > 0


class TestStatCollectorGetAllStats:
    """get_all_stats() returns a snapshot with 'routing' and 'gradient' keys."""

    def test_get_all_stats_keys(self, stat_collector: StatCollector) -> None:
        stats = stat_collector.get_all_stats()
        assert "routing" in stats
        assert "gradient" in stats

    def test_populated_collector_has_correct_layers(
        self, populated_collector: StatCollector
    ) -> None:
        stats = populated_collector.get_all_stats()
        assert "layers.0.gate" in stats["routing"]
        assert "layers.1.gate" in stats["routing"]

    def test_get_all_stats_returns_snapshot_copy(
        self, populated_collector: StatCollector
    ) -> None:
        """Mutating the returned dict must not affect subsequent calls."""
        stats1 = populated_collector.get_all_stats()
        stats1["routing"].clear()
        stats2 = populated_collector.get_all_stats()
        assert len(stats2["routing"]) == 2, (
            "Mutating get_all_stats() result corrupted internal state"
        )


class TestStatCollectorThreadSafety:
    """Concurrent writes from multiple threads must not corrupt state."""

    def test_concurrent_routing_writes(self, default_config: WatchConfig) -> None:
        collector = StatCollector(default_config)
        collector.register_layer("layers.0.gate", n_experts=4)
        errors: List[Exception] = []

        def writer(thread_id: int) -> None:
            try:
                for i in range(20):
                    collector.write_routing_event(
                        make_routing_event(
                            layer_name="layers.0.gate",
                            n_experts=4,
                            global_step=thread_id * 100 + i,
                        )
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors during concurrent writes: {errors}"
        # No crash means success; just verify data is readable
        stats = collector.get_all_stats()
        assert "layers.0.gate" in stats["routing"]

    def test_concurrent_gradient_writes(self, default_config: WatchConfig) -> None:
        collector = StatCollector(default_config)
        collector.register_layer("layers.0.gate", n_experts=4)
        errors: List[Exception] = []

        def grad_writer(expert_id: int) -> None:
            try:
                for step in range(20):
                    collector.write_gradient_event(
                        make_gradient_event(
                            layer_name="layers.0.experts",
                            expert_id=expert_id,
                            gradient_norm=0.1,
                            global_step=step,
                        )
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=grad_writer, args=(e,)) for e in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors during concurrent gradient writes: {errors}"


class TestStatCollectorEdgeCases:
    """Edge cases and defensive behaviour."""

    def test_no_events_layer_stats_step_is_zero(
        self, stat_collector: StatCollector
    ) -> None:
        """A registered but empty layer should return a sane default LayerStats."""
        stat_collector.register_layer("layers.0.gate", n_experts=4)
        ls = stat_collector.get_layer_stats("layers.0.gate")
        assert ls.step == 0

    def test_expert_utilization_valid_with_no_events(
        self, stat_collector: StatCollector
    ) -> None:
        """With zero events, utilization should default to uniform (no crash)."""
        stat_collector.register_layer("layers.0.gate", n_experts=4)
        ls = stat_collector.get_layer_stats("layers.0.gate")
        total = ls.expert_utilization.sum().item()
        assert abs(total - 1.0) < 1e-4 or total == 0.0, (
            f"Unexpected utilization sum for empty layer: {total}"
        )

    def test_load_imbalance_is_float(self, stat_collector: StatCollector) -> None:
        stat_collector.register_layer("layers.0.gate", n_experts=4)
        stat_collector.write_routing_event(
            make_routing_event(layer_name="layers.0.gate", n_experts=4)
        )
        ls = stat_collector.get_layer_stats("layers.0.gate")
        assert isinstance(ls.load_imbalance_ratio, float)
