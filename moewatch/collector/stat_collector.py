# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# moewatch/collector/stat_collector.py
# =============================================================================
#
# Project      : MoEWatch
# Version      : v0.2.0
# Description  : Central event aggregator for MoEWatch. Receives
#                 RoutingEvent objects (from RouterForwardHook) and
#                 GradientEvent objects (from GradientStarvationHook),
#                 stores them in per-layer / per-expert RingBuffer
#                 instances, and computes derived statistics on demand:
#
#                   LayerStats     — expert token counts, utilization,
#                                     load imbalance ratio, recent routing
#                                     logits window (Tier 2 / collapse
#                                     analyzer input)
#
#                   GradientStats  — per-expert gradient norm history,
#                                     rolling mean and standard deviation
#                                     (Tier 1 analyzer input) [v0.2.0]
#
#                 All public methods are thread-safe: writes may arrive
#                 from forward/backward hook callbacks while reads occur
#                 from the main monitoring loop.
#
# Author       : Abinesh N (@Abineshabee)
# Repository   : https://github.com/Abineshabee/MoEWatch
# License      : Apache 2.0
#
# Contents
# --------
#   LayerStats      — dataclass: per-layer routing statistics snapshot
#   GradientStats   — dataclass: per-expert gradient statistics snapshot [v0.2.0]
#   StatCollector   — central event aggregator
#
# Usage
# -----
#   from moewatch.collector.stat_collector import StatCollector
#
#   collector = StatCollector(config)
#   collector.register_layer("layers.5.moe_router", n_experts=8)
#   collector.write_routing_event(event)
#   collector.write_gradient_event(grad_event)
#
#   stats = collector.get_all_stats()
#   layer_stats = collector.get_layer_stats("layers.5.moe_router")
#
# =============================================================================

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from moewatch.collector.ring_buffer import RingBuffer
from moewatch.config import WatchConfig
from moewatch.hooks.gradient_hook import GradientEvent
from moewatch.hooks.router_hook import RoutingEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default RingBuffer capacity per routing/gradient buffer. Bounds memory
# usage independently of training run length.
_DEFAULT_BUFFER_CAPACITY: int = 1000

# Default window size (number of most-recent events) used when computing
# on-demand LayerStats / GradientStats snapshots.
_DEFAULT_STATS_WINDOW: int = 100


# ---------------------------------------------------------------------------
# LayerStats
# ---------------------------------------------------------------------------


@dataclass
class LayerStats:
    """Snapshot of per-layer routing statistics derived from recent events.

    Computed on demand by :class:`StatCollector` from the most recent
    window of :class:`RoutingEvent` objects for a given layer. Consumed
    by :class:`~moewatch.analyzer.entropy.EntropyAnalyzer` and
    :class:`~moewatch.analyzer.collapse.CollapseDetector`.

    Attributes
    ----------
    layer_name : str
        Router module name this snapshot describes.
    expert_token_counts : torch.Tensor
        1D integer tensor of shape ``[n_experts]``. Element ``i`` is the
        total number of tokens routed to expert ``i`` across all
        :class:`RoutingEvent` objects in the considered window.
    expert_utilization : torch.Tensor
        1D float tensor of shape ``[n_experts]``. Element ``i`` is
        ``expert_token_counts[i] / sum(expert_token_counts)``. If the
        total token count is ``0`` (no events yet), this is a uniform
        distribution ``1 / n_experts`` for every expert.
    load_imbalance_ratio : float
        ``max(expert_utilization) / mean(expert_utilization)``. A value
        of ``1.0`` indicates perfectly balanced routing; larger values
        indicate increasing skew toward a subset of experts. If
        ``mean(expert_utilization) == 0`` (no tokens routed at all),
        this is ``1.0`` (treated as balanced/no-signal).
    raw_logits_window : torch.Tensor
        Stacked routing logits from the considered window, shape
        ``[window_size, ..., n_experts]`` where the middle dimensions
        mirror whatever shape individual :class:`RoutingEvent.routing_logits`
        tensors had (e.g. ``[batch_size, n_experts]`` per event becomes
        ``[window_size, batch_size, n_experts]``). If events in the
        window have inconsistent shapes (e.g. varying batch sizes), only
        the most recent event's logits are returned with a leading
        window dimension of ``1`` — see :meth:`StatCollector.get_layer_stats`
        for the exact fallback behavior.
    step : int
        ``global_step`` of the most recent :class:`RoutingEvent`
        considered in this snapshot. ``0`` if no events have been
        recorded yet.
    """

    layer_name: str
    expert_token_counts: torch.Tensor
    expert_utilization: torch.Tensor
    load_imbalance_ratio: float
    raw_logits_window: torch.Tensor
    step: int


# ---------------------------------------------------------------------------
# GradientStats  [v0.2.0]
# ---------------------------------------------------------------------------


@dataclass
class GradientStats:
    """Snapshot of per-expert gradient statistics derived from recent events.

    Computed on demand by :class:`StatCollector` from the most recent
    window of :class:`GradientEvent` objects for a given
    ``(layer_name, expert_id)`` pair. Consumed by
    :class:`~moewatch.analyzer.gradient_starvation.GradientStarvationAnalyzer`
    (Tier 1 signal).

    Attributes
    ----------
    layer_name : str
        Expert layer name this snapshot describes (e.g.
        ``"layers.5.block_sparse_moe.experts"``).
    expert_id : int
        Index of the expert within ``layer_name``.
    gradient_norm_history : list[float]
        Gradient L2 norm values from the considered window, in
        chronological order (oldest first, newest last). Empty if no
        :class:`GradientEvent` objects have been recorded for this
        expert yet.
    gradient_norm_mean : float
        Arithmetic mean of :attr:`gradient_norm_history`. ``0.0`` if the
        history is empty.
    gradient_norm_rolling_std : float
        Population standard deviation of :attr:`gradient_norm_history`.
        ``0.0`` if the history has fewer than 2 elements.
    step : int
        ``global_step`` of the most recent :class:`GradientEvent`
        considered in this snapshot. ``0`` if no events have been
        recorded yet.
    """

    layer_name: str
    expert_id: int
    gradient_norm_history: List[float] = field(default_factory=list)
    gradient_norm_mean: float = 0.0
    gradient_norm_rolling_std: float = 0.0
    step: int = 0


# ---------------------------------------------------------------------------
# StatCollector
# ---------------------------------------------------------------------------


class StatCollector:
    """Central thread-safe aggregator for routing and gradient events.

    Owns one :class:`RingBuffer` of :class:`RoutingEvent` per registered
    router layer, and one :class:`RingBuffer` of :class:`GradientEvent`
    per ``(layer_name, expert_id)`` pair. Provides on-demand computation
    of :class:`LayerStats` and :class:`GradientStats` snapshots from the
    most recent window of buffered events.

    All write methods (:meth:`write_routing_event`,
    :meth:`write_gradient_event`) and read methods
    (:meth:`get_all_stats`, :meth:`get_layer_stats`) are safe to call
    concurrently — buffer-level locking is provided by
    :class:`~moewatch.collector.ring_buffer.RingBuffer`, and a
    collector-level lock guards the registration dictionaries themselves
    (so a layer can be registered concurrently with reads/writes without
    raising ``KeyError``/``RuntimeError`` from dict mutation during
    iteration).

    Parameters
    ----------
    config : WatchConfig
        Shared configuration object. Currently informs the statistics
        window size indirectly via ``config.sample_every`` /
        ``config.log_every`` is *not* used directly here; a fixed
        internal default window (:data:`_DEFAULT_STATS_WINDOW`) is used
        for on-demand statistics, independent of buffer capacity
        (:data:`_DEFAULT_BUFFER_CAPACITY`).

    Attributes
    ----------
    config : WatchConfig
        Reference to the shared configuration.
    """

    def __init__(self, config: WatchConfig) -> None:
        self.config: WatchConfig = config

        # layer_name -> RingBuffer[RoutingEvent]
        self._routing_buffers: Dict[str, RingBuffer] = {}

        # layer_name -> n_experts (as registered)
        self._expert_counts: Dict[str, int] = {}

        # layer_name -> { expert_id -> RingBuffer[GradientEvent] }
        self._gradient_buffers: Dict[str, Dict[int, RingBuffer]] = {}

        self._lock = threading.Lock()
        self._last_step: int = 0

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_layer(self, layer_name: str, n_experts: int) -> None:
        """Create routing and gradient buffers for a newly detected layer.

        Idempotent: if ``layer_name`` is already registered, this method
        only updates :attr:`_expert_counts` if ``n_experts`` differs (a
        layer may be re-registered with a more accurate expert count
        once the first real :class:`RoutingEvent` arrives), and
        lazily creates any missing per-expert gradient buffers — existing
        buffers and their contents are preserved.

        Parameters
        ----------
        layer_name : str
            Fully-qualified router module name.
        n_experts : int
            Number of experts in this layer. Must be a positive integer;
            values ``< 1`` are clamped to ``1`` (a degenerate but
            non-crashing configuration).

        Returns
        -------
        None

        Notes
        -----
        Thread-safe.
        """
        n_experts = max(1, int(n_experts))

        with self._lock:
            if layer_name not in self._routing_buffers:
                self._routing_buffers[layer_name] = RingBuffer(
                    capacity=_DEFAULT_BUFFER_CAPACITY
                )

            self._expert_counts[layer_name] = n_experts

            if layer_name not in self._gradient_buffers:
                self._gradient_buffers[layer_name] = {}

            expert_buffers = self._gradient_buffers[layer_name]
            for expert_id in range(n_experts):
                if expert_id not in expert_buffers:
                    expert_buffers[expert_id] = RingBuffer(
                        capacity=_DEFAULT_BUFFER_CAPACITY
                    )

        logger.debug(
            "[MoEWatch] StatCollector: registered layer '%s' with %d "
            "expert(s).",
            layer_name,
            n_experts,
        )

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def write_routing_event(self, event: RoutingEvent) -> None:
        """Append a routing event to its layer's buffer.

        Parameters
        ----------
        event : RoutingEvent
            Routing observation produced by
            :class:`~moewatch.hooks.router_hook.RouterForwardHook`.

        Returns
        -------
        None

        Notes
        -----
        If ``event.layer_name`` has not been registered via
        :meth:`register_layer`, it is auto-registered here using
        ``event.expert_count`` — this guards against ordering issues
        where a forward pass fires before
        :class:`~moewatch.hooks.manager.HookManager` completes
        registration (defensive; normal operation always registers
        layers during ``attach()`` before any forward pass occurs).

        Also updates :attr:`_last_step` to ``event.global_step`` if it is
        the largest step number seen so far. Thread-safe.
        """
        if event.layer_name not in self._routing_buffers:
            self.register_layer(event.layer_name, event.expert_count)

        self._routing_buffers[event.layer_name].write(event)

        with self._lock:
            if event.global_step > self._last_step:
                self._last_step = event.global_step

    def write_gradient_event(self, event: GradientEvent) -> None:
        """Append a gradient event to its ``(layer, expert)`` buffer.

        Parameters
        ----------
        event : GradientEvent
            Gradient norm observation produced by
            :class:`~moewatch.hooks.gradient_hook.GradientStarvationHook`.

        Returns
        -------
        None

        Notes
        -----
        If ``event.layer_name`` / ``event.expert_id`` has not been
        registered, the necessary nested buffer is created on demand
        (defensive auto-registration, mirroring
        :meth:`write_routing_event`). The layer's registered expert count
        is widened if ``event.expert_id`` falls outside the currently
        known range.

        Also updates :attr:`_last_step` to ``event.global_step`` if it is
        the largest step number seen so far. Thread-safe.
        """
        with self._lock:
            layer_buffers = self._gradient_buffers.setdefault(
                event.layer_name, {}
            )

            if event.expert_id not in layer_buffers:
                layer_buffers[event.expert_id] = RingBuffer(
                    capacity=_DEFAULT_BUFFER_CAPACITY
                )

            known_experts = self._expert_counts.get(event.layer_name, 0)
            if event.expert_id + 1 > known_experts:
                self._expert_counts[event.layer_name] = event.expert_id + 1

            buffer = layer_buffers[event.expert_id]

            if event.global_step > self._last_step:
                self._last_step = event.global_step

        buffer.write(event)

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def get_all_stats(
        self,
        window: Optional[int] = None,
    ) -> Dict[str, Dict[str, object]]:
        """Return a snapshot of all routing and gradient statistics.

        Parameters
        ----------
        window : int, optional
            Number of most-recent events to consider per buffer when
            computing statistics. Defaults to
            :data:`_DEFAULT_STATS_WINDOW` (``100``) if not provided.

        Returns
        -------
        dict[str, dict]
            A dictionary with exactly two top-level keys:

            ``"routing"``
                ``dict[str, LayerStats]`` — one entry per registered
                router layer.

            ``"gradient"``
                ``dict[str, dict[int, GradientStats]]`` — one outer
                entry per registered layer, with one inner entry per
                registered expert index.

            Layers / experts with no events yet still appear in the
            output with zero-valued / empty statistics (see
            :meth:`get_layer_stats` and :meth:`_compute_gradient_stats`
            for the exact defaults).

        Notes
        -----
        Thread-safe: the set of registered layer/expert names is copied
        under lock before any per-buffer computation begins, so this
        method never raises due to concurrent registration during
        iteration. Returned :class:`LayerStats` / :class:`GradientStats`
        objects are freshly computed (not references into internal
        state) — callers may freely retain or mutate them.
        """
        effective_window = window if window is not None else _DEFAULT_STATS_WINDOW

        with self._lock:
            layer_names = list(self._routing_buffers.keys())
            gradient_layer_experts: Dict[str, List[int]] = {
                layer_name: sorted(expert_buffers.keys())
                for layer_name, expert_buffers in self._gradient_buffers.items()
            }

        routing_stats: Dict[str, LayerStats] = {}
        for layer_name in layer_names:
            routing_stats[layer_name] = self.get_layer_stats(
                layer_name, window=effective_window
            )

        gradient_stats: Dict[str, Dict[int, GradientStats]] = {}
        for layer_name, expert_ids in gradient_layer_experts.items():
            gradient_stats[layer_name] = {}
            for expert_id in expert_ids:
                gradient_stats[layer_name][expert_id] = self._compute_gradient_stats(
                    layer_name, expert_id, window=effective_window
                )

        return {
            "routing": routing_stats,
            "gradient": gradient_stats,
        }

    def get_layer_stats(
        self,
        layer_name: str,
        window: Optional[int] = None,
    ) -> LayerStats:
        """Compute a :class:`LayerStats` snapshot for one layer.

        Parameters
        ----------
        layer_name : str
            Router module name. Must have been registered via
            :meth:`register_layer` (directly or via auto-registration in
            :meth:`write_routing_event`).
        window : int, optional
            Number of most-recent :class:`RoutingEvent` objects to
            consider. Defaults to :data:`_DEFAULT_STATS_WINDOW` (``100``)
            if not provided.

        Returns
        -------
        LayerStats
            Computed statistics for ``layer_name``. If no events have
            been recorded yet, returns a snapshot with
            ``expert_token_counts`` and ``expert_utilization`` as
            zero/uniform tensors of the registered expert count,
            ``load_imbalance_ratio == 1.0``, an empty
            ``raw_logits_window`` (shape ``[0]``), and ``step == 0``.

        Raises
        ------
        KeyError
            If ``layer_name`` has never been registered.

        Notes
        -----
        Aggregation algorithm:

        1. Read the most recent ``window`` :class:`RoutingEvent` objects
           from the layer's :class:`RingBuffer`.
        2. For each event, accumulate per-expert token counts: for every
           row of ``event.selected_experts`` (shape
           ``[batch_size, top_k]``), increment the count for each
           selected expert index by ``1``.
        3. ``expert_utilization = expert_token_counts / total_tokens``
           (uniform ``1 / n_experts`` if ``total_tokens == 0``).
        4. ``load_imbalance_ratio = max(utilization) / mean(utilization)``
           (``1.0`` if ``mean(utilization) == 0``).
        5. ``raw_logits_window`` is built by stacking
           ``event.routing_logits`` along a new leading dimension when
           all events in the window share the same logits shape;
           otherwise only the most recent event's logits are used with a
           singleton leading dimension.
        """
        if layer_name not in self._routing_buffers:
            raise KeyError(
                f"[MoEWatch] StatCollector.get_layer_stats(): layer "
                f"'{layer_name}' is not registered."
            )

        effective_window = window if window is not None else _DEFAULT_STATS_WINDOW
        events: List[RoutingEvent] = self._routing_buffers[layer_name].read_window(
            effective_window
        )

        n_experts = self._expert_counts.get(layer_name, 1)

        if not events:
            uniform = torch.full(
                (n_experts,), 1.0 / n_experts, dtype=torch.float32
            )
            return LayerStats(
                layer_name=layer_name,
                expert_token_counts=torch.zeros(n_experts, dtype=torch.long),
                expert_utilization=uniform,
                load_imbalance_ratio=1.0,
                raw_logits_window=torch.empty(0),
                step=0,
            )

        token_counts = torch.zeros(n_experts, dtype=torch.long)
        for event in events:
            selected = event.selected_experts
            if selected.numel() == 0:
                continue
            flat = selected.reshape(-1).to(dtype=torch.long)
            # Clamp indices defensively in case expert_count grew between
            # events (e.g. due to a shape change mid-run).
            flat = flat.clamp(min=0, max=n_experts - 1)
            counts = torch.bincount(flat, minlength=n_experts)
            if counts.shape[0] > n_experts:
                counts = counts[:n_experts]
            token_counts += counts.to(dtype=torch.long)

        total_tokens = int(token_counts.sum().item())
        if total_tokens > 0:
            expert_utilization = token_counts.to(dtype=torch.float32) / total_tokens
        else:
            expert_utilization = torch.full(
                (n_experts,), 1.0 / n_experts, dtype=torch.float32
            )

        mean_util = float(expert_utilization.mean().item())
        if mean_util > 0.0:
            max_util = float(expert_utilization.max().item())
            load_imbalance_ratio = max_util / mean_util
        else:
            load_imbalance_ratio = 1.0

        raw_logits_window = self._stack_logits_window(events)

        return LayerStats(
            layer_name=layer_name,
            expert_token_counts=token_counts,
            expert_utilization=expert_utilization,
            load_imbalance_ratio=load_imbalance_ratio,
            raw_logits_window=raw_logits_window,
            step=events[-1].global_step,
        )

    # ------------------------------------------------------------------
    # Internal: gradient statistics
    # ------------------------------------------------------------------

    def _compute_gradient_stats(
        self,
        layer_name: str,
        expert_id: int,
        window: int,
    ) -> GradientStats:
        """Compute a :class:`GradientStats` snapshot for one expert.

        Parameters
        ----------
        layer_name : str
            Expert layer name.
        expert_id : int
            Expert index within ``layer_name``.
        window : int
            Number of most-recent :class:`GradientEvent` objects to
            consider.

        Returns
        -------
        GradientStats
            Computed statistics. If no :class:`GradientEvent` objects
            have been recorded for this ``(layer_name, expert_id)`` pair,
            returns a snapshot with an empty
            ``gradient_norm_history``, ``gradient_norm_mean == 0.0``,
            ``gradient_norm_rolling_std == 0.0``, and ``step == 0``.

        Notes
        -----
        Standard deviation is computed as the *population* standard
        deviation (denominator ``N``, not ``N - 1``) for consistency
        across very small windows where a sample-corrected estimate would
        be unstable; with ``N < 2`` the result is ``0.0``.
        """
        buffer = self._gradient_buffers.get(layer_name, {}).get(expert_id)
        if buffer is None:
            return GradientStats(layer_name=layer_name, expert_id=expert_id)

        events: List[GradientEvent] = buffer.read_window(window)
        if not events:
            return GradientStats(layer_name=layer_name, expert_id=expert_id)

        history = [e.gradient_norm for e in events]
        n = len(history)
        mean = sum(history) / n

        if n >= 2:
            variance = sum((x - mean) ** 2 for x in history) / n
            std = variance**0.5
        else:
            std = 0.0

        return GradientStats(
            layer_name=layer_name,
            expert_id=expert_id,
            gradient_norm_history=history,
            gradient_norm_mean=mean,
            gradient_norm_rolling_std=std,
            step=events[-1].global_step,
        )

    # ------------------------------------------------------------------
    # Internal: logits window assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _stack_logits_window(events: List[RoutingEvent]) -> torch.Tensor:
        """Stack routing logits from a window of events into one tensor.

        Parameters
        ----------
        events : list[RoutingEvent]
            Non-empty list of routing events, ordered oldest-to-newest.

        Returns
        -------
        torch.Tensor
            If every event's ``routing_logits`` tensor has the same
            shape, returns a tensor of shape
            ``[len(events), *routing_logits.shape]`` produced via
            ``torch.stack``. If shapes differ (e.g. variable batch size
            across forward passes), returns the most recent event's
            ``routing_logits`` with an added leading dimension of size
            ``1`` (shape ``[1, *routing_logits.shape]``), so downstream
            consumers can always rely on a leading "window" dimension
            being present.
        """
        first_shape = events[0].routing_logits.shape
        if all(e.routing_logits.shape == first_shape for e in events):
            return torch.stack([e.routing_logits for e in events], dim=0)

        latest = events[-1].routing_logits
        return latest.unsqueeze(0)

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    @property
    def last_step(self) -> int:
        """The largest ``global_step`` seen across all written events.

        Returns
        -------
        int
            ``0`` if no events have been written yet.
        """
        with self._lock:
            return self._last_step

    def __repr__(self) -> str:
        with self._lock:
            n_layers = len(self._routing_buffers)
            n_experts_total = sum(
                len(experts) for experts in self._gradient_buffers.values()
            )
        return (
            f"StatCollector(layers={n_layers}, "
            f"gradient_buffers={n_experts_total}, "
            f"last_step={self.last_step})"
        )
