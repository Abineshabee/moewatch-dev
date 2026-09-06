# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# moewatch/hooks/router_hook.py
# =============================================================================
#
# Project      : MoEWatch
# Version      : v0.2.0
# Description  : Forward hook attached to MoE router modules. Captures the
#                 raw routing decision (logits + selected expert indices)
#                 produced at each forward pass and forwards it to the
#                 StatCollector as a RoutingEvent.
#
#                 This hook is designed to sit on the hot path of every
#                 forward pass, so it is intentionally minimal: no Python
#                 loops over tokens, no extra tensor copies beyond what is
#                 strictly necessary to detach from the autograd graph.
#
# Author       : Abinesh N (@Abineshabee)
# Repository   : https://github.com/Abineshabee/MoEWatch
# License      : Apache 2.0
#
# Contents
# --------
#   RoutingEvent       — dataclass describing a single routing observation
#   RouterForwardHook  — forward-hook callable registered on router modules
#
# Usage
# -----
#   from moewatch.hooks.router_hook import RouterForwardHook
#
#   hook = RouterForwardHook("layers.5.moe_router", stat_collector, config)
#   handle = router_module.register_forward_hook(hook)
#   ...
#   handle.remove()
#
# =============================================================================

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from moewatch.config import WatchConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RoutingEvent
# ---------------------------------------------------------------------------


@dataclass
class RoutingEvent:
    """A single observation of a router's forward-pass output.

    Instances of this dataclass are produced by :class:`RouterForwardHook`
    and consumed by
    :meth:`moewatch.collector.stat_collector.StatCollector.write_routing_event`.

    Attributes
    ----------
    timestamp : float
        Unix timestamp (seconds since epoch) at which the event was
        captured, via ``time.time()``.
    global_step : int
        Training step associated with this forward pass. Populated from
        the hook's last known step counter (updated externally by
        :class:`~moewatch.hooks.manager.HookManager` or the watcher).
    layer_name : str
        Fully-qualified name of the router module (e.g.
        ``"model.layers.5.block_sparse_moe.gate"``).
    routing_logits : torch.Tensor
        Raw router logits, detached from the autograd graph and moved to
        CPU is *not* performed here (kept on-device for downstream
        analysis); shape is typically ``[batch_size * seq_len, n_experts]``
        or ``[batch_size, seq_len, n_experts]``.
    selected_experts : torch.Tensor
        Expert selection data. Interpretation depends on ``is_expert_counts``:

        ``is_expert_counts=True`` (produced by RouterForwardHook v0.2+):
            Shape ``[n_experts]``, dtype ``int64``.
            Per-expert token counts — ``selected_experts[i]`` is the number
            of tokens routed to expert ``i`` in this forward pass.
            Memory: 64 B / event (Mixtral, n_experts=8).

        ``is_expert_counts=False`` (legacy / external callers):
            Shape ``[batch_size * seq_len * top_k]`` or ``[batch_size, top_k]``,
            dtype ``int64``.
            Flat expert index tensor produced by ``topk`` — each element is
            an expert index in ``[0, n_experts)``.
            Memory: up to 32 KB / event (Mixtral, seq=2048, top_k=2).
    is_expert_counts : bool
        Unambiguous format tag. ``True`` when ``selected_experts`` holds
        per-expert token counts; ``False`` (default) when it holds raw
        expert indices. Always set explicitly by ``RouterForwardHook``.
    expert_count : int
        Number of experts ``n_experts``, inferred from the last dimension
        of ``routing_logits``.
    batch_size : int
        Number of tokens/rows represented in this routing event (product
        of all leading dimensions of ``routing_logits``).
    """

    timestamp: float
    global_step: int
    layer_name: str
    routing_logits: torch.Tensor
    selected_experts: torch.Tensor
    expert_count: int
    batch_size: int
    is_expert_counts: bool = False
    # is_expert_counts=True  → selected_experts is shape [n_experts] per-expert token COUNTS
    # is_expert_counts=False → selected_experts is flat/2-D expert INDEX tensor (legacy format)


# ---------------------------------------------------------------------------
# RouterForwardHook
# ---------------------------------------------------------------------------


class RouterForwardHook:
    """Forward hook callable for MoE router modules.

    Registered via ``module.register_forward_hook(hook)``. On every
    forward pass, captures the router's output tensor, infers the number
    of experts and the top-k selected experts, and writes a
    :class:`RoutingEvent` to the configured :class:`StatCollector`.

    This hook is strictly read-only: it never modifies ``input`` or
    ``output``, and all tensor operations are wrapped in
    ``torch.no_grad()`` to avoid contributing to the autograd graph or
    retaining unnecessary memory.

    Parameters
    ----------
    layer_name : str
        Name of the router module this hook is attached to (used as the
        key for :class:`StatCollector` buffers).
    stat_collector : StatCollector
        Destination for emitted :class:`RoutingEvent` objects.
    config : WatchConfig
        Shared configuration object. Currently used for forward
        compatibility (e.g. future sampling controls).

    Notes
    -----
    Top-k selection defaults to ``k=1`` (argmax routing) when the router
    output shape does not unambiguously indicate ``k``. Most MoE routers
    (Mixtral, Qwen3-MoE, DeepSeek-MoE, OLMoE) emit a 2D logits tensor of
    shape ``[tokens, n_experts]``; ``k`` itself is a property of the
    *model*, not the router output, so this hook conservatively reports
    the top-1 expert unless overridden by examining ``module`` attributes
    (``top_k`` / ``num_experts_per_tok``) when present.
    """

    __slots__ = ("layer_name", "stat_collector", "config", "_global_step", "_model")

    def __init__(
        self,
        layer_name: str,
        stat_collector: "StatCollector",  # noqa: F821 - forward ref, avoids import cycle
        config: WatchConfig,
        model: "nn.Module | None" = None,
    ) -> None:
        self.layer_name: str = layer_name
        self.stat_collector = stat_collector
        self.config: WatchConfig = config
        self._model: "nn.Module | None" = model

        # Updated externally (by HookManager) before each forward pass so
        # that emitted events carry an accurate training step number.
        self._global_step: int = 0

    # ------------------------------------------------------------------
    # Hook entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        module: nn.Module,
        input: Tuple,  # noqa: A002 - matches torch hook signature
        output: object,
    ) -> None:
        """Forward hook callback.

        Parameters
        ----------
        module : torch.nn.Module
            The router module that produced ``output``. May expose
            ``top_k`` or ``num_experts_per_tok`` attributes which are
            used (if present) to determine how many experts are selected
            per token.
        input : tuple
            The module's forward-pass input arguments. Unused, present
            to satisfy the ``register_forward_hook`` signature.
        output : Any
            The module's forward-pass return value. May be a raw
            ``torch.Tensor`` of logits, or a tuple/namedtuple whose first
            element is the logits tensor (common in HuggingFace MoE
            implementations that return ``(logits, ...)`` or
            ``(hidden_states, router_logits)``).

        Returns
        -------
        None
            Forward hooks that return ``None`` do not modify the
            module's output. This hook never returns a replacement value.

        Notes
        -----
        Any exception raised while processing the event is caught and
        logged at DEBUG level rather than propagated, so a malformed or
        unexpected router output can never interrupt training.
        """
        try:
            logits = self._extract_logits(output)
            if logits is None:
                logger.debug(
                    "[MoEWatch] RouterForwardHook('%s'): could not extract "
                    "logits from output of type %s; skipping event.",
                    self.layer_name,
                    type(output).__name__,
                )
                return

            with torch.no_grad():
                if logits.ndim < 2:
                    logger.debug(
                        "[MoEWatch] RouterForwardHook('%s'): logits tensor "
                        "has unexpected ndim=%d (expected >= 2); skipping.",
                        self.layer_name,
                        logits.ndim,
                    )
                    return

                expert_count = logits.shape[-1]
                if expert_count < 1:
                    logger.debug(
                        "[MoEWatch] RouterForwardHook('%s'): expert_count "
                        "< 1; skipping event.",
                        self.layer_name,
                    )
                    return

                # Detach to avoid retaining the autograd graph through the
                # hook's reference. No device transfer (keep on-device for
                # downstream analyzer efficiency).
                logits_detached = logits.detach()

                top_k = self._infer_top_k(module, expert_count, self._model, self.layer_name)
                selected_experts = self._select_top_k(logits_detached, top_k)

                batch_size = 1
                for dim_size in logits_detached.shape[:-1]:
                    batch_size *= int(dim_size)

                # Memory fix: immediately reduce full [batch*seq, n_experts]
                # tensors to compact CPU representations before storing in
                # the ring buffer.
                #
                # Without this, each RoutingEvent holds:
                #   routing_logits:   [batch*seq, n_experts] float32
                #   selected_experts: [batch*seq, top_k]    int64
                # For Mixtral (batch=1, seq=2048, n_experts=8): ~96 KB/event.
                # With 1000-event buffers × 32 layers = ~3 GB — OOM risk.
                #
                # Fix: reduce to CPU immediately:
                #   routing_logits   → mean softmax prob (n_experts,) CPU  ~256 B
                #   selected_experts → flat index tensor CPU               ~seq*top_k B
                # stat_collector.get_all_stats() calls torch.bincount() on
                # selected_experts (needs index values, not pre-aggregated
                # counts), and _stack_logits_window() stacks routing_logits
                # (1-D mean-prob vectors work for entropy computation).
                flat = logits_detached.reshape(-1, expert_count).float()
                probs = torch.softmax(flat, dim=-1)
                mean_probs_cpu = probs.mean(dim=0).cpu()          # [n_experts] mean prob

                # Store per-expert token COUNTS instead of flat token indices.
                # Old: selected_experts_cpu = selected_experts.reshape(-1).cpu()
                #      shape [batch*seq*top_k] — up to 32 KB per event (Mixtral).
                # New: bincount at write time → shape [n_experts] — 64 B per event.
                # stat_collector.get_all_stats() now simply sums these count tensors
                # across the window instead of running bincount at read time.
                flat_idx = selected_experts.reshape(-1).to(dtype=torch.long)
                flat_idx = flat_idx.clamp(min=0, max=int(expert_count) - 1)
                expert_counts_cpu = torch.bincount(
                    flat_idx.cpu(), minlength=int(expert_count)
                ).to(dtype=torch.long)                            # [n_experts] counts

                event = RoutingEvent(
                    timestamp=time.time(),
                    global_step=self._global_step,
                    layer_name=self.layer_name,
                    routing_logits=mean_probs_cpu,          # [n_experts] mean prob, CPU
                    selected_experts=expert_counts_cpu,     # [n_experts] counts, CPU
                    expert_count=int(expert_count),
                    batch_size=int(batch_size),
                    is_expert_counts=True,                  # explicit format tag
                )

            self.stat_collector.write_routing_event(event)

        except Exception as exc:  # pylint: disable=broad-except
            logger.debug(
                "[MoEWatch] RouterForwardHook('%s'): unexpected error "
                "(event skipped): %s",
                self.layer_name,
                exc,
            )

    # ------------------------------------------------------------------
    # Step counter management (driven by HookManager)
    # ------------------------------------------------------------------

    def set_global_step(self, global_step: int) -> None:
        """Update the training step number reported in future events.

        Called by :class:`~moewatch.hooks.manager.HookManager` before each
        training step so that :class:`RoutingEvent` objects carry an
        accurate ``global_step``.

        Parameters
        ----------
        global_step : int
            Current training step number.
        """
        self._global_step = global_step

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_logits(output: object) -> "torch.Tensor | None":
        """Extract a router logits tensor from a module's forward output.

        Handles three common shapes of HuggingFace MoE router outputs:

        1. A bare ``torch.Tensor`` — returned as-is.
        2. A ``tuple``/``list`` whose first element is a ``torch.Tensor``
           — that element is returned (covers ``(logits, indices)`` or
           ``(hidden_states, router_logits)`` style returns; the *first*
           tensor is assumed to be logits-shaped).
        3. Anything else (e.g. ``dict``, custom objects) — attempts to
           read a ``.logits`` or ``.router_logits`` attribute/key.

        Parameters
        ----------
        output : object
            Raw forward-pass return value of the hooked module.

        Returns
        -------
        torch.Tensor or None
            The extracted logits tensor, or ``None`` if no tensor could
            be located.
        """
        if isinstance(output, torch.Tensor):
            return output

        if isinstance(output, (tuple, list)):
            for item in output:
                if isinstance(item, torch.Tensor):
                    return item
            return None

        if isinstance(output, dict):
            for key in ("router_logits", "logits", "routing_logits"):
                value = output.get(key)
                if isinstance(value, torch.Tensor):
                    return value
            return None

        for attr in ("router_logits", "logits", "routing_logits"):
            value = getattr(output, attr, None)
            if isinstance(value, torch.Tensor):
                return value

        return None

    @staticmethod
    def _infer_top_k(
        module: nn.Module,
        expert_count: int,
        model: "nn.Module | None" = None,
        layer_name: str = "",
    ) -> int:
        """Infer the number of experts selected per token (``top_k``).

        Checks common attribute names on the gate module first, then walks
        to the parent MoE block (or the root model itself, when the gate
        has no dotted parent path), before falling back to a conservative
        default of ``k=1``.

        Without the parent-walk, a bare ``nn.Linear`` gate (which carries
        no ``top_k`` attribute) always produces ``k=1``, making the hook
        report top-1 routing on a top-8 model — load-imbalance stays near
        1.0 regardless of how severe the collapse is.

        Parameters
        ----------
        module : torch.nn.Module
            The hooked router module, possibly exposing ``top_k`` or
            ``num_experts_per_tok``.
        expert_count : int
            Total number of experts, used to clamp the result.
        model : nn.Module or None, optional
            Root model reference for parent-module resolution.
        layer_name : str, optional
            Fully-qualified name of the gate module (e.g.
            ``"layers.5.mlp.gate"``). Used to derive the parent name
            ``"layers.5.mlp"`` for the attribute walk. A name with no
            dot (e.g. ``"gate"``) means the gate is a direct top-level
            attribute of the root model — the root model itself is then
            treated as the parent to check.

        Returns
        -------
        int
            The inferred ``top_k`` value clamped to ``[1, expert_count]``.
        """
        _ATTRS = ("top_k", "num_experts_per_tok", "k")

        # 1. Check the gate module itself.
        for attr in _ATTRS:
            value = getattr(module, attr, None)
            if isinstance(value, int) and value > 0:
                return min(value, expert_count)

        # 2. Walk to parent MoE block and check there.
        if model is not None and layer_name:
            if "." in layer_name:
                parent_name = layer_name.rsplit(".", 1)[0]
                try:
                    parent = model.get_submodule(parent_name)
                except AttributeError:
                    parent = None
            else:
                # No dot in layer_name: the gate is a direct top-level
                # attribute of the root model (e.g. `self.gate = ...`
                # with no nesting under a submodule) — a common shape
                # for small/custom MoE models, including this library's
                # own example models. Architecturally the root model
                # itself IS the parent in this case. Without this
                # branch, such models always fell through to the k=1
                # default below regardless of their real top_k, silently
                # reporting top-1 routing (and therefore near-1.0
                # utilization for whichever expert wins that single
                # slot) on a model that actually routes to k>1 experts
                # per token.
                parent = model

            if parent is not None:
                for attr in _ATTRS:
                    value = getattr(parent, attr, None)
                    if isinstance(value, int) and value > 0:
                        return min(value, expert_count)

        return min(1, expert_count)

    @staticmethod
    def _select_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Select the indices of the top-``k`` experts per row.

        Parameters
        ----------
        logits : torch.Tensor
            Detached router logits of shape ``[..., n_experts]``.
        top_k : int
            Number of experts to select per row.

        Returns
        -------
        torch.Tensor
            Integer tensor of shape ``[..., top_k]`` containing the
            indices of the highest-scoring experts along the last
            dimension.
        """
        top_k = max(1, min(top_k, logits.shape[-1]))
        _, indices = torch.topk(logits, k=top_k, dim=-1)
        return indices
