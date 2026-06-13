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
        Indices of the top-k selected experts derived from
        ``routing_logits`` via ``topk``; shape mirrors
        ``routing_logits`` with the last dimension reduced to ``k``.
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

    __slots__ = ("layer_name", "stat_collector", "config", "_global_step")

    def __init__(
        self,
        layer_name: str,
        stat_collector: "StatCollector",  # noqa: F821 - forward ref, avoids import cycle
        config: WatchConfig,
    ) -> None:
        self.layer_name: str = layer_name
        self.stat_collector = stat_collector
        self.config: WatchConfig = config

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

                top_k = self._infer_top_k(module, expert_count)
                selected_experts = self._select_top_k(logits_detached, top_k)

                batch_size = 1
                for dim_size in logits_detached.shape[:-1]:
                    batch_size *= int(dim_size)

                event = RoutingEvent(
                    timestamp=time.time(),
                    global_step=self._global_step,
                    layer_name=self.layer_name,
                    routing_logits=logits_detached,
                    selected_experts=selected_experts,
                    expert_count=int(expert_count),
                    batch_size=int(batch_size),
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
    def _infer_top_k(module: nn.Module, expert_count: int) -> int:
        """Infer the number of experts selected per token (``top_k``).

        Checks common attribute names used by popular MoE implementations
        before falling back to a conservative default.

        Parameters
        ----------
        module : torch.nn.Module
            The hooked router module, possibly exposing ``top_k`` or
            ``num_experts_per_tok``.
        expert_count : int
            Total number of experts available in this router, used to
            clamp the inferred ``top_k`` to a valid range.

        Returns
        -------
        int
            The inferred ``top_k`` value, clamped to ``[1, expert_count]``.
            Defaults to ``1`` (argmax routing) if no attribute is found.
        """
        for attr in ("top_k", "num_experts_per_tok", "k"):
            value = getattr(module, attr, None)
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
