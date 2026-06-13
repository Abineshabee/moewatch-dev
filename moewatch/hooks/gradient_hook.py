# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# moewatch/hooks/gradient_hook.py [v0.2.0]
# =============================================================================
#
# Project      : MoEWatch
# Version      : v0.2.0
# Description  : Backward hook attached to per-expert weight tensors.
#                 Captures the L2 norm of the gradient flowing into each
#                 expert's weight at every (sampled) backward pass. This
#                 is the primary data source for the Tier 1 "gradient
#                 starvation" signal — empirically the earliest available
#                 precursor to MoE routing collapse (50-200 steps before
#                 collapse becomes visible via utilization or entropy).
#
#                 Sampling is controlled by ``config.sample_every`` to keep
#                 backward-pass overhead negligible: the hook performs work
#                 only on steps where ``global_step % sample_every == 0``.
#
# Author       : Abinesh N (@Abineshabee)
# Repository   : https://github.com/Abineshabee/MoEWatch
# License      : Apache 2.0
#
# Contents
# --------
#   GradientEvent             — dataclass describing a single gradient
#                                norm observation for one expert
#   GradientStarvationHook    — backward-hook callable registered via
#                                ``register_hook`` on a parameter tensor,
#                                or via ``register_full_backward_hook`` on
#                                an expert submodule
#
# Usage
# -----
#   from moewatch.hooks.gradient_hook import GradientStarvationHook
#
#   hook = GradientStarvationHook(
#       layer_name="layers.5.experts",
#       expert_id=2,
#       stat_collector=stat_collector,
#       config=config,
#   )
#   handle = expert_weight_param.register_hook(hook)
#   ...
#   handle.remove()
#
# =============================================================================

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import torch

from moewatch.config import WatchConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GradientEvent
# ---------------------------------------------------------------------------


@dataclass
class GradientEvent:
    """A single observation of an expert's gradient L2 norm.

    Instances of this dataclass are produced by
    :class:`GradientStarvationHook` and consumed by
    :meth:`moewatch.collector.stat_collector.StatCollector.write_gradient_event`.

    Attributes
    ----------
    timestamp : float
        Unix timestamp (seconds since epoch) at which the event was
        captured, via ``time.time()``.
    global_step : int
        Training step at which this gradient was observed.
    layer_name : str
        Name of the expert layer this gradient belongs to (e.g.
        ``"model.layers.5.block_sparse_moe.experts"``).
    expert_id : int
        Index of the specific expert within ``layer_name``
        (``0`` to ``n_experts - 1``).
    gradient_norm : float
        L2 norm of the gradient tensor: ``sqrt(sum(grad ** 2))``.
    gradient_magnitude : float
        Alias for :attr:`gradient_norm`. Retained as a separate field for
        downstream consumers (e.g. JSON reporters, dashboards) that
        prefer an explicitly named "magnitude" key without needing to
        know it is identical to the norm.
    """

    timestamp: float
    global_step: int
    layer_name: str
    expert_id: int
    gradient_norm: float
    gradient_magnitude: float


# ---------------------------------------------------------------------------
# GradientStarvationHook
# ---------------------------------------------------------------------------


class GradientStarvationHook:
    """Backward-pass hook capturing per-expert gradient L2 norms.

    This callable is intended for registration via
    ``torch.Tensor.register_hook`` on an individual expert weight
    parameter (the recommended approach, since
    ``register_full_backward_hook`` on a module receives the gradient
    with respect to the module's *output*, not its weights). When
    registered this way, PyTorch invokes the hook with the gradient
    tensor for that parameter during ``backward()``.

    To minimize backward-pass overhead, the hook only performs norm
    computation and event emission every
    ``config.sample_every`` steps (controlled via :meth:`set_global_step`,
    which is called once per training step by
    :class:`~moewatch.hooks.manager.HookManager`).

    Per the gradient-hook contract in PyTorch, this callable **must**
    return either ``None`` (gradient unmodified) or a replacement
    gradient tensor of the same shape. This hook always returns ``None``,
    guaranteeing zero modification to the optimization process — it is
    purely an observability tap.

    Parameters
    ----------
    layer_name : str
        Name of the expert layer this hook monitors (used as the outer
        key for :class:`StatCollector`'s nested gradient buffers).
    expert_id : int
        Index of the specific expert within ``layer_name``.
    stat_collector : StatCollector
        Destination for emitted :class:`GradientEvent` objects.
    config : WatchConfig
        Shared configuration object; ``config.sample_every`` controls the
        sampling rate of this hook.

    Notes
    -----
    Execution time per invocation is designed to stay under ~0.5ms even
    for large expert weight tensors, since ``torch.norm`` on a
    contiguous tensor is a single fused reduction kernel.
    """

    __slots__ = (
        "layer_name",
        "expert_id",
        "stat_collector",
        "config",
        "_global_step",
    )

    def __init__(
        self,
        layer_name: str,
        expert_id: int,
        stat_collector: "StatCollector",  # noqa: F821 - forward ref, avoids import cycle
        config: WatchConfig,
    ) -> None:
        self.layer_name: str = layer_name
        self.expert_id: int = expert_id
        self.stat_collector = stat_collector
        self.config: WatchConfig = config

        # Updated externally (by HookManager) once per training step.
        self._global_step: int = 0

    # ------------------------------------------------------------------
    # Hook entry point
    # ------------------------------------------------------------------

    def __call__(self, grad: torch.Tensor) -> Optional[torch.Tensor]:
        """Backward-pass gradient hook callback.

        Parameters
        ----------
        grad : torch.Tensor
            The gradient tensor with respect to the parameter this hook
            is registered on (typically a 1D or 2D weight tensor, e.g.
            ``[in_features, out_features]`` for a linear expert layer).

        Returns
        -------
        torch.Tensor or None
            Always returns ``None``. Per the
            ``torch.Tensor.register_hook`` contract, returning ``None``
            leaves the gradient unmodified, preserving MoEWatch's
            zero-weight-modification / zero-gradient-modification
            guarantee.

        Notes
        -----
        - Sampling: if ``self._global_step % config.sample_every != 0``,
          this hook returns immediately without any tensor operations.
        - All processing is wrapped in ``torch.no_grad()`` and any
          exception is caught and logged at DEBUG level rather than
          propagated, since a failure here must never interrupt
          ``backward()``.
        """
        sample_every = max(1, self.config.sample_every)
        if self._global_step % sample_every != 0:
            return None

        try:
            if grad is None:
                return None

            with torch.no_grad():
                gradient_norm = float(torch.norm(grad.detach(), p=2).item())

            event = GradientEvent(
                timestamp=time.time(),
                global_step=self._global_step,
                layer_name=self.layer_name,
                expert_id=self.expert_id,
                gradient_norm=gradient_norm,
                gradient_magnitude=gradient_norm,
            )

            self.stat_collector.write_gradient_event(event)

        except Exception as exc:  # pylint: disable=broad-except
            logger.debug(
                "[MoEWatch] GradientStarvationHook('%s', expert=%d): "
                "unexpected error (event skipped): %s",
                self.layer_name,
                self.expert_id,
                exc,
            )

        # Always return None: never modify the gradient.
        return None

    # ------------------------------------------------------------------
    # Step counter management (driven by HookManager)
    # ------------------------------------------------------------------

    def set_global_step(self, global_step: int) -> None:
        """Update the training step number used for sampling and events.

        Called by :class:`~moewatch.hooks.manager.HookManager` before each
        training step so that this hook can decide whether to sample on
        the upcoming backward pass and so that emitted
        :class:`GradientEvent` objects carry an accurate ``global_step``.

        Parameters
        ----------
        global_step : int
            Current training step number.
        """
        self._global_step = global_step
