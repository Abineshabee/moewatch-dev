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
#   GradientStarvationHook    — per-parameter backward hook (legacy /
#                                test-facing API); registered via
#                                ``Tensor.register_hook`` on a single
#                                expert weight parameter; 1 hook per expert
#   MoEBlockGradientHook      — module-level backward hook (production
#                                path); registered once per MoE block via
#                                ``register_full_backward_hook``; reads all
#                                expert ``.grad`` attributes in one sweep,
#                                ~64× fewer Python→C++ calls per backward
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
from typing import TYPE_CHECKING, List, Optional

import torch

if TYPE_CHECKING:
    import torch.nn as nn

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
        "last_fired_step",
        "_pending_norm",
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

        # Tracks the last step this hook actually fired (i.e. wrote an event).
        # Used by HookManager.flush_missing_gradient_events() to detect experts
        # that received zero tokens this step and need a zero-norm stamp.
        # None means the hook has never fired.
        self.last_fired_step: Optional[int] = None

        # Deferred norm tensor (GPU). Set during __call__ instead of
        # immediately calling .item(), so all norms from all hooks
        # (28 layers × 64 experts = 1,792 per step) can be
        # batch-transferred to CPU in a single .cpu() call by
        # HookManager.flush_missing_gradient_events(), reducing backward
        # overhead from O(N_experts × N_layers) GPU→CPU syncs to O(1).
        self._pending_norm: "Optional[torch.Tensor]" = None

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
            # Record that this hook successfully fired for this step so
            # HookManager.flush_missing_gradient_events() can skip it.
            self.last_fired_step = self._global_step

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


# ---------------------------------------------------------------------------
# MoEBlockGradientHook  (module-level; production path)
# ---------------------------------------------------------------------------


class MoEBlockGradientHook:
    """Module-level backward hook for all experts in one MoE block.

    Registered via ``nn.Module.register_full_backward_hook`` on the MoE
    block that is the *parent* of the router module (e.g.
    ``layers.5.mlp`` rather than ``layers.5.mlp.gate``). PyTorch fires
    this hook **once per backward pass** for the block, after the block's
    weight gradients have been accumulated into ``.grad``.

    Reading 64 ``.grad`` attributes in one Python function is
    **~64× cheaper** than 64 individual ``Tensor.register_hook`` callbacks
    — each of which enters and exits the Python interpreter separately
    during the backward graph traversal.  With 28 layers the saving is
    1 764 avoided Python→C++ round-trips per backward step.

    Zero-norm recording is handled inline: experts that received no
    tokens have ``param.grad is None``; the hook writes ``gradient_norm=0.0``
    for them directly, making the separate
    :meth:`~moewatch.hooks.manager.HookManager.flush_missing_gradient_events`
    complement step redundant.

    Parameters
    ----------
    layer_name : str
        Fully-qualified name of the *router* module (e.g.
        ``"layers.5.mlp.gate"``), used as the outer key for
        :class:`~moewatch.collector.stat_collector.StatCollector`'s
        gradient buffers — unchanged from the per-param convention so
        downstream consumers need no changes.
    expert_params : list[torch.nn.Parameter or None]
        One entry per expert, in expert-index order. ``None`` entries are
        skipped. Typically the first weight parameter of each expert's
        submodule as located by
        :meth:`~moewatch.hooks.manager.HookManager._find_expert_weight_parameters`.
    stat_collector : StatCollector
        Destination for emitted :class:`GradientEvent` objects.
    config : WatchConfig
        Shared configuration; ``config.sample_every`` controls sampling.
    """

    __slots__ = (
        "layer_name",
        "expert_params",
        "stat_collector",
        "config",
        "_global_step",
    )

    def __init__(
        self,
        layer_name: str,
        expert_params: "List[Optional[torch.nn.Parameter]]",
        stat_collector: "StatCollector",  # noqa: F821
        config: "WatchConfig",            # noqa: F821
    ) -> None:
        self.layer_name    = layer_name
        self.expert_params = expert_params
        self.stat_collector = stat_collector
        self.config        = config
        self._global_step: int = 0

    # ------------------------------------------------------------------
    # Hook entry point (module full-backward-hook signature)
    # ------------------------------------------------------------------

    def __call__(
        self,
        module: "nn.Module",
        grad_input: tuple,
        grad_output: tuple,
    ) -> None:
        """Module full-backward hook: record all expert gradient norms.

        Parameters
        ----------
        module : nn.Module
            The MoE block whose backward just completed.
        grad_input : tuple
            Gradients w.r.t. the module's inputs (not used).
        grad_output : tuple
            Gradients w.r.t. the module's outputs (not used).

        Returns
        -------
        None
            This hook never modifies gradients.

        Notes
        -----
        By the time this hook fires, each expert's weight parameter has
        already had its gradient accumulated into ``.grad`` (or left as
        ``None`` if the expert received no tokens). Reading ``.grad``
        here is therefore always valid.

        All per-expert norms for this block are computed on the GPU in a
        single ``torch.stack(...).cpu()`` call rather than one ``.item()``
        per expert. This reduces GPU→CPU synchronisations from
        ``N_experts`` (64) to **1 per block per sampled step**, cutting
        the total sync count from 1 792 to 28 per backward pass.
        """
        sample_every = max(1, self.config.sample_every)
        if self._global_step % sample_every != 0:
            return

        try:
            # ---- 1. Collect per-expert grad tensors ----------------------
            # Separate experts into two groups:
            #   active  — param.grad is not None  → compute norm on GPU
            #   silent  — param.grad is None       → norm is 0.0
            active_indices: list = []   # expert_id for active experts
            active_norms_gpu: list = [] # 0-dim GPU tensors, one per active
            silent_indices: list = []   # expert_id for silent experts

            for expert_id, param in enumerate(self.expert_params):
                if param is None or not param.requires_grad:
                    continue
                if param.grad is None:
                    silent_indices.append(expert_id)
                else:
                    active_indices.append(expert_id)
                    active_norms_gpu.append(
                        param.grad.detach().norm(p=2)  # 0-dim GPU tensor
                    )

            # ---- 2. Single GPU→CPU transfer for all active norms ----------
            if active_norms_gpu:
                # stack → one contiguous GPU tensor → one .cpu() call
                batch_cpu = torch.stack(active_norms_gpu).cpu()
                active_norms = batch_cpu.tolist()
            else:
                active_norms = []

            # ---- 3. Write events ------------------------------------------
            ts = time.time()
            active_iter = iter(zip(active_indices, active_norms))

            # Pre-build a lookup for quick dispatch
            norm_map: dict = {
                eid: n for eid, n in zip(active_indices, active_norms)
            }
            for eid in silent_indices:
                norm_map[eid] = 0.0

            for expert_id in range(len(self.expert_params)):
                if expert_id not in norm_map:
                    continue  # param is None or doesn't require grad
                event = GradientEvent(
                    timestamp=ts,
                    global_step=self._global_step,
                    layer_name=self.layer_name,
                    expert_id=expert_id,
                    gradient_norm=norm_map[expert_id],
                    gradient_magnitude=norm_map[expert_id],
                )
                self.stat_collector.write_gradient_event(event)

        except Exception as exc:  # pylint: disable=broad-except
            logger.debug(
                "[MoEWatch] MoEBlockGradientHook('%s'): "
                "unexpected error (events skipped): %s",
                self.layer_name,
                exc,
            )

    # ------------------------------------------------------------------
    # Step counter management (driven by HookManager)
    # ------------------------------------------------------------------

    def set_global_step(self, global_step: int) -> None:
        """Update the training step number used for sampling and events.

        Parameters
        ----------
        global_step : int
            Current training step number.
        """
        self._global_step = global_step
