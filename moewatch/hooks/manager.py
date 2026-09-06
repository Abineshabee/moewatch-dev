# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# moewatch/hooks/manager.py
# =============================================================================
#
# Project      : MoEWatch
# Version      : v0.2.0
# Description  : HookManager owns the full lifecycle of every hook MoEWatch
#                 registers on a model: forward hooks on detected router
#                 modules (RouterForwardHook) and, when gradient-based
#                 analysis is enabled, backward hooks on per-expert weight
#                 parameters (GradientStarvationHook).
#
#                 attach() performs auto-detection, registers all hooks,
#                 and stores every returned handle. detach() removes every
#                 handle and is idempotent and safe to call multiple times,
#                 including after a partial/failed attach() (guaranteed via
#                 try/finally in attach()).
#
#                 This is the single chokepoint through which MoEWatch
#                 touches the model — all hook registration and removal
#                 flows through this class, which keeps the
#                 zero-weight-modification guarantee auditable in one
#                 place.
#
# Author       : Abinesh N (@Abineshabee)
# Repository   : https://github.com/Abineshabee/MoEWatch
# License      : Apache 2.0
#
# Contents
# --------
#   HookManager — attach/detach lifecycle owner for all MoEWatch hooks
#
# Usage
# -----
#   from moewatch.hooks.manager import HookManager
#
#   manager = HookManager(model, stat_collector, config)
#   manager.attach()
#   ...
#   manager.set_global_step(step)   # before each forward/backward pass
#   ...
#   manager.detach()
#
# =============================================================================

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List

import torch
import torch.nn as nn

from moewatch.config import WatchConfig
from moewatch.hooks.detection import detect_router_modules
from moewatch.hooks.gradient_hook import (
    GradientEvent,
    GradientStarvationHook,
    MoEBlockGradientHook,
)
from moewatch.hooks.router_hook import RouterForwardHook

if TYPE_CHECKING:
    from moewatch.collector.stat_collector import StatCollector

logger = logging.getLogger(__name__)


class HookManager:
    """Owns the lifecycle of all forward and backward hooks on the model.

    Responsible for:

    - Auto-detecting MoE router modules (via
      :func:`~moewatch.hooks.detection.detect_router_modules`).
    - Registering one :class:`~moewatch.hooks.router_hook.RouterForwardHook`
      per detected router via ``register_forward_hook``.
    - Registering one
      :class:`~moewatch.hooks.gradient_hook.GradientStarvationHook` per
      detected expert weight parameter via ``Tensor.register_hook``
      (Tier 1 gradient-starvation signal source).
    - Tracking every returned
      :class:`torch.utils.hooks.RemovableHandle` so that :meth:`detach`
      can remove all of them, guaranteed via ``try/finally`` in
      :meth:`attach`.
    - Propagating the current training step to every active hook via
      :meth:`set_global_step`, so emitted events carry accurate step
      numbers and gradient sampling (``config.sample_every``) behaves
      correctly.

    Parameters
    ----------
    model : torch.nn.Module
        The MoE model to instrument. Never modified — all hooks are
        read-only observers.
    stat_collector : StatCollector
        Destination for all :class:`RoutingEvent` and :class:`GradientEvent`
        objects produced by registered hooks. Layers are registered with
        the collector (via ``register_layer``) as part of :meth:`attach`.
    config : WatchConfig
        Shared configuration. ``config.router_modules`` controls
        auto-detection (see :func:`detect_router_modules`);
        ``config.sample_every`` controls gradient-hook sampling.

    Attributes
    ----------
    model : torch.nn.Module
        Reference to the monitored model (unchanged).
    stat_collector : StatCollector
        Reference to the event destination.
    config : WatchConfig
        Reference to the shared configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        stat_collector: "StatCollector",
        config: WatchConfig,
    ) -> None:
        self.model: nn.Module = model
        self.stat_collector = stat_collector
        self.config: WatchConfig = config

        # All removable handles for forward and backward hooks. Cleared on
        # successful detach().
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

        # Detected router modules: {layer_name: module}.
        self._layer_map: Dict[str, nn.Module] = {}

        # Active hook callables, kept so set_global_step() can fan out the
        # current training step to every hook without re-walking the model.
        self._router_hooks: List[RouterForwardHook] = []
        self._gradient_hooks: List[
            "GradientStarvationHook | MoEBlockGradientHook"
        ] = []

        self._attached: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def attach(self) -> None:
        """Attach forward and backward hooks to detected router modules.

        Performs, in order:

        1. Auto-detection of MoE router modules via
           :func:`detect_router_modules` (or resolution of
           ``config.router_modules`` if set).
        2. Registration of one :class:`RouterForwardHook` per detected
           router via ``register_forward_hook``.
        3. Registration of the layer with ``stat_collector`` (via
           ``register_layer``), so routing/gradient buffers exist before
           any event arrives.
        4. Registration of one :class:`GradientStarvationHook` per
           detected expert weight parameter, via ``Tensor.register_hook``,
           if expert parameters can be located for the layer.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If no router modules are detected (propagated from
            :func:`detect_router_modules`'s manual-override path, or
            raised here if auto-detection returns an empty mapping).

        Notes
        -----
        If any step after detection raises an exception, all hooks
        registered so far during this call are removed before the
        exception propagates (via ``try/finally`` calling
        :meth:`detach`), guaranteeing no partial/leaked hook state.
        Calling :meth:`attach` while already attached is a no-op (logs a
        warning and returns immediately).
        """
        if self._attached:
            logger.warning(
                "[MoEWatch] HookManager.attach() called while already "
                "attached. Ignored."
            )
            return

        try:
            self._layer_map = detect_router_modules(self.model, self.config)

            if not self._layer_map:
                raise ValueError(
                    "[MoEWatch] HookManager.attach(): no MoE router "
                    "modules detected. Set config.router_modules to "
                    "override auto-detection."
                )

            for layer_name, router_module in self._layer_map.items():
                self._attach_router_hook(layer_name, router_module)
                self._attach_gradient_hooks(layer_name, router_module)

            self._attached = True
            logger.info(
                "[MoEWatch] HookManager: attached hooks to %d router "
                "layer(s): %s",
                len(self._layer_map),
                list(self._layer_map.keys()),
            )

        except Exception:
            # Guarantee no partial hook state on failure.
            self._cleanup_handles()
            self._attached = False
            raise

    def detach(self) -> None:
        """Remove every registered hook handle.

        Iterates ``self._handles`` and calls ``.remove()`` on each,
        clears all internal bookkeeping (``_handles``, ``_router_hooks``,
        ``_gradient_hooks``, ``_layer_map``), and sets
        ``self._attached = False``.

        Returns
        -------
        None

        Notes
        -----
        Idempotent: safe to call multiple times, including when no hooks
        are currently attached (no-op in that case). Individual handle
        removal failures are caught and logged at WARNING level rather
        than aborting the loop, so a single bad handle cannot prevent
        cleanup of the remaining hooks.
        """
        if not self._handles and not self._attached:
            return

        self._cleanup_handles()
        self._attached = False
        logger.debug("[MoEWatch] HookManager: all hooks detached.")

    def is_attached(self) -> bool:
        """Return the current attachment state.

        Returns
        -------
        bool
            ``True`` if :meth:`attach` has completed successfully and
            :meth:`detach` has not since been called.
        """
        return self._attached

    # ------------------------------------------------------------------
    # Step propagation
    # ------------------------------------------------------------------

    def set_global_step(self, global_step: int) -> None:
        """Propagate the current training step to all active hooks.

        Should be called once per training step (typically at the start
        of ``MoEWatch.step()``) before the next forward/backward pass, so
        that all hooks tag emitted events with the correct step and
        compute the sampling decision (``global_step % config.sample_every``)
        correctly.

        Parameters
        ----------
        global_step : int
            Current training step number.

        Returns
        -------
        None
        """
        for hook in self._router_hooks:
            hook.set_global_step(global_step)
        for hook in self._gradient_hooks:
            hook.set_global_step(global_step)

    def flush_missing_gradient_events(self, step: int = 0) -> None:
        """Stamp zero-norm events for experts whose param hook did not fire.

        ``Tensor.register_hook`` only fires for parameters that participated
        in the backward graph, i.e. experts that received at least one token
        this step.  Experts that received zero tokens have no backward edge
        and their hook is silently skipped by autograd.

        This method iterates every registered :class:`GradientStarvationHook`
        and, for any hook whose ``last_fired_step`` is not the current step,
        writes a ``gradient_norm=0.0`` event so the downstream analyzer sees
        a zero rather than a missing observation for that expert.

        Called by :class:`~moewatch._watcher.MoEWatch` once per training
        step, after ``optimizer.step()`` and before ``watch.step()``.

        Parameters
        ----------
        step : int, optional
            Current global training step. Defaults to 0 if not provided.

        Returns
        -------
        None
        """
        import time as _time
        ts = _time.time()
        from moewatch.collector.stat_collector import GradientEvent

        for hook in self._gradient_hooks:
            if not isinstance(hook, GradientStarvationHook):
                continue
            if hook.last_fired_step == step:
                continue  # hook already fired for this step — expert was routed

            # Expert received no tokens this step: record a zero-norm event.
            event = GradientEvent(
                timestamp=ts,
                global_step=step,
                layer_name=hook.layer_name,
                expert_id=hook.expert_id,
                gradient_norm=0.0,
                gradient_magnitude=0.0,
            )
            try:
                hook.stat_collector.write_gradient_event(event)
            except Exception:
                pass  # never let flushing interrupt training

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def get_layer_map(self) -> Dict[str, nn.Module]:
        """Return the mapping of detected router layer names to modules.

        Returns
        -------
        dict[str, torch.nn.Module]
            Shallow copy of the internal layer map. Empty if
            :meth:`attach` has not been called or detection found
            nothing.
        """
        return dict(self._layer_map)

    # ------------------------------------------------------------------
    # Internal: hook registration
    # ------------------------------------------------------------------

    def _attach_router_hook(self, layer_name: str, router_module: nn.Module) -> None:
        """Register a :class:`RouterForwardHook` on ``router_module``.

        Also registers ``layer_name`` with ``stat_collector`` using the
        expert count inferred from the router module's ``out_features``
        attribute (falling back to ``2`` if unavailable; the collector's
        buffers will simply be created lazily/empty until the first real
        event arrives with the true shape).

        Parameters
        ----------
        layer_name : str
            Fully-qualified name of the router module.
        router_module : torch.nn.Module
            The router module instance to hook.

        Returns
        -------
        None
        """
        hook = RouterForwardHook(
            layer_name=layer_name,
            stat_collector=self.stat_collector,
            config=self.config,
            model=self.model,
        )
        handle = router_module.register_forward_hook(hook)

        self._handles.append(handle)
        self._router_hooks.append(hook)

        n_experts = self._infer_expert_count(router_module)

        try:
            self.stat_collector.register_layer(layer_name, n_experts)
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug(
                "[MoEWatch] HookManager: register_layer('%s', %d) "
                "failed (non-fatal): %s",
                layer_name,
                n_experts,
                exc,
            )

    def _attach_gradient_hooks(
        self, layer_name: str, router_module: nn.Module
    ) -> None:
        """Register per-expert Tensor.register_hook callbacks for gradient tracking.

        Locates the expert weight parameters associated with the gate's
        sibling ``nn.ModuleList`` and registers a ``Tensor.register_hook``
        on the **first weight parameter of each expert** via
        :class:`~moewatch.hooks.gradient_hook.GradientStarvationHook`.

        ``Tensor.register_hook`` fires *after* the gradient has been fully
        accumulated into ``param.grad`` for that tensor — which is the
        correct point to read it.  The previous approach used
        ``register_full_backward_hook`` on the parent module, which fires
        when the *module's* backward completes but *before* child weight
        gradients are accumulated, so ``param.grad`` was always ``None``
        inside that hook and every expert reported ``gradient_norm=0.0``.

        Parameters
        ----------
        layer_name : str
            Fully-qualified name of the router module (e.g.
            ``"layers.5.mlp.gate"``). Used as the outer key in
            :class:`~moewatch.collector.stat_collector.StatCollector`'s
            gradient buffers.
        router_module : torch.nn.Module
            The router module instance; used only to derive the layer name.

        Returns
        -------
        None

        Notes
        -----
        If no expert parameters can be located for this layer (e.g. an
        architecture where experts are not exposed as a discoverable
        ``nn.ModuleList`` sibling), this method logs at DEBUG level and
        returns without raising — gradient-starvation analysis for this
        layer will be unavailable, while entropy/collapse analysis
        (which only depends on routing events) continues to function.
        """
        expert_params = self._find_expert_weight_parameters(layer_name)

        if not expert_params:
            logger.debug(
                "[MoEWatch] HookManager: no expert weight parameters "
                "found for layer '%s'; Tier 1 gradient-starvation signal "
                "will be unavailable for this layer.",
                layer_name,
            )
            return

        registered = 0
        for expert_id, param in enumerate(expert_params):
            if param is None or not param.requires_grad:
                continue

            hook = GradientStarvationHook(
                layer_name=layer_name,
                expert_id=expert_id,
                stat_collector=self.stat_collector,
                config=self.config,
            )
            # Tensor.register_hook fires after param.grad is fully accumulated,
            # which is the correct timing to read per-expert gradient norms.
            handle = param.register_hook(hook)

            self._handles.append(handle)
            self._gradient_hooks.append(hook)
            registered += 1

        logger.debug(
            "[MoEWatch] HookManager: registered %d per-expert Tensor.register_hook "
            "callback(s) for '%s'.",
            registered,
            layer_name,
        )


    def _infer_expert_count(self, router_module: nn.Module) -> int:
        """Infer the number of experts from a router module's shape.

        Checked in order:

        1. A ``num_experts`` integer attribute — present directly on the
           modern "TopKRouter" gate classes used by current Mixtral,
           Qwen3-MoE, DeepSeek-V3, and OLMoE implementations (e.g.
           ``Qwen3MoeTopKRouter.num_experts``), and the most direct
           signal available when present.
        2. ``out_features`` — for ``nn.Linear``-based gates (legacy
           Mixtral, custom architectures).
        3. ``weight.shape[0]`` — for the same modern "TopKRouter"
           classes when ``num_experts`` isn't exposed under that name;
           these store their projection as a raw
           ``nn.Parameter(torch.zeros(num_experts, hidden_dim))``
           rather than wrapping it in ``nn.Linear``, so it has no
           ``out_features`` but ``weight.shape[0]`` is the expert count
           (mirrors ``nn.Linear.weight``'s own
           ``[out_features, in_features]`` layout).

        Getting this wrong is not merely cosmetic: the returned value is
        used to pre-register this layer's expert-count with
        :class:`~moewatch.collector.stat_collector.StatCollector`
        *before* any forward pass occurs. ``StatCollector`` only trusts
        that count on a layer's *very first* registration — by the time
        the first real ``RoutingEvent`` arrives, the layer is already
        registered, so a wrong count here is never self-corrected. Every
        subsequent per-token expert index is then clamped into that
        (wrong, too-small) range before histogramming, silently folding
        e.g. a real 128-expert model's routing decisions into 2 phantom
        buckets and fabricating near-total "collapse" regardless of true
        routing health. Falling back to ``2`` is intentionally a last
        resort, not a normal case.

        Parameters
        ----------
        router_module : torch.nn.Module
            Router module to inspect.

        Returns
        -------
        int
            The inferred expert count from the first matching signal
            above, otherwise ``2`` as a conservative non-zero default
            (buffers sized this way are harmless placeholders; real
            shapes are derived from the first :class:`RoutingEvent` —
            for the layers where that self-correction path actually
            applies, see the note above for why it usually doesn't).
        """
        num_experts_attr = getattr(router_module, "num_experts", None)
        if isinstance(num_experts_attr, int) and num_experts_attr >= 1:
            return num_experts_attr

        out_features = getattr(router_module, "out_features", None)
        if isinstance(out_features, int) and out_features >= 1:
            return out_features

        weight = getattr(router_module, "weight", None)
        if isinstance(weight, torch.nn.Parameter) and weight.dim() == 2:
            n_experts = int(weight.shape[0])
            if n_experts >= 1:
                return n_experts

        return 2

    def _find_expert_weight_parameters(
        self, layer_name: str
    ) -> List["torch.nn.Parameter | None"]:
        """Locate per-expert weight parameters for gradient hooking.

        Strategy
        --------
        1. Resolve the router module's parent module via
           ``model.get_submodule(parent_name)`` where ``parent_name`` is
           ``layer_name`` with its final dotted component removed.
        2. Search the parent's *named children* for an attribute that is
           an ``nn.ModuleList`` (the conventional container for
           per-expert submodules across Mixtral / Qwen3-MoE / DeepSeek-MoE
           / OLMoE implementations), excluding the router module itself.
        3. For each module in that ``ModuleList``, select its first
           parameter (by ``named_parameters()`` iteration order) as the
           "primary weight" to hook — typically the first linear
           projection (``w1`` / ``gate_proj`` / ``fc1``), which is a
           representative proxy for that expert's gradient flow.

        Parameters
        ----------
        layer_name : str
            Fully-qualified dotted name of the router module.

        Returns
        -------
        list[torch.nn.Parameter | None]
            One entry per expert found in the located ``ModuleList``, in
            order. Entries are ``None`` if a given expert submodule has
            no parameters at all (skipped during hook registration).
            Returns an empty list if no suitable ``ModuleList`` could be
            located.
        """
        parent_name = self._parent_module_name(layer_name)

        try:
            if parent_name:
                parent_module = self.model.get_submodule(parent_name)
            else:
                parent_module = self.model
        except AttributeError:
            return []

        router_leaf_name = layer_name.rsplit(".", 1)[-1]

        experts_container: "nn.ModuleList | None" = None
        for child_name, child_module in parent_module.named_children():
            if child_name == router_leaf_name:
                continue
            if isinstance(child_module, nn.ModuleList) and len(child_module) > 1:
                experts_container = child_module
                break

        if experts_container is None:
            return []

        params: List["torch.nn.Parameter | None"] = []
        for expert_module in experts_container:
            primary_param: "torch.nn.Parameter | None" = None
            for _, param in expert_module.named_parameters():
                primary_param = param
                break
            params.append(primary_param)

        return params

    @staticmethod
    def _parent_module_name(qualified_name: str) -> str:
        """Return the dotted parent path of a qualified module name.

        Parameters
        ----------
        qualified_name : str
            Fully-qualified dotted module name (e.g.
            ``"model.layers.5.block_sparse_moe.gate"``).

        Returns
        -------
        str
            The dotted path with the final component removed (e.g.
            ``"model.layers.5.block_sparse_moe"``), or ``""`` if
            ``qualified_name`` has no ``"."`` (i.e. it is a top-level
            module).
        """
        if "." not in qualified_name:
            return ""
        return qualified_name.rsplit(".", 1)[0]

    # ------------------------------------------------------------------
    # Internal: cleanup
    # ------------------------------------------------------------------

    def _cleanup_handles(self) -> None:
        """Remove all stored hook handles and clear bookkeeping state.

        Each ``.remove()`` call is individually wrapped in a try/except
        so that a single failure does not prevent the remaining handles
        from being cleaned up. Always clears ``_handles``,
        ``_router_hooks``, ``_gradient_hooks``, and ``_layer_map``
        regardless of how many removals succeeded.

        Returns
        -------
        None
        """
        for handle in self._handles:
            try:
                handle.remove()
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "[MoEWatch] HookManager: failed to remove a hook "
                    "handle (non-fatal): %s",
                    exc,
                )

        self._handles.clear()
        self._router_hooks.clear()
        self._gradient_hooks.clear()
        self._layer_map.clear()
