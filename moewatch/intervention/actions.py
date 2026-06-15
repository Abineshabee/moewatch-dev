# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# moewatch/intervention/actions.py [v0.2.0]
# =============================================================================
#
# Project      : MoEWatch
# Version      : v0.2.0
# Description  : Reversible intervention action classes. Each action wraps a
#                 single, targeted, reversible change to live training
#                 hyperparameters or forward-pass behaviour for one MoE
#                 router layer. Every action records the information it
#                 needs to undo itself, and reports a human-readable log
#                 line for auditing.
#
#                 Four candidate actions are provided:
#
#                   - AuxLossAction       — soft load balancing via an
#                                            increased auxiliary loss
#                                            coefficient.
#                   - RouterNoiseAction    — forced exploration via Gaussian
#                                            noise injected into router
#                                            logits.
#                   - ExpertDropoutAction  — reduced over-reliance via an
#                                            increased expert dropout rate.
#                   - NoOpAction           — conservative baseline; does
#                                            nothing.
#
#                 All actions are applied/reverted against the model
#                 (``torch.nn.Module``) being trained. Hooks registered by
#                 actions are read-only with respect to model *weights*
#                 (zero weight modification guarantee) — only forward-pass
#                 *behaviour* (e.g. injected noise) or model/config
#                 hyperparameters are touched, and every change is
#                 reversible via :meth:`InterventionAction.revert`.
#
# Author       : Abinesh N (@Abineshabee)
# Repository   : https://github.com/Abineshabee/MoEWatch
# License      : Apache 2.0
#
# Contents
# --------
#   InterventionAction   — abstract base class for all actions
#   AuxLossAction        — increase aux_loss_coef on a target layer
#   RouterNoiseAction     — inject Gaussian noise into router logits
#   ExpertDropoutAction  — increase expert dropout probability
#   NoOpAction            — do nothing (conservative baseline)
#
# Usage
# -----
#   from moewatch.intervention.actions import AuxLossAction
#
#   action = AuxLossAction(layer_name="model.layers.5.block_sparse_moe.gate",
#                           delta=0.05)
#   action.apply(model)
#   ...
#   action.revert(model)
#
# =============================================================================

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

try:
    import torch
    from torch import Tensor

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch is a hard dependency in practice
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[assignment, misc]
    _TORCH_AVAILABLE = False


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class InterventionAction(ABC):
    """Abstract base class for all reversible intervention actions.

    Every concrete action targets a single named module (``layer_name``)
    within the model owned by the ``trainer`` it is applied to, makes one
    bounded, reversible change, and records everything required to undo
    that change in :meth:`revert`.

    Parameters
    ----------
    action_type : str
        Short identifier for the action kind, e.g. ``"aux_loss"``,
        ``"router_noise"``, ``"expert_dropout"``, or ``"noop"``. Used by
        policies, the :class:`~moewatch.intervention.engine.InterventionEngine`,
        and reports for logging and bookkeeping.
    layer_name : str
        Dotted module path of the target MoE layer (router or expert
        block), resolvable via ``model.get_submodule(layer_name)``.
    delta : float
        Signed magnitude of the change this action represents. Used by
        :class:`~moewatch.intervention.safety.SafetyGuard` to enforce
        ``config.intervention_max_delta``. ``0.0`` for actions with no
        scalar magnitude (e.g. :class:`NoOpAction`).

    Attributes
    ----------
    action_type : str
        See above.
    layer_name : str
        See above.
    delta : float
        See above.
    applied_step : int
        Training step at which :meth:`apply` was called, or ``-1`` if the
        action has not yet been applied.

    Notes
    -----
    Subclasses must implement :meth:`apply` and :meth:`revert`. Both
    methods must be safe to call multiple times: ``apply`` after an
    already-applied action, and ``revert`` on an action that was never
    applied (or already reverted), should be no-ops rather than errors.
    This makes the action safe to use from
    :class:`~moewatch.intervention.engine.InterventionEngine`, which may
    call :meth:`revert` defensively during error handling.
    """

    def __init__(self, action_type: str, layer_name: str, delta: float = 0.0) -> None:
        if not isinstance(layer_name, str) or not layer_name:
            raise ValueError(
                f"[MoEWatch] {type(self).__name__}: 'layer_name' must be a "
                f"non-empty string, got {layer_name!r}."
            )

        self.action_type: str = action_type
        self.layer_name: str = layer_name
        self.delta: float = float(delta)
        self.applied_step: int = -1

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def apply(self, model: Any) -> None:
        """Apply this action to live training.

        Modifies ``model`` state (configuration, hyperparameters, or
        forward-pass hooks) in a targeted, bounded, and reversible way.

        Parameters
        ----------
        model : torch.nn.Module
            The model being trained. Concrete subclasses access
            ``model.config``, named submodules via
            ``model.get_submodule(...)``, or similar as required.

        Returns
        -------
        None

        Notes
        -----
        Implementations must record :attr:`applied_step` (set by the
        caller via :meth:`mark_applied`, see
        :class:`~moewatch.intervention.engine.InterventionEngine`) and any
        original values required to undo the change in :meth:`revert`.
        Must be idempotent: calling ``apply`` twice in a row should not
        compound the change.
        """
        raise NotImplementedError

    @abstractmethod
    def revert(self, model: Any) -> None:
        """Undo this action, restoring previous values.

        Parameters
        ----------
        model : torch.nn.Module
            The model being trained.

        Returns
        -------
        None

        Notes
        -----
        Must be safe to call on an action that was never applied (no-op
        in that case), and idempotent if called multiple times.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def mark_applied(self, step: int) -> None:
        """Record the training step at which this action was applied.

        Parameters
        ----------
        step : int
            Current training step.

        Returns
        -------
        None

        Notes
        -----
        Intended to be called by
        :class:`~moewatch.intervention.engine.InterventionEngine`
        immediately after :meth:`apply` succeeds.
        """
        self.applied_step = int(step)

    def log(self) -> str:
        """Return a human-readable description of this action.

        Returns
        -------
        str
            One-line summary including action type, target layer, signed
            delta, and the step at which it was applied (or "not applied"
            if :attr:`applied_step` is ``-1``).
        """
        applied = (
            f"step={self.applied_step}" if self.applied_step >= 0 else "not applied"
        )
        return (
            f"[{self.action_type}] layer={self.layer_name!r} "
            f"delta={self.delta:+.4f} ({applied})"
        )

    # ------------------------------------------------------------------
    # Module resolution helper
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_module(model: Any, layer_name: str) -> Optional["torch.nn.Module"]:
        """Resolve ``layer_name`` to a submodule on ``model``.

        Parameters
        ----------
        model : torch.nn.Module
            Model to search.
        layer_name : str
            Dotted module path, e.g.
            ``"model.layers.5.block_sparse_moe.gate"``.

        Returns
        -------
        torch.nn.Module or None
            The resolved submodule, or ``None`` if ``model`` is ``None``
            or the path does not resolve (logged as a warning rather than
            raised, so that a single unresolvable layer downgrades
            gracefully rather than crashing training).

        Notes
        -----
        Uses ``torch.nn.Module.get_submodule``, which raises
        ``AttributeError`` for an invalid path; that error is caught and
        converted into a logged warning + ``None`` return.
        """
        if model is None:
            logger.warning(
                "[MoEWatch] InterventionAction: model is None; cannot "
                "resolve layer '%s'.",
                layer_name,
            )
            return None

        try:
            return model.get_submodule(layer_name)
        except AttributeError as exc:
            logger.warning(
                "[MoEWatch] InterventionAction: failed to resolve layer "
                "'%s' on model (%s).",
                layer_name,
                exc,
            )
            return None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(layer_name={self.layer_name!r}, "
            f"delta={self.delta!r}, applied_step={self.applied_step})"
        )


# ---------------------------------------------------------------------------
# AuxLossAction
# ---------------------------------------------------------------------------


class AuxLossAction(InterventionAction):
    """Soft load balancing via an increased auxiliary loss coefficient.

    Increases the model's load-balancing auxiliary loss coefficient
    (``aux_loss_coef`` / ``router_aux_loss_coef``, depending on model
    family) by :attr:`delta`. A larger auxiliary loss penalises uneven
    expert utilisation more heavily, encouraging the router to
    redistribute tokens more evenly across experts without directly
    touching any model weights.

    Parameters
    ----------
    layer_name : str
        Target MoE layer. Retained for logging and safety-check
        attribution; the auxiliary loss coefficient itself is a
        model-config-level (global) hyperparameter for most architectures,
        so this action conceptually applies "on behalf of" ``layer_name``
        even though the underlying config field is shared.
    delta : float, default=0.05
        Amount by which to increase ``aux_loss_coef``. Must be positive.

    Attributes
    ----------
    delta : float
        Increase amount (inherited, see :class:`InterventionAction`).
    _original_coef : float or None
        Backup of the coefficient's value prior to :meth:`apply`, used by
        :meth:`revert`. ``None`` before :meth:`apply` has been called.

    Notes
    -----
    Looks for the coefficient attribute under several common names, in
    order: ``aux_loss_coef``, ``router_aux_loss_coef``,
    ``router_jitter_noise`` is *not* used here (see
    :class:`RouterNoiseAction` for jitter-style perturbation). If none of
    the recognised attribute names are found on ``trainer.model.config``,
    :meth:`apply` logs a warning and becomes a no-op (the action degrades
    gracefully rather than raising, so a single unsupported model family
    cannot crash a training run).
    """

    #: Candidate attribute names for the auxiliary loss coefficient,
    #: checked in order on ``trainer.model.config``.
    _COEF_ATTRS: tuple[str, ...] = ("aux_loss_coef", "router_aux_loss_coef")

    def __init__(self, layer_name: str, delta: float = 0.05) -> None:
        if delta <= 0.0:
            raise ValueError(
                f"[MoEWatch] AuxLossAction: 'delta' must be positive, "
                f"got {delta}."
            )

        super().__init__(action_type="aux_loss", layer_name=layer_name, delta=delta)
        self._original_coef: Optional[float] = None
        self._coef_attr: Optional[str] = None

    def apply(self, model: Any) -> None:
        """Increase ``aux_loss_coef`` on the model config by :attr:`delta`.

        Parameters
        ----------
        model : torch.nn.Module
            Model whose ``.config`` will be modified.

        Returns
        -------
        None

        Notes
        -----
        Idempotent: if already applied (``_original_coef`` already
        recorded), this call is a no-op so the increase is not compounded
        by repeated calls.
        """
        if self._original_coef is not None:
            logger.debug(
                "[MoEWatch] AuxLossAction: already applied to '%s'; "
                "skipping duplicate apply().",
                self.layer_name,
            )
            return

        config = self._resolve_config(model)
        if config is None:
            return

        attr = self._find_coef_attr(config)
        if attr is None:
            logger.warning(
                "[MoEWatch] AuxLossAction: model config has none of %s; "
                "action is a no-op for layer '%s'.",
                self._COEF_ATTRS,
                self.layer_name,
            )
            return

        original = float(getattr(config, attr))
        self._original_coef = original
        self._coef_attr = attr

        new_value = original + self.delta
        setattr(config, attr, new_value)

        logger.info(
            "[MoEWatch] AuxLossAction: increased '%s' from %.6f to %.6f "
            "(layer='%s').",
            attr,
            original,
            new_value,
            self.layer_name,
        )

    def revert(self, model: Any) -> None:
        """Restore the original auxiliary loss coefficient.

        Parameters
        ----------
        model : torch.nn.Module
            Model whose ``.config`` will be restored.

        Returns
        -------
        None

        Notes
        -----
        No-op if :meth:`apply` was never successfully applied (i.e.
        ``_original_coef`` is ``None``).
        """
        if self._original_coef is None or self._coef_attr is None:
            return

        config = self._resolve_config(model)
        if config is None:
            return

        setattr(config, self._coef_attr, self._original_coef)

        logger.info(
            "[MoEWatch] AuxLossAction: restored '%s' to %.6f (layer='%s').",
            self._coef_attr,
            self._original_coef,
            self.layer_name,
        )

        self._original_coef = None
        self._coef_attr = None

    @staticmethod
    def _resolve_config(model: Any) -> Optional[Any]:
        """Return ``model.config``, or ``None`` if unavailable."""
        config = getattr(model, "config", None)
        if config is None:
            logger.warning(
                "[MoEWatch] AuxLossAction: model.config is "
                "unavailable; action is a no-op."
            )
            return None
        return config

    def _find_coef_attr(self, config: Any) -> Optional[str]:
        """Return the first recognised coefficient attribute present."""
        for attr in self._COEF_ATTRS:
            if hasattr(config, attr):
                return attr
        return None


# ---------------------------------------------------------------------------
# RouterNoiseAction
# ---------------------------------------------------------------------------


class RouterNoiseAction(InterventionAction):
    """Forced exploration via Gaussian noise injected into router logits.

    Registers a temporary forward hook on the target router module that
    adds zero-mean Gaussian noise (standard deviation
    :attr:`noise_scale`) to its output logits. This breaks
    self-reinforcing "rich get richer" routing feedback loops by injecting
    randomness into the top-k expert selection, without modifying any
    model weights.

    Parameters
    ----------
    layer_name : str
        Dotted path to the target router module (the module whose forward
        output is the routing logits tensor).
    noise_scale : float, default=0.1
        Standard deviation of the injected Gaussian noise. Must be
        positive. Exposed as :attr:`delta` for safety-check purposes.

    Attributes
    ----------
    noise_scale : float
        See above. Equal to :attr:`delta`.
    _hook_handle : torch.utils.hooks.RemovableHandle or None
        Handle for the registered forward hook, used by :meth:`revert` to
        remove it. ``None`` if no hook is currently registered.

    Notes
    -----
    The injected noise is added only to the primary output tensor of the
    router's forward pass. If the router returns a tuple (common for
    HuggingFace MoE gates, e.g. ``(logits, ...)``), noise is added to the
    first element only and the rest of the tuple is passed through
    unchanged. If the output is neither a tensor nor a tuple with a tensor
    first element, the hook logs a warning once and passes the output
    through unmodified (graceful degradation).
    """

    def __init__(self, layer_name: str, noise_scale: float = 0.1) -> None:
        if noise_scale <= 0.0:
            raise ValueError(
                f"[MoEWatch] RouterNoiseAction: 'noise_scale' must be "
                f"positive, got {noise_scale}."
            )

        super().__init__(
            action_type="router_noise", layer_name=layer_name, delta=noise_scale
        )
        self.noise_scale: float = noise_scale
        self._hook_handle: Any = None
        self._warned_unsupported_output: bool = False

    def apply(self, model: Any) -> None:
        """Register a forward hook injecting noise into router logits.

        Parameters
        ----------
        model : torch.nn.Module
            Model containing the target router module.

        Returns
        -------
        None

        Notes
        -----
        Idempotent: if a hook is already registered (
        :attr:`_hook_handle` is not ``None``), this call is a no-op.
        If the target layer cannot be resolved or ``torch`` is
        unavailable, logs a warning and becomes a no-op.
        """
        if self._hook_handle is not None:
            logger.debug(
                "[MoEWatch] RouterNoiseAction: hook already registered on "
                "'%s'; skipping duplicate apply().",
                self.layer_name,
            )
            return

        if not _TORCH_AVAILABLE:
            logger.warning(
                "[MoEWatch] RouterNoiseAction: torch is unavailable; "
                "action is a no-op."
            )
            return

        module = self._resolve_module(model, self.layer_name)
        if module is None:
            return

        noise_scale = self.noise_scale

        def _noise_hook(
            _module: "torch.nn.Module", _inputs: Any, output: Any
        ) -> Any:
            if isinstance(output, Tensor):
                return output + torch.randn_like(output) * noise_scale

            if isinstance(output, tuple) and len(output) > 0 and isinstance(
                output[0], Tensor
            ):
                noisy_first = output[0] + torch.randn_like(output[0]) * noise_scale
                return (noisy_first,) + tuple(output[1:])

            if not self._warned_unsupported_output:
                logger.warning(
                    "[MoEWatch] RouterNoiseAction: router '%s' returned an "
                    "unsupported output type (%s); noise injection "
                    "skipped for this forward pass.",
                    self.layer_name,
                    type(output),
                )
                self._warned_unsupported_output = True

            return output

        self._hook_handle = module.register_forward_hook(_noise_hook)

        logger.info(
            "[MoEWatch] RouterNoiseAction: registered noise injection "
            "hook (std=%.4f) on '%s'.",
            noise_scale,
            self.layer_name,
        )

    def revert(self, model: Any) -> None:
        """Remove the noise-injection forward hook.

        Parameters
        ----------
        model : torch.nn.Module
            Unused directly; accepted for interface consistency with
            :class:`InterventionAction`.

        Returns
        -------
        None

        Notes
        -----
        No-op if no hook is currently registered.
        """
        if self._hook_handle is None:
            return

        self._hook_handle.remove()
        self._hook_handle = None

        logger.info(
            "[MoEWatch] RouterNoiseAction: removed noise injection hook "
            "from '%s'.",
            self.layer_name,
        )


# ---------------------------------------------------------------------------
# ExpertDropoutAction
# ---------------------------------------------------------------------------


class ExpertDropoutAction(InterventionAction):
    """Prevent over-reliance via a temporarily increased expert dropout rate.

    Increases the dropout probability of the target expert module(s) by
    :attr:`dropout_delta`, forcing the router to rely less on a small set
    of dominant experts and explore underutilized ones.

    Parameters
    ----------
    layer_name : str
        Dotted path to the target expert module (an
        ``nn.Module`` expected to either *be* an ``nn.Dropout`` instance,
        or to contain one or more ``nn.Dropout`` submodules whose ``p``
        attributes will all be adjusted together).
    dropout_delta : float, default=0.1
        Amount by which to increase the dropout probability ``p``. Must be
        positive. The resulting ``p`` is clamped to ``[0.0, 1.0]``.

    Attributes
    ----------
    dropout_delta : float
        See above. Equal to :attr:`delta`.
    _original_dropout : dict[str, float] or None
        Mapping of dotted submodule names (relative to ``layer_name``,
        empty string if ``layer_name`` itself is the dropout module) to
        their original ``p`` values, used by :meth:`revert`. ``None``
        before :meth:`apply` has been called successfully.

    Notes
    -----
    If the resolved module is neither a ``nn.Dropout`` instance nor
    contains any ``nn.Dropout`` submodules, :meth:`apply` logs a warning
    and becomes a no-op.
    """

    def __init__(self, layer_name: str, dropout_delta: float = 0.1) -> None:
        if dropout_delta <= 0.0:
            raise ValueError(
                f"[MoEWatch] ExpertDropoutAction: 'dropout_delta' must be "
                f"positive, got {dropout_delta}."
            )

        super().__init__(
            action_type="expert_dropout", layer_name=layer_name, delta=dropout_delta
        )
        self.dropout_delta: float = dropout_delta
        self._original_dropout: Optional[dict[str, float]] = None

    def apply(self, model: Any) -> None:
        """Increase dropout probability on the target expert module(s).

        Parameters
        ----------
        model : torch.nn.Module
            Model containing the target expert module.

        Returns
        -------
        None

        Notes
        -----
        Idempotent: if already applied (``_original_dropout`` recorded),
        this call is a no-op. If ``torch`` is unavailable, the target
        layer cannot be resolved, or no ``nn.Dropout`` modules are found,
        logs a warning and becomes a no-op.
        """
        if self._original_dropout is not None:
            logger.debug(
                "[MoEWatch] ExpertDropoutAction: already applied to '%s'; "
                "skipping duplicate apply().",
                self.layer_name,
            )
            return

        if not _TORCH_AVAILABLE:
            logger.warning(
                "[MoEWatch] ExpertDropoutAction: torch is unavailable; "
                "action is a no-op."
            )
            return

        module = self._resolve_module(model, self.layer_name)
        if module is None:
            return

        dropout_modules = dict(self._find_dropout_modules(module))
        if not dropout_modules:
            logger.warning(
                "[MoEWatch] ExpertDropoutAction: no nn.Dropout submodules "
                "found under '%s'; action is a no-op.",
                self.layer_name,
            )
            return

        original: dict[str, float] = {}
        for sub_name, dropout_module in dropout_modules.items():
            original[sub_name] = float(dropout_module.p)
            new_p = min(1.0, max(0.0, dropout_module.p + self.dropout_delta))
            dropout_module.p = new_p

        self._original_dropout = original

        logger.info(
            "[MoEWatch] ExpertDropoutAction: increased dropout by %.4f on "
            "%d submodule(s) under '%s'.",
            self.dropout_delta,
            len(original),
            self.layer_name,
        )

    def revert(self, model: Any) -> None:
        """Restore the original dropout probabilities.

        Parameters
        ----------
        model : torch.nn.Module
            Model containing the target expert module.

        Returns
        -------
        None

        Notes
        -----
        No-op if :meth:`apply` was never successfully applied. If the
        target layer can no longer be resolved (e.g. model restructured),
        logs a warning and clears the recorded state without raising.
        """
        if self._original_dropout is None:
            return

        module = self._resolve_module(model, self.layer_name)
        if module is None:
            self._original_dropout = None
            return

        dropout_modules = dict(self._find_dropout_modules(module))

        for sub_name, original_p in self._original_dropout.items():
            dropout_module = dropout_modules.get(sub_name)
            if dropout_module is None:
                logger.warning(
                    "[MoEWatch] ExpertDropoutAction: submodule '%s' under "
                    "'%s' no longer found during revert; skipping.",
                    sub_name or "<self>",
                    self.layer_name,
                )
                continue
            dropout_module.p = original_p

        logger.info(
            "[MoEWatch] ExpertDropoutAction: restored dropout for %d "
            "submodule(s) under '%s'.",
            len(self._original_dropout),
            self.layer_name,
        )

        self._original_dropout = None

    @staticmethod
    def _find_dropout_modules(
        module: "torch.nn.Module",
    ) -> list[tuple[str, "torch.nn.Module"]]:
        """Return ``(relative_name, module)`` pairs for all dropout layers.

        Parameters
        ----------
        module : torch.nn.Module
            Module to search. If ``module`` itself is an ``nn.Dropout``,
            it is returned with relative name ``""``. Otherwise, all
            ``nn.Dropout`` descendants are returned with their dotted
            relative names via ``named_modules()``.

        Returns
        -------
        list[tuple[str, torch.nn.Module]]
            Possibly empty list of dropout modules found.
        """
        if isinstance(module, torch.nn.Dropout):
            return [("", module)]

        return [
            (name, sub)
            for name, sub in module.named_modules()
            if isinstance(sub, torch.nn.Dropout)
        ]


# ---------------------------------------------------------------------------
# NoOpAction
# ---------------------------------------------------------------------------


class NoOpAction(InterventionAction):
    """Conservative baseline action that does nothing.

    Used as the safe default when no intervention is warranted, and as
    the downgrade target when
    :class:`~moewatch.intervention.safety.SafetyGuard` rejects a proposed
    action. Monitoring continues unaffected.

    Parameters
    ----------
    layer_name : str, default="<none>"
        Target layer for logging/attribution purposes only. No module
        resolution is performed.

    Notes
    -----
    :attr:`delta` is always ``0.0``, so this action always passes
    :class:`~moewatch.intervention.safety.SafetyGuard`'s delta-limit
    check.
    """

    def __init__(self, layer_name: str = "<none>") -> None:
        super().__init__(action_type="noop", layer_name=layer_name, delta=0.0)

    def apply(self, model: Any) -> None:
        """No-op (do nothing).

        Parameters
        ----------
        model : torch.nn.Module
            Unused; accepted for interface consistency.

        Returns
        -------
        None
        """
        return None

    def revert(self, model: Any) -> None:
        """No-op (do nothing).

        Parameters
        ----------
        model : torch.nn.Module
            Unused; accepted for interface consistency.

        Returns
        -------
        None
        """
        return None
