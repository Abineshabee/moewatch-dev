# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# moewatch/policy/rule_policy.py [v0.2.0]
# =============================================================================
#
# Project      : MoEWatch
# Version      : v0.2.0
# Description  : Phase 1 deterministic intervention policy. Maps a layer's
#                 fused risk_score directly onto one of four candidate
#                 actions via fixed thresholds:
#
#                     risk < 0.3  -> NoOpAction          (do nothing)
#                     risk < 0.6  -> AuxLossAction        (soft load balance)
#                     risk < 0.8  -> RouterNoiseAction    (force exploration)
#                     risk >= 0.8 -> ExpertDropoutAction  (strongest action)
#
#                 Deterministic and explainable: identical
#                 (risk_score, intervention_history) inputs always yield
#                 the same action. Serves as the stable behavioural
#                 baseline against which Phase 2's
#                 :class:`~moewatch.policy.bandit_policy.BanditPolicy` is
#                 compared.
#
#                 Guards against oscillation (rapidly alternating between
#                 two actions) and cascade (repeatedly escalating the same
#                 layer) by tracking each layer's recent action history and
#                 downgrading to the next-weaker action when the
#                 threshold-selected action would repeat too soon.
#
# Author       : Abinesh N (@Abineshabee)
# Repository   : https://github.com/Abineshabee/MoEWatch
# License      : Apache 2.0
#
# Contents
# --------
#   RulePolicy — Phase 1 deterministic threshold-based policy
#
# Usage
# -----
#   from moewatch.policy.rule_policy import RulePolicy
#   from moewatch.policy.base import PolicyState
#
#   policy = RulePolicy(config)
#   action = policy.select_action(state)
#   ...
#   policy.update(state, action, reward)
#
# =============================================================================

from __future__ import annotations

import json
import logging
from collections import deque
from typing import TYPE_CHECKING, Deque, Dict

from moewatch.intervention.actions import (
    AuxLossAction,
    ExpertDropoutAction,
    InterventionAction,
    NoOpAction,
    RouterNoiseAction,
)
from moewatch.policy.base import PolicyBase, PolicyState

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance
    from moewatch.config import WatchConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thresholds and ordering
# ---------------------------------------------------------------------------

#: Risk-score thresholds separating the four action tiers. A risk_score
#: below ``_RISK_NOOP_MAX`` maps to NoOp, below ``_RISK_AUXLOSS_MAX`` maps
#: to AuxLoss, below ``_RISK_ROUTERNOISE_MAX`` maps to RouterNoise, and
#: anything at or above that maps to ExpertDropout.
_RISK_NOOP_MAX = 0.3
_RISK_AUXLOSS_MAX = 0.6
_RISK_ROUTERNOISE_MAX = 0.8

#: Action names in increasing order of intervention strength. Used to
#: locate the "next weaker" action when oscillation/cascade guards
#: downgrade a selection.
_ACTION_ORDER = ("noop", "aux_loss", "router_noise", "expert_dropout")

#: How many of a layer's most recent actions are retained for oscillation
#: and cascade detection.
_HISTORY_LEN = 5

#: If the threshold-selected action equals the immediately preceding
#: action for this layer more than this many times in the retained
#: history, it is considered a potential cascade and downgraded by one
#: tier.
_CASCADE_REPEAT_LIMIT = 3


class RulePolicy(PolicyBase):
    """Phase 1 deterministic intervention policy.

    Maps ``risk_score`` to an action via fixed thresholds:

    - ``risk_score < 0.3``  -> :class:`~moewatch.intervention.actions.NoOpAction`
    - ``risk_score < 0.6``  -> :class:`~moewatch.intervention.actions.AuxLossAction`
    - ``risk_score < 0.8``  -> :class:`~moewatch.intervention.actions.RouterNoiseAction`
    - ``risk_score >= 0.8`` -> :class:`~moewatch.intervention.actions.ExpertDropoutAction`

    Avoids oscillation (rapid alternation between two actions for the same
    layer) and cascade (the same escalating action repeating many times in
    a row for the same layer) by tracking each layer's recent action
    history and, when either pattern is detected, downgrading the
    threshold-selected action to the next-weaker tier (never below
    :class:`~moewatch.intervention.actions.NoOpAction`).

    Parameters
    ----------
    config : WatchConfig
        Configuration. ``config.intervention_max_delta`` caps the
        magnitude of every constructed action (``AuxLossAction``,
        ``RouterNoiseAction``, ``ExpertDropoutAction``); each action's
        built-in default magnitude is used unless it would exceed this
        cap, in which case it is clamped down to it.

    Attributes
    ----------
    config : WatchConfig
        See above.

    Notes
    -----
    This policy does not learn: :meth:`update` only records
    (state, action, reward) information for diagnostics and for the
    oscillation/cascade guards in :meth:`select_action`. Given the same
    ``(risk_score, layer history)``, :meth:`select_action` always returns
    the same action type.
    """

    def __init__(self, config: "WatchConfig") -> None:
        self.config: "WatchConfig" = config

        # layer_name -> recent action_type history (most recent last).
        self._recent_actions: Dict[str, Deque[str]] = {}

        # layer_name -> most recently *selected* action_type (prior to any
        # downgrade), used purely for diagnostics/logging.
        self._action_sequence: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # PolicyBase interface
    # ------------------------------------------------------------------

    def select_action(self, state: PolicyState) -> InterventionAction:
        """Select an action for ``state`` based on risk-score thresholds.

        Parameters
        ----------
        state : PolicyState
            Current per-layer state. ``state.layer_id`` identifies the
            target layer; ``state.risk_score`` drives the threshold
            selection.

        Returns
        -------
        InterventionAction
            One of :class:`~moewatch.intervention.actions.NoOpAction`,
            :class:`~moewatch.intervention.actions.AuxLossAction`,
            :class:`~moewatch.intervention.actions.RouterNoiseAction`, or
            :class:`~moewatch.intervention.actions.ExpertDropoutAction`,
            constructed with ``layer_name`` derived from
            ``state.layer_id``.

        Notes
        -----
        Algorithm:

        1. Map ``state.risk_score`` to a candidate action type via the
           fixed thresholds (see module docstring).
        2. If the candidate action type would repeat the immediately
           preceding action type for this layer **and** that action type
           has appeared at least :data:`_CASCADE_REPEAT_LIMIT` times in
           the retained history (a cascade), downgrade the candidate to
           the next-weaker tier.
        3. If the candidate action type alternates with the
           second-most-recent action type in a strict back-and-forth
           pattern across the retained history (oscillation), downgrade
           the candidate to the next-weaker tier.
        4. Construct and return the corresponding
           :class:`~moewatch.intervention.actions.InterventionAction`
           subclass, with magnitude capped to
           ``config.intervention_max_delta`` (see :meth:`_build_action`),
           targeting ``layer_name``.
        5. Record the *candidate* (pre-downgrade) action type in
           :attr:`_action_sequence` for diagnostics, and append the
           *final* (post-downgrade) action type to
           :attr:`_recent_actions` for this layer.

        ``layer_name`` is derived as ``f"layer_{state.layer_id}"`` —
        callers that need a real dotted module path should construct one
        from ``state.layer_id`` via
        :func:`~moewatch.hooks.detection.detect_router_modules` and pass
        that mapping in, or post-process the returned action's
        ``layer_name`` attribute before calling ``apply()``.
        """
        layer_name = state.layer_name if state.layer_name else f"layer_{state.layer_id}"
        history = self._recent_actions.setdefault(
            layer_name, deque(maxlen=_HISTORY_LEN)
        )

        candidate = self._risk_to_action_type(state.risk_score)
        final_action_type = self._apply_guards(layer_name, history, candidate)

        self._action_sequence[layer_name] = candidate
        history.append(final_action_type)

        if final_action_type != candidate:
            logger.info(
                "[MoEWatch] RulePolicy: layer='%s' risk=%.4f candidate='%s' "
                "downgraded to '%s' (oscillation/cascade guard).",
                layer_name,
                state.risk_score,
                candidate,
                final_action_type,
            )
        else:
            logger.debug(
                "[MoEWatch] RulePolicy: layer='%s' risk=%.4f -> action='%s'.",
                layer_name,
                state.risk_score,
                final_action_type,
            )

        return self._build_action(
            final_action_type, layer_name, max_delta=self.config.intervention_max_delta
        )

    def update(
        self, state: PolicyState, action: InterventionAction, reward: float
    ) -> None:
        """Log the observed (state, action, reward) for analysis.

        Parameters
        ----------
        state : PolicyState
            State at the time :paramref:`action` was selected.
        action : InterventionAction
            Action that was selected.
        reward : float
            Observed discounted counterfactual reward.

        Returns
        -------
        None

        Notes
        -----
        :class:`RulePolicy` does not learn from rewards — this method is a
        diagnostic hook only. It logs the observation at ``DEBUG`` level
        and otherwise does not alter future :meth:`select_action`
        behaviour (which depends only on ``risk_score`` and the
        oscillation/cascade history already updated by
        :meth:`select_action`).
        """
        logger.debug(
            "[MoEWatch] RulePolicy.update: layer_id=%d step=%d "
            "action='%s' reward=%.6f (no-op for deterministic policy).",
            state.layer_id,
            state.training_step,
            action.action_type,
            reward,
        )

    def save_checkpoint(self, path: str) -> None:
        """Save the per-layer action history to ``path`` as JSON.

        Parameters
        ----------
        path : str
            Destination file path.

        Returns
        -------
        None

        Notes
        -----
        :class:`RulePolicy` has minimal state: only the oscillation and
        cascade detection history per layer. This is saved so that, if
        monitoring is interrupted and resumed, the guards continue from
        where they left off rather than resetting to an empty history
        (which could allow a brief window of unguarded cascading
        immediately after resume).
        """
        payload = {
            "recent_actions": {
                layer: list(hist) for layer, hist in self._recent_actions.items()
            },
            "action_sequence": dict(self._action_sequence),
        }

        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        except OSError as exc:
            logger.warning(
                "[MoEWatch] RulePolicy: failed to write checkpoint to "
                "'%s' (%s).",
                path,
                exc,
            )
            raise

    def load_checkpoint(self, path: str) -> None:
        """Load the per-layer action history from ``path``.

        Parameters
        ----------
        path : str
            Source file path, previously written by
            :meth:`save_checkpoint`.

        Returns
        -------
        None

        Notes
        -----
        If :paramref:`path` does not exist or contains invalid JSON, logs
        a warning and leaves the current (empty) history unchanged,
        rather than raising — a missing checkpoint at the start of a run
        is treated as "no prior history" rather than an error.
        """
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "[MoEWatch] RulePolicy: failed to load checkpoint from "
                "'%s' (%s); starting with empty history.",
                path,
                exc,
            )
            return

        recent_actions = payload.get("recent_actions", {})
        self._recent_actions = {
            layer: deque(actions, maxlen=_HISTORY_LEN)
            for layer, actions in recent_actions.items()
        }
        self._action_sequence = dict(payload.get("action_sequence", {}))

        logger.info(
            "[MoEWatch] RulePolicy: loaded action history for %d layer(s) "
            "from '%s'.",
            len(self._recent_actions),
            path,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _risk_to_action_type(risk_score: float) -> str:
        """Map a risk score to a candidate action type via fixed thresholds."""
        if risk_score < _RISK_NOOP_MAX:
            return "noop"
        if risk_score < _RISK_AUXLOSS_MAX:
            return "aux_loss"
        if risk_score < _RISK_ROUTERNOISE_MAX:
            return "router_noise"
        return "expert_dropout"

    @staticmethod
    def _downgrade(action_type: str) -> str:
        """Return the next-weaker action type, or ``"noop"`` if already weakest."""
        idx = _ACTION_ORDER.index(action_type)
        if idx == 0:
            return action_type
        return _ACTION_ORDER[idx - 1]

    def _apply_guards(
        self, layer_name: str, history: Deque[str], candidate: str
    ) -> str:
        """Apply oscillation and cascade guards, returning the final action type.

        Parameters
        ----------
        layer_name : str
            Target layer (used only for logging by the caller).
        history : Deque[str]
            This layer's recent *final* action types, most recent last.
            Not mutated by this method.
        candidate : str
            The threshold-selected action type before any guard is
            applied.

        Returns
        -------
        str
            ``candidate``, or ``candidate`` downgraded by one tier if a
            cascade or oscillation pattern is detected.

        Notes
        -----
        - **Cascade guard**: if ``candidate`` is not ``"noop"`` and
          ``candidate`` appears at least :data:`_CASCADE_REPEAT_LIMIT`
          times in ``history`` (which has length at most
          :data:`_HISTORY_LEN`), the same strong action has fired
          repeatedly for this layer without apparent improvement;
          downgrade by one tier.
        - **Oscillation guard**: if ``history`` has at least 2 entries and
          the last entry equals ``candidate`` while the second-to-last
          entry differs from ``candidate`` — i.e. the policy is about to
          flip back to an action it just moved away from — downgrade by
          one tier to break the alternation.

        At most one downgrade is applied per call (guards are not
        re-checked against the downgraded value), keeping the policy's
        behaviour easy to reason about: a single call to
        :meth:`select_action` changes the intervention strength by at
        most one tier relative to the pure threshold mapping.
        """
        if candidate == "noop":
            return candidate

        # Cascade guard: same strong action repeating too often.
        if list(history).count(candidate) >= _CASCADE_REPEAT_LIMIT:
            return self._downgrade(candidate)

        # Oscillation guard: about to flip back to an action just left.
        if len(history) >= 2 and history[-1] == candidate and history[-2] != candidate:
            return self._downgrade(candidate)

        return candidate

    @staticmethod
    def _build_action(
        action_type: str,
        layer_name: str,
        max_delta: "float | None" = None,
    ) -> InterventionAction:
        """Construct the :class:`InterventionAction` for ``action_type``.

        Parameters
        ----------
        action_type : str
            One of ``"noop"``, ``"aux_loss"``, ``"router_noise"``,
            ``"expert_dropout"``.
        layer_name : str
            Target layer name passed to the action's constructor.
        max_delta : float or None, default=None
            Upper bound on the constructed action's magnitude, normally
            ``config.intervention_max_delta``. Each action type's
            built-in default magnitude (``0.05`` for
            :class:`~moewatch.intervention.actions.AuxLossAction``,
            ``0.1`` for :class:`~moewatch.intervention.actions.RouterNoiseAction`
            and :class:`~moewatch.intervention.actions.ExpertDropoutAction`)
            is capped to ``max_delta`` when it would otherwise exceed it.
            If ``None``, the built-in defaults are used unmodified — kept
            for backward-compatible direct calls that don't have a config
            to consult.

        Returns
        -------
        InterventionAction
            A freshly constructed action instance with magnitude
            parameters that respect ``max_delta``.

        Notes
        -----
        Without this clamp, :class:`~moewatch.intervention.safety.SafetyGuard`'s
        delta-limit check would compare the action's hardcoded default
        magnitude against ``config.intervention_max_delta`` and, whenever
        the configured cap is tighter than the default (e.g. a fine-tuning
        config that sets ``intervention_max_delta=0.01``), *every* non-NoOp
        action would be downgraded to a NoOp regardless of risk score —
        silently disabling the entire intervention system. Clamping here
        ensures ``intervention_max_delta`` actually controls the applied
        magnitude instead of being a hard veto.
        """

        def _capped(default: float) -> float:
            return min(default, max_delta) if max_delta is not None else default

        if action_type == "aux_loss":
            return AuxLossAction(layer_name=layer_name, delta=_capped(0.05))
        if action_type == "router_noise":
            return RouterNoiseAction(layer_name=layer_name, noise_scale=_capped(0.1))
        if action_type == "expert_dropout":
            return ExpertDropoutAction(layer_name=layer_name, dropout_delta=_capped(0.1))
        return NoOpAction(layer_name=layer_name)
