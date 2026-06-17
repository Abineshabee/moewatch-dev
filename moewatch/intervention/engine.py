# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# moewatch/intervention/engine.py [v0.2.0]
# =============================================================================
#
# Project      : MoEWatch
# Version      : v0.2.0
# Description  : Orchestrates the full intervention lifecycle for live MoE
#                 training. Given an action proposed by the active policy,
#                 InterventionEngine:
#
#                   1. Validates the action against SafetyGuard.
#                   2. Applies the (possibly downgraded) action to the
#                      model being trained.
#                   3. Marks the affected layer's intervention-exclusion
#                      window in BaselineTracker, to prevent baseline
#                      contamination.
#                   4. Schedules an observation window
#                      (``config.reward_window_steps`` steps).
#                   5. Once the window expires, computes a counterfactual
#                      reward via BaselineTracker, reverts the action if the
#                      reward is non-positive, and feeds the
#                      ``(state, action, reward)`` tuple back to the active
#                      policy via :meth:`PolicyBase.update`.
#
#                 At most one intervention is active per layer at a time.
#                 All interventions and outcomes are recorded in
#                 :attr:`InterventionEngine._intervention_log` for
#                 auditability.
#
# Author       : Abinesh N (@Abineshabee)
# Repository   : https://github.com/Abineshabee/MoEWatch
# License      : Apache 2.0
#
# Contents
# --------
#   InterventionEngine — applies, tracks, and evaluates interventions
#
# Usage
# -----
#   from moewatch.intervention.engine import InterventionEngine
#
#   engine = InterventionEngine(config, model, baseline_tracker)
#
#   validated = engine.propose_intervention(
#       action, current_loss, risk_scores, layer_order, step
#   )
#   engine.apply_intervention(validated, step)
#   ...
#   engine.check_observation_windows(step, risk_scores, policy)
#
# =============================================================================

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from moewatch.collector.baseline_tracker import BaselineTracker
from moewatch.config import WatchConfig
from moewatch.intervention.actions import InterventionAction, NoOpAction
from moewatch.intervention.safety import SafetyGuard

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance
    from moewatch.policy.base import PolicyBase, PolicyState

logger = logging.getLogger(__name__)


class InterventionEngine:
    """Applies interventions to running MoE training and evaluates outcomes.

    Coordinates :class:`~moewatch.intervention.safety.SafetyGuard` (pre-flight
    validation), :class:`~moewatch.collector.baseline_tracker.BaselineTracker`
    (counterfactual baseline / exclusion windows), and the active
    :class:`~moewatch.policy.base.PolicyBase` implementation (feedback loop).

    Workflow
    --------
    1. :meth:`propose_intervention` — validate a policy-selected action with
       :class:`~moewatch.intervention.safety.SafetyGuard`; returns the
       original action or a downgraded
       :class:`~moewatch.intervention.actions.NoOpAction`.
    2. :meth:`apply_intervention` — apply the validated action to
       :attr:`model`, record it as active, mark the baseline-exclusion
       window, and schedule an observation window.
    3. :meth:`check_observation_windows` — called every step; for each
       expired observation window, compute the counterfactual reward,
       revert on non-positive reward, and update the policy.

    Parameters
    ----------
    config : WatchConfig
        Shared configuration. Uses ``config.reward_window_steps``.
    model : torch.nn.Module
        The model being trained. Passed through to
        :meth:`InterventionAction.apply` / :meth:`InterventionAction.revert`.
    baseline_tracker : BaselineTracker
        Per-layer counterfactual baseline tracker, shared with
        :class:`~moewatch.analyzer.risk_score.RiskScoreFuser` consumers and
        the rest of :class:`~moewatch._watcher.MoEWatch`.

    Attributes
    ----------
    config : WatchConfig
        See above.
    model : torch.nn.Module
        See above.
    safety_guard : SafetyGuard
        Pre-flight safety validator, constructed internally.
    baseline_tracker : BaselineTracker
        See above.
    """

    def __init__(
        self,
        config: WatchConfig,
        model: Any,
        baseline_tracker: BaselineTracker,
    ) -> None:
        self.config: WatchConfig = config
        self.model: Any = model
        self.safety_guard: SafetyGuard = SafetyGuard(config)
        self.baseline_tracker: BaselineTracker = baseline_tracker

        # layer_name -> (action, applied_step). At most one active
        # intervention per layer.
        self._active_interventions: Dict[str, Tuple[InterventionAction, int]] = {}

        # layer_name -> (start_step, end_step) observation window.
        self._observation_windows: Dict[str, Tuple[int, int]] = {}

        # layer_name -> PolicyState recorded at the time the intervention
        # was proposed, needed to call policy.update() once the
        # observation window expires.
        self._pending_states: Dict[str, "PolicyState"] = {}

        # Full history of intervention lifecycle events, for debugging and
        # auditing. Each entry is a dict with at least "event" and "step"
        # keys.
        self._intervention_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Step 1: propose / validate
    # ------------------------------------------------------------------

    def propose_intervention(
        self,
        action: InterventionAction,
        current_loss: float,
        risk_scores: Dict[str, float],
        layer_order: List[str],
        step: int,
    ) -> InterventionAction:
        """Validate a policy-selected action with :class:`SafetyGuard`.

        Parameters
        ----------
        action : InterventionAction
            Action proposed by the active policy.
        current_loss : float
            Training loss observed at ``step``.
        risk_scores : dict[str, float]
            Mapping of layer name to current collapse risk score.
        layer_order : list[str]
            Layer names in model order (used for the neighbor check).
        step : int
            Current training step.

        Returns
        -------
        InterventionAction
            ``action`` unchanged if all safety checks pass, otherwise a
            :class:`~moewatch.intervention.actions.NoOpAction` targeting
            the same layer.

        Notes
        -----
        Sets ``action.applied_step = step`` (via
        :meth:`InterventionAction.mark_applied`) *before* invoking
        :class:`SafetyGuard`, since the cooldown check compares against
        the proposed application step. If the action is downgraded, the
        downgrade reason(s) are logged via
        :class:`SafetyGuard` and recorded in :attr:`_intervention_log`.

        If a layer already has an active intervention (present in
        :attr:`_active_interventions`), the proposed action is downgraded
        to :class:`~moewatch.intervention.actions.NoOpAction` regardless of
        :class:`SafetyGuard`'s verdict — at most one intervention is active
        per layer at a time.

        Separately, if ``action.is_global_resource`` is ``True`` (e.g.
        :class:`~moewatch.intervention.actions.AuxLossAction`, which
        mutates a single shared ``model.config`` field rather than its
        own layer's submodule) and a *different* layer already has an
        active intervention of the same ``action_type``, the proposed
        action is likewise downgraded to NoOp — even though the two
        layers are otherwise unrelated, allowing both to be active
        simultaneously would mean two independent actions mutating the
        same underlying field, each tracking its own pre-apply snapshot;
        whichever reverts first would restore the field to *its* snapshot
        and silently erase the other's still-active contribution.
        """
        action.mark_applied(step)

        if (
            not isinstance(action, NoOpAction)
            and action.layer_name in self._active_interventions
        ):
            logger.info(
                "[MoEWatch] InterventionEngine: layer '%s' already has an "
                "active intervention; downgrading %s to NoOp.",
                action.layer_name,
                action.log(),
            )
            self._intervention_log.append(
                {
                    "event": "downgraded",
                    "step": step,
                    "layer": action.layer_name,
                    "reason": "layer already has an active intervention",
                    "original_action": action.action_type,
                }
            )
            downgraded = NoOpAction(layer_name=action.layer_name)
            downgraded.mark_applied(step)
            return downgraded

        if not isinstance(action, NoOpAction) and action.is_global_resource:
            conflicting_layer = self._find_conflicting_global_intervention(action)
            if conflicting_layer is not None:
                logger.info(
                    "[MoEWatch] InterventionEngine: '%s' on layer '%s' "
                    "targets a global resource already controlled by an "
                    "active '%s' intervention on layer '%s'; downgrading "
                    "to NoOp to avoid corrupting the shared field.",
                    action.action_type,
                    action.layer_name,
                    action.action_type,
                    conflicting_layer,
                )
                reason = (
                    f"global resource for '{action.action_type}' already "
                    f"has an active intervention on layer '{conflicting_layer}'"
                )
                self._intervention_log.append(
                    {
                        "event": "downgraded",
                        "step": step,
                        "layer": action.layer_name,
                        "reason": reason,
                        "original_action": action.action_type,
                    }
                )
                downgraded = NoOpAction(layer_name=action.layer_name)
                downgraded.mark_applied(step)
                return downgraded

        result = self.safety_guard.check(action, current_loss, risk_scores, layer_order)

        if not result.passed:
            self._intervention_log.append(
                {
                    "event": "downgraded",
                    "step": step,
                    "layer": action.layer_name,
                    "reason": "; ".join(result.failures),
                    "original_action": action.action_type,
                }
            )
            result.recommended_action.mark_applied(step)

        return result.recommended_action

    # ------------------------------------------------------------------
    # Step 2: apply
    # ------------------------------------------------------------------

    def apply_intervention(
        self,
        action: InterventionAction,
        step: int,
        state: "PolicyState | None" = None,
    ) -> None:
        """Apply ``action`` to live training and record bookkeeping state.

        Parameters
        ----------
        action : InterventionAction
            Action to apply, typically the return value of
            :meth:`propose_intervention` (already validated / possibly
            downgraded).
        step : int
            Current training step.
        state : PolicyState or None, default=None
            Policy state at the time this intervention was selected.
            Stored so :meth:`check_observation_windows` can later call
            :meth:`PolicyBase.update` with the matching state. Required for
            non-NoOp actions if the active policy's :meth:`update` is to be
            called when the observation window expires; if omitted, the
            window will still be checked and the action reverted on
            negative reward, but no ``policy.update`` call will be made for
            it.

        Returns
        -------
        None

        Notes
        -----
        :class:`~moewatch.intervention.actions.NoOpAction` instances are
        applied (a no-op) and logged, but do **not** register an active
        intervention, baseline-exclusion window, or observation window —
        there is nothing to observe or revert.

        For non-NoOp actions:

        - ``action.apply(self.model)`` is called.
        - The action is recorded in :attr:`_active_interventions` as
          ``(action, step)``.
        - :meth:`SafetyGuard.record_intervention` is called so future
          cooldown checks see this intervention.
        - :meth:`BaselineTracker.mark_intervention` is called to exclude
          the upcoming ``config.reward_window_steps`` steps from this
          layer's clean baseline history.
        - An observation window
          ``(step, step + config.reward_window_steps)`` is recorded in
          :attr:`_observation_windows`.
        """
        action.apply(self.model)

        self._intervention_log.append(
            {
                "event": "applied",
                "step": step,
                "layer": action.layer_name,
                "action": action.action_type,
                "delta": action.delta,
            }
        )

        if isinstance(action, NoOpAction):
            logger.debug(
                "[MoEWatch] InterventionEngine: applied NoOp at step %d "
                "(layer='%s').",
                step,
                action.layer_name,
            )
            return

        self._active_interventions[action.layer_name] = (action, step)
        self.safety_guard.record_intervention(action.layer_name, step)
        self.baseline_tracker.mark_intervention(action.layer_name, start_step=step)

        window_end = step + self.config.reward_window_steps
        self._observation_windows[action.layer_name] = (step, window_end)

        if state is not None:
            self._pending_states[action.layer_name] = state

        logger.info(
            "[MoEWatch] InterventionEngine: applied %s at step %d; "
            "observation window=[%d, %d).",
            action.log(),
            step,
            step,
            window_end,
        )

    # ------------------------------------------------------------------
    # Step 3: check expired observation windows
    # ------------------------------------------------------------------

    def check_observation_windows(
        self,
        step: int,
        risk_scores: Dict[str, float],
        policy: "PolicyBase",
    ) -> None:
        """Resolve expired observation windows, reverting and updating policy.

        Parameters
        ----------
        step : int
            Current training step.
        risk_scores : dict[str, float]
            Mapping of layer name to current collapse risk score, used as
            the "actual" signal value for counterfactual reward
            computation.
        policy : PolicyBase
            Active policy. :meth:`PolicyBase.update` is called for any
            expired window whose layer has a recorded pending
            :class:`~moewatch.policy.base.PolicyState`
            (see :meth:`apply_intervention`).

        Returns
        -------
        None

        Notes
        -----
        For each layer with an observation window
        ``(start_step, end_step)`` where ``step >= end_step``:

        1. The risk score for that layer (``risk_scores.get(layer, ...)``)
           is compared against the counterfactual baseline via
           :meth:`BaselineTracker.compute_counterfactual_delta`. If the
           baseline is not yet valid for this layer
           (:meth:`BaselineTracker.is_baseline_valid` is ``False``), the
           reward defaults to ``0.0`` (treated as neutral — neither
           success nor failure) and the action is **not** reverted on this
           basis alone.
        2. The reward is computed as ``baseline_projected - actual_risk``:
           since lower risk is healthier, a *positive* reward means the
           observed risk score is *below* what the no-intervention
           baseline trajectory would predict (the intervention helped); a
           *negative* reward means risk is at or above the baseline
           projection (the intervention did not help, or risk worsened).
        3. If ``reward <= 0.0``: the action is reverted via
           :meth:`InterventionAction.revert`, and the outcome is logged as
           ``"failure"``.
        4. If ``reward > 0.0``: the action is left in place, and the
           outcome is logged as ``"success"``.
        5. If a pending :class:`~moewatch.policy.base.PolicyState` was
           recorded for this layer (see :meth:`apply_intervention`),
           ``policy.update(state, action, reward)`` is called.
        6. The layer's entries are removed from
           :attr:`_active_interventions`, :attr:`_observation_windows`, and
           :attr:`_pending_states`.

        Layers whose risk score is missing from ``risk_scores`` are skipped
        for this call (their window remains pending and will be re-checked
        on a subsequent call) — this avoids prematurely resolving a window
        based on incomplete data, e.g. for layers temporarily absent from a
        partial forward pass.
        """
        expired_layers = [
            layer_name
            for layer_name, (_, window_end) in self._observation_windows.items()
            if step >= window_end
        ]

        for layer_name in expired_layers:
            if layer_name not in risk_scores:
                logger.debug(
                    "[MoEWatch] InterventionEngine: risk score for '%s' "
                    "unavailable at step %d; deferring window resolution.",
                    layer_name,
                    step,
                )
                continue

            action, applied_step = self._active_interventions[layer_name]
            actual_risk = risk_scores[layer_name]

            reward = self._compute_reward(layer_name, actual_risk)

            if reward <= 0.0:
                action.revert(self.model)
                outcome = "failure"
                logger.info(
                    "[MoEWatch] InterventionEngine: %s at step %d "
                    "(applied step %d) -> reward=%.6f; reverted.",
                    action.log(),
                    step,
                    applied_step,
                    reward,
                )
            else:
                outcome = "success"
                logger.info(
                    "[MoEWatch] InterventionEngine: %s at step %d "
                    "(applied step %d) -> reward=%.6f; kept.",
                    action.log(),
                    step,
                    applied_step,
                    reward,
                )

            self._intervention_log.append(
                {
                    "event": "resolved",
                    "step": step,
                    "layer": layer_name,
                    "action": action.action_type,
                    "applied_step": applied_step,
                    "reward": reward,
                    "outcome": outcome,
                }
            )

            state = self._pending_states.get(layer_name)
            if state is not None:
                policy.update(state, action, reward)

            self._active_interventions.pop(layer_name, None)
            self._observation_windows.pop(layer_name, None)
            self._pending_states.pop(layer_name, None)

    def _find_conflicting_global_intervention(
        self, action: InterventionAction
    ) -> "str | None":
        """Find another layer with an active intervention of the same type.

        Parameters
        ----------
        action : InterventionAction
            Proposed action, with ``action.is_global_resource is True``.

        Returns
        -------
        str or None
            The ``layer_name`` of another layer currently holding an
            active intervention with the same ``action_type`` as
            ``action``, or ``None`` if no such conflict exists.

        Notes
        -----
        Only relevant for actions that mutate shared/global model state
        (see :attr:`InterventionAction.is_global_resource`). Per-layer
        actions never conflict this way, since each targets its own
        distinct submodule.
        """
        for other_layer, (other_action, _) in self._active_interventions.items():
            if (
                other_layer != action.layer_name
                and other_action.action_type == action.action_type
            ):
                return other_layer
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_reward(self, layer_name: str, actual_risk: float) -> float:
        """Compute the counterfactual reward for an expired observation window.

        Parameters
        ----------
        layer_name : str
            Layer whose intervention is being evaluated.
        actual_risk : float
            Observed risk score for ``layer_name`` at window expiry.

        Returns
        -------
        float
            ``baseline_projected - actual_risk``, or ``0.0`` (neutral) if
            :meth:`BaselineTracker.is_baseline_valid` is ``False`` for
            ``layer_name``.

        Notes
        -----
        Lower risk scores are healthier, so a positive result indicates
        the intervention outperformed the no-intervention counterfactual.
        """
        if not self.baseline_tracker.is_baseline_valid(layer_name):
            return 0.0

        # compute_counterfactual_delta() returns (actual - baseline); we
        # want (baseline - actual) since lower risk is "better" here, so
        # negate the result.
        delta = self.baseline_tracker.compute_counterfactual_delta(
            layer_name, actual_risk
        )
        return -delta

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_intervention_log(self) -> List[Dict[str, Any]]:
        """Return a copy of the full intervention lifecycle log.

        Returns
        -------
        list[dict]
            Shallow copy of :attr:`_intervention_log`. Each entry is a
            dict describing a single lifecycle event (``"applied"``,
            ``"downgraded"``, or ``"resolved"``) with at least ``"event"``
            and ``"step"`` keys.
        """
        return list(self._intervention_log)

    def has_active_intervention(self, layer_name: str) -> bool:
        """Check whether ``layer_name`` currently has an active intervention.

        Parameters
        ----------
        layer_name : str
            Layer to check.

        Returns
        -------
        bool
            ``True`` if ``layer_name`` is present in
            :attr:`_active_interventions`.
        """
        return layer_name in self._active_interventions

    def __repr__(self) -> str:
        return (
            f"InterventionEngine(active={len(self._active_interventions)}, "
            f"pending_windows={len(self._observation_windows)}, "
            f"log_entries={len(self._intervention_log)})"
        )
