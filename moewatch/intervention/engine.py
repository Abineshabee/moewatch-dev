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
#                      reward is negative (a reward of exactly zero is
#                      treated as neutral and the action is kept), and
#                      feeds the
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
       revert on negative reward (zero counts as neutral, not reverted),
       and update the policy.

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

        # layer_name -> (action, applied_step). Interventions that completed
        # their observation window with reward > 0 and were intentionally
        # kept in the model. The engine retains the action object so it can
        # revert the hook/change when the same layer is intervened on again.
        # Without this dict the action handle is orphaned after a successful
        # window: the hook stays installed in the model forever while the
        # engine forgets it existed, breaking the "at most one intervention
        # per layer" invariant and stacking hooks on every new cycle.
        self._persistent_interventions: Dict[str, Tuple[InterventionAction, int]] = {}

        # Staging area for persistent actions that have been validated by safety
        # and are waiting to be atomically reverted inside apply_intervention().
        # propose_intervention() parks old persistent actions here after safety
        # passes. If apply_intervention() raises, the rescue path restores the
        # entry to _persistent_interventions so the model is never left with
        # neither the old nor the new intervention installed.
        self._pending_reverts: Dict[str, Tuple[InterventionAction, int]] = {}

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
        active or persistent intervention of the same ``action_type``,
        the new action is logged as an ``"accumulated"`` event but is
        **not** downgraded or specially applied here — it is tracked as
        its own independent owned intervention via the normal
        staging/apply path, so it gets its own observation window and can
        later be reverted or made persistent on its own. This relies on
        the action's :meth:`~moewatch.intervention.actions.InterventionAction.revert`
        removing only its own contribution from the shared field (as
        :class:`~moewatch.intervention.actions.AuxLossAction` does) rather
        than resetting to an absolute pre-apply snapshot, so two
        independent owners on the same global resource cannot erase each
        other's contribution.
        """
        action.mark_applied(step)

        if not isinstance(action, NoOpAction):
            # A layer under active observation cannot accept a second
            # intervention — block it and let the current window expire first.
            if action.layer_name in self._active_interventions:
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

        # Run SafetyGuard FIRST — before any model mutations.
        # Previously the global-resource accumulation block ran before this
        # check, calling existing_action.apply() and action.apply() on the
        # live model and then returning NoOp — bypassing SafetyGuard entirely
        # for that code path. propose_intervention() is the *proposal* phase;
        # all model mutations now happen only after safety validation passes.
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
            # Safety rejected — persistent intervention (if any) is left
            # completely untouched (earlier fix: revert only after safety
            # passes). Return without mutating the model at all.
            return result.recommended_action

        # Safety PASSED. Now safe to perform model mutations.

        if not isinstance(action, NoOpAction) and action.is_global_resource:
            conflicting_layer = self._find_conflicting_global_intervention(action)
            if conflicting_layer is not None:
                # A different layer already holds an active or persistent
                # intervention on the same global resource. We used to
                # apply() both actions here and discard this one as a NoOp,
                # because no owner was ever recorded for its delta — nobody
                # would call revert() on it, so on a long run the shared
                # config field could drift upward forever, invisibly, across
                # unrelated safety windows/cooldowns/rollback (see engine
                # history for the "orphaned accumulation" bug this caused).
                #
                # AuxLossAction.revert() now subtracts only its OWN delta
                # from whatever value is currently on the config, instead of
                # resetting to an absolute pre-apply snapshot (see
                # AuxLossAction.revert() docstring) — so two independent
                # owners on the same global resource no longer risk erasing
                # each other's contribution when one reverts before the
                # other. That means this action can now be treated like any
                # other proposal: it is NOT special-cased or mutated here;
                # it falls through to the normal staging/apply path below,
                # so apply_intervention() registers it as a real active
                # intervention (own observation window, own eventual
                # revert-or-persist decision, own ownership). Its apply()
                # will naturally add its delta on top of whatever the
                # conflicting layer's action already put on the shared
                # config field — no manual pre-application needed.
                logger.info(
                    "[MoEWatch] InterventionEngine: '%s' on layer '%s' "
                    "will accumulate onto existing '%s' intervention owned "
                    "by layer '%s' (global resource shared); tracking as "
                    "its own owned intervention so it can later be "
                    "reverted or made persistent independently.",
                    action.action_type,
                    action.layer_name,
                    action.action_type,
                    conflicting_layer,
                )
                self._intervention_log.append(
                    {
                        "event": "accumulated",
                        "step": step,
                        "layer": action.layer_name,
                        "onto_layer": conflicting_layer,
                        "action": action.action_type,
                        "delta": action.delta,
                    }
                )
                # Falls through — no early return. action is staged/applied
                # via the normal path below and by apply_intervention().

        # Safety PASSED and no global-resource conflict.
        # Stage the existing persistent intervention (if any) for atomic
        # revert-then-apply inside apply_intervention(). The revert is NOT done
        # here — propose_intervention() is the validation phase and must not
        # touch the model for the replacement case. If apply_intervention()
        # subsequently raises, it restores the entry to _persistent_interventions
        # so the model is never left with no active intervention at all.
        if not isinstance(action, NoOpAction):
            if action.layer_name in self._persistent_interventions:
                staged_action, staged_step = self._persistent_interventions.pop(
                    action.layer_name
                )
                self._pending_reverts[action.layer_name] = (staged_action, staged_step)
                # Log the intent here so callers and tests can observe it at
                # proposal time. The actual model.revert() call happens inside
                # apply_intervention() for atomicity.
                logger.info(
                    "[MoEWatch] InterventionEngine: staged revert of persistent %s "
                    "(applied step %d) on layer '%s'; revert executes in "
                    "apply_intervention() for atomicity.",
                    staged_action.log(),
                    staged_step,
                    action.layer_name,
                )
                self._intervention_log.append(
                    {
                        "event": "persistent_reverted",
                        "step": step,
                        "layer": action.layer_name,
                        "reverted_action": staged_action.action_type,
                        "reverted_applied_step": staged_step,
                    }
                )

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
        # Atomic replacement: if propose_intervention staged an old persistent
        # action for this layer (because safety passed and we're replacing it),
        # revert it and apply the new one in a single block. If action.apply()
        # raises, the old persistent action is restored to _persistent_interventions
        # so the engine's state stays consistent and the model is not left with
        # no active intervention.
        pending = self._pending_reverts.pop(action.layer_name, None)
        if pending is not None:
            old_action, old_step = pending
            old_action.revert(self.model)
            try:
                action.apply(self.model)
            except Exception as apply_error:
                # New action failed — the old action was already reverted from
                # the model above, so at this point the model has NO
                # intervention installed on this layer. Restoring only the
                # bookkeeping dict (_persistent_interventions) is not enough:
                # the model and the engine's records would disagree about
                # what's actually installed. Re-apply the old action to the
                # model itself so state and model stay consistent, then
                # restore the bookkeeping entry to match.
                try:
                    old_action.apply(self.model)
                except Exception as restore_error:
                    logger.critical(
                        "[MoEWatch] InterventionEngine: failed to restore old "
                        "intervention %s on layer '%s' after replacement "
                        "failure; model has NO intervention installed on "
                        "this layer.",
                        old_action.log(),
                        action.layer_name,
                    )
                    raise RuntimeError(
                        "Replacement of persistent intervention on layer "
                        f"'{action.layer_name}' failed and the old "
                        "intervention could not be restored to the model"
                    ) from restore_error

                self._persistent_interventions[action.layer_name] = (old_action, old_step)
                logger.warning(
                    "[MoEWatch] InterventionEngine: apply_intervention raised "
                    "while replacing persistent %s on layer '%s'; "
                    "old intervention re-applied to the model and restored "
                    "to _persistent_interventions.",
                    old_action.log(),
                    action.layer_name,
                )
                raise apply_error
            self._intervention_log.append(
                {
                    "event": "persistent_reverted",
                    "step": step,
                    "layer": action.layer_name,
                    "reverted_action": old_action.action_type,
                    "reverted_applied_step": old_step,
                }
            )
        else:
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
        3. If ``reward < 0.0``: the action is reverted via
           :meth:`InterventionAction.revert`, and the outcome is logged as
           ``"failure"``. ``reward == 0.0`` is the neutral/invalid-baseline
           case (step 1) and does **not** trigger a revert.
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

            # Risk at the moment the intervention was applied (from the
            # PolicyState recorded in apply_intervention). Used as a
            # secondary signal when the linear clean-baseline projection
            # is unreliable under a sustained distribution shift: the
            # pre-pressure baseline stays low, so even a clearly helpful
            # intervention that leaves residual risk (e.g. router noise
            # fighting a held gate-bias of 2.0) can score reward < 0 and
            # be incorrectly reverted — producing the L2 entropy spikes
            # seen in the Base-comparison demo.
            pending_state = self._pending_states.get(layer_name)
            risk_at_apply = (
                float(pending_state.risk_score)
                if pending_state is not None
                else actual_risk
            )
            improved_vs_apply = actual_risk < risk_at_apply - 1e-6

            if reward < 0.0 and not improved_vs_apply:
                # Counterfactual says no help AND risk did not fall since
                # apply — genuine failure. Revert.
                action.revert(self.model)
                outcome = "failure"
                logger.info(
                    "[MoEWatch] InterventionEngine: %s at step %d "
                    "(applied step %d) -> reward=%.6f "
                    "risk_at_apply=%.4f actual=%.4f; reverted.",
                    action.log(),
                    step,
                    applied_step,
                    reward,
                    risk_at_apply,
                    actual_risk,
                )
            else:
                # success: positive/neutral counterfactual reward, OR risk
                # improved vs apply time despite a misleading baseline
                # projection under regime shift.
                outcome = "success"
                self._persistent_interventions[layer_name] = (action, applied_step)
                logger.info(
                    "[MoEWatch] InterventionEngine: %s at step %d "
                    "(applied step %d) -> reward=%.6f "
                    "risk_at_apply=%.4f actual=%.4f; kept (persistent)%s.",
                    action.log(),
                    step,
                    applied_step,
                    reward,
                    risk_at_apply,
                    actual_risk,
                    " [improved-vs-apply override]" if (reward < 0.0 and improved_vs_apply) else "",
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
        # Search both active and persistent interventions.
        # A persistent AuxLossAction is still installed in the model and
        # still owns the shared resource — a new proposal from a different
        # layer must see it as a conflict.  Previously only
        # _active_interventions was checked, so a persistent AuxLossAction
        # was invisible and a second layer could start a competing one,
        # both modifying the same router_aux_loss_coef field.
        all_owned: Dict[str, Tuple[InterventionAction, int]] = {
            **self._persistent_interventions,
            **self._active_interventions,  # active takes precedence on key clash
        }
        for other_layer, (other_action, _) in all_owned.items():
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

        Returns ``True`` for layers under observation
        (:attr:`_active_interventions`) **and** for layers whose previous
        intervention succeeded and is still installed in the model
        (:attr:`_persistent_interventions`).

        Parameters
        ----------
        layer_name : str
            Layer to check.

        Returns
        -------
        bool
            ``True`` if ``layer_name`` is present in either
            :attr:`_active_interventions` or :attr:`_persistent_interventions`.
        """
        return (
            layer_name in self._active_interventions
            or layer_name in self._persistent_interventions
        )

    def revert_all(self) -> List[str]:
        """Revert every currently-installed intervention and clear bookkeeping.

        Reverts every action in :attr:`_active_interventions` and
        :attr:`_persistent_interventions` (via
        :meth:`InterventionAction.revert`), then clears
        :attr:`_active_interventions`, :attr:`_persistent_interventions`,
        :attr:`_observation_windows`, :attr:`_pending_states`, and
        :attr:`_pending_reverts`.

        Intended for use when monitoring is being torn down (e.g.
        :meth:`MoEWatch.stop`) so that no intervention is left modifying
        model execution — a forward hook still installed, a
        ``model.config`` field still shifted, a dropout probability still
        raised — after the engine that owns it has stopped observing and
        can no longer revert it on a negative reward.

        A single action's :meth:`~moewatch.intervention.actions.InterventionAction.revert`
        raising is logged and does not prevent the remaining actions from
        being reverted.

        Returns
        -------
        list[str]
            Layer names whose intervention was successfully reverted.
        """
        reverted: List[str] = []
        # A layer can appear in both dicts only transiently; in steady state
        # each layer is in at most one. Iterate over a combined, deduplicated
        # view so a persistent-only layer (no pending observation window)
        # is not skipped.
        all_layers = {**self._persistent_interventions, **self._active_interventions}

        for layer_name, (action, _applied_step) in all_layers.items():
            try:
                action.revert(self.model)
                reverted.append(layer_name)
                logger.info(
                    "[MoEWatch] InterventionEngine.revert_all(): reverted "
                    "%s on layer '%s'.",
                    action.log(),
                    layer_name,
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.error(
                    "[MoEWatch] InterventionEngine.revert_all(): failed to "
                    "revert %s on layer '%s': %s",
                    action.log(),
                    layer_name,
                    exc,
                )

        self._active_interventions.clear()
        self._persistent_interventions.clear()
        self._observation_windows.clear()
        self._pending_states.clear()
        self._pending_reverts.clear()

        return reverted

    def __repr__(self) -> str:
        return (
            f"InterventionEngine(active={len(self._active_interventions)}, "
            f"persistent={len(self._persistent_interventions)}, "
            f"pending_windows={len(self._observation_windows)}, "
            f"log_entries={len(self._intervention_log)})"
        )
