# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# moewatch/intervention/safety.py [v0.2.0]
# =============================================================================
#
# Project      : MoEWatch
# Version      : v0.2.0
# Description  : Pre-flight safety validation for the intervention engine.
#                 Before any proposed InterventionAction is applied to live
#                 training, SafetyGuard runs four independent checks:
#
#                   1. Cooldown      — minimum spacing between consecutive
#                                       interventions on the same layer.
#                   2. Delta limit   — bounded intervention magnitude.
#                   3. Loss guard    — freeze all interventions if training
#                                       loss is spiking relative to baseline.
#                   4. Neighbor check — adjacent layers must not themselves
#                                       be at critical risk (avoid stacking
#                                       interventions on a locally unstable
#                                       region of the model).
#
#                 If any check fails, the proposed action is *downgraded* to
#                 a NoOpAction rather than raising an error or aborting
#                 training. This makes SafetyGuard the final backstop that
#                 guarantees interventions can never destabilize training
#                 beyond configured bounds.
#
# Author       : Abinesh N (@Abineshabee)
# Repository   : https://github.com/Abineshabee/MoEWatch
# License      : Apache 2.0
#
# Contents
# --------
#   SafetyCheckResult   — result dataclass returned by SafetyGuard.check()
#   SafetyGuard          — pre-flight safety validator
#
# Usage
# -----
#   from moewatch.intervention.safety import SafetyGuard
#
#   guard = SafetyGuard(config)
#   guard.update_baseline_loss(initial_loss)
#   ...
#   result = guard.check(action, current_loss, risk_scores, layer_order)
#   if not result.passed:
#       action_to_apply = result.recommended_action  # NoOpAction
#
# =============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

from moewatch.config import WatchConfig
from moewatch.intervention.actions import InterventionAction, NoOpAction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SafetyCheckResult
# ---------------------------------------------------------------------------


@dataclass
class SafetyCheckResult:
    """Result of running all :class:`SafetyGuard` pre-flight checks.

    Attributes
    ----------
    passed : bool
        ``True`` if every check passed and ``recommended_action`` is the
        original proposed action unchanged. ``False`` if one or more
        checks failed and ``recommended_action`` has been downgraded to a
        :class:`~moewatch.intervention.actions.NoOpAction`.
    failures : list[str]
        Human-readable descriptions of each failed check, empty if
        ``passed`` is ``True``. Possible entries include strings such as
        ``"cooldown: 42 < 200 steps since last intervention on layer X"``.
    recommended_action : InterventionAction
        The action :class:`~moewatch.intervention.engine.InterventionEngine`
        should actually apply: either the original proposed action
        (``passed=True``) or a
        :class:`~moewatch.intervention.actions.NoOpAction`
        (``passed=False``).
    """

    passed: bool
    failures: List[str] = field(default_factory=list)
    recommended_action: InterventionAction = field(
        default_factory=lambda: NoOpAction()
    )


# ---------------------------------------------------------------------------
# SafetyGuard
# ---------------------------------------------------------------------------


class SafetyGuard:
    """Pre-flight safety validation for interventions.

    Enforces four independent checks before any
    :class:`~moewatch.intervention.actions.InterventionAction` is applied
    to live training:

    - **Cooldown** — at least ``config.intervention_cooldown`` training
      steps must have elapsed since the last intervention on the same
      layer.
    - **Delta limit** — ``action.delta`` must not exceed
      ``config.intervention_max_delta``.
    - **Loss guard** — the current training loss must not exceed
      ``config.loss_guard_threshold`` times the recorded baseline loss
      (a global freeze: a loss spike on *any* layer disables
      interventions on *all* layers until loss recovers).
    - **Neighbor check** — the immediately adjacent layers (by position in
      ``layer_order``) must not themselves be at ``CRITICAL`` risk; an
      intervention on a layer surrounded by already-unstable neighbours is
      considered too risky and is downgraded.

    If *any* check fails, :meth:`check` returns a
    :class:`SafetyCheckResult` with ``passed=False`` and
    ``recommended_action`` set to a
    :class:`~moewatch.intervention.actions.NoOpAction` — interventions are
    downgraded, never aborted with an exception.

    Parameters
    ----------
    config : WatchConfig
        Shared configuration. Uses ``config.intervention_cooldown``,
        ``config.intervention_max_delta``, and
        ``config.loss_guard_threshold``.

    Attributes
    ----------
    config : WatchConfig
        Reference to the shared configuration.
    """

    #: Risk score threshold above which a neighbouring layer is considered
    #: "CRITICAL" for the purposes of the neighbor check. Mirrors the
    #: CRITICAL bucket boundary used by
    #: :class:`~moewatch.analyzer.risk_score.RiskScoreFuser`
    #: (``score >= 0.8``).
    _NEIGHBOR_CRITICAL_THRESHOLD: float = 0.8

    def __init__(self, config: WatchConfig) -> None:
        self.config: WatchConfig = config

        # layer_name -> training step of the most recent intervention.
        self._intervention_history: Dict[str, int] = {}

        # Reference training loss for spike detection. ``None`` until
        # update_baseline_loss() is first called.
        self._loss_baseline: float | None = None

        # layer_name -> most recently observed risk score (updated by
        # check(), used for neighbor lookups on subsequent calls).
        self._risk_scores: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Baseline loss tracking
    # ------------------------------------------------------------------

    def update_baseline_loss(self, loss: float) -> None:
        """Update the reference training loss for spike detection.

        Parameters
        ----------
        loss : float
            Current training loss. Typically called once at the start of
            training to establish the baseline, and may be called
            periodically thereafter to let the baseline track a slowly
            improving loss curve.

        Returns
        -------
        None

        Notes
        -----
        Non-finite values (``NaN``/``inf``) are ignored with a logged
        warning, since they would make the loss-guard check meaningless
        (any finite ``current_loss`` would either always or never trip the
        spike condition).
        """
        if loss != loss or loss in (float("inf"), float("-inf")):  # NaN / inf
            logger.warning(
                "[MoEWatch] SafetyGuard: ignoring non-finite baseline loss "
                "value (%r).",
                loss,
            )
            return

        self._loss_baseline = float(loss)

    # ------------------------------------------------------------------
    # Main check
    # ------------------------------------------------------------------

    def check(
        self,
        action: InterventionAction,
        current_loss: float,
        risk_scores: Dict[str, float],
        layer_order: List[str],
    ) -> SafetyCheckResult:
        """Run all pre-flight safety checks for a proposed action.

        Parameters
        ----------
        action : InterventionAction
            Proposed action selected by the active policy.
        current_loss : float
            Training loss observed at the current step.
        risk_scores : dict[str, float]
            Mapping of layer name to current collapse risk score
            (``0.0``-``1.0``), as produced by
            :class:`~moewatch.analyzer.risk_score.RiskScoreFuser`.
        layer_order : list[str]
            Layer names in model order, used to determine
            :attr:`action`'s neighbours for the neighbor check.

        Returns
        -------
        SafetyCheckResult
            ``passed=True`` with ``recommended_action=action`` if every
            check passes. Otherwise ``passed=False``, ``failures``
            populated with one entry per failed check, and
            ``recommended_action`` set to
            ``NoOpAction(layer_name=action.layer_name)``.

        Notes
        -----
        :class:`~moewatch.intervention.actions.NoOpAction` instances
        always pass every check (``delta == 0.0`` trivially satisfies the
        delta limit, and a NoOp has no meaningful cooldown/neighbor
        implications) — there is nothing to downgrade further.

        This method updates :attr:`_risk_scores` with ``risk_scores`` as a
        side effect, regardless of outcome, so that subsequent calls have
        an up-to-date view even for layers not directly involved in this
        check.
        """
        # Keep the latest risk-score snapshot for bookkeeping / future
        # neighbor lookups, even if this call doesn't need every entry.
        self._risk_scores.update(risk_scores)

        if isinstance(action, NoOpAction):
            return SafetyCheckResult(passed=True, failures=[], recommended_action=action)

        failures: List[str] = []

        self._check_cooldown(action, failures)
        self._check_delta_limit(action, failures)
        self._check_loss_guard(current_loss, failures)
        self._check_neighbors(action, risk_scores, layer_order, failures)

        if not failures:
            return SafetyCheckResult(passed=True, failures=[], recommended_action=action)

        logger.info(
            "[MoEWatch] SafetyGuard: downgrading action %s to NoOp "
            "(%d check(s) failed: %s).",
            action.log(),
            len(failures),
            "; ".join(failures),
        )

        return SafetyCheckResult(
            passed=False,
            failures=failures,
            recommended_action=NoOpAction(layer_name=action.layer_name),
        )

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_cooldown(
        self, action: InterventionAction, failures: List[str]
    ) -> None:
        """Verify the cooldown period has elapsed for ``action.layer_name``.

        Parameters
        ----------
        action : InterventionAction
            Proposed action. Its :attr:`InterventionAction.applied_step`
            (set by the caller, e.g.
            :class:`~moewatch.intervention.engine.InterventionEngine`,
            *before* calling :meth:`check`) is used as the "current step"
            for cooldown comparison.
        failures : list[str]
            Failure-message accumulator; appended to in place if this
            check fails.

        Returns
        -------
        None

        Notes
        -----
        If ``action.layer_name`` has never had an intervention recorded
        (no entry in :attr:`_intervention_history`), the cooldown check
        trivially passes. The intervention history itself is recorded by
        :class:`~moewatch.intervention.engine.InterventionEngine` via
        :meth:`record_intervention`, *not* by this check.
        """
        last_step = self._intervention_history.get(action.layer_name)
        if last_step is None:
            return

        current_step = action.applied_step
        elapsed = current_step - last_step

        if elapsed < self.config.intervention_cooldown:
            failures.append(
                f"cooldown: only {elapsed} step(s) since last intervention "
                f"on '{action.layer_name}' "
                f"(requires >= {self.config.intervention_cooldown})"
            )

    def _check_delta_limit(
        self, action: InterventionAction, failures: List[str]
    ) -> None:
        """Verify ``action.delta`` is within ``config.intervention_max_delta``.

        Parameters
        ----------
        action : InterventionAction
            Proposed action.
        failures : list[str]
            Failure-message accumulator; appended to in place if this
            check fails.

        Returns
        -------
        None

        Notes
        -----
        Compares ``abs(action.delta)`` against
        ``config.intervention_max_delta`` to handle actions with
        negative deltas symmetrically.
        """
        max_delta = self.config.intervention_max_delta

        if abs(action.delta) > max_delta:
            failures.append(
                f"delta_limit: |{action.delta:.4f}| exceeds "
                f"intervention_max_delta={max_delta:.4f}"
            )

    def _check_loss_guard(self, current_loss: float, failures: List[str]) -> None:
        """Verify training loss has not spiked relative to the baseline.

        Parameters
        ----------
        current_loss : float
            Training loss observed at the current step.
        failures : list[str]
            Failure-message accumulator; appended to in place if this
            check fails.

        Returns
        -------
        None

        Notes
        -----
        If :meth:`update_baseline_loss` has never been called (baseline is
        ``None``), or ``current_loss`` is non-finite, this check is
        skipped (passes trivially) — there is no reference point to detect
        a spike against, and a non-finite loss is reported as its own
        training-level concern elsewhere, not silently used to freeze
        interventions forever.
        """
        if self._loss_baseline is None:
            return

        if current_loss != current_loss or current_loss in (
            float("inf"),
            float("-inf"),
        ):
            return

        threshold = self._loss_baseline * self.config.loss_guard_threshold
        if current_loss > threshold:
            failures.append(
                f"loss_guard: current_loss={current_loss:.6f} exceeds "
                f"baseline*threshold={threshold:.6f} "
                f"(baseline={self._loss_baseline:.6f}, "
                f"threshold={self.config.loss_guard_threshold:.2f})"
            )

    def _check_neighbors(
        self,
        action: InterventionAction,
        risk_scores: Dict[str, float],
        layer_order: List[str],
        failures: List[str],
    ) -> None:
        """Verify layers adjacent to ``action.layer_name`` are not CRITICAL.

        Parameters
        ----------
        action : InterventionAction
            Proposed action.
        risk_scores : dict[str, float]
            Mapping of layer name to current risk score.
        layer_order : list[str]
            Layer names in model order.
        failures : list[str]
            Failure-message accumulator; appended to in place if this
            check fails.

        Returns
        -------
        None

        Notes
        -----
        If ``action.layer_name`` is not present in ``layer_order`` (e.g.
        an action targeting a non-router submodule, or ``layer_order`` is
        empty), this check is skipped (passes trivially) — there is no
        well-defined neighbourhood to evaluate. Boundary layers (first or
        last in ``layer_order``) are checked only against the neighbour(s)
        that exist.

        A neighbour is considered CRITICAL if its risk score is
        ``>= _NEIGHBOR_CRITICAL_THRESHOLD``. Missing risk scores for a
        neighbour are treated as ``0.0`` (unknown == not critical), so
        that incomplete ``risk_scores`` data does not spuriously block
        interventions.
        """
        if action.layer_name not in layer_order:
            return

        idx = layer_order.index(action.layer_name)
        neighbor_indices = [i for i in (idx - 1, idx + 1) if 0 <= i < len(layer_order)]

        for n_idx in neighbor_indices:
            neighbor_name = layer_order[n_idx]
            neighbor_risk = risk_scores.get(neighbor_name, 0.0)

            if neighbor_risk >= self._NEIGHBOR_CRITICAL_THRESHOLD:
                failures.append(
                    f"neighbor_check: neighbor layer '{neighbor_name}' "
                    f"risk_score={neighbor_risk:.3f} >= "
                    f"{self._NEIGHBOR_CRITICAL_THRESHOLD:.2f} (CRITICAL)"
                )

    # ------------------------------------------------------------------
    # Intervention history bookkeeping
    # ------------------------------------------------------------------

    def record_intervention(self, layer_name: str, step: int) -> None:
        """Record that an intervention was applied to ``layer_name``.

        Parameters
        ----------
        layer_name : str
            Layer the intervention was applied to.
        step : int
            Training step at which the intervention was applied.

        Returns
        -------
        None

        Notes
        -----
        Called by
        :class:`~moewatch.intervention.engine.InterventionEngine` after an
        action successfully applies (i.e. one that was *not* downgraded to
        :class:`~moewatch.intervention.actions.NoOpAction`). Used by
        :meth:`_check_cooldown` on subsequent :meth:`check` calls.
        """
        self._intervention_history[layer_name] = int(step)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SafetyGuard(tracked_layers={len(self._intervention_history)}, "
            f"loss_baseline={self._loss_baseline!r}, "
            f"cooldown={self.config.intervention_cooldown}, "
            f"max_delta={self.config.intervention_max_delta}, "
            f"loss_guard_threshold={self.config.loss_guard_threshold})"
        )
