# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# MoEWatch — moewatch/report/watch_report.py  [v0.2.0]
# =============================================================================
#
# Project      : MoEWatch
# Version      : v0.2.0
# License      : Apache 2.0
# Author       : Abinesh N (@Abineshabee)
# Repository   : https://github.com/Abineshabee/MoEWatch
#
# Purpose
# -------
# Live training report objects generated at every ``MoEWatch.step()`` call.
# Two dataclasses are provided:
#
#   StepReport  — snapshot for a single training step.  Records risk scores,
#                 active interventions, policy decisions, counterfactual
#                 rewards, new alerts, training loss, and the dominant signal
#                 per layer.
#
#   WatchReport — rolling accumulator that collects ``StepReport`` objects
#                 over a configurable window.  Provides aggregation helpers
#                 (total interventions, alert counts) and serialisation.
#
# These replace the raw ``Alert`` list pattern from v0.1.0 with richer,
# structured introspection objects suitable for dashboarding, logging, and
# programmatic querying.
#
# Contents
# --------
#   StepReport    — per-step snapshot dataclass
#   WatchReport   — rolling accumulator with summary and serialisation
#
# Dependencies
# ------------
#   moewatch.__init__               — Alert
#   moewatch.config                 — AlertLevel
#   moewatch.intervention.actions   — InterventionAction
#   moewatch.analyzer.risk_score    — RiskReport, RiskLevel
#   json, datetime, logging, collections
#
# Usage
# -----
#   # MoEWatch.step() constructs and returns a WatchReport.
#   report = watch.step(global_step=1000)
#   print(report.summary())
#   report.to_json("watch_log.json")
#
# =============================================================================

from __future__ import annotations

import collections
import datetime
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

# Alert is defined at the package root to avoid circular imports.
from moewatch import Alert
from moewatch.analyzer.risk_score import RiskLevel, RiskReport
from moewatch.intervention.actions import InterventionAction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default maximum number of StepReport objects retained in WatchReport.
_DEFAULT_MAX_STEPS: int = 1_000

# Risk level label ordering (ascending severity) for display sorting.
_RISK_LEVEL_ORDER: Dict[str, int] = {
    RiskLevel.LOW.value: 0,
    RiskLevel.MID.value: 1,
    RiskLevel.HIGH.value: 2,
    RiskLevel.CRITICAL.value: 3,
}


# ---------------------------------------------------------------------------
# Internal serialisation helper
# ---------------------------------------------------------------------------


def _serialize_intervention(action: InterventionAction) -> Dict[str, Any]:
    """Convert an :class:`InterventionAction` to a JSON-safe dict.

    Falls back gracefully to a minimal representation if the action does not
    expose a ``to_dict()`` method.

    Parameters
    ----------
    action:
        Any concrete ``InterventionAction`` subclass instance.

    Returns
    -------
    dict
        JSON-serialisable representation.
    """
    if hasattr(action, "to_dict") and callable(action.to_dict):
        return action.to_dict()
    return {
        "action_type": type(action).__name__,
        "layer_name": getattr(action, "layer_name", None),
        "log_line": getattr(action, "log_line", str(action)),
    }


# ---------------------------------------------------------------------------
# StepReport
# ---------------------------------------------------------------------------


@dataclass
class StepReport:
    """Snapshot of MoEWatch monitoring state at a single training step.

    Constructed by ``MoEWatch.step()`` and immediately appended to the
    parent ``WatchReport``.  Fields are populated from all active analyzers,
    the intervention engine, and the policy selector.

    Attributes
    ----------
    step : int
        Global training step number.
    timestamp : float
        Unix timestamp at step completion.
    risk_scores : dict[str, float]
        Per-layer scalar risk score in [0.0, 1.0].
        Key: fully-qualified router module name.
    risk_levels : dict[str, str]
        Per-layer risk level string (``"low"`` / ``"mid"`` / ``"high"``
        / ``"critical"``).  Derived from risk_scores at step time.
    active_interventions : list[InterventionAction]
        All intervention actions applied at this step.  Empty list if no
        action was taken (e.g. risk below threshold or cooldown active).
    policy_decisions : dict[str, str]
        Per-layer action name selected by the policy for this step.
        Example: ``{"layers.5.moe": "aux_loss", "layers.9.moe": "noop"}``.
    counterfactual_rewards : dict[str, float]
        Per-layer counterfactual rewards from observation windows that
        closed at this step.  Populated by ``RewardComputer``; may be
        empty for most steps.
    alerts : list[Alert]
        New alerts emitted during this step (since the previous step).
        Does not include previously reported alerts.
    loss : float
        Training loss at this step.  ``float("nan")`` if not available.
    dominant_signals : dict[str, str]
        Per-layer name of the signal tier that contributed most to the
        risk score (``"gradient"`` / ``"entropy"`` / ``"cross_layer"``
        / ``"none"``).
    """

    step: int
    timestamp: float
    risk_scores: Dict[str, float] = field(default_factory=dict)
    risk_levels: Dict[str, str] = field(default_factory=dict)
    active_interventions: List[InterventionAction] = field(default_factory=list)
    policy_decisions: Dict[str, str] = field(default_factory=dict)
    counterfactual_rewards: Dict[str, float] = field(default_factory=dict)
    alerts: List[Alert] = field(default_factory=list)
    loss: float = float("nan")
    dominant_signals: Dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def has_critical(self) -> bool:
        """``True`` if any layer reached CRITICAL risk level at this step."""
        return any(
            v == RiskLevel.CRITICAL.value for v in self.risk_levels.values()
        )

    @property
    def num_interventions(self) -> int:
        """Number of interventions applied at this step."""
        return len(self.active_interventions)

    @property
    def num_alerts(self) -> int:
        """Number of new alerts emitted at this step."""
        return len(self.alerts)

    @property
    def worst_layer(self) -> Optional[str]:
        """Layer name with the highest risk score, or ``None`` if empty."""
        if not self.risk_scores:
            return None
        return max(self.risk_scores, key=lambda k: self.risk_scores[k])

    @property
    def step_datetime(self) -> datetime.datetime:
        """Human-readable :class:`datetime.datetime` of this step."""
        return datetime.datetime.fromtimestamp(self.timestamp)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise this step's snapshot to a JSON-safe dictionary.

        Returns
        -------
        dict
            All fields as primitive types or nested dicts/lists.
        """
        import math

        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "step_datetime": self.step_datetime.isoformat(),
            "risk_scores": dict(self.risk_scores),
            "risk_levels": dict(self.risk_levels),
            "active_interventions": [
                _serialize_intervention(a) for a in self.active_interventions
            ],
            "policy_decisions": dict(self.policy_decisions),
            "counterfactual_rewards": dict(self.counterfactual_rewards),
            "alerts": [a.to_dict() for a in self.alerts],
            "loss": self.loss if not math.isnan(self.loss) else None,
            "dominant_signals": dict(self.dominant_signals),
        }

    def __repr__(self) -> str:
        return (
            f"StepReport(step={self.step}, "
            f"interventions={self.num_interventions}, "
            f"alerts={self.num_alerts}, "
            f"worst={self.worst_layer!r})"
        )


# ---------------------------------------------------------------------------
# WatchReport
# ---------------------------------------------------------------------------


class WatchReport:
    """Rolling accumulator of per-step MoEWatch monitoring snapshots.

    ``MoEWatch.step()`` appends a new :class:`StepReport` to this object
    after every training step.  The report maintains at most *max_steps*
    snapshots to bound memory usage; older entries are automatically evicted
    using a deque.

    Aggregated statistics (``num_interventions``, ``num_alerts``) reflect
    only the currently retained window, not the full training run.

    Parameters
    ----------
    max_steps : int, optional
        Maximum number of per-step snapshots to retain.  Defaults to
        ``1000``.  Setting to ``0`` disables eviction (unlimited).

    Attributes
    ----------
    steps : collections.deque[StepReport]
        Chronologically ordered deque of per-step snapshots.
    start_step : int
        Global step number of the first retained snapshot, or ``-1`` if
        no snapshots have been added yet.
    end_step : int
        Global step number of the most recently added snapshot, or ``-1``
        if empty.
    """

    def __init__(self, max_steps: int = _DEFAULT_MAX_STEPS) -> None:
        if max_steps < 0:
            raise ValueError(
                f"max_steps must be >= 0, got {max_steps!r}. "
                "Pass 0 for unlimited retention."
            )
        self._max_steps = max_steps
        self.steps: Deque[StepReport] = collections.deque(
            maxlen=max_steps if max_steps > 0 else None
        )

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def append(self, step_report: StepReport) -> None:
        """Append a new per-step snapshot.

        Older snapshots are automatically evicted once the deque reaches
        ``max_steps`` entries.

        Parameters
        ----------
        step_report:
            Completed snapshot from one ``MoEWatch.step()`` call.

        Raises
        ------
        TypeError
            If *step_report* is not a :class:`StepReport` instance.
        """
        if not isinstance(step_report, StepReport):
            raise TypeError(
                f"Expected StepReport, got {type(step_report).__name__!r}."
            )
        self.steps.append(step_report)

    # ------------------------------------------------------------------
    # Derived aggregates
    # ------------------------------------------------------------------

    @property
    def start_step(self) -> int:
        """Global step of the oldest retained snapshot (``-1`` if empty)."""
        return self.steps[0].step if self.steps else -1

    @property
    def end_step(self) -> int:
        """Global step of the newest retained snapshot (``-1`` if empty)."""
        return self.steps[-1].step if self.steps else -1

    @property
    def num_interventions(self) -> int:
        """Total intervention actions applied across all retained steps."""
        return sum(sr.num_interventions for sr in self.steps)

    @property
    def num_alerts(self) -> int:
        """Total alerts emitted across all retained steps."""
        return sum(sr.num_alerts for sr in self.steps)

    @property
    def num_critical_steps(self) -> int:
        """Number of retained steps where at least one layer was CRITICAL."""
        return sum(1 for sr in self.steps if sr.has_critical)

    @property
    def is_empty(self) -> bool:
        """``True`` if no snapshots have been added yet."""
        return len(self.steps) == 0

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def latest(self) -> Optional[StepReport]:
        """Return the most recently appended :class:`StepReport`, or ``None``."""
        return self.steps[-1] if self.steps else None

    def steps_since(self, since_step: int) -> List[StepReport]:
        """Return all retained snapshots for steps > *since_step*.

        Parameters
        ----------
        since_step:
            Lower bound (exclusive) on the step counter.

        Returns
        -------
        list[StepReport]
            Ordered list of snapshots with ``step > since_step``.
        """
        return [sr for sr in self.steps if sr.step > since_step]

    def alerts_since(self, since_step: int = 0) -> List[Alert]:
        """Return all new alerts from retained steps > *since_step*.

        Parameters
        ----------
        since_step:
            Lower bound (exclusive) on the step counter.

        Returns
        -------
        list[Alert]
            Flat, chronologically ordered list of alerts.
        """
        alerts: List[Alert] = []
        for sr in self.steps:
            if sr.step > since_step:
                alerts.extend(sr.alerts)
        return alerts

    def interventions_since(self, since_step: int = 0) -> List[InterventionAction]:
        """Return all intervention actions from retained steps > *since_step*.

        Parameters
        ----------
        since_step:
            Lower bound (exclusive) on the step counter.

        Returns
        -------
        list[InterventionAction]
            Flat, chronologically ordered list of applied actions.
        """
        actions: List[InterventionAction] = []
        for sr in self.steps:
            if sr.step > since_step:
                actions.extend(sr.active_interventions)
        return actions

    def risk_history(self, layer_name: str) -> List[tuple[int, float]]:
        """Return the risk score history for a specific layer.

        Parameters
        ----------
        layer_name:
            Fully-qualified router module name.

        Returns
        -------
        list[tuple[int, float]]
            Each element is ``(step, risk_score)`` ordered chronologically.
            Steps where the layer was not monitored are omitted.
        """
        history: List[tuple[int, float]] = []
        for sr in self.steps:
            if layer_name in sr.risk_scores:
                history.append((sr.step, sr.risk_scores[layer_name]))
        return history

    def worst_layer_history(self) -> List[tuple[int, str, float]]:
        """Return the layer with the highest risk at each retained step.

        Returns
        -------
        list[tuple[int, str, float]]
            Each element is ``(step, layer_name, risk_score)`` chronologically.
            Steps with no risk data are omitted.
        """
        history: List[tuple[int, str, float]] = []
        for sr in self.steps:
            worst = sr.worst_layer
            if worst is not None:
                history.append((sr.step, worst, sr.risk_scores[worst]))
        return history

    def mean_risk(self, layer_name: Optional[str] = None) -> float:
        """Compute the mean risk score over all retained steps.

        Parameters
        ----------
        layer_name:
            If provided, computes the mean for that specific layer.
            If ``None``, computes the mean across *all* layers and steps.

        Returns
        -------
        float
            Mean risk score in [0.0, 1.0], or ``float("nan")`` if no data.
        """
        values: List[float] = []
        for sr in self.steps:
            if layer_name is not None:
                if layer_name in sr.risk_scores:
                    values.append(sr.risk_scores[layer_name])
            else:
                values.extend(sr.risk_scores.values())
        return sum(values) / len(values) if values else float("nan")

    # ------------------------------------------------------------------
    # Summary text
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a concise human-readable summary of the retained window.

        Returns
        -------
        str
            Multi-line plain-text summary of aggregate statistics.
        """
        if self.is_empty:
            return "WatchReport: no steps recorded yet."

        latest = self.latest()
        sep = "=" * 72

        lines: List[str] = []
        lines.append(sep)
        lines.append("  MoEWatch Live Report")
        lines.append(sep)
        lines.append(f"  Steps            : {self.start_step} → {self.end_step}")
        lines.append(f"  Retained steps   : {len(self.steps)} / {self._max_steps or '∞'}")
        lines.append(f"  Total alerts     : {self.num_alerts}")
        lines.append(f"  Total interventions : {self.num_interventions}")
        lines.append(f"  Critical steps   : {self.num_critical_steps}")
        lines.append("")

        # Latest step snapshot
        if latest is not None:
            lines.append(f"  Latest step      : {latest.step}")
            import math

            if not math.isnan(latest.loss):
                lines.append(f"  Training loss    : {latest.loss:.6f}")
            if latest.risk_scores:
                lines.append("  Current risk scores:")
                for lname, score in sorted(
                    latest.risk_scores.items(),
                    key=lambda x: x[1],
                    reverse=True,
                ):
                    level = latest.risk_levels.get(lname, "?")
                    bar = _risk_bar(score, width=16)
                    lines.append(
                        f"    {lname:<40s}  {bar}  {score:.3f}  [{level}]"
                    )
            if latest.active_interventions:
                lines.append("  Active interventions at latest step:")
                for action in latest.active_interventions:
                    log_line = getattr(action, "log_line", str(action))
                    lines.append(f"    → {log_line}")

        lines.append(sep)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self, max_steps_in_output: Optional[int] = None) -> Dict[str, Any]:
        """Serialise the report to a JSON-safe dictionary.

        Parameters
        ----------
        max_steps_in_output:
            If set, only the most recent N step reports are included in the
            ``"steps"`` list.  ``None`` (default) includes all retained steps.

        Returns
        -------
        dict
            JSON-serialisable representation.
        """
        steps_list = list(self.steps)
        if max_steps_in_output is not None:
            steps_list = steps_list[-max_steps_in_output:]

        return {
            "start_step": self.start_step,
            "end_step": self.end_step,
            "num_retained_steps": len(self.steps),
            "max_steps": self._max_steps,
            "num_alerts": self.num_alerts,
            "num_interventions": self.num_interventions,
            "num_critical_steps": self.num_critical_steps,
            "steps": [sr.to_dict() for sr in steps_list],
        }

    def to_json(
        self,
        path: str,
        indent: int = 2,
        max_steps_in_output: Optional[int] = None,
    ) -> None:
        """Serialise the rolling report to a JSON file at *path*.

        Parameters
        ----------
        path:
            Destination file path.
        indent:
            JSON indentation level.  Defaults to ``2``.
        max_steps_in_output:
            If set, cap the number of per-step records in the output.
            Useful for large runs where writing all steps would produce
            very large files.

        Raises
        ------
        OSError
            If the file cannot be written.
        """
        data = self.to_dict(max_steps_in_output=max_steps_in_output)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=indent)
            logger.info(
                "WatchReport saved to %s (%d steps)", path, len(self.steps)
            )
        except OSError as exc:
            logger.error("Failed to write WatchReport to %s: %s", path, exc)
            raise

    # ------------------------------------------------------------------
    # dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        return (
            f"WatchReport("
            f"steps={self.start_step}→{self.end_step}, "
            f"retained={len(self.steps)}, "
            f"alerts={self.num_alerts}, "
            f"interventions={self.num_interventions})"
        )


# ---------------------------------------------------------------------------
# Module-private helper
# ---------------------------------------------------------------------------


def _risk_bar(score: float, width: int = 16) -> str:
    """Render a compact ASCII risk bar for *score* in [0.0, 1.0].

    Parameters
    ----------
    score:
        Risk score in [0.0, 1.0].
    width:
        Bar width in characters.

    Returns
    -------
    str
        A bar string like ``"[########........]"``.
    """
    score = max(0.0, min(1.0, score))
    filled = int(round(score * width))
    empty = width - filled
    return f"[{'#' * filled}{'.' * empty}]"
