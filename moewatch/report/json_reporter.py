# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# MoEWatch — moewatch/report/json_reporter.py  [v0.2.0]
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
# Newline-delimited JSON (NDJSON) emitter for MoEWatch monitoring events.
# Designed for integration with log aggregation and dashboarding systems:
#
#   Grafana   — Loki datasource reads NDJSON from file
#   Splunk    — universal forwarder ingests NDJSON
#   W&B       — custom metrics tables via wandb.log()
#   Elasticsearch — Filebeat or direct ingest API
#   Stdout    — containerised training (piped to log collector)
#
# Every call to ``write_step()`` emits exactly one JSON line containing all
# monitoring metrics for that training step.  The format is flat and human-
# readable by design: fields are top-level where possible and nested only
# for logically grouped structures (e.g., per-layer risks).
#
# Output can be directed to a file (append mode) or stdout.  If an output
# file is specified but cannot be opened, a warning is logged and the line
# is written to stdout as a fallback — monitoring should never crash training.
#
# Contents
# --------
#   JSONReporter   — NDJSON emitter class
#
# JSON Line Schema (one object per step):
#   {
#     "moewatch_version": "v0.2.0",
#     "step": 1000,
#     "timestamp": 1723456789.123,
#     "datetime": "2024-08-12T14:39:49",
#     "loss": 2.345,
#     "risks": {
#       "model.layers.5.moe": 0.72,
#       "model.layers.9.moe": 0.21
#     },
#     "risk_levels": {
#       "model.layers.5.moe": "high",
#       "model.layers.9.moe": "low"
#     },
#     "dominant_signals": {
#       "model.layers.5.moe": "gradient"
#     },
#     "interventions": [
#       {"action": "AuxLossAction", "layer": "model.layers.5.moe", "delta": 0.05}
#     ],
#     "alerts": [
#       {"step": 1000, "level": "WARNING", "layer": "model.layers.5.moe",
#        "signal": "entropy_drift", "message": "..."}
#     ],
#     "num_interventions": 1,
#     "num_alerts": 1
#   }
#
# Dependencies
# ------------
#   moewatch.__init__               — Alert
#   moewatch.config                 — WatchConfig
#   moewatch.intervention.actions   — InterventionAction
#   json, sys, os, datetime, logging, math
#
# Usage
# -----
#   reporter = JSONReporter(config, output_file="moewatch_log.jsonl")
#   line = reporter.emit_step(
#       step=1000,
#       timestamp=time.time(),
#       risk_scores={"layers.5": 0.72},
#       loss=2.345,
#       interventions=[...],
#       alerts=[...],
#   )
#   reporter.write_step(...)   # emit + write in one call
#   reporter.close()           # flush and close file handle
#
# =============================================================================

from __future__ import annotations

import datetime
import json
import logging
import math
import os
import sys
from typing import Any, Dict, List, Optional, TextIO

from moewatch import Alert
from moewatch.config import WatchConfig
from moewatch.intervention.actions import InterventionAction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema version embedded in every emitted line.
# ---------------------------------------------------------------------------

_SCHEMA_VERSION = "v0.2.0"


# ---------------------------------------------------------------------------
# Internal serialisation helpers
# ---------------------------------------------------------------------------


def _serialize_alert(alert: Alert) -> Dict[str, Any]:
    """Convert an :class:`Alert` to a compact JSON-safe dict.

    Parameters
    ----------
    alert:
        Alert to serialise.

    Returns
    -------
    dict
        Compact representation: level, step, layer, signal, message, metrics.
    """
    level_val = (
        alert.level.value if hasattr(alert.level, "value") else str(alert.level)
    )
    return {
        "step": alert.step,
        "level": level_val.upper(),
        "layer": alert.layer_id,
        "signal": alert.signal_type,
        "message": alert.message,
        "metrics": dict(alert.metrics),
    }


def _serialize_intervention(action: InterventionAction) -> Dict[str, Any]:
    """Convert an :class:`InterventionAction` to a compact JSON-safe dict.

    Falls back to a minimal representation if ``to_dict()`` is not available.

    Parameters
    ----------
    action:
        Action to serialise.

    Returns
    -------
    dict
        Compact representation: action type, layer, delta, and log line.
    """
    if hasattr(action, "to_dict") and callable(action.to_dict):
        d = action.to_dict()
        # Normalise key names to schema spec if the action uses different names.
        return {
            "action": d.get("action_type", type(action).__name__),
            "layer": d.get("layer_name", getattr(action, "layer_name", None)),
            "delta": d.get("delta", getattr(action, "delta", None)),
            "log_line": d.get("log_line", getattr(action, "log_line", "")),
        }
    return {
        "action": type(action).__name__,
        "layer": getattr(action, "layer_name", None),
        "delta": getattr(action, "delta", None),
        "log_line": getattr(action, "log_line", ""),
    }


def _safe_float(value: float) -> Optional[float]:
    """Convert a float to None if it is NaN or infinite.

    JSON does not support ``NaN`` / ``Infinity`` as number literals, so we
    substitute ``null`` (Python ``None``) to avoid serialisation errors.

    Parameters
    ----------
    value:
        Float value that may be NaN or infinite.

    Returns
    -------
    float or None
        The original value, or ``None`` if it is NaN/infinite.
    """
    if math.isnan(value) or math.isinf(value):
        return None
    return value


# ---------------------------------------------------------------------------
# JSONReporter
# ---------------------------------------------------------------------------


class JSONReporter:
    """Emits newline-delimited JSON for log aggregation systems.

    One JSON line per training step. Contains all MoEWatch metrics in a flat
    or nested structure suitable for Elasticsearch indexing and Grafana /
    Splunk / W&B dashboarding.

    Output is written to a file in append mode, or to stdout if no file is
    configured.  File I/O errors are caught and logged; in that case the
    line is printed to stdout as a fallback so monitoring never blocks or
    crashes training.

    Parameters
    ----------
    config : WatchConfig
        Shared monitoring configuration.
    output_file : str or None, optional
        Path to the output ``.jsonl`` file.  If ``None``, all lines are
        written to stdout (suitable for containerised logging).
    """

    def __init__(
        self,
        config: WatchConfig,
        output_file: Optional[str] = None,
    ) -> None:
        self.config = config
        self.output_file: Optional[str] = output_file

        # File handle — opened lazily on first write to avoid holding an empty
        # open file descriptor until training starts.
        self._file_handle: Optional[TextIO] = None
        self._file_open_attempted: bool = False

    # ------------------------------------------------------------------
    # Public API — emit helpers
    # ------------------------------------------------------------------

    def emit_step(
        self,
        step: int,
        timestamp: float,
        risk_scores: Dict[str, float],
        loss: float,
        interventions: List[InterventionAction],
        alerts: List[Alert],
        *,
        risk_levels: Optional[Dict[str, str]] = None,
        dominant_signals: Optional[Dict[str, str]] = None,
        counterfactual_rewards: Optional[Dict[str, float]] = None,
        policy_decisions: Optional[Dict[str, str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a single NDJSON line for one training step.

        All monitoring metrics are serialised to a single compact JSON object.
        The result is a valid JSON string terminated by a newline character.

        Parameters
        ----------
        step:
            Current global training step number.
        timestamp:
            Unix timestamp (seconds since epoch) for this step.
        risk_scores:
            Per-layer scalar risk score in [0.0, 1.0].
            Key: fully-qualified router module name.
        loss:
            Training loss at this step.  Pass ``float("nan")`` if unavailable.
        interventions:
            All :class:`InterventionAction` objects applied at this step.
        alerts:
            All :class:`Alert` objects emitted at this step.
        risk_levels:
            Optional per-layer risk level string (``"low"`` / ``"mid"``
            / ``"high"`` / ``"critical"``).
        dominant_signals:
            Optional per-layer dominant signal tier name.
        counterfactual_rewards:
            Optional per-layer counterfactual reward from closed observation
            windows.
        policy_decisions:
            Optional per-layer policy action name.
        extra:
            Optional dict of additional key-value pairs to embed in the JSON
            line.  Keys must not conflict with reserved schema fields.

        Returns
        -------
        str
            A single NDJSON line (JSON object + newline character).
        """
        record: Dict[str, Any] = {
            "moewatch_version": _SCHEMA_VERSION,
            "step": step,
            "timestamp": timestamp,
            "datetime": datetime.datetime.fromtimestamp(timestamp).isoformat(),
            "loss": _safe_float(loss),
            "risks": {
                k: _safe_float(v) for k, v in (risk_scores or {}).items()
            },
            "num_interventions": len(interventions),
            "num_alerts": len(alerts),
        }

        # Optional enrichment fields — only included if provided.
        if risk_levels:
            record["risk_levels"] = dict(risk_levels)

        if dominant_signals:
            record["dominant_signals"] = dict(dominant_signals)

        if counterfactual_rewards:
            record["counterfactual_rewards"] = {
                k: _safe_float(v) for k, v in counterfactual_rewards.items()
            }

        if policy_decisions:
            record["policy_decisions"] = dict(policy_decisions)

        # Interventions — serialise to compact schema.
        record["interventions"] = [
            _serialize_intervention(a) for a in (interventions or [])
        ]

        # Alerts — serialise to compact schema.
        record["alerts"] = [_serialize_alert(a) for a in (alerts or [])]

        # Caller-provided extras (e.g. experiment metadata, W&B run ID).
        if extra:
            # Protect reserved schema fields.
            reserved = set(record.keys())
            for k, v in extra.items():
                if k in reserved:
                    logger.warning(
                        "JSONReporter: extra key %r conflicts with reserved "
                        "schema field. Skipping.", k
                    )
                    continue
                record[k] = v

        try:
            line = json.dumps(record, ensure_ascii=False) + "\n"
        except (TypeError, ValueError) as exc:
            # Fallback: strip un-serialisable values and retry.
            logger.warning(
                "JSONReporter: JSON serialisation failed (%s). "
                "Emitting reduced record.", exc
            )
            line = json.dumps(
                {
                    "moewatch_version": _SCHEMA_VERSION,
                    "step": step,
                    "timestamp": timestamp,
                    "error": "serialisation_failed",
                    "detail": str(exc),
                },
                ensure_ascii=False,
            ) + "\n"

        return line

    def write_step(
        self,
        step: int,
        timestamp: float,
        risk_scores: Dict[str, float],
        loss: float,
        interventions: List[InterventionAction],
        alerts: List[Alert],
        *,
        risk_levels: Optional[Dict[str, str]] = None,
        dominant_signals: Optional[Dict[str, str]] = None,
        counterfactual_rewards: Optional[Dict[str, float]] = None,
        policy_decisions: Optional[Dict[str, str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a NDJSON step record and write it to the configured output.

        This is the primary method called by ``MoEWatch.step()`` at every
        log interval.  It delegates generation to :meth:`emit_step` and
        writes the result to the output file or stdout.

        Parameters
        ----------
        (all parameters are identical to :meth:`emit_step`)

        Side effects
        ------------
        Appends one JSON line to ``self.output_file`` (or stdout).
        Logs a warning and falls back to stdout on I/O error.
        """
        line = self.emit_step(
            step=step,
            timestamp=timestamp,
            risk_scores=risk_scores,
            loss=loss,
            interventions=interventions,
            alerts=alerts,
            risk_levels=risk_levels,
            dominant_signals=dominant_signals,
            counterfactual_rewards=counterfactual_rewards,
            policy_decisions=policy_decisions,
            extra=extra,
        )
        self._write_line(line)

    # ------------------------------------------------------------------
    # Convenience: write a pre-built StepReport
    # ------------------------------------------------------------------

    def write_step_report(self, step_report: Any) -> None:
        """Write a :class:`~moewatch.report.watch_report.StepReport` as a JSON line.

        Convenience wrapper that unpacks a ``StepReport`` and delegates to
        :meth:`write_step`.  Useful when the caller already has a ``StepReport``
        object from ``MoEWatch.step()``.

        Parameters
        ----------
        step_report:
            A :class:`~moewatch.report.watch_report.StepReport` instance.
        """
        # Import here to avoid circular import at module load time.
        from moewatch.report.watch_report import StepReport  # noqa: PLC0415

        if not isinstance(step_report, StepReport):
            raise TypeError(
                f"Expected StepReport, got {type(step_report).__name__!r}."
            )
        self.write_step(
            step=step_report.step,
            timestamp=step_report.timestamp,
            risk_scores=step_report.risk_scores,
            loss=step_report.loss,
            interventions=step_report.active_interventions,
            alerts=step_report.alerts,
            risk_levels=step_report.risk_levels,
            dominant_signals=step_report.dominant_signals,
            counterfactual_rewards=step_report.counterfactual_rewards,
            policy_decisions=step_report.policy_decisions,
        )

    # ------------------------------------------------------------------
    # File management
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Flush the output file buffer.

        No-op if writing to stdout or if the file handle is not open.
        """
        if self._file_handle is not None and not self._file_handle.closed:
            try:
                self._file_handle.flush()
            except OSError as exc:
                logger.warning("JSONReporter: flush failed: %s", exc)

    def close(self) -> None:
        """Flush and close the output file handle.

        Safe to call multiple times; subsequent calls are no-ops.
        """
        if self._file_handle is not None and not self._file_handle.closed:
            try:
                self._file_handle.flush()
                self._file_handle.close()
                logger.debug("JSONReporter: closed %s", self.output_file)
            except OSError as exc:
                logger.warning("JSONReporter: close failed: %s", exc)
            finally:
                self._file_handle = None

    def rotate(self, new_path: str) -> None:
        """Close the current output file and start writing to a new path.

        Useful for log rotation during very long training runs.

        Parameters
        ----------
        new_path:
            Destination path for subsequent writes.
        """
        self.close()
        self.output_file = new_path
        self._file_open_attempted = False
        logger.info("JSONReporter: rotated to %s", new_path)

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "JSONReporter":
        """Context manager entry — returns self."""
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:
        """Context manager exit — flushes and closes the file handle."""
        self.close()

    # ------------------------------------------------------------------
    # Internal I/O
    # ------------------------------------------------------------------

    def _get_file_handle(self) -> Optional[TextIO]:
        """Lazily open the output file in append mode.

        Returns the open file handle, or ``None`` if the file cannot be
        opened (in which case a warning is logged and stdout is used).

        Returns
        -------
        TextIO or None
            Open file handle, or ``None`` on error.
        """
        # Already open and not closed — return immediately.
        if self._file_handle is not None and not self._file_handle.closed:
            return self._file_handle

        # Already attempted and failed — don't retry every step.
        if self._file_open_attempted and self._file_handle is None:
            return None

        if self.output_file is None:
            return None

        self._file_open_attempted = True
        try:
            # Ensure the parent directory exists.
            parent = os.path.dirname(os.path.abspath(self.output_file))
            os.makedirs(parent, exist_ok=True)

            self._file_handle = open(  # noqa: WPS515
                self.output_file, "a", encoding="utf-8", buffering=1
            )
            logger.debug("JSONReporter: opened %s (append)", self.output_file)
            return self._file_handle
        except OSError as exc:
            logger.warning(
                "JSONReporter: cannot open %r for writing (%s). "
                "Falling back to stdout.",
                self.output_file,
                exc,
            )
            self._file_handle = None
            return None

    def _write_line(self, line: str) -> None:
        """Write a single NDJSON line to the output destination.

        Falls back to stdout on I/O error rather than raising, to ensure
        monitoring never crashes training.

        Parameters
        ----------
        line:
            A complete NDJSON line (JSON object + trailing newline).
        """
        fh = self._get_file_handle()

        if fh is not None:
            try:
                fh.write(line)
                return
            except OSError as exc:
                logger.warning(
                    "JSONReporter: write to %r failed (%s). "
                    "Falling back to stdout for this line.",
                    self.output_file,
                    exc,
                )

        # Stdout fallback — always safe.
        sys.stdout.write(line)
        sys.stdout.flush()

    # ------------------------------------------------------------------
    # dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        dest = repr(self.output_file) if self.output_file else "stdout"
        return f"JSONReporter(output={dest})"
