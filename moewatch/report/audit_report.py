# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# MoEWatch — moewatch/report/audit_report.py
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
# Structured, immutable result object returned by the offline ``audit()``
# function.  Aggregates the output of every analyzer that ran during the
# diagnostic pass: entropy, collapse, gradient starvation, cross-layer
# correlation, and fused risk scores.
#
# The report is intentionally frozen (``frozen=True``) to prevent accidental
# mutation after the audit pass completes.  All per-layer results are keyed
# by the fully-qualified router module name (e.g. ``"model.layers.5.moe"``).
#
# Serialisation is provided through ``to_json()`` (newline-safe JSON file),
# and optional tabular export through ``to_dataframe()`` (requires pandas).
#
# Contents
# --------
#   AuditReport   — frozen dataclass: per-layer audit results + helpers
#
# Signal Hierarchy (for context)
# --------------------------------
#   Tier 1 — Gradient Starvation   (50–200 steps before collapse)
#   Tier 2 — Entropy Drift         (30–100 steps before collapse)
#   Tier 3 — Cross-Layer Spread    (system-level localization)
#
# Dependencies
# ------------
#   moewatch.analyzer.entropy             — LayerEntropyReport
#   moewatch.analyzer.collapse            — LayerCollapseReport
#   moewatch.analyzer.gradient_starvation — GradientStarvationReport
#   moewatch.analyzer.cross_layer         — CrossLayerReport
#   moewatch.analyzer.risk_score          — RiskReport, RiskLevel
#   json, datetime, logging
#   pandas (optional — only for to_dataframe())
#
# Usage
# -----
#   report = audit(model, dataloader, num_batches=50)
#   print(report.summary())
#   report.to_json("audit_results.json")
#   df = report.to_dataframe()   # requires pandas
#
# =============================================================================

from __future__ import annotations

import datetime
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from moewatch.analyzer.collapse import LayerCollapseReport
from moewatch.analyzer.cross_layer import CrossLayerReport
from moewatch.analyzer.entropy import LayerEntropyReport
from moewatch.analyzer.gradient_starvation import GradientStarvationReport
from moewatch.analyzer.risk_score import RiskLevel, RiskReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal serialisation helpers
# ---------------------------------------------------------------------------


def _to_serializable(obj: Any) -> Any:
    """Recursively convert dataclasses, enums, and other types to JSON-safe
    primitives.

    Parameters
    ----------
    obj:
        Any Python object.

    Returns
    -------
    Any
        A JSON-serialisable representation of ``obj``.
    """
    # Dataclasses that expose to_dict() — use it for controlled serialisation.
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()

    # Enums — fall back to .value.
    if hasattr(obj, "value"):
        return obj.value

    # Dicts — recurse into values.
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}

    # Lists and tuples — recurse.
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(item) for item in obj]

    # Primitives — return as-is.
    return obj


# ---------------------------------------------------------------------------
# AuditReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditReport:
    """Immutable structured result from the offline :func:`moewatch.audit`
    diagnostic pass.

    Each field holds the aggregated output of one analyzer.  The report
    is frozen to prevent accidental mutation; create a new instance if you
    need modified data.

    Attributes
    ----------
    model_name : str
        Name of the audited model (typically ``type(model).__name__``).
    timestamp : float
        Unix timestamp (seconds since epoch) recorded at audit completion.
    num_batches : int
        Number of forward-pass batches processed during the audit.
    entropy_results : dict[str, LayerEntropyReport]
        Per-layer Tier 2 entropy drift analysis, keyed by router module name.
        Empty dict if the entropy analyzer was skipped or raised.
    collapse_results : dict[str, LayerCollapseReport]
        Per-layer expert health-state analysis (HEALTHY / COLD / DEAD).
        Empty dict if the collapse detector was skipped or raised.
    gradient_results : dict[str, list[GradientStarvationReport]]
        Per-layer list of Tier 1 gradient starvation analyses, one entry
        per expert that has gradient data. Key is the router layer name;
        each report exposes its own ``expert_id``.
        Empty dict if gradient hooks were not enabled or no events collected.
    cross_layer_results : CrossLayerReport or None
        Tier 3 inter-layer entropy correlation matrix.  ``None`` if the
        cross-layer analyzer did not run (e.g. insufficient layer count).
    risk_scores : dict[str, RiskReport]
        Per-layer fused risk score in [0.0, 1.0] with explainability breakdown.
        Populated only when risk_score fusion ran without error.
    dead_experts_count : int
        Total number of experts classified as DEAD across all layers.
        Derived from ``collapse_results`` at construction time.
    critical_layers : list[str]
        List of layer names where ``risk_level == CRITICAL``.
        Derived from ``risk_scores`` at construction time.
    """

    # Core metadata
    model_name: str
    timestamp: float
    num_batches: int

    # Per-layer analyzer results
    entropy_results: Dict[str, LayerEntropyReport] = field(default_factory=dict)
    collapse_results: Dict[str, LayerCollapseReport] = field(default_factory=dict)
    gradient_results: Dict[str, List[GradientStarvationReport]] = field(
        default_factory=dict
    )
    cross_layer_results: Optional[CrossLayerReport] = field(default=None)
    risk_scores: Dict[str, RiskReport] = field(default_factory=dict)

    # Derived summary fields (computed by _audit.py before construction)
    dead_experts_count: int = 0
    critical_layers: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Read-only derived properties
    # ------------------------------------------------------------------

    @property
    def num_layers(self) -> int:
        """Total number of monitored MoE layers in this audit."""
        return len(self.entropy_results) or len(self.collapse_results)

    @property
    def has_critical_risk(self) -> bool:
        """``True`` if at least one layer is classified as CRITICAL."""
        return bool(self.critical_layers)

    @property
    def audit_datetime(self) -> datetime.datetime:
        """Human-readable :class:`datetime.datetime` of the audit timestamp."""
        return datetime.datetime.fromtimestamp(self.timestamp)

    # ------------------------------------------------------------------
    # Public API — querying
    # ------------------------------------------------------------------

    def dead_experts(self) -> List[Tuple[str, int]]:
        """Return a list of all dead experts across all layers.

        Returns
        -------
        list[tuple[str, int]]
            Each element is ``(layer_name, expert_id)`` for every expert
            whose health state is DEAD in ``collapse_results``.  Returns an
            empty list if collapse results are not available.

        Examples
        --------
        >>> dead = report.dead_experts()
        >>> for layer, eid in dead:
        ...     print(f"  Dead expert {eid} in {layer}")
        """
        dead: List[Tuple[str, int]] = []
        for layer_name, collapse_report in self.collapse_results.items():
            for expert_state in collapse_report.expert_states.values():
                if expert_state.status.value == "dead":
                    dead.append((layer_name, expert_state.expert_id))
        return dead

    def layer_risk(self, layer_name: str) -> Optional[RiskReport]:
        """Look up the fused risk report for a specific layer.

        Parameters
        ----------
        layer_name:
            Fully-qualified router module name.

        Returns
        -------
        RiskReport or None
            The risk report if available, otherwise ``None``.
        """
        return self.risk_scores.get(layer_name)

    def get_risk_for_layer(self, layer_name: str) -> Optional[RiskReport]:
        """Look up the fused risk report for a specific layer.

        Alias for :meth:`layer_risk`, provided for callers that prefer the
        explicit ``get_*`` accessor naming convention.

        Parameters
        ----------
        layer_name:
            Fully-qualified router module name.

        Returns
        -------
        RiskReport or None
            The risk report if available, otherwise ``None``.
        """
        return self.risk_scores.get(layer_name)

    def layers_by_risk(self) -> List[Tuple[str, float]]:
        """Return all layers sorted by descending risk score.

        Returns
        -------
        list[tuple[str, float]]
            Each element is ``(layer_name, risk_score)`` in descending order.
        """
        return sorted(
            [(name, rr.risk_score) for name, rr in self.risk_scores.items()],
            key=lambda x: x[1],
            reverse=True,
        )

    def gradient_starved_experts(
        self, threshold: float = 0.01
    ) -> List[Tuple[str, int, float]]:
        """Return experts whose gradient norm falls below the given threshold.

        Parameters
        ----------
        threshold:
            Gradient norm threshold. Experts with ``mean_gradient_norm`` below
            this value are considered starved. Defaults to ``0.01``.

        Returns
        -------
        list[tuple[str, int, float]]
            Each element is ``(layer_name, expert_id, mean_gradient_norm)``
            for every starved expert, sorted by gradient norm ascending.
        """
        starved: List[Tuple[str, int, float]] = []
        for layer_name, expert_reports in self.gradient_results.items():
            for gs_report in expert_reports:
                expert_id = getattr(gs_report, "expert_id", -1)
                norm = getattr(gs_report, "gradient_norm_mean", float("inf"))
                if norm < threshold:
                    starved.append((layer_name, expert_id, norm))
        return sorted(starved, key=lambda x: x[2])

    # ------------------------------------------------------------------
    # Public API — summary text
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a concise human-readable text summary of audit findings.

        Covers: audit metadata, risk overview, critical layers, dead experts,
        gradient starvation summary, and cross-layer spread status.

        Returns
        -------
        str
            Multi-line plain-text summary suitable for logging or printing.
        """
        lines: List[str] = []
        sep = "=" * 72

        # Header
        lines.append(sep)
        lines.append("  MoEWatch Audit Report")
        lines.append(sep)
        lines.append(f"  Model        : {self.model_name}")
        lines.append(
            f"  Audit time   : {self.audit_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        lines.append(f"  Batches      : {self.num_batches}")
        lines.append(f"  Layers       : {self.num_layers}")
        lines.append("")

        # Risk overview
        lines.append("  Risk Overview")
        lines.append("  " + "-" * 68)
        if self.risk_scores:
            for layer_name, rr in sorted(self.risk_scores.items()):
                bar = _risk_bar(rr.risk_score, width=20)
                lines.append(
                    f"  {layer_name:<40s}  {bar}  {rr.risk_score:.3f}"
                    f"  [{rr.risk_level.value.upper()}]"
                )
        else:
            lines.append("  (risk scores not available)")
        lines.append("")

        # Critical layers
        if self.critical_layers:
            lines.append(
                f"  Critical layers ({len(self.critical_layers)} total):"
            )
            for lname in self.critical_layers:
                lines.append(f"    ⚠  {lname}")
        else:
            lines.append("  No critical layers detected.")
        lines.append("")

        # Dead experts
        dead = self.dead_experts()
        lines.append(f"  Dead experts : {self.dead_experts_count}")
        if dead:
            for layer_name, eid in dead[:10]:  # cap display at 10
                lines.append(f"    — expert {eid:>3d}  in  {layer_name}")
            if len(dead) > 10:
                lines.append(f"    ... and {len(dead) - 10} more")
        lines.append("")

        # Gradient starvation summary
        starved = self.gradient_starved_experts()
        if starved:
            lines.append(
                f"  Gradient-starved experts (norm < 0.01) : {len(starved)}"
            )
            for layer_name, eid, norm in starved[:5]:
                lines.append(
                    f"    — expert {eid:>3d}  norm={norm:.5f}  in  {layer_name}"
                )
            if len(starved) > 5:
                lines.append(f"    ... and {len(starved) - 5} more")
        else:
            lines.append("  No gradient-starved experts detected.")
        lines.append("")

        # Entropy summary
        if self.entropy_results:
            critical_entropy = [
                (name, r)
                for name, r in self.entropy_results.items()
                if getattr(r, "alert_level", None) in ("WARNING", "CRITICAL")
            ]
            lines.append(
                f"  Entropy alerts : {len(critical_entropy)} layer(s) with"
                " WARNING or CRITICAL entropy."
            )
        else:
            lines.append("  Entropy results not available.")
        lines.append("")

        # Cross-layer spread
        if self.cross_layer_results is not None:
            lines.append(
                "  Cross-layer correlation analysis: available  "
                f"({len(getattr(self.cross_layer_results, 'layer_names', []))}"
                " layers)"
            )
        else:
            lines.append(
                "  Cross-layer correlation analysis: not available."
            )

        lines.append(sep)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public API — serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the full report to a JSON-safe dictionary.

        All nested dataclasses and enums are recursively converted to
        primitives.  The ``to_dict()`` method of each sub-report is used
        where available to ensure consistent serialisation.

        Returns
        -------
        dict
            JSON-serialisable representation of the audit report.
        """
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "audit_datetime": self.audit_datetime.isoformat(),
            "num_batches": self.num_batches,
            "num_layers": self.num_layers,
            "dead_experts_count": self.dead_experts_count,
            "critical_layers": list(self.critical_layers),
            "entropy_results": {
                k: _to_serializable(v)
                for k, v in self.entropy_results.items()
            },
            "collapse_results": {
                k: _to_serializable(v)
                for k, v in self.collapse_results.items()
            },
            "gradient_results": {
                layer: {
                    str(getattr(gs_report, "expert_id", idx)): _to_serializable(gs_report)
                    for idx, gs_report in enumerate(expert_reports)
                }
                for layer, expert_reports in self.gradient_results.items()
            },
            "cross_layer_results": _to_serializable(self.cross_layer_results),
            "risk_scores": {
                k: _to_serializable(v) for k, v in self.risk_scores.items()
            },
        }

    def to_json(self, path: str, indent: int = 2) -> None:
        """Serialise the audit report to a JSON file at *path*.

        The file is written with UTF-8 encoding and pretty-printed with
        *indent* spaces.  Any existing file at *path* is overwritten.

        Parameters
        ----------
        path:
            Destination file path (e.g. ``"audit_2024.json"``).
        indent:
            JSON indentation level.  Defaults to ``2``.

        Raises
        ------
        OSError
            If the file cannot be written.
        """
        data = self.to_dict()
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=indent)
            logger.info("AuditReport saved to %s", path)
        except OSError as exc:
            logger.error("Failed to write AuditReport to %s: %s", path, exc)
            raise

    def to_dataframe(self) -> "Any":  # pandas.DataFrame
        """Export per-layer audit data as a pandas DataFrame.

        Each row corresponds to one MoE layer and contains: layer name,
        risk score, risk level, dead expert count, entropy (normalised),
        dominant signal, and gradient starvation flag.

        Returns
        -------
        pandas.DataFrame
            Tabular summary of all per-layer results.

        Raises
        ------
        ImportError
            If ``pandas`` is not installed.
        RuntimeError
            If no per-layer data is available (empty report).
        """
        try:
            import pandas as pd  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "pandas is required for AuditReport.to_dataframe(). "
                "Install it with: pip install pandas"
            ) from exc

        # Gather per-layer rows.
        all_layers = set(self.entropy_results) | set(self.collapse_results) | set(
            self.risk_scores
        )
        if not all_layers:
            raise RuntimeError(
                "AuditReport contains no per-layer data. "
                "Ensure audit() ran successfully."
            )

        rows: List[Dict[str, Any]] = []
        for layer_name in sorted(all_layers):
            row: Dict[str, Any] = {"layer_name": layer_name}

            # Risk score
            rr = self.risk_scores.get(layer_name)
            row["risk_score"] = rr.risk_score if rr is not None else None
            row["risk_level"] = (
                rr.risk_level.value if rr is not None else None
            )
            row["dominant_signal"] = (
                rr.dominant_signal if rr is not None else None
            )

            # Collapse
            cr = self.collapse_results.get(layer_name)
            if cr is not None:
                row["dead_expert_count"] = sum(
                    1
                    for es in cr.expert_states.values()
                    if es.status.value == "dead"
                )
                row["load_imbalance"] = getattr(cr, "load_imbalance", None)
            else:
                row["dead_expert_count"] = None
                row["load_imbalance"] = None

            # Entropy
            er = self.entropy_results.get(layer_name)
            if er is not None:
                row["entropy_norm"] = getattr(er, "entropy_norm", None)
                row["entropy_trend"] = getattr(er, "trend", None)
                row["entropy_alert"] = getattr(er, "alert_level", None)
            else:
                row["entropy_norm"] = None
                row["entropy_trend"] = None
                row["entropy_alert"] = None

            # Gradient starvation
            gs_list = self.gradient_results.get(layer_name, [])
            if gs_list:
                all_norms = [
                    getattr(gsr, "gradient_norm_mean", 0.0)
                    for gsr in gs_list
                ]
                row["min_expert_gradient_norm"] = min(all_norms)
                row["starved_expert_count"] = sum(
                    1 for n in all_norms if n < 0.01
                )
            else:
                row["min_expert_gradient_norm"] = None
                row["starved_expert_count"] = None

            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"AuditReport("
            f"model={self.model_name!r}, "
            f"layers={self.num_layers}, "
            f"dead_experts={self.dead_experts_count}, "
            f"critical={len(self.critical_layers)})"
        )


# ---------------------------------------------------------------------------
# Module-private helper
# ---------------------------------------------------------------------------


def _risk_bar(score: float, width: int = 20) -> str:
    """Render a simple ASCII progress bar for a risk score in [0, 1].

    Parameters
    ----------
    score:
        Risk score in [0.0, 1.0].
    width:
        Total bar width in characters.

    Returns
    -------
    str
        A bar string like ``"[##########          ]"``.
    """
    score = max(0.0, min(1.0, score))
    filled = int(round(score * width))
    empty = width - filled
    return f"[{'#' * filled}{'.' * empty}]"
