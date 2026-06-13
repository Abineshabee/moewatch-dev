# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# MoEWatch — moewatch/analyzer/risk_score.py
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
# Fuses the three empirically ordered precursor signals into a single
# per-layer collapse risk score in [0.0, 1.0]:
#
#   risk = w1 * T1 + w2 * T2 + w3 * T3
#
# where:
#   T1  = Tier 1 normalised gradient starvation signal (weight 0.6)
#   T2  = Tier 2 normalised entropy drift signal       (weight 0.3)
#   T3  = Tier 3 normalised cross-layer spread signal  (weight 0.1)
#   w1 > w2 > w3 (Tier 1 weighted heaviest — earliest precursor)
#
# Each tier signal is normalised to [0, 1] using the configured thresholds
# before the weighted sum is computed.  The result is then bucketed into
# one of four risk levels: LOW, MID, HIGH, CRITICAL.
#
# Explainability is preserved through the ``dominant_signal`` field and
# per-tier contribution values in the returned ``RiskReport``.
#
# Contents
# --------
#   RiskLevel          — enum: LOW / MID / HIGH / CRITICAL
#   RiskReport         — per-layer risk score dataclass
#   RiskScoreFuser     — fuses T1 + T2 + T3 → RiskReport
#
# Risk Level Thresholds
# ---------------------
#   score < 0.3   → LOW
#   score < 0.6   → MID
#   score < 0.8   → HIGH
#   score ≥ 0.8   → CRITICAL
#
# Dependencies
# ------------
#   moewatch.analyzer.gradient_starvation — GradientStarvationReport
#   moewatch.analyzer.entropy             — LayerEntropyReport
#   moewatch.analyzer.cross_layer         — CrossLayerReport
#   moewatch.config                       — WatchConfig
#   numpy
#
# Usage
# -----
#   fuser  = RiskScoreFuser(config)
#   report = fuser.fuse(gradient_report, entropy_report, cross_layer_report)
# -----------------------------------------------------------------------------

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import numpy as np

from moewatch.config import WatchConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RiskLevel enum
# ---------------------------------------------------------------------------


class RiskLevel(Enum):
    """Categorical collapse risk classification.

    Attributes
    ----------
    LOW
        Risk score < 0.3.  Routing is healthy; continue monitoring.
    MID
        0.3 ≤ score < 0.6.  Early warning; increase monitoring frequency.
    HIGH
        0.6 ≤ score < 0.8.  Significant risk; consider intervention.
    CRITICAL
        score ≥ 0.8.  Imminent collapse; intervention strongly recommended.
    """

    LOW = "low"
    MID = "mid"
    HIGH = "high"
    CRITICAL = "critical"


# Risk score thresholds separating the four levels.
_LEVEL_THRESHOLDS: Dict[float, RiskLevel] = {
    0.3: RiskLevel.LOW,
    0.6: RiskLevel.MID,
    0.8: RiskLevel.HIGH,
}

# Default fusion weights (must sum to 1.0).
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "tier1": 0.6,
    "tier2": 0.3,
    "tier3": 0.1,
}


# ---------------------------------------------------------------------------
# RiskReport dataclass
# ---------------------------------------------------------------------------


@dataclass
class RiskReport:
    """Per-layer fused collapse risk score and explainability breakdown.

    Attributes
    ----------
    layer_name : str
        Fully-qualified router module name.
    risk_score : float
        Weighted fusion of T1, T2, T3 signals, range [0.0, 1.0].
        0.0 = fully healthy; 1.0 = imminent collapse.
    risk_level : RiskLevel
        Categorical classification: LOW / MID / HIGH / CRITICAL.
    dominant_signal : str
        Which tier contributed most to the risk score.
        One of ``"gradient"`` (T1), ``"entropy"`` (T2),
        ``"cross_layer"`` (T3), or ``"none"`` if all contributions are
        zero.
    tier1_contribution : float
        Weighted Tier 1 contribution: ``w1 * T1_normalised``.
    tier2_contribution : float
        Weighted Tier 2 contribution: ``w2 * T2_normalised``.
    tier3_contribution : float
        Weighted Tier 3 contribution: ``w3 * T3_normalised``.
    step : int
        Training step associated with this risk assessment.
    """

    layer_name: str
    risk_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    dominant_signal: str = "none"
    tier1_contribution: float = 0.0
    tier2_contribution: float = 0.0
    tier3_contribution: float = 0.0
    step: int = 0

    def to_dict(self) -> Dict:
        """Serialise to a JSON-safe dictionary.

        Returns
        -------
        dict
            All fields as primitive types suitable for JSON output.
        """
        return {
            "layer_name": self.layer_name,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level.value,
            "dominant_signal": self.dominant_signal,
            "tier1_contribution": self.tier1_contribution,
            "tier2_contribution": self.tier2_contribution,
            "tier3_contribution": self.tier3_contribution,
            "step": self.step,
        }


# ---------------------------------------------------------------------------
# RiskScoreFuser
# ---------------------------------------------------------------------------


class RiskScoreFuser:
    """Fuses per-tier precursor signals into a unified per-layer risk score.

    Each call to ``fuse()`` accepts the three tier reports for a single layer
    and returns a ``RiskReport``.  The fuser also maintains a running cache
    of the most recent ``RiskReport`` per layer (keyed by ``layer_name``),
    accessible via ``get_latest_score()``.

    Parameters
    ----------
    config : WatchConfig
        Shared configuration.  Used to access signal thresholds that govern
        the per-tier normalisation.
    weights : dict[str, float] or None, optional
        Custom fusion weights.  Must be a dict with keys ``"tier1"``,
        ``"tier2"``, ``"tier3"`` and values that sum to 1.0 (within 1e-4
        tolerance).  If None (default), uses
        ``{tier1: 0.6, tier2: 0.3, tier3: 0.1}``.

    Raises
    ------
    ValueError
        If ``weights`` is provided but does not sum to ~1.0, or if any
        weight is negative.

    Attributes
    ----------
    config : WatchConfig
        Configuration reference.
    weights : dict[str, float]
        Active fusion weights (possibly overridden).
    """

    def __init__(
        self,
        config: WatchConfig,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.config: WatchConfig = config
        self.weights: Dict[str, float] = self._validate_weights(weights)

        # Cache of most recent RiskReport per layer.
        self._latest_reports: Dict[str, RiskReport] = {}

    # ------------------------------------------------------------------
    # Primary fusion method
    # ------------------------------------------------------------------

    def fuse(
        self,
        gradient_report: object,
        entropy_report: object,
        cross_layer_report: Optional[object] = None,
    ) -> RiskReport:
        """Fuse three tier signals into a single RiskReport for one layer.

        Normalises each signal to [0, 1] using the configured thresholds,
        applies the fusion weights, and classifies the resulting score.

        Parameters
        ----------
        gradient_report : GradientStarvationReport
            Tier 1 signal.  Must expose ``starvation_score`` (float [0,1]),
            ``layer_name`` (str), and ``step`` (int).
        entropy_report : LayerEntropyReport
            Tier 2 signal.  Must expose ``normalized_entropy`` (float [0,1]),
            ``drift_detected`` (bool), and ``step`` (int).
        cross_layer_report : CrossLayerReport or None, optional
            Tier 3 signal.  Must expose ``spread_score`` (float [0,1]) and
            ``victim_layers`` (list).  If None, Tier 3 contribution is 0.0
            and its weight is redistributed proportionally to Tier 1 and Tier 2.

        Returns
        -------
        RiskReport
            Fully populated risk report.

        Notes
        -----
        The fuser is tolerant of missing or partial tier data — any signal
        component that cannot be extracted defaults to 0.0.
        """
        # ------------------------------------------------------------------
        # Extract fields from tier reports (tolerant of duck-typed inputs)
        # ------------------------------------------------------------------
        layer_name: str = str(
            getattr(gradient_report, "layer_name", None)
            or getattr(entropy_report, "layer_name", "unknown")
        )
        step: int = max(
            int(getattr(gradient_report, "step", 0)),
            int(getattr(entropy_report, "step", 0)),
        )

        # Tier 1: gradient starvation score already normalised to [0, 1].
        t1_raw: float = float(
            np.clip(getattr(gradient_report, "starvation_score", 0.0), 0.0, 1.0)
        )

        # Tier 2: entropy drift.  Normalised entropy is LOWER = worse, so
        # we invert: T2 = 1 - normalized_entropy.  A value of 0 = perfectly
        # healthy; 1 = fully collapsed entropy.
        norm_entropy: float = float(
            np.clip(getattr(entropy_report, "normalized_entropy", 1.0), 0.0, 1.0)
        )
        drift_detected: bool = bool(getattr(entropy_report, "drift_detected", False))

        # Invert: healthy entropy (high value) → low T2 risk; collapsed → high.
        t2_raw: float = float(np.clip(1.0 - norm_entropy, 0.0, 1.0))

        # Boost T2 slightly when CUSUM drift detector has fired.
        if drift_detected:
            t2_raw = float(np.clip(t2_raw * 1.2, 0.0, 1.0))

        # Tier 3: spread score from cross-layer correlation.
        t3_raw: float = 0.0
        if cross_layer_report is not None:
            raw_spread = float(
                np.clip(getattr(cross_layer_report, "spread_score", 0.0), 0.0, 1.0)
            )
            # Boost T3 if this layer is a source layer in the report.
            source_layer = getattr(cross_layer_report, "source_layer", None)
            if source_layer == layer_name:
                raw_spread = float(np.clip(raw_spread * 1.5, 0.0, 1.0))
            t3_raw = raw_spread

        # ------------------------------------------------------------------
        # Resolve effective weights (redistribute T3 weight if no T3 data)
        # ------------------------------------------------------------------
        w1, w2, w3 = self._resolve_weights(t3_available=(cross_layer_report is not None))

        # ------------------------------------------------------------------
        # Weighted fusion
        # ------------------------------------------------------------------
        tier1_contrib = float(w1 * t1_raw)
        tier2_contrib = float(w2 * t2_raw)
        tier3_contrib = float(w3 * t3_raw)

        risk_score = float(
            np.clip(tier1_contrib + tier2_contrib + tier3_contrib, 0.0, 1.0)
        )

        # ------------------------------------------------------------------
        # Risk level classification
        # ------------------------------------------------------------------
        risk_level = self._classify_level(risk_score)

        # ------------------------------------------------------------------
        # Dominant signal identification
        # ------------------------------------------------------------------
        contributions = {
            "gradient": tier1_contrib,
            "entropy": tier2_contrib,
            "cross_layer": tier3_contrib,
        }
        max_contrib = max(contributions.values())
        if max_contrib <= 0.0:
            dominant_signal = "none"
        else:
            dominant_signal = max(contributions, key=contributions.__getitem__)

        # ------------------------------------------------------------------
        # Build report and cache it
        # ------------------------------------------------------------------
        report = RiskReport(
            layer_name=layer_name,
            risk_score=risk_score,
            risk_level=risk_level,
            dominant_signal=dominant_signal,
            tier1_contribution=tier1_contrib,
            tier2_contribution=tier2_contrib,
            tier3_contribution=tier3_contrib,
            step=step,
        )

        self._latest_reports[layer_name] = report

        logger.debug(
            "[MoEWatch] RiskScoreFuser: layer='%s' risk=%.3f [%s] "
            "T1=%.3f T2=%.3f T3=%.3f dominant='%s' step=%d",
            layer_name,
            risk_score,
            risk_level.value,
            tier1_contrib,
            tier2_contrib,
            tier3_contrib,
            dominant_signal,
            step,
        )

        return report

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def get_latest_score(self, layer_name: str) -> Optional[float]:
        """Return the most recently computed risk score for a layer.

        Parameters
        ----------
        layer_name : str
            Layer to query.

        Returns
        -------
        float or None
            Most recent ``risk_score`` in [0, 1], or None if no report
            has been computed for this layer yet.
        """
        report = self._latest_reports.get(layer_name)
        return report.risk_score if report is not None else None

    def get_latest_report(self, layer_name: str) -> Optional[RiskReport]:
        """Return the most recently computed RiskReport for a layer.

        Parameters
        ----------
        layer_name : str
            Layer to query.

        Returns
        -------
        RiskReport or None
        """
        return self._latest_reports.get(layer_name)

    def get_all_latest_reports(self) -> Dict[str, RiskReport]:
        """Return the most recent RiskReport for every known layer.

        Returns
        -------
        dict[str, RiskReport]
            Shallow copy of the internal report cache.
        """
        return dict(self._latest_reports)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_level(risk_score: float) -> RiskLevel:
        """Map a risk score to a categorical RiskLevel.

        Parameters
        ----------
        risk_score : float
            Value in [0, 1].

        Returns
        -------
        RiskLevel
        """
        if risk_score >= 0.8:
            return RiskLevel.CRITICAL
        if risk_score >= 0.6:
            return RiskLevel.HIGH
        if risk_score >= 0.3:
            return RiskLevel.MID
        return RiskLevel.LOW

    def _resolve_weights(
        self, t3_available: bool
    ) -> tuple:
        """Return effective (w1, w2, w3) fusion weights.

        When T3 data is unavailable, the T3 weight is redistributed to T1
        and T2 in proportion to their existing shares.

        Parameters
        ----------
        t3_available : bool
            Whether a valid cross-layer report was supplied.

        Returns
        -------
        tuple[float, float, float]
            Effective (w1, w2, w3) that sum to 1.0.
        """
        w1 = self.weights["tier1"]
        w2 = self.weights["tier2"]
        w3 = self.weights["tier3"]

        if not t3_available and w3 > 0.0:
            # Redistribute T3 weight proportionally to T1 and T2.
            total_12 = w1 + w2
            if total_12 > 0.0:
                w1 = w1 + w3 * (w1 / total_12)
                w2 = w2 + w3 * (w2 / total_12)
            w3 = 0.0

        return float(w1), float(w2), float(w3)

    @staticmethod
    def _validate_weights(
        weights: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        """Validate and normalise custom fusion weights.

        Parameters
        ----------
        weights : dict or None
            Custom weights dict or None to use defaults.

        Returns
        -------
        dict[str, float]
            Validated weights dict.

        Raises
        ------
        ValueError
            If weights are negative or do not sum to ~1.0.
        """
        if weights is None:
            return dict(_DEFAULT_WEIGHTS)

        required_keys = {"tier1", "tier2", "tier3"}
        missing = required_keys - weights.keys()
        if missing:
            raise ValueError(
                f"[MoEWatch] RiskScoreFuser: custom weights missing keys: "
                f"{missing}. Required: {required_keys}."
            )

        for key, val in weights.items():
            if val < 0.0:
                raise ValueError(
                    f"[MoEWatch] RiskScoreFuser: weight for '{key}' must be "
                    f">= 0.0, got {val}."
                )

        total = sum(weights[k] for k in required_keys)
        if abs(total - 1.0) > 1e-4:
            raise ValueError(
                f"[MoEWatch] RiskScoreFuser: fusion weights must sum to 1.0, "
                f"got {total:.6f}. Received: {weights}."
            )

        return {k: float(weights[k]) for k in required_keys}

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Clear the internal latest-report cache.

        After calling this, ``get_latest_score()`` will return None for all
        layers until ``fuse()`` is called again.
        """
        self._latest_reports.clear()

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"RiskScoreFuser("
            f"weights={self.weights}, "
            f"layers_cached={len(self._latest_reports)})"
        )
