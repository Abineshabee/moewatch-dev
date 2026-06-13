# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# MoEWatch — moewatch/analyzer/entropy.py
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
# Tier 2 signal: per-layer Shannon entropy drift detection. Monitors the
# routing distribution entropy across all detected MoE layers and identifies
# when the distribution is narrowing (routing collapse precursor, observed
# 30–100 steps before expert utilisation drops).
#
# Entropy is normalised against H_max = log2(n_experts) so scores are
# comparable across architectures with different expert counts.  A CUSUM
# detector (see cusum.py) triggers change-point alerts on the normalised
# entropy trajectory, providing a statistically principled alternative to
# naive threshold crossing.
#
# Contents
# --------
#   compute_entropy        — Shannon entropy of a probability vector (nats)
#   compute_entropy_norm   — entropy normalised to [0, 1]
#   LayerEntropyReport     — per-layer entropy snapshot dataclass
#   EntropyAnalyzer        — analyzes all layers; returns LayerEntropyReport
#
# Signal Hierarchy
# ----------------
#   Tier 1 — Gradient Starvation   (50–200 steps before collapse)
#   Tier 2 — Entropy Drift         (30–100 steps)  ← this file
#   Tier 3 — Cross-Layer Spread    (system-level)
#
# Dependencies
# ------------
#   moewatch.analyzer.cusum        — CUSUMDetector
#   moewatch.collector.stat_collector — StatCollector
#   moewatch.config                — WatchConfig
#   torch, numpy
#
# Usage
# -----
#   analyzer = EntropyAnalyzer(config)
#   reports  = analyzer.analyze(stat_collector)
#   # reports: dict[str, LayerEntropyReport]
# -----------------------------------------------------------------------------

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from moewatch.analyzer.cusum import CUSUMDetector
from moewatch.collector.stat_collector import StatCollector
from moewatch.config import WatchConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum number of entropy history points required before trend
# classification is attempted.  Below this count, trend is "UNKNOWN".
_MIN_TREND_HISTORY: int = 5

# Number of recent points used for the "recent" window in trend analysis.
_TREND_RECENT_WINDOW: int = 5

# Number of recent points used for the "long" window in trend analysis.
_TREND_LONG_WINDOW: int = 20

# Small epsilon to prevent log(0) in entropy computation.
_ENTROPY_EPS: float = 1e-10

# CUSUM parameters for the online entropy drift detector.
# Threshold tuned for normalised entropy (range [0,1]).
_CUSUM_THRESHOLD: float = 4.0
_CUSUM_DRIFT: float = 0.3


# ---------------------------------------------------------------------------
# Entropy computation helpers
# ---------------------------------------------------------------------------


def compute_entropy(probs: np.ndarray) -> float:
    """Compute Shannon entropy (in nats) of a probability distribution.

    Uses the convention 0 * log(0) = 0, which is the correct limit as
    p → 0+.  The input is expected to be a valid probability distribution
    (non-negative, summing to approximately 1.0), but the function is
    tolerant of unnormalised distributions — it normalises internally.

    Parameters
    ----------
    probs : numpy.ndarray
        1-D (or flattened) array of probabilities. Non-negative values
        only.  Will be normalised if they do not sum to 1.

    Returns
    -------
    float
        Shannon entropy H in nats. Returns 0.0 if the array is empty,
        all-zero, or contains only a single non-zero element.

    Examples
    --------
    >>> compute_entropy(np.array([0.25, 0.25, 0.25, 0.25]))
    1.3862943611198906   # log(4) nats
    >>> compute_entropy(np.array([1.0, 0.0, 0.0, 0.0]))
    0.0
    """
    arr = np.asarray(probs, dtype=np.float64).ravel()

    if arr.size == 0:
        return 0.0

    # Clamp negatives that may arise from floating-point precision errors.
    arr = np.clip(arr, 0.0, None)

    total = arr.sum()
    if total <= 0.0:
        return 0.0

    # Normalise so that we get a proper probability distribution.
    p = arr / total

    # Apply mask to avoid log(0): p=0 terms contribute 0 to entropy.
    mask = p > _ENTROPY_EPS
    entropy = -np.sum(p[mask] * np.log(p[mask]))

    return float(entropy)


def compute_entropy_norm(probs: np.ndarray, n_experts: int) -> float:
    """Compute Shannon entropy normalised to [0, 1].

    Divides the raw entropy by H_max = log2(n_experts), making the score
    architecture-agnostic and directly comparable across layers with
    different expert counts.

    Parameters
    ----------
    probs : numpy.ndarray
        Probability distribution (1-D, non-negative).
    n_experts : int
        Total number of experts (denominator for normalisation).
        Must be >= 2; if < 2, returns 0.0 to avoid division errors.

    Returns
    -------
    float
        Normalised entropy in [0, 1].  A value of 1.0 means perfectly
        uniform routing; 0.0 means routing is entirely collapsed onto one
        expert.
    """
    if n_experts < 2:
        return 0.0

    raw_entropy = compute_entropy(probs)
    h_max = math.log(n_experts)  # log in nats, consistent with compute_entropy

    if h_max <= 0.0:
        return 0.0

    return float(np.clip(raw_entropy / h_max, 0.0, 1.0))


def _softmax_to_probs(logits: torch.Tensor) -> np.ndarray:
    """Convert router logits to a mean probability distribution over experts.

    Accepts logits of shape ``[..., n_experts]`` (any number of leading
    batch / sequence dimensions) and:
      1. Computes softmax along the last dimension.
      2. Averages across all leading dimensions to obtain a single
         per-expert mean probability vector of shape ``[n_experts]``.

    Parameters
    ----------
    logits : torch.Tensor
        Raw router logits.

    Returns
    -------
    numpy.ndarray
        1-D float64 array of shape ``[n_experts]``.  Returns uniform
        distribution if logits has no expert dimension.
    """
    with torch.no_grad():
        if logits.ndim < 1 or logits.shape[-1] < 1:
            return np.array([1.0], dtype=np.float64)

        probs = torch.softmax(logits.float(), dim=-1)

        # Average over all batch/token dimensions.
        if probs.ndim > 1:
            probs = probs.reshape(-1, probs.shape[-1]).mean(dim=0)

        return probs.cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# LayerEntropyReport dataclass
# ---------------------------------------------------------------------------


@dataclass
class LayerEntropyReport:
    """Per-layer entropy analysis snapshot produced by EntropyAnalyzer.

    Attributes
    ----------
    layer_name : str
        Fully-qualified router module name.
    current_entropy : float
        Raw Shannon entropy (nats) of the most-recent routing distribution.
    normalized_entropy : float
        ``current_entropy / log(n_experts)``.  Range [0, 1].
        1.0 = perfectly uniform routing; 0.0 = fully collapsed.
    trend : str
        Qualitative trend classification for the recent entropy trajectory:
        ``"DECLINING"``, ``"STABLE"``, ``"IMPROVING"``, or ``"UNKNOWN"``
        (insufficient history).
    drift_detected : bool
        True if the CUSUM detector has fired on this layer's normalised
        entropy sequence, signalling a statistically significant change.
    step : int
        Training step of the most recent event included in this snapshot.
        0 if no events have been collected for this layer yet.
    entropy_history : list[float]
        Recent normalised entropy values in chronological order (up to
        the last ``_TREND_LONG_WINDOW`` values). Exposed for downstream
        consumers such as CrossLayerCorrelation.
    drop_rate : float
        Mean per-step change in normalised entropy over the recent trend
        window (negative = declining).  ``0.0`` if history is too short.
    n_experts : int
        Expert count inferred from the routing logits shape.  0 if unknown.
    """

    layer_name: str
    current_entropy: float = 0.0
    normalized_entropy: float = 0.0
    trend: str = "UNKNOWN"
    drift_detected: bool = False
    step: int = 0
    entropy_history: List[float] = field(default_factory=list)
    drop_rate: float = 0.0
    n_experts: int = 0


# ---------------------------------------------------------------------------
# EntropyAnalyzer
# ---------------------------------------------------------------------------


class EntropyAnalyzer:
    """Tier 2 signal: per-layer routing entropy drift detection.

    Consumes routing logits collected by ``StatCollector`` and produces a
    ``LayerEntropyReport`` per layer at each ``analyze()`` call.  Maintains
    a per-layer ``CUSUMDetector`` instance to track drift across successive
    calls, so entropy trends accumulate across training steps rather than
    being computed from scratch each time.

    The analyzer is stateful: it is designed to be called repeatedly during
    training (e.g., via ``MoEWatch.step()``) with the same ``StatCollector``
    instance.  Calling ``reset()`` clears all accumulated history.

    Parameters
    ----------
    config : WatchConfig
        Shared configuration.  Relevant fields:
          ``entropy_warn``, ``entropy_critical``, ``entropy_drop_warn``.

    Attributes
    ----------
    config : WatchConfig
        Configuration reference.
    """

    def __init__(self, config: WatchConfig) -> None:
        self.config: WatchConfig = config

        # Per-layer normalised entropy history (most recent values, used
        # for trend analysis and as input to CrossLayerCorrelation).
        self._entropy_history: Dict[str, List[float]] = {}

        # Per-layer CUSUM detector instances (stateful, persistent).
        self._cusum_detectors: Dict[str, CUSUMDetector] = {}

        # Per-layer cached expert count (inferred from first event shape).
        self._expert_counts: Dict[str, int] = {}

        # Per-layer latest step number seen.
        self._latest_steps: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Primary analysis method
    # ------------------------------------------------------------------

    def analyze(self, stat_collector: StatCollector) -> Dict[str, LayerEntropyReport]:
        """Analyze all registered layers and return per-layer entropy reports.

        Iterates over every layer registered in ``stat_collector``, reads the
        recent routing logits window, computes the mean routing distribution,
        derives entropy metrics, updates the CUSUM detector, classifies the
        trend, and returns the results as a dict.

        Layers with no events yet (empty logits window) produce a report with
        all-default values and ``trend = "UNKNOWN"``, so downstream consumers
        always receive a complete dict keyed by all known layer names.

        Parameters
        ----------
        stat_collector : StatCollector
            Source of ``LayerStats`` (routing logits and expert counts).

        Returns
        -------
        dict[str, LayerEntropyReport]
            One report per registered layer.  Keys are the fully-qualified
            layer (router module) names.

        Notes
        -----
        Per-layer failures are caught and logged at WARNING level rather than
        propagating, so a single malformed layer cannot block analysis of the
        remaining layers.
        """
        reports: Dict[str, LayerEntropyReport] = {}

        # Retrieve current stats snapshot (thread-safe copy).
        try:
            all_stats = stat_collector.get_all_stats()
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "[MoEWatch] EntropyAnalyzer.analyze(): failed to read stats "
                "from StatCollector: %s",
                exc,
            )
            return reports

        routing_stats = all_stats.get("routing", {})

        for layer_name, layer_stats in routing_stats.items():
            try:
                report = self._analyze_layer(layer_name, layer_stats)
                reports[layer_name] = report
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "[MoEWatch] EntropyAnalyzer: error analyzing layer '%s' "
                    "(producing default report): %s",
                    layer_name,
                    exc,
                )
                reports[layer_name] = LayerEntropyReport(
                    layer_name=layer_name,
                    step=self._latest_steps.get(layer_name, 0),
                )

        return reports

    # ------------------------------------------------------------------
    # Internal: single-layer analysis
    # ------------------------------------------------------------------

    def _analyze_layer(
        self,
        layer_name: str,
        layer_stats: object,
    ) -> LayerEntropyReport:
        """Compute entropy report for one layer.

        Parameters
        ----------
        layer_name : str
            Layer name.
        layer_stats : LayerStats
            Statistics snapshot from StatCollector.

        Returns
        -------
        LayerEntropyReport
            Fully populated report.
        """
        # ------------------------------------------------------------------
        # Initialise per-layer state if first encounter
        # ------------------------------------------------------------------
        if layer_name not in self._cusum_detectors:
            self._cusum_detectors[layer_name] = CUSUMDetector(
                threshold=_CUSUM_THRESHOLD,
                drift=_CUSUM_DRIFT,
            )
            self._entropy_history[layer_name] = []
            self._expert_counts[layer_name] = 0
            self._latest_steps[layer_name] = 0

        step = int(getattr(layer_stats, "step", 0))
        raw_logits_window = getattr(layer_stats, "raw_logits_window", None)
        expert_util = getattr(layer_stats, "expert_utilization", None)

        # ------------------------------------------------------------------
        # Determine expert count
        # ------------------------------------------------------------------
        n_experts = self._expert_counts.get(layer_name, 0)

        # Prefer utilization tensor for expert count (most reliable).
        if expert_util is not None and hasattr(expert_util, "shape"):
            n_experts = int(expert_util.shape[-1])
        elif raw_logits_window is not None and hasattr(raw_logits_window, "shape"):
            if raw_logits_window.ndim >= 1 and raw_logits_window.shape[-1] > 1:
                n_experts = int(raw_logits_window.shape[-1])

        if n_experts > 0:
            self._expert_counts[layer_name] = n_experts

        # ------------------------------------------------------------------
        # Compute entropy from routing distribution
        # ------------------------------------------------------------------
        current_entropy: float = 0.0
        normalized_entropy: float = 0.0

        if expert_util is not None and hasattr(expert_util, "cpu"):
            # Use pre-computed expert utilization (most efficient path).
            probs_np = expert_util.cpu().float().numpy().astype(np.float64)
            if probs_np.ndim > 1:
                probs_np = probs_np.mean(axis=0)
            current_entropy = compute_entropy(probs_np)
            if n_experts >= 2:
                normalized_entropy = compute_entropy_norm(probs_np, n_experts)

        elif (
            raw_logits_window is not None
            and hasattr(raw_logits_window, "shape")
            and raw_logits_window.numel() > 0
        ):
            # Fall back to logits window.
            # Take the mean distribution across the entire window.
            if raw_logits_window.ndim >= 2 and raw_logits_window.shape[-1] > 1:
                probs_np = _softmax_to_probs(raw_logits_window)
                current_entropy = compute_entropy(probs_np)
                if n_experts >= 2:
                    normalized_entropy = compute_entropy_norm(probs_np, n_experts)

        # ------------------------------------------------------------------
        # Update CUSUM and entropy history
        # ------------------------------------------------------------------
        drift_detected = False
        if normalized_entropy > 0.0 or len(self._entropy_history[layer_name]) > 0:
            history = self._entropy_history[layer_name]
            history.append(normalized_entropy)

            # Bound history length to prevent unbounded growth.
            if len(history) > _TREND_LONG_WINDOW * 4:
                self._entropy_history[layer_name] = history[-(_TREND_LONG_WINDOW * 2):]

            # CUSUM expects values that are "too low" = signal of collapse.
            # We feed the negative entropy shift so that a *drop* in entropy
            # triggers the positive CUSUM direction.
            drift_detected = self._cusum_detectors[layer_name].update(
                normalized_entropy
            )

            self._latest_steps[layer_name] = step

        # ------------------------------------------------------------------
        # Trend classification
        # ------------------------------------------------------------------
        trend, drop_rate = self._classify_trend(layer_name)

        # ------------------------------------------------------------------
        # Build and return report
        # ------------------------------------------------------------------
        return LayerEntropyReport(
            layer_name=layer_name,
            current_entropy=current_entropy,
            normalized_entropy=normalized_entropy,
            trend=trend,
            drift_detected=drift_detected,
            step=step,
            entropy_history=list(self._entropy_history.get(layer_name, [])),
            drop_rate=drop_rate,
            n_experts=n_experts,
        )

    # ------------------------------------------------------------------
    # Internal: trend classification
    # ------------------------------------------------------------------

    def _classify_trend(self, layer_name: str) -> Tuple[str, float]:
        """Classify entropy trend and compute per-step drop rate.

        Compares the mean of the most recent ``_TREND_RECENT_WINDOW``
        values against the mean of the preceding ``_TREND_LONG_WINDOW``
        values to decide direction.

        Parameters
        ----------
        layer_name : str
            Layer to classify.

        Returns
        -------
        trend : str
            ``"DECLINING"`` | ``"STABLE"`` | ``"IMPROVING"`` | ``"UNKNOWN"``
        drop_rate : float
            Mean per-step change (negative = declining).
        """
        history = self._entropy_history.get(layer_name, [])

        if len(history) < _MIN_TREND_HISTORY:
            return "UNKNOWN", 0.0

        recent = np.array(history[-_TREND_RECENT_WINDOW:], dtype=np.float64)
        recent_mean = float(recent.mean())

        # Per-step slope over recent window using finite differences.
        if len(recent) >= 2:
            drop_rate = float(np.diff(recent).mean())
        else:
            drop_rate = 0.0

        # Use a longer reference window if sufficient history is available.
        if len(history) >= _TREND_LONG_WINDOW:
            prior = np.array(
                history[-_TREND_LONG_WINDOW : -_TREND_RECENT_WINDOW],
                dtype=np.float64,
            )
        else:
            prior = np.array(history[: max(1, len(history) - _TREND_RECENT_WINDOW)], dtype=np.float64)

        if prior.size == 0:
            return "UNKNOWN", drop_rate

        prior_mean = float(prior.mean())
        delta = recent_mean - prior_mean

        # Classify based on delta relative to entropy_drop_warn threshold.
        drop_threshold = self.config.entropy_drop_warn
        if delta < -drop_threshold:
            trend = "DECLINING"
        elif delta > drop_threshold * 0.5:
            trend = "IMPROVING"
        else:
            trend = "STABLE"

        return trend, drop_rate

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def reset(self, layer_name: Optional[str] = None) -> None:
        """Reset accumulated state for one or all layers.

        Parameters
        ----------
        layer_name : str, optional
            If provided, resets only the specified layer. If None (default),
            all layers are reset.
        """
        if layer_name is not None:
            self._entropy_history.pop(layer_name, None)
            self._cusum_detectors.pop(layer_name, None)
            self._expert_counts.pop(layer_name, None)
            self._latest_steps.pop(layer_name, None)
        else:
            self._entropy_history.clear()
            self._cusum_detectors.clear()
            self._expert_counts.clear()
            self._latest_steps.clear()

    def get_entropy_history(self, layer_name: str) -> List[float]:
        """Return the stored normalised entropy history for a layer.

        Parameters
        ----------
        layer_name : str
            Layer to query.

        Returns
        -------
        list[float]
            Copy of the stored history (oldest first).  Empty list if
            ``layer_name`` is unregistered or has no events yet.
        """
        return list(self._entropy_history.get(layer_name, []))

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_layers = len(self._cusum_detectors)
        return (
            f"EntropyAnalyzer("
            f"layers={n_layers}, "
            f"config=({self.config.entropy_warn}/{self.config.entropy_critical}))"
        )
