# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# MoEWatch — moewatch/analyzer/cross_layer.py
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
# Tier 3 signal: cross-layer routing entropy correlation analysis.
# Maintains a sliding window of per-layer normalised entropy sequences and
# computes a pairwise Pearson correlation matrix.  Identifies the source layer
# (highest entropy drop rate) and downstream victim layers (high correlation
# with source), estimates propagation velocity, and packages the result into
# a CrossLayerReport for consumption by RiskScoreFuser.
#
# Tier 3 is a localisation signal — it does not predict collapse earlier than
# Tier 1/2, but tells the system WHERE collapse is spreading and at what rate,
# enabling targeted per-layer interventions rather than global ones.
#
# Contents
# --------
#   CrossLayerReport        — analysis result dataclass
#   CrossLayerCorrelation   — analyzer class
#
# Algorithm Overview
# ------------------
#   1. Ingest per-layer normalised entropy from EntropyAnalyzer reports.
#   2. Maintain a sliding window (default 100 steps) per layer.
#   3. When ≥ min_layers have sufficient history:
#      a. Assemble matrix M[t, l] = entropy at time t for layer l.
#      b. Compute Pearson correlation matrix C[i, j] = corr(M[:, i], M[:, j]).
#      c. Identify source layer: highest absolute entropy drop rate
#         (most negative slope of linear fit on entropy history).
#      d. Identify victims: layers with |C[source, l]| > correlation_threshold.
#      e. Estimate propagation velocity from correlation lag structure.
#   4. Return CrossLayerReport.
#
# Dependencies
# ------------
#   moewatch.analyzer.entropy   — LayerEntropyReport (as input, not imported
#                                 directly to avoid circular import)
#   moewatch.config             — WatchConfig
#   numpy
#
# Usage
# -----
#   # Construct once; call analyze() at each step.
#   analyzer = CrossLayerCorrelation(config)
#   report   = analyzer.analyze(entropy_reports)
#   # entropy_reports: dict[str, LayerEntropyReport]
# -----------------------------------------------------------------------------

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from moewatch.config import WatchConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum number of layers required before cross-layer analysis is attempted.
_MIN_LAYERS_FOR_ANALYSIS: int = 2

# Minimum number of entropy samples per layer before that layer is included
# in the correlation computation.
_MIN_HISTORY_LENGTH: int = 10

# Default sliding window size (number of most-recent steps retained).
_DEFAULT_WINDOW_SIZE: int = 100

# Pearson correlation coefficient threshold above which a layer is considered
# a "victim" of the identified source layer's collapse spread.
_CORRELATION_VICTIM_THRESHOLD: float = 0.6

# Minimum absolute slope magnitude (nats/step) to classify a layer as the
# source layer rather than a bystander.
_MIN_SOURCE_SLOPE: float = 0.001


# ---------------------------------------------------------------------------
# CrossLayerReport dataclass
# ---------------------------------------------------------------------------


@dataclass
class CrossLayerReport:
    """Cross-layer routing entropy correlation analysis result.

    Attributes
    ----------
    source_layer : str or None
        Name of the layer with the steepest entropy decline rate (most
        likely initiator of a spreading collapse), or None if insufficient
        data or no layer shows a significant drop.
    victim_layers : list[str]
        Names of layers with high Pearson correlation to the source
        (|r| >= ``_CORRELATION_VICTIM_THRESHOLD``), excluding the source
        itself.  Empty if no source was identified or no victims qualify.
    correlation_matrix : numpy.ndarray
        Square pairwise Pearson correlation matrix of shape
        ``[n_layers, n_layers]``.  Row/column ordering matches
        ``layer_order`` (the order of layers passed into the analyzer).
        A 0×0 array if insufficient data for computation.
    layer_order : list[str]
        Layer names in the order that rows/columns of ``correlation_matrix``
        correspond to.
    propagation_velocity : float or None
        Estimated number of training steps per layer for collapse
        propagation from source to victims, or None if not computable
        (fewer than 2 victims, or lag detection failed).
    spread_score : float
        Overall normalised spread severity in [0, 1].  0.0 = no spread;
        1.0 = strong correlated collapse across all layers.
        Computed as ``num_victims / max(n_layers - 1, 1)``.
    step : int
        Training step of the most recently ingested entropy observation.
    """

    source_layer: Optional[str] = None
    victim_layers: List[str] = field(default_factory=list)
    correlation_matrix: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float64)
    )
    layer_order: List[str] = field(default_factory=list)
    propagation_velocity: Optional[float] = None
    spread_score: float = 0.0
    step: int = 0


# ---------------------------------------------------------------------------
# CrossLayerCorrelation
# ---------------------------------------------------------------------------


class CrossLayerCorrelation:
    """Tier 3 signal: cross-layer entropy correlation and spread detection.

    Maintains a sliding window of per-layer normalised entropy values and
    computes the Pearson correlation structure to identify which layer is
    the collapse source and which are victims.

    The analyzer is designed to be called incrementally via
    ``analyze(entropy_reports)`` at each training step, passing in the dict
    returned by ``EntropyAnalyzer.analyze()``.  It accumulates its own
    internal entropy history rather than relying on the EntropyAnalyzer's
    stored history, so it remains correct even if the EntropyAnalyzer is
    reset independently.

    Parameters
    ----------
    config : WatchConfig
        Shared configuration.

    Attributes
    ----------
    config : WatchConfig
        Configuration reference.
    """

    def __init__(self, config: WatchConfig) -> None:
        self.config: WatchConfig = config

        # Per-layer sliding window of normalised entropy values.
        # {layer_name: [entropy_t0, entropy_t1, ...]}
        self._entropy_sequences: Dict[str, List[float]] = {}

        # Per-layer step index corresponding to each entropy value in the
        # sequence (used for lag detection).
        self._step_sequences: Dict[str, List[int]] = {}

        # Current window size (may be changed via the config in future).
        self._window_size: int = _DEFAULT_WINDOW_SIZE

        # Layer names in stable order (established on first call with ≥ 2 layers).
        self._layer_order: List[str] = []

        # Step of the most recently ingested data point.
        self._latest_step: int = 0

    # ------------------------------------------------------------------
    # Primary analysis method
    # ------------------------------------------------------------------

    def analyze(
        self,
        entropy_reports: Dict[str, object],
    ) -> CrossLayerReport:
        """Update internal entropy history and return cross-layer analysis.

        Parameters
        ----------
        entropy_reports : dict[str, LayerEntropyReport]
            Output of ``EntropyAnalyzer.analyze()``.  Keys are layer names;
            values must expose at minimum:
              ``normalized_entropy`` (float), ``step`` (int).

        Returns
        -------
        CrossLayerReport
            Analysis result.  If fewer than ``_MIN_LAYERS_FOR_ANALYSIS``
            layers have sufficient history, most fields will be empty/default.

        Notes
        -----
        The function is tolerant of partial data: layers that have not yet
        accumulated ``_MIN_HISTORY_LENGTH`` values are silently excluded from
        the correlation computation without affecting reports for other layers.
        """
        if not entropy_reports:
            return CrossLayerReport(step=self._latest_step)

        # ------------------------------------------------------------------
        # 1. Ingest new entropy values into sliding windows.
        # ------------------------------------------------------------------
        self._ingest_entropy_reports(entropy_reports)

        # ------------------------------------------------------------------
        # 2. Determine which layers are eligible for correlation analysis.
        # ------------------------------------------------------------------
        eligible_layers = self._get_eligible_layers()

        if len(eligible_layers) < _MIN_LAYERS_FOR_ANALYSIS:
            return CrossLayerReport(step=self._latest_step)

        # ------------------------------------------------------------------
        # 3. Compute correlation matrix.
        # ------------------------------------------------------------------
        corr_matrix, layer_order = self._compute_correlation_matrix(eligible_layers)

        if corr_matrix.shape[0] < _MIN_LAYERS_FOR_ANALYSIS:
            return CrossLayerReport(step=self._latest_step)

        # ------------------------------------------------------------------
        # 4. Identify source layer (steepest entropy decline rate).
        # ------------------------------------------------------------------
        source_layer, source_idx = self._identify_source_layer(layer_order)

        # ------------------------------------------------------------------
        # 5. Identify victim layers (high correlation with source).
        # ------------------------------------------------------------------
        victim_layers: List[str] = []
        if source_layer is not None and source_idx is not None:
            victim_layers = self._identify_victims(
                corr_matrix=corr_matrix,
                source_idx=source_idx,
                layer_order=layer_order,
                source_layer=source_layer,
            )

        # ------------------------------------------------------------------
        # 6. Estimate propagation velocity.
        # ------------------------------------------------------------------
        propagation_velocity: Optional[float] = None
        if source_layer is not None and len(victim_layers) >= 1:
            propagation_velocity = self._estimate_propagation_velocity(
                source_layer=source_layer,
                victim_layers=victim_layers,
            )

        # ------------------------------------------------------------------
        # 7. Compute spread score.
        # ------------------------------------------------------------------
        n_layers = len(layer_order)
        spread_score = (
            float(len(victim_layers)) / max(n_layers - 1, 1)
            if n_layers > 1
            else 0.0
        )

        return CrossLayerReport(
            source_layer=source_layer,
            victim_layers=victim_layers,
            correlation_matrix=corr_matrix,
            layer_order=layer_order,
            propagation_velocity=propagation_velocity,
            spread_score=float(np.clip(spread_score, 0.0, 1.0)),
            step=self._latest_step,
        )

    # ------------------------------------------------------------------
    # Internal: data ingestion
    # ------------------------------------------------------------------

    def _ingest_entropy_reports(
        self,
        entropy_reports: Dict[str, object],
    ) -> None:
        """Update per-layer sliding windows from entropy reports.

        Parameters
        ----------
        entropy_reports : dict[str, LayerEntropyReport]
            New entropy values to incorporate.
        """
        for layer_name, report in entropy_reports.items():
            norm_ent = float(getattr(report, "normalized_entropy", 0.0))
            step = int(getattr(report, "step", 0))

            if layer_name not in self._entropy_sequences:
                self._entropy_sequences[layer_name] = []
                self._step_sequences[layer_name] = []

            self._entropy_sequences[layer_name].append(norm_ent)
            self._step_sequences[layer_name].append(step)

            # Enforce sliding window bound.
            if len(self._entropy_sequences[layer_name]) > self._window_size:
                self._entropy_sequences[layer_name] = (
                    self._entropy_sequences[layer_name][-self._window_size :]
                )
                self._step_sequences[layer_name] = (
                    self._step_sequences[layer_name][-self._window_size :]
                )

            if step > self._latest_step:
                self._latest_step = step

    # ------------------------------------------------------------------
    # Internal: eligible layer selection
    # ------------------------------------------------------------------

    def _get_eligible_layers(self) -> List[str]:
        """Return layers with sufficient history for correlation analysis.

        Returns
        -------
        list[str]
            Layer names with at least ``_MIN_HISTORY_LENGTH`` values,
            sorted deterministically by layer name.
        """
        eligible = [
            name
            for name, history in self._entropy_sequences.items()
            if len(history) >= _MIN_HISTORY_LENGTH
        ]
        # Sort for stable ordering across calls.
        return sorted(eligible)

    # ------------------------------------------------------------------
    # Internal: correlation matrix computation
    # ------------------------------------------------------------------

    def _compute_correlation_matrix(
        self,
        eligible_layers: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """Build entropy matrix and compute Pearson correlations.

        Parameters
        ----------
        eligible_layers : list[str]
            Layers to include.

        Returns
        -------
        corr_matrix : numpy.ndarray
            Pearson correlation matrix, shape [n, n].
        layer_order : list[str]
            Row/column order.
        """
        n = len(eligible_layers)

        # Determine the common window length (minimum across all eligible layers
        # so that the matrix is rectangular).
        min_len = min(
            len(self._entropy_sequences[name]) for name in eligible_layers
        )
        min_len = max(min_len, _MIN_HISTORY_LENGTH)

        # Assemble matrix M[time, layer].
        entropy_matrix = np.zeros((min_len, n), dtype=np.float64)
        for col_idx, layer_name in enumerate(eligible_layers):
            seq = self._entropy_sequences[layer_name]
            # Take the most recent min_len values.
            entropy_matrix[:, col_idx] = np.array(seq[-min_len:], dtype=np.float64)

        # Handle constant sequences (zero variance → correlation is undefined).
        # Replace constant columns with NaN-safe uniform random jitter so that
        # numpy.corrcoef does not produce NaN/Inf; we then clamp those entries
        # to 0.0 post-computation.
        std_per_layer = entropy_matrix.std(axis=0)
        constant_mask = std_per_layer < 1e-10

        if constant_mask.all():
            # All layers constant — no meaningful correlation to compute.
            return np.eye(n, dtype=np.float64), list(eligible_layers)

        # Add tiny jitter to constant columns to prevent divide-by-zero inside
        # numpy.corrcoef.  Resulting correlations for those columns will be near
        # zero and will be clamped anyway.
        rng = np.random.default_rng(seed=0)  # deterministic seed for reproducibility
        for col_idx in np.where(constant_mask)[0]:
            entropy_matrix[:, col_idx] += rng.normal(0, 1e-8, size=min_len)

        try:
            # numpy.corrcoef returns [n, n] correlation matrix.
            corr_matrix = np.corrcoef(entropy_matrix.T)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "[MoEWatch] CrossLayerCorrelation: correlation computation "
                "failed (%s). Returning identity matrix.",
                exc,
            )
            return np.eye(n, dtype=np.float64), list(eligible_layers)

        # Clamp out-of-range values (can arise from floating-point precision).
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)

        # Zero out correlations for constant-column pairs (meaningless).
        for col_idx in np.where(constant_mask)[0]:
            corr_matrix[col_idx, :] = 0.0
            corr_matrix[:, col_idx] = 0.0

        return corr_matrix, list(eligible_layers)

    # ------------------------------------------------------------------
    # Internal: source layer identification
    # ------------------------------------------------------------------

    def _identify_source_layer(
        self,
        layer_order: List[str],
    ) -> Tuple[Optional[str], Optional[int]]:
        """Identify the layer with the steepest normalised entropy decline.

        Fits a linear trend to each layer's entropy history and selects the
        layer with the most negative slope (fastest declining entropy).

        Parameters
        ----------
        layer_order : list[str]
            Eligible layers to consider.

        Returns
        -------
        source_layer : str or None
            Name of the source layer.
        source_idx : int or None
            Index of source layer in ``layer_order``.
        """
        slopes: List[float] = []

        for layer_name in layer_order:
            seq = self._entropy_sequences.get(layer_name, [])
            if len(seq) < 2:
                slopes.append(0.0)
                continue

            x = np.arange(len(seq), dtype=np.float64)
            y = np.array(seq, dtype=np.float64)

            try:
                slope, _ = np.polyfit(x, y, deg=1)
            except Exception:  # pylint: disable=broad-except
                slope = 0.0

            slopes.append(float(slope))

        if not slopes:
            return None, None

        # Source layer = most negative slope (fastest entropy decline).
        min_slope_idx = int(np.argmin(slopes))
        min_slope = slopes[min_slope_idx]

        # Only report a source if the slope is meaningfully negative.
        if min_slope > -_MIN_SOURCE_SLOPE:
            return None, None

        return layer_order[min_slope_idx], min_slope_idx

    # ------------------------------------------------------------------
    # Internal: victim identification
    # ------------------------------------------------------------------

    def _identify_victims(
        self,
        corr_matrix: np.ndarray,
        source_idx: int,
        layer_order: List[str],
        source_layer: str,
    ) -> List[str]:
        """Identify layers highly correlated with the source layer's decline.

        Parameters
        ----------
        corr_matrix : numpy.ndarray
            Pearson correlation matrix.
        source_idx : int
            Row/column index of the source layer.
        layer_order : list[str]
            Layer names parallel to rows/columns of corr_matrix.
        source_layer : str
            Name of the source layer (excluded from victims).

        Returns
        -------
        list[str]
            Victim layer names, in descending order of absolute correlation
            with the source.
        """
        n = len(layer_order)
        source_row = corr_matrix[source_idx, :]

        victim_candidates: List[Tuple[float, str]] = []
        for col_idx in range(n):
            if col_idx == source_idx:
                continue
            corr_val = abs(float(source_row[col_idx]))
            if corr_val >= _CORRELATION_VICTIM_THRESHOLD:
                victim_candidates.append((corr_val, layer_order[col_idx]))

        # Sort by descending correlation (highest correlation = primary victim).
        victim_candidates.sort(key=lambda t: -t[0])
        return [name for _, name in victim_candidates]

    # ------------------------------------------------------------------
    # Internal: propagation velocity estimation
    # ------------------------------------------------------------------

    def _estimate_propagation_velocity(
        self,
        source_layer: str,
        victim_layers: List[str],
    ) -> Optional[float]:
        """Estimate collapse propagation velocity in steps per layer.

        Measures the lag between the source entropy decline and each victim's
        decline using cross-correlation (argmax of the cross-correlation
        function between the source entropy sequence and each victim sequence).
        The mean lag across all victims, divided by some spatial unit, gives
        a rough propagation velocity estimate.

        Parameters
        ----------
        source_layer : str
            Source layer name.
        victim_layers : list[str]
            Victim layer names.

        Returns
        -------
        float or None
            Estimated steps per layer, or None if not computable.
        """
        source_seq = self._entropy_sequences.get(source_layer, [])
        if len(source_seq) < _MIN_HISTORY_LENGTH:
            return None

        src = np.array(source_seq, dtype=np.float64)
        # Normalise for cross-correlation.
        src_std = src.std()
        if src_std < 1e-10:
            return None
        src_norm = (src - src.mean()) / src_std

        lags: List[float] = []
        for victim_name in victim_layers:
            vic_seq = self._entropy_sequences.get(victim_name, [])
            if len(vic_seq) < _MIN_HISTORY_LENGTH:
                continue

            # Align to same length.
            min_len = min(len(src_norm), len(vic_seq))
            s = src_norm[-min_len:]
            v = np.array(vic_seq[-min_len:], dtype=np.float64)
            v_std = v.std()
            if v_std < 1e-10:
                continue
            v_norm = (v - v.mean()) / v_std

            # Full cross-correlation using numpy.
            xcorr = np.correlate(s, v_norm, mode="full")
            # Lag 0 is at index (min_len - 1).
            lag_idx = int(np.argmax(xcorr))
            lag_steps = lag_idx - (min_len - 1)

            # Only count positive lags (source precedes victim).
            if lag_steps > 0:
                lags.append(float(lag_steps))

        if not lags:
            return None

        # Mean lag across victims / 1 layer spacing = steps per layer.
        return float(np.mean(lags))

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def reset(self, layer_name: Optional[str] = None) -> None:
        """Clear stored entropy sequences for one or all layers.

        Parameters
        ----------
        layer_name : str, optional
            Layer to clear.  If None, clears all layers.
        """
        if layer_name is not None:
            self._entropy_sequences.pop(layer_name, None)
            self._step_sequences.pop(layer_name, None)
        else:
            self._entropy_sequences.clear()
            self._step_sequences.clear()
            self._latest_step = 0

    def get_entropy_window(self, layer_name: str) -> List[float]:
        """Return the stored normalised entropy window for a layer.

        Parameters
        ----------
        layer_name : str
            Layer to query.

        Returns
        -------
        list[float]
            Copy of the stored window (oldest first).
        """
        return list(self._entropy_sequences.get(layer_name, []))

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_layers = len(self._entropy_sequences)
        return (
            f"CrossLayerCorrelation("
            f"layers_tracked={n_layers}, "
            f"window_size={self._window_size}, "
            f"latest_step={self._latest_step})"
        )
