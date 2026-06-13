# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# MoEWatch — moewatch/analyzer/cusum.py
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
# Cumulative Sum (CUSUM) change-point detection for streaming time series.
# Used by EntropyAnalyzer and GradientStarvationAnalyzer to detect statistically
# principled drift in signal sequences. Replaces naive recent-vs-prior window
# thresholding with a sequential hypothesis test that controls false alarm rate.
#
# Contents
# --------
#   detect_change      — stateless batch CUSUM on a full series
#   CUSUMDetector      — stateful online CUSUM for streaming updates
#
# Algorithm
# ---------
#   CUSUM tracks a cumulative sum of (value - expected_drift):
#     S_pos[t] = max(0, S_pos[t-1] + (x[t] - drift))
#   A change is flagged when S_pos crosses threshold h.
#   The detector also tracks a negative direction (S_neg) to detect
#   upward changes in signals where decrease = healthy (e.g., loss).
#
#   References:
#     Page, E.S. (1954) "Continuous Inspection Schemes"
#     Montgomery, D.C. (2009) "Statistical Quality Control"
#
# Dependencies
# ------------
#   numpy — array operations and statistics
#
# Usage
# -----
#   # Batch (stateless)
#   detected, idx, score = detect_change(series, threshold=5.0, drift=0.5)
#
#   # Streaming (stateful)
#   detector = CUSUMDetector(threshold=5.0, drift=0.5)
#   for value in stream:
#       fired = detector.update(value)
#       if fired:
#           detector.reset()
# -----------------------------------------------------------------------------

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stateless batch function
# ---------------------------------------------------------------------------


def detect_change(
    series: Union[List[float], np.ndarray],
    threshold: float = 5.0,
    drift: float = 1.0,
) -> Tuple[bool, int, float]:
    """Detect a change-point in a time series using the CUSUM algorithm.

    Applies the two-sided CUSUM test to the entire series in one pass.
    Detects the *first* step at which the cumulative sum crosses the
    alert threshold in either direction (upward = positive CUSUM,
    downward = negative CUSUM tracked via negated series).

    The standard one-sided positive-direction formula is:
        S[0]   = 0
        S[t]   = max(0,  S[t-1] + (x[t] - drift))
    A change is signalled when S[t] > threshold.

    Parameters
    ----------
    series : list[float] or numpy.ndarray
        Time-ordered sequence of scalar observations. Must contain at
        least one element; empty series immediately returns (False, -1, 0.0).
    threshold : float, optional
        Decision threshold h. When the cumulative sum exceeds h, a
        change is declared. Larger values reduce false alarms at the
        cost of detection delay. Default: 5.0.
    drift : float, optional
        Expected in-control mean shift magnitude (the "allowance" k).
        Acts as a noise tolerance parameter. Set to roughly half the
        minimum detectable shift. Default: 1.0.

    Returns
    -------
    change_detected : bool
        True if the cumulative sum exceeded threshold at any point.
    change_point_step : int
        Index into series where the change was first detected, or -1
        if no change was detected.
    cusum_value : float
        The value of the positive cumulative sum at detection (or at
        end of series if no detection occurred).

    Raises
    ------
    TypeError
        If series contains non-numeric elements.

    Examples
    --------
    >>> detected, idx, score = detect_change([0.9, 0.8, 0.7, 0.3, 0.1])
    >>> assert detected  # entropy drop triggers detection
    """
    if len(series) == 0:
        return False, -1, 0.0

    arr = np.asarray(series, dtype=np.float64)

    if arr.ndim != 1:
        raise TypeError(
            f"[MoEWatch] detect_change(): series must be 1-D, "
            f"got shape {arr.shape}."
        )

    # Guard against NaN / Inf values — treat them as zero deviation.
    arr = np.where(np.isfinite(arr), arr, 0.0)

    cusum_pos: float = 0.0
    cusum_neg: float = 0.0

    for i, x in enumerate(arr):
        # Positive direction: detect upward shift (increase in x)
        cusum_pos = max(0.0, cusum_pos + (x - drift))
        # Negative direction: detect downward shift (decrease in x)
        cusum_neg = max(0.0, cusum_neg + (-x - drift))

        if cusum_pos > threshold or cusum_neg > threshold:
            dominant = cusum_pos if cusum_pos >= cusum_neg else cusum_neg
            return True, int(i), float(dominant)

    return False, -1, float(cusum_pos)


# ---------------------------------------------------------------------------
# Stateful streaming detector
# ---------------------------------------------------------------------------


class CUSUMDetector:
    """Stateful online CUSUM detector for streaming time series.

    Maintains the cumulative sum state across successive ``update()``
    calls, making it suitable for real-time use where the full series is
    not available in advance. After a detection event, the caller should
    call ``reset()`` to restart the statistic from zero.

    Both a positive-direction CUSUM (detects increasing drift) and a
    negative-direction CUSUM (detects decreasing drift) are tracked
    simultaneously. The ``update()`` return value is True when *either*
    direction exceeds the threshold, enabling detection of both
    collapsing (decreasing entropy) and recovering routing distributions.

    Parameters
    ----------
    threshold : float, optional
        Alert threshold h. Default: 5.0.
    drift : float, optional
        Allowance parameter k (expected in-control shift tolerance).
        Default: 1.0.

    Attributes
    ----------
    threshold : float
        Decision threshold for change declaration.
    drift : float
        Noise tolerance / allowance parameter.

    Examples
    --------
    >>> detector = CUSUMDetector(threshold=3.0, drift=0.5)
    >>> for val in [0.8, 0.8, 0.5, 0.3, 0.1]:
    ...     if detector.update(val):
    ...         print("Change detected!")
    ...         detector.reset()
    """

    def __init__(
        self,
        threshold: float = 5.0,
        drift: float = 1.0,
    ) -> None:
        if threshold <= 0.0:
            raise ValueError(
                f"[MoEWatch] CUSUMDetector: threshold must be > 0, "
                f"got {threshold}."
            )
        if drift < 0.0:
            raise ValueError(
                f"[MoEWatch] CUSUMDetector: drift must be >= 0, "
                f"got {drift}."
            )

        self.threshold: float = float(threshold)
        self.drift: float = float(drift)

        # Current cumulative sum values (positive and negative directions)
        self._cusum_pos: float = 0.0
        self._cusum_neg: float = 0.0

        # Number of values processed since last reset
        self._n_updates: int = 0

        # Step index of the most recent detection (if any), -1 if none
        self._last_detection_step: int = -1

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, value: float) -> bool:
        """Process one observation and return True if a change is detected.

        Updates both the positive and negative cumulative sums using the
        CUSUM recursion:
            S_pos[t] = max(0, S_pos[t-1] + (value - drift))
            S_neg[t] = max(0, S_neg[t-1] + (-value - drift))

        Parameters
        ----------
        value : float
            The next observation in the time series. Non-finite values
            (NaN, Inf) are treated as zero deviation and do not advance
            the cumulative sum.

        Returns
        -------
        bool
            True if either ``S_pos`` or ``S_neg`` exceeds ``threshold``
            after incorporating this observation. The caller should call
            ``reset()`` to restart monitoring after a detection.
        """
        if not np.isfinite(value):
            self._n_updates += 1
            return False

        v = float(value)
        self._cusum_pos = max(0.0, self._cusum_pos + (v - self.drift))
        self._cusum_neg = max(0.0, self._cusum_neg + (-v - self.drift))
        self._n_updates += 1

        if self._cusum_pos > self.threshold or self._cusum_neg > self.threshold:
            self._last_detection_step = self._n_updates - 1
            return True

        return False

    def reset(self) -> None:
        """Reset cumulative sums to zero without clearing update count.

        Should be called after a detection event to restart the statistic.
        The ``_n_updates`` counter is preserved so that
        ``last_detection_step`` continues to be meaningful relative to the
        full stream index.
        """
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0
        logger.debug(
            "[MoEWatch] CUSUMDetector: reset at update %d.", self._n_updates
        )

    def reset_full(self) -> None:
        """Reset all internal state, including update counter.

        Use this when restarting an entirely new signal stream (e.g.,
        new training run). Unlike ``reset()``, this clears the
        ``_n_updates`` counter and ``_last_detection_step``.
        """
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0
        self._n_updates = 0
        self._last_detection_step = -1

    # ------------------------------------------------------------------
    # Query properties
    # ------------------------------------------------------------------

    @property
    def cusum_pos(self) -> float:
        """Current positive-direction cumulative sum."""
        return self._cusum_pos

    @property
    def cusum_neg(self) -> float:
        """Current negative-direction cumulative sum."""
        return self._cusum_neg

    @property
    def cusum_value(self) -> float:
        """Maximum of the two directional cumulative sums.

        Represents the current overall "tension" in the series. Values
        near threshold indicate the series is close to triggering a
        change detection event.
        """
        return max(self._cusum_pos, self._cusum_neg)

    @property
    def n_updates(self) -> int:
        """Total number of observations processed since last full reset."""
        return self._n_updates

    @property
    def last_detection_step(self) -> int:
        """Update index of the most recent detection, or -1 if none yet."""
        return self._last_detection_step

    def is_near_threshold(self, margin: float = 0.8) -> bool:
        """Return True if the current cusum_value exceeds margin * threshold.

        Useful for implementing "early warning" logic before a full
        detection fires.

        Parameters
        ----------
        margin : float, optional
            Fraction of threshold to check against. Default: 0.8
            (fires at 80% of threshold).

        Returns
        -------
        bool
            True if ``cusum_value >= margin * threshold``.
        """
        return self.cusum_value >= margin * self.threshold

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"CUSUMDetector("
            f"threshold={self.threshold}, "
            f"drift={self.drift}, "
            f"cusum_pos={self._cusum_pos:.4f}, "
            f"cusum_neg={self._cusum_neg:.4f}, "
            f"n_updates={self._n_updates})"
        )
