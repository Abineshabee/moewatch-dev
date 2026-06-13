# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# MoEWatch — moewatch/analyzer/gradient_starvation.py
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
# Tier 1 signal — the earliest consistent collapse precursor, empirically
# observed 50–200 steps before routing utilisation visibly drops.
#
# Experts stop receiving useful gradient updates before their token counts
# change, making gradient norm monitoring the most actionable early warning
# available.  This analyzer consumes per-expert gradient L2 norms collected
# by GradientStarvationHook (moewatch/hooks/gradient_hook.py) and stored in
# StatCollector's gradient buffers, computes rolling statistics, and
# classifies each expert as starving or not.
#
# Contents
# --------
#   GradientStarvationReport   — per-expert starvation analysis dataclass
#   GradientStarvationAnalyzer — Tier 1 analyzer
#
# Starvation Detection Logic
# --------------------------
#   starvation_score  = max(0, 1 - norm_mean / cold_threshold)
#     → 0.0 = fully healthy,  1.0 = norm_mean is at or below zero
#   starvation_detected = True when score > 0 AND
#     consecutive_cold_steps >= config.cold_steps_limit
#   starvation_onset_step recorded at first crossing of cold_threshold
#
# Dependencies
# ------------
#   moewatch.collector.stat_collector — StatCollector
#   moewatch.config                   — WatchConfig
#   numpy
#
# Usage
# -----
#   analyzer = GradientStarvationAnalyzer(config)
#   reports  = analyzer.analyze(stat_collector)
#   # reports: dict[str, list[GradientStarvationReport]]
# -----------------------------------------------------------------------------

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from moewatch.collector.stat_collector import StatCollector
from moewatch.config import WatchConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum number of gradient norm samples required before starvation
# detection is attempted for a given expert.  Below this count, the
# analyzer returns a default report with starvation_detected=False.
_MIN_SAMPLES_FOR_DETECTION: int = 3

# Maximum gradient history retained per expert in the internal bookkeeping
# dicts (onset step tracking only — actual norm history lives in StatCollector).
_MAX_ONSET_HISTORY: int = 1000


# ---------------------------------------------------------------------------
# GradientStarvationReport dataclass
# ---------------------------------------------------------------------------


@dataclass
class GradientStarvationReport:
    """Per-expert gradient starvation analysis snapshot.

    Produced by ``GradientStarvationAnalyzer.analyze()`` for every
    ``(layer_name, expert_id)`` pair that has gradient data.

    Attributes
    ----------
    layer_name : str
        Fully-qualified name of the router/expert layer.
    expert_id : int
        Zero-indexed expert identifier within its layer.
    gradient_norm_mean : float
        Arithmetic mean of the gradient L2 norms over the most recent
        analysis window.  ``0.0`` if no samples collected yet.
    gradient_norm_std : float
        Population standard deviation of the gradient norms in the window.
        ``0.0`` if fewer than two samples.
    starvation_score : float
        Continuous starvation severity metric in [0.0, 1.0].
        ``max(0, 1 - gradient_norm_mean / config.cold_threshold)``
        A value of 0.0 means the expert is fully healthy; 1.0 means the
        mean gradient norm is at or below zero (complete starvation).
    starvation_detected : bool
        True when the expert has been continuously below the cold threshold
        for at least ``config.cold_steps_limit`` consecutive analysis calls.
    starvation_onset_step : int or None
        Training step at which starvation first began (first step below
        cold_threshold), or None if not currently starving.
    step : int
        Training step of the most recent gradient event in the window.
    n_samples : int
        Number of gradient norm samples used to compute this report.
    """

    layer_name: str
    expert_id: int
    gradient_norm_mean: float = 0.0
    gradient_norm_std: float = 0.0
    starvation_score: float = 0.0
    starvation_detected: bool = False
    starvation_onset_step: Optional[int] = None
    step: int = 0
    n_samples: int = 0


# ---------------------------------------------------------------------------
# GradientStarvationAnalyzer
# ---------------------------------------------------------------------------


class GradientStarvationAnalyzer:
    """Tier 1 signal: per-expert gradient starvation detection.

    Consumes ``GradientStats`` objects from ``StatCollector`` and produces
    one ``GradientStarvationReport`` per ``(layer_name, expert_id)`` pair.

    State is persistent across successive ``analyze()`` calls so that:
      - ``starvation_onset_step`` is recorded once and kept until recovery.
      - ``consecutive_cold_steps`` accumulates across analysis calls rather
        than being recomputed from scratch each time.
      - Recovery (norm rising above cold_threshold) resets the counter and
        clears the onset step.

    Parameters
    ----------
    config : WatchConfig
        Shared configuration.  Relevant fields:
          ``cold_threshold``, ``dead_threshold``, ``cold_steps_limit``,
          ``sample_every``.

    Attributes
    ----------
    config : WatchConfig
        Configuration reference.
    """

    def __init__(self, config: WatchConfig) -> None:
        self.config: WatchConfig = config

        # Persistent consecutive-below-threshold counters.
        # {layer_name: {expert_id: consecutive_cold_steps}}
        self._starvation_counters: Dict[str, Dict[int, int]] = {}

        # Step at which starvation began for each expert (if currently starving).
        # {layer_name: {expert_id: onset_step}}
        self._onset_steps: Dict[str, Dict[int, Optional[int]]] = {}

    # ------------------------------------------------------------------
    # Primary analysis method
    # ------------------------------------------------------------------

    def analyze(
        self,
        stat_collector: StatCollector,
    ) -> Dict[str, List[GradientStarvationReport]]:
        """Analyze gradient norms and return per-layer starvation reports.

        Reads ``GradientStats`` from ``stat_collector`` for every registered
        ``(layer_name, expert_id)`` pair, computes rolling statistics, and
        updates the internal starvation state machine.

        Parameters
        ----------
        stat_collector : StatCollector
            Source of gradient statistics.

        Returns
        -------
        dict[str, list[GradientStarvationReport]]
            Keys are layer names; values are lists of reports, one per expert
            in that layer (ordered by ascending expert_id).  Layers with no
            gradient data produce an empty list.

        Notes
        -----
        Per-layer failures are caught and logged at WARNING level.
        """
        reports: Dict[str, List[GradientStarvationReport]] = {}

        try:
            all_stats = stat_collector.get_all_stats()
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "[MoEWatch] GradientStarvationAnalyzer.analyze(): failed to "
                "read stats: %s",
                exc,
            )
            return reports

        gradient_stats = all_stats.get("gradient", {})

        for layer_name, expert_stats_map in gradient_stats.items():
            layer_reports: List[GradientStarvationReport] = []

            # Ensure per-layer state containers exist.
            if layer_name not in self._starvation_counters:
                self._starvation_counters[layer_name] = {}
                self._onset_steps[layer_name] = {}

            # Process experts in deterministic order.
            for expert_id in sorted(expert_stats_map.keys()):
                grad_stats = expert_stats_map[expert_id]
                try:
                    report = self._analyze_expert(layer_name, expert_id, grad_stats)
                    layer_reports.append(report)
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning(
                        "[MoEWatch] GradientStarvationAnalyzer: error on "
                        "layer '%s' expert %d: %s",
                        layer_name,
                        expert_id,
                        exc,
                    )
                    layer_reports.append(
                        GradientStarvationReport(
                            layer_name=layer_name,
                            expert_id=expert_id,
                        )
                    )

            if layer_reports:
                reports[layer_name] = layer_reports

        return reports

    # ------------------------------------------------------------------
    # Internal: single-expert analysis
    # ------------------------------------------------------------------

    def _analyze_expert(
        self,
        layer_name: str,
        expert_id: int,
        grad_stats: object,
    ) -> GradientStarvationReport:
        """Compute starvation report for one expert.

        Parameters
        ----------
        layer_name : str
            Layer name.
        expert_id : int
            Expert index.
        grad_stats : GradientStats
            Statistics snapshot from StatCollector.

        Returns
        -------
        GradientStarvationReport
        """
        # Extract fields from GradientStats dataclass.
        norm_history: List[float] = list(
            getattr(grad_stats, "gradient_norm_history", []) or []
        )
        step: int = int(getattr(grad_stats, "step", 0))

        n_samples = len(norm_history)

        # ------------------------------------------------------------------
        # Insufficient data guard
        # ------------------------------------------------------------------
        if n_samples < _MIN_SAMPLES_FOR_DETECTION:
            return GradientStarvationReport(
                layer_name=layer_name,
                expert_id=expert_id,
                step=step,
                n_samples=n_samples,
            )

        # ------------------------------------------------------------------
        # Rolling statistics
        # ------------------------------------------------------------------
        norm_array = np.array(norm_history, dtype=np.float64)

        # Guard against NaN / Inf values (can arise from gradient explosions).
        finite_mask = np.isfinite(norm_array)
        if not finite_mask.any():
            return GradientStarvationReport(
                layer_name=layer_name,
                expert_id=expert_id,
                step=step,
                n_samples=n_samples,
            )
        norm_array = norm_array[finite_mask]

        norm_mean = float(norm_array.mean())
        norm_std = float(norm_array.std()) if len(norm_array) >= 2 else 0.0

        # ------------------------------------------------------------------
        # Starvation score: continuous metric
        # ------------------------------------------------------------------
        cold_threshold = max(self.config.cold_threshold, 1e-9)
        starvation_score = float(
            np.clip(1.0 - norm_mean / cold_threshold, 0.0, 1.0)
        )

        # ------------------------------------------------------------------
        # State machine: consecutive cold steps
        # ------------------------------------------------------------------
        counter_map = self._starvation_counters[layer_name]
        onset_map = self._onset_steps[layer_name]

        if expert_id not in counter_map:
            counter_map[expert_id] = 0
            onset_map[expert_id] = None

        below_cold = norm_mean < cold_threshold

        if below_cold:
            counter_map[expert_id] += 1
            if onset_map[expert_id] is None:
                onset_map[expert_id] = step
                logger.debug(
                    "[MoEWatch] GradientStarvationAnalyzer: expert %d in "
                    "'%s' fell below cold threshold at step %d "
                    "(norm_mean=%.5f < threshold=%.5f).",
                    expert_id,
                    layer_name,
                    step,
                    norm_mean,
                    cold_threshold,
                )
        else:
            # Expert has recovered — reset counter and onset.
            if counter_map[expert_id] > 0:
                logger.debug(
                    "[MoEWatch] GradientStarvationAnalyzer: expert %d in "
                    "'%s' recovered at step %d (norm_mean=%.5f).",
                    expert_id,
                    layer_name,
                    step,
                    norm_mean,
                )
            counter_map[expert_id] = 0
            onset_map[expert_id] = None

        consecutive_cold = counter_map[expert_id]
        starvation_detected = consecutive_cold >= self.config.cold_steps_limit

        if starvation_detected and not below_cold:
            # Recovered this step but threshold not yet cleared — keep onset.
            starvation_detected = False

        return GradientStarvationReport(
            layer_name=layer_name,
            expert_id=expert_id,
            gradient_norm_mean=norm_mean,
            gradient_norm_std=norm_std,
            starvation_score=starvation_score,
            starvation_detected=starvation_detected,
            starvation_onset_step=onset_map.get(expert_id),
            step=step,
            n_samples=n_samples,
        )

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def reset(self, layer_name: Optional[str] = None) -> None:
        """Reset starvation state for one or all layers.

        Parameters
        ----------
        layer_name : str, optional
            Specific layer to reset.  If None, resets all layers.
        """
        if layer_name is not None:
            self._starvation_counters.pop(layer_name, None)
            self._onset_steps.pop(layer_name, None)
        else:
            self._starvation_counters.clear()
            self._onset_steps.clear()

    def get_starvation_count(self, layer_name: str, expert_id: int) -> int:
        """Return current consecutive-cold-steps count for one expert.

        Parameters
        ----------
        layer_name : str
            Layer name.
        expert_id : int
            Expert index.

        Returns
        -------
        int
            Consecutive steps below cold threshold, or 0 if untracked.
        """
        return self._starvation_counters.get(layer_name, {}).get(expert_id, 0)

    def is_layer_registered(self, layer_name: str) -> bool:
        """Return True if this layer has ever been analyzed.

        Parameters
        ----------
        layer_name : str
            Layer name to check.

        Returns
        -------
        bool
        """
        return layer_name in self._starvation_counters

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_layers = len(self._starvation_counters)
        n_experts = sum(len(m) for m in self._starvation_counters.values())
        return (
            f"GradientStarvationAnalyzer("
            f"layers={n_layers}, "
            f"experts_tracked={n_experts}, "
            f"cold_threshold={self.config.cold_threshold}, "
            f"cold_steps_limit={self.config.cold_steps_limit})"
        )
