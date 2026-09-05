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
#   cold_threshold is RELATIVE whenever a layer has >= 2 experts with
#   sufficient samples: threshold = _RELATIVE_COLD_FRACTION * median(norm
#   across layer-mates). Using the median (not the mean) keeps a single
#   outlier expert from skewing the reference for everyone else. This
#   self-calibrates to whatever gradient-norm scale the model actually
#   produces. config.cold_threshold (an absolute value) is used only as
#   a fallback when there's no peer group to compare against (e.g. a
#   single-expert layer) — see `_compute_layer_mean_norm`.
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

# Fraction of a layer's mean expert gradient norm below which an expert is
# considered "cold" relative to its peers. Used instead of the absolute
# config.cold_threshold whenever at least two experts in the same layer
# have enough samples to compute a meaningful peer average — see
# `_compute_layer_mean_norm` / the `layer_mean_norm` fix in `_analyze_expert`.
_RELATIVE_COLD_FRACTION: float = 0.5


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
            all_stats = stat_collector.get_all_stats(window=self.config.stats_window)
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

            # Compute a peer reference norm for this layer once, so each
            # expert can be judged relative to its layer-mates rather than
            # only against a single hardcoded absolute value (see
            # `_compute_layer_mean_norm` docstring for rationale).
            layer_mean_norm, n_valid_experts = self._compute_layer_mean_norm(
                expert_stats_map
            )

            # Process experts in deterministic order.
            for expert_id in sorted(expert_stats_map.keys()):
                grad_stats = expert_stats_map[expert_id]
                try:
                    report = self._analyze_expert(
                        layer_name,
                        expert_id,
                        grad_stats,
                        layer_mean_norm=layer_mean_norm,
                        n_valid_experts=n_valid_experts,
                    )
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
    # Internal: layer-level peer reference norm
    # ------------------------------------------------------------------

    def _compute_layer_mean_norm(
        self,
        expert_stats_map: Dict[int, object],
    ) -> "tuple[float, int]":
        """Compute a robust peer-reference gradient norm for a layer.

        Used to derive a *relative* cold threshold (see
        ``_RELATIVE_COLD_FRACTION``) instead of relying solely on the
        absolute ``config.cold_threshold``. The absolute default is
        calibrated for typical production-scale gradient norms
        (e.g. 0.1–10.0); for a small model whose gradient norms sit at
        0.001–0.01, the absolute threshold is always far above every
        expert's norm, so every expert is permanently misclassified as
        starving. Comparing each expert against its layer-mates
        self-calibrates to whatever scale the model actually produces,
        regardless of absolute magnitude.

        The **median** (not the mean) of the experts' norms is used as
        the reference. A plain mean is not robust to a single outlier:
        if one expert dominated routing during a prior collapse and
        still carries a disproportionately larger gradient norm even
        after routing recovers, averaging it in with the rest inflates
        the reference and can permanently misclassify every *other*,
        genuinely healthy expert as "cold relative to the outlier" —
        exactly the kind of stuck, never-clearing alert this fix is
        meant to prevent. The median is unaffected by a single such
        outlier, so recovery is judged against what most experts are
        actually doing.

        Only experts with at least ``_MIN_SAMPLES_FOR_DETECTION`` finite
        samples contribute, matching the sample-sufficiency gate used in
        ``_analyze_expert``.

        Parameters
        ----------
        expert_stats_map : dict[int, GradientStats]
            All experts' gradient stats for one layer, as returned by
            ``StatCollector.get_all_stats()``.

        Returns
        -------
        tuple[float, int]
            ``(layer_reference_norm, n_valid_experts)``. ``n_valid_experts``
            is the number of experts that contributed. Callers should
            only trust the relative threshold when this is >= 2 (a
            single expert has no peers to be judged against).
        """
        means: List[float] = []
        for grad_stats in expert_stats_map.values():
            norm_history = list(
                getattr(grad_stats, "gradient_norm_history", []) or []
            )
            if len(norm_history) < _MIN_SAMPLES_FOR_DETECTION:
                continue
            arr = np.array(norm_history, dtype=np.float64)
            finite = arr[np.isfinite(arr)]
            if len(finite) == 0:
                continue
            means.append(float(finite.mean()))

        if not means:
            return 0.0, 0
        return float(np.median(means)), len(means)

    # ------------------------------------------------------------------
    # Internal: single-expert analysis
    # ------------------------------------------------------------------

    def _analyze_expert(
        self,
        layer_name: str,
        expert_id: int,
        grad_stats: object,
        layer_mean_norm: float = 0.0,
        n_valid_experts: int = 0,
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
        layer_mean_norm : float, optional
            Mean gradient norm across this expert's layer-mates (see
            ``_compute_layer_mean_norm``). Used as the reference for a
            relative cold threshold when ``n_valid_experts >= 2``.
        n_valid_experts : int, optional
            Number of experts that contributed to ``layer_mean_norm``.
            Below 2, there is no meaningful peer group to compare
            against, so the absolute ``config.cold_threshold`` is used
            instead.

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
            # Insufficient samples for reliable trend detection.
            # However, if we have at least 1 sample and the norm is
            # exactly zero, this is a confirmed dead expert — compute
            # the starvation score rather than returning the default 0.0,
            # which would be misread by the fuser as "healthy".
            if n_samples >= 1:
                _early_array = np.array(norm_history, dtype=np.float64)
                _finite = _early_array[np.isfinite(_early_array)]
                if len(_finite) >= 1 and float(_finite.mean()) == 0.0:
                    _cold_thr = max(self.config.cold_threshold, 1e-9)
                    return GradientStarvationReport(
                        layer_name=layer_name,
                        expert_id=expert_id,
                        gradient_norm_mean=0.0,
                        gradient_norm_std=0.0,
                        starvation_score=float(
                            np.clip(1.0 - 0.0 / _cold_thr, 0.0, 1.0)
                        ),
                        step=step,
                        n_samples=n_samples,
                    )
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
        # Prefer a threshold relative to this expert's layer-mates when at
        # least one other expert is available for comparison — this is
        # what actually self-calibrates to the model's real gradient
        # scale (see `_compute_layer_mean_norm`). Fall back to the
        # absolute config.cold_threshold only when there's no peer group
        # to compare against (e.g. a layer with a single expert), since a
        # relative threshold is meaningless without peers.
        if n_valid_experts >= 2 and layer_mean_norm > 1e-12:
            cold_threshold = max(
                layer_mean_norm * _RELATIVE_COLD_FRACTION, 1e-9
            )
        else:
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
                # Determine the onset step as accurately as possible.
                #
                # When ALL n_samples in the window are zero, starvation
                # started before the window — back-calculate to the first
                # event in the buffer: step - (n_samples - 1).
                # Example: step=2, n_samples=3 (all zero since step 0)
                #          → onset = 2 - 2 = 0  ✓
                #
                # When only the RECENT tail is zero (expert died mid-run),
                # we can't back-calculate precisely without per-event steps
                # in GradientStats. Record the current step as a conservative
                # upper bound — at most _MIN_SAMPLES_FOR_DETECTION steps late.
                all_zero = (norm_mean == 0.0)
                if all_zero:
                    onset_step = max(0, step - (n_samples - 1))
                else:
                    onset_step = step
                onset_map[expert_id] = onset_step
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
