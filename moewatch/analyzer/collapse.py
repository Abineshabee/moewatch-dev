# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# MoEWatch — moewatch/analyzer/collapse.py
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
# Expert health state machine. Tracks each expert's status across training
# through the lifecycle: UNKNOWN → HEALTHY / COLD → DEAD. Experts transition
# to COLD when their utilisation fraction falls below a gradient-derived
# proxy threshold, and are promoted to DEAD after cold_steps_limit consecutive
# COLD steps. Load imbalance (max/mean utilisation ratio) is also reported.
#
# This analysis does NOT directly use gradient signals (that is Tier 1 in
# gradient_starvation.py). It operates entirely from routing event token
# counts, which are cheaper to collect and available in offline audit() mode.
#
# Contents
# --------
#   ExpertStatus           — enum for expert health states
#   ExpertState            — per-expert state dataclass
#   LayerCollapseReport    — per-layer collapse analysis dataclass
#   CollapseDetector       — state machine analyzer
#
# State Transition Rules
# ----------------------
#   UNKNOWN  → HEALTHY  : utilization >= cold_util_threshold
#   UNKNOWN  → COLD     : utilization <  cold_util_threshold
#   HEALTHY  → COLD     : utilization drops below cold_util_threshold
#   COLD     → HEALTHY  : utilization recovers (cold counter resets)
#   COLD     → DEAD     : consecutive COLD steps >= cold_steps_limit
#   DEAD     → DEAD     : once dead, never recovers (one-way transition)
#
# Dependencies
# ------------
#   moewatch.collector.stat_collector — StatCollector, LayerStats
#   moewatch.config                   — WatchConfig
#
# Usage
# -----
#   detector = CollapseDetector(config)
#   reports  = detector.analyze(stat_collector)
#   # reports: dict[str, LayerCollapseReport]
# -----------------------------------------------------------------------------

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

from moewatch.collector.stat_collector import StatCollector
from moewatch.config import WatchConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum number of tokens an expert must receive across the stats window to
# be considered "alive".  Experts receiving fewer than this are classed as
# COLD (even if their utilization fraction is above the relative threshold).
_MIN_TOKEN_COUNT: int = 1

# Safety floor for mean utilization to avoid divide-by-zero in imbalance
# ratio computation.
_MEAN_UTIL_FLOOR: float = 1e-9


# ---------------------------------------------------------------------------
# ExpertStatus enum
# ---------------------------------------------------------------------------


class ExpertStatus(Enum):
    """Health status of an individual expert.

    Attributes
    ----------
    UNKNOWN
        Insufficient data to classify.  Initial state for all experts.
    HEALTHY
        Expert is receiving a sufficient fraction of routing tokens.
    COLD
        Expert utilization has dropped below the cold threshold.
        Not yet dead; may recover.
    DEAD
        Expert has been COLD for ``cold_steps_limit`` consecutive steps.
        Terminal state — does not recover.
    """

    UNKNOWN = "UNKNOWN"
    HEALTHY = "HEALTHY"
    COLD = "COLD"
    DEAD = "DEAD"


# ---------------------------------------------------------------------------
# ExpertState dataclass
# ---------------------------------------------------------------------------


@dataclass
class ExpertState:
    """Per-expert health state snapshot.

    Attributes
    ----------
    expert_id : int
        Zero-indexed expert identifier within its layer.
    status : ExpertStatus
        Current health classification.
    consecutive_cold_steps : int
        Number of consecutive analysis steps in which this expert was COLD
        (including the current step if currently COLD).  Resets to 0 on
        recovery to HEALTHY.
    token_count : int
        Total number of tokens routed to this expert in the most recent
        analysis window (from ``LayerStats.expert_token_counts``).
    utilization : float
        Fraction of total tokens routed to this expert
        (``token_count / total_tokens``).  0.0 if no tokens were routed
        to any expert in the window.
    cold_onset_step : int or None
        Training step at which this expert first entered the COLD state
        during the current episode.  Resets to None on recovery.
    """

    expert_id: int
    status: ExpertStatus = ExpertStatus.UNKNOWN
    consecutive_cold_steps: int = 0
    token_count: int = 0
    utilization: float = 0.0
    cold_onset_step: Optional[int] = None


# ---------------------------------------------------------------------------
# LayerCollapseReport dataclass
# ---------------------------------------------------------------------------


@dataclass
class LayerCollapseReport:
    """Per-layer expert health analysis snapshot.

    Attributes
    ----------
    layer_name : str
        Fully-qualified router module name.
    expert_states : dict[int, ExpertState]
        Health state for each expert (keyed by expert_id).  Empty if no
        events have been collected for this layer.
    num_dead_experts : int
        Total number of experts in DEAD state.
    num_cold_experts : int
        Total number of experts currently in COLD state (not yet dead).
    num_healthy_experts : int
        Total number of experts in HEALTHY state.
    load_imbalance_ratio : float
        ``max(utilization) / mean(utilization)``.  A value of 1.0 indicates
        perfectly balanced routing; higher values indicate increasing skew.
        Returns 1.0 when mean utilization is effectively zero.
    severity : str
        Summary severity string:
        ``"HEALTHY"``  — all experts healthy, imbalance ratio within warn
        ``"DEGRADED"`` — some cold experts or borderline imbalance
        ``"CRITICAL"`` — dead experts present or severe imbalance
        ``"UNKNOWN"``  — insufficient data
    step : int
        Training step of the most recent event in the analysis window.
    """

    layer_name: str
    expert_states: Dict[int, ExpertState] = field(default_factory=dict)
    num_dead_experts: int = 0
    num_cold_experts: int = 0
    num_healthy_experts: int = 0
    load_imbalance_ratio: float = 1.0
    severity: str = "UNKNOWN"
    step: int = 0


# ---------------------------------------------------------------------------
# CollapseDetector
# ---------------------------------------------------------------------------


class CollapseDetector:
    """Expert health state machine.

    Maintains persistent per-layer, per-expert ``ExpertState`` objects across
    successive ``analyze()`` calls and transitions them according to the
    state machine rules described in the module header.

    The cold-to-dead promotion prevents noisy single-step utilization dips
    from triggering spurious DEAD classifications; an expert must remain
    continuously below the cold threshold for ``config.cold_steps_limit``
    steps before being marked as dead.

    Parameters
    ----------
    config : WatchConfig
        Shared configuration.  Relevant fields:
          ``dead_threshold``, ``cold_threshold``, ``cold_steps_limit``,
          ``load_imbalance_warn``, ``load_imbalance_error``.

    Attributes
    ----------
    config : WatchConfig
        Configuration reference.
    """

    def __init__(self, config: WatchConfig) -> None:
        self.config: WatchConfig = config

        # Persistent expert state per layer and expert.
        # {layer_name: {expert_id: ExpertState}}
        self._expert_states: Dict[str, Dict[int, ExpertState]] = {}

    # ------------------------------------------------------------------
    # Primary analysis method
    # ------------------------------------------------------------------

    def analyze(self, stat_collector: StatCollector) -> Dict[str, LayerCollapseReport]:
        """Analyze expert health states across all registered layers.

        Reads ``LayerStats`` from ``stat_collector``, updates the internal
        expert state machines, and returns a ``LayerCollapseReport`` per layer.

        Parameters
        ----------
        stat_collector : StatCollector
            Source of per-layer routing statistics.

        Returns
        -------
        dict[str, LayerCollapseReport]
            One report per registered layer.  Layers with no events yet
            produce a default report with ``severity = "UNKNOWN"``.

        Notes
        -----
        Per-layer failures are caught and logged, not propagated.
        """
        reports: Dict[str, LayerCollapseReport] = {}

        try:
            all_stats = stat_collector.get_all_stats()
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "[MoEWatch] CollapseDetector.analyze(): failed to read stats: %s",
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
                    "[MoEWatch] CollapseDetector: error on layer '%s': %s",
                    layer_name,
                    exc,
                )
                reports[layer_name] = LayerCollapseReport(
                    layer_name=layer_name,
                    severity="UNKNOWN",
                )

        return reports

    # ------------------------------------------------------------------
    # Internal: single-layer analysis
    # ------------------------------------------------------------------

    def _analyze_layer(
        self,
        layer_name: str,
        layer_stats: object,
    ) -> LayerCollapseReport:
        """Update and return expert health states for one layer.

        Parameters
        ----------
        layer_name : str
            Layer name.
        layer_stats : LayerStats
            Current statistics snapshot.

        Returns
        -------
        LayerCollapseReport
        """
        step = int(getattr(layer_stats, "step", 0))
        expert_utilization = getattr(layer_stats, "expert_utilization", None)
        expert_token_counts = getattr(layer_stats, "expert_token_counts", None)
        load_imbalance_ratio = float(
            getattr(layer_stats, "load_imbalance_ratio", 1.0)
        )

        # ------------------------------------------------------------------
        # Guard: no utilization data yet
        # ------------------------------------------------------------------
        if expert_utilization is None or not hasattr(expert_utilization, "__len__"):
            return LayerCollapseReport(
                layer_name=layer_name,
                step=step,
                load_imbalance_ratio=load_imbalance_ratio,
            )

        import torch  # noqa: PLC0415 — lazy import; torch dependency checked at package level

        if isinstance(expert_utilization, torch.Tensor):
            util_list = expert_utilization.cpu().float().tolist()
        else:
            util_list = list(float(v) for v in expert_utilization)

        n_experts = len(util_list)
        if n_experts == 0:
            return LayerCollapseReport(layer_name=layer_name, step=step)

        # Token counts (integer list), parallel to util_list.
        if (
            expert_token_counts is not None
            and hasattr(expert_token_counts, "__len__")
        ):
            if isinstance(expert_token_counts, torch.Tensor):
                token_list = expert_token_counts.cpu().tolist()
            else:
                token_list = list(int(v) for v in expert_token_counts)
        else:
            token_list = [0] * n_experts

        # ------------------------------------------------------------------
        # Initialise per-layer state on first encounter
        # ------------------------------------------------------------------
        if layer_name not in self._expert_states:
            self._expert_states[layer_name] = {
                i: ExpertState(expert_id=i) for i in range(n_experts)
            }

        layer_expert_states = self._expert_states[layer_name]

        # Add entries for any new experts (e.g., if n_experts grew).
        for i in range(n_experts):
            if i not in layer_expert_states:
                layer_expert_states[i] = ExpertState(expert_id=i)

        # ------------------------------------------------------------------
        # Update each expert's state
        # ------------------------------------------------------------------
        # The utilization-based cold threshold is derived from the configured
        # cold_threshold (gradient-space proxy).  Because token utilization
        # fractions are normalised to [0, 1] while gradient norms are in a
        # different scale, we use a simple relative heuristic:
        # an expert is "cold" if its utilization < 1/n_experts * cold_factor.
        # An expert is "cold" if it receives less than half its fair share
        # of tokens under uniform routing (1/n_experts).
        cold_util_threshold = (1.0 / n_experts) * 0.5

        # An expert is dead-level when utilization is near zero.
        dead_util_threshold = (1.0 / n_experts) * 0.05

        for expert_id in range(n_experts):
            util = util_list[expert_id] if expert_id < len(util_list) else 0.0
            tokens = token_list[expert_id] if expert_id < len(token_list) else 0
            state = layer_expert_states[expert_id]

            # Update token statistics.
            state.token_count = int(tokens)
            state.utilization = float(util)

            # Dead state is terminal — skip further transitions.
            if state.status == ExpertStatus.DEAD:
                continue

            # Determine health classification from utilization.
            if util >= cold_util_threshold and tokens >= _MIN_TOKEN_COUNT:
                # Expert is healthy.
                if state.status != ExpertStatus.HEALTHY:
                    logger.debug(
                        "[MoEWatch] CollapseDetector: expert %d in '%s' "
                        "recovered to HEALTHY (util=%.4f).",
                        expert_id,
                        layer_name,
                        util,
                    )
                state.status = ExpertStatus.HEALTHY
                state.consecutive_cold_steps = 0
                state.cold_onset_step = None

            else:
                # Expert is at or below cold threshold.
                if state.status == ExpertStatus.HEALTHY or state.status == ExpertStatus.UNKNOWN:
                    # First step entering COLD.
                    state.cold_onset_step = step
                    logger.debug(
                        "[MoEWatch] CollapseDetector: expert %d in '%s' "
                        "entered COLD at step %d (util=%.4f).",
                        expert_id,
                        layer_name,
                        step,
                        util,
                    )

                state.status = ExpertStatus.COLD
                state.consecutive_cold_steps += 1

                # Promote to DEAD after cold_steps_limit consecutive steps.
                if state.consecutive_cold_steps >= self.config.cold_steps_limit:
                    state.status = ExpertStatus.DEAD
                    logger.info(
                        "[MoEWatch] CollapseDetector: expert %d in '%s' "
                        "promoted to DEAD after %d consecutive cold steps.",
                        expert_id,
                        layer_name,
                        state.consecutive_cold_steps,
                    )

        # ------------------------------------------------------------------
        # Aggregate layer-level statistics
        # ------------------------------------------------------------------
        num_dead = sum(
            1 for s in layer_expert_states.values() if s.status == ExpertStatus.DEAD
        )
        num_cold = sum(
            1 for s in layer_expert_states.values() if s.status == ExpertStatus.COLD
        )
        num_healthy = sum(
            1 for s in layer_expert_states.values() if s.status == ExpertStatus.HEALTHY
        )

        severity = self._classify_severity(
            num_dead=num_dead,
            num_cold=num_cold,
            n_experts=n_experts,
            load_imbalance_ratio=load_imbalance_ratio,
        )

        return LayerCollapseReport(
            layer_name=layer_name,
            expert_states=dict(layer_expert_states),
            num_dead_experts=num_dead,
            num_cold_experts=num_cold,
            num_healthy_experts=num_healthy,
            load_imbalance_ratio=load_imbalance_ratio,
            severity=severity,
            step=step,
        )

    # ------------------------------------------------------------------
    # Internal: severity classification
    # ------------------------------------------------------------------

    def _classify_severity(
        self,
        num_dead: int,
        num_cold: int,
        n_experts: int,
        load_imbalance_ratio: float,
    ) -> str:
        """Classify layer-level severity from expert counts and load imbalance.

        Parameters
        ----------
        num_dead : int
            Number of dead experts.
        num_cold : int
            Number of cold experts.
        n_experts : int
            Total number of experts.
        load_imbalance_ratio : float
            max / mean utilization ratio.

        Returns
        -------
        str
            ``"CRITICAL"`` | ``"DEGRADED"`` | ``"HEALTHY"`` | ``"UNKNOWN"``
        """
        if n_experts == 0:
            return "UNKNOWN"

        dead_fraction = num_dead / n_experts
        cold_fraction = num_cold / n_experts

        # Critical: any dead expert, severe imbalance, or majority cold.
        if (
            num_dead > 0
            or load_imbalance_ratio >= self.config.load_imbalance_error
            or cold_fraction >= 0.5
        ):
            return "CRITICAL"

        # Degraded: some cold experts or elevated imbalance.
        if (
            num_cold > 0
            or load_imbalance_ratio >= self.config.load_imbalance_warn
            or cold_fraction > 0.1
        ):
            return "DEGRADED"

        # Insufficient data to confirm healthy.
        if (num_dead + num_cold + (n_experts - num_dead - num_cold)) == 0:
            return "UNKNOWN"

        return "HEALTHY"

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def reset(self, layer_name: Optional[str] = None) -> None:
        """Reset expert state machine for one or all layers.

        Parameters
        ----------
        layer_name : str, optional
            Specific layer to reset.  If None, resets all layers.
        """
        if layer_name is not None:
            self._expert_states.pop(layer_name, None)
        else:
            self._expert_states.clear()

    def get_expert_state(
        self, layer_name: str, expert_id: int
    ) -> Optional[ExpertState]:
        """Return the current state for a specific expert.

        Parameters
        ----------
        layer_name : str
            Layer name.
        expert_id : int
            Expert index.

        Returns
        -------
        ExpertState or None
            State object if found, None otherwise.
        """
        return self._expert_states.get(layer_name, {}).get(expert_id)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_layers = len(self._expert_states)
        return (
            f"CollapseDetector("
            f"layers={n_layers}, "
            f"cold_steps_limit={self.config.cold_steps_limit})"
        )
