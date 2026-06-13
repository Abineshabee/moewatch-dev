# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# moewatch/analyzer/__init__.py
# =============================================================================
#
# Project      : MoEWatch
# Version      : v0.2.0
# Description  : Public surface of the moewatch.analyzer submodule. Re-exports
#                 all signal analyzers, their report dataclasses, supporting
#                 enumerations, and the CUSUM change-point detection utilities.
#                 Consumers should import from this package rather than from
#                 individual sub-modules so that internal layout can evolve
#                 without breaking downstream code.
#
# Author       : Abinesh N (@Abineshabee)
# Repository   : https://github.com/Abineshabee/MoEWatch
# License      : Apache 2.0
#
# Signal Tier Hierarchy
# ---------------------
#   Tier 1  — Gradient Starvation   earliest precursor (50–200 steps ahead)
#   Tier 2  — Entropy Drift         intermediate signal (30–100 steps ahead)
#   Tier 3  — Cross-Layer Spread    system-level localization
#   Fusion  — RiskScoreFuser        weighted combination → risk_score ∈ [0, 1]
#             risk = 0.6 * T1 + 0.3 * T2 + 0.1 * T3
#
# Exports
# -------
#   --- Tier 1: Gradient Starvation (v0.2.0) ---
#   GradientStarvationAnalyzer   — per-expert gradient norm monitoring
#   GradientStarvationReport     — per-expert starvation result dataclass
#
#   --- Tier 2: Entropy Drift ---
#   EntropyAnalyzer              — per-layer Shannon entropy drift detection
#   LayerEntropyReport           — per-layer entropy result dataclass
#   compute_entropy              — standalone entropy computation helper
#   compute_entropy_norm         — normalized entropy helper (H / H_max)
#
#   --- Tier 3: Cross-Layer Correlation (v0.2.0) ---
#   CrossLayerCorrelation        — cross-layer propagation analysis
#   CrossLayerReport             — correlation/source/victim result dataclass
#
#   --- Risk Fusion (v0.2.0) ---
#   RiskScoreFuser               — three-tier signal fusion into risk_score
#   RiskReport                   — fused risk result dataclass
#   RiskLevel                    — risk classification enum (LOW/MID/HIGH/CRITICAL)
#
#   --- Expert Health State Machine ---
#   CollapseDetector             — HEALTHY/COLD/DEAD state machine per expert
#   LayerCollapseReport          — per-layer collapse result dataclass
#   ExpertState                  — per-expert state dataclass
#   ExpertStatus                 — expert health enum (HEALTHY/COLD/DEAD/UNKNOWN)
#
#   --- CUSUM Change-Point Detection (v0.2.0) ---
#   CUSUMDetector                — stateful incremental CUSUM detector
#   detect_change                — one-shot CUSUM over a value series
#
# =============================================================================

from __future__ import annotations

# ---------------------------------------------------------------------------
# Tier 1: Gradient Starvation  [v0.2.0]
# ---------------------------------------------------------------------------
from moewatch.analyzer.gradient_starvation import (
    GradientStarvationAnalyzer,
    GradientStarvationReport,
)

# ---------------------------------------------------------------------------
# Tier 2: Entropy Drift
# ---------------------------------------------------------------------------
from moewatch.analyzer.entropy import (
    EntropyAnalyzer,
    LayerEntropyReport,
    compute_entropy,
    compute_entropy_norm,
)

# ---------------------------------------------------------------------------
# Tier 3: Cross-Layer Correlation  [v0.2.0]
# ---------------------------------------------------------------------------
from moewatch.analyzer.cross_layer import (
    CrossLayerCorrelation,
    CrossLayerReport,
)

# ---------------------------------------------------------------------------
# Risk Fusion  [v0.2.0]
# ---------------------------------------------------------------------------
from moewatch.analyzer.risk_score import (
    RiskLevel,
    RiskReport,
    RiskScoreFuser,
)

# ---------------------------------------------------------------------------
# Expert Health State Machine
# ---------------------------------------------------------------------------
from moewatch.analyzer.collapse import (
    CollapseDetector,
    ExpertState,
    ExpertStatus,
    LayerCollapseReport,
)

# ---------------------------------------------------------------------------
# CUSUM Change-Point Detection  [v0.2.0]
# ---------------------------------------------------------------------------
from moewatch.analyzer.cusum import (
    CUSUMDetector,
    detect_change,
)

# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------
__all__: list[str] = [
    # Tier 1
    "GradientStarvationAnalyzer",
    "GradientStarvationReport",
    # Tier 2
    "EntropyAnalyzer",
    "LayerEntropyReport",
    "compute_entropy",
    "compute_entropy_norm",
    # Tier 3
    "CrossLayerCorrelation",
    "CrossLayerReport",
    # Risk Fusion
    "RiskScoreFuser",
    "RiskReport",
    "RiskLevel",
    # Expert Health
    "CollapseDetector",
    "LayerCollapseReport",
    "ExpertState",
    "ExpertStatus",
    # CUSUM
    "CUSUMDetector",
    "detect_change",
]
