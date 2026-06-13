# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# moewatch/intervention/__init__.py [v0.2.0]
# =============================================================================
#
# Project      : MoEWatch
# Version      : v0.2.0
# Description  : Public surface of the moewatch.intervention submodule.
#                 Re-exports the intervention action classes, the
#                 SafetyGuard pre-flight validator, and the
#                 InterventionEngine orchestrator. Consumers should import
#                 from this package rather than from individual sub-modules
#                 so that internal layout can evolve without breaking
#                 downstream code.
#
# Author       : Abinesh N (@Abineshabee)
# Repository   : https://github.com/Abineshabee/MoEWatch
# License      : Apache 2.0
#
# Exports
# -------
#   --- Actions ---
#   InterventionAction    — abstract base class for all actions
#   AuxLossAction         — increase aux_loss_coef on a target layer
#   RouterNoiseAction     — inject Gaussian noise into router logits
#   ExpertDropoutAction   — increase expert dropout probability
#   NoOpAction            — do nothing (conservative baseline)
#
#   --- Safety ---
#   SafetyGuard           — pre-flight validation (cooldown, delta limit,
#                            loss guard, neighbor check)
#   SafetyCheckResult     — result dataclass returned by SafetyGuard.check()
#
#   --- Orchestration ---
#   InterventionEngine    — applies, tracks, and evaluates interventions
#
# =============================================================================

from __future__ import annotations

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
from moewatch.intervention.actions import (
    AuxLossAction,
    ExpertDropoutAction,
    InterventionAction,
    NoOpAction,
    RouterNoiseAction,
)

# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------
from moewatch.intervention.safety import SafetyCheckResult, SafetyGuard

# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
from moewatch.intervention.engine import InterventionEngine

# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------
__all__: list[str] = [
    # Actions
    "InterventionAction",
    "AuxLossAction",
    "RouterNoiseAction",
    "ExpertDropoutAction",
    "NoOpAction",
    # Safety
    "SafetyGuard",
    "SafetyCheckResult",
    # Orchestration
    "InterventionEngine",
]
