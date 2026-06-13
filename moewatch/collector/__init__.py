# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# moewatch/collector/__init__.py
# =============================================================================
#
# Project      : MoEWatch
# Version      : v0.2.0
# Description  : Public surface of the moewatch.collector submodule.
#                 Re-exports the central event aggregator (StatCollector),
#                 its derived-statistics dataclasses (LayerStats,
#                 GradientStats), the underlying fixed-capacity circular
#                 buffer (RingBuffer), and the intervention-conditioned
#                 counterfactual baseline tracker (BaselineTracker).
#
# Author       : Abinesh N (@Abineshabee)
# Repository   : https://github.com/Abineshabee/MoEWatch
# License      : Apache 2.0
#
# Exports
# -------
#   StatCollector     — central event aggregator
#   LayerStats        — per-layer routing statistics snapshot
#   GradientStats     — per-expert gradient statistics snapshot [v0.2.0]
#   RingBuffer        — fixed-capacity circular buffer
#   BaselineTracker   — counterfactual baseline tracker [v0.2.0]
#
# =============================================================================

from __future__ import annotations

from moewatch.collector.baseline_tracker import BaselineTracker
from moewatch.collector.ring_buffer import RingBuffer
from moewatch.collector.stat_collector import (
    GradientStats,
    LayerStats,
    StatCollector,
)

__all__ = [
    "StatCollector",
    "LayerStats",
    "GradientStats",
    "RingBuffer",
    "BaselineTracker",
]
