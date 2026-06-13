# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# moewatch/hooks/__init__.py
# =============================================================================
#
# Project      : MoEWatch
# Version      : v0.2.0
# Description  : Public surface of the moewatch.hooks submodule. Re-exports
#                 the hook lifecycle manager, the forward/backward hook
#                 callables, the event dataclasses they produce, and the
#                 router auto-detection function.
#
# Author       : Abinesh N (@Abineshabee)
# Repository   : https://github.com/Abineshabee/MoEWatch
# License      : Apache 2.0
#
# Exports
# -------
#   HookManager              — attach/detach lifecycle owner
#   RouterForwardHook        — forward hook on router modules
#   RoutingEvent             — dataclass produced by RouterForwardHook
#   GradientStarvationHook   — backward hook on expert weight parameters [v0.2.0]
#   GradientEvent            — dataclass produced by GradientStarvationHook [v0.2.0]
#   detect_router_modules    — auto-detection function
#
# =============================================================================

from __future__ import annotations

from moewatch.hooks.detection import detect_router_modules
from moewatch.hooks.gradient_hook import GradientEvent, GradientStarvationHook
from moewatch.hooks.manager import HookManager
from moewatch.hooks.router_hook import RouterForwardHook, RoutingEvent

__all__ = [
    "HookManager",
    "RouterForwardHook",
    "RoutingEvent",
    "GradientStarvationHook",
    "GradientEvent",
    "detect_router_modules",
]
