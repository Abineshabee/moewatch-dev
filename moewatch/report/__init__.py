# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# MoEWatch — moewatch/report/__init__.py
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
# Public re-export surface for the report subpackage. Importing from this
# module is the canonical way to access any reporting class; internal module
# paths are considered private and subject to change between minor versions.
#
# Contents
# --------
#   AuditReport    — structured result from offline audit() (audit_report.py)
#   WatchReport    — aggregated live training report (watch_report.py) [v0.2.0]
#   CLIReporter    — ANSI / Rich terminal dashboard (cli_reporter.py)
#   JSONReporter   — newline-delimited JSON emitter (json_reporter.py) [v0.2.0]
#
# Usage
# -----
#   from moewatch.report import AuditReport, WatchReport
#   from moewatch.report import CLIReporter, JSONReporter
#
# =============================================================================

from __future__ import annotations

from moewatch.report.audit_report import AuditReport
from moewatch.report.cli_reporter import CLIReporter
from moewatch.report.json_reporter import JSONReporter
from moewatch.report.watch_report import StepReport, WatchReport

__all__: list[str] = [
    "AuditReport",
    "WatchReport",
    "StepReport",
    "CLIReporter",
    "JSONReporter",
]
