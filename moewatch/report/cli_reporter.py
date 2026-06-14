# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# MoEWatch — moewatch/report/cli_reporter.py
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
# Rich-based terminal visualization backend for live MoEWatch training
# monitoring. Provides structured, well-aligned, colorful, minimal, and
# detailed real-time dashboards for training-time observability and
# intervention tracking. Uses Rich Panels, Tables, Rules, Columns, and
# styled Text for a polished multi-section terminal UI.
#
# Sections
# --------
#   1. Header              — Branded banner with step and timestamp
#   2. Routing Health      — Per-layer risk panels with gradient bars
#   3. Expert Health       — Per-layer expert utilization and health states
#   4. Entropy Trend       — Tier-2 normalized entropy with CUSUM markers
#   5. Active Interventions— Applied action log with delta and details
#   6. Policy Decisions    — Compact per-layer action summary
#
# =============================================================================

from __future__ import annotations

import io
import logging
import math
import os
import shutil
import sys
from typing import Dict, List, Optional, Tuple

from moewatch import Alert
from moewatch.analyzer.collapse import CollapseDetector, LayerCollapseReport
from moewatch.analyzer.entropy import EntropyAnalyzer, LayerEntropyReport
from moewatch.analyzer.risk_score import RiskLevel, RiskReport, RiskScoreFuser
from moewatch.config import AlertLevel, WatchConfig
from moewatch.report.watch_report import StepReport, WatchReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rich availability
# ---------------------------------------------------------------------------

try:
    from rich import box
    from rich.columns import Columns
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table
    from rich.text import Text
    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RICH_AVAILABLE = False

# ---------------------------------------------------------------------------
# ANSI fallback constants
# ---------------------------------------------------------------------------

_ANSI_RESET   = "\033[0m"
_ANSI_BOLD    = "\033[1m"
_ANSI_DIM     = "\033[2m"
_ANSI_RED     = "\033[31m"
_ANSI_GREEN   = "\033[32m"
_ANSI_YELLOW  = "\033[33m"
_ANSI_CYAN    = "\033[36m"
_ANSI_MAGENTA = "\033[35m"
_ANSI_WHITE   = "\033[37m"
_ANSI_ORANGE  = "\033[38;5;208m"

# Risk level → (rich_color, ansi_code)
_RISK_COLORS: Dict[str, Tuple[str, str]] = {
    RiskLevel.LOW.value:      ("bright_green",  _ANSI_GREEN),
    RiskLevel.MID.value:      ("yellow",         _ANSI_YELLOW),
    RiskLevel.HIGH.value:     ("orange1",        _ANSI_ORANGE),
    RiskLevel.CRITICAL.value: ("bold red",       _ANSI_RED),
}

_ALERT_COLORS: Dict[str, Tuple[str, str]] = {
    AlertLevel.INFO.value:     ("bright_cyan", _ANSI_CYAN),
    AlertLevel.WARNING.value:  ("yellow",      _ANSI_YELLOW),
    AlertLevel.CRITICAL.value: ("bold red",    _ANSI_RED),
}

_BAR_FILL  = "█"
_BAR_EMPTY = "░"
_VERSION   = "v0.2.0"

# Risk level → panel border style
_RISK_BORDER: Dict[str, str] = {
    RiskLevel.LOW.value:      "bright_green",
    RiskLevel.MID.value:      "yellow",
    RiskLevel.HIGH.value:     "orange1",
    RiskLevel.CRITICAL.value: "red",
}


# ---------------------------------------------------------------------------
# CLIReporter
# ---------------------------------------------------------------------------


class CLIReporter:
    """Rich terminal dashboard renderer for live MoEWatch training monitoring.

    Renders a structured, colorful, aligned multi-section terminal UI using
    Rich Panels, Tables, Columns, Rules, and styled Text. Falls back to
    plain ANSI text when Rich is unavailable. Respects NO_COLOR env var
    and WatchConfig.no_color.

    Parameters
    ----------
    config : WatchConfig
        Shared monitoring configuration.
    """

    def __init__(self, config: WatchConfig) -> None:
        self.config = config
        self._no_color: bool = (
            config.no_color or bool(os.environ.get("NO_COLOR", "").strip())
        )
        self._term_width: int = shutil.get_terminal_size(fallback=(120, 24)).columns

        if _RICH_AVAILABLE:
            self._console: Optional[Console] = Console(
                width=self._term_width,
                highlight=False,
                markup=True,
                no_color=self._no_color,
            )
        else:
            self._console = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_dashboard(
        self,
        watch_report: WatchReport,
        entropy_analyzer: EntropyAnalyzer,
        collapse_detector: CollapseDetector,
        risk_fuser: RiskScoreFuser,
    ) -> str:
        """Render the full monitoring dashboard to terminal and return as string.

        Parameters
        ----------
        watch_report : WatchReport
            Current rolling report. Must contain at least one StepReport.
        entropy_analyzer : EntropyAnalyzer
            Live analyzer for entropy trend data.
        collapse_detector : CollapseDetector
            Live detector for expert health states.
        risk_fuser : RiskScoreFuser
            Live fuser for per-layer risk reports.

        Returns
        -------
        str
            Full rendered dashboard as a plain string.
        """
        latest: Optional[StepReport] = watch_report.latest()
        if latest is None:
            msg = "CLIReporter: WatchReport is empty — nothing to render."
            logger.debug(msg)
            return msg

        entropy_reports: Dict[str, LayerEntropyReport] = (
            entropy_analyzer.last_reports
            if hasattr(entropy_analyzer, "last_reports") else {}
        )
        collapse_reports: Dict[str, LayerCollapseReport] = (
            collapse_detector.last_reports
            if hasattr(collapse_detector, "last_reports") else {}
        )
        risk_reports: Dict[str, RiskReport] = risk_fuser.get_all_latest_reports()

        if _RICH_AVAILABLE and self._console is not None:
            return self._render_rich(
                latest, watch_report,
                entropy_reports, collapse_reports, risk_reports,
            )
        return self._render_plain(
            latest, watch_report,
            entropy_reports, collapse_reports, risk_reports,
        )

    def render_alert(self, alert: Alert) -> str:
        """Format and print a single alert line to the terminal.

        Parameters
        ----------
        alert : Alert
            The alert to render.

        Returns
        -------
        str
            Formatted alert string.
        """
        if _RICH_AVAILABLE and self._console is not None:
            return self._render_alert_rich(alert)
        return self._render_alert_plain(alert)

    # ------------------------------------------------------------------
    # Rich rendering — single-pass, no duplication
    # ------------------------------------------------------------------

    def _render_rich(
        self,
        latest: StepReport,
        watch_report: WatchReport,
        entropy_reports: Dict[str, LayerEntropyReport],
        collapse_reports: Dict[str, LayerCollapseReport],
        risk_reports: Dict[str, RiskReport],
    ) -> str:
        """Render the full dashboard to terminal using Rich. Returns plain string."""
        assert self._console is not None

        loss_str = (
            f"loss={latest.loss:.5f}" if not math.isnan(latest.loss) else "loss=N/A"
        )
        ts = latest.step_datetime.strftime("%H:%M:%S")

        # Build all sections upfront
        risk_table   = self._build_risk_table(latest, risk_reports)
        util_table   = self._build_utilisation_table(collapse_reports)
        entropy_tbl  = self._build_entropy_table(entropy_reports)
        interv_table = self._build_intervention_table(latest) if latest.active_interventions else None
        policy_line  = self._build_policy_line(latest)

        # ── Render to terminal ──────────────────────────────────────────
        self._console.rule(
            f"[bold cyan] MoEWatch {_VERSION} [/]"
            f"[dim]  step=[/][bold white]{latest.step}[/]"
            f"[dim]  {ts}[/]",
            style="bold cyan",
            characters="─",
        )
        self._console.print()
        self._console.print(risk_table)
        if util_table is not None:
            self._console.print(util_table)
        if entropy_tbl is not None:
            self._console.print(entropy_tbl)
        if interv_table is not None:
            self._console.print(interv_table)
        if policy_line:
            self._console.print(
                Panel(
                    policy_line,
                    title="[bold white]Policy Decisions[/]",
                    border_style="dim white",
                    padding=(0, 2),
                )
            )
        self._console.rule(
            f"[dim]alerts={watch_report.num_alerts}  "
            f"interventions={watch_report.num_interventions}  "
            f"{loss_str}[/]",
            style="dim",
            characters="─",
        )
        self._console.print()

        # ── Capture plain string (no-color, no markup) for return value ─
        buf = io.StringIO()
        cap = Console(
            file=buf,
            width=self._term_width,
            highlight=False,
            markup=True,
            no_color=True,
        )
        cap.rule(f"MoEWatch {_VERSION}  step={latest.step}  {ts}")
        cap.print(risk_table)
        if util_table is not None:
            cap.print(util_table)
        if entropy_tbl is not None:
            cap.print(entropy_tbl)
        if interv_table is not None:
            cap.print(interv_table)
        if policy_line:
            cap.print(policy_line)
        cap.rule(
            f"alerts={watch_report.num_alerts}  "
            f"interventions={watch_report.num_interventions}  "
            f"{loss_str}"
        )
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _build_risk_table(
        self,
        step: StepReport,
        risk_reports: Dict[str, RiskReport],
    ) -> "Table":
        """Build the Routing Health table with styled risk bars."""
        table = Table(
            title="Routing Health",
            title_style="bold cyan",
            caption="Tier 1+2+3 fused risk per MoE layer",
            caption_style="dim",
            box=box.SIMPLE_HEAD,
            show_header=True,
            show_edge=True,
            header_style="bold cyan",
            border_style="cyan",
            width=min(self._term_width - 2, 112),
            padding=(0, 1),
            expand=False,
        )
        table.add_column("Layer",          style="white",        min_width=30, max_width=46, no_wrap=True)
        table.add_column("Risk Meter",     justify="left",       min_width=18)
        table.add_column("Score",          justify="center",     min_width=7)
        table.add_column("Level",          justify="center",     min_width=10)
        table.add_column("Signal",         justify="center",     min_width=12)
        table.add_column("Trend",          justify="center",     min_width=5)

        sorted_layers = sorted(
            step.risk_scores.items(), key=lambda kv: kv[1], reverse=True
        )

        for layer_name, score in sorted_layers:
            level_str  = step.risk_levels.get(layer_name, RiskLevel.LOW.value)
            rich_color, _ = _RISK_COLORS.get(level_str, ("white", _ANSI_WHITE))
            bar        = _ascii_risk_bar(score, width=16)
            rr         = risk_reports.get(layer_name)
            dominant   = str(rr.dominant_signal) if rr and rr.dominant_signal is not None else "—"
            trend      = _trend_arrow(step, layer_name, prev_score=None)

            table.add_row(
                f"[dim]{layer_name}[/]",
                f"[{rich_color}]{bar}[/]",
                f"[{rich_color}]{score:.3f}[/]",
                f"[{rich_color}]{level_str.upper()}[/]",
                f"[dim]{dominant}[/]",
                trend,
            )

        return table

    def _build_utilisation_table(
        self,
        collapse_reports: Dict[str, LayerCollapseReport],
    ) -> Optional["Table"]:
        """Build the Expert Health table. Returns None if no data."""
        if not collapse_reports:
            return None

        table = Table(
            title="Expert Utilisation",
            title_style="bold magenta",
            caption="Per-expert routing load and health state",
            caption_style="dim",
            box=box.SIMPLE_HEAD,
            show_header=True,
            show_edge=True,
            header_style="bold magenta",
            border_style="magenta",
            width=min(self._term_width - 2, 112),
            padding=(0, 1),
            expand=False,
        )
        table.add_column("Layer",         style="white",    min_width=30, no_wrap=True)
        table.add_column("Distribution",  min_width=32)
        table.add_column("Imbalance",     justify="center", min_width=10)
        table.add_column("Healthy",       justify="center", min_width=8)
        table.add_column("Cold",          justify="center", min_width=6)
        table.add_column("Dead",          justify="center", min_width=6)

        for layer_name, cr in sorted(collapse_reports.items()):
            expert_states   = getattr(cr, "expert_states", {})
            util_fractions  = {
                eid: es.utilization
                for eid, es in expert_states.items()
            }
            bar_str    = _expert_util_bar(util_fractions, width=28)
            imbalance  = getattr(cr, "load_imbalance_ratio", None)
            if imbalance is None:
                imbalance = getattr(cr, "load_imbalance", None)

            if imbalance is not None:
                if imbalance > 5.0:
                    imb_str = f"[bold red]{imbalance:.2f}x[/]"
                elif imbalance > 3.0:
                    imb_str = f"[yellow]{imbalance:.2f}x[/]"
                else:
                    imb_str = f"[bright_green]{imbalance:.2f}x[/]"
            else:
                imb_str = "[dim]—[/]"

            healthy = getattr(cr, "num_healthy_experts", 0)
            cold    = getattr(cr, "num_cold_experts", 0)
            dead    = getattr(cr, "num_dead_experts", 0)

            table.add_row(
                f"[dim]{layer_name}[/]",
                bar_str,
                imb_str,
                f"[bright_green]{healthy}[/]" if healthy > 0 else "[dim]0[/]",
                f"[yellow]{cold}[/]"           if cold > 0    else "[dim]0[/]",
                f"[bold red]{dead}[/]"          if dead > 0    else "[dim]0[/]",
            )

        return table

    def _build_entropy_table(
        self,
        entropy_reports: Dict[str, LayerEntropyReport],
    ) -> Optional["Table"]:
        """Build the Entropy Trend table. Returns None if no data."""
        if not entropy_reports:
            return None

        table = Table(
            title="Entropy Trend  (Tier 2)",
            title_style="bold yellow",
            caption="Normalised routing entropy — lower = more collapsed",
            caption_style="dim",
            box=box.SIMPLE_HEAD,
            show_header=True,
            show_edge=True,
            header_style="bold yellow",
            border_style="yellow",
            width=min(self._term_width - 2, 112),
            padding=(0, 1),
            expand=False,
        )
        table.add_column("Layer",      style="white",    min_width=30, no_wrap=True)
        table.add_column("H_norm",     justify="center", min_width=8)
        table.add_column("Entropy Bar",                  min_width=18)
        table.add_column("Trend",      justify="center", min_width=9)
        table.add_column("CUSUM",      justify="center", min_width=7)
        table.add_column("Drop/step",  justify="center", min_width=10)

        for layer_name, er in sorted(entropy_reports.items()):
            # Support both attribute names
            h_norm    = getattr(er, "normalized_entropy", None)
            if h_norm is None:
                h_norm = getattr(er, "entropy_norm", None)
            trend     = getattr(er, "trend", "UNKNOWN")
            drift     = getattr(er, "drift_detected", False)
            if not drift:
                drift = getattr(er, "cusum_triggered", False)
            drop_rate = getattr(er, "drop_rate", 0.0)

            if h_norm is not None:
                if h_norm >= 0.6:
                    ec = "bright_green"
                elif h_norm >= 0.3:
                    ec = "yellow"
                else:
                    ec = "bold red"
                h_display = f"[{ec}]{h_norm:.4f}[/]"
                ent_bar   = f"[{ec}]{_ascii_risk_bar(h_norm, width=16)}[/]"
            else:
                h_display = "[dim]—[/]"
                ent_bar   = "[dim]—[/]"

            cusum_str = "[bold red]ALERT[/]" if drift else "[dim]—[/]"
            drop_str  = (
                f"[yellow]{drop_rate:+.4f}[/]"
                if abs(drop_rate) > 0.001
                else f"[dim]{drop_rate:+.4f}[/]"
            )

            table.add_row(
                f"[dim]{layer_name}[/]",
                h_display,
                ent_bar,
                _entropy_trend_arrow(trend),
                cusum_str,
                drop_str,
            )

        return table

    def _build_intervention_table(self, step: StepReport) -> "Table":
        """Build the Active Interventions table."""
        table = Table(
            title="Active Interventions",
            title_style="bold red",
            caption="Interventions applied at this step",
            caption_style="dim",
            box=box.SIMPLE_HEAD,
            show_header=True,
            show_edge=True,
            header_style="bold red",
            border_style="red",
            width=min(self._term_width - 2, 112),
            padding=(0, 1),
            expand=False,
        )
        table.add_column("Action",  min_width=18)
        table.add_column("Layer",   style="white", min_width=30, no_wrap=True)
        table.add_column("Delta",   justify="right", min_width=8)
        table.add_column("Details", min_width=20)

        for action in step.active_interventions:
            action_type = getattr(action, "action_type", type(action).__name__)
            layer_name  = getattr(action, "layer_name", "—")
            delta       = getattr(action, "delta", None)
            delta_str   = f"{delta:+.4f}" if delta is not None else "—"
            log_line    = getattr(action, "log_line", "") or getattr(action, "description", "")

            table.add_row(
                f"[bold red]{action_type}[/]",
                f"[dim]{layer_name}[/]",
                f"[yellow]{delta_str}[/]",
                f"[dim]{log_line}[/]" if log_line else "[dim]—[/]",
            )

        return table

    def _build_policy_line(self, step: StepReport) -> Optional[str]:
        """Build a compact single-line policy decision summary."""
        if not step.policy_decisions:
            return None
        parts = [
            f"[dim]{layer}[/]  [bold cyan]{action}[/]"
            for layer, action in sorted(step.policy_decisions.items())
        ]
        return "  [bold white]Policy decisions:[/]   " + "     [dim]|[/]     ".join(parts)

    # ------------------------------------------------------------------
    # Alert rendering
    # ------------------------------------------------------------------

    def _render_alert_rich(self, alert: Alert) -> str:
        """Render a single alert using Rich markup and print it."""
        assert self._console is not None

        level_val = (
            alert.level.value if hasattr(alert.level, "value") else str(alert.level)
        )
        rich_color, _ = _ALERT_COLORS.get(level_val, ("white", _ANSI_WHITE))
        icons = {"critical": "[bold red]⬛[/]", "warning": "[yellow]▲[/]", "info": "[cyan]▸[/]"}
        icon  = icons.get(level_val, "•")

        line = (
            f"{icon}  [{rich_color}][{level_val.upper()}][/]  "
            f"step=[bold]{alert.step}[/]  "
            f"[dim]{alert.layer_id}[/]  "
            f"[italic]{alert.signal_type}[/]  "
            f"{alert.message}"
        )
        self._console.print(line)
        return line

    # ------------------------------------------------------------------
    # Plain-text fallback
    # ------------------------------------------------------------------

    def _render_plain(
        self,
        latest: StepReport,
        watch_report: WatchReport,
        entropy_reports: Dict[str, LayerEntropyReport],
        collapse_reports: Dict[str, LayerCollapseReport],
        risk_reports: Dict[str, RiskReport],
    ) -> str:
        """Render dashboard as plain ANSI text (no Rich)."""
        lines: List[str] = []

        def _c(code: str, text: str) -> str:
            return text if self._no_color else f"{code}{text}{_ANSI_RESET}"

        sep  = "─" * min(self._term_width, 80)
        sep2 = "━" * min(self._term_width, 80)
        ts   = latest.step_datetime.strftime("%Y-%m-%d %H:%M:%S")

        lines.append(sep2)
        lines.append(
            _c(_ANSI_CYAN + _ANSI_BOLD,
               f"  MoEWatch {_VERSION}   step={latest.step}   {ts}")
        )
        lines.append(sep2)
        lines.append("")

        # Routing Health
        lines.append(_c(_ANSI_BOLD, "  Routing Health"))
        lines.append(sep)
        for layer_name, score in sorted(
            latest.risk_scores.items(), key=lambda kv: kv[1], reverse=True
        ):
            level_str = latest.risk_levels.get(layer_name, RiskLevel.LOW.value)
            _, ansi   = _RISK_COLORS.get(level_str, ("white", _ANSI_WHITE))
            bar       = _ascii_risk_bar(score, width=16)
            lines.append(
                f"  {layer_name:<44s}  "
                + _c(ansi, f"{bar}  {score:.3f}  [{level_str.upper()}]")
            )
        lines.append("")

        # Entropy Trend
        if entropy_reports:
            lines.append(_c(_ANSI_BOLD, "  Entropy Trend"))
            lines.append(sep)
            for layer_name, er in sorted(entropy_reports.items()):
                h_norm = getattr(er, "normalized_entropy", None)
                if h_norm is None:
                    h_norm = getattr(er, "entropy_norm", None)
                trend  = getattr(er, "trend", "?")
                arrow  = _entropy_trend_arrow(trend)
                h_str  = f"{h_norm:.4f}" if h_norm is not None else "N/A"
                lines.append(f"  {layer_name:<44s}  H={h_str}  {arrow}")
            lines.append("")

        # Active Interventions
        if latest.active_interventions:
            lines.append(_c(_ANSI_BOLD + _ANSI_RED, "  Active Interventions"))
            lines.append(sep)
            for action in latest.active_interventions:
                action_type = getattr(action, "action_type", type(action).__name__)
                layer_name  = getattr(action, "layer_name", "—")
                delta       = getattr(action, "delta", None)
                delta_str   = f"{delta:+.4f}" if delta is not None else ""
                lines.append(f"  {action_type:<20s}  {layer_name}  {delta_str}")
            lines.append("")

        loss_str = (
            f"loss={latest.loss:.5f}" if not math.isnan(latest.loss) else "loss=N/A"
        )
        lines.append(
            _c(_ANSI_DIM,
               f"  alerts={watch_report.num_alerts}  "
               f"interventions={watch_report.num_interventions}  "
               f"{loss_str}")
        )
        lines.append(sep2)

        rendered = "\n".join(lines)
        print(rendered, file=sys.stdout)
        return rendered

    def _render_alert_plain(self, alert: Alert) -> str:
        """Render a single alert as plain ANSI text."""
        def _c(code: str, text: str) -> str:
            return text if self._no_color else f"{code}{text}{_ANSI_RESET}"

        level_val = (
            alert.level.value if hasattr(alert.level, "value") else str(alert.level)
        )
        _, ansi  = _ALERT_COLORS.get(level_val, ("white", _ANSI_WHITE))
        prefix   = "! " if level_val == AlertLevel.WARNING.value else "X "
        line = (
            f"{prefix}{_c(ansi, level_val.upper())}  "
            f"step={alert.step}  {alert.layer_id}  "
            f"{alert.signal_type}  {alert.message}"
        )
        print(line, file=sys.stdout)
        return line


# ---------------------------------------------------------------------------
# Module-private rendering helpers
# ---------------------------------------------------------------------------


def _ascii_risk_bar(score: float, width: int = 16) -> str:
    """Render a Unicode block progress bar for a score in [0, 1]."""
    score  = max(0.0, min(1.0, score))
    filled = int(round(score * width))
    return _BAR_FILL * filled + _BAR_EMPTY * (width - filled)


def _expert_util_bar(
    util_fractions: Dict[int, float], width: int = 28
) -> str:
    """Render a compact per-expert utilisation bar chart string.

    Returns a plain string (no Rich markup) so it is always renderable
    in a Rich Table cell.
    """
    if not util_fractions:
        return "(no data)"

    max_util = max(util_fractions.values()) or 1.0
    n        = len(util_fractions)
    w        = max(1, width // n)
    parts: List[str] = []

    for eid in sorted(util_fractions):
        frac   = util_fractions[eid] / max_util
        filled = int(round(frac * w))
        bar    = _BAR_FILL * filled + _BAR_EMPTY * (w - filled)
        parts.append(bar)

    return "|".join(parts)


def _entropy_trend_arrow(trend: str) -> str:
    """Map an entropy trend label to a short arrow string."""
    mapping = {
        "RISING":    "↑",
        "IMPROVING": "↑",
        "DECLINING": "↓",
        "DROPPING":  "↓",
        "STABLE":    "→",
        "UNKNOWN":   "?",
    }
    key = trend.upper() if isinstance(trend, str) else "UNKNOWN"
    return mapping.get(key, "?")


def _trend_arrow(
    step: StepReport,
    layer_name: str,
    prev_score: Optional[float],
) -> str:
    """Infer a plain risk trend arrow. Returns → if no previous score."""
    if prev_score is None:
        return "→"
    current = step.risk_scores.get(layer_name)
    if current is None:
        return "?"
    if current > prev_score + 0.01:
        return "↑"
    if current < prev_score - 0.01:
        return "↓"
    return "→"
