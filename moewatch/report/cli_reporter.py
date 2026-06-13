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
# Rich-based terminal dashboard for live MoEWatch training monitoring.
# Renders a structured, colorful, real-time CLI display with six sections:
#
#   1. Header                — MoEWatch banner, version, step, timestamp
#   2. Routing Health        — per-layer risk score gauges with color coding
#   3. Expert Utilisation    — per-layer expert load bar charts
#   4. Entropy Trend         — normalised entropy with CUSUM alert status
#   5. Active Interventions  — tabular log of applied actions
#   6. Policy Memory Stats   — action success rates from PolicyMemory
#
# The dashboard adapts to terminal width (shutil.get_terminal_size()).
# NO_COLOR environment variable and ``WatchConfig.no_color`` are both
# respected: if either is set, all ANSI color codes are suppressed and
# a plain-text fallback is used.
#
# Rich is used as the primary rendering backend.  If Rich is not installed,
# the reporter falls back to a minimal ANSI/plain-text mode with no layout
# engine (no panels, no tables).
#
# Contents
# --------
#   CLIReporter          — main dashboard renderer class
#
# Dependencies
# ------------
#   moewatch.report.watch_report    — WatchReport, StepReport
#   moewatch.analyzer.entropy       — EntropyAnalyzer, LayerEntropyReport
#   moewatch.analyzer.collapse      — CollapseDetector, LayerCollapseReport
#   moewatch.analyzer.risk_score    — RiskScoreFuser, RiskReport, RiskLevel
#   moewatch.config                 — WatchConfig, AlertLevel
#   moewatch.__init__               — Alert
#   rich (optional, graceful fallback)
#   os, shutil, sys, datetime, logging
#
# Usage
# -----
#   reporter = CLIReporter(config)
#   reporter.render_dashboard(watch_report, entropy_analyzer,
#                              collapse_detector, risk_fuser)
#   reporter.render_alert(alert)
#
# =============================================================================

from __future__ import annotations

import datetime
import logging
import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple

from moewatch import Alert
from moewatch.analyzer.collapse import CollapseDetector, LayerCollapseReport
from moewatch.analyzer.entropy import EntropyAnalyzer, LayerEntropyReport
from moewatch.analyzer.risk_score import RiskLevel, RiskReport, RiskScoreFuser
from moewatch.config import AlertLevel, WatchConfig
from moewatch.report.watch_report import StepReport, WatchReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rich availability check
# ---------------------------------------------------------------------------

try:
    from rich import box
    from rich.columns import Columns
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, TextColumn
    from rich.table import Table
    from rich.text import Text

    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RICH_AVAILABLE = False
    logger.debug(
        "rich is not installed. CLIReporter will use plain-text fallback. "
        "Install it with: pip install rich"
    )

# ---------------------------------------------------------------------------
# ANSI color constants (used in fallback mode)
# ---------------------------------------------------------------------------

_ANSI_RESET = "\033[0m"
_ANSI_BOLD = "\033[1m"
_ANSI_RED = "\033[31m"
_ANSI_YELLOW = "\033[33m"
_ANSI_GREEN = "\033[32m"
_ANSI_CYAN = "\033[36m"
_ANSI_MAGENTA = "\033[35m"
_ANSI_WHITE = "\033[37m"
_ANSI_DIM = "\033[2m"

# Risk level → Rich color string and fallback ANSI code.
_RISK_COLORS: Dict[str, Tuple[str, str]] = {
    RiskLevel.LOW.value: ("bright_green", _ANSI_GREEN),
    RiskLevel.MID.value: ("yellow", _ANSI_YELLOW),
    RiskLevel.HIGH.value: ("orange1", "\033[38;5;208m"),
    RiskLevel.CRITICAL.value: ("bold red", _ANSI_RED),
}

# Alert level → Rich color and ANSI code.
_ALERT_COLORS: Dict[str, Tuple[str, str]] = {
    AlertLevel.INFO.value: ("bright_cyan", _ANSI_CYAN),
    AlertLevel.WARNING.value: ("yellow", _ANSI_YELLOW),
    AlertLevel.CRITICAL.value: ("bold red", _ANSI_RED),
}

# Risk score fill characters for ASCII bar rendering.
_BAR_FILL = "█"
_BAR_EMPTY = "░"

# Version string embedded in dashboard header.
_VERSION = "v0.2.0"


# ---------------------------------------------------------------------------
# CLIReporter
# ---------------------------------------------------------------------------


class CLIReporter:
    """Renders a structured Rich terminal dashboard for live MoEWatch monitoring.

    Sections:
      1. Header: MoEWatch banner, version, current step, timestamp.
      2. Routing health: per-layer risk score meter with color coding.
      3. Expert utilisation: per-expert load bar chart.
      4. Entropy trend: normalised entropy with direction arrow.
      5. Active interventions: formatted log table.
      6. Policy memory: action success rate summary.

    Falls back to minimal plain-text ANSI output if Rich is not installed.
    Respects ``NO_COLOR`` env var and ``WatchConfig.no_color``.

    Parameters
    ----------
    config : WatchConfig
        Shared monitoring configuration.  ``no_color`` and ``output`` fields
        are consumed here.
    """

    def __init__(self, config: WatchConfig) -> None:
        self.config = config
        self._no_color: bool = (
            config.no_color or bool(os.environ.get("NO_COLOR", "").strip())
        )
        self._term_width: int = shutil.get_terminal_size(fallback=(120, 24)).columns

        # Rich Console — force_terminal ensures colour even in notebooks or CI.
        if _RICH_AVAILABLE and not self._no_color:
            self._console: Optional[Console] = Console(
                width=self._term_width,
                highlight=False,
                markup=True,
            )
        elif _RICH_AVAILABLE and self._no_color:
            self._console = Console(
                width=self._term_width,
                no_color=True,
                highlight=False,
                markup=False,
            )
        else:
            self._console = None  # plain-text fallback

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
        """Render the full monitoring dashboard and print it to the terminal.

        All six sections are composed and emitted.  The rendered string is
        also returned for callers that need to capture or log the output.

        Parameters
        ----------
        watch_report:
            Current rolling :class:`WatchReport`.  Must contain at least one
            :class:`StepReport`.
        entropy_analyzer:
            Live :class:`EntropyAnalyzer` instance for entropy trend data.
        collapse_detector:
            Live :class:`CollapseDetector` instance for expert health states.
        risk_fuser:
            Live :class:`RiskScoreFuser` for the latest per-layer risk reports.

        Returns
        -------
        str
            The full rendered dashboard as a plain string (ANSI codes stripped
            or included depending on ``no_color`` setting).
        """
        latest: Optional[StepReport] = watch_report.latest()
        if latest is None:
            msg = "CLIReporter: WatchReport is empty — nothing to render."
            logger.debug(msg)
            return msg

        # Gather data
        entropy_reports: Dict[str, LayerEntropyReport] = (
            entropy_analyzer.last_reports if hasattr(entropy_analyzer, "last_reports")
            else {}
        )
        collapse_reports: Dict[str, LayerCollapseReport] = (
            collapse_detector.last_reports if hasattr(collapse_detector, "last_reports")
            else {}
        )
        risk_reports: Dict[str, RiskReport] = risk_fuser.latest_scores()

        if _RICH_AVAILABLE and self._console is not None:
            return self._render_rich(
                latest,
                watch_report,
                entropy_reports,
                collapse_reports,
                risk_reports,
            )
        else:
            return self._render_plain(
                latest,
                watch_report,
                entropy_reports,
                collapse_reports,
                risk_reports,
            )

    def render_alert(self, alert: Alert) -> str:
        """Format and print a single alert line to the terminal.

        Parameters
        ----------
        alert:
            The :class:`Alert` to render.

        Returns
        -------
        str
            Formatted alert string (with or without ANSI codes).
        """
        if _RICH_AVAILABLE and self._console is not None:
            return self._render_alert_rich(alert)
        return self._render_alert_plain(alert)

    # ------------------------------------------------------------------
    # Rich rendering path
    # ------------------------------------------------------------------

    def _render_rich(
        self,
        latest: StepReport,
        watch_report: WatchReport,
        entropy_reports: Dict[str, LayerEntropyReport],
        collapse_reports: Dict[str, LayerCollapseReport],
        risk_reports: Dict[str, RiskReport],
    ) -> str:
        """Render dashboard using Rich layout engine.

        Returns the rendered string captured from the Console.
        """
        assert self._console is not None

        import io

        buf = io.StringIO()
        capture_console = Console(
            file=buf,
            width=self._term_width,
            highlight=False,
            markup=True,
            no_color=self._no_color,
        )

        # 1. Header
        capture_console.rule(
            f"[bold cyan]MoEWatch {_VERSION}[/]   "
            f"step=[bold]{latest.step}[/]   "
            f"{latest.step_datetime.strftime('%H:%M:%S')}",
            style="cyan",
        )

        # 2. Routing health table
        capture_console.print(self._build_risk_table_rich(latest, risk_reports))

        # 3. Expert utilisation table
        util_table = self._build_utilisation_table_rich(collapse_reports)
        if util_table is not None:
            capture_console.print(util_table)

        # 4. Entropy trend table
        entropy_table = self._build_entropy_table_rich(entropy_reports)
        if entropy_table is not None:
            capture_console.print(entropy_table)

        # 5. Active interventions
        if latest.active_interventions:
            capture_console.print(
                self._build_intervention_table_rich(latest)
            )

        # 6. Policy memory summary
        policy_line = self._build_policy_line(latest)
        if policy_line:
            capture_console.print(policy_line)

        # Footer
        import math

        loss_str = (
            f"loss={latest.loss:.5f}" if not math.isnan(latest.loss) else "loss=N/A"
        )
        capture_console.rule(
            f"[dim]alerts={watch_report.num_alerts}  "
            f"interventions={watch_report.num_interventions}  "
            f"{loss_str}[/]",
            style="dim",
        )

        rendered = buf.getvalue()

        # Print to real console
        self._console.print(self._build_risk_table_rich(latest, risk_reports))
        if util_table is not None:
            self._console.print(util_table)
        if entropy_table is not None:
            self._console.print(entropy_table)
        if latest.active_interventions:
            self._console.print(self._build_intervention_table_rich(latest))
        if policy_line:
            self._console.print(policy_line)
        self._console.rule(
            f"[dim]alerts={watch_report.num_alerts}  "
            f"interventions={watch_report.num_interventions}  "
            f"{loss_str}[/]",
            style="dim",
        )
        self._console.rule(style="cyan")

        return rendered

    def _build_risk_table_rich(
        self,
        step: StepReport,
        risk_reports: Dict[str, RiskReport],
    ) -> "Table":
        """Build the per-layer risk score Rich table."""
        table = Table(
            title="Routing Health",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold cyan",
            width=min(self._term_width, 110),
        )
        table.add_column("Layer", style="dim", min_width=30, max_width=50)
        table.add_column("Risk Score", justify="center", min_width=22)
        table.add_column("Level", justify="center", min_width=10)
        table.add_column("Dominant Signal", min_width=16)
        table.add_column("Trend", min_width=6, justify="center")

        # Sort by risk score descending.
        sorted_layers = sorted(
            step.risk_scores.items(), key=lambda kv: kv[1], reverse=True
        )

        for layer_name, score in sorted_layers:
            level_str = step.risk_levels.get(layer_name, RiskLevel.LOW.value)
            rich_color, _ = _RISK_COLORS.get(level_str, ("white", _ANSI_WHITE))
            bar = _ascii_risk_bar(score, width=14)
            rr = risk_reports.get(layer_name)
            dominant = rr.dominant_signal if rr else "—"
            trend = _trend_arrow(
                step, layer_name, prev_score=None
            )
            table.add_row(
                layer_name,
                f"[{rich_color}]{bar} {score:.3f}[/]",
                f"[{rich_color}]{level_str.upper()}[/]",
                dominant,
                trend,
            )

        return table

    def _build_utilisation_table_rich(
        self,
        collapse_reports: Dict[str, LayerCollapseReport],
    ) -> Optional["Table"]:
        """Build the expert utilisation Rich table.

        Returns ``None`` if no collapse reports are available.
        """
        if not collapse_reports:
            return None

        table = Table(
            title="Expert Utilisation",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold cyan",
            width=min(self._term_width, 110),
        )
        table.add_column("Layer", style="dim", min_width=30)
        table.add_column("Expert Distribution", min_width=40)
        table.add_column("Load Imbalance", justify="right", min_width=14)
        table.add_column("Dead", justify="center", min_width=6)

        for layer_name, cr in sorted(collapse_reports.items()):
            # Build a compact bar chart of expert utilisation fractions.
            util_fractions = getattr(cr, "utilisation_fractions", {})
            bar_str = _expert_util_bar(util_fractions, width=30)
            imbalance = getattr(cr, "load_imbalance", None)
            imbalance_str = f"{imbalance:.2f}x" if imbalance is not None else "—"
            dead_count = sum(
                1
                for es in cr.expert_states.values()
                if es.status.value == "dead"
            )
            dead_str = (
                f"[bold red]{dead_count}[/]" if dead_count > 0 else "0"
            )
            table.add_row(
                layer_name, bar_str, imbalance_str, dead_str
            )

        return table

    def _build_entropy_table_rich(
        self,
        entropy_reports: Dict[str, LayerEntropyReport],
    ) -> Optional["Table"]:
        """Build the entropy trend Rich table.

        Returns ``None`` if no entropy reports are available.
        """
        if not entropy_reports:
            return None

        table = Table(
            title="Entropy Trend (Tier 2)",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold cyan",
            width=min(self._term_width, 110),
        )
        table.add_column("Layer", style="dim", min_width=30)
        table.add_column("H_norm", justify="center", min_width=12)
        table.add_column("Trend", justify="center", min_width=6)
        table.add_column("CUSUM Alert", min_width=14)
        table.add_column("Status", min_width=12)

        for layer_name, er in sorted(entropy_reports.items()):
            h_norm = getattr(er, "entropy_norm", None)
            trend = getattr(er, "trend", "UNKNOWN")
            cusum_alert = getattr(er, "cusum_triggered", False)
            alert_level = getattr(er, "alert_level", None)

            h_str = f"{h_norm:.4f}" if h_norm is not None else "—"
            trend_arrow = _entropy_trend_arrow(trend)
            cusum_str = "[bold red]YES[/]" if cusum_alert else "—"
            status_color = (
                "bold red"
                if alert_level == "CRITICAL"
                else "yellow"
                if alert_level == "WARNING"
                else "bright_green"
            )
            status_str = f"[{status_color}]{alert_level or 'OK'}[/]"

            table.add_row(layer_name, h_str, trend_arrow, cusum_str, status_str)

        return table

    def _build_intervention_table_rich(
        self,
        step: StepReport,
    ) -> "Table":
        """Build the active interventions Rich table."""
        table = Table(
            title="Active Interventions",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold magenta",
            width=min(self._term_width, 110),
        )
        table.add_column("Action", min_width=22)
        table.add_column("Layer", style="dim", min_width=30)
        table.add_column("Delta", justify="right", min_width=8)
        table.add_column("Details", min_width=30)

        for action in step.active_interventions:
            action_type = type(action).__name__
            layer_name = getattr(action, "layer_name", "—")
            delta = getattr(action, "delta", None)
            delta_str = f"{delta:+.4f}" if delta is not None else "—"
            log_line = getattr(action, "log_line", "")
            table.add_row(
                f"[bold magenta]{action_type}[/]",
                layer_name,
                delta_str,
                log_line,
            )

        return table

    def _build_policy_line(self, step: StepReport) -> Optional[str]:
        """Build a compact single-line policy decision summary.

        Returns ``None`` if no policy decisions were recorded.
        """
        if not step.policy_decisions:
            return None
        parts = [
            f"{layer}: [bold]{action}[/]"
            for layer, action in sorted(step.policy_decisions.items())
        ]
        return "  [cyan]Policy decisions:[/] " + "  |  ".join(parts)

    def _render_alert_rich(self, alert: Alert) -> str:
        """Render a single alert using Rich markup and print it.

        Parameters
        ----------
        alert:
            Alert to render.

        Returns
        -------
        str
            Formatted alert string.
        """
        assert self._console is not None

        level_val = (
            alert.level.value if hasattr(alert.level, "value") else str(alert.level)
        )
        rich_color, _ = _ALERT_COLORS.get(level_val, ("white", _ANSI_WHITE))
        prefix_map = {
            AlertLevel.CRITICAL.value: "⛔",
            AlertLevel.WARNING.value: "⚠️ ",
            AlertLevel.INFO.value: "ℹ️ ",
        }
        prefix = prefix_map.get(level_val, "•")
        line = (
            f"{prefix}  [{rich_color}][{level_val.upper()}][/]  "
            f"step=[bold]{alert.step}[/]  "
            f"[dim]{alert.layer_id}[/]  "
            f"[italic]{alert.signal_type}[/]  "
            f"{alert.message}"
        )
        self._console.print(line)
        return line

    # ------------------------------------------------------------------
    # Plain-text fallback rendering path
    # ------------------------------------------------------------------

    def _render_plain(
        self,
        latest: StepReport,
        watch_report: WatchReport,
        entropy_reports: Dict[str, LayerEntropyReport],
        collapse_reports: Dict[str, LayerCollapseReport],
        risk_reports: Dict[str, RiskReport],
    ) -> str:
        """Render dashboard as plain ANSI text (no Rich dependency)."""
        c = "" if self._no_color else None  # sentinel
        lines: List[str] = []

        def _c(code: str, text: str) -> str:
            if self._no_color:
                return text
            return f"{code}{text}{_ANSI_RESET}"

        sep = "─" * min(self._term_width, 80)

        # Header
        lines.append(sep)
        lines.append(
            _c(
                _ANSI_CYAN + _ANSI_BOLD,
                f"  MoEWatch {_VERSION}   "
                f"step={latest.step}   "
                f"{latest.step_datetime.strftime('%Y-%m-%d %H:%M:%S')}",
            )
        )
        lines.append(sep)

        # Risk scores
        lines.append(_c(_ANSI_BOLD, "  Routing Health"))
        lines.append("")
        for layer_name, score in sorted(
            latest.risk_scores.items(), key=lambda kv: kv[1], reverse=True
        ):
            level_str = latest.risk_levels.get(layer_name, RiskLevel.LOW.value)
            _, ansi_code = _RISK_COLORS.get(level_str, ("white", _ANSI_WHITE))
            bar = _ascii_risk_bar(score, width=16)
            lines.append(
                f"  {layer_name:<42s}  "
                + _c(ansi_code, f"{bar} {score:.3f}  [{level_str.upper()}]")
            )
        lines.append("")

        # Entropy trend (compact)
        if entropy_reports:
            lines.append(_c(_ANSI_BOLD, "  Entropy Trend"))
            lines.append("")
            for layer_name, er in sorted(entropy_reports.items()):
                h_norm = getattr(er, "entropy_norm", None)
                trend = getattr(er, "trend", "?")
                arrow = _entropy_trend_arrow(trend)
                h_str = f"{h_norm:.4f}" if h_norm is not None else "N/A"
                lines.append(f"  {layer_name:<42s}  H={h_str}  {arrow}")
            lines.append("")

        # Active interventions (compact)
        if latest.active_interventions:
            lines.append(_c(_ANSI_BOLD + _ANSI_MAGENTA, "  Active Interventions"))
            lines.append("")
            for action in latest.active_interventions:
                log_line = getattr(action, "log_line", str(action))
                lines.append(f"  → {log_line}")
            lines.append("")

        # Footer
        import math

        loss_str = (
            f"loss={latest.loss:.5f}" if not math.isnan(latest.loss) else "loss=N/A"
        )
        lines.append(
            _c(
                _ANSI_DIM,
                f"  alerts={watch_report.num_alerts}  "
                f"interventions={watch_report.num_interventions}  "
                f"{loss_str}",
            )
        )
        lines.append(sep)

        rendered = "\n".join(lines)
        print(rendered, file=sys.stdout)
        return rendered

    def _render_alert_plain(self, alert: Alert) -> str:
        """Render a single alert as plain ANSI text.

        Parameters
        ----------
        alert:
            Alert to render.

        Returns
        -------
        str
            Formatted alert string.
        """

        def _c(code: str, text: str) -> str:
            if self._no_color:
                return text
            return f"{code}{text}{_ANSI_RESET}"

        level_val = (
            alert.level.value if hasattr(alert.level, "value") else str(alert.level)
        )
        _, ansi_code = _ALERT_COLORS.get(level_val, ("white", _ANSI_WHITE))
        prefix = "⚠ " if level_val == AlertLevel.WARNING.value else "⛔ "
        line = (
            f"{prefix}{_c(ansi_code, level_val.upper())}  "
            f"step={alert.step}  "
            f"{alert.layer_id}  "
            f"{alert.signal_type}  "
            f"{alert.message}"
        )
        print(line, file=sys.stdout)
        return line


# ---------------------------------------------------------------------------
# Module-private rendering helpers
# ---------------------------------------------------------------------------


def _ascii_risk_bar(score: float, width: int = 14) -> str:
    """Render a Unicode block progress bar for a risk score in [0, 1].

    Parameters
    ----------
    score:
        Risk score in [0.0, 1.0].
    width:
        Bar width in characters.

    Returns
    -------
    str
        A compact bar like ``"████████░░░░░░"`` with no brackets.
    """
    score = max(0.0, min(1.0, score))
    filled = int(round(score * width))
    empty = width - filled
    return _BAR_FILL * filled + _BAR_EMPTY * empty


def _expert_util_bar(
    util_fractions: Dict[int, float], width: int = 30
) -> str:
    """Render a compact expert utilisation bar chart.

    Each expert is represented by a single character column whose fill level
    indicates its share of routed tokens.

    Parameters
    ----------
    util_fractions:
        Mapping of ``expert_id → utilisation_fraction`` in [0.0, 1.0].
    width:
        Approximate total character width for the chart.

    Returns
    -------
    str
        Compact bar chart string.
    """
    if not util_fractions:
        return "(no data)"

    # Normalise to [0, 1] relative to max utilisation for visual clarity.
    max_util = max(util_fractions.values()) or 1.0
    n_experts = len(util_fractions)
    chars_per_expert = max(1, width // n_experts)

    parts: List[str] = []
    for eid in sorted(util_fractions):
        frac = util_fractions[eid] / max_util
        bar_len = int(round(frac * chars_per_expert))
        empty_len = chars_per_expert - bar_len
        parts.append(_BAR_FILL * bar_len + _BAR_EMPTY * empty_len)

    return "|".join(parts)


def _entropy_trend_arrow(trend: str) -> str:
    """Map an entropy trend label to a Unicode directional arrow.

    Parameters
    ----------
    trend:
        Trend label from ``LayerEntropyReport.trend``.
        Expected values: ``"STABLE"``, ``"DROPPING"``, ``"RISING"``,
        ``"UNKNOWN"``.

    Returns
    -------
    str
        Unicode arrow: ``↑`` / ``↓`` / ``→`` / ``?``.
    """
    mapping = {
        "RISING": "↑",
        "DROPPING": "↓",
        "STABLE": "→",
        "UNKNOWN": "?",
    }
    return mapping.get(trend.upper() if isinstance(trend, str) else trend, "?")


def _trend_arrow(
    step: StepReport,
    layer_name: str,
    prev_score: Optional[float],
) -> str:
    """Infer a risk trend arrow for a layer from the current step.

    In the absence of previous-step data, returns ``"→"``.

    Parameters
    ----------
    step:
        Current ``StepReport``.
    layer_name:
        Target layer.
    prev_score:
        Risk score at the previous step, or ``None`` if unavailable.

    Returns
    -------
    str
        Arrow string.
    """
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
