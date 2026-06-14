# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# MoEWatch — moewatch/report/cli_reporter.py  [redesigned]
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
#
# This is a ground-up redesign of the original CLIReporter. Layout:
#
#   +----------------------------------------------------------------+
#   |  Top banner: title, step, timestamp, overall fused health gauge |
#   +----------------------------------------------------------------+
#   |  Layer Risk Map        |  Expert Health Grid                    |
#   |  (sparkline trend +    |  (per-layer expert bars, status        |
#   |   risk meter + level)  |   counts, imbalance ratio)             |
#   +----------------------------------------------------------------+
#   |  Entropy Radar (Tier 2 normalized entropy + drift markers)      |
#   +----------------------------------------------------------------+
#   |  Interventions Timeline   |  Policy Decisions                   |
#   +----------------------------------------------------------------+
#   |  Status bar: alerts, interventions, loss, worst layer           |
#   +----------------------------------------------------------------+
#
# Public API is unchanged from the previous implementation:
#   CLIReporter(config).render_dashboard(watch_report, entropy_analyzer,
#                                          collapse_detector, risk_fuser)
#   CLIReporter(config).render_alert(alert)
#
# Both still print to the terminal AND return a plain-text capture of the
# rendered output. render_dashboard now performs a SINGLE render pass
# (no duplicate printing).
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
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.progress_bar import ProgressBar
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

# Risk level -> (rich_color, ansi_code)
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

_ALERT_ICONS: Dict[str, str] = {
    AlertLevel.INFO.value:     "i",
    AlertLevel.WARNING.value:  "!",
    AlertLevel.CRITICAL.value: "X",
}

_RISK_ICONS: Dict[str, str] = {
    RiskLevel.LOW.value:      "o",
    RiskLevel.MID.value:      "~",
    RiskLevel.HIGH.value:     "^",
    RiskLevel.CRITICAL.value: "X",
}

_BAR_FILL     = "#"
_BAR_EMPTY    = "."
_SPARK_CHARS  = " .:-=+*#%@"
_VERSION      = "v0.2.0"

# Risk level -> panel / border style
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

    Redesigned multi-panel layout: a top status banner with an overall
    fused-health gauge, a side-by-side Layer Risk Map (with sparkline risk
    trends) and Expert Health grid, an Entropy Radar table, an
    Interventions / Policy panel pair, and a compact status bar footer.

    Falls back to a structured plain-ANSI rendering when Rich is
    unavailable. Respects ``NO_COLOR`` env var and ``WatchConfig.no_color``.

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
            # `force_terminal=True` ensures styling is emitted even when
            # stdout is piped, redirected, or running under a terminal
            # that Rich cannot positively identify as a TTY (common on
            # Windows shells and captured subprocesses). When the user
            # explicitly requests no color (config.no_color / NO_COLOR),
            # we still force a color system so layout/markup is parsed
            # consistently, but `no_color=True` strips the actual codes.
            self._console: Optional[Console] = Console(
                width=self._term_width,
                highlight=False,
                markup=True,
                no_color=self._no_color,
                force_terminal=True,
                color_system="standard" if not self._no_color else None,
                legacy_windows=False,
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
            Full rendered dashboard as a plain string (ANSI codes stripped
            when ``no_color`` is in effect).
        """
        latest: Optional[StepReport] = watch_report.latest()
        if latest is None:
            msg = "CLIReporter: WatchReport is empty -- nothing to render."
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
    # Rich rendering -- single pass, single print
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

        group = self._build_dashboard_group(
            latest, watch_report, entropy_reports, collapse_reports, risk_reports,
        )

        # -- Single render pass: print to the real console -------------
        self._console.print(group)

        # -- Capture an equivalent plain-text string for the return value -
        buf = io.StringIO()
        cap = Console(
            file=buf,
            width=self._term_width,
            highlight=False,
            markup=True,
            no_color=True,
            force_terminal=False,
            legacy_windows=False,
        )
        cap.print(group)
        return buf.getvalue()

    def _build_dashboard_group(
        self,
        latest: StepReport,
        watch_report: WatchReport,
        entropy_reports: Dict[str, LayerEntropyReport],
        collapse_reports: Dict[str, LayerCollapseReport],
        risk_reports: Dict[str, RiskReport],
    ) -> "Group":
        """Assemble every dashboard section into a single renderable Group."""
        renderables: List[object] = []

        renderables.append(self._build_banner(latest, watch_report))
        renderables.append(Text(""))

        # Side-by-side: Layer Risk Map | Expert Health
        risk_panel = self._build_risk_panel(latest, watch_report, risk_reports)
        expert_panel = self._build_expert_panel(collapse_reports)
        if expert_panel is not None:
            renderables.append(Columns([risk_panel, expert_panel], equal=False, expand=True))
        else:
            renderables.append(risk_panel)

        # Entropy Radar
        entropy_panel = self._build_entropy_panel(entropy_reports)
        if entropy_panel is not None:
            renderables.append(entropy_panel)

        # Side-by-side: Interventions | Policy Decisions
        interv_panel = self._build_intervention_panel(latest)
        policy_panel = self._build_policy_panel(latest)
        bottom_row: List[object] = [p for p in (interv_panel, policy_panel) if p is not None]
        if len(bottom_row) == 2:
            renderables.append(Columns(bottom_row, equal=True, expand=True))
        elif len(bottom_row) == 1:
            renderables.append(bottom_row[0])

        # Status bar footer
        renderables.append(self._build_status_bar(latest, watch_report))

        return Group(*renderables)

    # ------------------------------------------------------------------
    # Banner / overall health gauge
    # ------------------------------------------------------------------

    def _build_banner(self, latest: StepReport, watch_report: WatchReport) -> "Panel":
        """Top banner with title, step/time, and an overall fused-health gauge."""
        ts = latest.step_datetime.strftime("%Y-%m-%d %H:%M:%S")

        overall_score = self._overall_health_score(latest)
        overall_level = self._score_to_level(overall_score)
        color, _ = _RISK_COLORS.get(overall_level, ("white", _ANSI_WHITE))
        icon = _RISK_ICONS.get(overall_level, "?")

        gauge = ProgressBar(
            total=100,
            completed=int(round(overall_score * 100)),
            width=30,
            style="grey23",
            complete_style=color,
            finished_style=color,
        )

        left = Text.from_markup(
            f"[bold cyan]MoEWatch[/] [dim]{_VERSION}[/]\n"
            f"[white]step [bold]{latest.step}[/][/]   [dim]{ts}[/]"
        )

        right = Table.grid(padding=(0, 1))
        right.add_column(justify="right")
        right.add_column(justify="left")
        right.add_row(
            Text("Fused Health", style="dim"),
            Text(f"{icon} {overall_score:.3f}  [{overall_level}]", style=f"bold {color}"),
        )
        right.add_row(Text(""), gauge)

        header = Table.grid(expand=True)
        header.add_column(ratio=1)
        header.add_column(justify="right")
        header.add_row(left, right)

        return Panel(
            header,
            border_style=color,
            box=box.HEAVY,
            padding=(1, 2),
        )

    def _overall_health_score(self, step: StepReport) -> float:
        """Aggregate per-layer risk scores into a single overall figure (max)."""
        if not step.risk_scores:
            return 0.0
        return max(step.risk_scores.values())

    def _score_to_level(self, score: float) -> str:
        """Map a fused score to a RiskLevel string using standard thresholds."""
        if score >= 0.8:
            return RiskLevel.CRITICAL.value
        if score >= 0.6:
            return RiskLevel.HIGH.value
        if score >= 0.3:
            return RiskLevel.MID.value
        return RiskLevel.LOW.value

    # ------------------------------------------------------------------
    # Layer Risk Map (with sparkline trend history)
    # ------------------------------------------------------------------

    def _build_risk_panel(
        self,
        step: StepReport,
        watch_report: WatchReport,
        risk_reports: Dict[str, RiskReport],
    ) -> "Panel":
        """Build the Layer Risk Map panel: meter, score, level, sparkline trend."""
        table = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            show_edge=False,
            header_style="bold cyan",
            padding=(0, 1),
            expand=True,
        )
        table.add_column("Layer", style="white", ratio=3, no_wrap=True)
        table.add_column("Risk Meter", ratio=2)
        table.add_column("Score", justify="center", width=6)
        table.add_column("Level", justify="center", width=11)
        table.add_column("Signal", justify="center", ratio=1)
        table.add_column("Trend", justify="center", width=11)

        history = self._layer_score_history(watch_report, max_points=12)

        sorted_layers = sorted(
            step.risk_scores.items(), key=lambda kv: kv[1], reverse=True
        )

        for layer_name, score in sorted_layers:
            level_str = step.risk_levels.get(layer_name, RiskLevel.LOW.value)
            color, _ = _RISK_COLORS.get(level_str, ("white", _ANSI_WHITE))
            icon = _RISK_ICONS.get(level_str, "?")
            bar = _ascii_risk_bar(score, width=14)
            rr = risk_reports.get(layer_name)
            dominant = str(rr.dominant_signal) if rr and rr.dominant_signal is not None else "-"
            spark = _sparkline(history.get(layer_name, []))

            table.add_row(
                f"[dim]{_short_layer_name(layer_name)}[/]",
                f"[{color}]{bar}[/]",
                f"[{color}]{score:.3f}[/]",
                f"[{color}]{icon} {level_str.upper()}[/]",
                f"[dim]{dominant}[/]",
                f"[{color}]{spark}[/]" if spark else "[dim]-[/]",
            )

        return Panel(
            table,
            title="[bold cyan]Layer Risk Map[/]",
            subtitle="[dim]Tier 1+2+3 fused risk - sparkline = recent trend[/]",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 1),
        )

    def _layer_score_history(
        self, watch_report: WatchReport, max_points: int = 12
    ) -> Dict[str, List[float]]:
        """Collect recent per-layer risk scores across retained steps."""
        history: Dict[str, List[float]] = {}
        recent_steps = list(watch_report.steps)[-max_points:]
        for sr in recent_steps:
            for layer_name, score in sr.risk_scores.items():
                history.setdefault(layer_name, []).append(score)
        return history

    # ------------------------------------------------------------------
    # Expert Health grid
    # ------------------------------------------------------------------

    def _build_expert_panel(
        self, collapse_reports: Dict[str, LayerCollapseReport]
    ) -> Optional["Panel"]:
        """Build the Expert Health panel. Returns None if no data."""
        if not collapse_reports:
            return None

        table = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            show_edge=False,
            header_style="bold magenta",
            padding=(0, 1),
            expand=True,
        )
        table.add_column("Layer", style="white", ratio=3, no_wrap=True)
        table.add_column("Expert Load", ratio=4)
        table.add_column("Imb.", justify="center", width=7)
        table.add_column("ok/cold/dead", justify="center", width=13)

        for layer_name, cr in sorted(collapse_reports.items()):
            expert_states = getattr(cr, "expert_states", {})
            util_fractions = {
                eid: es.utilization for eid, es in expert_states.items()
            }
            bar_str = _expert_util_bar(util_fractions, width=24)

            imbalance = getattr(cr, "load_imbalance_ratio", None)
            if imbalance is not None:
                if imbalance > 5.0:
                    imb_str = f"[bold red]{imbalance:.1f}x[/]"
                elif imbalance > 3.0:
                    imb_str = f"[yellow]{imbalance:.1f}x[/]"
                else:
                    imb_str = f"[bright_green]{imbalance:.1f}x[/]"
            else:
                imb_str = "[dim]-[/]"

            healthy = getattr(cr, "num_healthy_experts", 0)
            cold = getattr(cr, "num_cold_experts", 0)
            dead = getattr(cr, "num_dead_experts", 0)

            counts = (
                f"[bright_green]{healthy}[/]/"
                f"[yellow]{cold}[/]/"
                f"[bold red]{dead}[/]"
            )

            table.add_row(
                f"[dim]{_short_layer_name(layer_name)}[/]",
                bar_str,
                imb_str,
                counts,
            )

        return Panel(
            table,
            title="[bold magenta]Expert Health[/]",
            subtitle="[dim]Per-expert load - healthy/cold/dead[/]",
            border_style="magenta",
            box=box.ROUNDED,
            padding=(0, 1),
        )

    # ------------------------------------------------------------------
    # Entropy Radar
    # ------------------------------------------------------------------

    def _build_entropy_panel(
        self, entropy_reports: Dict[str, LayerEntropyReport]
    ) -> Optional["Panel"]:
        """Build the Entropy Radar panel. Returns None if no data."""
        if not entropy_reports:
            return None

        table = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            show_edge=False,
            header_style="bold yellow",
            padding=(0, 1),
            expand=True,
        )
        table.add_column("Layer", style="white", ratio=3, no_wrap=True)
        table.add_column("H_norm", justify="center", width=8)
        table.add_column("Entropy", ratio=3)
        table.add_column("Trend", justify="center", width=7)
        table.add_column("Drift", justify="center", width=8)
        table.add_column("Delta/step", justify="center", width=11)

        for layer_name, er in sorted(entropy_reports.items()):
            h_norm = getattr(er, "normalized_entropy", None)
            trend = getattr(er, "trend", "UNKNOWN")
            drift = getattr(er, "drift_detected", False)
            drop_rate = getattr(er, "drop_rate", 0.0)

            if h_norm is not None:
                if h_norm >= 0.6:
                    ec = "bright_green"
                elif h_norm >= 0.3:
                    ec = "yellow"
                else:
                    ec = "bold red"
                h_display = f"[{ec}]{h_norm:.4f}[/]"
                ent_bar = f"[{ec}]{_ascii_risk_bar(h_norm, width=18)}[/]"
            else:
                h_display = "[dim]-[/]"
                ent_bar = "[dim]-[/]"

            drift_str = "[bold red]DRIFT[/]" if drift else "[dim]stable[/]"
            drop_str = (
                f"[yellow]{drop_rate:+.4f}[/]"
                if abs(drop_rate) > 0.001
                else f"[dim]{drop_rate:+.4f}[/]"
            )

            table.add_row(
                f"[dim]{_short_layer_name(layer_name)}[/]",
                h_display,
                ent_bar,
                _entropy_trend_arrow(trend),
                drift_str,
                drop_str,
            )

        return Panel(
            table,
            title="[bold yellow]Entropy Radar[/]  [dim](Tier 2)[/]",
            subtitle="[dim]Normalised routing entropy -- lower = more collapsed[/]",
            border_style="yellow",
            box=box.ROUNDED,
            padding=(0, 1),
        )

    # ------------------------------------------------------------------
    # Interventions timeline + Policy decisions
    # ------------------------------------------------------------------

    def _build_intervention_panel(self, step: StepReport) -> Optional["Panel"]:
        """Build the Active Interventions panel."""
        if not step.active_interventions:
            return Panel(
                Text("No interventions applied this step.", style="dim italic"),
                title="[bold red]Interventions[/]",
                border_style="dim red",
                box=box.ROUNDED,
                padding=(0, 1),
            )

        table = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            show_edge=False,
            header_style="bold red",
            padding=(0, 1),
            expand=True,
        )
        table.add_column("Action", ratio=2)
        table.add_column("Layer", style="white", ratio=3, no_wrap=True)
        table.add_column("Delta", justify="right", width=9)

        for action in step.active_interventions:
            action_type = getattr(action, "action_type", type(action).__name__)
            layer_name = getattr(action, "layer_name", "-")
            delta = getattr(action, "delta", None)
            delta_str = f"{delta:+.4f}" if delta is not None else "-"

            table.add_row(
                f"[bold red]{action_type}[/]",
                f"[dim]{_short_layer_name(layer_name)}[/]",
                f"[yellow]{delta_str}[/]",
            )

        return Panel(
            table,
            title="[bold red]Interventions[/]",
            subtitle="[dim]Applied at this step[/]",
            border_style="red",
            box=box.ROUNDED,
            padding=(0, 1),
        )

    def _build_policy_panel(self, step: StepReport) -> Optional["Panel"]:
        """Build the Policy Decisions panel."""
        if not step.policy_decisions:
            return Panel(
                Text("No policy decisions recorded.", style="dim italic"),
                title="[bold white]Policy Decisions[/]",
                border_style="dim white",
                box=box.ROUNDED,
                padding=(0, 1),
            )

        table = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            show_edge=False,
            header_style="bold white",
            padding=(0, 1),
            expand=True,
        )
        table.add_column("Layer", style="white", ratio=3, no_wrap=True)
        table.add_column("Action", justify="left", ratio=2)

        for layer_name, action in sorted(step.policy_decisions.items()):
            table.add_row(
                f"[dim]{_short_layer_name(layer_name)}[/]",
                f"[bold cyan]{action}[/]",
            )

        return Panel(
            table,
            title="[bold white]Policy Decisions[/]",
            border_style="dim white",
            box=box.ROUNDED,
            padding=(0, 1),
        )

    # ------------------------------------------------------------------
    # Status bar footer
    # ------------------------------------------------------------------

    def _build_status_bar(self, step: StepReport, watch_report: WatchReport) -> "Table":
        """Build a compact one-line status bar with key run-level stats."""
        loss_str = (
            f"{step.loss:.5f}" if not math.isnan(step.loss) else "N/A"
        )
        worst = step.worst_layer
        worst_str = _short_layer_name(worst) if worst else "-"
        worst_score = step.risk_scores.get(worst, 0.0) if worst else 0.0
        worst_level = step.risk_levels.get(worst, RiskLevel.LOW.value) if worst else RiskLevel.LOW.value
        worst_color, _ = _RISK_COLORS.get(worst_level, ("white", _ANSI_WHITE))

        bar = Table.grid(expand=True, padding=(0, 2))
        bar.add_column(justify="left", ratio=1)
        bar.add_column(justify="left", ratio=1)
        bar.add_column(justify="left", ratio=1)
        bar.add_column(justify="left", ratio=1)
        bar.add_row(
            Text.from_markup(f"[dim]Alerts[/] [bold]{watch_report.num_alerts}[/]"),
            Text.from_markup(f"[dim]Interventions[/] [bold]{watch_report.num_interventions}[/]"),
            Text.from_markup(f"[dim]Loss[/] [bold]{loss_str}[/]"),
            Text.from_markup(
                f"[dim]Worst layer[/] [{worst_color}]{worst_str} "
                f"({worst_score:.3f})[/]"
            ),
        )
        return bar

    # ------------------------------------------------------------------
    # Alert rendering
    # ------------------------------------------------------------------

    def _render_alert_rich(self, alert: Alert) -> str:
        """Render a single alert using Rich markup and print it.

        Returns a plain-text (markup-stripped) capture of the rendered
        line, matching the contract of :meth:`render_alert` /
        :meth:`render_dashboard`.
        """
        assert self._console is not None

        level_val = (
            alert.level.value if hasattr(alert.level, "value") else str(alert.level)
        )
        rich_color, _ = _ALERT_COLORS.get(level_val, ("white", _ANSI_WHITE))
        icon = _ALERT_ICONS.get(level_val, "*")

        line = (
            f"[{rich_color}]{icon} [{level_val.upper()}][/]  "
            f"[bold]step={alert.step}[/]  "
            f"[dim]{_short_layer_name(alert.layer_id)}[/]  "
            f"[italic]{alert.signal_type}[/]  "
            f"{alert.message}"
        )
        self._console.print(line)

        # -- Capture an equivalent plain-text string for the return value -
        buf = io.StringIO()
        cap = Console(
            file=buf,
            width=self._term_width,
            highlight=False,
            markup=True,
            no_color=True,
            force_terminal=False,
            legacy_windows=False,
        )
        cap.print(line)
        return buf.getvalue().rstrip("\n")


    # ------------------------------------------------------------------
    # Plain-text fallback (no Rich)
    # ------------------------------------------------------------------

    def _render_plain(
        self,
        latest: StepReport,
        watch_report: WatchReport,
        entropy_reports: Dict[str, LayerEntropyReport],
        collapse_reports: Dict[str, LayerCollapseReport],
        risk_reports: Dict[str, RiskReport],
    ) -> str:
        """Render dashboard as structured plain ANSI text (no Rich)."""
        lines: List[str] = []

        def _c(code: str, text: str) -> str:
            return text if self._no_color else f"{code}{text}{_ANSI_RESET}"

        width = min(self._term_width, 88)
        sep = "-" * width
        sep2 = "=" * width
        ts = latest.step_datetime.strftime("%Y-%m-%d %H:%M:%S")

        overall_score = self._overall_health_score(latest)
        overall_level = self._score_to_level(overall_score)
        _, overall_ansi = _RISK_COLORS.get(overall_level, ("white", _ANSI_WHITE))
        overall_icon = _RISK_ICONS.get(overall_level, "?")
        overall_bar = _ascii_risk_bar(overall_score, width=30)

        # -- Banner -------------------------------------------------
        lines.append(sep2)
        lines.append(
            _c(_ANSI_CYAN + _ANSI_BOLD, f"  MoEWatch {_VERSION}")
            + f"   step={latest.step}   {ts}"
        )
        lines.append(
            f"  Fused Health: "
            + _c(overall_ansi, f"{overall_icon} {overall_score:.3f} [{overall_level.upper()}]  {overall_bar}")
        )
        lines.append(sep2)
        lines.append("")

        # -- Layer Risk Map -------------------------------------------
        history = self._layer_score_history(watch_report, max_points=12)
        lines.append(_c(_ANSI_BOLD + _ANSI_CYAN, "  Layer Risk Map"))
        lines.append(sep)
        for layer_name, score in sorted(
            latest.risk_scores.items(), key=lambda kv: kv[1], reverse=True
        ):
            level_str = latest.risk_levels.get(layer_name, RiskLevel.LOW.value)
            _, ansi = _RISK_COLORS.get(level_str, ("white", _ANSI_WHITE))
            icon = _RISK_ICONS.get(level_str, "?")
            bar = _ascii_risk_bar(score, width=14)
            spark = _sparkline(history.get(layer_name, [])) or "-"
            lines.append(
                f"  {_short_layer_name(layer_name):<28s}  "
                + _c(ansi, f"{bar}  {score:.3f}  {icon} {level_str.upper():<8s}  {spark}")
            )
        lines.append("")

        # -- Expert Health ----------------------------------------------
        if collapse_reports:
            lines.append(_c(_ANSI_BOLD + _ANSI_MAGENTA, "  Expert Health"))
            lines.append(sep)
            for layer_name, cr in sorted(collapse_reports.items()):
                expert_states = getattr(cr, "expert_states", {})
                util_fractions = {eid: es.utilization for eid, es in expert_states.items()}
                bar_str = _expert_util_bar(util_fractions, width=24)
                imbalance = getattr(cr, "load_imbalance_ratio", None)
                imb_str = f"{imbalance:.1f}x" if imbalance is not None else "-"
                healthy = getattr(cr, "num_healthy_experts", 0)
                cold = getattr(cr, "num_cold_experts", 0)
                dead = getattr(cr, "num_dead_experts", 0)
                lines.append(
                    f"  {_short_layer_name(layer_name):<28s}  {bar_str}  "
                    f"imb={imb_str:<6s}  ok={healthy} cold={cold} dead={dead}"
                )
            lines.append("")

        # -- Entropy Radar ------------------------------------------------
        if entropy_reports:
            lines.append(_c(_ANSI_BOLD + _ANSI_YELLOW, "  Entropy Radar (Tier 2)"))
            lines.append(sep)
            for layer_name, er in sorted(entropy_reports.items()):
                h_norm = getattr(er, "normalized_entropy", None)
                trend = getattr(er, "trend", "?")
                drift = getattr(er, "drift_detected", False)
                arrow = _entropy_trend_arrow(trend)
                h_str = f"{h_norm:.4f}" if h_norm is not None else "N/A"
                drift_str = "DRIFT!" if drift else "stable"
                lines.append(
                    f"  {_short_layer_name(layer_name):<28s}  H={h_str}  {arrow}  {drift_str}"
                )
            lines.append("")

        # -- Active Interventions -----------------------------------------
        if latest.active_interventions:
            lines.append(_c(_ANSI_BOLD + _ANSI_RED, "  Interventions"))
            lines.append(sep)
            for action in latest.active_interventions:
                action_type = getattr(action, "action_type", type(action).__name__)
                layer_name = getattr(action, "layer_name", "-")
                delta = getattr(action, "delta", None)
                delta_str = f"{delta:+.4f}" if delta is not None else ""
                lines.append(f"  {action_type:<18s}  {_short_layer_name(layer_name)}  {delta_str}")
            lines.append("")

        # -- Policy Decisions -----------------------------------------
        if latest.policy_decisions:
            lines.append(_c(_ANSI_BOLD, "  Policy Decisions"))
            lines.append(sep)
            for layer_name, action in sorted(latest.policy_decisions.items()):
                lines.append(f"  {_short_layer_name(layer_name):<28s}  {action}")
            lines.append("")

        # -- Status bar -------------------------------------------------
        loss_str = (
            f"loss={latest.loss:.5f}" if not math.isnan(latest.loss) else "loss=N/A"
        )
        worst = latest.worst_layer
        worst_str = f"  worst={_short_layer_name(worst)}" if worst else ""
        lines.append(
            _c(_ANSI_DIM,
               f"  alerts={watch_report.num_alerts}  "
               f"interventions={watch_report.num_interventions}  "
               f"{loss_str}{worst_str}")
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
        _, ansi = _ALERT_COLORS.get(level_val, ("white", _ANSI_WHITE))
        icon = _ALERT_ICONS.get(level_val, "*")
        line = (
            f"{icon} {_c(ansi, level_val.upper())}  "
            f"step={alert.step}  {_short_layer_name(alert.layer_id)}  "
            f"{alert.signal_type}  {alert.message}"
        )
        print(line, file=sys.stdout)
        return line


# ---------------------------------------------------------------------------
# Module-private rendering helpers
# ---------------------------------------------------------------------------


def _ascii_risk_bar(score: float, width: int = 16) -> str:
    """Render a block progress bar for a score in [0, 1]."""
    score = max(0.0, min(1.0, score))
    filled = int(round(score * width))
    return _BAR_FILL * filled + _BAR_EMPTY * (width - filled)


def _expert_util_bar(util_fractions: Dict[int, float], width: int = 24) -> str:
    """Render a compact per-expert utilisation bar chart string.

    Returns a plain string (no Rich markup) so it is always renderable
    in a Rich Table cell.
    """
    if not util_fractions:
        return "(no data)"

    max_util = max(util_fractions.values()) or 1.0
    n = len(util_fractions)
    w = max(1, width // n)
    parts: List[str] = []

    for eid in sorted(util_fractions):
        frac = util_fractions[eid] / max_util
        filled = int(round(frac * w))
        bar = _BAR_FILL * filled + _BAR_EMPTY * (w - filled)
        parts.append(bar)

    return "|".join(parts)


def _sparkline(values: List[float]) -> str:
    """Render a compact sparkline for a list of values in [0, 1].

    Returns an empty string if fewer than 2 points are available.
    """
    if not values or len(values) < 2:
        return ""

    lo, hi = min(values), max(values)
    span = hi - lo
    chars: List[str] = []
    for v in values:
        if span <= 1e-9:
            idx = 0
        else:
            idx = int(round((v - lo) / span * (len(_SPARK_CHARS) - 1)))
        idx = max(0, min(len(_SPARK_CHARS) - 1, idx))
        chars.append(_SPARK_CHARS[idx])
    return "".join(chars)


def _entropy_trend_arrow(trend: str) -> str:
    """Map an entropy trend label to a short arrow string."""
    mapping = {
        "RISING":    "UP",
        "IMPROVING": "UP",
        "DECLINING": "DN",
        "DROPPING":  "DN",
        "STABLE":    "--",
        "UNKNOWN":   "??",
    }
    key = trend.upper() if isinstance(trend, str) else "UNKNOWN"
    return mapping.get(key, "??")


def _short_layer_name(layer_name: Optional[str], max_len: int = 32) -> str:
    """Shorten a fully-qualified layer/module name for compact display.

    Keeps the last two dotted segments (most identifying) and truncates
    the rest with an ellipsis if the result is still too long.

    Examples
    --------
    >>> _short_layer_name("model.layers.5.mlp.moe.router")
    'mlp.moe.router'
    """
    if not layer_name:
        return "-"

    parts = layer_name.split(".")
    if len(parts) > 2:
        short = ".".join(parts[-2:])
    else:
        short = layer_name

    if len(short) > max_len:
        short = short[: max_len - 1] + "."

    return short
