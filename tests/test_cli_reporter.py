# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# tests/test_cli_reporter_coverage.py
# =============================================================================
#
# Project      : MoEWatch
# Description  : Coverage-focused tests for moewatch/report/cli_reporter.py.
#
#                 Targets both the Rich rendering path (_render_rich and its
#                 _build_*_rich table builders) and the plain-text fallback
#                 path (_render_plain), plus render_alert (rich + plain),
#                 __init__ variants (no_color, NO_COLOR env var), and the
#                 module-private helper functions (_ascii_risk_bar,
#                 _expert_util_bar, _entropy_trend_arrow, _trend_arrow).
#
#                 entropy_analyzer / collapse_detector / risk_fuser are
#                 supplied as MagicMock objects exposing exactly the
#                 attributes/methods CLIReporter.render_dashboard() reads:
#                   - entropy_analyzer.last_reports  -> dict[str, LayerEntropyReport-like]
#                   - collapse_detector.last_reports -> dict[str, LayerCollapseReport]
#                   - risk_fuser.latest_scores()     -> dict[str, RiskReport]
#
# =============================================================================

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from moewatch import Alert
from moewatch.analyzer.collapse import ExpertState, ExpertStatus, LayerCollapseReport
from moewatch.analyzer.risk_score import RiskLevel, RiskReport
from moewatch.config import AlertLevel, OutputMode, WatchConfig
from moewatch.intervention.actions import AuxLossAction, NoOpAction
from moewatch.report.cli_reporter import (
    CLIReporter,
    _ascii_risk_bar,
    _entropy_trend_arrow,
    _expert_util_bar,
    _trend_arrow,
)
from moewatch.report.watch_report import StepReport, WatchReport


# ===========================================================================
# ── Helpers ───────────────────────────────────────────────────────────────
# ===========================================================================


def _config(**kwargs) -> WatchConfig:
    return WatchConfig(output=OutputMode.SILENT, **kwargs)


def _entropy_report(**kwargs) -> SimpleNamespace:
    """Build an object exposing the attributes _build_entropy_table_rich /
    _render_plain read via getattr (entropy_norm, trend, cusum_triggered,
    alert_level)."""
    defaults = dict(entropy_norm=0.5, trend="STABLE", cusum_triggered=False, alert_level=None)
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _watch_report_with_step(step_report: StepReport, max_steps: int = 10) -> WatchReport:
    wr = WatchReport(max_steps=max_steps)
    wr.append(step_report)
    return wr


def _mock_analyzers(entropy_reports=None, collapse_reports=None, risk_reports=None):
    entropy_analyzer = MagicMock()
    entropy_analyzer.last_reports = entropy_reports or {}

    collapse_detector = MagicMock()
    collapse_detector.last_reports = collapse_reports or {}

    risk_fuser = MagicMock()
    risk_fuser.latest_scores.return_value = risk_reports or {}

    return entropy_analyzer, collapse_detector, risk_fuser


# ===========================================================================
# ── 1. __init__ variants ─────────────────────────────────────────────────
# ===========================================================================


class TestCLIReporterInit:
    def test_default_construction_with_color(self) -> None:
        config = _config()
        reporter = CLIReporter(config)
        assert reporter._no_color is False
        assert reporter._console is not None
        assert reporter._term_width > 0

    def test_no_color_via_config(self) -> None:
        config = _config(no_color=True)
        reporter = CLIReporter(config)
        assert reporter._no_color is True
        # Rich is available in this environment, so console still exists
        # but with no_color/markup disabled.
        assert reporter._console is not None

    def test_no_color_via_env_var(self, monkeypatch) -> None:
        monkeypatch.setenv("NO_COLOR", "1")
        config = _config(no_color=False)
        reporter = CLIReporter(config)
        assert reporter._no_color is True

    def test_no_color_env_var_whitespace_only_is_ignored(self, monkeypatch) -> None:
        monkeypatch.setenv("NO_COLOR", "   ")
        config = _config(no_color=False)
        reporter = CLIReporter(config)
        assert reporter._no_color is False

    def test_rich_unavailable_falls_back_to_plain_console(self, monkeypatch) -> None:
        import moewatch.report.cli_reporter as cli_mod

        monkeypatch.setattr(cli_mod, "_RICH_AVAILABLE", False)
        config = _config()
        reporter = CLIReporter(config)
        assert reporter._console is None


# ===========================================================================
# ── 2. render_dashboard — empty report ──────────────────────────────────
# ===========================================================================


class TestRenderDashboardEmpty:
    def test_render_dashboard_with_empty_report_rich(self) -> None:
        config = _config()
        reporter = CLIReporter(config)
        empty_report = WatchReport(max_steps=10)
        entropy_analyzer, collapse_detector, risk_fuser = _mock_analyzers()

        result = reporter.render_dashboard(
            empty_report, entropy_analyzer, collapse_detector, risk_fuser
        )
        assert "nothing to render" in result.lower()

    def test_render_dashboard_with_empty_report_plain(self, monkeypatch) -> None:
        import moewatch.report.cli_reporter as cli_mod

        monkeypatch.setattr(cli_mod, "_RICH_AVAILABLE", False)
        config = _config()
        reporter = CLIReporter(config)
        empty_report = WatchReport(max_steps=10)
        entropy_analyzer, collapse_detector, risk_fuser = _mock_analyzers()

        result = reporter.render_dashboard(
            empty_report, entropy_analyzer, collapse_detector, risk_fuser
        )
        assert "nothing to render" in result.lower()


# ===========================================================================
# ── 3. render_dashboard — Rich path, full data ──────────────────────────
# ===========================================================================


class TestRenderDashboardRich:
    def _full_step_report(self) -> StepReport:
        action = AuxLossAction(layer_name="layers.0.gate", delta=0.05)
        action.mark_applied(10)
        no_op = NoOpAction(layer_name="layers.1.gate")

        return StepReport(
            step=10,
            timestamp=1700000000.0,
            risk_scores={
                "layers.0.gate": 0.85,
                "layers.1.gate": 0.30,
                "layers.2.gate": 0.10,
            },
            risk_levels={
                "layers.0.gate": RiskLevel.CRITICAL.value,
                "layers.1.gate": RiskLevel.MID.value,
                "layers.2.gate": RiskLevel.LOW.value,
            },
            active_interventions=[action, no_op],
            policy_decisions={"layers.0.gate": "aux_loss", "layers.1.gate": "noop"},
            loss=1.23456,
        )

    def _full_risk_reports(self) -> dict:
        return {
            "layers.0.gate": RiskReport(
                layer_name="layers.0.gate",
                risk_score=0.85,
                risk_level=RiskLevel.CRITICAL,
                dominant_signal="gradient",
            ),
            "layers.1.gate": RiskReport(
                layer_name="layers.1.gate",
                risk_score=0.30,
                risk_level=RiskLevel.MID,
                dominant_signal="entropy",
            ),
            "layers.2.gate": RiskReport(
                layer_name="layers.2.gate",
                risk_score=0.10,
                risk_level=RiskLevel.LOW,
                dominant_signal="none",
            ),
        }

    def _full_entropy_reports(self) -> dict:
        return {
            "layers.0.gate": _entropy_report(
                entropy_norm=0.15, trend="DROPPING", cusum_triggered=True, alert_level="CRITICAL"
            ),
            "layers.1.gate": _entropy_report(
                entropy_norm=0.6, trend="STABLE", cusum_triggered=False, alert_level="WARNING"
            ),
            "layers.2.gate": _entropy_report(
                entropy_norm=0.9, trend="RISING", cusum_triggered=False, alert_level=None
            ),
        }

    def _full_collapse_reports(self) -> dict:
        return {
            "layers.0.gate": SimpleNamespace(
                expert_states={
                    0: ExpertState(expert_id=0, status=ExpertStatus.DEAD),
                    1: ExpertState(expert_id=1, status=ExpertStatus.HEALTHY),
                },
                utilisation_fractions={0: 0.0, 1: 1.0},
                load_imbalance=2.5,
            ),
            "layers.1.gate": SimpleNamespace(
                expert_states={
                    0: ExpertState(expert_id=0, status=ExpertStatus.HEALTHY),
                    1: ExpertState(expert_id=1, status=ExpertStatus.COLD),
                },
                utilisation_fractions={0: 0.7, 1: 0.3},
                load_imbalance=None,
            ),
        }

    def test_full_dashboard_rich_with_color(self) -> None:
        config = _config()
        reporter = CLIReporter(config)
        watch_report = _watch_report_with_step(self._full_step_report())
        entropy_analyzer, collapse_detector, risk_fuser = _mock_analyzers(
            entropy_reports=self._full_entropy_reports(),
            collapse_reports=self._full_collapse_reports(),
            risk_reports=self._full_risk_reports(),
        )

        result = reporter.render_dashboard(
            watch_report, entropy_analyzer, collapse_detector, risk_fuser
        )

        assert "Routing Health" in result
        assert "layers.0.gate" in result
        assert "Expert Utilisation" in result
        assert "Entropy Trend" in result
        assert "Active Interventions" in result
        assert "Policy decisions" in result
        assert "MoEWatch" in result

    def test_full_dashboard_rich_no_color(self) -> None:
        config = _config(no_color=True)
        reporter = CLIReporter(config)
        watch_report = _watch_report_with_step(self._full_step_report())
        entropy_analyzer, collapse_detector, risk_fuser = _mock_analyzers(
            entropy_reports=self._full_entropy_reports(),
            collapse_reports=self._full_collapse_reports(),
            risk_reports=self._full_risk_reports(),
        )

        result = reporter.render_dashboard(
            watch_report, entropy_analyzer, collapse_detector, risk_fuser
        )
        assert "Routing Health" in result
        assert "layers.0.gate" in result

    def test_dashboard_with_no_interventions_or_policy(self) -> None:
        config = _config()
        reporter = CLIReporter(config)
        sr = StepReport(
            step=1,
            timestamp=1700000000.0,
            risk_scores={"layers.0.gate": 0.4},
            risk_levels={"layers.0.gate": RiskLevel.MID.value},
            loss=float("nan"),
        )
        watch_report = _watch_report_with_step(sr)
        entropy_analyzer, collapse_detector, risk_fuser = _mock_analyzers(
            risk_reports={
                "layers.0.gate": RiskReport(
                    layer_name="layers.0.gate", risk_score=0.4, risk_level=RiskLevel.MID
                )
            }
        )

        result = reporter.render_dashboard(
            watch_report, entropy_analyzer, collapse_detector, risk_fuser
        )
        assert "Routing Health" in result
        # No interventions table and no policy line should be present.
        assert "Active Interventions" not in result
        assert "Policy decisions" not in result
        # Loss N/A path
        assert "loss=N/A" in result or "loss" in result

    def test_dashboard_with_empty_risk_scores(self) -> None:
        config = _config()
        reporter = CLIReporter(config)
        sr = StepReport(step=1, timestamp=1700000000.0)
        watch_report = _watch_report_with_step(sr)
        entropy_analyzer, collapse_detector, risk_fuser = _mock_analyzers()

        result = reporter.render_dashboard(
            watch_report, entropy_analyzer, collapse_detector, risk_fuser
        )
        assert "Routing Health" in result


# ===========================================================================
# ── 4. render_dashboard — plain-text fallback path ──────────────────────
# ===========================================================================


class TestRenderDashboardPlain:
    def _patch_no_rich(self, monkeypatch) -> None:
        import moewatch.report.cli_reporter as cli_mod

        monkeypatch.setattr(cli_mod, "_RICH_AVAILABLE", False)

    def _full_step_report(self) -> StepReport:
        action = AuxLossAction(layer_name="layers.0.gate", delta=0.05)
        action.mark_applied(10)
        return StepReport(
            step=10,
            timestamp=1700000000.0,
            risk_scores={"layers.0.gate": 0.85, "layers.1.gate": 0.10},
            risk_levels={
                "layers.0.gate": RiskLevel.CRITICAL.value,
                "layers.1.gate": RiskLevel.LOW.value,
            },
            active_interventions=[action],
            loss=2.5,
        )

    def test_full_dashboard_plain_with_color(self, monkeypatch, capsys) -> None:
        self._patch_no_rich(monkeypatch)
        config = _config()
        reporter = CLIReporter(config)
        watch_report = _watch_report_with_step(self._full_step_report())
        entropy_analyzer, collapse_detector, risk_fuser = _mock_analyzers(
            entropy_reports={
                "layers.0.gate": _entropy_report(entropy_norm=0.2, trend="DROPPING"),
            },
            risk_reports={
                "layers.0.gate": RiskReport(
                    layer_name="layers.0.gate", risk_score=0.85, risk_level=RiskLevel.CRITICAL
                )
            },
        )

        result = reporter.render_dashboard(
            watch_report, entropy_analyzer, collapse_detector, risk_fuser
        )
        assert "MoEWatch" in result
        assert "Routing Health" in result
        assert "Entropy Trend" in result
        assert "Active Interventions" in result
        assert "layers.0.gate" in result
        assert "loss=2.50000" in result

        captured = capsys.readouterr()
        assert "MoEWatch" in captured.out

    def test_full_dashboard_plain_no_color(self, monkeypatch) -> None:
        self._patch_no_rich(monkeypatch)
        config = _config(no_color=True)
        reporter = CLIReporter(config)
        watch_report = _watch_report_with_step(self._full_step_report())
        entropy_analyzer, collapse_detector, risk_fuser = _mock_analyzers(
            risk_reports={
                "layers.0.gate": RiskReport(
                    layer_name="layers.0.gate", risk_score=0.85, risk_level=RiskLevel.CRITICAL
                )
            }
        )

        result = reporter.render_dashboard(
            watch_report, entropy_analyzer, collapse_detector, risk_fuser
        )
        # No ANSI escape codes should be present.
        assert "\033[" not in result

    def test_plain_dashboard_with_nan_loss_and_no_extras(self, monkeypatch) -> None:
        self._patch_no_rich(monkeypatch)
        config = _config()
        reporter = CLIReporter(config)
        sr = StepReport(
            step=1,
            timestamp=1700000000.0,
            risk_scores={"layers.0.gate": 0.5},
            risk_levels={"layers.0.gate": RiskLevel.MID.value},
        )
        watch_report = _watch_report_with_step(sr)
        entropy_analyzer, collapse_detector, risk_fuser = _mock_analyzers()

        result = reporter.render_dashboard(
            watch_report, entropy_analyzer, collapse_detector, risk_fuser
        )
        assert "loss=N/A" in result
        assert "Entropy Trend" not in result
        assert "Active Interventions" not in result

    def test_plain_dashboard_unknown_risk_level_uses_default_color(self, monkeypatch) -> None:
        self._patch_no_rich(monkeypatch)
        config = _config()
        reporter = CLIReporter(config)
        sr = StepReport(
            step=1,
            timestamp=1700000000.0,
            risk_scores={"layers.0.gate": 0.5},
            risk_levels={},  # missing -> defaults to RiskLevel.LOW.value
        )
        watch_report = _watch_report_with_step(sr)
        entropy_analyzer, collapse_detector, risk_fuser = _mock_analyzers()

        result = reporter.render_dashboard(
            watch_report, entropy_analyzer, collapse_detector, risk_fuser
        )
        assert "layers.0.gate" in result


# ===========================================================================
# ── 5. render_alert ───────────────────────────────────────────────────────
# ===========================================================================


class TestRenderAlert:
    @pytest.mark.parametrize(
        "level",
        [AlertLevel.INFO, AlertLevel.WARNING, AlertLevel.CRITICAL],
    )
    def test_render_alert_rich_all_levels(self, level) -> None:
        config = _config()
        reporter = CLIReporter(config)
        alert = Alert(
            step=5,
            level=level,
            layer_id="layers.0.gate",
            signal_type="entropy_drift",
            message="something happened",
        )
        result = reporter.render_alert(alert)
        assert "layers.0.gate" in result
        assert "something happened" in result
        assert level.value.upper() in result

    @pytest.mark.parametrize(
        "level",
        [AlertLevel.INFO, AlertLevel.WARNING, AlertLevel.CRITICAL],
    )
    def test_render_alert_plain_all_levels(self, monkeypatch, level) -> None:
        import moewatch.report.cli_reporter as cli_mod

        monkeypatch.setattr(cli_mod, "_RICH_AVAILABLE", False)
        config = _config()
        reporter = CLIReporter(config)
        alert = Alert(
            step=7,
            level=level,
            layer_id="layers.1.gate",
            signal_type="risk_score",
            message="plain alert message",
        )
        result = reporter.render_alert(alert)
        assert "layers.1.gate" in result
        assert "plain alert message" in result
        assert "step=7" in result

    def test_render_alert_plain_no_color(self, monkeypatch) -> None:
        import moewatch.report.cli_reporter as cli_mod

        monkeypatch.setattr(cli_mod, "_RICH_AVAILABLE", False)
        config = _config(no_color=True)
        reporter = CLIReporter(config)
        alert = Alert(
            step=1, level=AlertLevel.CRITICAL, layer_id="l0",
            signal_type="entropy_drift", message="msg",
        )
        result = reporter.render_alert(alert)
        assert "\033[" not in result

    def test_render_alert_rich_no_color(self) -> None:
        config = _config(no_color=True)
        reporter = CLIReporter(config)
        alert = Alert(
            step=2, level=AlertLevel.WARNING, layer_id="l1",
            signal_type="load_imbalance", message="imbalance high",
        )
        result = reporter.render_alert(alert)
        assert "l1" in result
        assert "imbalance high" in result


# ===========================================================================
# ── 6. Module-private helper functions ──────────────────────────────────
# ===========================================================================


class TestAsciiRiskBar:
    def test_zero_score(self) -> None:
        bar = _ascii_risk_bar(0.0, width=10)
        assert bar == "░" * 10

    def test_full_score(self) -> None:
        bar = _ascii_risk_bar(1.0, width=10)
        assert bar == "█" * 10

    def test_partial_score(self) -> None:
        bar = _ascii_risk_bar(0.5, width=10)
        assert bar.count("█") == 5
        assert bar.count("░") == 5

    def test_clamps_out_of_range(self) -> None:
        assert _ascii_risk_bar(-1.0, width=4) == "░" * 4
        assert _ascii_risk_bar(2.0, width=4) == "█" * 4


class TestExpertUtilBar:
    def test_empty_returns_no_data(self) -> None:
        assert _expert_util_bar({}, width=30) == "(no data)"

    def test_single_expert_full_bar(self) -> None:
        result = _expert_util_bar({0: 1.0}, width=10)
        assert "|" not in result  # single expert -> no separator
        assert "█" in result

    def test_multiple_experts_separated(self) -> None:
        result = _expert_util_bar({0: 1.0, 1: 0.5, 2: 0.0}, width=12)
        parts = result.split("|")
        assert len(parts) == 3

    def test_zero_max_util_handled(self) -> None:
        # All zero -> max_util defaults to 1.0 via "or 1.0"
        result = _expert_util_bar({0: 0.0, 1: 0.0}, width=8)
        assert "|" in result
        assert "█" not in result  # frac == 0 for all


class TestEntropyTrendArrow:
    @pytest.mark.parametrize(
        "trend,expected",
        [
            ("RISING", "↑"),
            ("rising", "↑"),
            ("DROPPING", "↓"),
            ("STABLE", "→"),
            ("UNKNOWN", "?"),
            ("something_else", "?"),
        ],
    )
    def test_known_and_unknown_trends(self, trend, expected) -> None:
        assert _entropy_trend_arrow(trend) == expected

    def test_non_string_trend(self) -> None:
        # isinstance check fails -> mapping.get(trend, "?") with non-str key
        assert _entropy_trend_arrow(None) == "?"
        assert _entropy_trend_arrow(123) == "?"


class TestTrendArrow:
    def test_no_prev_score_returns_flat(self) -> None:
        sr = StepReport(step=1, timestamp=1.0, risk_scores={"l0": 0.5})
        assert _trend_arrow(sr, "l0", prev_score=None) == "→"

    def test_layer_missing_from_current_returns_unknown(self) -> None:
        sr = StepReport(step=1, timestamp=1.0, risk_scores={})
        assert _trend_arrow(sr, "l0", prev_score=0.5) == "?"

    def test_increase_returns_up_arrow(self) -> None:
        sr = StepReport(step=1, timestamp=1.0, risk_scores={"l0": 0.6})
        assert _trend_arrow(sr, "l0", prev_score=0.4) == "↑"

    def test_decrease_returns_down_arrow(self) -> None:
        sr = StepReport(step=1, timestamp=1.0, risk_scores={"l0": 0.2})
        assert _trend_arrow(sr, "l0", prev_score=0.4) == "↓"

    def test_no_significant_change_returns_flat(self) -> None:
        sr = StepReport(step=1, timestamp=1.0, risk_scores={"l0": 0.401})
        assert _trend_arrow(sr, "l0", prev_score=0.4) == "→"
