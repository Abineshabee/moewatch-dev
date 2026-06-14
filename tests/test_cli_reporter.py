# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# tests/test_cli_reporter.py
# =============================================================================
#
# Project      : MoEWatch
# Description  : Coverage-focused tests for the redesigned
#                moewatch/report/cli_reporter.py (v0.2.0).
#
#                - Covers __init__ variants (no_color, NO_COLOR env var, Rich
#                unavailable), render_dashboard (empty report, Rich path,
#                plain-ANSI fallback path, with/without entropy/collapse
#                data, NaN loss, unknown risk level), render_alert (Rich +
#                plain, all AlertLevels, no_color), and the module-private
#                helper functions (_ascii_risk_bar, _expert_util_bar,
#                _sparkline, _entropy_trend_arrow, _short_layer_name).
#
#                - entropy_analyzer / collapse_detector are supplied as
#                MagicMock objects exposing `.last_reports`; risk_fuser
#                exposes get_all_latest_reports() returning a real
#                dict[str, RiskReport].
#
# =============================================================================

from __future__ import annotations

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
    _short_layer_name,
    _sparkline,
)
from moewatch.report.watch_report import StepReport, WatchReport


# ===========================================================================
# Helpers
# ===========================================================================


def _config(**kwargs) -> WatchConfig:
    return WatchConfig(output=OutputMode.SILENT, **kwargs)


def _entropy_report(**kwargs) -> SimpleNamespace:
    """Build a SimpleNamespace exposing all attributes CLIReporter reads."""
    defaults = dict(
        normalized_entropy=0.5,
        trend="STABLE",
        drift_detected=False,
        drop_rate=0.0,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _watch_report_with_step(step_report: StepReport, max_steps: int = 10) -> WatchReport:
    wr = WatchReport(max_steps=max_steps)
    wr.append(step_report)
    return wr


def _mock_analyzers(entropy_reports=None, collapse_reports=None, risk_reports=None):
    """Build mock analyzer objects. risk_fuser uses get_all_latest_reports()."""
    entropy_analyzer = MagicMock()
    entropy_analyzer.last_reports = entropy_reports or {}

    collapse_detector = MagicMock()
    collapse_detector.last_reports = collapse_reports or {}

    risk_fuser = MagicMock()
    risk_fuser.get_all_latest_reports.return_value = risk_reports or {}

    return entropy_analyzer, collapse_detector, risk_fuser


def _patch_no_rich(monkeypatch) -> None:
    import moewatch.report.cli_reporter as cli_mod
    monkeypatch.setattr(cli_mod, "_RICH_AVAILABLE", False)


# ===========================================================================
# 1. __init__ variants
# ===========================================================================


class TestCLIReporterInit:
    def test_default_construction_with_color(self) -> None:
        config = _config()
        reporter = CLIReporter(config)
        assert reporter._no_color is False
        assert reporter._term_width > 0

    def test_no_color_via_config(self) -> None:
        config = _config(no_color=True)
        reporter = CLIReporter(config)
        assert reporter._no_color is True

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
        _patch_no_rich(monkeypatch)
        config = _config()
        reporter = CLIReporter(config)
        assert reporter._console is None


# ===========================================================================
# 2. render_dashboard — empty report
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
        _patch_no_rich(monkeypatch)
        config = _config()
        reporter = CLIReporter(config)
        empty_report = WatchReport(max_steps=10)
        entropy_analyzer, collapse_detector, risk_fuser = _mock_analyzers()

        result = reporter.render_dashboard(
            empty_report, entropy_analyzer, collapse_detector, risk_fuser
        )
        assert "nothing to render" in result.lower()


# ===========================================================================
# 3. render_dashboard — full data (Rich + plain)
# ===========================================================================


class TestRenderDashboardFull:
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
                normalized_entropy=0.15, trend="DROPPING",
                drift_detected=True, drop_rate=-0.02,
            ),
            "layers.1.gate": _entropy_report(
                normalized_entropy=0.6, trend="STABLE",
                drift_detected=False, drop_rate=0.0,
            ),
        }

    def _full_collapse_reports(self) -> dict:
        expert_states = {
            0: ExpertState(expert_id=0, status=ExpertStatus.HEALTHY, utilization=0.4),
            1: ExpertState(expert_id=1, status=ExpertStatus.COLD, utilization=0.01),
            2: ExpertState(expert_id=2, status=ExpertStatus.DEAD, utilization=0.0),
        }
        return {
            "layers.0.gate": LayerCollapseReport(
                layer_name="layers.0.gate",
                expert_states=expert_states,
                num_dead_experts=1,
                num_cold_experts=1,
                num_healthy_experts=1,
                load_imbalance_ratio=6.5,
            ),
            "layers.1.gate": LayerCollapseReport(
                layer_name="layers.1.gate",
                expert_states={0: ExpertState(expert_id=0, status=ExpertStatus.HEALTHY, utilization=0.5)},
                num_dead_experts=0,
                num_cold_experts=0,
                num_healthy_experts=1,
                load_imbalance_ratio=2.0,
            ),
        }

    # ---- Rich path ------------------------------------------------

    def test_full_dashboard_rich(self, capsys) -> None:
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
        assert "MoEWatch" in result
        assert "v0.2.0" in result
        assert "0.gate" in result
        assert "Interventions" in result
        assert "Policy Decisions" in result
        assert "Entropy Radar" in result
        assert "Expert Health" in result

        captured = capsys.readouterr()
        assert "MoEWatch" in captured.out

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
        assert isinstance(result, str) and len(result) > 0

    def test_dashboard_rich_no_interventions_no_policy(self) -> None:
        """Exercise the 'No interventions applied' / 'No policy decisions' branches."""
        config = _config()
        reporter = CLIReporter(config)
        sr = StepReport(
            step=1,
            timestamp=1700000000.0,
            risk_scores={"layers.0.gate": 0.4},
            risk_levels={"layers.0.gate": RiskLevel.MID.value},
        )
        watch_report = _watch_report_with_step(sr)
        entropy_analyzer, collapse_detector, risk_fuser = _mock_analyzers(
            risk_reports={
                "layers.0.gate": RiskReport(
                    layer_name="layers.0.gate",
                    risk_score=0.4,
                    risk_level=RiskLevel.MID,
                )
            }
        )

        result = reporter.render_dashboard(
            watch_report, entropy_analyzer, collapse_detector, risk_fuser
        )
        assert "No interventions" in result
        assert "No policy decisions" in result

    def test_dashboard_rich_multi_step_history_sparkline(self) -> None:
        """Multiple steps build up risk score history feeding sparklines."""
        config = _config()
        reporter = CLIReporter(config)
        watch_report = WatchReport(max_steps=20)
        for i, score in enumerate([0.1, 0.3, 0.5, 0.7, 0.85], start=1):
            sr = StepReport(
                step=i,
                timestamp=1700000000.0 + i,
                risk_scores={"layers.0.gate": score},
                risk_levels={"layers.0.gate": RiskLevel.HIGH.value},
            )
            watch_report.append(sr)

        entropy_analyzer, collapse_detector, risk_fuser = _mock_analyzers(
            risk_reports={
                "layers.0.gate": RiskReport(
                    layer_name="layers.0.gate",
                    risk_score=0.85,
                    risk_level=RiskLevel.HIGH,
                    dominant_signal="cross_layer",
                )
            }
        )

        result = reporter.render_dashboard(
            watch_report, entropy_analyzer, collapse_detector, risk_fuser
        )
        assert "Layer Risk Map" in result

    # ---- Plain (no Rich) path --------------------------------------

    def test_full_dashboard_plain(self, monkeypatch, capsys) -> None:
        _patch_no_rich(monkeypatch)
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
        assert "Interventions" in result
        assert "Policy Decisions" in result
        assert "Entropy Radar" in result
        assert "Expert Health" in result
        assert "Layer Risk Map" in result
        assert "loss=1.23456" in result

        captured = capsys.readouterr()
        assert "MoEWatch" in captured.out

    def test_full_dashboard_plain_no_color(self, monkeypatch) -> None:
        _patch_no_rich(monkeypatch)
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
        assert "\033[" not in result

    def test_plain_dashboard_with_nan_loss_and_no_extras(self, monkeypatch) -> None:
        _patch_no_rich(monkeypatch)
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
        assert "Entropy Radar" not in result
        assert "Interventions" not in result
        assert "Policy Decisions" not in result

    def test_plain_dashboard_unknown_risk_level_uses_default_color(self, monkeypatch) -> None:
        _patch_no_rich(monkeypatch)
        config = _config()
        reporter = CLIReporter(config)
        sr = StepReport(
            step=1,
            timestamp=1700000000.0,
            risk_scores={"layers.0.gate": 0.5},
            risk_levels={},
        )
        watch_report = _watch_report_with_step(sr)
        entropy_analyzer, collapse_detector, risk_fuser = _mock_analyzers()

        result = reporter.render_dashboard(
            watch_report, entropy_analyzer, collapse_detector, risk_fuser
        )
        assert "0.gate" in result

    def test_plain_dashboard_with_collapse_no_entropy(self, monkeypatch) -> None:
        _patch_no_rich(monkeypatch)
        config = _config()
        reporter = CLIReporter(config)
        sr = StepReport(
            step=1,
            timestamp=1700000000.0,
            risk_scores={"layers.0.gate": 0.9},
            risk_levels={"layers.0.gate": RiskLevel.CRITICAL.value},
        )
        watch_report = _watch_report_with_step(sr)
        entropy_analyzer, collapse_detector, risk_fuser = _mock_analyzers(
            collapse_reports=self._full_collapse_reports(),
        )

        result = reporter.render_dashboard(
            watch_report, entropy_analyzer, collapse_detector, risk_fuser
        )
        assert "Expert Health" in result
        assert "Entropy Radar" not in result


# ===========================================================================
# 4. render_alert
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
        assert "0.gate" in result
        assert "something happened" in result
        assert level.value.upper() in result

    @pytest.mark.parametrize(
        "level",
        [AlertLevel.INFO, AlertLevel.WARNING, AlertLevel.CRITICAL],
    )
    def test_render_alert_plain_all_levels(self, monkeypatch, level) -> None:
        _patch_no_rich(monkeypatch)
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
        assert "1.gate" in result
        assert "plain alert message" in result
        assert "step=7" in result

    def test_render_alert_plain_no_color(self, monkeypatch) -> None:
        _patch_no_rich(monkeypatch)
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
# 5. Module-private helper functions
# ===========================================================================


class TestAsciiRiskBar:
    def test_zero_score(self) -> None:
        bar = _ascii_risk_bar(0.0, width=10)
        assert bar == "." * 10

    def test_full_score(self) -> None:
        bar = _ascii_risk_bar(1.0, width=10)
        assert bar == "#" * 10

    def test_partial_score(self) -> None:
        bar = _ascii_risk_bar(0.5, width=10)
        assert bar.count("#") == 5
        assert bar.count(".") == 5

    def test_clamps_out_of_range(self) -> None:
        assert _ascii_risk_bar(-1.0, width=4) == "." * 4
        assert _ascii_risk_bar(2.0, width=4) == "#" * 4


class TestExpertUtilBar:
    def test_empty_returns_no_data(self) -> None:
        assert _expert_util_bar({}, width=30) == "(no data)"

    def test_single_expert_full_bar(self) -> None:
        result = _expert_util_bar({0: 1.0}, width=10)
        assert "|" not in result
        assert "#" in result

    def test_multiple_experts_separated(self) -> None:
        result = _expert_util_bar({0: 1.0, 1: 0.5, 2: 0.0}, width=12)
        parts = result.split("|")
        assert len(parts) == 3

    def test_zero_max_util_handled(self) -> None:
        result = _expert_util_bar({0: 0.0, 1: 0.0}, width=8)
        assert "|" in result
        assert "#" not in result


class TestSparkline:
    def test_empty_returns_empty_string(self) -> None:
        assert _sparkline([]) == ""

    def test_single_value_returns_empty_string(self) -> None:
        assert _sparkline([0.5]) == ""

    def test_multiple_values_returns_chars_per_value(self) -> None:
        result = _sparkline([0.1, 0.5, 0.9])
        assert len(result) == 3

    def test_constant_values_handled(self) -> None:
        """All-equal values should not raise (span == 0 branch)."""
        result = _sparkline([0.5, 0.5, 0.5])
        assert len(result) == 3


class TestEntropyTrendArrow:
    @pytest.mark.parametrize(
        "trend,expected",
        [
            ("RISING", "UP"),
            ("rising", "UP"),
            ("IMPROVING", "UP"),
            ("DECLINING", "DN"),
            ("DROPPING", "DN"),
            ("STABLE", "--"),
            ("UNKNOWN", "??"),
            ("something_else", "??"),
        ],
    )
    def test_known_and_unknown_trends(self, trend, expected) -> None:
        assert _entropy_trend_arrow(trend) == expected

    def test_non_string_trend(self) -> None:
        assert _entropy_trend_arrow(None) == "??"
        assert _entropy_trend_arrow(123) == "??"


class TestShortLayerName:
    def test_none_returns_dash(self) -> None:
        assert _short_layer_name(None) == "-"

    def test_short_name_unchanged(self) -> None:
        assert _short_layer_name("router") == "router"

    def test_long_dotted_path_keeps_last_two_segments(self) -> None:
        assert _short_layer_name("model.layers.5.mlp.moe.router") == "moe.router"

    def test_truncates_overly_long_result(self) -> None:
        long_name = "a" * 10 + "." + "b" * 40
        result = _short_layer_name(long_name, max_len=20)
        assert len(result) == 20
        assert result.endswith(".")
