"""
Step 3 — Interventions, Policy Decisions, and Alert rendering.

Exercises:
  - active_interventions populated (AuxLossAction / RouterNoiseAction)
  - policy_decisions populated
  - render_alert() for INFO / WARNING / CRITICAL levels
"""
import time
from moewatch import Alert
from moewatch.config import WatchConfig, AlertLevel
from moewatch.report.watch_report import WatchReport, StepReport
from moewatch.report.cli_reporter import CLIReporter
from moewatch.analyzer.entropy import EntropyAnalyzer
from moewatch.analyzer.collapse import CollapseDetector
from moewatch.analyzer.risk_score import RiskScoreFuser
from moewatch.intervention.actions import AuxLossAction, RouterNoiseAction, NoOpAction

config = WatchConfig()
watch_report = WatchReport()
entropy_analyzer = EntropyAnalyzer(config)
collapse_detector = CollapseDetector(config)
risk_fuser = RiskScoreFuser(config)
reporter = CLIReporter(config)

# -- Build interventions -------------------------------------------------
aux_action = AuxLossAction(layer_name="layers.4.moe.router", delta=0.05)
aux_action.mark_applied(12)

noise_action = RouterNoiseAction(layer_name="layers.4.moe.router", noise_scale=0.02)
noise_action.mark_applied(12)

noop_action = NoOpAction(layer_name="layers.8.moe.router")

step = StepReport(
    step=12,
    timestamp=time.time(),
    risk_scores={"layers.4.moe.router": 0.91, "layers.8.moe.router": 0.12},
    risk_levels={"layers.4.moe.router": "CRITICAL", "layers.8.moe.router": "LOW"},
    active_interventions=[aux_action, noise_action, noop_action],
    policy_decisions={
        "layers.4.moe.router": "aux_loss_bump",
        "layers.8.moe.router": "noop",
    },
    alerts=[
        Alert(
            step=12, level=AlertLevel.CRITICAL,
            layer_id="layers.4.moe.router", signal_type="entropy",
            message="entropy collapse detected, intervention applied",
        ),
    ],
    loss=3.1415,
)
watch_report.steps.append(step)

print("===== DASHBOARD =====")
reporter.render_dashboard(watch_report, entropy_analyzer, collapse_detector, risk_fuser)

print("\n===== ALERTS =====")
reporter.render_alert(Alert(
    step=12, level=AlertLevel.INFO, layer_id="layers.1.moe.router",
    signal_type="gradient", message="all signals nominal",
))
reporter.render_alert(Alert(
    step=12, level=AlertLevel.WARNING, layer_id="layers.4.moe.router",
    signal_type="load_imbalance", message="expert load imbalance rising",
))
reporter.render_alert(Alert(
    step=12, level=AlertLevel.CRITICAL, layer_id="layers.4.moe.router",
    signal_type="entropy", message="entropy collapse detected, intervention applied",
))
