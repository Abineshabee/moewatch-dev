"""
Step 1 — Multi-step live dashboard simulation.

Feeds several StepReports into WatchReport so the Layer Risk Map's
sparkline / trend history has real data to render.
"""
import time
from moewatch.config import WatchConfig
from moewatch.report.watch_report import WatchReport, StepReport
from moewatch.report.cli_reporter import CLIReporter
from moewatch.analyzer.entropy import EntropyAnalyzer
from moewatch.analyzer.collapse import CollapseDetector
from moewatch.analyzer.risk_score import RiskScoreFuser

config = WatchConfig()
watch_report = WatchReport()
entropy_analyzer = EntropyAnalyzer(config)
collapse_detector = CollapseDetector(config)
risk_fuser = RiskScoreFuser(config)
reporter = CLIReporter(config)

# Simulate a rising-risk trend on layer 5, stable on layer 9
scores_5 = [0.10, 0.25, 0.40, 0.55, 0.72, 0.80]
scores_9 = [0.20, 0.21, 0.19, 0.22, 0.21, 0.20]

for i, (s5, s9) in enumerate(zip(scores_5, scores_9), start=1):
    step = StepReport(
        step=i,
        timestamp=time.time(),
        risk_scores={"layers.5.moe": s5, "layers.9.moe": s9},
        risk_levels={
            "layers.5.moe": "CRITICAL" if s5 >= 0.8 else ("HIGH" if s5 >= 0.6 else ("MID" if s5 >= 0.3 else "LOW")),
            "layers.9.moe": "LOW",
        },
        loss=2.5 - i * 0.05,
    )
    watch_report.steps.append(step)

output = reporter.render_dashboard(
    watch_report, entropy_analyzer, collapse_detector, risk_fuser
)
