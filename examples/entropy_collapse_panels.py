"""
Step 2 — Entropy Radar + Expert Health panels.

CLIReporter reads `entropy_analyzer.last_reports` and
`collapse_detector.last_reports` via hasattr(). Real analyzers don't
populate these unless `.analyze(stat_collector)` is called (which needs
torch tensors). For a lightweight CLI-only test, attach the report dicts
directly -- this exercises exactly what render_dashboard reads.
"""
import time
from moewatch.config import WatchConfig
from moewatch.report.watch_report import WatchReport, StepReport
from moewatch.report.cli_reporter import CLIReporter
from moewatch.analyzer.entropy import EntropyAnalyzer, LayerEntropyReport
from moewatch.analyzer.collapse import (
    CollapseDetector,
    LayerCollapseReport,
    ExpertState,
    ExpertStatus,
)
from moewatch.analyzer.risk_score import RiskScoreFuser

config = WatchConfig()
watch_report = WatchReport()
entropy_analyzer = EntropyAnalyzer(config)
collapse_detector = CollapseDetector(config)
risk_fuser = RiskScoreFuser(config)
reporter = CLIReporter(config)

step = StepReport(
    step=1,
    timestamp=time.time(),
    risk_scores={"layers.3.moe": 0.78, "layers.7.moe": 0.15},
    risk_levels={"layers.3.moe": "HIGH", "layers.7.moe": "LOW"},
    loss=1.9876,
)
watch_report.steps.append(step)

# -- Entropy Radar data ------------------------------------------------
entropy_analyzer.last_reports = {
    "layers.3.moe": LayerEntropyReport(
        layer_name="layers.3.moe",
        normalized_entropy=0.18,
        trend="DROPPING",
        drift_detected=True,
        drop_rate=-0.04,
    ),
    "layers.7.moe": LayerEntropyReport(
        layer_name="layers.7.moe",
        normalized_entropy=0.82,
        trend="STABLE",
        drift_detected=False,
        drop_rate=0.0,
    ),
}

# -- Expert Health data -------------------------------------------------
expert_states_3 = {
    0: ExpertState(expert_id=0, status=ExpertStatus.HEALTHY, utilization=0.35),
    1: ExpertState(expert_id=1, status=ExpertStatus.COLD, utilization=0.02),
    2: ExpertState(expert_id=2, status=ExpertStatus.DEAD, utilization=0.0),
    3: ExpertState(expert_id=3, status=ExpertStatus.HEALTHY, utilization=0.63),
}
expert_states_7 = {
    0: ExpertState(expert_id=0, status=ExpertStatus.HEALTHY, utilization=0.5),
    1: ExpertState(expert_id=1, status=ExpertStatus.HEALTHY, utilization=0.5),
}

collapse_detector.last_reports = {
    "layers.3.moe": LayerCollapseReport(
        layer_name="layers.3.moe",
        expert_states=expert_states_3,
        num_dead_experts=1,
        num_cold_experts=1,
        num_healthy_experts=2,
        load_imbalance_ratio=7.1,
    ),
    "layers.7.moe": LayerCollapseReport(
        layer_name="layers.7.moe",
        expert_states=expert_states_7,
        num_dead_experts=0,
        num_cold_experts=0,
        num_healthy_experts=2,
        load_imbalance_ratio=1.0,
    ),
}

output = reporter.render_dashboard(
    watch_report, entropy_analyzer, collapse_detector, risk_fuser
)
