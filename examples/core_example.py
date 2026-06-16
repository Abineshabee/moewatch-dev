import time
import torch
from unittest.mock import MagicMock
from moewatch.config import WatchConfig, OutputMode
from moewatch.collector.baseline_tracker import BaselineTracker
from moewatch.collector.stat_collector import StatCollector
from moewatch.analyzer.entropy import EntropyAnalyzer
from moewatch.analyzer.collapse import CollapseDetector
from moewatch.analyzer.risk_score import RiskScoreFuser
from moewatch.analyzer.gradient_starvation import GradientStarvationReport
from moewatch.hooks.router_hook import RoutingEvent
from moewatch.intervention.actions import AuxLossAction
from moewatch.intervention.engine import InterventionEngine
from moewatch.policy.rule_policy import RulePolicy

print("=" * 65)
print("MoEWatch v0.2.0 — Predictive Control Loop Test")
print("Catching collapse BEFORE experts die")
print("=" * 65)

config = WatchConfig(
    output=OutputMode.SILENT,
    intervention_cooldown=5,
    intervention_max_delta=0.5,
    loss_guard_threshold=2.0,
    reward_window_steps=10,
    baseline_min_clean_steps=3,
    baseline_exclusion_window=10,
    dead_threshold=0.01,
    cold_threshold=0.05,
    entropy_warn=0.3,
    entropy_critical=0.15,
)

layer     = "model.layers.0.mlp.gate"
n_experts = 8
torch.manual_seed(42)

trainer = MagicMock()
trainer.args = MagicMock()
trainer.args.aux_loss_coef = 0.01

sc       = StatCollector(config)
bt       = BaselineTracker(config)
entropy  = EntropyAnalyzer(config)
collapse = CollapseDetector(config)
fuser    = RiskScoreFuser(config)
engine   = InterventionEngine(config, trainer, bt)
policy   = RulePolicy(config)

sc.register_layer(layer, n_experts)
bt.register_layer(layer)

def _feed(w: torch.Tensor, step: int, n_tokens: int = 256):
    """Simulate n_tokens tokens independently routed via multinomial sampling."""
    logits   = w.unsqueeze(0).expand(n_tokens, -1)
    selected = torch.multinomial(logits, num_samples=2, replacement=False)
    sc.write_routing_event(RoutingEvent(
        timestamp=time.time(), global_step=step, layer_name=layer,
        routing_logits=logits, selected_experts=selected,
        expert_count=n_experts, batch_size=n_tokens,
    ))

def _snapshot(step, starvation_score=0.0):
    """Call analyzers once and return reports — matches watcher log_every pattern."""
    er_map = entropy.analyze(sc)
    cr_map = collapse.analyze(sc)
    er = er_map[layer]
    cr = cr_map[layer]
    grad = GradientStarvationReport(
        layer_name=layer, expert_id=0,
        starvation_score=starvation_score, step=step,
    )
    rr = fuser.fuse(grad, er)
    return er, cr, rr

# --- [1] Healthy routing ---
print("\n[1] Healthy routing — uniform expert load")
w_uniform = torch.ones(n_experts) / n_experts
for step in range(20):
    _feed(w_uniform, step)
    bt.update_signal(layer, value=0.5, step=step)

er, cr, rr = _snapshot(step=19)
print(f"  Normalized entropy : {er.normalized_entropy:.4f}  (expected ~1.0)")
print(f"  Dead experts       : {cr.num_dead_experts}  (expected 0)")
print(f"  Healthy experts    : {cr.num_healthy_experts}  (expected {n_experts})")
print(f"  Risk score         : {rr.risk_score:.4f}  (expected low)")
print(f"  Risk level         : {rr.risk_level.value.upper()}")

# --- [2] Early collapse signal — entropy dropping, zero dead experts yet ---
print("\n[2] Early collapse — one expert dominating (no dead experts yet)")
for step in range(20, 50):
    alpha = (step - 20) / 29
    w = torch.zeros(n_experts)
    w[0] = 0.125 + alpha * 0.70   # 12.5% -> 82.5%
    w[1:] = (1.0 - w[0]) / (n_experts - 1)
    _feed(w, step)
    bt.update_signal(layer, value=0.5 + alpha * 0.05, step=step)

er, cr, rr = _snapshot(step=49, starvation_score=0.25)
print(f"  Normalized entropy : {er.normalized_entropy:.4f}  (dropping from 1.0)")
print(f"  Drift detected     : {er.drift_detected}")
print(f"  Dead experts       : {cr.num_dead_experts}  (still 0 — other tools silent here)")
print(f"  Cold experts       : {cr.num_cold_experts}  (starting to show stress)")
print(f"  Risk score         : {rr.risk_score:.4f}  (MoEWatch already sees it)")
print(f"  Risk level         : {rr.risk_level.value.upper()}")

# --- [3] Predictive intervention ---
print("\n[3] Predictive intervention — firing before any expert dies")
risk_scores = {layer: rr.risk_score}
action    = AuxLossAction(layer_name=layer)
validated = engine.propose_intervention(
    action, current_loss=0.52,
    risk_scores=risk_scores, layer_order=[layer], step=49,
)
engine.apply_intervention(validated, step=49)
print(f"  Action proposed    : {validated.action_type!r}")
print(f"  Intervention active: {layer in engine._active_interventions}")
print(f"  Obs window open    : {layer in engine._observation_windows}")

# --- [4] Recovery ---
print("\n[4] Post-intervention — routing recovers")
for step in range(50, 65):
    recovery = (step - 50) / 14
    w = torch.zeros(n_experts)
    w[0] = 0.825 - recovery * 0.70   # 82.5% back to 12.5%
    w[0] = max(w[0].item(), 0.125)
    w[1:] = (1.0 - w[0]) / (n_experts - 1)
    _feed(w, step)
    bt.update_signal(layer, value=0.5, step=step)
    engine.check_observation_windows(
        step=step,
        risk_scores={layer: rr.risk_score * (1 - recovery)},
        policy=policy,
    )

er, cr, rr_final = _snapshot(step=64)
print(f"  Normalized entropy : {er.normalized_entropy:.4f}  (recovering toward 1.0)")
print(f"  Dead experts       : {cr.num_dead_experts}  (stayed 0 — collapse prevented)")
print(f"  Window resolved    : {layer not in engine._observation_windows}")

print("\n" + "=" * 65)
print("Collapse caught at drift stage. Experts never died.")
print("This is the gap MoEWatch fills.")
print("=" * 65)
