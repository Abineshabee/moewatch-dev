"""
Test 4 — BanditPolicy (Phase 2 learning) end-to-end smoke test.

Verifies BanditPolicy can be constructed via policy_type="bandit",
selects actions across an escalating-risk run, and that policy.update()
is invoked through the normal observation-window resolution path
(check via intervention log "resolved" entries).
"""
import torch, torch.nn as nn
from moewatch import MoEWatch
from moewatch.config import WatchConfig, OutputMode


class FakeConfig:
    def __init__(self):
        self.router_aux_loss_coef = 0.01


class FakeMoELayer(nn.Module):
    def __init__(self, num_experts=8):
        super().__init__()
        self.router = nn.Module()
        self.router.gate = nn.Linear(16, num_experts, bias=True)

    def forward(self, x):
        logits = self.router.gate(x)
        probs = torch.softmax(logits, dim=-1)
        return x[:, :1] * probs.sum(-1, keepdim=True)


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = FakeConfig()
        self.layers = nn.ModuleList([FakeMoELayer() for _ in range(3)])

    def set_collapse_pressure(self, pressure: float):
        with torch.no_grad():
            for layer in self.layers:
                gate = layer.router.gate
                gate.bias.zero_()
                gate.bias[0] = pressure

    def forward(self, x):
        out = x[:, :1]
        for layer in self.layers:
            out = layer(x)
        return out


config = WatchConfig(
    log_every=1,
    output=OutputMode.SILENT,
    intervention_enabled=True,
    policy_type="bandit",
    entropy_warn=0.6,
    entropy_critical=0.3,
    intervention_cooldown=2,
    intervention_max_delta=0.5,
    reward_window_steps=4,
    baseline_min_clean_steps=2,
    bandit_epsilon=0.2,
)

model = FakeModel()
watcher = MoEWatch(model, config)
watcher.start()
watcher.intervention_engine.safety_guard.update_baseline_loss(1.0)

print(f"policy class: {type(watcher.policy).__name__}")
print()
print("Step | Pressure | Risk   | Interventions | Action types       | aux_loss_coef")
print("-" * 80)

torch.manual_seed(0)

for step in range(1, 61):
    watcher.pre_step(step)
    pressure = min(step * 0.25, 8.0)
    model.set_collapse_pressure(pressure)

    x = torch.randn(32, 16)
    out = model(x)
    loss = out.mean()
    loss.backward()

    report = watcher.step(global_step=step, current_loss=loss.item())
    worst_score = max(report.risk_scores.values(), default=0.0)
    n_interventions = len(report.active_interventions)
    action_types = [a.action_type for a in report.active_interventions]
    coef = model.config.router_aux_loss_coef

    if step % 5 == 0 or n_interventions:
        print(f"  {step:3d}  | {pressure:.2f}     | {worst_score:.4f} | {n_interventions}              "
              f"| {str(action_types):<18} | {coef:.4f}")

print("\n--- Resolved windows (BanditPolicy.update calls) ---")
for entry in watcher.intervention_engine.get_intervention_log():
    if entry.get("event") == "resolved":
        print(entry)
