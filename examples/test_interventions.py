import torch, torch.nn as nn
from moewatch import MoEWatch
from moewatch.config import WatchConfig, OutputMode

# Minimal config object that AuxLossAction._resolve_config() can find
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
        self.config = FakeConfig()          # <-- AuxLossAction needs this
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
    entropy_warn=0.6,
    entropy_critical=0.3,
    intervention_cooldown=2,
    intervention_max_delta=0.5,
)

model = FakeModel()
watcher = MoEWatch(model, config)
watcher.start()

print(f"intervention_engine: {watcher.intervention_engine}")
print(f"policy: {watcher.policy}")
print()
print("Step | Pressure | Worst Risk | Alerts | Interventions | Action types       | aux_loss_coef")
print("-" * 90)

for step in range(1, 16):
    watcher.pre_step(step)
    pressure = step * 0.4
    model.set_collapse_pressure(pressure)

    x = torch.randn(32, 16)
    out = model(x)
    loss = out.mean()
    loss.backward()

    report = watcher.step(global_step=step, current_loss=loss.item())

    worst_score = max(report.risk_scores.values(), default=0.0)
    n_alerts = len(report.alerts)
    n_interventions = len(report.active_interventions)
    action_types = [a.action_type for a in report.active_interventions]
    coef = model.config.router_aux_loss_coef

    print(f"  {step:3d}  | {pressure:.1f}      | {worst_score:.4f}     | {n_alerts}      | {n_interventions}              | {str(action_types):<20} | {coef:.4f}")
