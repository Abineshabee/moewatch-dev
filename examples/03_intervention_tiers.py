# ============================================================
#  MoEWatch — examples/03_intervention_tiers.py
#  Three intervention tiers: what fires, what changes.
# ============================================================
"""
MoEWatch three intervention tiers explained with working code.

  Tier 1 — AuxLossAction      (risk 0.30–0.60)
  Tier 2 — RouterNoiseAction  (risk 0.60–0.80)
  Tier 3 — ExpertDropoutAction (risk >= 0.80)

Instead of relying on the automatic policy (which has a cascade
guard that limits repeated actions), this example shows each
action applied directly so you can see its exact side-effect:

  AuxLossAction      → model.config.router_aux_loss_coef grows
  RouterNoiseAction  → forward hook added to gate modules
  ExpertDropoutAction → expert.dropout.p raises above 0

Then we run a real monitored loop and print the WatchReport
summary, alerts by severity, and final risk-per-layer table.

Run:
    python examples/03_intervention_tiers.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import MagicMock

from moewatch import MoEWatch, WatchConfig, OutputMode, AlertLevel
from moewatch.intervention.actions import (
    AuxLossAction,
    RouterNoiseAction,
    ExpertDropoutAction,
)

# ── Model ───────────────────────────────────────────────────

class Expert(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.up      = nn.Linear(dim, dim * 4)
        self.dropout = nn.Dropout(p=0.0)
        self.down    = nn.Linear(dim * 4, dim)
    def forward(self, x):
        return self.down(self.dropout(F.gelu(self.up(x))))


class MoELayer(nn.Module):
    def __init__(self, dim: int, n: int = 4) -> None:
        super().__init__()
        self.gate    = nn.Linear(dim, n, bias=True)
        self.experts = nn.ModuleList([Expert(dim) for _ in range(n)])
    def forward(self, x):
        B, S, D = x.shape; flat = x.reshape(-1, D)
        probs = torch.softmax(self.gate(flat), -1); top_i = probs.argmax(-1)
        with torch.no_grad():
            f_e = F.one_hot(top_i, len(self.experts)).float().mean(0)
        self.last_aux = len(self.experts) * (f_e * probs.mean(0)).sum()
        out = torch.zeros_like(flat)
        for e in range(len(self.experts)):
            mask = top_i == e
            if mask.any(): out[mask] = self.experts[e](flat[mask])
        return out.reshape(B, S, D)


class TinyMoE(nn.Module):
    def __init__(self, vocab=64, dim=32):
        super().__init__()
        self.embed  = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList([MoELayer(dim) for _ in range(3)])
        self.head   = nn.Linear(dim, vocab, bias=False)
        self.config = MagicMock()
        self.config.router_aux_loss_coef = 0.0
    def forward(self, ids):
        x = self.embed(ids); self._aux = 0.0
        for l in self.layers:
            x = x + l(x); self._aux += l.last_aux
        return self.head(x)
    def inject_bias(self, li, ei, s):
        with torch.no_grad(): self.layers[li].gate.bias[ei] = s


# ── Part A: Apply each action directly ──────────────────────

def demo_actions() -> None:
    print("=" * 58)
    print("  Part A — Direct action API (what each tier does)")
    print("=" * 58)

    model = TinyMoE()
    layer_name = "layers.0.gate"   # first detected router

    # Tier 1: AuxLossAction
    action1 = AuxLossAction(layer_name=layer_name, delta=0.5)
    coef_before = float(model.config.router_aux_loss_coef)
    action1.apply(model)
    coef_after = float(model.config.router_aux_loss_coef)
    print(f"\n  Tier 1 — AuxLossAction(delta=0.5)")
    print(f"    router_aux_loss_coef: {coef_before:.2f} → {coef_after:.2f}")
    print(f"    Effect: training loop uses  loss += {coef_after} * aux_loss")

    # Tier 2: RouterNoiseAction
    hooks_before = len(model.layers[0].gate._forward_hooks)
    action2 = RouterNoiseAction(layer_name=layer_name, noise_scale=0.3)
    action2.apply(model)
    hooks_after = len(model.layers[0].gate._forward_hooks)
    print(f"\n  Tier 2 — RouterNoiseAction(noise_scale=0.3)")
    print(f"    Gate forward hooks: {hooks_before} → {hooks_after}")
    print(f"    Effect: Gaussian noise N(0, 0.3) added to gate logits each step")

    # Tier 3: ExpertDropoutAction
    drop_before = model.layers[0].experts[0].dropout.p
    action3 = ExpertDropoutAction(layer_name=layer_name, dropout_delta=0.2)
    action3.apply(model)
    drop_after = max(e.dropout.p for l in model.layers for e in l.experts)
    print(f"\n  Tier 3 — ExpertDropoutAction(dropout_delta=0.2)")
    print(f"    max expert dropout.p: {drop_before:.2f} → {drop_after:.2f}")
    print(f"    Effect: dominant expert's dropout raised, forcing reliance on others")


# ── Part B: Monitored loop with alert summary ────────────────

def demo_monitored_loop() -> None:
    print("\n" + "=" * 58)
    print("  Part B — Monitored training loop with risk tracking")
    print("=" * 58)

    torch.manual_seed(0)
    VOCAB, DIM, STEPS = 64, 32, 200
    model     = TinyMoE(VOCAB, DIM)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    config = WatchConfig(
        output=OutputMode.SILENT,
        entropy_warn=0.65,
        entropy_critical=0.40,
        entropy_drop_warn=0.06,
        dead_threshold=0.05,
        cold_threshold=0.15,
        cold_steps_limit=10,
        intervention_enabled=True,
        policy_type="rule",
        intervention_cooldown=80,
        intervention_max_delta=1.5,
        loss_guard_threshold=5.0,
        reward_window_steps=300,
        baseline_min_clean_steps=5,
        baseline_exclusion_window=5,
        log_every=10,
        sample_every=1,
    )
    watcher = MoEWatch(model, config)
    watcher.start()

    for step in range(1, STEPS + 1):
        # Collapse pressure on layer 2 from step 50
        s = min((step - 50) / 100 * 2.2, 2.2) if step >= 50 else 0.0
        model.inject_bias(2, 1, s)

        watcher.pre_step(step)
        ids    = torch.randint(0, VOCAB, (8, 16))
        logits = model(ids)
        ce     = F.cross_entropy(
            logits[:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1)
        )
        aux_c = float(model.config.router_aux_loss_coef)
        loss  = ce + aux_c * model._aux
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        watcher.step(step, ce.item())

    watcher.stop()

    # Print a compact summary — no raw alert spam
    all_alerts = watcher.get_alerts()
    by_level: dict[str, int] = {}
    by_signal: dict[str, int] = {}
    for a in all_alerts:
        by_level[a.level.value]   = by_level.get(a.level.value, 0) + 1
        by_signal[a.signal_type]  = by_signal.get(a.signal_type, 0) + 1

    print(f"\n  Training done. {STEPS} steps, {len(all_alerts)} total alerts.\n")
    print("  Alerts by severity:")
    for lvl in ["critical", "error", "warning", "info"]:
        cnt = by_level.get(lvl, 0)
        if cnt:
            print(f"    {lvl.upper():<10}: {cnt}")
    print("\n  Alerts by signal type (what triggered them):")
    for sig, cnt in sorted(by_signal.items(), key=lambda kv: -kv[1]):
        print(f"    {sig:<30}: {cnt}")

    print(f"\n  Final risk per layer:")
    for name, score in watcher.get_risk_summary().items():
        short = name.split(".")[-2]
        bar   = "#" * int(score * 18) + "." * (18 - int(score * 18))
        print(f"    {short}  [{bar}]  {score:.3f}")

    print(f"\n  router_aux_loss_coef (Tier 1 side-effect): "
          f"{model.config.router_aux_loss_coef:.3f}")
    print(f"  WatchReport summary:\n")
    print(watcher.watch_report.summary())


# ── Run both parts ───────────────────────────────────────────

if __name__ == "__main__":
    demo_actions()
    demo_monitored_loop()
