# ============================================================
#  MoEWatch — examples/03_intervention_deep_dive.py
#  Show all three intervention tiers firing under deliberate
#  collapse pressure and inspect every intervention event.
# ============================================================

"""
03_intervention_deep_dive.py
============================
Demonstrates the three intervention tiers MoEWatch can apply:

  Tier 1 — AuxLossAction      (risk 0.30–0.60)
            Raises router_aux_loss_coef, adding a load-balancing
            penalty to the cross-entropy loss. The training loop
            must include this penalty (see below).

  Tier 2 — RouterNoiseAction  (risk 0.60–0.80)
            Injects Gaussian noise into gate logits via a forward
            hook — forces the softmax to spread tokens more evenly.
            No change to model weights.

  Tier 3 — ExpertDropoutAction (risk >= 0.80)
            Raises nn.Dropout.p on the dominant expert, forcing
            the model to rely on other experts.

How collapse pressure is injected
----------------------------------
We directly SET the gate bias toward one expert (increasing by
0.01 per step) to simulate a sustained input-distribution shift.
This is the same mechanism used in the comparison demo.

What to look for in the output
--------------------------------
  • risk_score climbs as bias accumulates
  • Intervention tier escalates: aux_loss → router_noise → expert_dropout
  • router_aux_loss_coef grows as AuxLossAction accumulates deltas
  • After pressure ends, entropy recovers

Run:
    python examples/03_intervention_deep_dive.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import MagicMock

from moewatch import MoEWatch, WatchConfig, OutputMode


# ---------------------------------------------------------------------------
# Model with Dropout in experts (required for Tier 3 ExpertDropoutAction)
# ---------------------------------------------------------------------------

class Expert(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.up      = nn.Linear(dim, dim * 4)
        self.dropout = nn.Dropout(p=0.0)   # ExpertDropoutAction target
        self.down    = nn.Linear(dim * 4, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.dropout(F.gelu(self.up(x))))


class MoELayer(nn.Module):
    def __init__(self, dim: int, n_experts: int = 4) -> None:
        super().__init__()
        self.gate    = nn.Linear(dim, n_experts, bias=True)
        self.experts = nn.ModuleList([Expert(dim) for _ in range(n_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        flat    = x.reshape(-1, D)
        logits  = self.gate(flat)
        probs   = torch.softmax(logits, dim=-1)
        top_i   = probs.argmax(-1)
        # Switch-Transformer aux loss (stored for training loop to use)
        with torch.no_grad():
            f_e = F.one_hot(top_i, len(self.experts)).float().mean(0)
        P_e = probs.mean(0)
        self.last_aux_loss = len(self.experts) * (f_e * P_e).sum()
        out = torch.zeros_like(flat)
        for e in range(len(self.experts)):
            mask = top_i == e
            if mask.any():
                out[mask] = self.experts[e](flat[mask])
        return out.reshape(B, S, D)


class TinyMoEModel(nn.Module):
    def __init__(self, vocab: int = 64, dim: int = 32) -> None:
        super().__init__()
        self.embed  = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList([MoELayer(dim) for _ in range(2)])
        self.head   = nn.Linear(dim, vocab, bias=False)
        # AuxLossAction writes to this field
        self.config = MagicMock()
        self.config.router_aux_loss_coef = 0.0

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        aux = 0.0
        for layer in self.layers:
            x   = x + layer(x)
            aux = aux + layer.last_aux_loss
        self.last_aux = aux
        return self.head(x)

    def inject_bias(self, layer_idx: int, expert_idx: int, strength: float) -> None:
        """Directly set the gate bias to simulate a distribution shift."""
        with torch.no_grad():
            self.layers[layer_idx].gate.bias[expert_idx] = strength


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    torch.manual_seed(0)
    VOCAB, DIM, STEPS = 64, 32, 300

    model     = TinyMoEModel(VOCAB, DIM)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    config = WatchConfig(
        output=OutputMode.SILENT,

        # Thresholds calibrated for HIDDEN_DIM=32
        entropy_warn=0.65,
        entropy_critical=0.40,
        entropy_drop_warn=0.06,
        dead_threshold=0.05,
        cold_threshold=0.15,
        cold_steps_limit=10,

        # Interventions enabled — all three tiers active
        intervention_enabled=True,
        policy_type="rule",
        intervention_cooldown=5,
        intervention_max_delta=1.5,   # allows large noise injection
        loss_guard_threshold=5.0,
        reward_window_steps=200,
        baseline_min_clean_steps=5,
        baseline_exclusion_window=5,

        log_every=10,
        sample_every=1,
    )

    watcher = MoEWatch(model, config)
    watcher.start()

    print("=" * 70)
    print("  Intervention Deep Dive")
    print(f"  {watcher.num_layers_monitored} layers monitored")
    print("=" * 70)
    print(f"  {'step':>5}  {'loss':>7}  {'bias':>6}  {'risk_L0':>8}  "
          f"{'aux_coef':>9}  action")
    print("-" * 70)

    action_log: list[dict] = []

    for step in range(1, STEPS + 1):
        # Ramp bias on layer 0 expert 0 from step 50 to 200
        if 50 <= step <= 200:
            strength = (step - 50) / 150 * 2.0   # 0 → 2.0 linearly
        elif step > 200:
            strength = 2.0                         # held at peak
        else:
            strength = 0.0
        model.inject_bias(layer_idx=0, expert_idx=0, strength=strength)

        watcher.pre_step(step)

        ids    = torch.randint(0, VOCAB, (8, 16))
        logits = model(ids)
        ce     = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1))
        aux_c  = float(model.config.router_aux_loss_coef)
        loss   = ce + aux_c * model.last_aux

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        report = watcher.step(step, ce.item())

        acts = [a.action_type for a in report.active_interventions]
        risk0 = report.risk_scores.get(watcher._layer_order[0], 0.0)

        if step % 20 == 0 or acts:
            print(f"  {step:>5}  {ce.item():>7.4f}  {strength:>6.2f}  "
                  f"{risk0:>8.3f}  {aux_c:>9.4f}  {','.join(acts) or '-'}")

        if acts:
            action_log.append({"step": step, "actions": acts,
                                "risk": risk0, "aux_coef": aux_c})

    watcher.stop()

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  INTERVENTION SUMMARY")
    print("=" * 70)

    all_alerts = watcher.get_alerts()
    by_level: dict[str, int] = {}
    for a in all_alerts:
        by_level[a.level.value] = by_level.get(a.level.value, 0) + 1
    print(f"  Total alerts: {len(all_alerts)}")
    for lvl, cnt in sorted(by_level.items()):
        print(f"    {lvl.upper():<10}: {cnt}")

    action_counts: dict[str, int] = {}
    for entry in action_log:
        for a in entry["actions"]:
            action_counts[a] = action_counts.get(a, 0) + 1
    print(f"\n  Interventions fired: {len(action_log)} events")
    for action, cnt in sorted(action_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {action:<20}: {cnt}")

    print(f"\n  Final router_aux_loss_coef: {model.config.router_aux_loss_coef:.4f}")
    print(f"  (AuxLossAction accumulated {model.config.router_aux_loss_coef:.2f} "
          f"above the initial 0.0)")


if __name__ == "__main__":
    main()
