# ============================================================
#  MoEWatch — examples/01_minimal_loop.py
#  Minimal plain-PyTorch training loop with live monitoring.
# ============================================================
"""
The smallest complete MoEWatch usage — observation mode.

Four steps to add MoEWatch to any training loop:
  1. watcher = MoEWatch(model, config)
  2. watcher.start()
  3. watcher.pre_step(step)   — before the forward pass
  4. report = watcher.step(step, loss)  — after optimizer.step()

This example uses OutputMode.SILENT and prints a compact table
so you can see exactly what MoEWatch tracks: risk score per
layer, dominant signal tier, and CRITICAL alerts.

Collapse pressure is injected artificially (gate bias ramp) to
show the monitoring signal rising in real-time.

Run:
    python examples/01_minimal_loop.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import MagicMock

from moewatch import MoEWatch, WatchConfig, OutputMode, AlertLevel

# ── Tiny MoE model ──────────────────────────────────────────

class Expert(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.up      = nn.Linear(dim, dim * 4)
        self.dropout = nn.Dropout(p=0.0)
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
        flat  = x.reshape(-1, D)
        probs = torch.softmax(self.gate(flat), dim=-1)
        top_i = probs.argmax(-1)
        out   = torch.zeros_like(flat)
        for e in range(len(self.experts)):
            mask = top_i == e
            if mask.any():
                out[mask] = self.experts[e](flat[mask])
        return out.reshape(B, S, D)


class TinyMoE(nn.Module):
    def __init__(self, vocab: int = 64, dim: int = 32) -> None:
        super().__init__()
        self.embed  = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList([MoELayer(dim) for _ in range(3)])
        self.head   = nn.Linear(dim, vocab, bias=False)
        self.config = MagicMock()
        self.config.router_aux_loss_coef = 0.0

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        for layer in self.layers:
            x = x + layer(x)
        return self.head(x)

    def inject_bias(self, layer_idx: int, expert: int, strength: float) -> None:
        with torch.no_grad():
            self.layers[layer_idx].gate.bias[expert] = strength


# ── Main ────────────────────────────────────────────────────

def main() -> None:
    torch.manual_seed(0)
    VOCAB, DIM, STEPS = 64, 32, 300

    model     = TinyMoE(VOCAB, DIM)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    # Observation-only config — no interventions, just watching
    config = WatchConfig(
        output=OutputMode.SILENT,
        entropy_warn=0.65,
        entropy_critical=0.40,
        entropy_drop_warn=0.06,
        dead_threshold=0.05,
        cold_threshold=0.15,
        cold_steps_limit=10,
        intervention_enabled=False,     # observe only
        log_every=10,
        sample_every=1,
    )

    # Step 1-2: create and start
    watcher = MoEWatch(model, config)
    watcher.start()

    print(f"MoEWatch started — {watcher.num_layers_monitored} layers monitored\n")
    print(f"  {'step':>5}  {'loss':>7}  {'bias':>6}  "
          f"{'L0_risk':>8}  {'L1_risk':>8}  {'L2_risk':>8}  new_crits")
    print("  " + "-" * 62)

    prev_alert_count = 0

    for step in range(1, STEPS + 1):
        # Simulate a distribution shift: bias layer 2 toward expert 1
        if 80 <= step <= 220:
            strength = (step - 80) / 140 * 2.2
        elif step > 220:
            strength = 2.2
        else:
            strength = 0.0
        model.inject_bias(layer_idx=2, expert=1, strength=strength)

        # Step 3: signal start of step (before forward pass)
        watcher.pre_step(step)

        ids    = torch.randint(0, VOCAB, (8, 16))
        logits = model(ids)
        loss   = F.cross_entropy(
            logits[:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Step 4: run full analysis pipeline
        report = watcher.step(step, loss.item())

        if step % 30 == 0:
            layers = watcher._layer_order
            r = [report.risk_scores.get(l, 0.0) for l in layers]
            all_alerts    = watcher.get_alerts()
            new_crits     = len([
                a for a in all_alerts
                if a.level == AlertLevel.CRITICAL and a.step == step
            ])
            print(
                f"  {step:>5}  {loss.item():>7.4f}  {strength:>6.2f}  "
                f"{r[0]:>8.3f}  {r[1]:>8.3f}  {r[2]:>8.3f}  {new_crits:>9}"
            )

    watcher.stop()

    # ── Final summary ────────────────────────────────────────
    all_alerts = watcher.get_alerts()
    crits      = [a for a in all_alerts if a.level == AlertLevel.CRITICAL]
    print("\n" + "=" * 50)
    print("  FINAL SUMMARY")
    print("=" * 50)
    print(f"  Total alerts   : {len(all_alerts)}")
    print(f"  Critical alerts: {len(crits)}")
    print(f"\n  Worst layer detected: {watcher.watch_report.latest().worst_layer}")
    print(f"\n  Risk per layer (final):")
    for name, score in watcher.get_risk_summary().items():
        bar   = "#" * int(score * 16) + "." * (16 - int(score * 16))
        short = name.split(".")[-2]
        print(f"    {short}  [{bar}]  {score:.3f}")


if __name__ == "__main__":
    main()
