# ============================================================
#  MoEWatch — examples/02_watchconfig_tour.py
#  WatchConfig key parameters explained with live output.
# ============================================================
"""
WatchConfig controls every aspect of MoEWatch.
This example walks through the three most important groups:

  A. Output modes     — SILENT / CLI / JSON
  B. Signal thresholds — when alerts fire
  C. Intervention safety — cooldown, max_delta, loss_guard

Each config group is shown with a short training run and its
effect on what gets reported.

Run:
    python examples/02_watchconfig_tour.py
"""

from __future__ import annotations

import json, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import MagicMock

from moewatch import MoEWatch, WatchConfig, OutputMode, AlertLevel


# ── Reusable tiny MoE model ─────────────────────────────────

class Expert(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.up      = nn.Linear(dim, dim * 4)
        self.dropout = nn.Dropout(p=0.0)
        self.down    = nn.Linear(dim * 4, dim)
    def forward(self, x): return self.down(self.dropout(F.gelu(self.up(x))))


class MoELayer(nn.Module):
    def __init__(self, dim: int, n: int = 4) -> None:
        super().__init__()
        self.gate    = nn.Linear(dim, n, bias=True)
        self.experts = nn.ModuleList([Expert(dim) for _ in range(n)])
    def forward(self, x):
        B, S, D = x.shape; flat = x.reshape(-1, D)
        probs = torch.softmax(self.gate(flat), -1); top_i = probs.argmax(-1)
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
        x = self.embed(ids)
        for l in self.layers: x = x + l(x)
        return self.head(x)
    def inject_bias(self, li, ei, s):
        with torch.no_grad(): self.layers[li].gate.bias[ei] = s


def quick_run(config: WatchConfig, steps: int = 100,
              pressure_from: int = 30) -> MoEWatch:
    """Run a short training loop and return the finished watcher."""
    torch.manual_seed(0)
    VOCAB = 64
    model     = TinyMoE()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    watcher   = MoEWatch(model, config)
    watcher.start()
    for step in range(1, steps + 1):
        s = (step - pressure_from) / max(steps - pressure_from, 1) * 2.2 \
            if step >= pressure_from else 0.0
        model.inject_bias(2, 1, s)
        watcher.pre_step(step)
        ids    = torch.randint(0, VOCAB, (8, 16))
        logits = model(ids)
        loss   = F.cross_entropy(
            logits[:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1)
        )
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        watcher.step(step, loss.item())
    watcher.stop()
    return watcher


# ── A: Output modes ─────────────────────────────────────────
print("=" * 55)
print("  A. Output Modes")
print("=" * 55)

# SILENT — you handle all output yourself
config_silent = WatchConfig(
    output=OutputMode.SILENT,       # no automatic output
    entropy_warn=0.65, entropy_critical=0.40,
    entropy_drop_warn=0.06,
    dead_threshold=0.05, cold_threshold=0.15, cold_steps_limit=10,
    intervention_enabled=False, log_every=10, sample_every=1,
)
w = quick_run(config_silent, steps=80)
crits = [a for a in w.get_alerts() if a.level == AlertLevel.CRITICAL]
print(f"SILENT mode — captured {len(w.get_alerts())} alerts "
      f"({len(crits)} critical) with no terminal output")

# JSON — structured log written to a file
JSON_PATH = "/tmp/moewatch_run.jsonl"
config_json = WatchConfig(
    output=OutputMode.JSON,
    entropy_warn=0.65, entropy_critical=0.40,
    entropy_drop_warn=0.06,
    dead_threshold=0.05, cold_threshold=0.15, cold_steps_limit=10,
    intervention_enabled=False, log_every=50, sample_every=5,
)
w2 = quick_run(config_json, steps=80)
# Save the full watch report as JSON
w2.watch_report.to_json(JSON_PATH)
with open(JSON_PATH) as f:
    data = json.load(f)
print(f"JSON mode  — report saved to {JSON_PATH} "
      f"({data['num_retained_steps']} step records)")

# CLI — live Rich dashboard (comment out if running non-interactively)
print("\nCLI mode outputs a Rich dashboard — set output=OutputMode.CLI")
print("to see it in an interactive terminal.\n")


# ── B: Signal thresholds ─────────────────────────────────────
print("=" * 55)
print("  B. Signal Thresholds")
print("=" * 55)

# Loose thresholds — only fire on severe collapse
config_loose = WatchConfig(
    output=OutputMode.SILENT,
    entropy_warn=0.30,          # warn only below 30% entropy
    entropy_critical=0.10,      # critical only below 10%
    entropy_drop_warn=0.15,
    dead_threshold=0.05, cold_threshold=0.15, cold_steps_limit=25,
    intervention_enabled=False, log_every=10, sample_every=1,
)
w_loose = quick_run(config_loose, steps=150)
print(f"Loose thresholds  (warn<0.30, crit<0.10): "
      f"{len(w_loose.get_alerts()):>5} total alerts")

# Tight thresholds — catch early drift
config_tight = WatchConfig(
    output=OutputMode.SILENT,
    entropy_warn=0.80,          # warn below 80% entropy
    entropy_critical=0.60,      # critical below 60%
    entropy_drop_warn=0.03,
    dead_threshold=0.05, cold_threshold=0.15, cold_steps_limit=5,
    intervention_enabled=False, log_every=10, sample_every=1,
)
w_tight = quick_run(config_tight, steps=150)
print(f"Tight thresholds  (warn<0.80, crit<0.60): "
      f"{len(w_tight.get_alerts()):>5} total alerts  "
      f"← catches early drift")


# ── C: Intervention safety ───────────────────────────────────
print("\n" + "=" * 55)
print("  C. Intervention Safety")
print("=" * 55)

# Conservative — long cooldown, small delta
config_conservative = WatchConfig(
    output=OutputMode.SILENT,
    entropy_warn=0.65, entropy_critical=0.40, entropy_drop_warn=0.06,
    dead_threshold=0.05, cold_threshold=0.15, cold_steps_limit=10,
    intervention_enabled=True,
    policy_type="rule",
    intervention_cooldown=200,      # at most one intervention per 200 steps
    intervention_max_delta=0.05,    # tiny aux_loss bump per intervention
    loss_guard_threshold=1.5,       # freeze if loss spikes 50%
    reward_window_steps=300,
    baseline_min_clean_steps=20, baseline_exclusion_window=100,
    log_every=10, sample_every=1,
)
wc = quick_run(config_conservative, steps=250)
print(f"Conservative (cooldown=200, delta=0.05): "
      f"{wc.watch_report.num_interventions} interventions fired")

# Aggressive — short cooldown, large delta
config_aggressive = WatchConfig(
    output=OutputMode.SILENT,
    entropy_warn=0.65, entropy_critical=0.40, entropy_drop_warn=0.06,
    dead_threshold=0.05, cold_threshold=0.15, cold_steps_limit=10,
    intervention_enabled=True,
    policy_type="rule",
    intervention_cooldown=60,       # can re-intervene every 60 steps
    intervention_max_delta=1.5,     # large aux_loss boost per intervention
    loss_guard_threshold=5.0,       # relaxed loss guard
    reward_window_steps=300,
    baseline_min_clean_steps=5, baseline_exclusion_window=5,
    log_every=10, sample_every=1,
)
wa = quick_run(config_aggressive, steps=250)
print(f"Aggressive (cooldown=60, delta=1.5):     "
      f"{wa.watch_report.num_interventions} interventions fired")

print("\nConfig tour complete.")
