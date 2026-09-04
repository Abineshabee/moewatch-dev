# ============================================================
#  MoEWatch — examples/02_watchconfig_tour.py
#  Walk through every WatchConfig knob: thresholds, output
#  modes, intervention safety, and the bandit policy.
# ============================================================

"""
02_watchconfig_tour.py
======================
Demonstrates how WatchConfig controls every aspect of MoEWatch —
thresholds, sampling rate, output format, intervention limits,
and which policy makes decisions.

Three configurations are shown side-by-side so you can see the
effect of tightening thresholds, switching output modes, and
enabling the learning bandit policy.

Run:
    python examples/02_watchconfig_tour.py
"""

from __future__ import annotations

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import MagicMock

from moewatch import MoEWatch, WatchConfig, OutputMode, AlertLevel


# ---------------------------------------------------------------------------
# Reusable tiny MoE model (same as 01_minimal_loop.py)
# ---------------------------------------------------------------------------

class Expert(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MoELayer(nn.Module):
    def __init__(self, dim: int, n_experts: int = 4) -> None:
        super().__init__()
        self.gate    = nn.Linear(dim, n_experts, bias=True)
        self.experts = nn.ModuleList([Expert(dim) for _ in range(n_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        flat    = x.reshape(-1, D)
        top_i   = self.gate(flat).argmax(-1)
        out     = torch.zeros_like(flat)
        for e in range(len(self.experts)):
            mask = top_i == e
            if mask.any():
                out[mask] = self.experts[e](flat[mask])
        return out.reshape(B, S, D)


class TinyMoEModel(nn.Module):
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


def train(model: nn.Module, config: WatchConfig, steps: int = 100) -> MoEWatch:
    """Run a short loop and return the watcher for inspection."""
    torch.manual_seed(42)
    VOCAB = 64
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    watcher   = MoEWatch(model, config)
    watcher.start()
    for step in range(1, steps + 1):
        watcher.pre_step(step)
        ids    = torch.randint(0, VOCAB, (8, 16))
        logits = model(ids)
        loss   = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        watcher.step(step, loss.item())
    watcher.stop()
    return watcher


# ---------------------------------------------------------------------------
# Config A — silent + observation-only (no interventions)
#            Useful for baseline measurement or debugging.
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Config A: SILENT output, intervention disabled")
print("=" * 60)

config_a = WatchConfig(
    output=OutputMode.SILENT,       # no terminal output at all

    # Signal thresholds — tighter than default to catch small drifts early
    entropy_warn=0.55,              # warn when entropy drops below 55%
    entropy_critical=0.35,          # critical below 35%
    entropy_drop_warn=0.05,         # flag rapid per-step drops

    # Collection
    log_every=10,
    sample_every=1,                 # sample gradients every step (small model)
    stats_window=50,

    # Intervention off — observe only
    intervention_enabled=False,
)

model_a  = TinyMoEModel()
watcher_a = train(model_a, config_a, steps=100)

alerts_a = watcher_a.get_alerts()
print(f"Alerts collected : {len(alerts_a)}")
by_level = {}
for a in alerts_a:
    by_level[a.level.value] = by_level.get(a.level.value, 0) + 1
for lvl, cnt in sorted(by_level.items()):
    print(f"  {lvl.upper():<10}: {cnt}")

risk = watcher_a.get_risk_summary()
print("Final risk scores:")
for name, score in risk.items():
    print(f"  {name}: {score:.3f}")


# ---------------------------------------------------------------------------
# Config B — JSON output, strict safety limits
#            Good for piping into log aggregators.
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Config B: JSON output, strict safety limits")
print("=" * 60)

config_b = WatchConfig(
    output=OutputMode.JSON,         # emits newline-delimited JSON to stdout

    entropy_warn=0.60,
    entropy_critical=0.40,
    entropy_drop_warn=0.06,

    # Intervention safety — tight limits for production
    intervention_enabled=True,
    policy_type="rule",
    intervention_cooldown=50,       # at least 50 steps between interventions
    intervention_max_delta=0.05,    # each action changes coef by at most 0.05
    loss_guard_threshold=1.8,       # freeze if loss spikes 80% above baseline

    log_every=25,
    sample_every=5,
)

model_b   = TinyMoEModel()
watcher_b = train(model_b, config_b, steps=100)
print(f"\nTotal interventions: {watcher_b.watch_report.num_interventions}")


# ---------------------------------------------------------------------------
# Config C — CLI output, bandit policy
#            The bandit policy learns from reward feedback over time.
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Config C: CLI output, bandit policy (epsilon-greedy)")
print("=" * 60)

config_c = WatchConfig(
    output=OutputMode.SILENT,       # keep terminal clean in this script

    entropy_warn=0.60,
    entropy_critical=0.40,
    entropy_drop_warn=0.06,

    intervention_enabled=True,
    policy_type="bandit",           # learning policy instead of rule-based

    # Bandit hyperparameters
    bandit_epsilon=0.20,            # 20% random exploration
    reward_discount_gamma=0.90,     # discount future reward at 0.90/step
    reward_window_steps=30,         # evaluate outcome 30 steps after action

    # Baseline tracker
    baseline_min_clean_steps=10,    # need 10 clean steps before baseline is valid
    baseline_exclusion_window=30,   # exclude 30 steps after each intervention

    intervention_cooldown=15,
    intervention_max_delta=0.10,
    loss_guard_threshold=2.0,

    log_every=20,
    sample_every=2,
)

model_c   = TinyMoEModel()
watcher_c = train(model_c, config_c, steps=150)
print(f"Total interventions: {watcher_c.watch_report.num_interventions}")
print(f"Total alerts       : {len(watcher_c.get_alerts())}")

# Inspect per-layer risk history for the first layer
layer0 = watcher_c._layer_order[0] if watcher_c._layer_order else None
if layer0:
    history = watcher_c.watch_report.risk_history(layer0)
    if history:
        steps_h, scores = zip(*history)
        print(f"Layer 0 risk — min={min(scores):.3f}  max={max(scores):.3f}  "
              f"mean={sum(scores)/len(scores):.3f}")

print("\nAll three configs ran successfully.")
