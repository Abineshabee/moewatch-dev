# ============================================================
#  MoEWatch — examples/01_minimal_loop.py
#  The smallest possible usage: monitor a custom training loop
#  with zero configuration. Copy this as your starting point.
# ============================================================

"""
01_minimal_loop.py
==================
Attach MoEWatch to a plain PyTorch training loop in 4 lines.

No HuggingFace Trainer, no extra config — just wrap your loop
with watcher.start() / watcher.stop(), call pre_step() before
each forward pass and step() after the optimizer step.

What you see
------------
A live CLI dashboard (OutputMode.CLI by default) that updates
every 10 steps showing per-layer risk scores, entropy levels,
and any alerts that fired.

Run:
    python examples/01_minimal_loop.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import MagicMock

from moewatch import MoEWatch, WatchConfig, OutputMode

# ---------------------------------------------------------------------------
# Tiny MoE model (4 experts, top-1 routing, 3 layers)
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
        # MoEWatch's AuxLossAction writes here — must be a real float
        self.config = MagicMock()
        self.config.router_aux_loss_coef = 0.0

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        for layer in self.layers:
            x = x + layer(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main() -> None:
    torch.manual_seed(0)
    VOCAB, DIM, STEPS = 64, 32, 200

    model     = TinyMoEModel(VOCAB, DIM)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    # 1. Create config — CLI output, all defaults
    config = WatchConfig(
        output=OutputMode.CLI,
        log_every=20,
        intervention_enabled=True,
    )

    # 2. Create watcher and start
    watcher = MoEWatch(model, config)
    watcher.start()
    print(f"Monitoring {watcher.num_layers_monitored} MoE layers.\n")

    for step in range(1, STEPS + 1):
        # 3. Tell MoEWatch the step is starting (before forward pass)
        watcher.pre_step(step)

        ids     = torch.randint(0, VOCAB, (8, 16))
        logits  = model(ids)
        loss    = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 4. Process this step through MoEWatch
        report = watcher.step(global_step=step, current_loss=loss.item())

        if report.active_interventions:
            acts = [a.action_type for a in report.active_interventions]
            print(f"  [step {step}] Interventions fired: {acts}")

    # 5. Stop and print summary
    watcher.stop()
    print(f"\nDone. Total alerts: {len(watcher.get_alerts())}")
    print(watcher.watch_report.summary())


if __name__ == "__main__":
    main()
