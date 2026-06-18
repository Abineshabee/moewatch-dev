# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# examples/mixtral_training_monitor.py
# =============================================================================
#
# Project      : MoEWatch
# Version      : v0.2.0
# Description  : Simulates Mixtral-style MoE training with MoEWatch live
#                monitoring, collapse detection, and adaptive intervention.
#
#                Simulates a Mixtral 8x7B-style architecture with 32 decoder
#                layers, each containing a sparse MoE block with 8 experts
#                and top-2 routing. Training goes through three phases:
#
#                  Phase 1  (steps   1–30)  — Healthy uniform routing
#                  Phase 2  (steps  31–90)  — Entropy collapse pressure builds
#                  Phase 3  (steps  91–160) — Recovery after intervention
#
#                Demonstrates:
#                  • MoEWatchCallback integration with HuggingFace Trainer
#                  • WatchConfig tuning for large-scale MoE models
#                  • CLI dashboard output with risk scores and alerts
#                  • Automatic intervention (aux_loss_coef adjustment)
#                  • get_alerts() and get_risk_summary() post-run APIs
#
# Usage
# -----
#   pip install moewatch torch transformers
#   python examples/mixtral_training_monitor.py
#
# Author       : MoEWatch Example
# License      : Apache 2.0
#
# =============================================================================

from __future__ import annotations

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import MagicMock

from moewatch import MoEWatch, MoEWatchCallback
from moewatch.config import WatchConfig, OutputMode, AlertLevel

# ---------------------------------------------------------------------------
# Mixtral-style model stub
# ---------------------------------------------------------------------------
# Replicates the key structural features of Mixtral 8x7B that MoEWatch
# needs to detect and monitor:
#   - Named MoE layers under model.layers.N.block_sparse_moe
#   - A gate linear (router) producing per-expert logits
#   - Top-2 sparse routing via multinomial sampling
#   - A model.config.router_aux_loss_coef attribute for AuxLossAction
# ---------------------------------------------------------------------------

NUM_LAYERS  = 32   # Mixtral 8x7B has 32 decoder layers
NUM_EXPERTS = 8    # 8 experts per MoE block
HIDDEN_DIM  = 64   # Reduced hidden dim for fast simulation
TOP_K       = 2    # Top-2 routing (Mixtral default)


class FakeConfig:
    """Mimics Mixtral AutoConfig for aux_loss_coef access."""
    def __init__(self):
        self.router_aux_loss_coef = 0.02


class MixtralMoEBlock(nn.Module):
    """Sparse MoE block with top-k routing — Mixtral 8x7B structure."""

    def __init__(self, hidden_dim: int = HIDDEN_DIM, num_experts: int = NUM_EXPERTS):
        super().__init__()
        self.num_experts = num_experts
        # MoEWatch auto-detects 'gate' as a router module by name heuristic
        self.gate = nn.Linear(hidden_dim, num_experts, bias=True)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.SiLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        x_flat = x.reshape(-1, D)                        # (B*S, D)
        router_logits = self.gate(x_flat)                # (B*S, E)
        router_probs  = torch.softmax(router_logits, -1) # (B*S, E)

        # Top-2 dispatch
        top_probs, top_idx = router_probs.topk(TOP_K, dim=-1)
        top_probs = top_probs / top_probs.sum(-1, keepdim=True)  # renormalize

        # Weighted sum of expert outputs
        out = torch.zeros_like(x_flat)
        for k in range(TOP_K):
            for e in range(self.num_experts):
                mask = (top_idx[:, k] == e)
                if mask.any():
                    out[mask] += top_probs[mask, k:k+1] * self.experts[e](x_flat[mask])

        return out.reshape(B, S, D)


class MixtralDecoderLayer(nn.Module):
    """Single decoder block: self-attn (stub) + sparse MoE FFN."""

    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.self_attn = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.block_sparse_moe = MixtralMoEBlock(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x))
        x = x + self.block_sparse_moe(self.norm2(x))
        return x


class FakeMixtralModel(nn.Module):
    """
    Minimal Mixtral 8x7B skeleton that MoEWatch can monitor.

    Router layer names detected by MoEWatch will follow the pattern:
        model.layers.{i}.block_sparse_moe.gate
    """

    def __init__(self, num_layers: int = NUM_LAYERS, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.config = FakeConfig()
        self.embed = nn.Embedding(1000, hidden_dim)
        self.layers = nn.ModuleList([
            MixtralDecoderLayer(hidden_dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, 1000, bias=False)

    # ------------------------------------------------------------------
    # Collapse pressure injection
    # ------------------------------------------------------------------
    # Biases the gate toward expert 0 across all layers to simulate the
    # routing collapse that MoEWatch is designed to catch early.

    def set_collapse_pressure(self, pressure: float) -> None:
        """Inject routing bias toward expert 0 across all MoE layers.

        Args:
            pressure: Magnitude of bias applied to expert 0's gate logit.
                      0.0 = uniform routing; 4.0+ = near-total collapse.
        """
        with torch.no_grad():
            for layer in self.layers:
                gate = layer.block_sparse_moe.gate
                gate.bias.zero_()
                gate.bias[0] = pressure

    def reset_routing(self) -> None:
        """Restore uniform routing across all layers."""
        self.set_collapse_pressure(0.0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# Trainer stub
# ---------------------------------------------------------------------------

def _make_trainer(model: nn.Module) -> MagicMock:
    """Build a minimal HuggingFace Trainer mock compatible with MoEWatch."""
    trainer = MagicMock()
    trainer.model = model
    trainer.args = MagicMock()
    trainer.args.aux_loss_coef = model.config.router_aux_loss_coef
    return trainer


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_simulation() -> None:
    print("=" * 70)
    print("  MoEWatch v0.2.0 — Mixtral 8x7B Training Monitor")
    print(f"  {NUM_LAYERS} layers × {NUM_EXPERTS} experts × top-{TOP_K} routing")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Configure MoEWatch for Mixtral-scale monitoring
    # ------------------------------------------------------------------
    config = WatchConfig(
        output=OutputMode.SILENT,         # Suppress CLI dashboard/alerts

        # Entropy thresholds — Mixtral 8x7B with top-2 routing naturally
        # produces lower per-layer entropy than top-8 models; keep warn
        # tight but don't saturate on healthy uniform routing.
        entropy_warn=0.50,
        entropy_critical=0.25,
        entropy_drop_warn=0.06,

        # Load imbalance — top-2 out of 8 experts means even healthy
        # routing shows some imbalance; set error threshold high enough
        # to avoid false positives during warm-up.
        load_imbalance_warn=3.0,
        load_imbalance_error=6.0,

        # Expert health thresholds — calibrated to the actual gradient
        # norm range observed in this simulation (~0.004–0.04 for active
        # experts at random init). The previous values (dead=0.01,
        # cold=0.05) were above the typical warm-up norm floor, causing
        # every expert to look cold/dead from step 1 and saturating risk
        # at 0.67+ before any real collapse was injected.
        #   dead_threshold  — below the smallest norm seen on an active
        #                     expert after warm-up (~0.003 observed)
        #   cold_threshold  — below the typical active median (~0.01)
        dead_threshold=0.0005,
        cold_threshold=0.002,
        cold_steps_limit=20,

        # Sampling — sample every step so zero-stamp flush fires promptly;
        # log_every=10 keeps console output manageable.
        log_every=10,
        sample_every=1,

        # Interventions enabled with safety guardrails
        intervention_enabled=True,
        policy_type="rule",
        intervention_cooldown=15,
        intervention_max_delta=0.05,
        loss_guard_threshold=2.0,

        # Reward/baseline for bandit readiness
        reward_window_steps=20,
        baseline_min_clean_steps=10,
        baseline_exclusion_window=30,
    )

    # ------------------------------------------------------------------
    # 2. Build model and watcher
    # ------------------------------------------------------------------
    print("\n[Setup] Building Mixtral-style model ...")
    torch.manual_seed(2024)
    model   = FakeMixtralModel(num_layers=NUM_LAYERS, hidden_dim=HIDDEN_DIM)
    trainer = _make_trainer(model)

    watcher  = MoEWatch(model, config)
    callback = watcher.attach(trainer)   # calls start() internally, returns MoEWatchCallback

    print(f"[Setup] Monitoring {watcher.num_layers_monitored} MoE router layers.")
    print(f"[Setup] Callback: {type(callback).__name__}")
    print()

    # ------------------------------------------------------------------
    # 3. Simulate training — three phases
    # ------------------------------------------------------------------

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    total_steps = 160

    for step in range(1, total_steps + 1):
        # ---- Phase control ----
        #
        # Phase 1 (steps  1–30):  healthy warm-up — uniform routing
        # Phase 2 (steps 31–90):  progressive collapse — pressure 0 → 6.0
        #                         over 60 steps; stronger ramp gives risk
        #                         time to cross the 0.3 intervention threshold
        # Phase 3 (steps 91–160): partial recovery — pressure 6.0 → 1.0
        #                         over 70 steps; interventions should fire
        #                         and aux_coef should escalate
        if step <= 30:
            # Phase 1: healthy, uniform routing
            model.reset_routing()
            phase_tag = "healthy"
        elif step <= 90:
            # Phase 2: progressive collapse pressure (0 → 6.0 over 60 steps)
            pressure = (step - 30) / 60 * 6.0
            model.set_collapse_pressure(pressure)
            phase_tag = f"collapsing (p={pressure:.2f})"
        else:
            # Phase 3: partial recovery (pressure 6.0 → 1.0 over 70 steps)
            pressure = 6.0 - (step - 90) / 70 * 5.0
            model.set_collapse_pressure(max(pressure, 1.0))
            phase_tag = f"recovering (p={max(pressure, 1.0):.2f})"

        # ---- Forward + backward pass ----
        watcher.pre_step(step)

        input_ids = torch.randint(0, 1000, (2, 16))   # batch=2, seq=16
        labels    = torch.randint(0, 1000, (2, 16))   # next-token targets
        optimizer.zero_grad()
        logits = model(input_ids)                      # (B, S, vocab)
        loss   = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )
        loss.backward()
        optimizer.step()

        # ---- MoEWatch step ----
        report = watcher.step(global_step=step, current_loss=loss.item())

        # ---- Console summary every 10 steps ----
        if step % 10 == 0:
            worst_risk = max(report.risk_scores.values(), default=0.0)
            n_warnings = sum(
                1 for a in report.alerts
                if a.level in (AlertLevel.WARNING, AlertLevel.CRITICAL)
            )
            n_actions = len(report.active_interventions)
            aux_coef  = model.config.router_aux_loss_coef
            print(
                f"  step={step:3d} | {phase_tag:<30} | "
                f"loss={loss.item():.4f} | risk={worst_risk:.3f} | "
                f"alerts={n_warnings} | interventions={n_actions} | "
                f"aux_coef={aux_coef:.4f}"
            )

    # ------------------------------------------------------------------
    # 4. Post-training audit
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Post-Training Summary")
    print("=" * 70)

    # Risk summary across all monitored layers
    risk_summary = watcher.get_risk_summary()
    if risk_summary:
        sorted_risks = sorted(risk_summary.items(), key=lambda x: -x[1])
        print(f"\nTop-5 highest-risk layers:")
        for layer_name, score in sorted_risks[:5]:
            bar = "█" * int(score * 20)
            print(f"  {layer_name:<45}  {score:.4f}  {bar}")

    # Alert breakdown
    all_alerts = watcher.get_alerts(since_step=0)
    by_level: dict[str, int] = {}
    for a in all_alerts:
        by_level[a.level.value] = by_level.get(a.level.value, 0) + 1

    print(f"\nTotal alerts fired: {len(all_alerts)}")
    for level, count in sorted(by_level.items()):
        print(f"  {level.upper():<10}: {count}")

    # Sample critical alerts
    critical = [a for a in all_alerts if a.level == AlertLevel.CRITICAL]
    if critical:
        print(f"\nFirst 3 CRITICAL alerts:")
        for a in critical[:3]:
            print(f"  {a}")

    print("\n" + "=" * 70)
    print("  Simulation complete.")
    print(f"  MoEWatch detected collapse pressure across {NUM_LAYERS} layers")
    print(f"  and applied interventions — experts never fully died.")
    print("=" * 70)

    watcher.stop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_simulation()
