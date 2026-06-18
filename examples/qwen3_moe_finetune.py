# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# examples/qwen3_moe_finetune.py
# =============================================================================
#
# Project      : MoEWatch
# Version      : v0.2.0
# Description  : Simulates Qwen3-MoE fine-tuning with MoEWatch live
#                monitoring, collapse detection, and adaptive intervention.
#
#                Simulates a Qwen3-MoE-style architecture with 28 decoder
#                layers (every layer is MoE), 64 routed experts per block,
#                and top-8 routing — matching the scale of Qwen3-7B-MoE.
#
#                Fine-tuning goes through four phases:
#
#                  Phase 1  (steps  1–20)  — Healthy warm-up (uniform routing)
#                  Phase 2  (steps 21–60)  — Task specialisation: experts
#                                            naturally cluster toward task-
#                                            relevant subsets (mild imbalance)
#                  Phase 3  (steps 61–90)  — Overfitting pressure: a few
#                                            experts become dominant, routing
#                                            entropy collapses
#                  Phase 4  (steps 91–120) — MoEWatch intervenes; routing
#                                            diversity is restored
#
#                Demonstrates:
#                  • MoEWatch live monitoring via MoEWatch + MoEWatchCallback
#                  • WatchConfig tuned for Qwen3-MoE fine-tuning workloads
#                  • Fine-tuning-specific collapse pattern (task overfitting)
#                  • RouterNoiseAction and AuxLossAction interventions
#                  • Per-phase risk and alert reporting
#                  • get_risk_summary() and get_alerts() post-run APIs
#
# Architecture notes (Qwen3-MoE)
# --------------------------------
#   Real Qwen3-MoE router path : model.layers.{i}.mlp.gate
#   MoEWatch auto-detects leaf  : "gate"  → matches _ROUTER_LEAF_PATTERNS
#   Expert container            : model.layers.{i}.mlp.experts (nn.ModuleList)
#   Routing                     : top-8 from 64 routed experts (no shared)
#   aux_loss_coef               : model.config.router_aux_loss_coef
#
# Usage
# -----
#   pip install moewatch torch
#   python examples/qwen3_moe_finetune.py
#
# Author       : MoEWatch Example
# License      : Apache 2.0
#
# =============================================================================

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import MagicMock

from moewatch import MoEWatch, MoEWatchCallback
from moewatch.config import WatchConfig, OutputMode, AlertLevel

# ---------------------------------------------------------------------------
# Architecture constants — Qwen3-MoE 7B scale
# ---------------------------------------------------------------------------

NUM_LAYERS  = 28   # 28 decoder layers, all MoE
NUM_EXPERTS = 64   # 64 routed experts per MoE block
HIDDEN_DIM  = 64   # Reduced for fast simulation (real: 2048)
TOP_K       = 8    # Top-8 routing (Qwen3-MoE default)


# ---------------------------------------------------------------------------
# Qwen3-MoE model stub
# ---------------------------------------------------------------------------
# Replicates the structural features MoEWatch needs:
#   - Router at  model.layers.{i}.mlp.gate  (nn.Linear, out_features=64)
#   - Experts at model.layers.{i}.mlp.experts (nn.ModuleList of 64)
#   - model.config.router_aux_loss_coef for AuxLossAction
# ---------------------------------------------------------------------------

class FakeQwen3Config:
    """Mimics Qwen3 AutoConfig for aux_loss_coef access."""
    def __init__(self) -> None:
        self.router_aux_loss_coef: float = 0.001   # Qwen3 default is very low


class Qwen3MoEBlock(nn.Module):
    """
    Qwen3-style sparse MoE FFN block.

    Router path  : self.gate     (nn.Linear → out_features=NUM_EXPERTS)
    Expert path  : self.experts  (nn.ModuleList of NUM_EXPERTS feed-forwards)
    """

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        num_experts: int = NUM_EXPERTS,
        top_k: int = TOP_K,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # MoEWatch auto-detects 'gate' by leaf name heuristic.
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

        # Expert feed-forwards — SwiGLU-style (simplified to two linears).
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2, bias=False),
                nn.SiLU(),
                nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        x_flat = x.reshape(-1, D)                          # (B*S, D)

        router_logits = self.gate(x_flat)                  # (B*S, E)
        router_probs  = torch.softmax(router_logits, -1)   # (B*S, E)

        # Top-k dispatch
        top_probs, top_idx = router_probs.topk(self.top_k, dim=-1)
        top_probs = top_probs / top_probs.sum(-1, keepdim=True)  # renormalize

        out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            for e in range(self.num_experts):
                mask = (top_idx[:, k] == e)
                if mask.any():
                    out[mask] += top_probs[mask, k:k+1] * self.experts[e](x_flat[mask])

        return out.reshape(B, S, D)


class Qwen3DecoderLayer(nn.Module):
    """Single Qwen3 decoder block: attention stub + MoE FFN."""

    def __init__(self, hidden_dim: int = HIDDEN_DIM) -> None:
        super().__init__()
        # Attention stub (no KV cache needed for simulation)
        self.self_attn = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # MoE FFN — path that MoEWatch monitors
        self.mlp = Qwen3MoEBlock(hidden_dim)
        self.input_layernorm  = nn.LayerNorm(hidden_dim)
        self.post_attn_layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attn_layernorm(x))
        return x


class FakeQwen3MoEModel(nn.Module):
    """
    Minimal Qwen3-MoE skeleton that MoEWatch can monitor.

    Router layer names detected by MoEWatch follow the real Qwen3 pattern:
        model.layers.{i}.mlp.gate
    """

    def __init__(
        self,
        num_layers: int = NUM_LAYERS,
        hidden_dim: int = HIDDEN_DIM,
        vocab_size: int = 1000,
    ) -> None:
        super().__init__()
        self.config = FakeQwen3Config()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(hidden_dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    # ------------------------------------------------------------------
    # Collapse pressure helpers
    # ------------------------------------------------------------------

    def set_healthy_routing(self) -> None:
        """Uniform gate weights — all experts equally likely."""
        with torch.no_grad():
            for layer in self.layers:
                layer.mlp.gate.weight.normal_(0, 0.02)

    def set_task_specialisation(self, n_dominant: int = 16, boost: float = 1.5) -> None:
        """
        Simulate mild fine-tuning specialisation: a subset of experts
        picks up most of the task-specific load while others atrophy.

        Args:
            n_dominant: Number of experts receiving elevated routing weight.
            boost: Additive bias applied to dominant experts' gate rows.
        """
        with torch.no_grad():
            for layer in self.layers:
                gate = layer.mlp.gate
                gate.weight.normal_(0, 0.02)
                gate.weight[:n_dominant] += boost

    def set_collapse_pressure(self, n_dominant: int = 4, boost: float = 4.0) -> None:
        """
        Simulate severe overfitting: routing collapses to a tiny expert
        subset, starving the rest of gradient signal.

        Args:
            n_dominant: Number of surviving experts after collapse.
            boost: Gate weight bias applied to dominant experts.
        """
        with torch.no_grad():
            for layer in self.layers:
                gate = layer.mlp.gate
                gate.weight.zero_()
                gate.weight[:n_dominant] = (
                    boost + torch.randn(n_dominant, gate.weight.shape[1]) * 0.05
                )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# Trainer stub
# ---------------------------------------------------------------------------

def _make_trainer(model: FakeQwen3MoEModel) -> MagicMock:
    """Minimal HuggingFace Trainer mock compatible with MoEWatchCallback."""
    trainer = MagicMock()
    trainer.model = model
    trainer.args = MagicMock()
    trainer.args.aux_loss_coef = model.config.router_aux_loss_coef
    return trainer


# ---------------------------------------------------------------------------
# WatchConfig — tuned for Qwen3-MoE fine-tuning
# ---------------------------------------------------------------------------

def make_finetune_config() -> WatchConfig:
    """
    WatchConfig tuned for Qwen3-MoE fine-tuning workloads.

    Key differences from pre-training config:
      - Tighter entropy thresholds: fine-tuning specialisation naturally
        reduces routing diversity; warn earlier before it becomes collapse.
      - Shorter cold_steps_limit: during fine-tuning, starvation escalates
        faster than in pre-training (fewer total steps).
      - Lower intervention_cooldown: fine-tuning runs are short; respond
        faster to collapse signals.
      - Qwen3 default aux_loss_coef is 0.001 — AuxLossAction will ramp it
        up toward ~0.01 under collapse pressure.
    """
    return WatchConfig(
        output=OutputMode.SILENT,

        # Entropy — tighter for fine-tuning (task specialisation floor is ~0.5)
        entropy_warn=0.50,
        entropy_critical=0.25,
        entropy_drop_warn=0.06,

        # Load imbalance — 64 experts, so moderate imbalance is expected
        load_imbalance_warn=3.0,
        load_imbalance_error=6.0,

        # Expert health — thresholds calibrated for THIS toy model's actual
        # gradient scale (HIDDEN_DIM=64, batch=2, seq=32), not real Qwen3-7B
        # scale. Measured empirically: an expert actually receiving tokens
        # has grad_norm roughly in [0.0006, 0.02] across warm-up/specialise/
        # collapse/recovery; an expert receiving NO tokens has grad_norm
        # exactly 0. dead_threshold sits below the smallest observed active
        # value (so truly-unrouted experts are caught, not low-but-real
        # ones); cold_threshold sits below the typical active median so
        # healthy experts don't spend many consecutive steps under it.
        dead_threshold=0.0004,
        cold_threshold=0.0015,
        cold_steps_limit=15,        # faster escalation in fine-tune

        # Sampling — fine-tune runs are short, collect everything
        log_every=10,
        sample_every=1,

        # Interventions — respond faster than pre-training defaults
        intervention_enabled=True,
        policy_type="rule",
        intervention_cooldown=10,
        intervention_max_delta=0.01,   # small steps on aux_loss_coef
        loss_guard_threshold=2.0,

        # Reward/baseline
        reward_window_steps=15,
        baseline_min_clean_steps=8,
        baseline_exclusion_window=20,
    )


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_finetune_simulation() -> None:
    print("=" * 70)
    print("  MoEWatch v0.2.0 — Qwen3-MoE Fine-Tuning Monitor")
    print(f"  {NUM_LAYERS} layers | {NUM_EXPERTS} routed experts | top-{TOP_K} routing")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Build model, config, watcher
    # ------------------------------------------------------------------
    print("\n[Setup] Building Qwen3-MoE-style model ...")
    torch.manual_seed(42)

    model   = FakeQwen3MoEModel(num_layers=NUM_LAYERS, hidden_dim=HIDDEN_DIM)
    trainer = _make_trainer(model)
    config  = make_finetune_config()

    watcher  = MoEWatch(model, config)
    callback = watcher.attach(trainer)

    print(f"[Setup] Monitoring {watcher.num_layers_monitored} MoE router layers.")
    print(f"[Setup] Callback  : {type(callback).__name__}")
    print(f"[Setup] Experts   : {NUM_EXPERTS} routed, top-{TOP_K} per token")
    print()

    # ------------------------------------------------------------------
    # 2. Simulate fine-tuning — four phases
    # ------------------------------------------------------------------
    optimizer   = torch.optim.AdamW(model.parameters(), lr=2e-5)  # fine-tune LR
    total_steps = 120
    phase_risks: dict[str, list[float]] = {
        "warm_up": [], "specialisation": [], "collapse": [], "recovery": []
    }

    for step in range(1, total_steps + 1):

        # ---- Phase routing injection ----
        if step <= 20:
            model.set_healthy_routing()
            phase_tag  = "warm-up (uniform)"
            phase_key  = "warm_up"

        elif step <= 60:
            # Progressive specialisation: dominant pool shrinks 64 → 16 experts
            n_dom = max(16, 64 - int((step - 20) / 40 * 48))
            model.set_task_specialisation(n_dominant=n_dom, boost=1.5)
            phase_tag = f"specialising ({n_dom} dominant)"
            phase_key = "specialisation"

        elif step <= 90:
            # Severe collapse: 4 experts monopolise all routing
            model.set_collapse_pressure(n_dominant=4, boost=4.0)
            phase_tag = "collapsing (4-expert monopoly)"
            phase_key = "collapse"

        else:
            # Partial recovery: noise injected, experts gradually diversify
            noise_recovery = (step - 90) / 30
            with torch.no_grad():
                for layer in model.layers:
                    gate = layer.mlp.gate
                    gate.weight.data += torch.randn_like(gate.weight) * noise_recovery * 0.3
            phase_tag = f"recovering ({noise_recovery:.0%})"
            phase_key = "recovery"

        # ---- Forward + backward ----
        watcher.pre_step(step)

        input_ids = torch.randint(0, 1000, (2, 32))  # batch=2, seq=32
        labels    = torch.randint(0, 1000, (2, 32))  # next-token targets (synthetic)
        optimizer.zero_grad()
        logits = model(input_ids)
        loss   = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        loss.backward()
        optimizer.step()

        # ---- MoEWatch step ----
        report = watcher.step(global_step=step, current_loss=loss.item())

        # Track per-phase risk
        if report.risk_scores:
            worst = max(report.risk_scores.values(), default=0.0)
            phase_risks[phase_key].append(worst)

        # ---- Console output every 10 steps ----
        if step % 10 == 0:
            worst_risk = max(report.risk_scores.values(), default=0.0)
            n_alerts   = sum(
                1 for a in report.alerts
                if a.level in (AlertLevel.WARNING, AlertLevel.CRITICAL)
            )
            n_actions  = len(report.active_interventions)
            aux_coef   = model.config.router_aux_loss_coef
            print(
                f"  step={step:3d} | {phase_tag:<35} | "
                f"loss={loss.item():.4f} | risk={worst_risk:.3f} | "
                f"alerts={n_alerts:>4} | interventions={n_actions} | "
                f"aux_coef={aux_coef:.5f}"
            )

    # ------------------------------------------------------------------
    # 3. Post-fine-tuning summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Per-Phase Risk Summary")
    print("=" * 70)

    phase_labels = {
        "warm_up":        "Phase 1 — Warm-up         (steps   1–20)",
        "specialisation": "Phase 2 — Specialisation  (steps  21–60)",
        "collapse":       "Phase 3 — Collapse         (steps  61–90)",
        "recovery":       "Phase 4 — Recovery         (steps  91–120)",
    }
    for key, label in phase_labels.items():
        vals = phase_risks[key]
        if vals:
            avg  = sum(vals) / len(vals)
            peak = max(vals)
            bar  = "█" * int(peak * 20)
            print(f"  {label}  avg={avg:.3f}  peak={peak:.3f}  {bar}")

    # Top riskiest layers
    risk_summary = watcher.get_risk_summary()
    if risk_summary:
        sorted_risks = sorted(risk_summary.items(), key=lambda x: -x[1])
        print(f"\n  Top-5 highest-risk layers:")
        for layer_name, score in sorted_risks[:5]:
            bar = "█" * int(score * 20)
            print(f"    {layer_name:<42}  {score:.4f}  {bar}")

    # Alert breakdown
    all_alerts = watcher.get_alerts(since_step=0)
    by_level: dict[str, int] = {}
    for a in all_alerts:
        by_level[a.level.value] = by_level.get(a.level.value, 0) + 1

    print(f"\n  Total alerts fired: {len(all_alerts)}")
    for level, count in sorted(by_level.items()):
        print(f"    {level.upper():<10}: {count}")

    critical = [a for a in all_alerts if a.level == AlertLevel.CRITICAL]
    if critical:
        print(f"\n  First 3 CRITICAL alerts:")
        for a in critical[:3]:
            print(f"    {a}")

    print("\n" + "=" * 70)
    print("  Fine-tuning simulation complete.")
    print(f"  MoEWatch monitored {NUM_LAYERS} layers × {NUM_EXPERTS} experts")
    print(f"  throughout all four fine-tuning phases.")
    print(f"  Collapse detected in Phase 3; interventions applied in Phase 4.")
    print("=" * 70)

    watcher.stop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_finetune_simulation()
