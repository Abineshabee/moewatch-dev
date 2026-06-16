# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# examples/deepseek_audit.py
# =============================================================================
#
# Project      : MoEWatch
# Version      : v0.2.0
# Description  : Offline post-training diagnostic audit for a DeepSeek-MoE
#                style architecture using the moewatch.audit() API.
#
#                Simulates a DeepSeek-V2-style MoE model with:
#                  - 27 decoder layers (alternating dense + MoE)
#                  - 64 routed experts + 2 shared experts per MoE block
#                  - Top-6 expert routing
#                  - Fine-grained expert segmentation (DeepSeek pattern)
#
#                Three post-training health scenarios are audited:
#                  Scenario A — Healthy checkpoint (uniform routing)
#                  Scenario B — Partially collapsed checkpoint (mid-training)
#                  Scenario C — Severely collapsed checkpoint (routing failure)
#
#                Demonstrates:
#                  • audit() offline API (no Trainer, no live hooks)
#                  • AuditReport.summary() — human-readable text report
#                  • AuditReport.layers_by_risk() — risk ranking
#                  • AuditReport.dead_experts() — dead expert enumeration
#                  • AuditReport.gradient_starved_experts() — starvation scan
#                  • AuditReport.to_json() — serialization to disk
#                  • AuditReport.to_dataframe() — pandas tabular export
#
# Usage
# -----
#   pip install moewatch torch
#   python examples/deepseek_audit.py
#
# Author       : MoEWatch Example
# License      : Apache 2.0
#
# =============================================================================

from __future__ import annotations

import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import List

from moewatch import audit
from moewatch.config import WatchConfig, OutputMode

# ---------------------------------------------------------------------------
# DeepSeek-MoE architecture stub
# ---------------------------------------------------------------------------
# Replicates the structural features MoEWatch needs to detect and monitor:
#   - Named MoE layers under layers.N.mlp (every 3rd layer is MoE)
#   - A gate linear producing per-expert logits for routed experts
#   - Shared experts (always active, not routed)
#   - Top-K sparse routing via topk selection
#   - model.config.router_aux_loss_coef for AuxLossAction compatibility
# ---------------------------------------------------------------------------

NUM_LAYERS       = 27    # DeepSeek-V2 has 27 transformer layers
NUM_ROUTED       = 64    # Routed experts per MoE block
NUM_SHARED       = 2     # Always-active shared experts
TOP_K            = 6     # Top-6 routing (DeepSeek-V2 default)
HIDDEN_DIM       = 64    # Reduced for fast simulation
MOE_LAYER_FREQ   = 3     # Every 3rd layer is MoE (layers 2, 5, 8 ...)


class DeepSeekConfig:
    """Mimics DeepSeek AutoConfig for aux_loss_coef access."""
    def __init__(self):
        self.router_aux_loss_coef    = 0.001   # DeepSeek uses very small coef
        self.num_experts             = NUM_ROUTED
        self.num_experts_per_tok     = TOP_K
        self.n_shared_experts        = NUM_SHARED


class DeepSeekSharedExpert(nn.Module):
    """Always-active shared expert (not routed)."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class DeepSeekRoutedExpert(nn.Module):
    """Single routed expert FFN."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class DeepSeekMoEBlock(nn.Module):
    """
    DeepSeek-V2 style MoE block.

    Combines shared experts (always active) with routed experts (top-k).
    MoEWatch detects 'gate' as the router module by name heuristic.
    """

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        num_routed: int = NUM_ROUTED,
        num_shared: int = NUM_SHARED,
        top_k: int = TOP_K,
    ):
        super().__init__()
        self.num_routed = num_routed
        self.num_shared = num_shared
        self.top_k      = top_k

        # Router gate — MoEWatch auto-detects this by 'gate' name
        self.gate = nn.Linear(hidden_dim, num_routed, bias=False)

        # Routed experts
        self.experts = nn.ModuleList([
            DeepSeekRoutedExpert(hidden_dim) for _ in range(num_routed)
        ])

        # Shared experts (always active, bypass routing)
        self.shared_experts = nn.ModuleList([
            DeepSeekSharedExpert(hidden_dim) for _ in range(num_shared)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        x_flat = x.reshape(-1, D)   # (B*S, D)

        # --- Shared expert output (always active) ---
        shared_out = torch.zeros_like(x_flat)
        for shared_expert in self.shared_experts:
            shared_out = shared_out + shared_expert(x_flat)

        # --- Routed expert output (top-k) ---
        router_logits = self.gate(x_flat)                        # (B*S, E)
        router_probs  = torch.softmax(router_logits, dim=-1)
        top_probs, top_idx = router_probs.topk(self.top_k, dim=-1)
        top_probs = top_probs / top_probs.sum(-1, keepdim=True)  # renormalize

        routed_out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            for e in range(self.num_routed):
                mask = (top_idx[:, k] == e)
                if mask.any():
                    routed_out[mask] += (
                        top_probs[mask, k:k+1] * self.experts[e](x_flat[mask])
                    )

        return (shared_out + routed_out).reshape(B, S, D)


class DeepSeekDecoderLayer(nn.Module):
    """Single decoder block: self-attn stub + (MoE or dense) FFN."""

    def __init__(self, hidden_dim: int, is_moe: bool):
        super().__init__()
        self.is_moe   = is_moe
        self.self_attn = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm1     = nn.LayerNorm(hidden_dim)
        self.norm2     = nn.LayerNorm(hidden_dim)

        if is_moe:
            self.mlp = DeepSeekMoEBlock(hidden_dim)
        else:
            # Dense FFN for non-MoE layers
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.SiLU(),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FakeDeepSeekModel(nn.Module):
    """
    Minimal DeepSeek-V2 MoE skeleton that moewatch.audit() can analyze.

    MoE layers are at indices where (layer_idx % MOE_LAYER_FREQ == 2),
    matching DeepSeek-V2's alternating dense/MoE pattern.

    Router layer names detected by MoEWatch follow the pattern:
        layers.{i}.mlp.gate
    """

    def __init__(self, num_layers: int = NUM_LAYERS, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.config  = DeepSeekConfig()
        self.embed   = nn.Embedding(32000, hidden_dim)
        self.layers  = nn.ModuleList([
            DeepSeekDecoderLayer(
                hidden_dim,
                is_moe=(i % MOE_LAYER_FREQ == 2),   # every 3rd layer is MoE
            )
            for i in range(num_layers)
        ])
        self.norm    = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, 32000, bias=False)

        self._moe_layer_indices = [
            i for i in range(num_layers) if i % MOE_LAYER_FREQ == 2
        ]

    # ------------------------------------------------------------------
    # Collapse scenario injection
    # ------------------------------------------------------------------

    def set_healthy_routing(self) -> None:
        """Reset all router gates to uniform routing."""
        with torch.no_grad():
            for i in self._moe_layer_indices:
                self.layers[i].mlp.gate.weight.normal_(0, 0.02)

    def set_partial_collapse(self, top_experts: int = 8) -> None:
        """Concentrate routing toward a subset of experts.

        Simulates mid-training partial collapse where top_experts
        receive most of the token load.
        """
        with torch.no_grad():
            for i in self._moe_layer_indices:
                gate = self.layers[i].mlp.gate
                gate.weight.normal_(0, 0.02)
                # Boost the top experts' output weights
                gate.weight[:top_experts] += 2.5

    def set_severe_collapse(self, dominant_expert: int = 0) -> None:
        """Route nearly all tokens to a single expert.

        Simulates late-stage routing collapse / expert monopolization.
        """
        with torch.no_grad():
            for i in self._moe_layer_indices:
                gate = self.layers[i].mlp.gate
                gate.weight.zero_()
                gate.weight[dominant_expert] += 6.0   # extreme bias

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))

    def moe_layer_count(self) -> int:
        return len(self._moe_layer_indices)


# ---------------------------------------------------------------------------
# Dataloader factory
# ---------------------------------------------------------------------------

def make_val_loader(
    num_batches: int = 50,
    batch_size: int = 4,
    seq_len: int = 32,
    vocab_size: int = 32000,
) -> DataLoader:
    """Build a synthetic validation dataloader for the audit pass."""
    torch.manual_seed(42)
    total_samples = num_batches * batch_size
    input_ids = torch.randint(0, vocab_size, (total_samples, seq_len))
    dataset   = TensorDataset(input_ids)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# Audit config
# ---------------------------------------------------------------------------

def make_audit_config() -> WatchConfig:
    """WatchConfig tuned for DeepSeek-scale offline audit."""
    return WatchConfig(
        output=OutputMode.SILENT,         # Suppress live output during audit

        # DeepSeek uses many fine-grained experts — tighter imbalance thresholds
        entropy_warn=0.40,
        entropy_critical=0.18,
        entropy_drop_warn=0.07,
        load_imbalance_warn=2.0,
        load_imbalance_error=4.5,

        # Expert health — DeepSeek has 64 experts, starvation is more common
        dead_threshold=0.008,
        cold_threshold=0.04,
        cold_steps_limit=30,

        # Sampling (for audit, collect everything)
        log_every=1,
        sample_every=1,

        # Interventions off — audit is read-only
        intervention_enabled=False,
    )


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_audit_results(
    scenario_name: str,
    report,
    show_full_summary: bool = True,
) -> None:
    """Print a formatted audit result section."""
    print(f"\n{'=' * 70}")
    print(f"  Scenario: {scenario_name}")
    print(f"{'=' * 70}")

    if show_full_summary:
        print(report.summary())
    else:
        # Compact view
        print(f"  Layers monitored : {report.num_layers}")
        print(f"  Dead experts     : {report.dead_experts_count}")
        print(f"  Critical layers  : {len(report.critical_layers)}")
        print(f"  Has critical risk: {report.has_critical_risk}")

    # Top-5 riskiest layers
    ranked = report.layers_by_risk()
    if ranked:
        print(f"\n  Top-5 riskiest layers:")
        for layer_name, score in ranked[:5]:
            bar = "█" * int(score * 20)
            rr  = report.layer_risk(layer_name)
            lvl = rr.risk_level.value.upper() if rr else "?"
            print(f"    {layer_name:<40}  {score:.4f}  {bar:<20}  [{lvl}]")

    # Gradient-starved experts
    starved = report.gradient_starved_experts(threshold=0.01)
    if starved:
        print(f"\n  Gradient-starved experts: {len(starved)}")
        for layer_name, eid, norm in starved[:5]:
            print(f"    expert {eid:>3d} | norm={norm:.6f} | {layer_name}")
        if len(starved) > 5:
            print(f"    ... and {len(starved) - 5} more")
    else:
        print(f"\n  No gradient-starved experts detected.")

    # Dead experts
    dead = report.dead_experts()
    if dead:
        print(f"\n  Dead experts ({len(dead)} total):")
        for layer_name, eid in dead[:5]:
            print(f"    expert {eid:>3d} in {layer_name}")
        if len(dead) > 5:
            print(f"    ... and {len(dead) - 5} more")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_audit() -> None:
    print("=" * 70)
    print("  MoEWatch v0.2.0 — DeepSeek-MoE Offline Audit")
    print(f"  {NUM_LAYERS} layers | {NUM_ROUTED} routed + {NUM_SHARED} shared experts | top-{TOP_K}")
    print(f"  MoE layers: every {MOE_LAYER_FREQ}rd layer")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Build model and shared config
    # ------------------------------------------------------------------
    torch.manual_seed(0)
    model  = FakeDeepSeekModel(num_layers=NUM_LAYERS, hidden_dim=HIDDEN_DIM)
    config = make_audit_config()

    print(f"\n[Model] {NUM_LAYERS} total layers | {model.moe_layer_count()} MoE layers")
    print(f"[Model] {NUM_ROUTED} routed experts + {NUM_SHARED} shared experts per MoE block")

    # ------------------------------------------------------------------
    # Scenario A — Healthy checkpoint
    # ------------------------------------------------------------------
    print("\n[Scenario A] Setting healthy routing ...")
    model.set_healthy_routing()
    loader_a = make_val_loader(num_batches=40, batch_size=4, seq_len=32)

    report_a = audit(
        model=model,
        dataloader=loader_a,
        num_batches=40,
        config=config,
        device="cpu",
    )
    print_audit_results("A — Healthy Checkpoint", report_a, show_full_summary=False)

    # ------------------------------------------------------------------
    # Scenario B — Partial collapse (mid-training)
    # ------------------------------------------------------------------
    print("\n[Scenario B] Injecting partial collapse (top-8 expert dominance) ...")
    model.set_partial_collapse(top_experts=8)
    loader_b = make_val_loader(num_batches=40, batch_size=4, seq_len=32)

    report_b = audit(
        model=model,
        dataloader=loader_b,
        num_batches=40,
        config=config,
        device="cpu",
    )
    print_audit_results("B — Partial Collapse (mid-training)", report_b, show_full_summary=False)

    # ------------------------------------------------------------------
    # Scenario C — Severe collapse (routing failure)
    # ------------------------------------------------------------------
    print("\n[Scenario C] Injecting severe collapse (single expert monopoly) ...")
    model.set_severe_collapse(dominant_expert=0)
    loader_c = make_val_loader(num_batches=40, batch_size=4, seq_len=32)

    report_c = audit(
        model=model,
        dataloader=loader_c,
        num_batches=40,
        config=config,
        device="cpu",
    )
    print_audit_results("C — Severe Collapse (routing failure)", report_c, show_full_summary=True)

    # ------------------------------------------------------------------
    # Cross-scenario comparison
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("  Cross-Scenario Comparison")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<35}  {'Scenario A':>12}  {'Scenario B':>12}  {'Scenario C':>12}")
    print(f"  {'-' * 35}  {'-' * 12}  {'-' * 12}  {'-' * 12}")

    def _top_risk(r):
        ranked = r.layers_by_risk()
        return f"{ranked[0][1]:.4f}" if ranked else "N/A"

    def _avg_risk(r):
        scores = [rr.risk_score for rr in r.risk_scores.values()]
        return f"{sum(scores)/len(scores):.4f}" if scores else "N/A"

    rows = [
        ("MoE layers monitored",  report_a.num_layers,         report_b.num_layers,         report_c.num_layers),
        ("Dead experts",          report_a.dead_experts_count,  report_b.dead_experts_count,  report_c.dead_experts_count),
        ("Critical layers",       len(report_a.critical_layers),len(report_b.critical_layers),len(report_c.critical_layers)),
        ("Starved experts",       len(report_a.gradient_starved_experts()), len(report_b.gradient_starved_experts()), len(report_c.gradient_starved_experts())),
        ("Top layer risk",        _top_risk(report_a),         _top_risk(report_b),         _top_risk(report_c)),
        ("Avg layer risk",        _avg_risk(report_a),         _avg_risk(report_b),         _avg_risk(report_c)),
    ]

    for label, a, b, c in rows:
        print(f"  {label:<35}  {str(a):>12}  {str(b):>12}  {str(c):>12}")

    # ------------------------------------------------------------------
    # Serialisation demo
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("  Serialisation")
    print(f"{'=' * 70}")

    # JSON export for each scenario
    for name, report, fname in [
        ("A (healthy)",  report_a, "deepseek_audit_A_healthy.json"),
        ("B (partial)",  report_b, "deepseek_audit_B_partial.json"),
        ("C (severe)",   report_c, "deepseek_audit_C_severe.json"),
    ]:
        report.to_json(fname)
        size_kb = os.path.getsize(fname) / 1024
        print(f"  Scenario {name:<15} → {fname}  ({size_kb:.1f} KB)")

    # DataFrame export (scenario C — most interesting)
    try:
        df = report_c.to_dataframe()
        print(f"\n  Scenario C dataframe: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"  Columns: {list(df.columns)}")
        if not df.empty:
            worst = df.sort_values("risk_score", ascending=False).head(3)
            print(f"\n  Top-3 highest-risk layers (dataframe):")
            for _, row in worst.iterrows():
                print(
                    f"    {row['layer_name']:<40}  "
                    f"risk={row['risk_score']:.4f}  "
                    f"level={row['risk_level']}"
                )
    except ImportError:
        print("\n  (pandas not installed — skipping to_dataframe() demo)")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("  Audit complete.")
    print("  MoEWatch successfully diagnosed all three checkpoint scenarios.")
    print("  Use report.to_json() to save results for CI/CD pipeline integration.")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_audit()
