# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# examples/audit_walkthrough.py
# =============================================================================
#
# Project      : MoEWatch
# Version      : v0.2.0
# Description  : Offline audit walkthrough — post-training diagnostic on a
#                saved model checkpoint using moewatch.audit().
#
#                This is the second major workflow in MoEWatch, complementary
#                to live monitoring. Use it when you want to:
#                  - Inspect a checkpoint you did not monitor during training
#                  - Run diagnostics before deployment
#                  - Compare a healthy baseline against a suspect checkpoint
#                  - Export findings as JSON for CI pipelines or dashboards
#
#                Covers:
#                  1. Building two model checkpoints: healthy vs. collapsed
#                  2. Running audit() on both with a shared validation DataLoader
#                  3. Reading the AuditReport: risk, entropy, expert health
#                  4. Querying dead and cold experts
#                  5. Side-by-side comparison across checkpoints
#                  6. Exporting findings to JSON and reading them back
#                  7. Using report.summary() for human-readable output
#
#                Design notes:
#                  - Both checkpoints are audited on the SAME fixed validation
#                    dataset — this is the only scientifically valid comparison.
#                  - Collapse is simulated via a gate bias (deterministic) rather
#                    than weight scaling (non-deterministic with random inputs).
#                  - with_backward=False is used throughout. The gradient-starvation
#                    hook (Tier 1) relies on register_full_backward_hook, which
#                    fires before weight .grad is accumulated in the backward graph.
#                    In an offline audit that means all expert grad norms are 0 —
#                    so this example focuses on the two signals that ARE reliable
#                    in an offline context: routing entropy (Tier 2) and expert
#                    utilisation / collapse detection.
#                  - audit() never modifies model weights. Safe to call on a model
#                    that continues training afterward.
#
#                No GPU or HuggingFace Trainer required.
#
# Author       : Abinesh N (@Abineshabee)
# Repository   : https://github.com/Abineshabee/MoEWatch
# License      : Apache 2.0
#
# Run
# ---
#   python examples/audit_walkthrough.py
#
# =============================================================================

from __future__ import annotations

import json
import math
import os
import tempfile
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud

from moewatch import WatchConfig, OutputMode
from moewatch._audit import audit

warnings.filterwarnings(
    "ignore",
    message="Full backward hook is firing",
    category=UserWarning,
)

# ── Formatting helpers ────────────────────────────────────────────────────────

SEP  = "─" * 60
SEP2 = "═" * 60

def _banner(title: str) -> None:
    print(f"\n{SEP2}")
    print(f"  {title}")
    print(SEP2)

def _row(label: str, value: object, width: int = 26) -> None:
    print(f"  {label:<{width}}: {value}")


# ── Model ─────────────────────────────────────────────────────────────────────
#
#  CheckpointMoE: 4 experts, top_k=2, standard sparse routing.
#
#  Collapse is simulated by adding a large positive bias to expert 0's gate
#  output logit. With bias=+4, expert 0 receives ~95% of all tokens regardless
#  of the input, giving normalised entropy ≈ 0.19 — firmly in the CRITICAL zone.
#
#  Using a bias (additive, input-independent) instead of weight scaling
#  (multiplicative, input-dependent) makes the collapse fully deterministic:
#  the same inputs always produce the same expert distribution, so the entropy
#  measurement is reproducible and predictable.
#
#  forward() accepts plain tensors OR (tensor, label) tuples — the latter
#  is the standard format produced by most DataLoaders. audit() handles
#  both automatically.

class CheckpointMoE(nn.Module):
    """Sparse MoE block with 4 experts and a bias-controllable router."""

    N_EXPERTS = 4
    TOP_K     = 2

    def __init__(self, d_model: int = 32):
        super().__init__()
        # bias=True so we can directly control each expert's base logit
        self.gate    = nn.Linear(d_model, self.N_EXPERTS, bias=True)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2, bias=False),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model, bias=False),
            )
            for _ in range(self.N_EXPERTS)
        ])
        nn.init.normal_(self.gate.weight, std=0.01)   # near-uniform routing initially
        nn.init.zeros_(self.gate.bias)                  # zero bias = no expert preference
        self.top_k = self.TOP_K

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, (list, tuple)):
            x = x[0]                    # accept (input, label) tuples from DataLoader
        B, S, D = x.shape
        xf       = x.reshape(B * S, D)

        logits           = self.gate(xf)
        weights, indices = torch.topk(
            F.softmax(logits, dim=-1), self.TOP_K, dim=-1
        )

        out = torch.zeros_like(xf)
        for k in range(self.TOP_K):
            for e in range(self.N_EXPERTS):
                mask = (indices[:, k] == e)
                if mask.any():
                    out[mask] += (
                        weights[mask, k].unsqueeze(-1) * self.experts[e](xf[mask])
                    )
        return out.reshape(B, S, D)

    def routing_entropy(self, xs: torch.Tensor) -> float:
        """Normalised Shannon entropy measured on a specific input tensor.

        Use the same xs that the DataLoader is built from so the number
        matches what audit() observes.
        """
        with torch.no_grad():
            xf = xs.reshape(-1, xs.shape[-1])
            p  = F.softmax(self.gate(xf), dim=-1).clamp(min=1e-9)
            H  = -(p * p.log()).sum(dim=-1).mean()
            return (H / math.log(self.N_EXPERTS)).item()

    def routing_fractions(self, xs: torch.Tensor) -> list[float]:
        """Mean token fraction received by each expert on xs."""
        with torch.no_grad():
            xf = xs.reshape(-1, xs.shape[-1])
            p  = F.softmax(self.gate(xf), dim=-1)
            return p.mean(dim=0).tolist()


# ── Section 1: Build two checkpoints ─────────────────────────────────────────

def section_1_build_checkpoints(
    xs_val: torch.Tensor,
) -> tuple[CheckpointMoE, CheckpointMoE]:
    _banner("1. Build two model checkpoints")
    print("""
  Two checkpoints represent snapshots at different training stages.

    checkpoint_A  — early training, routing still healthy.
                    Gate bias = 0 everywhere (no expert preference).
                    All four experts receive ≈25% of tokens.
                    Normalised entropy ≈ 1.00.

    checkpoint_B  — later training, routing has collapsed.
                    Expert 0's gate bias set to +4.0.
                    Expert 0 attracts ~95% of tokens; experts 1–3 starve.
                    Normalised entropy ≈ 0.19 (below CRITICAL threshold 0.40).

  Why bias instead of weight scaling?
    Bias is input-independent, so the collapse is deterministic:
    the same validation data always produces the same entropy.
    Weight scaling interacts with random inputs and gives unpredictable
    results depending on the batch, making the example hard to follow.

  In a real workflow these would be loaded from .pt files:
      model.load_state_dict(torch.load("checkpoint_A.pt"))
""")
    torch.manual_seed(42)

    # Checkpoint A — healthy
    ckpt_a = CheckpointMoE(d_model=32)

    # Checkpoint B — collapsed: bias +4.0 on expert 0
    ckpt_b = CheckpointMoE(d_model=32)
    ckpt_b.load_state_dict(ckpt_a.state_dict())   # identical base weights
    with torch.no_grad():
        ckpt_b.gate.bias[0] = 4.0                 # expert 0 always gets +4 logit

    # Measure entropy on the SAME fixed validation data used by audit()
    ent_a = ckpt_a.routing_entropy(xs_val)
    ent_b = ckpt_b.routing_entropy(xs_val)
    fra_a = ckpt_a.routing_fractions(xs_val)
    fra_b = ckpt_b.routing_fractions(xs_val)

    print(f"  Router probability entropy (same {xs_val.shape[0]}-sample set used by audit()):\n")
    print(f"  checkpoint_A  prob_entropy = {ent_a:.4f}  fractions = {[f'{f:.3f}' for f in fra_a]}")
    print(f"  checkpoint_B  prob_entropy = {ent_b:.4f}  fractions = {[f'{f:.3f}' for f in fra_b]}")
    print()
    print("  Note: audit() also reports *empirical* routing entropy from")
    print("  actual top-k selections, not raw softmax probabilities.")
    print("  With top_k=2 and expert 0 winning every slot for checkpoint_B,")
    print("  empirical utilisation ≈ [1.0, 0.0, 0.0, 0.0] → entropy ≈ 0.0.")
    print("  Both figures are correct; they measure different things.")
    print(f"\n  Both: {CheckpointMoE.N_EXPERTS} experts, top_k={CheckpointMoE.TOP_K}, d_model=32")
    return ckpt_a, ckpt_b


# ── Section 2: Build a shared validation DataLoader ───────────────────────────

def section_2_build_dataloader() -> tuple[tud.DataLoader, torch.Tensor]:
    _banner("2. Build a shared validation DataLoader")
    print("""
  Both checkpoints are audited on the SAME fixed dataset.
  This is essential for a fair comparison: any difference in the
  audit results is then solely due to the model weights, not the data.

  audit() accepts any standard PyTorch DataLoader. Batches can be:
    - A plain tensor:             x  (shape [B, S, D])
    - A (tensor, label) tuple:   (x, y)
    - A dict:                    {"input_ids": x, ...}

  Here we use (tensor, label) tuples — the most common format.
  audit() automatically extracts the input tensor and ignores the label.
""")
    torch.manual_seed(0)
    xs = torch.randn(120, 6, 32)   # 120 samples, seq_len=6, d_model=32
    ys = torch.randint(0, 4, (120,))
    dataset = tud.TensorDataset(xs, ys)
    loader  = tud.DataLoader(dataset, batch_size=10, shuffle=False)

    print(f"  Dataset size   : {len(dataset)} samples")
    print(f"  Batch size     : 10")
    print(f"  Total batches  : {len(loader)}")
    print(f"  num_batches=12 — audit will consume all 12 batches")
    return loader, xs


# ── Section 3: Run audit() on both checkpoints ───────────────────────────────

def section_3_run_audit(
    ckpt_a: CheckpointMoE,
    ckpt_b: CheckpointMoE,
    loader: tud.DataLoader,
) -> tuple:
    _banner("3. Run audit() on both checkpoints")
    print("""
  audit(model, dataloader, num_batches, config, with_backward)

    model         — model to inspect (weights NEVER changed by audit)
    dataloader    — validation data source (same loader for both here)
    num_batches   — how many batches to process
    config        — WatchConfig with your detection thresholds
    with_backward — set False for routing-only analysis (entropy + collapse).
                    The Tier-1 gradient-starvation signal requires a live
                    training loop (backward hooks read .grad before it is
                    accumulated in a short offline audit), so we use False
                    here to keep the output accurate and reproducible.

  WatchConfig thresholds used:
    entropy_warn     = 0.70   (WARNING if norm_entropy < 0.70)
    entropy_critical = 0.40   (CRITICAL if norm_entropy < 0.40)
    cold_steps_limit = 5      (expert declared DEAD after 5 cold steps)
""")
    config = WatchConfig(
        output           = OutputMode.SILENT,
        entropy_warn     = 0.70,
        entropy_critical = 0.40,
        cold_steps_limit = 5,
        stats_window     = 15,
    )

    print("  Auditing checkpoint_A ... ", end="", flush=True)
    report_a = audit(ckpt_a, loader, num_batches=12, config=config, with_backward=False)
    print(f"done  ({report_a.num_batches} batches, "
          f"{report_a.num_layers} layer(s) detected)")

    print("  Auditing checkpoint_B ... ", end="", flush=True)
    report_b = audit(ckpt_b, loader, num_batches=12, config=config, with_backward=False)
    print(f"done  ({report_b.num_batches} batches, "
          f"{report_b.num_layers} layer(s) detected)")

    return report_a, report_b, config


# ── Section 4: Read the AuditReport ──────────────────────────────────────────

def section_4_read_report(report_a, report_b) -> None:
    _banner("4. Read the AuditReport")
    print("""
  AuditReport exposes the full analysis as structured fields and
  convenience properties. Everything is computed and cached at audit
  time — no re-processing needed.
""")
    for label, report in [
        ("checkpoint_A  (healthy)",   report_a),
        ("checkpoint_B  (collapsed)", report_b),
    ]:
        print(f"  ── {label} {'─' * (35 - len(label))}")

        _row("  audit_datetime",  report.audit_datetime.strftime("%Y-%m-%d %H:%M:%S"))
        _row("  num_layers",      report.num_layers)
        _row("  num_batches",     report.num_batches)
        _row("  has_critical",    report.has_critical_risk)
        _row("  critical_layers", report.critical_layers or "none")

        print(f"\n    Layers by risk (descending):")
        for layer, score in report.layers_by_risk():
            rr    = report.layer_risk(layer)
            level = rr.risk_level.value if rr else "—"
            bar   = "█" * int(score * 20)
            print(f"      {layer:<16}  [{bar:<20}]  {score:.4f}  {level}")

        # checkpoint_A shows HIGH even with entropy ≈ 1.0.
        # Reason: the fused risk score is 60% Tier-1 (gradient starvation).
        # With_backward=False, Tier-1 is unavailable and contributes a
        # non-zero baseline — raising the fused score into HIGH territory.
        # The entropy and collapse signals (Tiers 2 & 3) are healthy.
        # checkpoint_B is CRITICAL because its near-zero entropy dominates.

        print(f"\n    Entropy:")
        for layer, er in report.entropy_results.items():
            rr = report.layer_risk(layer)
            _row(f"      {layer}  norm_entropy",   f"{er.normalized_entropy:.4f}")
            _row(f"      {layer}  trend",          er.trend)
            _row(f"      {layer}  drift_detected", er.drift_detected)
            _row(f"      {layer}  drop_rate",      f"{er.drop_rate:.4f}")
            _row(f"      {layer}  risk_level",     rr.risk_level.value if rr else "—")
        print()


# ── Section 5: Expert health ──────────────────────────────────────────────────

def section_5_expert_health(report_a, report_b) -> None:
    _banner("5. Expert health — utilisation and collapse states")
    print("""
  CollapseDetector tracks each expert's status through a state machine:

    UNKNOWN → HEALTHY  when utilisation >= cold_threshold
    HEALTHY → COLD     when utilisation drops below cold_threshold
    COLD    → DEAD     after cold_steps_limit consecutive COLD steps

  In a short audit (12 batches) experts may not progress all the way
  to DEAD even when their utilisation is near zero — they show as COLD.
  This is expected: DEAD is a conservative temporal criterion designed
  for live monitoring over many training steps.

  The utilisation figures here are what actually drive the risk score:
  experts with near-zero utilisation push the fused risk toward CRITICAL.
""")
    for label, report in [
        ("checkpoint_A  (healthy)",   report_a),
        ("checkpoint_B  (collapsed)", report_b),
    ]:
        print(f"  ── {label}")
        for layer, col in report.collapse_results.items():
            print(f"\n    layer: {layer}")
            _row("    load_imbalance_ratio", f"{col.load_imbalance_ratio:.2f}x")
            _row("    num_dead_experts",     col.num_dead_experts)
            _row("    num_cold_experts",     col.num_cold_experts)
            print(f"\n    Expert states:")
            print(f"      {'id':>3}  {'status':<10}  {'utilisation':>12}  {'cold_steps':>11}")
            print(f"      {'─'*44}")
            for eid, state in col.expert_states.items():
                bar = {"HEALTHY": "■", "COLD": "□", "DEAD": "✗", "UNKNOWN": "?"}.get(
                    state.status.value.upper(), "?"
                )
                print(
                    f"      {eid:>3}  {bar} {state.status.value:<8}  "
                    f"{state.utilization:>12.4f}  "
                    f"{state.consecutive_cold_steps:>11}"
                )
        print()


# ── Section 6: Side-by-side comparison ───────────────────────────────────────

def section_6_comparison(report_a, report_b) -> None:
    _banner("6. Side-by-side comparison across checkpoints")
    print("""
  A comparison table makes it easy to spot exactly what changed.
  Because both reports were produced from the same validation data,
  every difference is attributable to the model weights alone.
""")
    shared = sorted(
        set(report_a.entropy_results) & set(report_b.entropy_results)
    )

    print(f"  {'layer':<16}  {'metric':<22}  {'ckpt_A':>10}  {'ckpt_B':>10}  delta")
    print(f"  {SEP}")

    for layer in shared:
        er_a = report_a.entropy_results[layer]
        er_b = report_b.entropy_results[layer]
        rr_a = report_a.layer_risk(layer)
        rr_b = report_b.layer_risk(layer)
        ca   = report_a.collapse_results.get(layer)
        cb   = report_b.collapse_results.get(layer)

        rows = [
            ("norm_entropy",
             f"{er_a.normalized_entropy:.4f}",
             f"{er_b.normalized_entropy:.4f}",
             er_b.normalized_entropy - er_a.normalized_entropy),
            ("drift_detected",
             str(er_a.drift_detected),
             str(er_b.drift_detected),
             None),
            ("risk_score",
             f"{rr_a.risk_score:.4f}" if rr_a else "—",
             f"{rr_b.risk_score:.4f}" if rr_b else "—",
             (rr_b.risk_score - rr_a.risk_score) if rr_a and rr_b else None),
            ("risk_level",
             rr_a.risk_level.value if rr_a else "—",
             rr_b.risk_level.value if rr_b else "—",
             None),
            ("cold_experts",
             str(ca.num_cold_experts) if ca else "—",
             str(cb.num_cold_experts) if cb else "—",
             (cb.num_cold_experts - ca.num_cold_experts) if ca and cb else None),
            ("load_imbalance",
             f"{ca.load_imbalance_ratio:.2f}x" if ca else "—",
             f"{cb.load_imbalance_ratio:.2f}x" if cb else "—",
             None),
        ]

        for metric, va, vb, delta in rows:
            delta_str = (
                f"{delta:+.4f}" if isinstance(delta, float) else
                f"{delta:+d}"   if isinstance(delta, int)   else
                ""
            )
            print(f"  {layer:<16}  {metric:<22}  {va:>10}  {vb:>10}  {delta_str}")
        print()


# ── Section 7: Export to JSON ─────────────────────────────────────────────────

def section_7_export_json(report_b) -> None:
    _banner("7. Export to JSON")
    print("""
  report.to_json(path) writes the full AuditReport as a JSON file.

  Use this for:
    - CI gates ("fail if any layer is CRITICAL")
    - Dashboard ingestion
    - Comparing audit results across runs in a script
""")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as fh:
        json_path = fh.name

    try:
        report_b.to_json(json_path)
        size = os.path.getsize(json_path)
        print(f"  Written : {json_path}")
        print(f"  Size    : {size:,} bytes")

        with open(json_path) as fh:
            data = json.load(fh)

        print(f"\n  Top-level keys:")
        for key in data.keys():
            val     = data[key]
            summary = (
                f"{len(val)} layer(s)" if isinstance(val, dict) else
                f"{len(val)} item(s)"  if isinstance(val, list) else
                str(val)[:60]
            )
            print(f"    {key:<28} → {summary}")

        # Show key entropy field from JSON
        for layer, er in data.get("entropy_results", {}).items():
            print(f"\n  entropy_results[\"{layer}\"]:")
            for k in ("normalized_entropy", "trend", "drift_detected", "drop_rate"):
                print(f"    {k:<20} = {er.get(k)}")

        # Example CI gate
        critical = data.get("critical_layers", [])
        print(f"\n  CI gate example:")
        print(f"    critical_layers = {critical}")
        if critical:
            print(f"    → would FAIL: critical layers detected — block deployment")
        else:
            print(f"    → PASS: no critical layers")

    finally:
        os.unlink(json_path)


# ── Section 8: Human-readable summary ────────────────────────────────────────

def section_8_summary(report_a, report_b) -> None:
    _banner("8. Human-readable summary — report.summary()")
    print("""
  report.summary() returns a formatted multi-line string suitable
  for logging or printing to a terminal.
""")
    print("  ── checkpoint_A  (healthy) ──")
    for line in report_a.summary().splitlines():
        print(f"  {line}")

    print("\n  ── checkpoint_B  (collapsed) ──")
    for line in report_b.summary().splitlines():
        print(f"  {line}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(SEP2)
    print("  MoEWatch — Offline Audit Walkthrough")
    print("  checkpoint → DataLoader → audit() → AuditReport")
    print("  → compare → export → summary")
    print(SEP2)

    # Build the fixed validation dataset FIRST — used by both checkpoints
    # for entropy measurement AND for the DataLoader passed to audit().
    # This ensures every number in the example is measured on identical data.
    loader, xs_val = section_2_build_dataloader()
    ckpt_a, ckpt_b  = section_1_build_checkpoints(xs_val)

    report_a, report_b, _ = section_3_run_audit(ckpt_a, ckpt_b, loader)

    section_4_read_report(report_a, report_b)
    section_5_expert_health(report_a, report_b)
    section_6_comparison(report_a, report_b)
    section_7_export_json(report_b)
    section_8_summary(report_a, report_b)

    print(f"\n{SEP2}")
    print("  Done.")
    print("  For live monitoring during training, see:")
    print("  examples/live_monitoring_walkthrough.py")
    print(SEP2)


if __name__ == "__main__":
    main()
