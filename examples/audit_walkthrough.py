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
#                  2. Running audit() on both with a validation DataLoader
#                  3. Reading the AuditReport: risk, entropy, expert health
#                  4. Querying dead experts and gradient-starved experts
#                  5. Side-by-side comparison across checkpoints
#                  6. Exporting findings to JSON and reading them back
#                  7. Using report.summary() for human-readable output
#
#                Key design note:
#                  audit() attaches hooks, runs forward (and optionally
#                  backward) passes, then removes all hooks. Model weights
#                  are never modified. It is safe to call on a model that
#                  will continue training afterward.
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

def _row(label: str, value: object, width: int = 22) -> None:
    print(f"  {label:<{width}}: {value}")


# ── Model ─────────────────────────────────────────────────────────────────────
#
#  CheckpointMoE: 6 experts, top_k=2, standard sparse routing.
#
#  "Collapse" is simulated by multiplying the gate weight for expert 0 by a
#  large scalar. This concentrates the softmax mass on expert 0, so tokens
#  almost never reach experts 1–5, which then become gradient-starved.
#
#  forward() accepts plain tensors OR (tensor, label) tuples — the latter
#  is the standard format produced by most DataLoaders. audit() handles
#  both automatically.

class CheckpointMoE(nn.Module):
    """Sparse MoE block with 6 experts. The gate is the monitored router."""

    N_EXPERTS = 6
    TOP_K     = 2

    def __init__(self, d_model: int = 48):
        super().__init__()
        self.gate    = nn.Linear(d_model, self.N_EXPERTS, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2, bias=False),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model, bias=False),
            )
            for _ in range(self.N_EXPERTS)
        ])
        nn.init.normal_(self.gate.weight, std=0.02)
        self.top_k = self.TOP_K

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        xf      = x.reshape(B * S, D)

        logits   = self.gate(xf)
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

    def norm_entropy(self) -> float:
        """Quick routing entropy sanity check (random batch, no grad)."""
        with torch.no_grad():
            x = torch.randn(64, self.gate.in_features)
            p = F.softmax(self.gate(x), dim=-1).clamp(min=1e-9)
            H = -(p * p.log()).sum(dim=-1).mean()
            return (H / math.log(self.N_EXPERTS)).item()


# ── Section 1: Build two checkpoints ─────────────────────────────────────────

def section_1_build_checkpoints() -> tuple[CheckpointMoE, CheckpointMoE]:
    _banner("1. Build two model checkpoints")
    print("""
  We create two model states that represent snapshots at different
  points in a hypothetical training run:

    checkpoint_A  — early training, routing still healthy.
                    Gate weights are near-zero (uniform softmax).
                    Normalised entropy ≈ 1.0.

    checkpoint_B  — later training, routing has collapsed.
                    Expert 0's gate weight is scaled up ×30,
                    concentrating nearly all tokens onto one expert.
                    Normalised entropy drops to ≈ 0.20.

  In a real workflow these would be loaded from .pt files:
      model.load_state_dict(torch.load("checkpoint_A.pt"))
""")
    torch.manual_seed(0)

    # Checkpoint A — healthy
    ckpt_a = CheckpointMoE(d_model=48)

    # Checkpoint B — collapsed: expert 0's row ×30
    ckpt_b = CheckpointMoE(d_model=48)
    ckpt_b.load_state_dict(ckpt_a.state_dict())   # same base weights
    with torch.no_grad():
        ckpt_b.gate.weight[0] *= 30.0

    print(f"  checkpoint_A  norm_entropy = {ckpt_a.norm_entropy():.4f}  (≈1.0 = uniform)")
    print(f"  checkpoint_B  norm_entropy = {ckpt_b.norm_entropy():.4f}  (low  = collapsed)")
    print(f"\n  Both checkpoints: {CheckpointMoE.N_EXPERTS} experts, "
          f"top_k={CheckpointMoE.TOP_K}, d_model=48")
    return ckpt_a, ckpt_b


# ── Section 2: Build a validation DataLoader ─────────────────────────────────

def section_2_build_dataloader() -> tud.DataLoader:
    _banner("2. Build a validation DataLoader")
    print("""
  audit() accepts any standard PyTorch DataLoader. Batches can be:
    - A plain tensor:             x  (shape [B, S, D])
    - A (tensor, label) tuple:   (x, y)
    - A dict:                    {"input_ids": x, "attention_mask": ...}

  Here we use (tensor, label) tuples — the most common format.
  audit() automatically extracts the input tensor and ignores the label.
""")
    # 120 samples, each (seq_len=6, d_model=48), with a dummy integer label
    xs = torch.randn(120, 6, 48)
    ys = torch.randint(0, 4, (120,))
    dataset = tud.TensorDataset(xs, ys)
    loader  = tud.DataLoader(dataset, batch_size=12, shuffle=False)

    print(f"  Dataset size   : {len(dataset)} samples")
    print(f"  Batch size     : 12")
    print(f"  Total batches  : {len(loader)}")
    print(f"  num_batches=10 — audit will consume the first 10 batches")
    return loader


# ── Section 3: Run audit() on both checkpoints ───────────────────────────────

def section_3_run_audit(
    ckpt_a: CheckpointMoE,
    ckpt_b: CheckpointMoE,
    loader: tud.DataLoader,
) -> tuple:
    _banner("3. Run audit() on both checkpoints")
    print("""
  audit(model, dataloader, num_batches, config, with_backward)

    model        — the model to inspect (weights unchanged after audit)
    dataloader   — validation data source
    num_batches  — how many batches to run (10 is enough for diagnostics)
    config       — WatchConfig with your thresholds
    with_backward — run a proxy backward pass so gradient starvation
                    analysis (Tier 1) is populated. Slightly slower but
                    gives a complete picture. Set False for routing-only.

  Both audits use the same config and the same DataLoader — the only
  difference is the model checkpoint being inspected.
""")
    config = WatchConfig(
        output           = OutputMode.SILENT,
        entropy_warn     = 0.70,
        entropy_critical = 0.40,
        cold_steps_limit = 5,
        stats_window     = 20,
    )

    print("  Auditing checkpoint_A ... ", end="", flush=True)
    report_a = audit(ckpt_a, loader, num_batches=10, config=config, with_backward=True)
    print(f"done  ({report_a.num_batches} batches)")

    print("  Auditing checkpoint_B ... ", end="", flush=True)
    report_b = audit(ckpt_b, loader, num_batches=10, config=config, with_backward=True)
    print(f"done  ({report_b.num_batches} batches)")

    return report_a, report_b, config


# ── Section 4: Read the AuditReport ──────────────────────────────────────────

def section_4_read_report(report_a, report_b) -> None:
    _banner("4. Read the AuditReport")
    print("""
  AuditReport exposes the full analysis as structured fields and
  convenience properties. No re-processing needed — everything is
  computed and cached at audit time.
""")

    for label, report in [("checkpoint_A (healthy)", report_a),
                           ("checkpoint_B (collapsed)", report_b)]:
        print(f"  ── {label} {'─' * (36 - len(label))}")

        # Top-level metadata
        _row("  audit_datetime", report.audit_datetime.strftime("%Y-%m-%d %H:%M:%S"))
        _row("  num_layers",     report.num_layers)
        _row("  num_batches",    report.num_batches)
        _row("  has_critical",   report.has_critical_risk)
        _row("  critical_layers", report.critical_layers or "none")

        # Per-layer risk: layers_by_risk() returns sorted (layer, score) pairs
        print(f"\n    Layers by risk (descending):")
        for layer, score in report.layers_by_risk():
            rr    = report.layer_risk(layer)
            level = rr.risk_level.value if rr else "—"
            bar   = "█" * int(score * 20)
            print(f"      {layer:<16}  [{bar:<20}]  {score:.4f}  {level}")

        # Entropy
        print(f"\n    Entropy results:")
        for layer, er in report.entropy_results.items():
            _row(f"      {layer} norm_entropy", f"{er.normalized_entropy:.4f}")
            _row(f"      {layer} drift_detected", er.drift_detected)

        print()


# ── Section 5: Query dead and starved experts ─────────────────────────────────

def section_5_expert_health(report_a, report_b) -> None:
    _banner("5. Expert health — dead and gradient-starved experts")
    print("""
  Two signals track expert health. They use different definitions:

    dead_experts_count  (from gradient reports)
        Counts experts whose mean gradient norm < config.dead_threshold
        (default 0.5) across the audit window. In a sparse top-k MoE,
        experts not selected in any of the audit batches will have
        norm=0 and are counted here. This signal is most meaningful
        with many batches or after calibrating dead_threshold to
        your model's typical gradient norm scale.

    dead_experts()  (from CollapseDetector state machine)
        Returns (layer, expert_id) only after cold_steps_limit
        consecutive cold steps — a conservative temporal criterion
        designed for live monitoring. May return none in a short
        offline audit even when gradient norms are zero.

    gradient_starved_experts(threshold)
        The most flexible query: returns (layer, expert_id, grad_norm)
        for every expert below a threshold you choose. Compare across
        checkpoints with the same threshold to see which experts
        lost gradient signal between two training stages.
""")

    for label, report in [("checkpoint_A", report_a), ("checkpoint_B", report_b)]:
        # Show all per-expert norms so the reader can see the actual values
        all_norms = sorted(
            report.gradient_starved_experts(threshold=1.0),
            key=lambda x: x[1]   # sort by expert_id
        )

        print(f"  ── {label}")
        print(f"    dead_experts_count (grad norm < dead_threshold) : {report.dead_experts_count}")
        print(f"    dead_experts()     (CollapseDetector state)     : {report.dead_experts() or 'none'}")

        if all_norms:
            print(f"    Per-expert gradient norms:")
            print(f"      {'expert':<8}  {'layer':<16}  grad_norm")
            print(f"      {'─' * 42}")
            for lname, eid, norm in all_norms:
                bar = "▓" * min(20, int(norm * 200))
                print(f"      {eid:<8}  {lname:<16}  {norm:.6f}  {bar}")
        print()


# ── Section 6: Side-by-side comparison ───────────────────────────────────────

def section_6_comparison(report_a, report_b) -> None:
    _banner("6. Side-by-side comparison across checkpoints")
    print("""
  A comparison table makes it easy to see exactly what changed
  between two checkpoints. The key diagnostic signals are:
    - normalised entropy (drops under collapse)
    - dead expert count  (rises under collapse)
    - drift_detected     (True when CUSUM detects distribution shift)
    - risk level         (classification from the fused score)
""")

    # Gather metrics for shared layers
    layers_a = set(report_a.entropy_results.keys())
    layers_b = set(report_b.entropy_results.keys())
    shared   = sorted(layers_a & layers_b)

    print(f"  {'layer':<16}  {'metric':<22}  {'ckpt_A':>10}  {'ckpt_B':>10}  delta")
    print(f"  {SEP}")

    for layer in shared:
        er_a = report_a.entropy_results[layer]
        er_b = report_b.entropy_results[layer]
        rr_a = report_a.layer_risk(layer)
        rr_b = report_b.layer_risk(layer)

        rows = [
            ("norm_entropy",   f"{er_a.normalized_entropy:.4f}",
                               f"{er_b.normalized_entropy:.4f}",
                               er_b.normalized_entropy - er_a.normalized_entropy),
            ("drift_detected", str(er_a.drift_detected),
                               str(er_b.drift_detected),
                               None),
            ("risk_score",     f"{rr_a.risk_score:.4f}" if rr_a else "—",
                               f"{rr_b.risk_score:.4f}" if rr_b else "—",
                               (rr_b.risk_score - rr_a.risk_score) if rr_a and rr_b else None),
            ("risk_level",     rr_a.risk_level.value if rr_a else "—",
                               rr_b.risk_level.value if rr_b else "—",
                               None),
            ("dead_experts",   str(report_a.dead_experts_count),
                               str(report_b.dead_experts_count),
                               report_b.dead_experts_count - report_a.dead_experts_count),
        ]

        for metric, va, vb, delta in rows:
            delta_str = (
                f"{delta:+.4f}" if isinstance(delta, float) else
                f"{delta:+d}"   if isinstance(delta, int) else
                ""
            )
            print(f"  {layer:<16}  {metric:<22}  {va:>10}  {vb:>10}  {delta_str}")
        print()


# ── Section 7: Export to JSON ─────────────────────────────────────────────────

def section_7_export_json(report_b) -> None:
    _banner("7. Export to JSON")
    print("""
  report.to_json(path) writes the full AuditReport as a JSON file.
  The structure is:
    {
      "model_name":      "...",
      "timestamp":       ...,
      "num_batches":     ...,
      "entropy_results": { "<layer>": { ... } },
      "collapse_results":{ "<layer>": { ... } },
      "gradient_results":{ "<layer>": [ ... ] },
      "risk_scores":     { "<layer>": { ... } },
      "critical_layers": [...],
      "dead_experts_count": ...
    }

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
        print(f"  Written: {json_path}")
        print(f"  Size   : {size:,} bytes")

        # Read back and inspect top-level keys
        with open(json_path) as fh:
            data = json.load(fh)

        print(f"\n  Top-level keys in JSON export:")
        for key in data.keys():
            val = data[key]
            summary = (
                f"{len(val)} layer(s)" if isinstance(val, dict) else
                f"{len(val)} item(s)"  if isinstance(val, list) else
                str(val)[:60]
            )
            print(f"    {key:<24} → {summary}")

        # Example CI gate: fail if any layer is CRITICAL
        critical = data.get("critical_layers", [])
        print(f"\n  CI gate example:")
        print(f"    critical_layers = {critical}")
        if critical:
            print(f"    → would FAIL: critical layers detected in checkpoint_B")
        else:
            print(f"    → PASS: no critical layers (risk is HIGH, not CRITICAL)")

    finally:
        os.unlink(json_path)


# ── Section 8: Human-readable summary ────────────────────────────────────────

def section_8_summary(report_a, report_b) -> None:
    _banner("8. Human-readable summary — report.summary()")
    print("""
  report.summary() returns a formatted multi-line string suitable
  for logging or printing to a terminal. Covers: metadata, risk
  overview with ASCII bar chart, critical layers, dead experts,
  gradient starvation, and cross-layer correlation status.
""")
    print("  ── checkpoint_A (healthy) ──")
    for line in report_a.summary().splitlines():
        print(f"  {line}")

    print("\n  ── checkpoint_B (collapsed) ──")
    for line in report_b.summary().splitlines():
        print(f"  {line}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(SEP2)
    print("  MoEWatch — Offline Audit Walkthrough")
    print("  checkpoint → DataLoader → audit() → AuditReport")
    print("  → compare → export → summary")
    print(SEP2)

    ckpt_a, ckpt_b  = section_1_build_checkpoints()
    loader          = section_2_build_dataloader()
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
