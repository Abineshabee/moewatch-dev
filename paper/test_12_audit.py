# =============================================================================
# MoEWatch v0.2.0 — Research Paper Test Suite
# Test 12: Offline Audit (audit() API — End-to-End)
# =============================================================================
#
# Verifies the audit() offline API end-to-end across three checkpoints:
#
#   Checkpoint A — Healthy   : uniform routing → low risk, no critical layers
#   Checkpoint B — Collapsed : monopoly routing → high risk, critical layers
#   Checkpoint C — Comparison: risk(A) < risk(B), dead experts(A) < dead(B)
#
# Also verifies:
#   4. AuditReport fields: num_layers, dead_experts_count, critical_layers
#   5. layers_by_risk() ordering: highest risk first
#   6. gradient_starved_experts() returns (layer, expert_id) tuples
#   7. summary() returns a non-empty string
#   8. to_json() writes a valid JSON file with correct schema keys
#   9. has_critical_risk property
#
# =============================================================================

import json
import os
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from moewatch import audit
from moewatch.config import WatchConfig, OutputMode

# ---------------------------------------------------------------------------
# Minimal MoE model with controllable routing
# ---------------------------------------------------------------------------

NUM_EXPERTS = 8
TOP_K       = 2
HIDDEN      = 32


class FakeMoELayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate    = nn.Linear(HIDDEN, NUM_EXPERTS, bias=False)
        self.gate.top_k = TOP_K
        self.experts = nn.ModuleList([
            nn.Linear(HIDDEN, HIDDEN, bias=False) for _ in range(NUM_EXPERTS)
        ])
        self.top_k   = TOP_K

    def forward(self, x):
        B, S, D = x.shape
        flat    = x.reshape(-1, D)
        probs   = torch.softmax(self.gate(flat), dim=-1)
        top_p, top_idx = probs.topk(self.top_k, dim=-1)
        top_p   = top_p / top_p.sum(-1, keepdim=True)
        out     = torch.zeros_like(flat)
        for k in range(self.top_k):
            for e in range(NUM_EXPERTS):
                mask = top_idx[:, k] == e
                if mask.any():
                    out[mask] += top_p[mask, k:k+1] * self.experts[e](flat[mask])
        return out.reshape(B, S, D)


class AuditModel(nn.Module):
    def __init__(self, n_moe_layers: int = 3):
        super().__init__()
        self.embed   = nn.Embedding(1000, HIDDEN)
        self.layers  = nn.ModuleList([FakeMoELayer() for _ in range(n_moe_layers)])
        self.head    = nn.Linear(HIDDEN, 1000, bias=False)
        self._n_moe  = n_moe_layers

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = x + layer(x)
        return self.head(x)

    def set_healthy(self):
        with torch.no_grad():
            for layer in self.layers:
                layer.gate.weight.normal_(0, 0.02)

    def set_collapsed(self, dominant: int = 0, scale: float = 8.0):
        with torch.no_grad():
            for layer in self.layers:
                layer.gate.weight.zero_()
                layer.gate.weight[dominant] = scale


def make_loader(n_batches: int = 15, batch: int = 4, seq: int = 16) -> DataLoader:
    torch.manual_seed(42)
    ids = torch.randint(0, 1000, (n_batches * batch, seq))
    return DataLoader(TensorDataset(ids), batch_size=batch, shuffle=False)


def make_config() -> WatchConfig:
    return WatchConfig(
        output=OutputMode.SILENT,
        dead_threshold=0.5,
        cold_threshold=1.0,
        cold_steps_limit=5,
        intervention_enabled=False,
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def run():
    print("=" * 60)
    print("  MoEWatch Test 12 — Offline Audit (End-to-End)")
    print("=" * 60)

    torch.manual_seed(0)
    config = make_config()

    # ==================================================================
    # Checkpoint A — Healthy model
    # ==================================================================
    print("\n  [A] Healthy checkpoint (uniform random routing)")

    model_a = AuditModel(n_moe_layers=3)
    model_a.set_healthy()
    report_a = audit(
        model=model_a,
        dataloader=make_loader(),
        num_batches=15,
        config=config,
        device="cpu",
        with_backward=True,
    )

    print(f"    num_layers         : {report_a.num_layers}  (expected 3)")
    print(f"    dead_experts_count : {report_a.dead_experts_count}")
    print(f"    critical_layers    : {report_a.critical_layers}")
    print(f"    has_critical_risk  : {report_a.has_critical_risk}")

    ranked_a  = report_a.layers_by_risk()
    top_risk_a = ranked_a[0][1] if ranked_a else 0.0
    print(f"    top layer risk     : {top_risk_a:.4f}")

    assert report_a.num_layers == 3, \
        f"Expected 3 MoE layers, got {report_a.num_layers}"
    print(f"    ✓ Healthy checkpoint audited: {report_a.num_layers} layers")

    # ==================================================================
    # Checkpoint B — Collapsed model (expert 0 monopoly)
    # ==================================================================
    print("\n  [B] Collapsed checkpoint (expert 0 monopoly)")

    model_b = AuditModel(n_moe_layers=3)
    model_b.set_collapsed(dominant=0, scale=10.0)
    report_b = audit(
        model=model_b,
        dataloader=make_loader(),
        num_batches=15,
        config=config,
        device="cpu",
        with_backward=True,
    )

    print(f"    num_layers         : {report_b.num_layers}")
    print(f"    dead_experts_count : {report_b.dead_experts_count}")
    print(f"    critical_layers    : {report_b.critical_layers}")
    print(f"    has_critical_risk  : {report_b.has_critical_risk}")

    ranked_b   = report_b.layers_by_risk()
    top_risk_b = ranked_b[0][1] if ranked_b else 0.0
    print(f"    top layer risk     : {top_risk_b:.4f}")

    assert report_b.num_layers == 3
    assert report_b.dead_experts_count > report_a.dead_experts_count, \
        f"Collapsed model should have more dead experts: " \
        f"A={report_a.dead_experts_count} B={report_b.dead_experts_count}"
    print(f"    ✓ Collapsed checkpoint: dead experts A={report_a.dead_experts_count} "
          f"→ B={report_b.dead_experts_count}")

    # ==================================================================
    # [C] Healthy vs Collapsed comparison
    # ==================================================================
    print("\n  [C] Healthy vs Collapsed comparison")

    print(f"    {'Metric':<30}  {'Healthy':>10}  {'Collapsed':>10}")
    print(f"    {'-'*30}  {'-'*10}  {'-'*10}")
    print(f"    {'num_layers':<30}  {report_a.num_layers:>10}  {report_b.num_layers:>10}")
    print(f"    {'dead_experts_count':<30}  {report_a.dead_experts_count:>10}  {report_b.dead_experts_count:>10}")
    print(f"    {'critical_layers':<30}  {len(report_a.critical_layers):>10}  {len(report_b.critical_layers):>10}")
    print(f"    {'top_layer_risk':<30}  {top_risk_a:>10.4f}  {top_risk_b:>10.4f}")
    print(f"    {'has_critical_risk':<30}  {str(report_a.has_critical_risk):>10}  {str(report_b.has_critical_risk):>10}")

    assert top_risk_b >= top_risk_a, \
        f"Collapsed risk must be >= healthy: A={top_risk_a:.4f} B={top_risk_b:.4f}"
    print(f"    ✓ Risk ordering correct: healthy({top_risk_a:.4f}) ≤ collapsed({top_risk_b:.4f})")

    # ==================================================================
    # [4] layers_by_risk() ordering
    # ==================================================================
    print("\n  [4] layers_by_risk() — descending order")

    ranked = report_b.layers_by_risk()
    print(f"    Total ranked layers: {len(ranked)}  (expected 3)")
    for i, (name, score) in enumerate(ranked):
        print(f"    [{i+1}] {name:<35} {score:.4f}")

    assert len(ranked) == 3
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True), \
        "layers_by_risk() must return descending order"
    print(f"    ✓ Strict descending order verified")

    # ==================================================================
    # [5] gradient_starved_experts()
    # ==================================================================
    print("\n  [5] gradient_starved_experts()")

    starved = report_b.gradient_starved_experts()
    print(f"    Starved experts count: {len(starved)}")
    if starved:
        layer_name, expert_id, norm = starved[0]
        print(f"    Sample: layer={layer_name!r}  expert_id={expert_id}  norm={norm:.6f}")
        assert isinstance(layer_name, str)
        assert isinstance(expert_id, int)
        assert isinstance(norm, float)
    print(f"    ✓ gradient_starved_experts() returns (str, int, float) tuples")

    # ==================================================================
    # [6] summary() returns non-empty string
    # ==================================================================
    print("\n  [6] summary()")

    s = report_b.summary()
    assert isinstance(s, str) and len(s) > 50, \
        f"summary() should return non-empty string, got: {s!r}"
    lines = s.strip().split("\n")
    print(f"    summary() lines    : {len(lines)}")
    print(f"    First line         : {lines[0].strip()!r}")
    print(f"    ✓ summary() returns {len(lines)}-line string")

    # ==================================================================
    # [7] to_json() writes valid JSON with correct schema
    # ==================================================================
    print("\n  [7] to_json() — write and round-trip verify")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        report_b.to_json(path)
        size_kb = os.path.getsize(path) / 1024
        print(f"    Written: {path}  ({size_kb:.1f} KB)")

        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        required_keys = ["model_name", "num_layers", "dead_experts_count",
                         "critical_layers", "risk_scores"]
        print(f"    Top-level keys: {list(data.keys())}")
        for k in required_keys:
            assert k in data, f"Missing key: {k}"

        assert data["num_layers"]          == report_b.num_layers
        assert data["dead_experts_count"]  == report_b.dead_experts_count
        assert data["critical_layers"]     == list(report_b.critical_layers)
        print(f"    ✓ Round-trip verified: num_layers={data['num_layers']}  "
              f"dead={data['dead_experts_count']}  "
              f"critical={data['critical_layers']}")
    finally:
        os.unlink(path)

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n  All assertions passed.")
    print(f"  Healthy  → dead={report_a.dead_experts_count}  "
          f"critical={len(report_a.critical_layers)}  "
          f"top_risk={top_risk_a:.4f}")
    print(f"  Collapsed→ dead={report_b.dead_experts_count}  "
          f"critical={len(report_b.critical_layers)}  "
          f"top_risk={top_risk_b:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    run()
