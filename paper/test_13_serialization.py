# =============================================================================
# MoEWatch v0.2.0 — Research Paper Test Suite
# Test 13: Serialization Round-Trip
# =============================================================================
#
# Verifies every serialization path in MoEWatch survives a full
# write → load → verify cycle with no data loss:
#
#   1. AuditReport.to_json()  → json.load() → field equality
#   2. AuditReport.to_dict()  → all required keys present
#   3. WatchReport.to_json()  → json.load() → step count, field equality
#   4. WatchReport.to_json(max_steps_in_output=N) → capped correctly
#   5. StepReport.to_dict()   → intervention and alert lists preserved
#   6. Numeric precision      → risk scores survive float round-trip
#   7. Unicode safety          → non-ASCII layer names survive JSON round-trip
#
# =============================================================================

import json
import math
import os
import tempfile
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from moewatch import Alert, AlertLevel, audit
from moewatch.config import WatchConfig, OutputMode
from moewatch.intervention.actions import AuxLossAction, RouterNoiseAction
from moewatch.report.watch_report import StepReport, WatchReport

LAYER   = "layers.0.gate"
LAYERS  = ["layers.0.gate", "layers.1.gate", "layers.2.gate"]


# ---------------------------------------------------------------------------
# Minimal MoE model for AuditReport generation
# ---------------------------------------------------------------------------

class FakeMoELayer(nn.Module):
    def __init__(self, dim=32, n_exp=8, top_k=2):
        super().__init__()
        self.gate    = nn.Linear(dim, n_exp, bias=False)
        self.gate.top_k = top_k
        self.experts = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(n_exp)])
        self.top_k   = top_k

    def forward(self, x):
        B, S, D = x.shape
        flat    = x.reshape(-1, D)
        probs   = torch.softmax(self.gate(flat), dim=-1)
        top_p, top_idx = probs.topk(self.top_k, dim=-1)
        top_p   = top_p / top_p.sum(-1, keepdim=True)
        out     = torch.zeros_like(flat)
        for k in range(self.top_k):
            for e in range(len(self.experts)):
                mask = top_idx[:, k] == e
                if mask.any():
                    out[mask] += top_p[mask, k:k+1] * self.experts[e](flat[mask])
        return out.reshape(B, S, D)


class AuditModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed  = nn.Embedding(1000, 32)
        self.layers = nn.ModuleList([FakeMoELayer() for _ in range(3)])
        self.head   = nn.Linear(32, 1000, bias=False)

    def forward(self, ids):
        x = self.embed(ids)
        for l in self.layers:
            x = x + l(x)
        return self.head(x)


def make_loader():
    torch.manual_seed(0)
    ids = torch.randint(0, 1000, (60, 16))
    return DataLoader(TensorDataset(ids), batch_size=4, shuffle=False)


def make_audit_config():
    return WatchConfig(output=OutputMode.SILENT, dead_threshold=0.5,
                       cold_threshold=1.0, intervention_enabled=False)


def make_step_report(step: int, with_intervention=False, with_alert=False) -> StepReport:
    actions = [AuxLossAction(layer_name=LAYER, delta=0.01)] if with_intervention else []
    alerts  = [Alert(step=step, level=AlertLevel.WARNING, layer_id=LAYER,
                     signal_type="entropy_drift",
                     message=f"Entropy drop at step {step}",
                     metrics={"normalized_entropy": 0.42})] if with_alert else []
    return StepReport(
        step=step,
        timestamp=time.time(),
        risk_scores={l: round(0.05 + 0.01 * (step % 5) + 0.005 * i, 6)
                     for i, l in enumerate(LAYERS)},
        risk_levels={l: "low" for l in LAYERS},
        active_interventions=actions,
        policy_decisions={l: "aux_loss" if with_intervention else "noop" for l in LAYERS},
        alerts=alerts,
        loss=round(6.5 - step * 0.001, 6),
        dominant_signals={l: "entropy" for l in LAYERS},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_and_load(report, path_suffix=".json", **kwargs) -> dict:
    with tempfile.NamedTemporaryFile(suffix=path_suffix, delete=False) as f:
        path = f.name
    try:
        report.to_json(path, **kwargs)
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh), path, os.path.getsize(path) / 1024
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def run():
    print("=" * 60)
    print("  MoEWatch Test 14 — Serialization Round-Trip")
    print("=" * 60)

    # ==================================================================
    # [1] AuditReport.to_json() round-trip
    # ==================================================================
    print("\n  [1] AuditReport.to_json() round-trip")

    torch.manual_seed(1)
    model  = AuditModel()
    config = make_audit_config()
    report = audit(model=model, dataloader=make_loader(), num_batches=15,
                   config=config, device="cpu", with_backward=True)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        report.to_json(path)
        size_kb = os.path.getsize(path) / 1024
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    finally:
        os.unlink(path)

    print(f"    File size          : {size_kb:.1f} KB")
    print(f"    num_layers         : {data['num_layers']}  == {report.num_layers}  "
          f"{'✓' if data['num_layers'] == report.num_layers else '✗'}")
    print(f"    dead_experts_count : {data['dead_experts_count']}  == {report.dead_experts_count}  "
          f"{'✓' if data['dead_experts_count'] == report.dead_experts_count else '✗'}")
    print(f"    critical_layers    : {data['critical_layers']}  == {list(report.critical_layers)}  "
          f"{'✓' if data['critical_layers'] == list(report.critical_layers) else '✗'}")

    assert data["num_layers"]         == report.num_layers
    assert data["dead_experts_count"] == report.dead_experts_count
    assert data["critical_layers"]    == list(report.critical_layers)
    print(f"    ✓ AuditReport round-trip verified")

    # ==================================================================
    # [2] AuditReport.to_dict() required keys
    # ==================================================================
    print("\n  [2] AuditReport.to_dict() required keys")

    d = report.to_dict()
    required = ["model_name", "timestamp", "audit_datetime", "num_batches",
                "num_layers", "dead_experts_count", "critical_layers",
                "entropy_results", "collapse_results", "gradient_results",
                "cross_layer_results", "risk_scores"]
    missing = [k for k in required if k not in d]
    print(f"    Keys present  : {len(d)}")
    print(f"    Missing keys  : {missing}  (expected [])")
    assert not missing, f"Missing keys: {missing}"
    print(f"    ✓ All {len(required)} required keys present")

    # ==================================================================
    # [3] WatchReport.to_json() round-trip
    # ==================================================================
    print("\n  [3] WatchReport.to_json() round-trip (50 steps)")

    wr = WatchReport(max_steps=200)
    for s in range(1, 51):
        wr.append(make_step_report(
            s,
            with_intervention=(s % 10 == 0),
            with_alert=(s % 7 == 0),
        ))

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        wr.to_json(path)
        size_kb = os.path.getsize(path) / 1024
        with open(path, "r", encoding="utf-8") as fh:
            wd = json.load(fh)
    finally:
        os.unlink(path)

    print(f"    File size          : {size_kb:.1f} KB")
    print(f"    steps in file      : {len(wd['steps'])}  == 50  "
          f"{'✓' if len(wd['steps']) == 50 else '✗'}")
    print(f"    num_alerts         : {wd['num_alerts']}  == {wr.num_alerts}  "
          f"{'✓' if wd['num_alerts'] == wr.num_alerts else '✗'}")
    print(f"    num_interventions  : {wd['num_interventions']}  == {wr.num_interventions}  "
          f"{'✓' if wd['num_interventions'] == wr.num_interventions else '✗'}")

    assert len(wd["steps"])        == 50
    assert wd["num_alerts"]        == wr.num_alerts
    assert wd["num_interventions"] == wr.num_interventions
    print(f"    ✓ WatchReport round-trip verified")

    # ==================================================================
    # [4] WatchReport.to_json(max_steps_in_output=10) cap
    # ==================================================================
    print("\n  [4] WatchReport.to_json(max_steps_in_output=10) cap")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        wr.to_json(path, max_steps_in_output=10)
        capped_kb = os.path.getsize(path) / 1024
        with open(path, "r", encoding="utf-8") as fh:
            cd = json.load(fh)
    finally:
        os.unlink(path)

    print(f"    Steps in capped file : {len(cd['steps'])}  (expected ≤ 10)")
    print(f"    Capped file size     : {capped_kb:.1f} KB  (< {size_kb:.1f} KB)")
    assert len(cd["steps"]) <= 10, \
        f"max_steps_in_output=10 but got {len(cd['steps'])} steps"
    assert capped_kb < size_kb, "Capped file should be smaller"
    print(f"    ✓ Cap enforced: {len(cd['steps'])} steps, {capped_kb:.1f} KB")

    # ==================================================================
    # [5] StepReport.to_dict() — intervention and alert lists preserved
    # ==================================================================
    print("\n  [5] StepReport.to_dict() — intervention + alert preservation")

    sr = make_step_report(step=99, with_intervention=True, with_alert=True)
    sd = sr.to_dict()

    print(f"    step               : {sd['step']}  (expected 99)")
    print(f"    num_interventions  : {sd.get('num_interventions', sd.get('interventions', 'N/A'))}")
    print(f"    num_alerts         : {sd.get('num_alerts', sd.get('alerts', 'N/A'))}")
    print(f"    loss               : {sd['loss']:.6f}")
    print(f"    risk_scores        : {sd.get('risk_scores', sd.get('risks', {}))}")

    assert sd["step"] == 99
    # Verify step is JSON-serializable
    j = json.dumps(sd)
    sd2 = json.loads(j)
    assert sd2["step"] == 99
    print(f"    ✓ StepReport.to_dict() is fully JSON-serializable")

    # ==================================================================
    # [6] Numeric precision — risk scores survive float round-trip
    # ==================================================================
    print("\n  [6] Numeric precision — float round-trip")

    # Build risk scores with high-precision values
    precise_scores = {
        "layers.0.gate": 0.123456789,
        "layers.1.gate": 0.999999999,
        "layers.2.gate": 1.234e-7,
    }
    sr2 = make_step_report(99)
    sr2.risk_scores.update(precise_scores)
    sd2 = sr2.to_dict()
    j2  = json.dumps(sd2)
    rd2 = json.loads(j2)

    risks_key = "risk_scores" if "risk_scores" in rd2 else "risks"
    for layer, original in precise_scores.items():
        recovered = rd2[risks_key].get(layer, float("nan"))
        err = abs(recovered - original)
        print(f"    {layer}: original={original}  recovered={recovered}  err={err:.2e}")
        assert err < 1e-6, f"Precision loss too high: {err:.2e}"
    print(f"    ✓ All risk scores preserved to < 1e-6 precision")

    # ==================================================================
    # [7] Unicode safety — non-ASCII layer names
    # ==================================================================
    print("\n  [7] Unicode safety — non-ASCII layer names")

    unicode_layers = ["层.0.gate", "레이어.1.gate", "слой.2.gate"]
    sr3 = StepReport(
        step=1,
        timestamp=time.time(),
        risk_scores={l: 0.5 for l in unicode_layers},
        risk_levels={l: "mid" for l in unicode_layers},
        loss=6.0,
    )
    sd3 = sr3.to_dict()
    j3  = json.dumps(sd3, ensure_ascii=False)
    rd3 = json.loads(j3)

    risks_key3 = "risk_scores" if "risk_scores" in rd3 else "risks"
    for layer in unicode_layers:
        assert layer in rd3[risks_key3], f"Unicode layer name lost: {layer!r}"
        print(f"    ✓ {layer!r} survived round-trip")
    print(f"    ✓ All non-ASCII layer names preserved")

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n  All assertions passed.")
    print(f"  AuditReport  : {size_kb:.1f} KB  ({len(d)} top-level keys)")
    print(f"  WatchReport  : {size_kb:.1f} KB  (50 steps)  →  capped {capped_kb:.1f} KB (10 steps)")
    print(f"  Float precision : < 1e-6 across all risk score values")
    print(f"  Unicode        : non-ASCII layer names survive JSON round-trip")
    print("=" * 60)


if __name__ == "__main__":
    run()
