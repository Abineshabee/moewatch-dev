# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# examples/json_output_pipeline.py
# =============================================================================
#
# Project      : MoEWatch
# Version      : v0.2.0
# Description  : Demonstrates every JSON output path in MoEWatch:
#
#                Pipeline 1 — JSONReporter (live training, NDJSON per step)
#                  • emit_step()  — generate a single NDJSON line in memory
#                  • write_step() — emit + write to .jsonl file
#                  • write_step_report() — write a pre-built StepReport
#                  • rotate()     — swap output file mid-run (log rotation)
#                  • context manager usage (with JSONReporter(...) as jr)
#
#                Pipeline 2 — WatchReport.to_json() (rolling session summary)
#                  • Serialise an entire training session to one JSON file
#                  • max_steps_in_output cap for large runs
#
#                Pipeline 3 — AuditReport.to_json() (offline audit export)
#                  • Post-training audit serialised to disk
#                  • Round-trip: load back and verify schema fields
#
#                Pipeline 4 — JSONReporter stdout mode (containerised logging)
#                  • output_file=None → all lines go to stdout
#                  • Useful for Kubernetes / Docker log aggregation
#
#                Pipeline 5 — extra metadata fields (W&B run ID, git SHA)
#                  • Embedding experiment metadata into every JSON line
#
# Usage
# -----
#   pip install moewatch torch
#   python examples/json_output_pipeline.py
#
# Author       : MoEWatch Example
# License      : Apache 2.0
#
# =============================================================================

from __future__ import annotations

import json
import math
import os
import time
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional

from moewatch import Alert, AlertLevel, audit
from moewatch.config import WatchConfig, OutputMode
from moewatch.intervention.actions import (
    AuxLossAction,
    NoOpAction,
    RouterNoiseAction,
)
from moewatch.report.json_reporter import JSONReporter
from moewatch.report.watch_report import StepReport, WatchReport

# ---------------------------------------------------------------------------
# Minimal MoE model stub
# ---------------------------------------------------------------------------

NUM_LAYERS  = 8
NUM_EXPERTS = 16
TOP_K       = 2
HIDDEN_DIM  = 64


class MoEBlock(nn.Module):
    def __init__(self, hidden_dim: int = HIDDEN_DIM, num_experts: int = NUM_EXPERTS, top_k: int = TOP_K):
        super().__init__()
        self.top_k      = top_k
        self.num_experts = num_experts
        self.gate       = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts    = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                          nn.Linear(hidden_dim, hidden_dim))
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D   = x.shape
        flat      = x.reshape(-1, D)
        logits    = self.gate(flat)
        probs     = torch.softmax(logits, dim=-1)
        top_p, top_idx = probs.topk(self.top_k, dim=-1)
        top_p     = top_p / top_p.sum(-1, keepdim=True)
        out       = torch.zeros_like(flat)
        for k in range(self.top_k):
            for e in range(self.num_experts):
                mask = top_idx[:, k] == e
                if mask.any():
                    out[mask] += top_p[mask, k:k+1] * self.experts[e](flat[mask])
        return out.reshape(B, S, D)


class SimpleMoEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed  = nn.Embedding(32000, HIDDEN_DIM)
        self.layers = nn.ModuleList([
            MoEBlock() if i % 2 == 1 else
            nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.SiLU(),
                          nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM))
            for i in range(NUM_LAYERS)
        ])
        self.norm    = nn.LayerNorm(HIDDEN_DIM)
        self.lm_head = nn.Linear(HIDDEN_DIM, 32000, bias=False)

        # MoE layer indices (odd layers)
        self._moe_indices = [i for i in range(NUM_LAYERS) if i % 2 == 1]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = x + layer(x) if isinstance(layer, MoEBlock) else layer(x)
        return self.lm_head(self.norm(x))

    def inject_collapse(self, dominant: int = 0, scale: float = 5.0) -> None:
        with torch.no_grad():
            for i in self._moe_indices:
                g = self.layers[i].gate
                g.weight.zero_()
                g.weight[dominant] += scale

    def restore_healthy(self) -> None:
        with torch.no_grad():
            for i in self._moe_indices:
                self.layers[i].gate.weight.normal_(0, 0.02)


def make_loader(num_batches: int = 30, batch_size: int = 4, seq_len: int = 16) -> DataLoader:
    torch.manual_seed(42)
    ids = torch.randint(0, 32000, (num_batches * batch_size, seq_len))
    return DataLoader(TensorDataset(ids), batch_size=batch_size, shuffle=False)


def make_config() -> WatchConfig:
    return WatchConfig(
        output=OutputMode.SILENT,
        entropy_warn=0.40,
        entropy_critical=0.18,
        dead_threshold=0.008,
        cold_threshold=0.04,
        cold_steps_limit=20,
        log_every=1,
        sample_every=1,
        intervention_enabled=False,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MoE_LAYERS = [f"layers.{i}.gate" for i in range(1, NUM_LAYERS, 2)]


def _make_step_report(
    step: int,
    risk_offset: float = 0.0,
    with_alert: bool = False,
    with_intervention: bool = False,
) -> StepReport:
    """Build a synthetic StepReport for demo purposes."""
    risk_scores = {
        layer: round(0.05 + risk_offset + 0.01 * (idx % 4), 4)
        for idx, layer in enumerate(MoE_LAYERS)
    }
    risk_levels = {
        layer: ("high" if v > 0.6 else "mid" if v > 0.3 else "low")
        for layer, v in risk_scores.items()
    }
    alerts: List[Alert] = []
    if with_alert:
        alerts.append(Alert(
            step=step,
            level=AlertLevel.WARNING,
            layer_id=MoE_LAYERS[0],
            signal_type="entropy_drift",
            message=f"Entropy drop detected at step {step}",
            metrics={"normalized_entropy": 0.41, "risk_score": risk_scores[MoE_LAYERS[0]]},
        ))
    interventions: List = []
    if with_intervention:
        interventions.append(AuxLossAction(
            layer_name=MoE_LAYERS[0],
            delta=0.01,
        ))

    return StepReport(
        step=step,
        timestamp=time.time(),
        risk_scores=risk_scores,
        risk_levels=risk_levels,
        active_interventions=interventions,
        policy_decisions={layer: ("aux_loss" if with_intervention and idx == 0 else "noop")
                          for idx, layer in enumerate(MoE_LAYERS)},
        alerts=alerts,
        loss=round(6.5 - step * 0.002 + 0.1 * math.sin(step), 4),
        dominant_signals={layer: "entropy" for layer in MoE_LAYERS},
    )


# ---------------------------------------------------------------------------
# Pipeline 1 — JSONReporter: live NDJSON output
# ---------------------------------------------------------------------------

def pipeline1_json_reporter() -> None:
    print("=" * 70)
    print("  Pipeline 1 — JSONReporter (live NDJSON per step)")
    print("=" * 70)

    out_file  = "pipeline1_live.jsonl"
    rot_file  = "pipeline1_live_rotated.jsonl"

    # ------------------------------------------------------------------
    # 1a. emit_step() — generate line in memory (no file I/O)
    # ------------------------------------------------------------------
    print("\n  [1a] emit_step() — generate NDJSON line in memory")
    config = make_config()
    jr = JSONReporter(config, output_file=None)

    sr0 = _make_step_report(step=1)
    line = jr.emit_step(
        step=sr0.step,
        timestamp=sr0.timestamp,
        risk_scores=sr0.risk_scores,
        loss=sr0.loss,
        interventions=sr0.active_interventions,
        alerts=sr0.alerts,
        risk_levels=sr0.risk_levels,
        dominant_signals=sr0.dominant_signals,
    )
    parsed = json.loads(line)
    print(f"    step={parsed['step']}  loss={parsed['loss']}  "
          f"num_alerts={parsed['num_alerts']}  "
          f"num_interventions={parsed['num_interventions']}")
    print(f"    schema fields: {list(parsed.keys())}")

    # ------------------------------------------------------------------
    # 1b. write_step() — emit + write to .jsonl file
    # ------------------------------------------------------------------
    print(f"\n  [1b] write_step() — writing steps 1-10 to {out_file!r}")
    with JSONReporter(config, output_file=out_file) as jr:
        for step in range(1, 11):
            sr = _make_step_report(
                step=step,
                risk_offset=0.05 * (step / 10),
                with_alert=(step % 4 == 0),
                with_intervention=(step % 5 == 0),
            )
            jr.write_step(
                step=sr.step,
                timestamp=sr.timestamp,
                risk_scores=sr.risk_scores,
                loss=sr.loss,
                interventions=sr.active_interventions,
                alerts=sr.alerts,
                risk_levels=sr.risk_levels,
                dominant_signals=sr.dominant_signals,
            )

    size_kb = os.path.getsize(out_file) / 1024
    with open(out_file, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    print(f"    Written: {len(lines)} lines  ({size_kb:.1f} KB)")
    first = json.loads(lines[0])
    last  = json.loads(lines[-1])
    print(f"    First line: step={first['step']}  loss={first['loss']}")
    print(f"    Last line:  step={last['step']}   loss={last['loss']}")

    # ------------------------------------------------------------------
    # 1c. write_step_report() — pass a pre-built StepReport directly
    # ------------------------------------------------------------------
    print(f"\n  [1c] write_step_report() — StepReport convenience wrapper")
    with JSONReporter(config, output_file=out_file) as jr:
        for step in range(11, 16):
            sr = _make_step_report(step=step, risk_offset=0.1)
            jr.write_step_report(sr)

    with open(out_file, "r", encoding="utf-8") as fh:
        total_lines = len(fh.readlines())
    print(f"    After write_step_report() × 5: {total_lines} total lines in file")

    # ------------------------------------------------------------------
    # 1d. rotate() — swap output file mid-run
    # ------------------------------------------------------------------
    print(f"\n  [1d] rotate() — swap output file to {rot_file!r} mid-run")
    jr = JSONReporter(config, output_file=out_file)
    for step in range(16, 19):
        sr = _make_step_report(step=step)
        jr.write_step_report(sr)
    print(f"    Written steps 16-18 to {out_file!r}")

    jr.rotate(rot_file)
    for step in range(19, 22):
        sr = _make_step_report(step=step, risk_offset=0.2)
        jr.write_step_report(sr)
    jr.close()
    print(f"    After rotate: steps 19-21 written to {rot_file!r}")

    with open(rot_file, "r", encoding="utf-8") as fh:
        rot_lines = fh.readlines()
    print(f"    {rot_file}: {len(rot_lines)} line(s)")
    for ln in rot_lines:
        rec = json.loads(ln)
        print(f"      step={rec['step']}  loss={rec['loss']}  risks={len(rec['risks'])} layers")

    print(f"\n  Pipeline 1 complete.")


# ---------------------------------------------------------------------------
# Pipeline 2 — WatchReport.to_json()
# ---------------------------------------------------------------------------

def pipeline2_watch_report() -> None:
    print("\n" + "=" * 70)
    print("  Pipeline 2 — WatchReport.to_json() (session summary)")
    print("=" * 70)

    watch_report = WatchReport(max_steps=200)

    print("\n  Building synthetic training session (100 steps) ...")
    for step in range(1, 101):
        # Simulate gradual collapse from step 50
        risk_offset = max(0.0, (step - 50) * 0.012)
        sr = _make_step_report(
            step=step,
            risk_offset=risk_offset,
            with_alert=(step > 60 and step % 5 == 0),
            with_intervention=(step > 70 and step % 10 == 0),
        )
        watch_report.append(sr)

    # Full export
    full_path = "pipeline2_watch_report_full.json"
    watch_report.to_json(full_path)
    size_kb = os.path.getsize(full_path) / 1024
    print(f"\n  Full export → {full_path!r}  ({size_kb:.1f} KB)")

    with open(full_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    print(f"    Top-level keys   : {list(data.keys())}")
    print(f"    Steps in file    : {len(data.get('steps', []))}")
    print(f"    num_interventions: {data.get('num_interventions', '?')}")
    print(f"    num_alerts       : {data.get('num_alerts', '?')}")

    # Capped export (last 20 steps only)
    capped_path = "pipeline2_watch_report_capped.json"
    watch_report.to_json(capped_path, max_steps_in_output=20)
    capped_kb = os.path.getsize(capped_path) / 1024
    print(f"\n  Capped export (max_steps=20) → {capped_path!r}  ({capped_kb:.1f} KB)")
    with open(capped_path, "r", encoding="utf-8") as fh:
        capped = json.load(fh)
    print(f"    Steps in capped file: {len(capped.get('steps', []))}")

    # WatchReport analytics
    print(f"\n  WatchReport analytics:")
    print(f"    Step range         : {watch_report.start_step} – {watch_report.end_step}")
    print(f"    Total alerts       : {watch_report.num_alerts}")
    print(f"    Total interventions: {watch_report.num_interventions}")
    print(f"    Critical steps     : {watch_report.num_critical_steps}")
    worst_layer = MoE_LAYERS[0]
    mean_r = watch_report.mean_risk(worst_layer)
    print(f"    Mean risk ({worst_layer}): {mean_r:.4f}")

    print(f"\n  Pipeline 2 complete.")


# ---------------------------------------------------------------------------
# Pipeline 3 — AuditReport.to_json() (offline audit)
# ---------------------------------------------------------------------------

def pipeline3_audit_report() -> None:
    print("\n" + "=" * 70)
    print("  Pipeline 3 — AuditReport.to_json() (offline audit export)")
    print("=" * 70)

    print("\n  Building model and running offline audit ...")
    torch.manual_seed(1)
    model  = SimpleMoEModel()
    config = make_config()

    # Healthy audit
    model.restore_healthy()
    loader_healthy = make_loader(num_batches=20)
    report_healthy = audit(
        model=model,
        dataloader=loader_healthy,
        num_batches=20,
        config=config,
        device="cpu",
        with_backward=True,
    )

    # Collapsed audit
    model.inject_collapse(dominant=0, scale=6.0)
    loader_collapsed = make_loader(num_batches=20)
    report_collapsed = audit(
        model=model,
        dataloader=loader_collapsed,
        num_batches=20,
        config=config,
        device="cpu",
        with_backward=True,
    )

    # Export both
    healthy_path   = "pipeline3_audit_healthy.json"
    collapsed_path = "pipeline3_audit_collapsed.json"

    report_healthy.to_json(healthy_path)
    report_collapsed.to_json(collapsed_path)

    for label, path, report in [
        ("Healthy",   healthy_path,   report_healthy),
        ("Collapsed", collapsed_path, report_collapsed),
    ]:
        size_kb = os.path.getsize(path) / 1024
        print(f"\n  [{label}] → {path!r}  ({size_kb:.1f} KB)")
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        print(f"    Top-level keys     : {list(data.keys())}")
        print(f"    model_name         : {data.get('model_name', '?')}")
        print(f"    num_layers         : {data.get('num_layers', '?')}")
        print(f"    dead_experts_count : {data.get('dead_experts_count', '?')}")
        print(f"    critical_layers    : {data.get('critical_layers', [])}")

        # Risk scores round-trip check
        risk_map = data.get("risk_scores", {})
        if risk_map:
            worst = max(risk_map, key=lambda k: risk_map[k].get("risk_score", 0))
            worst_score = risk_map[worst].get("risk_score", "?")
            worst_level = risk_map[worst].get("risk_level", "?")
            print(f"    Worst layer        : {worst}")
            print(f"      risk_score       : {worst_score}")
            print(f"      risk_level       : {worst_level}")

    print(f"\n  Pipeline 3 complete.")


# ---------------------------------------------------------------------------
# Pipeline 4 — JSONReporter stdout mode (containerised logging)
# ---------------------------------------------------------------------------

def pipeline4_stdout_mode() -> None:
    print("\n" + "=" * 70)
    print("  Pipeline 4 — JSONReporter stdout mode (output_file=None)")
    print("=" * 70)

    print("\n  Writing 3 steps to stdout (each line is a valid JSON object):\n")
    config = make_config()

    # output_file=None → all output goes to stdout
    with JSONReporter(config, output_file=None) as jr:
        for step in [1, 2, 3]:
            sr = _make_step_report(step=step, risk_offset=0.05 * step)
            jr.write_step_report(sr)

    print("\n  (stdout lines above are valid NDJSON — pipe to jq, Splunk, etc.)")
    print(f"\n  Pipeline 4 complete.")


# ---------------------------------------------------------------------------
# Pipeline 5 — extra metadata fields (W&B run ID, git SHA, experiment name)
# ---------------------------------------------------------------------------

def pipeline5_extra_metadata() -> None:
    print("\n" + "=" * 70)
    print("  Pipeline 5 — Extra metadata fields in JSON output")
    print("=" * 70)

    out_file = "pipeline5_with_metadata.jsonl"
    config   = make_config()

    # Simulate experiment metadata — inject once per step
    experiment_meta = {
        "experiment_name": "deepseek_finetune_v3",
        "wandb_run_id":    "abc123xyz",
        "git_sha":         "f4e2b1c",
        "dataset":         "openhermes-2.5",
        "lr":              3e-4,
        "batch_size":      16,
    }

    print(f"\n  Writing 5 steps with experiment metadata to {out_file!r}")
    with JSONReporter(config, output_file=out_file) as jr:
        for step in range(1, 6):
            sr = _make_step_report(
                step=step,
                risk_offset=0.05 * step,
                with_alert=(step == 3),
                with_intervention=(step == 5),
            )
            jr.write_step(
                step=sr.step,
                timestamp=sr.timestamp,
                risk_scores=sr.risk_scores,
                loss=sr.loss,
                interventions=sr.active_interventions,
                alerts=sr.alerts,
                risk_levels=sr.risk_levels,
                dominant_signals=sr.dominant_signals,
                extra=experiment_meta,
            )

    with open(out_file, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    print(f"  Written: {len(lines)} lines")
    sample = json.loads(lines[2])  # step 3 has an alert
    print(f"\n  Sample line (step 3) — all fields:")
    for k, v in sample.items():
        if k in ("risks", "interventions", "alerts"):
            print(f"    {k:<28}: [{len(v)} item(s)]")
        else:
            print(f"    {k:<28}: {v}")

    # Verify metadata round-trip
    print(f"\n  Metadata round-trip check:")
    for key in experiment_meta:
        val = sample.get(key, "MISSING")
        status = "✓" if val == experiment_meta[key] else "✗"
        print(f"    {status}  {key} = {val!r}")

    print(f"\n  Pipeline 5 complete.")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_output_files() -> None:
    print("\n" + "=" * 70)
    print("  Output files written")
    print("=" * 70)
    files = [
        ("pipeline1_live.jsonl",                 "JSONReporter NDJSON (steps 1-18)"),
        ("pipeline1_live_rotated.jsonl",          "JSONReporter rotated file (steps 19-21)"),
        ("pipeline2_watch_report_full.json",      "WatchReport full session (100 steps)"),
        ("pipeline2_watch_report_capped.json",    "WatchReport capped export (20 steps)"),
        ("pipeline3_audit_healthy.json",          "AuditReport — healthy checkpoint"),
        ("pipeline3_audit_collapsed.json",        "AuditReport — collapsed checkpoint"),
        ("pipeline5_with_metadata.jsonl",         "JSONReporter with experiment metadata"),
    ]
    for fname, desc in files:
        if os.path.exists(fname):
            size_kb = os.path.getsize(fname) / 1024
            print(f"  {fname:<45}  {size_kb:>7.1f} KB   {desc}")
        else:
            print(f"  {fname:<45}  (not written)   {desc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  MoEWatch v0.2.0 — JSON Output Pipeline Example                  ║")
    print("║  JSONReporter · WatchReport · AuditReport · stdout · metadata    ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    pipeline1_json_reporter()
    pipeline2_watch_report()
    pipeline3_audit_report()
    pipeline4_stdout_mode()
    pipeline5_extra_metadata()
    print_output_files()

    print("\n" + "=" * 70)
    print("  All five pipelines complete.")
    print("  MoEWatch JSON output is compatible with:")
    print("    • Elasticsearch / OpenSearch (NDJSON ingest)")
    print("    • Grafana (JSON datasource)")
    print("    • W&B / MLflow (json artifact upload)")
    print("    • CI/CD failure gates (parse risk_scores from AuditReport JSON)")
    print("=" * 70)


if __name__ == "__main__":
    main()
