"""
================================================================================
  MoEWatch v0.2.0 — Hook Overhead Profiler
================================================================================
  Purpose   : Measures MoEWatch forward+backward hook overhead vs unmonitored
              baseline. Reports mean latency, p95 latency, tokens/sec, and peak
              GPU memory delta. Outputs results to overhead_results.json.

  Targets   : Forward overhead  < 3%
              Backward overhead < 5%

  Setup
  -----
  OS        : Windows 10.0.26200
  CPU       : AMD64 Family 26 Model 36 Stepping 0, AuthenticAMD
              10 physical cores / 20 logical threads
  RAM       : 24.8 GB
  GPU       : NVIDIA GeForce RTX 4050 Laptop GPU  (6.4 GB VRAM)
  CUDA      : 12.6
  PyTorch   : 2.11.0+cu126
  Python    : 3.13.7 [MSC v.1944 64 bit (AMD64)]

  Model stub: FakeQwen3MoEModel
              28 layers | 64 routed experts | top-8 routing | hidden_dim=64
  Batch     : 2 sequences × 32 tokens = 64 tokens/step
  Iterations: 500 timed + 50 warm-up (discarded)
================================================================================
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Inline model stub (mirrors examples/qwen3_moe_finetune.py)
# ---------------------------------------------------------------------------

NUM_LAYERS  = 28
NUM_EXPERTS = 64
HIDDEN_DIM  = 64
TOP_K       = 8
VOCAB_SIZE  = 1000
BATCH_SIZE  = 2
SEQ_LEN     = 32


class _FakeQwen3Config:
    router_aux_loss_coef: float = 0.001


class _MoEBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate    = nn.Linear(HIDDEN_DIM, NUM_EXPERTS, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2, bias=False),
                nn.SiLU(),
                nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM, bias=False),
            )
            for _ in range(NUM_EXPERTS)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        x_flat      = x.reshape(-1, D)
        logits      = self.gate(x_flat)
        probs       = torch.softmax(logits, -1)
        top_probs, top_idx = probs.topk(TOP_K, dim=-1)
        top_probs   = top_probs / top_probs.sum(-1, keepdim=True)
        out         = torch.zeros_like(x_flat)
        for k in range(TOP_K):
            for e in range(NUM_EXPERTS):
                mask = (top_idx[:, k] == e)
                if mask.any():
                    out[mask] += top_probs[mask, k:k+1] * self.experts[e](x_flat[mask])
        return out.reshape(B, S, D)


class _DecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn           = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.mlp                 = _MoEBlock()
        self.input_layernorm     = nn.LayerNorm(HIDDEN_DIM)
        self.post_attn_layernorm = nn.LayerNorm(HIDDEN_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attn_layernorm(x))
        return x


class _FakeQwen3MoEModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config       = _FakeQwen3Config()
        self.embed_tokens = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.layers       = nn.ModuleList([_DecoderLayer() for _ in range(NUM_LAYERS)])
        self.norm         = nn.LayerNorm(HIDDEN_DIM)
        self.lm_head      = nn.Linear(HIDDEN_DIM, VOCAB_SIZE, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))


# ---------------------------------------------------------------------------
# MoEWatch trainer shim (same as used in the example)
# ---------------------------------------------------------------------------

class _DummyTrainer:
    """Minimal trainer shim so watcher.attach() succeeds outside HF Trainer."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def compute_loss(self, model, inputs, return_outputs=False):  # noqa: ANN001
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _gpu_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _peak_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def _reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _timed_step(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    watcher=None,
    step: int = 0,
) -> Tuple[float, float, float]:
    """
    Run one forward+backward step and return (fwd_ms, bwd_ms, total_ms).
    If watcher is provided, wraps with pre_step / step calls.
    """
    if watcher is not None:
        watcher.pre_step(step)

    optimizer.zero_grad()

    # --- Forward ---
    _gpu_sync()
    t0 = time.perf_counter()
    logits = model(input_ids)
    loss   = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
    )
    _gpu_sync()
    t1 = time.perf_counter()

    # --- Backward ---
    loss.backward()
    _gpu_sync()
    t2 = time.perf_counter()

    optimizer.step()

    if watcher is not None:
        watcher.step(global_step=step, current_loss=loss.item())

    fwd_ms   = (t1 - t0) * 1_000
    bwd_ms   = (t2 - t1) * 1_000
    total_ms = (t2 - t0) * 1_000
    return fwd_ms, bwd_ms, total_ms


def _run_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_warmup: int,
    n_iters: int,
    watcher=None,
) -> Dict[str, List[float]]:
    """Warm up then collect n_iters timing samples."""
    fwd_times:   List[float] = []
    bwd_times:   List[float] = []
    total_times: List[float] = []

    step = 0
    for i in range(n_warmup + n_iters):
        step += 1
        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        labels    = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        fwd, bwd, total = _timed_step(model, input_ids, labels, optimizer, watcher, step)

        if i >= n_warmup:   # discard warm-up
            fwd_times.append(fwd)
            bwd_times.append(bwd)
            total_times.append(total)

    return {"fwd": fwd_times, "bwd": bwd_times, "total": total_times}


def _summarise(times: List[float]) -> Dict[str, float]:
    s = sorted(times)
    n = len(s)
    return {
        "mean_ms":   statistics.mean(s),
        "median_ms": statistics.median(s),
        "p95_ms":    s[int(n * 0.95)],
        "p99_ms":    s[int(n * 0.99)],
        "min_ms":    s[0],
        "max_ms":    s[-1],
        "std_ms":    statistics.stdev(s),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    N_WARMUP = 50
    N_ITERS  = 500
    TOKENS_PER_STEP = BATCH_SIZE * SEQ_LEN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*72}")
    print(f"  MoEWatch v0.2.0 — Hook Overhead Profiler")
    print(f"{'='*72}")
    print(f"  Device      : {device}")
    print(f"  Model       : FakeQwen3MoEModel  "
          f"({NUM_LAYERS}L | {NUM_EXPERTS}E | top-{TOP_K} | dim={HIDDEN_DIM})")
    print(f"  Batch       : {BATCH_SIZE} × {SEQ_LEN} = {TOKENS_PER_STEP} tokens/step")
    print(f"  Warm-up     : {N_WARMUP} steps  |  Timed: {N_ITERS} steps")
    print(f"{'='*72}\n")

    # ------------------------------------------------------------------
    # 1. BASELINE — no MoEWatch
    # ------------------------------------------------------------------
    print("[1/2] Running BASELINE (no MoEWatch) ...")
    baseline_model = _FakeQwen3MoEModel().to(device)
    baseline_opt   = torch.optim.AdamW(baseline_model.parameters(), lr=2e-5)

    _reset_peak_memory()
    mem_before_baseline = _peak_memory_mb()
    baseline_times = _run_loop(
        baseline_model, baseline_opt, device, N_WARMUP, N_ITERS, watcher=None
    )
    mem_after_baseline = _peak_memory_mb()
    baseline_mem_delta = mem_after_baseline - mem_before_baseline

    baseline_fwd   = _summarise(baseline_times["fwd"])
    baseline_bwd   = _summarise(baseline_times["bwd"])
    baseline_total = _summarise(baseline_times["total"])
    baseline_tps   = TOKENS_PER_STEP / (baseline_total["mean_ms"] / 1_000)

    print(f"  Baseline fwd  mean={baseline_fwd['mean_ms']:.3f}ms  "
          f"p95={baseline_fwd['p95_ms']:.3f}ms")
    print(f"  Baseline bwd  mean={baseline_bwd['mean_ms']:.3f}ms  "
          f"p95={baseline_bwd['p95_ms']:.3f}ms")
    print(f"  Baseline tok/s: {baseline_tps:.1f}")
    print()

    # ------------------------------------------------------------------
    # 2. MONITORED — with MoEWatch
    # ------------------------------------------------------------------
    print("[2/2] Running MONITORED (with MoEWatch) ...")

    from moewatch import MoEWatch
    from moewatch.config import WatchConfig, OutputMode

    monitored_model = _FakeQwen3MoEModel().to(device)
    monitored_opt   = torch.optim.AdamW(monitored_model.parameters(), lr=2e-5)

    config  = WatchConfig(
        output=OutputMode.SILENT,
        sample_every=1,
        log_every=1,
        intervention_enabled=True,
        policy_type="rule",
        dead_threshold=0.0005,
        cold_threshold=0.002,
    )
    watcher = MoEWatch(model=monitored_model, config=config)
    trainer = _DummyTrainer(monitored_model)
    watcher.attach(trainer)

    _reset_peak_memory()
    mem_before_monitored = _peak_memory_mb()
    monitored_times = _run_loop(
        monitored_model, monitored_opt, device, N_WARMUP, N_ITERS, watcher=watcher
    )
    mem_after_monitored = _peak_memory_mb()
    monitored_mem_delta = mem_after_monitored - mem_before_monitored

    monitored_fwd   = _summarise(monitored_times["fwd"])
    monitored_bwd   = _summarise(monitored_times["bwd"])
    monitored_total = _summarise(monitored_times["total"])
    monitored_tps   = TOKENS_PER_STEP / (monitored_total["mean_ms"] / 1_000)

    print(f"  Monitored fwd  mean={monitored_fwd['mean_ms']:.3f}ms  "
          f"p95={monitored_fwd['p95_ms']:.3f}ms")
    print(f"  Monitored bwd  mean={monitored_bwd['mean_ms']:.3f}ms  "
          f"p95={monitored_bwd['p95_ms']:.3f}ms")
    print(f"  Monitored tok/s: {monitored_tps:.1f}")
    print()

    # ------------------------------------------------------------------
    # 3. Overhead calculation
    # ------------------------------------------------------------------
    def _pct(base: float, monitored: float) -> float:
        return (monitored - base) / base * 100 if base > 0 else 0.0

    fwd_overhead_pct   = _pct(baseline_fwd["mean_ms"],   monitored_fwd["mean_ms"])
    bwd_overhead_pct   = _pct(baseline_bwd["mean_ms"],   monitored_bwd["mean_ms"])
    total_overhead_pct = _pct(baseline_total["mean_ms"], monitored_total["mean_ms"])
    tps_delta_pct      = _pct(baseline_tps, monitored_tps)

    FWD_TARGET_PCT = 3.0
    BWD_TARGET_PCT = 5.0
    fwd_pass  = fwd_overhead_pct  <= FWD_TARGET_PCT
    bwd_pass  = bwd_overhead_pct  <= BWD_TARGET_PCT

    # ------------------------------------------------------------------
    # 4. Results summary
    # ------------------------------------------------------------------
    print(f"{'='*72}")
    print(f"  Overhead Summary  ({N_ITERS} timed steps, {N_WARMUP} warm-up discarded)")
    print(f"{'='*72}")
    print(f"  {'Metric':<30} {'Baseline':>12} {'Monitored':>12} {'Overhead':>10}")
    print(f"  {'-'*66}")
    print(f"  {'Fwd mean latency (ms)':<30} "
          f"{baseline_fwd['mean_ms']:>12.3f} "
          f"{monitored_fwd['mean_ms']:>12.3f} "
          f"{fwd_overhead_pct:>+9.2f}%")
    print(f"  {'Fwd p95 latency (ms)':<30} "
          f"{baseline_fwd['p95_ms']:>12.3f} "
          f"{monitored_fwd['p95_ms']:>12.3f}")
    print(f"  {'Bwd mean latency (ms)':<30} "
          f"{baseline_bwd['mean_ms']:>12.3f} "
          f"{monitored_bwd['mean_ms']:>12.3f} "
          f"{bwd_overhead_pct:>+9.2f}%")
    print(f"  {'Bwd p95 latency (ms)':<30} "
          f"{baseline_bwd['p95_ms']:>12.3f} "
          f"{monitored_bwd['p95_ms']:>12.3f}")
    print(f"  {'Total mean latency (ms)':<30} "
          f"{baseline_total['mean_ms']:>12.3f} "
          f"{monitored_total['mean_ms']:>12.3f} "
          f"{total_overhead_pct:>+9.2f}%")
    print(f"  {'Tokens / sec':<30} "
          f"{baseline_tps:>12.1f} "
          f"{monitored_tps:>12.1f} "
          f"{tps_delta_pct:>+9.2f}%")
    print(f"  {'Peak GPU mem delta (MB)':<30} "
          f"{baseline_mem_delta:>12.1f} "
          f"{monitored_mem_delta:>12.1f}")
    print(f"  {'-'*66}")
    print(f"  Target: fwd < {FWD_TARGET_PCT}%  →  "
          f"{'PASS ✓' if fwd_pass else 'FAIL ✗'}  "
          f"({fwd_overhead_pct:+.2f}%)")
    print(f"  Target: bwd < {BWD_TARGET_PCT}%  →  "
          f"{'PASS ✓' if bwd_pass else 'FAIL ✗'}  "
          f"({bwd_overhead_pct:+.2f}%)")
    print(f"{'='*72}\n")

    # ------------------------------------------------------------------
    # 5. Write JSON
    # ------------------------------------------------------------------
    results = {
        "config": {
            "n_warmup":        N_WARMUP,
            "n_iters":         N_ITERS,
            "batch_size":      BATCH_SIZE,
            "seq_len":         SEQ_LEN,
            "tokens_per_step": TOKENS_PER_STEP,
            "num_layers":      NUM_LAYERS,
            "num_experts":     NUM_EXPERTS,
            "hidden_dim":      HIDDEN_DIM,
            "top_k":           TOP_K,
            "device":          str(device),
            "environment": {
                "os":      "Windows 10.0.26200",
                "cpu":     "AMD64 Family 26 Model 36 Stepping 0, AuthenticAMD",
                "cores":   "10 physical / 20 logical",
                "ram_gb":  24.8,
                "gpu":     "NVIDIA GeForce RTX 4050 Laptop GPU",
                "vram_gb": 6.4,
                "cuda":    "12.6",
                "pytorch": "2.11.0+cu126",
                "python":  "3.13.7",
            },
        },
        "targets": {
            "fwd_overhead_pct": FWD_TARGET_PCT,
            "bwd_overhead_pct": BWD_TARGET_PCT,
        },
        "baseline": {
            "fwd":        baseline_fwd,
            "bwd":        baseline_bwd,
            "total":      baseline_total,
            "tokens_per_sec": baseline_tps,
            "peak_gpu_mem_delta_mb": baseline_mem_delta,
        },
        "monitored": {
            "fwd":        monitored_fwd,
            "bwd":        monitored_bwd,
            "total":      monitored_total,
            "tokens_per_sec": monitored_tps,
            "peak_gpu_mem_delta_mb": monitored_mem_delta,
        },
        "overhead": {
            "fwd_mean_pct":   fwd_overhead_pct,
            "bwd_mean_pct":   bwd_overhead_pct,
            "total_mean_pct": total_overhead_pct,
            "tps_delta_pct":  tps_delta_pct,
            "fwd_pass":       fwd_pass,
            "bwd_pass":       bwd_pass,
        },
    }

    out_path = Path(__file__).parent / "overhead_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"  Results saved → {out_path}")


if __name__ == "__main__":
    main()
