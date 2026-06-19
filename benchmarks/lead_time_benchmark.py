"""
================================================================================
  MoEWatch v0.2.0 — Lead-Time Advantage Benchmark
  File: benchmarks/lead_time_benchmark.py
================================================================================
  Purpose   : Measures T_moewatch vs T_utilization vs T_entropy vs T_aux_loss
              across 20 training runs with different random seeds. Computes mean
              and std of lead-time advantage for each baseline. Core empirical
              proof of the paper claim. Outputs lead_time_results.json.

  Definition of "lead time"
  -------------------------
  The step at which a signal first triggers a WARNING or CRITICAL alert on a
  collapsing layer is called T_signal. "Lead-time advantage" of MoEWatch over
  a baseline is:

      lead_time_advantage = T_baseline − T_moewatch   (steps)

  Positive values mean MoEWatch detected earlier than the baseline.

  Baselines
  ---------
  T_utilization : naïve per-expert token-count threshold  (load imbalance)
  T_entropy     : per-layer routing-entropy threshold (no drift tracking)
  T_aux_loss    : auxiliary load-balancing loss spike threshold

  Speed optimisations (v0.2.0)
  ----------------------------
  1. Batched expert dispatch  — replaces the 64-iteration Python loop with a
     single stacked Linear + scatter_add over all experts at once.
  2. Routing-prob cache       — hooks capture probs during the fwd pass so
     get_routing_probs() never triggers a second forward pass.
  3. SGD instead of AdamW     — 3-4× faster optimizer for a timing benchmark.
  4. torch.compile            — optional graph-mode compilation (PyTorch ≥ 2.0).
  5. Parallel seeds           — multiprocessing.Pool runs seeds concurrently
     across CPU cores when CUDA is unavailable; serialised on GPU to avoid
     context thrash.
  6. Early-exit               — once all four detectors have fired, the run
     stops immediately without completing remaining steps.

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
  Runs      : 20 (seeded 0..19)
  Steps/run : 200 (with collapse injected at step 80)
================================================================================
"""

from __future__ import annotations

import json
import logging
import math
import multiprocessing as mp
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.getLogger("moewatch").setLevel(logging.CRITICAL)

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

NUM_LAYERS   = 28
NUM_EXPERTS  = 64
HIDDEN_DIM   = 64
TOP_K        = 8
VOCAB_SIZE   = 1000
BATCH_SIZE   = 2
SEQ_LEN      = 32

# Benchmark parameters
N_SEEDS           = 20
N_STEPS           = 200
COLLAPSE_STEP     = 80
COLLAPSE_LAYER    = 0
COLLAPSE_STRENGTH = 12.0

# Baseline detection thresholds
UTIL_IMBALANCE_THRESHOLD  = 0.70
ENTROPY_THRESHOLD         = 0.35
AUX_LOSS_SPIKE_THRESHOLD  = 0.30

# Set True to try torch.compile (requires PyTorch >= 2.0, adds ~30s first-run cost)
USE_COMPILE = False


# ---------------------------------------------------------------------------
# Optimised model
# ---------------------------------------------------------------------------

class _FakeQwen3Config:
    router_aux_loss_coef: float = 0.001


class _MoEBlock(nn.Module):
    """
    Batched expert dispatch — all 64 experts share weight tensors stacked into
    (NUM_EXPERTS, in_features, out_features) so a single bmm replaces the
    64-iteration Python loop. ~10-20× faster than the naïve loop.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Linear(HIDDEN_DIM, NUM_EXPERTS, bias=False)

        # Stack expert weights: shape (E, D, 2D) and (E, 2D, D)
        self.w1 = nn.Parameter(torch.empty(NUM_EXPERTS, HIDDEN_DIM, HIDDEN_DIM * 2))
        self.w2 = nn.Parameter(torch.empty(NUM_EXPERTS, HIDDEN_DIM * 2, HIDDEN_DIM))
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

        self.register_buffer("collapse_bias", torch.zeros(NUM_EXPERTS))

        # Routing-prob cache written during forward; read by baseline detectors.
        self._cached_routing_probs: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        x_flat = x.reshape(-1, D)          # (T, D)
        T = x_flat.shape[0]

        logits = self.gate(x_flat) + self.collapse_bias   # (T, E)
        probs  = torch.softmax(logits, -1)                # (T, E)

        # Cache mean probs so baseline detectors don't need a second fwd pass
        self._cached_routing_probs = probs.detach().mean(0)  # (E,)

        top_probs, top_idx = probs.topk(TOP_K, dim=-1)       # (T, K)
        top_probs = top_probs / top_probs.sum(-1, keepdim=True)

        # ------------------------------------------------------------------
        # Batched dispatch: for each top-k slot, look up the expert weight
        # via indexing, run a single batched matmul across all (T*K) tokens.
        # ------------------------------------------------------------------
        # flat_idx : (T*K,)   which expert each (token, slot) pair uses
        flat_idx    = top_idx.reshape(-1)                          # (T*K,)
        flat_weight = top_probs.reshape(-1, 1)                     # (T*K, 1)
        flat_x      = x_flat[torch.arange(T, device=x.device)
                              .unsqueeze(1).expand(T, TOP_K)
                              .reshape(-1)]                        # (T*K, D)

        # Gather per-token expert weights: (T*K, D, 2D) and (T*K, 2D, D)
        w1_sel = self.w1[flat_idx]     # (T*K, D, 2D)
        w2_sel = self.w2[flat_idx]     # (T*K, 2D, D)

        # bmm: (T*K, 1, D) @ (T*K, D, 2D) → (T*K, 1, 2D)
        h = torch.bmm(flat_x.unsqueeze(1), w1_sel).squeeze(1)     # (T*K, 2D)
        h = F.silu(h)
        h = torch.bmm(h.unsqueeze(1), w2_sel).squeeze(1)          # (T*K, D)

        # Weighted accumulation back into output
        weighted = h * flat_weight                                 # (T*K, D)
        token_ids = (torch.arange(T, device=x.device)
                     .unsqueeze(1).expand(T, TOP_K).reshape(-1))  # (T*K,)
        out = torch.zeros_like(x_flat)
        out.scatter_add_(0, token_ids.unsqueeze(1).expand_as(weighted), weighted)

        return out.reshape(B, S, D)

    def _inject_collapse(self) -> None:
        self.collapse_bias[0] = COLLAPSE_STRENGTH

    def _clear_collapse(self) -> None:
        self.collapse_bias.zero_()


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

    def inject_collapse(self, layer_idx: int = COLLAPSE_LAYER) -> None:
        self.layers[layer_idx].mlp._inject_collapse()

    def get_cached_routing_probs(self, layer_idx: int = COLLAPSE_LAYER) -> torch.Tensor:
        """Return routing probs cached during the most recent forward pass — zero extra compute."""
        probs = self.layers[layer_idx].mlp._cached_routing_probs
        if probs is None:
            raise RuntimeError("No cached routing probs yet; run a forward pass first.")
        return probs


class _DummyTrainer:
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def add_callback(self, cb) -> None:
        pass

    def compute_loss(self, model, inputs, return_outputs=False):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Baseline detectors  (operate on cached probs — no extra compute)
# ---------------------------------------------------------------------------

def _detect_utilization(probs: torch.Tensor) -> bool:
    return bool(probs.max().item() > UTIL_IMBALANCE_THRESHOLD)


def _routing_entropy_norm(probs: torch.Tensor) -> float:
    n = probs.shape[0]
    p = probs.clamp(min=1e-9)
    return -(p * p.log()).sum().item() / math.log(n)


def _detect_entropy(probs: torch.Tensor) -> bool:
    return _routing_entropy_norm(probs) < ENTROPY_THRESHOLD


def _detect_aux_loss(probs: torch.Tensor) -> bool:
    n = probs.shape[0]
    return float(n * (probs * probs).sum().item()) > AUX_LOSS_SPIKE_THRESHOLD


# ---------------------------------------------------------------------------
# Single-run experiment
# ---------------------------------------------------------------------------

def _run_one_seed(args: Tuple[int, str]) -> Dict:
    """
    Worker function — runs one seed. Accepts (seed, device_str) so it can
    be called from a multiprocessing pool without pickling a device object.
    """
    seed, device_str = args
    device = torch.device(device_str)

    torch.manual_seed(seed)
    random.seed(seed)

    model     = _FakeQwen3MoEModel().to(device)
    # SGD is 3-4× faster than AdamW for a timing benchmark
    optimizer = torch.optim.SGD(model.parameters(), lr=2e-5, momentum=0.9)

    if USE_COMPILE:
        try:
            model = torch.compile(model)
        except Exception:
            pass

    from moewatch import MoEWatch
    from moewatch.config import WatchConfig, OutputMode

    config = WatchConfig(
        output=OutputMode.SILENT,
        sample_every=1,
        log_every=1,
        intervention_enabled=False,
        dead_threshold=0.0005,
        cold_threshold=0.002,
    )
    watcher = MoEWatch(model=model, config=config)
    trainer = _DummyTrainer(model)
    watcher.attach(trainer)

    t_moewatch:     Optional[int] = None
    t_utilization:  Optional[int] = None
    t_entropy:      Optional[int] = None
    t_aux_loss_det: Optional[int] = None

    for step in range(1, N_STEPS + 1):
        if step == COLLAPSE_STEP:
            model.inject_collapse(COLLAPSE_LAYER)

        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        labels    = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)

        watcher.pre_step(step)
        optimizer.zero_grad()
        logits = model(input_ids)
        loss   = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        loss.backward()
        optimizer.step()

        report = watcher.step(global_step=step, current_loss=loss.item())

        if t_moewatch is None:
            for alert in report.alerts:
                if alert.level.value in ("warning", "critical"):
                    t_moewatch = step
                    break

        # Use probs cached inside the MoEBlock during the forward pass above —
        # no second forward pass needed.
        probs = model.get_cached_routing_probs(COLLAPSE_LAYER)

        if t_utilization  is None and _detect_utilization(probs): t_utilization  = step
        if t_entropy       is None and _detect_entropy(probs):      t_entropy       = step
        if t_aux_loss_det  is None and _detect_aux_loss(probs):     t_aux_loss_det  = step

        # Early-exit once all four detectors have fired
        if all(v is not None for v in (t_moewatch, t_utilization, t_entropy, t_aux_loss_det)):
            break

    watcher.stop()

    return {
        "seed":          seed,
        "t_moewatch":    t_moewatch,
        "t_utilization": t_utilization,
        "t_entropy":     t_entropy,
        "t_aux_loss":    t_aux_loss_det,
    }


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _lead_time_advantage(
    t_mw: List[Optional[int]],
    t_bl: List[Optional[int]],
) -> Tuple[float, float, int]:
    deltas = [tb - tm for tm, tb in zip(t_mw, t_bl) if tm is not None and tb is not None]
    if not deltas:
        return float("nan"), float("nan"), 0
    return statistics.mean(deltas), (statistics.stdev(deltas) if len(deltas) > 1 else 0.0), len(deltas)


def _fire_rate(lst: List[Optional[int]]) -> float:
    return sum(1 for v in lst if v is not None) / len(lst)


def _mean_detect(lst: List[Optional[int]]) -> str:
    vals = [v for v in lst if v is not None]
    return f"{statistics.mean(vals):.1f}" if vals else "N/A"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = str(device)

    print(f"\n{'='*72}")
    print(f"  MoEWatch v0.2.0 — Lead-Time Advantage Benchmark")
    print(f"{'='*72}")
    print(f"  Device        : {device}")
    print(f"  Model         : FakeQwen3MoEModel  "
          f"({NUM_LAYERS}L | {NUM_EXPERTS}E | top-{TOP_K} | dim={HIDDEN_DIM})")
    print(f"  Batch         : {BATCH_SIZE} × {SEQ_LEN} = {BATCH_SIZE * SEQ_LEN} tokens/step")
    print(f"  Seeds         : {N_SEEDS}  (0 .. {N_SEEDS - 1})")
    print(f"  Steps/run     : {N_STEPS}")
    print(f"  Collapse at   : step {COLLAPSE_STEP}  "
          f"(layer {COLLAPSE_LAYER}, bias={COLLAPSE_STRENGTH})")
    print(f"  Dispatch      : batched bmm  (no per-expert Python loop)")
    print(f"  Optimizer     : SGD+momentum  (no AdamW state overhead)")
    print(f"  Routing probs : forward-pass cache  (no second fwd pass)")

    # On GPU: run seeds serially (CUDA context is not fork-safe).
    # On CPU: use a multiprocessing pool to parallelise across cores.
    use_parallel = (device.type == "cpu")
    n_workers    = min(N_SEEDS, max(1, mp.cpu_count() - 1)) if use_parallel else 1
    print(f"  Parallelism   : {'pool ×' + str(n_workers) if use_parallel else 'serial (GPU)'}")
    print(f"{'='*72}\n")

    seed_args = [(s, device_str) for s in range(N_SEEDS)]
    t0 = time.perf_counter()

    if use_parallel and n_workers > 1:
        with mp.Pool(processes=n_workers) as pool:
            results_list = []
            for res in pool.imap(_run_one_seed, seed_args):
                results_list.append(res)
                s = res["seed"]
                print(
                    f"  Seed {s:02d}/{N_SEEDS-1}  "
                    f"T_moewatch={res['t_moewatch'] or 'N/A':>5}  "
                    f"T_util={res['t_utilization'] or 'N/A':>5}  "
                    f"T_ent={res['t_entropy'] or 'N/A':>5}  "
                    f"T_aux={res['t_aux_loss'] or 'N/A':>5}"
                )
            results_list.sort(key=lambda r: r["seed"])
    else:
        results_list = []
        for args in seed_args:
            res = _run_one_seed(args)
            results_list.append(res)
            s = res["seed"]
            sys.stdout.write(
                f"  Seed {s:02d}/{N_SEEDS-1}  "
                f"T_moewatch={str(res['t_moewatch'] or 'N/A'):>5}  "
                f"T_util={str(res['t_utilization'] or 'N/A'):>5}  "
                f"T_ent={str(res['t_entropy'] or 'N/A'):>5}  "
                f"T_aux={str(res['t_aux_loss'] or 'N/A'):>5}\n"
            )
            sys.stdout.flush()

    elapsed = time.perf_counter() - t0

    t_moewatch_all    = [r["t_moewatch"]    for r in results_list]
    t_utilization_all = [r["t_utilization"] for r in results_list]
    t_entropy_all     = [r["t_entropy"]     for r in results_list]
    t_aux_loss_all    = [r["t_aux_loss"]    for r in results_list]

    lta_util_mean, lta_util_std, n_util = _lead_time_advantage(t_moewatch_all, t_utilization_all)
    lta_ent_mean,  lta_ent_std,  n_ent  = _lead_time_advantage(t_moewatch_all, t_entropy_all)
    lta_aux_mean,  lta_aux_std,  n_aux  = _lead_time_advantage(t_moewatch_all, t_aux_loss_all)

    fr_moewatch    = _fire_rate(t_moewatch_all)
    fr_utilization = _fire_rate(t_utilization_all)
    fr_entropy     = _fire_rate(t_entropy_all)
    fr_aux_loss    = _fire_rate(t_aux_loss_all)

    print(f"\n{'='*72}")
    print(f"  Lead-Time Advantage Summary  ({N_SEEDS} seeds, collapse @ step {COLLAPSE_STEP})")
    print(f"{'='*72}")
    print(f"  {'Signal':<22} {'Fire Rate':>10} {'Mean T_detect':>14} "
          f"{'Lead-time adv':>15} {'Std':>8}  {'Pairs':>6}")
    print(f"  {'-'*76}")
    print(f"  {'MoEWatch':<22} {fr_moewatch:>10.0%} {_mean_detect(t_moewatch_all):>14} "
          f"{'(reference)':>15} {'—':>8}  {'—':>6}")
    print(f"  {'vs Utilisation':<22} {fr_utilization:>10.0%} "
          f"{_mean_detect(t_utilization_all):>14} "
          f"{f'{lta_util_mean:+.1f} steps':>15} {lta_util_std:>8.2f}  {n_util:>6}")
    print(f"  {'vs Entropy':<22} {fr_entropy:>10.0%} "
          f"{_mean_detect(t_entropy_all):>14} "
          f"{f'{lta_ent_mean:+.1f} steps':>15} {lta_ent_std:>8.2f}  {n_ent:>6}")
    print(f"  {'vs Aux-Loss':<22} {fr_aux_loss:>10.0%} "
          f"{_mean_detect(t_aux_loss_all):>14} "
          f"{f'{lta_aux_mean:+.1f} steps':>15} {lta_aux_std:>8.2f}  {n_aux:>6}")
    print(f"  {'-'*76}")
    print(f"\n  Positive lead-time = MoEWatch detected BEFORE the baseline.")
    print(f"  Total wall time: {elapsed:.1f}s")
    print(f"{'='*72}\n")

    # ------------------------------------------------------------------
    # Write JSON
    # ------------------------------------------------------------------
    def _safe_mean(lst):
        vals = [v for v in lst if v is not None]
        return statistics.mean(vals) if vals else None

    results = {
        "config": {
            "n_seeds":           N_SEEDS,
            "n_steps":           N_STEPS,
            "collapse_step":     COLLAPSE_STEP,
            "collapse_layer":    COLLAPSE_LAYER,
            "collapse_strength": COLLAPSE_STRENGTH,
            "batch_size":        BATCH_SIZE,
            "seq_len":           SEQ_LEN,
            "num_layers":        NUM_LAYERS,
            "num_experts":       NUM_EXPERTS,
            "hidden_dim":        HIDDEN_DIM,
            "top_k":             TOP_K,
            "device":            device_str,
            "optimisations": [
                "batched_bmm_dispatch",
                "routing_prob_cache",
                "sgd_optimizer",
                "early_exit_per_seed",
                "parallel_seeds_on_cpu",
            ],
            "baselines": {
                "utilization_threshold": UTIL_IMBALANCE_THRESHOLD,
                "entropy_threshold":     ENTROPY_THRESHOLD,
                "aux_loss_threshold":    AUX_LOSS_SPIKE_THRESHOLD,
            },
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
        "wall_time_seconds": round(elapsed, 2),
        "per_seed":          results_list,
        "fire_rates": {
            "moewatch":    fr_moewatch,
            "utilization": fr_utilization,
            "entropy":     fr_entropy,
            "aux_loss":    fr_aux_loss,
        },
        "mean_detection_step": {
            "moewatch":    _safe_mean(t_moewatch_all),
            "utilization": _safe_mean(t_utilization_all),
            "entropy":     _safe_mean(t_entropy_all),
            "aux_loss":    _safe_mean(t_aux_loss_all),
        },
        "lead_time_advantage": {
            "vs_utilization": {"mean_steps": lta_util_mean, "std_steps": lta_util_std, "n_pairs": n_util},
            "vs_entropy":     {"mean_steps": lta_ent_mean,  "std_steps": lta_ent_std,  "n_pairs": n_ent},
            "vs_aux_loss":    {"mean_steps": lta_aux_mean,  "std_steps": lta_aux_std,  "n_pairs": n_aux},
        },
    }

    out_path = Path(__file__).parent / "results" / "lead_time_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"  Results saved → {out_path}")


if __name__ == "__main__":
    main()
