"""
================================================================================
  MoEWatch v0.2.0 — Intervention Efficiency Score (IES) Benchmark
  File: benchmarks/ies_benchmark.py
================================================================================
  Purpose
  -------
  Measures Intervention Efficiency Score (IES) across all four action types:

      IES = stability_improvement / intervention_cost

  Runs controlled experiments: train WITH MoEWatch interventions vs train
  WITHOUT (control). Computes:
    - Dead expert rate reduction     (stability improvement component)
    - Loss trajectory deviation      (stability improvement component)
    - Compute overhead per step      (intervention cost component)
    - IES per action type            (final metric)

  Outputs ies_results.json (same results/ folder as other benchmarks).

  Definitions
  -----------
  stability_improvement
      Composite score ∈ [0, 1]:
        0.6 * dead_expert_rate_reduction  +  0.4 * loss_improvement_ratio

      dead_expert_rate_reduction:
          (control_dead_rate − monitored_dead_rate) / (control_dead_rate + ε)
          Positive → intervention reduced dead experts.

      loss_improvement_ratio:
          (control_loss_auc − monitored_loss_auc) / (control_loss_auc + ε)
          AUC computed over the post-collapse window [COLLAPSE_STEP, N_STEPS].
          Positive → intervention kept loss lower.

  intervention_cost
      Mean extra wall-clock time per step (ms) when interventions are active,
      normalised by the per-step baseline time so cost ∈ [0, ∞).
      cost = (mean_monitored_step_ms − mean_control_step_ms) / mean_control_step_ms
      Clamped to ε to avoid division-by-zero in IES.

  IES
      stability_improvement / max(intervention_cost, ε)
      Higher IES = more stability gain per unit of extra compute.

  Action types
  ------------
  Four action types from moewatch.intervention.actions:
    aux_loss       — bump model.config.router_aux_loss_coef by +delta
    router_noise   — inject Gaussian noise into router logits
    expert_dropout — raise per-expert dropout probability
    noop           — conservative baseline; does nothing

  For each action type we run N_TRIALS independent seeds with that type
  forced as the sole policy output (via ForcedPolicy), plus a matching
  no-intervention control run using the same seed.

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
  Batch     : 2 × 32 = 64 tokens/step
  Trials    : 5 seeds per action type  (4 types × 5 = 20 total experiments)
  Steps/run : 150  (collapse injected at step 50)
================================================================================
"""

from __future__ import annotations

import json
import logging
import math
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
# Configuration
# ---------------------------------------------------------------------------

NUM_LAYERS        = 28
NUM_EXPERTS       = 64
HIDDEN_DIM        = 64
TOP_K             = 8
VOCAB_SIZE        = 1000
BATCH_SIZE        = 2
SEQ_LEN           = 32

N_TRIALS          = 5       # seeds per action type
N_STEPS           = 150     # steps per run
COLLAPSE_STEP     = 50      # inject collapse at this step
COLLAPSE_LAYER    = 0
COLLAPSE_STRENGTH = 10.0

# Dead-expert threshold: expert is "dead" if mean routing prob < this
DEAD_EXPERT_THRESHOLD = 1.0 / NUM_EXPERTS / 4   # << 1/64

EPS = 1e-8


# ---------------------------------------------------------------------------
# Model (same as lead_time_benchmark — ModuleList kept for gradient hooks)
# ---------------------------------------------------------------------------

class _FakeQwen3Config:
    router_aux_loss_coef: float = 0.001


class _ExpertMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2, bias=False)
        self.w2 = nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)))


class _MoEBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate    = nn.Linear(HIDDEN_DIM, NUM_EXPERTS, bias=False)
        self.experts = nn.ModuleList([_ExpertMLP() for _ in range(NUM_EXPERTS)])
        self.register_buffer("collapse_bias", torch.zeros(NUM_EXPERTS))
        self._cached_routing_probs: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        x_flat  = x.reshape(-1, D)
        T       = x_flat.shape[0]

        logits = self.gate(x_flat) + self.collapse_bias
        probs  = torch.softmax(logits, -1)
        self._cached_routing_probs = probs.detach().mean(0)

        top_probs, top_idx = probs.topk(TOP_K, dim=-1)
        top_probs = top_probs / top_probs.sum(-1, keepdim=True)

        # Batched BMM dispatch
        w1 = torch.stack([e.w1.weight.T for e in self.experts])
        w2 = torch.stack([e.w2.weight.T for e in self.experts])

        flat_idx    = top_idx.reshape(-1)
        flat_weight = top_probs.reshape(-1, 1)
        tok_ids     = (torch.arange(T, device=x.device)
                       .unsqueeze(1).expand(T, TOP_K).reshape(-1))
        flat_x      = x_flat[tok_ids]

        h = torch.bmm(flat_x.unsqueeze(1), w1[flat_idx]).squeeze(1)
        h = F.silu(h)
        h = torch.bmm(h.unsqueeze(1), w2[flat_idx]).squeeze(1)

        out = torch.zeros_like(x_flat)
        out.scatter_add_(0, tok_ids.unsqueeze(1).expand_as(h), h * flat_weight)
        return out.reshape(B, S, D)

    def _inject_collapse(self) -> None:
        self.collapse_bias[0] = COLLAPSE_STRENGTH

    def routing_probs(self) -> Optional[torch.Tensor]:
        return self._cached_routing_probs


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

    def inject_collapse(self) -> None:
        self.layers[COLLAPSE_LAYER].mlp._inject_collapse()

    def dead_expert_rate(self) -> float:
        """Fraction of experts with mean routing prob below DEAD_EXPERT_THRESHOLD."""
        probs = self.layers[COLLAPSE_LAYER].mlp.routing_probs()
        if probs is None:
            return 0.0
        dead = (probs < DEAD_EXPERT_THRESHOLD).sum().item()
        return dead / NUM_EXPERTS


class _DummyTrainer:
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def add_callback(self, cb) -> None:
        pass

    def compute_loss(self, model, inputs, return_outputs=False):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Forced policy — always returns a fixed action type for the target layer
# ---------------------------------------------------------------------------

class _ForcedPolicy:
    """
    Minimal policy shim that instructs MoEWatch to always apply a fixed
    action type on the collapse layer, regardless of the risk score.

    Mimics the PolicyBase interface: select() returns an action, update()
    is a no-op (no learning).
    """

    def __init__(self, action_type: str, layer_name: str) -> None:
        from moewatch.intervention.actions import (
            AuxLossAction, RouterNoiseAction, ExpertDropoutAction, NoOpAction,
        )
        self._action_type = action_type
        self._layer_name  = layer_name
        self._action_map  = {
            "aux_loss":       lambda: AuxLossAction(layer_name=layer_name, delta=0.05),
            "router_noise":   lambda: RouterNoiseAction(layer_name=layer_name, noise_scale=0.1),
            "expert_dropout": lambda: ExpertDropoutAction(layer_name=layer_name, dropout_delta=0.05),
            "noop":           lambda: NoOpAction(layer_name=layer_name),
        }

    def select_action(self, state: object) -> object:
        layer_name = getattr(state, "layer_name", "")
        if layer_name == self._layer_name:
            return self._action_map.get(self._action_type,
                                        self._action_map["noop"])()
        from moewatch.intervention.actions import NoOpAction
        return NoOpAction(layer_name=layer_name or "<none>")

    def update(self, *args, **kwargs) -> None:
        pass


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _auc(values: List[float]) -> float:
    """Trapezoidal AUC (unnormalised, just sum for equal-spaced steps)."""
    if not values:
        return 0.0
    return sum(values) / len(values)   # mean == normalised AUC


def _dead_expert_rate_reduction(ctrl_rates: List[float], mon_rates: List[float]) -> float:
    ctrl_mean = _auc(ctrl_rates)
    mon_mean  = _auc(mon_rates)
    return (ctrl_mean - mon_mean) / (ctrl_mean + EPS)


def _loss_improvement_ratio(ctrl_losses: List[float], mon_losses: List[float]) -> float:
    ctrl_auc = _auc(ctrl_losses)
    mon_auc  = _auc(mon_losses)
    return (ctrl_auc - mon_auc) / (ctrl_auc + EPS)


def _stability_improvement(dead_reduction: float, loss_ratio: float) -> float:
    return 0.6 * max(dead_reduction, 0.0) + 0.4 * max(loss_ratio, 0.0)


def _intervention_cost(ctrl_ms: float, mon_ms: float) -> float:
    return (mon_ms - ctrl_ms) / (ctrl_ms + EPS)


def _ies(stability: float, cost: float) -> float:
    return stability / max(cost, EPS)


# ---------------------------------------------------------------------------
# Single run — returns per-step loss + dead_rate + mean step time (ms)
# ---------------------------------------------------------------------------

def _run(
    seed: int,
    device: torch.device,
    action_type: Optional[str],   # None = control (no interventions)
    target_layer_name: str,
) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    model     = _FakeQwen3MoEModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=2e-5, momentum=0.9)

    losses_post:    List[float] = []   # post-collapse only
    dead_rates_post: List[float] = []
    step_times_ms:  List[float] = []
    n_interventions = 0

    from moewatch import MoEWatch
    from moewatch.config import WatchConfig, OutputMode

    _common_config = dict(
        output=OutputMode.SILENT,
        sample_every=1,
        log_every=1,
        # gradient health — calibrated for hidden_dim=64, top-8/64
        dead_threshold=0.002,
        cold_threshold=0.008,
        # imbalance thresholds
        load_imbalance_warn=2.5,
        load_imbalance_error=4.0,
        # fast-response rolling window
        stats_window=20,
        # suppress CUSUM false-positives before collapse window
        cusum_warmup=COLLAPSE_STEP - 5,
    )

    if action_type is not None:
        config = WatchConfig(
            **_common_config,
            intervention_enabled=True,
            intervention_cooldown=5,
            intervention_max_delta=0.5,
        )
        watcher = MoEWatch(model=model, config=config)
        # Inject the forced policy after attach() so it overrides the default
        trainer = _DummyTrainer(model)
        watcher.attach(trainer)
        forced_policy = _ForcedPolicy(
            action_type=action_type,
            layer_name=target_layer_name,
        )
        watcher.policy = forced_policy
    else:
        config = WatchConfig(
            **_common_config,
            intervention_enabled=False,
        )
        watcher = MoEWatch(model=model, config=config)
        trainer = _DummyTrainer(model)
        watcher.attach(trainer)

    for step in range(1, N_STEPS + 1):
        if step == COLLAPSE_STEP:
            model.inject_collapse()

        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        labels    = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)

        t0 = time.perf_counter()

        watcher.pre_step(step)
        optimizer.zero_grad()
        logits = model(input_ids)
        loss   = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        loss.backward()
        optimizer.step()
        report = watcher.step(global_step=step, current_loss=loss.item())

        step_ms = (time.perf_counter() - t0) * 1000.0
        step_times_ms.append(step_ms)

        if step >= COLLAPSE_STEP:
            losses_post.append(loss.item())
            dead_rates_post.append(model.dead_expert_rate())
            n_interventions += report.num_interventions

    watcher.stop()

    return {
        "seed":             seed,
        "action_type":      action_type or "control",
        "losses_post":      losses_post,
        "dead_rates_post":  dead_rates_post,
        "mean_step_ms":     statistics.mean(step_times_ms),
        "n_interventions":  n_interventions,
    }


# ---------------------------------------------------------------------------
# Per-action-type experiment
# ---------------------------------------------------------------------------

def _run_action_type(
    action_type: str,
    device: torch.device,
    target_layer_name: str,
) -> Dict:
    """
    Runs N_TRIALS paired experiments (monitored + control) for one action type.
    Returns aggregated IES metrics.
    """
    stabilities:  List[float] = []
    costs:        List[float] = []
    ies_values:   List[float] = []
    dead_reductions:  List[float] = []
    loss_ratios:      List[float] = []
    per_seed: List[Dict] = []

    for seed in range(N_TRIALS):
        # Control run (no interventions)
        ctrl = _run(seed, device, action_type=None,        target_layer_name=target_layer_name)
        # Monitored run (forced action type)
        mon  = _run(seed, device, action_type=action_type, target_layer_name=target_layer_name)

        dead_red = _dead_expert_rate_reduction(ctrl["dead_rates_post"], mon["dead_rates_post"])
        loss_rat = _loss_improvement_ratio(ctrl["losses_post"],         mon["losses_post"])
        stab     = _stability_improvement(dead_red, loss_rat)
        cost     = _intervention_cost(ctrl["mean_step_ms"],             mon["mean_step_ms"])
        ies      = _ies(stab, cost)

        dead_reductions.append(dead_red)
        loss_ratios.append(loss_rat)
        stabilities.append(stab)
        costs.append(cost)
        ies_values.append(ies)

        per_seed.append({
            "seed":              seed,
            "dead_reduction":    dead_red,
            "loss_ratio":        loss_rat,
            "stability":         stab,
            "cost":              cost,
            "ies":               ies,
            "ctrl_mean_step_ms": ctrl["mean_step_ms"],
            "mon_mean_step_ms":  mon["mean_step_ms"],
            "n_interventions":   mon["n_interventions"],
        })

    def _s(lst: List[float]) -> Tuple[float, float]:
        return statistics.mean(lst), (statistics.stdev(lst) if len(lst) > 1 else 0.0)

    stab_mean, stab_std  = _s(stabilities)
    cost_mean, cost_std  = _s(costs)
    ies_mean,  ies_std   = _s(ies_values)
    dead_mean, dead_std  = _s(dead_reductions)
    loss_mean, loss_std  = _s(loss_ratios)

    return {
        "action_type":         action_type,
        "n_trials":            N_TRIALS,
        "dead_reduction":      {"mean": dead_mean, "std": dead_std},
        "loss_ratio":          {"mean": loss_mean, "std": loss_std},
        "stability":           {"mean": stab_mean, "std": stab_std},
        "cost":                {"mean": cost_mean, "std": cost_std},
        "ies":                 {"mean": ies_mean,  "std": ies_std},
        "per_seed":            per_seed,
    }


# ---------------------------------------------------------------------------
# Discover the collapse layer's gate module path
# ---------------------------------------------------------------------------

def _find_gate_name(model: nn.Module) -> str:
    """Return the dotted module path of the gate linear in COLLAPSE_LAYER."""
    for name, _ in model.named_modules():
        # Matches pattern: layers.0.mlp.gate
        if f"layers.{COLLAPSE_LAYER}." in name and name.endswith(".gate"):
            return name
    # Fallback
    return f"layers.{COLLAPSE_LAYER}.mlp.gate"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ACTION_TYPES = ["aux_loss", "router_noise", "expert_dropout", "noop"]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Discover the gate module path on a temporary model
    _tmp = _FakeQwen3MoEModel()
    target_layer_name = _find_gate_name(_tmp)
    del _tmp

    print(f"\n{'='*72}")
    print(f"  MoEWatch v0.2.0 — Intervention Efficiency Score (IES) Benchmark")
    print(f"{'='*72}")
    print(f"  Device        : {device}")
    print(f"  Model         : FakeQwen3MoEModel  "
          f"({NUM_LAYERS}L | {NUM_EXPERTS}E | top-{TOP_K} | dim={HIDDEN_DIM})")
    print(f"  Batch         : {BATCH_SIZE} × {SEQ_LEN} = {BATCH_SIZE * SEQ_LEN} tokens/step")
    print(f"  Action types  : {ACTION_TYPES}")
    print(f"  Trials/type   : {N_TRIALS} seeds  ×  2 runs (monitored + control)")
    print(f"  Steps/run     : {N_STEPS}  (collapse @ step {COLLAPSE_STEP})")
    print(f"  Target layer  : {target_layer_name}")
    print(f"  IES formula   : stability_improvement / intervention_cost")
    print(f"{'='*72}\n")

    t0 = time.perf_counter()
    all_results: List[Dict] = []

    for action_type in ACTION_TYPES:
        print(f"  [{action_type.upper():<14}]  running {N_TRIALS} trials ...")
        sys.stdout.flush()

        result = _run_action_type(action_type, device, target_layer_name)
        all_results.append(result)

        r = result
        print(
            f"    dead_reduction: {r['dead_reduction']['mean']:+.4f} ± {r['dead_reduction']['std']:.4f}  |  "
            f"loss_ratio: {r['loss_ratio']['mean']:+.4f} ± {r['loss_ratio']['std']:.4f}"
        )
        print(
            f"    stability:      {r['stability']['mean']:.4f} ± {r['stability']['std']:.4f}  |  "
            f"cost: {r['cost']['mean']:+.4f} ± {r['cost']['std']:.4f}  |  "
            f"IES: {r['ies']['mean']:.4f} ± {r['ies']['std']:.4f}"
        )
        print()

    elapsed = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print(f"{'='*72}")
    print(f"  IES Summary  ({N_TRIALS} trials × 4 action types, collapse @ step {COLLAPSE_STEP})")
    print(f"{'='*72}")
    print(f"  {'Action':<16} {'Stability ↑':>12} {'Cost':>10} {'IES ↑':>12}  {'Rank':>5}")
    print(f"  {'-'*56}")

    ranked = sorted(all_results, key=lambda r: r["ies"]["mean"], reverse=True)
    rank_map = {r["action_type"]: i + 1 for i, r in enumerate(ranked)}

    for r in all_results:
        print(
            f"  {r['action_type']:<16} "
            f"{r['stability']['mean']:>10.4f}   "
            f"{r['cost']['mean']:>10.4f}   "
            f"{r['ies']['mean']:>10.4f}    "
            f"#{rank_map[r['action_type']]}"
        )

    best = ranked[0]
    print(f"  {'-'*56}")
    print(f"\n  Best action type: {best['action_type'].upper()}  "
          f"(IES = {best['ies']['mean']:.4f} ± {best['ies']['std']:.4f})")
    print(f"  Total wall time : {elapsed:.1f}s")
    print(f"{'='*72}\n")

    # ------------------------------------------------------------------
    # Write JSON
    # ------------------------------------------------------------------
    results = {
        "config": {
            "n_trials":          N_TRIALS,
            "n_steps":           N_STEPS,
            "collapse_step":     COLLAPSE_STEP,
            "collapse_layer":    COLLAPSE_LAYER,
            "collapse_strength": COLLAPSE_STRENGTH,
            "target_layer":      target_layer_name,
            "batch_size":        BATCH_SIZE,
            "seq_len":           SEQ_LEN,
            "num_layers":        NUM_LAYERS,
            "num_experts":       NUM_EXPERTS,
            "hidden_dim":        HIDDEN_DIM,
            "top_k":             TOP_K,
            "device":            str(device),
            "dead_expert_threshold": DEAD_EXPERT_THRESHOLD,
            "ies_formula": {
                "stability": "0.6 * dead_reduction + 0.4 * loss_ratio",
                "cost":      "(mon_step_ms - ctrl_step_ms) / ctrl_step_ms",
                "ies":       "stability / max(cost, eps)",
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
        "action_types":      all_results,
        "ranking": [
            {
                "rank":        rank_map[r["action_type"]],
                "action_type": r["action_type"],
                "ies_mean":    r["ies"]["mean"],
                "ies_std":     r["ies"]["std"],
            }
            for r in ranked
        ],
    }

    out_path = Path(__file__).parent / "results" / "ies_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"  Results saved → {out_path}")


if __name__ == "__main__":
    main()
