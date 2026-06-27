# =============================================================================
# MoEWatch v0.2.0 — Research Paper Test Suite
# Test 02: Routing Statistics Collection
# =============================================================================
#
# Verifies that MoEWatch correctly captures routing events and computes
# accurate load distribution statistics:
#
#   1. Routing events accumulate in stat_collector after forward passes
#   2. Load imbalance ratio = 1.0 for perfectly uniform routing
#   3. Load imbalance ratio is HIGH when one expert dominates
#   4. Expert utilization vector sums to 1.0
#   5. Raw logits are captured and have correct shape (num_experts,)
#
# =============================================================================

import torch
import torch.nn as nn

from moewatch import MoEWatch
from moewatch.config import WatchConfig, OutputMode

# ---------------------------------------------------------------------------
# Controllable MoE model — routing behaviour injected externally
# ---------------------------------------------------------------------------

class ControlledRouter(nn.Module):
    """Gate whose output distribution is set manually via .set_logits()."""
    def __init__(self, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.out_features = num_experts  # needed by HookManager._infer_expert_count
        self.top_k        = top_k
        # Learnable weight (required by MoEWatch hook); we override output.
        self.weight       = nn.Parameter(torch.zeros(num_experts, 16))
        self._fixed_logits: torch.Tensor | None = None

    def set_logits(self, logits: torch.Tensor) -> None:
        """Fix output logits for all tokens on the next forward pass."""
        self._fixed_logits = logits.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[0]  # num tokens
        if self._fixed_logits is not None:
            return self._fixed_logits.unsqueeze(0).expand(T, -1)
        return x @ self.weight.T


class ControlledMoELayer(nn.Module):
    def __init__(self, dim: int = 16, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.gate    = ControlledRouter(num_experts, top_k)
        self.experts = nn.ModuleList([
            nn.Linear(dim, dim, bias=False) for _ in range(num_experts)
        ])
        self.top_k   = top_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        flat    = x.reshape(-1, D)
        logits  = self.gate(flat)
        probs   = torch.softmax(logits, dim=-1)
        top_p, top_idx = probs.topk(self.top_k, dim=-1)
        top_p   = top_p / top_p.sum(-1, keepdim=True)
        out     = torch.zeros_like(flat)
        for k in range(self.top_k):
            for e in range(len(self.experts)):
                mask = top_idx[:, k] == e
                if mask.any():
                    out[mask] += top_p[mask, k:k+1] * self.experts[e](flat[mask])
        return out.reshape(B, S, D)


class SimpleModel(nn.Module):
    def __init__(self, top_k: int = 2):
        super().__init__()
        self.moe = ControlledMoELayer(dim=16, num_experts=8, top_k=top_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.moe(x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_watcher(model):
    config = WatchConfig(output=OutputMode.SILENT, intervention_enabled=False,
                         sample_every=1, log_every=1,
                         router_modules=["gate"])
    w = MoEWatch(model, config)
    w.start()
    return w


def run_forward(model, watcher, n_steps: int = 10,
                batch: int = 4, seq: int = 8):
    for step in range(1, n_steps + 1):
        watcher.pre_step(step)
        x = torch.randn(batch, seq, 16)
        with torch.no_grad():
            model(x)
        watcher.step(step)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def run():
    print("=" * 60)
    print("  MoEWatch Test 02 — Routing Statistics Collection")
    print("=" * 60)

    NUM_EXPERTS = 8
    TOP_K       = 2
    torch.manual_seed(1)

    # ==================================================================
    # Scenario A — Near-uniform routing (random weights, many tokens)
    # Natural routing from random normal gate weights produces near-
    # uniform load distribution across experts over many steps.
    # ==================================================================
    print("\n  [Scenario A] Near-uniform routing (random gate weights, top_k=2)")

    model_a   = SimpleModel(top_k=2)
    watcher_a = make_watcher(model_a)

    # Do NOT fix logits — let the random normal gate weights (default init)
    # produce natural routing. With enough tokens this converges to ~uniform.
    # We run 40 steps × batch=4 × seq=8 = 1280 tokens total.
    run_forward(model_a, watcher_a, n_steps=40, batch=4, seq=8)

    sc_a       = watcher_a.hook_manager.stat_collector
    layer_name = watcher_a._layer_order[0]
    stats_a    = sc_a.get_layer_stats(layer_name)

    print(f"    layer              : {layer_name}")
    print(f"    events collected   : {stats_a.step}  (expected ≥ 20)")
    print(f"    load_imbalance_ratio: {stats_a.load_imbalance_ratio:.4f}  (expected < 6.0 for healthy routing)")
    print(f"    utilization sum    : {stats_a.expert_utilization.sum().item():.4f}  (expected 1.0)")

    assert stats_a.step >= 40, "Should have collected ≥ 40 routing events"
    assert abs(stats_a.expert_utilization.sum().item() - 1.0) < 1e-4, "Utilization must sum to 1.0"
    assert stats_a.load_imbalance_ratio < 6.0, \
        f"Healthy routing imbalance should be < 6.0, got {stats_a.load_imbalance_ratio:.3f}"
    print(f"    ✓ Healthy routing: moderate imbalance, utilization sums to 1.0")

    watcher_a.stop()

    # ==================================================================
    # Scenario B — Monopoly routing (expert 0 gets all tokens, top_k=1)
    # ==================================================================
    print("\n  [Scenario B] Monopoly routing (expert 0 dominates, top_k=1)")

    # top_k=1: only one expert selected per token — expert 0 with logit=10
    # gets 100% of tokens. utilization[0] ≈ 1.0, all others ≈ 0.0.
    model_b   = SimpleModel(top_k=1)
    watcher_b = make_watcher(model_b)

    # Expert 0 gets logit=10, all others get 0 → softmax ≈ [1, 0, 0, ...]
    monopoly_logits = torch.zeros(NUM_EXPERTS)
    monopoly_logits[0] = 10.0
    model_b.moe.gate.set_logits(monopoly_logits)
    run_forward(model_b, watcher_b, n_steps=40)

    sc_b    = watcher_b.hook_manager.stat_collector
    stats_b = sc_b.get_layer_stats(layer_name)

    print(f"    load_imbalance_ratio: {stats_b.load_imbalance_ratio:.4f}  (expected >> 3.0)")
    print(f"    utilization[0]     : {stats_b.expert_utilization[0].item():.4f}  (expected ≈ 1.0)")
    print(f"    utilization[1:]    : {stats_b.expert_utilization[1:].sum().item():.4f}  (expected ≈ 0.0)")

    assert stats_b.load_imbalance_ratio > 3.0, \
        f"Monopoly should produce high imbalance, got {stats_b.load_imbalance_ratio:.3f}"
    assert stats_b.expert_utilization[0].item() > 0.8, \
        f"Expert 0 should dominate, got {stats_b.expert_utilization[0].item():.3f}"
    print(f"    ✓ Monopoly routing: high imbalance, expert 0 dominates")

    watcher_b.stop()

    # ==================================================================
    # Scenario C — Raw logits shape check
    # ==================================================================
    print("\n  [Scenario C] Raw logits shape verification")

    model_c   = SimpleModel(top_k=2)
    watcher_c = make_watcher(model_c)

    custom_logits = torch.linspace(-2, 2, NUM_EXPERTS)
    model_c.moe.gate.set_logits(custom_logits)
    run_forward(model_c, watcher_c, n_steps=10)

    sc_c    = watcher_c.hook_manager.stat_collector
    stats_c = sc_c.get_layer_stats(layer_name)

    raw = stats_c.raw_logits_window
    print(f"    raw_logits_window shape: {tuple(raw.shape)}")
    print(f"    last logits (mean over tokens): {raw[-1].mean(0).tolist()[:4]} ...")

    assert raw.shape[-1] == NUM_EXPERTS, \
        f"Last dim should be num_experts={NUM_EXPERTS}, got {raw.shape[-1]}"
    print(f"    ✓ Logits captured with correct expert dimension")

    watcher_c.stop()

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n  All assertions passed.")
    print(f"  Imbalance ratio — Uniform: {stats_a.load_imbalance_ratio:.3f}  "
          f"Monopoly: {stats_b.load_imbalance_ratio:.3f}  "
          f"(ratio increase: {stats_b.load_imbalance_ratio / max(stats_a.load_imbalance_ratio, 0.01):.1f}×)")
    print("=" * 60)


if __name__ == "__main__":
    run()
