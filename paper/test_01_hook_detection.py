# =============================================================================
# MoEWatch v0.2.0 — Research Paper Test Suite
# Test 01: Hook Attachment & Router Detection
# =============================================================================
#
# Verifies that MoEWatch correctly:
#   1. Auto-detects MoE router layers by name heuristic
#   2. Attaches forward hooks without modifying model weights
#   3. Reports the correct number of experts per layer
#   4. Detaches all hooks cleanly after stop()
#
# =============================================================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from moewatch import MoEWatch
from moewatch.config import WatchConfig, OutputMode

# ---------------------------------------------------------------------------
# Minimal fake MoE model (no downloads required)
# ---------------------------------------------------------------------------

class FakeExpert(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.fc(x)


class FakeMoELayer(nn.Module):
    def __init__(self, dim: int = 32, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.gate    = nn.Linear(dim, num_experts, bias=False)
        self.gate.top_k = top_k
        self.experts = nn.ModuleList([FakeExpert(dim) for _ in range(num_experts)])
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


class FakeMoEModel(nn.Module):
    """4-layer model: 2 MoE layers (8 experts) + 2 dense layers."""
    def __init__(self, dim: int = 32):
        super().__init__()
        self.embed  = nn.Embedding(1000, dim)
        self.layer0 = FakeMoELayer(dim, num_experts=8,  top_k=2)  # MoE
        self.layer1 = nn.Linear(dim, dim)                          # dense
        self.layer2 = FakeMoELayer(dim, num_experts=16, top_k=4)  # MoE
        self.layer3 = nn.Linear(dim, dim)                          # dense
        self.head   = nn.Linear(dim, 1000, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        x = self.layer0(x)
        x = self.layer1(x.reshape(x.shape[0], -1, x.shape[-1]))
        x = self.layer2(x)
        x = self.layer3(x.reshape(x.shape[0], -1, x.shape[-1]))
        return self.head(x)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def run():
    print("=" * 60)
    print("  MoEWatch Test 01 — Hook Attachment & Router Detection")
    print("=" * 60)

    torch.manual_seed(0)
    model  = FakeMoEModel(dim=32)
    config = WatchConfig(output=OutputMode.SILENT, intervention_enabled=False)

    # ---- 1. Attach watcher ----
    watcher = MoEWatch(model, config)
    watcher.start()

    num_monitored = watcher.num_layers_monitored
    print(f"\n  [1] Layers monitored     : {num_monitored}  (expected 2)")
    assert num_monitored == 2, f"Expected 2 MoE layers, got {num_monitored}"

    # ---- 2. Verify layer names ----
    layers = watcher._layer_order
    print(f"  [2] Monitored layer names:")
    for name in layers:
        print(f"        {name}")
    assert any("layer0" in n for n in layers), "layer0.gate not detected"
    assert any("layer2" in n for n in layers), "layer2.gate not detected"

    # ---- 3. Verify expert counts ----
    print(f"  [3] Expert counts per layer:")
    sc = watcher.hook_manager.stat_collector
    for name in layers:
        n_exp = sc._expert_counts.get(name, "?")
        print(f"        {name:<35} → {n_exp} experts")
    layer0_name = next(n for n in layers if "layer0" in n)
    layer2_name = next(n for n in layers if "layer2" in n)
    assert sc._expert_counts.get(layer0_name) == 8,  "layer0 should have 8 experts"
    assert sc._expert_counts.get(layer2_name) == 16, "layer2 should have 16 experts"

    # ---- 4. Confirm weights unchanged after attach ----
    w_before = model.layer0.gate.weight.data.clone()
    x = torch.randint(0, 1000, (2, 8))
    with torch.no_grad():
        _ = model(x)
    w_after = model.layer0.gate.weight.data
    assert torch.allclose(w_before, w_after), "Gate weights changed after hook!"
    print(f"  [4] Gate weights unchanged after forward pass ✓")

    # ---- 5. Verify dense layers NOT monitored ----
    dense_monitored = any("layer1" in n or "layer3" in n for n in layers)
    assert not dense_monitored, "Dense layers should not be monitored"
    print(f"  [5] Dense layers correctly excluded ✓")

    # ---- 6. Stop and verify hook removal ----
    hook_count_before = len(watcher.hook_manager._handles)
    watcher.stop()
    print(f"  [6] Hooks before stop: {hook_count_before}  |  after stop: {len(watcher.hook_manager._handles)}")
    assert len(watcher.hook_manager._handles) == 0, "Hooks not fully removed after stop()"

    print(f"\n  All 6 assertions passed.")
    print("=" * 60)


if __name__ == "__main__":
    run()
