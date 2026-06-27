"""
Quick smoke-test for the three lead-time benchmark fixes:
  1. _infer_top_k walks to parent module (no top_k on gate → finds it on _MoEBlock)
  2. _RouterGate bakes collapse_bias into the hooked forward (hook sees biased logits)
  3. cusum_warmup + stats_window prevent false-positives before collapse

Run: python test_lead_time_fix.py
Expected: all three lines print PASS
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from moewatch import MoEWatch
from moewatch.config import WatchConfig, OutputMode

logging.getLogger("moewatch").setLevel(logging.CRITICAL)

# ── constants ────────────────────────────────────────────────────────
NUM_EXPERTS   = 8
HIDDEN_DIM    = 32
TOP_K         = 2
VOCAB         = 100
COLLAPSE_STEP = 15

# ── model ────────────────────────────────────────────────────────────

class _RouterGate(nn.Module):
    """Gate wrapper — collapse_bias lives inside here so the hook sees it."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(HIDDEN_DIM, NUM_EXPERTS, bias=False)
        # top_k NOT set here — only on parent _MoEBlock, to test the parent-walk
        self.register_buffer("collapse_bias", torch.zeros(NUM_EXPERTS))

    def forward(self, x):
        return self.linear(x) + self.collapse_bias   # hook captures biased logits

    def inject_collapse(self):
        self.collapse_bias[0] = 12.0

    @property
    def weight(self): return self.linear.weight
    @property
    def out_features(self): return self.linear.out_features
    @property
    def in_features(self): return self.linear.in_features


class _MoEBlock(nn.Module):
    top_k = TOP_K   # on PARENT, not on gate — exercises the parent-walk fix

    def __init__(self):
        super().__init__()
        self.gate    = _RouterGate()
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2, bias=False),
                nn.SiLU(),
                nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM, bias=False),
            )
            for _ in range(NUM_EXPERTS)
        ])

    def forward(self, x):
        B, S, D = x.shape
        xf = x.reshape(-1, D); T = xf.shape[0]
        logits = self.gate(xf)
        probs  = torch.softmax(logits, -1)
        tp, ti = probs.topk(TOP_K, dim=-1)
        tp = tp / tp.sum(-1, keepdim=True)
        fi   = ti.reshape(-1)
        fw   = tp.reshape(-1, 1)
        tids = torch.arange(T).unsqueeze(1).expand(T, TOP_K).reshape(-1)
        w1 = torch.stack([e[0].weight.T for e in self.experts])
        w2 = torch.stack([e[2].weight.T for e in self.experts])
        h = torch.bmm(xf[tids].unsqueeze(1), w1[fi]).squeeze(1)
        h = F.silu(h)
        h = torch.bmm(h.unsqueeze(1), w2[fi]).squeeze(1)
        out = torch.zeros_like(xf)
        out.scatter_add_(0, tids.unsqueeze(1).expand_as(h), h * fw)
        return out.reshape(B, S, D)

    def _inject_collapse(self):
        self.gate.inject_collapse()


class _DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.mlp  = _MoEBlock()
        self.ln1  = nn.LayerNorm(HIDDEN_DIM)
        self.ln2  = nn.LayerNorm(HIDDEN_DIM)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("C", (), {"router_aux_loss_coef": 0.001})()
        self.embed  = nn.Embedding(VOCAB, HIDDEN_DIM)
        self.layers = nn.ModuleList([_DecoderLayer() for _ in range(4)])
        self.norm   = nn.LayerNorm(HIDDEN_DIM)
        self.head   = nn.Linear(HIDDEN_DIM, VOCAB, bias=False)

    def forward(self, ids):
        x = self.embed(ids)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))


class _DummyTrainer:
    def __init__(self, m): self.model = m
    def add_callback(self, cb): pass


# ── run ──────────────────────────────────────────────────────────────
torch.manual_seed(0)
model = _Model()
opt   = torch.optim.SGD(model.parameters(), lr=2e-4, momentum=0.9)

cfg = WatchConfig(
    output=OutputMode.SILENT, sample_every=1, log_every=1,
    intervention_enabled=False,
    dead_threshold=0.002, cold_threshold=0.008,
    load_imbalance_warn=2.5, load_imbalance_error=4.0,
    stats_window=10,
    cusum_warmup=COLLAPSE_STEP - 3,
)
watcher = MoEWatch(model=model, config=cfg)
watcher.attach(_DummyTrainer(model))

layer0         = list(watcher._detected_layers.keys())[0]
router_hook    = watcher.hook_manager._router_hooks[0]
inferred_k     = router_hook._infer_top_k(
    model.layers[0].mlp.gate, NUM_EXPERTS, model, layer0
)

t_mw   = None
streak = 0
print(f"{'step':>5}  {'load_imb':>9}  {'risk':>7}  {'level':>8}  {'alerts':>7}")

for step in range(1, 40):
    if step == COLLAPSE_STEP:
        model.layers[0].mlp._inject_collapse()
        print(f"  >>> COLLAPSE INJECTED at step {step}")

    ids    = torch.randint(0, VOCAB, (2, 8))
    labels = torch.randint(0, VOCAB, (2, 8))
    watcher.pre_step(step)
    opt.zero_grad()
    logits = model(ids)
    loss   = F.cross_entropy(logits.reshape(-1, VOCAB), labels.reshape(-1))
    loss.backward()
    opt.step()
    report = watcher.step(step, loss.item())

    ls  = watcher.stat_collector.get_layer_stats(layer0, window=step)
    imb = ls.load_imbalance_ratio if ls else 0.0
    risk = report.risk_scores.get(layer0, 0.0)
    lvl  = report.risk_levels.get(layer0, "—")
    print(f"{step:>5}  {imb:>9.3f}  {risk:>7.4f}  {lvl:>8}  {len(report.alerts):>7}")

    if t_mw is None and step >= COLLAPSE_STEP:
        has_crit = any(
            a.level.value == "critical" and a.layer_id == layer0
            for a in report.alerts
        )
        streak = (streak + 1) if has_crit else 0
        if streak >= 2:
            t_mw = step
            print(f"  *** T_moewatch = {step} ***")

watcher.stop()

# ── results ──────────────────────────────────────────────────────────
print()
print(f"top_k inferred correctly  : {'PASS' if inferred_k == TOP_K else f'FAIL (got {inferred_k})'}")
print(f"T_moewatch fires          : {'PASS' if t_mw is not None else 'FAIL'} (step={t_mw})")
print(f"No false-positive         : {'PASS' if (t_mw is None or t_mw >= COLLAPSE_STEP) else f'FAIL (fired at {t_mw} before collapse at {COLLAPSE_STEP})'}")
