"""
moe_model.py
============
A tiny, from-scratch Mixture-of-Experts language model used to demonstrate
MoEWatch on a *small custom model* (no pretrained/large models involved).

The model and the synthetic training corpus are shared by both comparison
scripts so the ONLY difference between runs is whether MoEWatch is present.

Dead-expert calibration:
  eval_batch = 128 seqs x SEQ_LEN=16 = 2048 token routing decisions.
  dead_threshold in expert_usage() = 0.02.

  Bias -> dominant expert fraction -> dead experts (usage < 0.02):
    bias=1.0 -> ~75% dominant, 0 dead
    bias=1.5 -> ~91% dominant, 0 dead (but 3 dead at <0.05 threshold)
    bias=2.0 -> ~98% dominant, 3 dead  <- target: used as peak

  PRESSURE_SCHEDULE ramps 0->peak, then HOLDS at peak (no reset).
  This models a real-world input-distribution shift that permanently
  corrupts the gate. Without MoEWatch: dead experts remain dead.
  With MoEWatch: AuxLoss+RouterNoise prevent the gate weights from
  learning to rely on the bias, so bias removal (which MoEWatch doesn't
  do -- it only acts on the training dynamics) still leaves a healthy gate.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Fixed hyperparameters
# ---------------------------------------------------------------------------

SEED            = 1234
VOCAB_SIZE      = 64
HIDDEN_DIM      = 32
NUM_LAYERS      = 3
NUM_EXPERTS     = 4
TOP_K           = 1
SEQ_LEN         = 16
BATCH_SIZE      = 16
TOTAL_STEPS     = 400
LR              = 3e-3
EVAL_BATCH_SIZE = 128   # 128 x 16 = 2048 routing decisions: stable counts

# ---------------------------------------------------------------------------
# Collapse-pressure schedule.
#
# Pressure ramps linearly from 0 -> peak over [start, end], then HOLDS
# at peak for the rest of training.  This models a sustained distribution
# shift (e.g. domain mismatch, long-tailed data imbalance).
#
# Without MoEWatch: the gate weights learn to rely on the biased logit,
#   driving 3 non-dominant experts to near-zero usage (dead).
# With MoEWatch: AuxLoss + RouterNoise counteract the bias during the ramp,
#   keeping gate weights from collapsing, so even at held peak the
#   router stays healthy.
#
# Staggered windows: layers 0, 1, 2 start their ramp at different steps
# so the plot shows a clear sequential story before the final held state
# puts all three under simultaneous stress.
# ---------------------------------------------------------------------------
PRESSURE_SCHEDULE = [
    # (layer_idx, dominant_expert, ramp_start, ramp_end, peak_strength)
    # peak=2.0-2.2 calibrated to produce 3 dead experts without MoEWatch.
    (0, 3,  40, 150, 2.0),   # layer 0 ramps early
    (1, 0,  80, 200, 2.2),   # layer 1 ramps mid-training
    (2, 1, 120, 260, 2.0),   # layer 2 ramps late (overlap with L0,L1)
]


# ---------------------------------------------------------------------------
# Synthetic corpus
# ---------------------------------------------------------------------------

def build_markov_transition(seed: int = SEED) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    trans = torch.rand(VOCAB_SIZE, VOCAB_SIZE, generator=g) * 0.05
    cycle = list(range(8))
    for i, tok in enumerate(cycle):
        nxt = cycle[(i + 1) % len(cycle)]
        trans[tok, nxt] += 3.0
    trans = trans / trans.sum(dim=-1, keepdim=True)
    return trans


def sample_batch(trans: torch.Tensor, batch_size: int, seq_len: int,
                  generator: torch.Generator) -> torch.Tensor:
    seqs = torch.zeros(batch_size, seq_len + 1, dtype=torch.long)
    seqs[:, 0] = torch.randint(0, VOCAB_SIZE, (batch_size,), generator=generator)
    for t in range(seq_len):
        probs = trans[seqs[:, t]]
        seqs[:, t + 1] = torch.multinomial(probs, num_samples=1,
                                            generator=generator).squeeze(-1)
    return seqs


# ---------------------------------------------------------------------------
# Tiny MoE model
# ---------------------------------------------------------------------------

class Expert(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.up      = nn.Linear(dim, dim * 4)
        self.dropout = nn.Dropout(p=0.0)   # ExpertDropoutAction target
        self.down    = nn.Linear(dim * 4, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.dropout(F.gelu(self.up(x))))


class MoEBlock(nn.Module):
    def __init__(self, dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        self.gate        = nn.Linear(dim, num_experts, bias=True)
        self.experts     = nn.ModuleList([Expert(dim) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        flat    = x.reshape(-1, D)
        logits  = self.gate(flat)
        probs   = torch.softmax(logits, dim=-1)
        top_p, top_i = probs.topk(self.top_k, dim=-1)
        top_p   = top_p / top_p.sum(-1, keepdim=True)

        out = torch.zeros_like(flat)
        for k in range(self.top_k):
            for e in range(self.num_experts):
                mask = top_i[:, k] == e
                if mask.any():
                    out[mask] += top_p[mask, k:k+1] * self.experts[e](flat[mask])

        with torch.no_grad():
            assign_onehot = F.one_hot(top_i[:, 0], num_classes=self.num_experts).float()
            f_e = assign_onehot.mean(dim=0)
        P_e = probs.mean(dim=0)
        self.last_aux_loss = self.num_experts * (f_e * P_e).sum()

        return out.reshape(B, S, D)


class TinyBlock(nn.Module):
    def __init__(self, dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.qkv   = nn.Linear(dim, dim * 3, bias=False)
        self.proj  = nn.Linear(dim, dim, bias=False)
        self.moe   = MoEBlock(dim, num_experts, top_k)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        attn = torch.softmax(q @ k.transpose(-2, -1) / (q.size(-1) ** 0.5), dim=-1)
        causal = torch.tril(torch.ones(x.size(1), x.size(1), device=x.device)).bool()
        attn   = attn.masked_fill(~causal, 0.0)
        attn   = attn / attn.sum(-1, keepdim=True).clamp(min=1e-6)
        x = x + self.proj(attn @ v)
        x = x + self.moe(self.norm2(x))
        return x


class TinyMoELM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed  = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.blocks = nn.ModuleList([
            TinyBlock(HIDDEN_DIM, NUM_EXPERTS, TOP_K) for _ in range(NUM_LAYERS)
        ])
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.head = nn.Linear(HIDDEN_DIM, VOCAB_SIZE, bias=False)
        self.config = MagicMock()
        self.config.router_aux_loss_coef = 0.0

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        aux_total = 0.0
        for block in self.blocks:
            x = block(x)
            aux_total = aux_total + block.moe.last_aux_loss
        self.last_aux_loss_total = aux_total
        return self.head(self.norm(x))

    def gate_names(self):
        return [f"blocks.{i}.moe.gate" for i in range(len(self.blocks))]

    def inject_collapse_pressure(self, layer_idx: int, dominant_expert: int,
                                  strength: float) -> None:
        """SET the gate bias for dominant_expert to strength (token-independent logit shift)."""
        with torch.no_grad():
            gate = self.blocks[layer_idx].moe.gate
            gate.bias[dominant_expert] = strength

    def apply_pressure_schedule(self, step: int) -> str:
        """Ramp bias 0->peak over [ramp_start, ramp_end], then HOLD at peak.

        The hold models a sustained distribution shift that permanently
        biases the router. Without MoEWatch, gate weights adapt to this
        bias and stay collapsed even if the bias were later removed.
        With MoEWatch, interventions prevent that adaptation.
        """
        active_labels = []
        for layer_idx, dom, ramp_start, ramp_end, peak in PRESSURE_SCHEDULE:
            if step < ramp_start:
                strength = 0.0
            elif step <= ramp_end:
                frac     = (step - ramp_start) / max(ramp_end - ramp_start, 1)
                strength = frac * peak
                active_labels.append(f"L{layer_idx}↑")   # ramping up
            else:
                strength = peak                            # held at peak
                active_labels.append(f"L{layer_idx}●")   # held
            self.inject_collapse_pressure(layer_idx, dom, strength=strength)

        if not active_labels:
            return "normal"
        return " ".join(active_labels)

    @torch.no_grad()
    def expert_usage(self, input_ids: torch.Tensor):
        usage   = {}
        entropy = {}
        x = self.embed(input_ids)
        for i, block in enumerate(self.blocks):
            h = block.norm1(x)
            q, k, v = block.qkv(h).chunk(3, dim=-1)
            attn = torch.softmax(q @ k.transpose(-2, -1) / (q.size(-1) ** 0.5), dim=-1)
            causal = torch.tril(torch.ones(x.size(1), x.size(1))).bool()
            attn   = attn.masked_fill(~causal, 0.0)
            attn   = attn / attn.sum(-1, keepdim=True).clamp(min=1e-6)
            x_attn = x + block.proj(attn @ v)
            flat   = block.norm2(x_attn).reshape(-1, HIDDEN_DIM)

            logits = block.moe.gate(flat)
            probs  = torch.softmax(logits, dim=-1)
            top_i  = probs.topk(block.moe.top_k, dim=-1).indices
            counts = torch.zeros(NUM_EXPERTS)
            for e in range(NUM_EXPERTS):
                counts[e] = (top_i == e).sum().item()
            frac = counts / counts.sum().clamp(min=1)
            usage[f"blocks.{i}.moe.gate"] = frac.tolist()

            p   = frac.clamp(min=1e-9)
            ent = -(p * p.log()).sum() / torch.log(torch.tensor(float(NUM_EXPERTS)))
            entropy[f"blocks.{i}.moe.gate"] = ent.item()

            x = x_attn + block.moe(block.norm2(x_attn))
        return usage, entropy
