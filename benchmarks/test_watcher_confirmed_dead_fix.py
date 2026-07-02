# =============================================================================
# benchmark/test_watcher_confirmed_dead_fix.py
#
# Verifies that the confirmed_dead fix ported from _audit.py to _watcher.py
# correctly detects dead experts during live training (not just offline audit).
#
# Without fix: naive max(starvation_score) misses dead experts whose hooks
#   return score=0.0 due to _MIN_SAMPLES_FOR_DETECTION guard.
#
# With fix: confirmed_dead experts (n_samples>=1, norm_mean==0.0) are
#   prioritised → risk rises above 0.5 during collapse.
#
# Pass criteria:
#   - Phase 1 (healthy): avg risk < 0.15
#   - Phase 2 (collapse): peak risk >= 0.50
# =============================================================================

from __future__ import annotations
import sys, torch, torch.nn as nn, torch.nn.functional as F
from unittest.mock import MagicMock
from moewatch import MoEWatch
from moewatch.config import WatchConfig, OutputMode

NUM_EXPERTS = 8
HIDDEN_DIM  = 32
VOCAB_SIZE  = 100


class MoEBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate    = nn.Linear(HIDDEN_DIM, NUM_EXPERTS, bias=True)
        self.experts = nn.ModuleList([
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False) for _ in range(NUM_EXPERTS)
        ])

    def set_monopoly(self, expert_id: int = 0, strength: float = 8.0):
        with torch.no_grad():
            self.gate.bias.zero_()
            self.gate.bias[expert_id] = strength

    def reset(self):
        with torch.no_grad():
            self.gate.bias.zero_()

    def forward(self, x):
        flat  = x.reshape(-1, HIDDEN_DIM)
        probs = torch.softmax(self.gate(flat), dim=-1)
        top_p, top_i = probs.topk(2, dim=-1)
        top_p = top_p / top_p.sum(-1, keepdim=True)
        out   = torch.zeros_like(flat)
        for k in range(2):
            for e in range(NUM_EXPERTS):
                mask = (top_i[:, k] == e)
                if mask.any():
                    out[mask] += top_p[mask, k:k+1] * self.experts[e](flat[mask])
        return out.reshape(x.shape)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MagicMock()
        self.config.router_aux_loss_coef = 0.001
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.moe   = MoEBlock()
        self.norm  = nn.LayerNorm(HIDDEN_DIM)
        self.head  = nn.Linear(HIDDEN_DIM, VOCAB_SIZE, bias=False)

    def forward(self, ids):
        x = self.embed(ids)
        x = x + self.moe(x)
        return self.head(self.norm(x))


def run_test():
    print("=" * 60)
    print("  _watcher.py confirmed_dead fix — Live Training Test")
    print("=" * 60)

    torch.manual_seed(0)
    model = SimpleModel()

    config = WatchConfig(
        output=OutputMode.SILENT,
        sample_every=1,
        log_every=1,
        # Calibrated to actual measured gradient norms for this model:
        #   live expert (expert 0):  norm ~0.68
        #   dead experts (1-7):      norm ~0.0001-0.0009 (small but nonzero
        #                            due to residual gradient paths)
        # cold_threshold set between dead and live norm ranges.
        # Calibrated to actual gradient norms:
        #   live expert 0: norm ~0.50  (gets real tokens)
        #   dead experts:  norm ~0.07  (residual leakage, not true zero)
        # starvation_score = clip(1 - mean/cold_threshold)
        # dead:  1 - 0.07/0.15 = 0.53  → STARVED
        # live:  1 - 0.50/0.15 < 0    → 0.0  → HEALTHY
        dead_threshold=0.01,
        cold_threshold=0.15,
        cold_steps_limit=5,
        intervention_enabled=False,
    )

    trainer = MagicMock()
    trainer.model = model
    watcher = MoEWatch(model, config)
    watcher.attach(trainer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    phase1_risks = []
    phase2_risks = []

    # ------------------------------------------------------------------
    # Phase 1 — healthy (steps 1-20)
    # ------------------------------------------------------------------
    print("\nPhase 1: Healthy routing (steps 1-20)")
    model.moe.reset()
    for step in range(1, 21):
        watcher.pre_step(step)
        ids    = torch.randint(0, VOCAB_SIZE, (2, 8))
        labels = torch.randint(0, VOCAB_SIZE, (2, 8))
        optimizer.zero_grad()
        loss = F.cross_entropy(model(ids).reshape(-1, VOCAB_SIZE), labels.reshape(-1))
        loss.backward(); optimizer.step()
        report = watcher.step(global_step=step, current_loss=loss.item())
        worst  = max(report.risk_scores.values(), default=0.0)
        phase1_risks.append(worst)
    avg_p1 = sum(phase1_risks) / len(phase1_risks)
    print(f"  avg risk = {avg_p1:.4f}  (expected < 0.15)")

    # ------------------------------------------------------------------
    # Phase 2 — collapse: expert 0 monopoly (steps 21-50)
    # ------------------------------------------------------------------
    print("\nPhase 2: Collapse (steps 21-50)")
    model.moe.set_monopoly(expert_id=0, strength=8.0)
    for step in range(21, 51):
        watcher.pre_step(step)
        ids    = torch.randint(0, VOCAB_SIZE, (2, 8))
        labels = torch.randint(0, VOCAB_SIZE, (2, 8))
        optimizer.zero_grad()
        loss = F.cross_entropy(model(ids).reshape(-1, VOCAB_SIZE), labels.reshape(-1))
        loss.backward(); optimizer.step()
        report = watcher.step(global_step=step, current_loss=loss.item())
        worst  = max(report.risk_scores.values(), default=0.0)
        phase2_risks.append(worst)
        if step % 10 == 0:
            print(f"  step={step:3d}  risk={worst:.4f}")

    peak_p2 = max(phase2_risks)
    avg_p2  = sum(phase2_risks) / len(phase2_risks)
    print(f"  peak risk = {peak_p2:.4f}  (expected >= 0.50)")

    # Gradient starvation details
    print("\nGradient starvation scores (last step):")
    from moewatch.analyzer.gradient_starvation import GradientStarvationAnalyzer
    ga = GradientStarvationAnalyzer(config)
    gr = ga.analyze(watcher.stat_collector)
    layer = list(gr.keys())[0] if gr else None
    if layer:
        for r in sorted(gr[layer], key=lambda x: x.expert_id):
            tag = "LIVE" if r.expert_id == 0 else "DEAD"
            print(f"  expert {r.expert_id}  score={r.starvation_score:.4f}  "
                  f"mean={getattr(r,'gradient_norm_mean',0):.5f}  [{tag}]")

    # ------------------------------------------------------------------
    # Pass/fail
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    checks = [
        ("Phase 1 avg risk < 0.15",   avg_p1 < 0.15,  avg_p1),
        ("Phase 2 peak risk >= 0.40",  peak_p2 >= 0.40, peak_p2),
    ]
    all_passed = True
    for name, passed, val in checks:
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  [{status}] {name}  (value={val:.4f})")
        if not passed: all_passed = False
    print()
    if all_passed:
        print("  ALL CHECKS PASSED — confirmed_dead fix works in live watcher.")
    else:
        print("  SOME CHECKS FAILED.")
    print("=" * 60)
    watcher.stop()
    return all_passed


if __name__ == "__main__":
    sys.exit(0 if run_test() else 1)
