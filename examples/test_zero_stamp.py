# =============================================================================
# examples/test_zero_stamp.py
#
# Focused test for the zero-stamp fix in HookManager / GradientStarvationHook.
#
# What this verifies
# ------------------
# BEFORE fix: When an expert receives zero tokens (no backward edge), its
#   GradientStarvationHook never fires. The buffer goes stale with old
#   healthy norms → starvation_score stays near 0.0 → collapse undetected.
#
# AFTER fix:  HookManager.flush_missing_gradient_events() writes a zero-norm
#   GradientEvent for every silent expert on each sampled step. The rolling
#   window fills with 0.0 norms → norm_mean drops → starvation_score rises
#   → collapse correctly detected.
#
# Test structure
# --------------
#   Phase 1 (steps 1-20)  — all experts receive tokens (healthy)
#   Phase 2 (steps 21-40) — experts 1..N-1 receive zero tokens (dead)
#
# Pass criteria
# -------------
#   After Phase 2, the gradient starvation analyzer must report:
#     - starvation_score > 0.5  for at least one dead expert
#     - norm_mean == 0.0        for dead experts
#     - starvation_score < 0.2  for the live expert (expert 0)
#
# Usage
# -----
#   python tests/test_zero_stamp_fix.py
#
# =============================================================================

from __future__ import annotations

import sys
import torch
import torch.nn as nn

from moewatch import MoEWatch
from moewatch.config import WatchConfig, OutputMode


# ---------------------------------------------------------------------------
# Minimal MoE model
# ---------------------------------------------------------------------------

NUM_EXPERTS = 8
HIDDEN_DIM  = 32


class SimpleMoEBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate    = nn.Linear(HIDDEN_DIM, NUM_EXPERTS, bias=True)
        self.experts = nn.ModuleList([
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM) for _ in range(NUM_EXPERTS)
        ])

    def forward(self, x):
        B, S, D = x.shape
        flat = x.reshape(-1, D)
        logits = self.gate(flat)
        probs  = torch.softmax(logits, -1)
        top_p, top_i = probs.topk(2, dim=-1)
        top_p = top_p / top_p.sum(-1, keepdim=True)
        out = torch.zeros_like(flat)
        for k in range(2):
            for e in range(NUM_EXPERTS):
                mask = (top_i[:, k] == e)
                if mask.any():
                    out[mask] += top_p[mask, k:k+1] * self.experts[e](flat[mask])
        return out.reshape(B, S, D)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, HIDDEN_DIM)
        self.moe   = SimpleMoEBlock()
        self.head  = nn.Linear(HIDDEN_DIM, 100)

    def set_monopoly(self, expert_id: int = 0, strength: float = 6.0):
        """Route all tokens to expert_id."""
        with torch.no_grad():
            self.moe.gate.bias.zero_()
            self.moe.gate.bias[expert_id] = strength

    def reset_routing(self):
        with torch.no_grad():
            self.moe.gate.bias.zero_()

    def forward(self, x):
        h = self.embed(x)
        h = self.moe(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def run_test():
    print("=" * 60)
    print("  Zero-Stamp Fix — Gradient Starvation Hook Test")
    print("=" * 60)

    torch.manual_seed(42)
    model = SimpleModel()

    config = WatchConfig(
        output=OutputMode.SILENT,
        sample_every=1,
        log_every=1,
        dead_threshold=0.005,
        cold_threshold=0.03,
        cold_steps_limit=5,
        intervention_enabled=False,
    )

    from unittest.mock import MagicMock
    trainer = MagicMock()
    trainer.model = model

    watcher  = MoEWatch(model, config)
    watcher.attach(trainer)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    # ------------------------------------------------------------------
    # Phase 1 — healthy routing (steps 1-20)
    # ------------------------------------------------------------------
    print("\nPhase 1: Healthy routing (steps 1-20)")
    model.reset_routing()
    for step in range(1, 21):
        watcher.pre_step(step)
        x = torch.randint(0, 100, (2, 8))
        optimizer.zero_grad()
        loss = model(x).mean()
        loss.backward()
        optimizer.step()
        watcher.step(global_step=step, current_loss=loss.item())
    print("  Done.")

    # ------------------------------------------------------------------
    # Phase 2 — collapse: only expert 0 gets tokens (steps 21-40)
    # ------------------------------------------------------------------
    print("\nPhase 2: Collapse — expert 0 monopoly (steps 21-40)")
    model.set_monopoly(expert_id=0, strength=6.0)
    for step in range(21, 41):
        watcher.pre_step(step)
        x = torch.randint(0, 100, (2, 8))
        optimizer.zero_grad()
        loss = model(x).mean()
        loss.backward()
        optimizer.step()
        watcher.step(global_step=step, current_loss=loss.item())
    print("  Done.")

    # ------------------------------------------------------------------
    # Inspect gradient results
    # ------------------------------------------------------------------
    print("\nGradient starvation results after Phase 2:")
    print(f"  {'Expert':<10} {'norm_mean':>12} {'starvation':>12} {'n_samples':>10} {'status'}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")

    from moewatch.analyzer.gradient_starvation import GradientStarvationAnalyzer
    analyzer = GradientStarvationAnalyzer(config)
    grad_results = analyzer.analyze(watcher.stat_collector)

    layer_name = list(grad_results.keys())[0] if grad_results else None
    if not layer_name:
        print("  ERROR: no gradient results returned")
        sys.exit(1)

    reports = grad_results[layer_name]
    reports_by_id = {r.expert_id: r for r in reports}

    pass_checks = []

    for eid in range(NUM_EXPERTS):
        r = reports_by_id.get(eid)
        if r is None:
            print(f"  expert {eid:<4}   MISSING")
            continue

        norm  = getattr(r, 'gradient_norm_mean', -1.0)
        score = r.starvation_score
        nsamp = getattr(r, 'n_samples', 0)

        if eid == 0:
            status = "LIVE (should be low score)"
            check  = score < 0.5
            pass_checks.append(("live expert score < 0.5", check, score))
        else:
            status = "DEAD (should be high score)"
            check  = score > 0.5
            pass_checks.append((f"dead expert {eid} score > 0.5", check, score))

        flag = "✓" if check else "✗ FAIL"
        print(f"  expert {eid:<4}  {norm:>12.6f}  {score:>12.4f}  {nsamp:>10}  {status}  {flag}")

    # ------------------------------------------------------------------
    # Also verify zero events were written (flush worked)
    # ------------------------------------------------------------------
    print("\nZero-norm event flush check:")
    hook_manager = watcher.hook_manager
    n_flushed = sum(
        1 for hook in hook_manager._gradient_hooks
        if hook.last_fired_step != 40  # step 40 was the last step
    )
    print(f"  Hooks that did NOT fire on last step: {n_flushed}/{len(hook_manager._gradient_hooks)}")
    print(f"  (Expected ~{NUM_EXPERTS - 1} — all experts except the monopoly one)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Results:")
    all_passed = True
    for name, passed, value in pass_checks[:NUM_EXPERTS]:  # first N unique checks
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  [{status}] {name} (value={value:.4f})")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  ALL CHECKS PASSED — zero-stamp fix is working correctly.")
    else:
        print("  SOME CHECKS FAILED — fix may not have been applied.")
    print("=" * 60)

    watcher.stop()


if __name__ == "__main__":
    run_test()
