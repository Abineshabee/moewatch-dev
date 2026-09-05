# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# examples/rule_policy_walkthrough.py
# =============================================================================
#
# Project      : MoEWatch
# Version      : v0.2.0
# Description  : Standalone walkthrough of the RulePolicy (Phase 1).
#
#                Covers:
#                  1. Basic risk-to-action threshold mapping across all tiers
#                  2. Simulating a realistic per-layer risk trajectory
#                  3. Cascade guard — same action repeating → downgrade
#                  4. Critical-risk guard — stronger action gets more runway
#                  5. Oscillation guard — A→B→A→B alternation → downgrade
#                  6. Multi-layer independence (each layer has own history)
#                  7. Checkpoint save and load
#
#                No model or trainer required — policy is exercised directly
#                via PolicyState objects, making this safe to run on CPU
#                without GPU or a real training loop.
#
# Author       : Abinesh N (@Abineshabee)
# Repository   : https://github.com/Abineshabee/MoEWatch
# License      : Apache 2.0
#
# Run
# ---
#   python examples/rule_policy_walkthrough.py
#
# =============================================================================

from __future__ import annotations

import tempfile
import os

from moewatch.config import WatchConfig, OutputMode
from moewatch.policy.base import PolicyState
from moewatch.policy.rule_policy import (
    RulePolicy,
    _CASCADE_REPEAT_LIMIT,
    _CASCADE_CRITICAL_REPEAT_LIMIT,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

SEP  = "─" * 58
SEP2 = "═" * 58

def _config() -> WatchConfig:
    """Shared config for all examples — silent output, interventions on."""
    return WatchConfig(output=OutputMode.SILENT)


def _state(
    risk: float,
    layer_id: int = 0,
    step: int = 0,
    layer_name: str = "",
    dominant_signal: str = "entropy",
) -> PolicyState:
    return PolicyState(
        risk_score=risk,
        layer_id=layer_id,
        training_step=step,
        layer_name=layer_name or f"layer_{layer_id}",
        dominant_signal=dominant_signal,
    )


def _banner(title: str) -> None:
    print(f"\n{SEP2}")
    print(f"  {title}")
    print(SEP2)


def _row(step: int, risk: float, action: str, note: str = "") -> None:
    note_col = f"  ← {note}" if note else ""
    print(f"    step={step:>3}  risk={risk:.2f}  →  {action:<18}{note_col}")


# ── Section 1: Threshold mapping ─────────────────────────────────────────────

def demo_threshold_mapping() -> None:
    _banner("1. Risk-to-action threshold mapping")
    print(f"""
  risk < 0.30  →  noop            (healthy, do nothing)
  risk < 0.60  →  aux_loss        (soft load balance)
  risk < 0.80  →  router_noise    (force exploration)
  risk >= 0.80 →  expert_dropout  (strongest — genuine collapse)
""")
    policy = RulePolicy(_config())

    tiers = [
        (0.10, "noop"),
        (0.29, "noop"),
        (0.30, "aux_loss"),
        (0.59, "aux_loss"),
        (0.60, "router_noise"),
        (0.79, "router_noise"),
        (0.80, "expert_dropout"),
        (1.00, "expert_dropout"),
    ]
    print(f"    {'risk':>6}   {'expected':<18}  {'got':<18}  {'ok'}")
    print(f"    {SEP}")
    all_ok = True
    for risk, expected in tiers:
        action = policy.select_action(_state(risk)).action_type
        ok = action == expected
        all_ok = all_ok and ok
        marker = "✓" if ok else "✗  <-- UNEXPECTED"
        print(f"    {risk:>6.2f}   {expected:<18}  {action:<18}  {marker}")

    print(f"\n  Result: {'all thresholds correct ✓' if all_ok else 'MISMATCH ✗'}")


# ── Section 2: Realistic risk trajectory ─────────────────────────────────────

def demo_risk_trajectory() -> None:
    _banner("2. Realistic risk trajectory (collapse + recovery)")
    print("""
  Simulates a layer whose risk rises to critical, stabilises,
  then recovers. Shows how the policy responds at each stage.
""")
    policy = RulePolicy(_config())

    trajectory = [
        # (step, risk, description)
        (0,  0.05, "healthy"),
        (1,  0.08, "healthy"),
        (2,  0.32, "mild degradation"),
        (3,  0.41, "degrading"),
        (4,  0.58, "degrading"),
        (5,  0.65, "router imbalance"),
        (6,  0.72, "routing skewed"),
        (7,  0.85, "near collapse"),
        (8,  0.94, "active collapse"),
        (9,  0.91, "still critical"),
        (10, 0.78, "stabilising"),
        (11, 0.55, "recovering"),
        (12, 0.28, "recovered"),
        (13, 0.10, "healthy"),
    ]

    print(f"    {'step':>5}  {'risk':>5}  {'action':<18}  description")
    print(f"    {SEP}")
    for step, risk, desc in trajectory:
        a = policy.select_action(_state(risk, step=step)).action_type
        print(f"    {step:>5}  {risk:>5.2f}  {a:<18}  {desc}")


# ── Section 3: Cascade guard ─────────────────────────────────────────────────

def demo_cascade_guard() -> None:
    _banner("3. Cascade guard — same action repeating too often")
    print(f"""
  When the same action fires more than {_CASCADE_REPEAT_LIMIT} consecutive times
  at non-critical risk, the cascade guard kicks in and downgrades
  to the next-weaker tier to break the stall.

  Here: risk stays at 0.70 (router_noise tier) for {_CASCADE_REPEAT_LIMIT + 5} steps.
""")
    policy = RulePolicy(_config())

    print(f"    step   risk    action             note")
    print(f"    {SEP}")
    for step in range(_CASCADE_REPEAT_LIMIT + 5):
        a = policy.select_action(_state(0.70, step=step)).action_type
        note = ""
        if step == _CASCADE_REPEAT_LIMIT:
            note = f"← cascade guard fires (>{_CASCADE_REPEAT_LIMIT} repeats)"
        elif step > _CASCADE_REPEAT_LIMIT:
            note = "← keeps downgrading while candidate unchanged"
        _row(step, 0.70, a, note)

    print(f"""
  Logic: risk=0.70 always maps to router_noise, but after
  {_CASCADE_REPEAT_LIMIT} steps without improvement the guard injects aux_loss.
  This persists as long as the underlying risk stays the same.
""")


# ── Section 4: Critical-risk cascade ─────────────────────────────────────────

def demo_critical_cascade_guard() -> None:
    _banner("4. Critical-risk cascade — longer runway before suppression")
    print(f"""
  During genuine collapse (risk >= 0.80), suppressing expert_dropout
  after only {_CASCADE_REPEAT_LIMIT} steps would be counter-productive.
  The critical cascade limit is {_CASCADE_CRITICAL_REPEAT_LIMIT} steps — much higher runway.

  Here: risk stays at 0.90 for {_CASCADE_CRITICAL_REPEAT_LIMIT + 3} steps.
""")
    policy = RulePolicy(_config())

    print(f"    step   risk    action             note")
    print(f"    {SEP}")
    for step in range(_CASCADE_CRITICAL_REPEAT_LIMIT + 3):
        a = policy.select_action(_state(0.90, step=step)).action_type
        note = ""
        if step < _CASCADE_REPEAT_LIMIT:
            note = "normal limit would fire here — but risk is critical"
        elif step == _CASCADE_CRITICAL_REPEAT_LIMIT:
            note = f"← critical cascade guard fires (>{_CASCADE_CRITICAL_REPEAT_LIMIT} repeats)"
        elif step > _CASCADE_CRITICAL_REPEAT_LIMIT:
            note = "← counter reset, expert_dropout resumes"
        _row(step, 0.90, a, note)


# ── Section 5: Oscillation guard ─────────────────────────────────────────────

def demo_oscillation_guard() -> None:
    _banner("5. Oscillation guard — A→B→A→B alternation")
    print("""
  If risk flips rapidly between two tiers (the policy would
  alternate router_noise ↔ expert_dropout every step), the
  oscillation guard detects the A→B→A→B pattern and downgrades
  on the repeated return, forcing the policy to commit.
""")
    policy = RulePolicy(_config())

    sequence = [
        (0, 0.70, "router_noise tier"),
        (1, 0.90, "critical tier"),
        (2, 0.70, "back to router_noise"),
        (3, 0.90, "back to critical"),
        (4, 0.70, "oscillation guard fires here"),
    ]

    print(f"    step   risk    action             note")
    print(f"    {SEP}")
    for step, risk, desc in sequence:
        a = policy.select_action(_state(risk, step=step)).action_type
        _row(step, risk, a, desc)

    print("""
  Step 4 returns aux_loss instead of router_noise because the guard
  detected: [router_noise, expert_dropout, router_noise, expert_dropout]
  and the next candidate was router_noise again — classic oscillation.
""")


# ── Section 6: Multi-layer independence ──────────────────────────────────────

def demo_multi_layer() -> None:
    _banner("6. Multi-layer independence")
    print("""
  Each layer maintains its own action history. A cascade on
  layer 0 has zero effect on layers 1 and 2.
""")
    policy = RulePolicy(_config())

    # Saturate cascade on layer 0
    for step in range(_CASCADE_REPEAT_LIMIT + 3):
        policy.select_action(_state(0.70, layer_id=0, step=step))

    print(f"  After {_CASCADE_REPEAT_LIMIT + 3} steps at risk=0.70 on layer 0:\n")

    for lid in range(3):
        a = policy.select_action(
            _state(0.70, layer_id=lid, step=_CASCADE_REPEAT_LIMIT + 3)
        )
        note = "cascade saturated" if lid == 0 else "clean history — not affected"
        print(f"    layer {lid}  →  {a.action_type:<18}  ({note})")

    print()


# ── Section 7: Checkpoint round-trip ─────────────────────────────────────────

def demo_checkpoint() -> None:
    _banner("7. Checkpoint — save and resume across sessions")
    print("""
  RulePolicy saves its per-layer action history so the cascade
  and oscillation guards work correctly after a restart, rather
  than treating every resume as a blank slate.
""")
    # Run policy for several steps
    policy_a = RulePolicy(_config())
    for step in range(_CASCADE_REPEAT_LIMIT + 2):
        policy_a.select_action(_state(0.70, layer_id=0, step=step))

    # What action does policy_a give at this point?
    probe = _state(0.70, layer_id=0, step=_CASCADE_REPEAT_LIMIT + 2)
    action_before = policy_a.select_action(probe).action_type

    # Save, load into a fresh policy, probe again
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as fh:
        path = fh.name

    try:
        policy_a.save_checkpoint(path)
        print(f"  Checkpoint saved → {os.path.basename(path)}")

        policy_b = RulePolicy(_config())
        policy_b.load_checkpoint(path)
        action_after = policy_b.select_action(probe).action_type

        match = action_before == action_after
        print(f"  Action before save : {action_before}")
        print(f"  Action after load  : {action_after}")
        print(f"  History preserved  : {'yes ✓' if match else 'no ✗'}")
    finally:
        os.unlink(path)

    print("""
  A fresh policy (without loading the checkpoint) would start its
  cascade counter from zero and return router_noise instead of the
  downgraded action — the checkpoint is what prevents that regression.
""")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(SEP2)
    print("  MoEWatch — RulePolicy Walkthrough")
    print(f"  CASCADE_REPEAT_LIMIT          = {_CASCADE_REPEAT_LIMIT}")
    print(f"  CASCADE_CRITICAL_REPEAT_LIMIT = {_CASCADE_CRITICAL_REPEAT_LIMIT}")
    print(SEP2)

    demo_threshold_mapping()
    demo_risk_trajectory()
    demo_cascade_guard()
    demo_critical_cascade_guard()
    demo_oscillation_guard()
    demo_multi_layer()
    demo_checkpoint()

    print(SEP2)
    print("  Done.")
    print(SEP2)


if __name__ == "__main__":
    main()
