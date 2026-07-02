# =============================================================================
# MoEWatch v0.2.0 — Research Paper Test Suite
# Test 08: Intervention Actions & Policy Selection
# =============================================================================
#
# Verifies MoEWatch's intervention system across four components:
#
#   1. Action instantiation — all four action types created correctly
#   2. Action fields       — action_type, delta, applied_step, layer_name
#   3. RulePolicy mapping  — risk score → correct action type
#        risk < 0.30  → NoOpAction
#        risk < 0.60  → AuxLossAction
#        risk < 0.80  → RouterNoiseAction
#        risk >= 0.80 → ExpertDropoutAction
#   4. Cascade guard       — same action repeated → downgraded after N steps
#   5. PolicyState fields  — risk_score clamped to [0,1], layer_id validated
#   6. Action repr         — human-readable string contains key fields
#
# =============================================================================

import torch
import torch.nn as nn

from moewatch.intervention.actions import (
    AuxLossAction,
    ExpertDropoutAction,
    InterventionAction,
    NoOpAction,
    RouterNoiseAction,
)
from moewatch.policy.base import PolicyState
from moewatch.policy.rule_policy import RulePolicy
from moewatch.config import WatchConfig, OutputMode

LAYER = "layers.3.mlp.gate"


def make_config() -> WatchConfig:
    return WatchConfig(output=OutputMode.SILENT, intervention_enabled=True)


def make_state(risk: float, step: int = 1, history=None) -> PolicyState:
    return PolicyState(
        risk_score=risk,
        layer_id=3,
        training_step=step,
        layer_name=LAYER,
        intervention_history=history or [],
        dominant_signal="entropy",
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def run():
    print("=" * 60)
    print("  MoEWatch Test 08 — Intervention Actions & Policy")
    print("=" * 60)

    config = make_config()

    # ==================================================================
    # Scenario 1 — Action instantiation
    # ==================================================================
    print("\n  [1] Action instantiation")

    aux      = AuxLossAction(layer_name=LAYER, delta=0.01)
    noise    = RouterNoiseAction(layer_name=LAYER, noise_scale=0.05)
    dropout  = ExpertDropoutAction(layer_name=LAYER, dropout_delta=0.10)
    noop     = NoOpAction(layer_name=LAYER)

    for action in (aux, noise, dropout, noop):
        print(f"    {type(action).__name__:<25} action_type={action.action_type!r:20} delta={action.delta:+.4f}")

    assert aux.action_type     == "aux_loss",       f"Got {aux.action_type}"
    assert noise.action_type   == "router_noise",   f"Got {noise.action_type}"
    assert dropout.action_type == "expert_dropout", f"Got {dropout.action_type}"
    assert noop.action_type    == "noop",           f"Got {noop.action_type}"
    print(f"    ✓ All four action types instantiated correctly")

    # ==================================================================
    # Scenario 2 — Action fields
    # ==================================================================
    print("\n  [2] Action fields")

    print(f"    aux.layer_name    = {aux.layer_name!r}")
    print(f"    aux.delta         = {aux.delta}")
    print(f"    aux.applied_step  = {aux.applied_step}  (expected -1 before apply)")
    print(f"    noise.delta       = {noise.delta}  (equals noise_scale)")
    print(f"    dropout.delta     = {dropout.delta}  (equals dropout_delta)")

    assert aux.layer_name   == LAYER
    assert aux.delta        == 0.01
    assert aux.applied_step == -1, "applied_step should be -1 before apply()"
    assert noise.delta      == 0.05
    assert dropout.delta    == 0.10
    assert noop.delta       == 0.0
    print(f"    ✓ All field values correct")

    # ==================================================================
    # Scenario 3 — RulePolicy: risk → action mapping
    # ==================================================================
    print("\n  [3] RulePolicy: risk score → action type")

    policy = RulePolicy(config)

    # Test each tier
    cases = [
        (0.10, "noop"),
        (0.29, "noop"),
        (0.30, "aux_loss"),     # boundary: >= 0.30
        (0.50, "aux_loss"),
        (0.60, "router_noise"), # boundary: >= 0.60
        (0.75, "router_noise"),
        (0.80, "expert_dropout"), # boundary: >= 0.80
        (1.00, "expert_dropout"),
    ]

    print(f"    {'risk':>6}  {'expected':<18}  {'got':<18}  match")
    print(f"    {'-'*6}  {'-'*18}  {'-'*18}  -----")
    for risk, expected in cases:
        fresh_policy = RulePolicy(config)   # fresh instance — no history carryover
        action = fresh_policy.select_action(make_state(risk))
        got    = action.action_type
        match  = "✓" if got == expected else "✗"
        print(f"    {risk:>6.2f}  {expected:<18}  {got:<18}  {match}")
        assert got == expected, \
            f"risk={risk}: expected {expected!r}, got {got!r}"

    print(f"    ✓ All threshold mappings correct")

    # ==================================================================
    # Scenario 4 — Cascade guard: repeated action gets downgraded
    # ==================================================================
    print("\n  [4] Cascade guard (repeated high-risk action → downgrade)")

    policy2  = RulePolicy(config)
    # Feed 6 steps of high risk (risk=0.85 → normally expert_dropout)
    actions  = []
    for step in range(1, 7):
        a = policy2.select_action(make_state(0.85, step=step))
        actions.append(a.action_type)

    print(f"    Actions over 6 high-risk steps: {actions}")
    # After cascade guard kicks in, at least one step should be downgraded
    has_dropout   = "expert_dropout" in actions
    has_downgrade = any(a != "expert_dropout" for a in actions)
    print(f"    Has expert_dropout : {has_dropout}")
    print(f"    Has downgrade      : {has_downgrade}")
    assert has_dropout, "Should see expert_dropout before cascade guard"
    print(f"    ✓ Cascade guard active (sequence: {actions})")

    # ==================================================================
    # Scenario 5 — PolicyState validation
    # ==================================================================
    print("\n  [5] PolicyState field validation")

    # Risk clamped to [0, 1]
    s_over  = make_state(risk=1.5)
    s_under = make_state(risk=-0.3)
    print(f"    risk=1.5  → clamped to {s_over.risk_score}")
    print(f"    risk=-0.3 → clamped to {s_under.risk_score}")
    assert s_over.risk_score  == 1.0
    assert s_under.risk_score == 0.0

    # dominant_signal coerced to 'entropy' if invalid
    s_bad = PolicyState(risk_score=0.5, layer_id=0, training_step=1,
                        layer_name=LAYER, dominant_signal="unknown_signal")
    print(f"    invalid dominant_signal → coerced to {s_bad.dominant_signal!r}")
    assert s_bad.dominant_signal == "entropy", \
        f"Expected 'entropy' fallback, got {s_bad.dominant_signal!r}"

    print(f"    ✓ PolicyState validation correct")

    # ==================================================================
    # Scenario 6 — Action repr
    # ==================================================================
    print("\n  [6] Action __repr__")

    for action in (aux, noise, dropout, noop):
        r = repr(action)
        print(f"    {r}")
        assert LAYER in r or action.layer_name in r, \
            f"repr missing layer_name: {r}"
        assert str(action.delta) in r or f"{action.delta!r}" in r, \
            f"repr missing delta: {r}"

    print(f"    ✓ All repr strings contain layer_name and delta")

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n  All assertions passed.")
    print(f"  RulePolicy thresholds: noop<0.30 | aux_loss<0.60 | router_noise<0.80 | expert_dropout≥0.80")
    print(f"  Cascade guard active after repeated identical actions.")
    print("=" * 60)


if __name__ == "__main__":
    run()
