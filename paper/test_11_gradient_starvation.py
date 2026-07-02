# =============================================================================
# MoEWatch v0.2.0 — Research Paper Test Suite
# Test 11: Gradient Starvation Detection (Tier 1 Signal)
# =============================================================================
#
# Verifies the GradientStarvationAnalyzer correctly measures per-expert
# gradient health across four scenarios:
#
#   1. Healthy experts  → starvation_score ≈ 0.0, starvation_detected=False
#   2. Starved expert   → starvation_score ≈ 1.0 after cold_steps_limit
#   3. Mixed layer      → some experts healthy, some starved
#   4. Recovery         → starvation_score drops when grad norms recover
#   5. Report fields    → all required fields populated correctly
#   6. starvation_score formula: max(0, 1 - mean_norm / cold_threshold)
#
# Gradient starvation is MoEWatch's Tier 1 signal (weight 0.6).
# It is the primary collapse precursor — an expert receiving zero
# gradient updates will never recover without intervention.
#
# =============================================================================

from moewatch.analyzer.gradient_starvation import GradientStarvationAnalyzer
from moewatch.collector.stat_collector import StatCollector
from moewatch.hooks.gradient_hook import GradientEvent
from moewatch.config import WatchConfig, OutputMode

LAYER       = "layers.0.gate"
NUM_EXPERTS = 8


def make_config(cold_steps_limit: int = 10) -> WatchConfig:
    return WatchConfig(
        output=OutputMode.SILENT,
        cold_threshold=1.0,       # experts below this norm → cold
        dead_threshold=0.01,
        cold_steps_limit=cold_steps_limit,
        stats_window=100,
    )


def make_sc(config: WatchConfig) -> StatCollector:
    sc = StatCollector(config)
    sc.register_layer(LAYER, NUM_EXPERTS)
    return sc


def inject_grad(
    sc: StatCollector,
    expert_norms: dict,     # {expert_id: norm_value}
    n_steps: int,
    start_step: int = 1,
) -> None:
    """Write GradientEvents for each expert with given norm for n_steps."""
    for step in range(start_step, start_step + n_steps):
        for expert_id, norm in expert_norms.items():
            sc.write_gradient_event(GradientEvent(
                timestamp=float(step),
                global_step=step,
                layer_name=LAYER,
                expert_id=expert_id,
                gradient_norm=norm,
                gradient_magnitude=norm,
            ))


def inject_grad_with_analyze(
    sc: StatCollector,
    analyzer,
    expert_norms: dict,
    n_steps: int,
    start_step: int = 1,
) -> list:
    """Write GradientEvents AND call analyze() each step so the consecutive
    cold-step counter increments on every step (as it would in live training)."""
    last_reports = []
    for step in range(start_step, start_step + n_steps):
        for expert_id, norm in expert_norms.items():
            sc.write_gradient_event(GradientEvent(
                timestamp=float(step),
                global_step=step,
                layer_name=LAYER,
                expert_id=expert_id,
                gradient_norm=norm,
                gradient_magnitude=norm,
            ))
        last_reports = analyzer.analyze(sc)
    return last_reports


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def run():
    print("=" * 60)
    print("  MoEWatch Test 11 — Gradient Starvation (Tier 1 Signal)")
    print("=" * 60)
    print(f"  cold_threshold={1.0}  cold_steps_limit=10")

    config   = make_config(cold_steps_limit=10)
    analyzer = GradientStarvationAnalyzer(config)

    # ==================================================================
    # Scenario 1 — All experts healthy (high gradient norms)
    # ==================================================================
    print("\n  [1] All experts healthy (norm=5.0 >> cold_threshold=1.0)")

    sc1 = make_sc(config)
    inject_grad(sc1, {e: 5.0 for e in range(NUM_EXPERTS)}, n_steps=20)

    reports1 = analyzer.analyze(sc1)
    layer_reports1 = reports1.get(LAYER, [])

    print(f"    num reports        : {len(layer_reports1)}  (expected {NUM_EXPERTS})")
    for r in layer_reports1[:3]:
        print(f"    expert {r.expert_id}: norm_mean={r.gradient_norm_mean:.3f}  "
              f"score={r.starvation_score:.4f}  detected={r.starvation_detected}")
    print(f"    ...")

    assert len(layer_reports1) == NUM_EXPERTS, \
        f"Expected {NUM_EXPERTS} reports, got {len(layer_reports1)}"
    for r in layer_reports1:
        assert r.starvation_score < 0.1, \
            f"Expert {r.expert_id}: healthy norm should give score≈0, got {r.starvation_score:.4f}"
        assert r.starvation_detected is False, \
            f"Expert {r.expert_id}: should not be detected as starved"
    print(f"    ✓ All {NUM_EXPERTS} experts healthy: score≈0, detected=False")

    # ==================================================================
    # Scenario 2 — Expert 0 fully starved (norm=0)
    # ==================================================================
    print("\n  [2] Expert 0 starved (norm=0) for cold_steps_limit steps")

    analyzer2 = GradientStarvationAnalyzer(config)
    sc2 = make_sc(config)
    norms = {e: 5.0 for e in range(NUM_EXPERTS)}
    norms[0] = 0.0
    # Call analyze() each step so consecutive_cold_steps counter accumulates
    reports2 = inject_grad_with_analyze(sc2, analyzer2, norms, n_steps=15)
    layer_reps2   = reports2.get(LAYER, [])
    r_expert0     = next(r for r in layer_reps2 if r.expert_id == 0)
    r_expert1     = next(r for r in layer_reps2 if r.expert_id == 1)

    print(f"    expert 0: norm_mean={r_expert0.gradient_norm_mean:.4f}  "
          f"score={r_expert0.starvation_score:.4f}  "
          f"detected={r_expert0.starvation_detected}")
    print(f"    expert 1: norm_mean={r_expert1.gradient_norm_mean:.4f}  "
          f"score={r_expert1.starvation_score:.4f}  "
          f"detected={r_expert1.starvation_detected}")

    assert r_expert0.starvation_score > 0.8, \
        f"Starved expert should have score≈1.0, got {r_expert0.starvation_score:.4f}"
    assert r_expert0.starvation_detected is True, \
        "Starved expert should be detected=True after cold_steps_limit"
    assert r_expert1.starvation_detected is False, \
        "Healthy expert 1 should not be detected"
    print(f"    ✓ Expert 0 starved: score={r_expert0.starvation_score:.4f}, detected=True")
    print(f"    ✓ Expert 1 healthy: score={r_expert1.starvation_score:.4f}, detected=False")

    # ==================================================================
    # Scenario 3 — starvation_score formula verification
    # ==================================================================
    print("\n  [3] starvation_score formula: max(0, 1 - norm_mean / cold_threshold)")

    test_norms = [0.0, 0.5, 1.0, 2.0]
    cold_th    = config.cold_threshold   # 1.0
    print(f"    {'norm':>6}  {'expected_score':>14}  {'actual_score':>12}  match")
    print(f"    {'-'*6}  {'-'*14}  {'-'*12}  -----")

    for norm_val in test_norms:
        a3 = GradientStarvationAnalyzer(config)
        s3 = make_sc(config)
        inject_grad(s3, {0: norm_val}, n_steps=5)
        reps3 = a3.analyze(s3)
        r3 = next((r for r in reps3.get(LAYER, []) if r.expert_id == 0), None)
        if r3 is None:
            print(f"    {norm_val:>6.2f}  {'N/A':>14}  {'N/A':>12}")
            continue
        expected = max(0.0, 1.0 - norm_val / cold_th)
        match = "✓" if abs(r3.starvation_score - expected) < 0.05 else "✗"
        print(f"    {norm_val:>6.2f}  {expected:>14.4f}  {r3.starvation_score:>12.4f}  {match}")
        assert abs(r3.starvation_score - expected) < 0.05, \
            f"norm={norm_val}: expected score={expected:.4f}, got {r3.starvation_score:.4f}"
    print(f"    ✓ Formula verified across four norm values")

    # ==================================================================
    # Scenario 4 — Recovery: starvation drops when norms recover
    # ==================================================================
    print("\n  [4] Recovery: norm recovers from 0.0 → 5.0")

    analyzer4 = GradientStarvationAnalyzer(config)
    sc4 = make_sc(config)

    # Phase 1: starvation (norm=0 for 15 steps, analyze each step)
    reps4_collapsed = inject_grad_with_analyze(sc4, analyzer4, {0: 0.0}, n_steps=15, start_step=1)
    r4_col = next(r for r in reps4_collapsed.get(LAYER, []) if r.expert_id == 0)

    # Phase 2: recovery (norm=5.0 for 15 steps, analyze each step)
    reps4_recovered = inject_grad_with_analyze(sc4, analyzer4, {0: 5.0}, n_steps=15, start_step=16)
    r4_rec = next(r for r in reps4_recovered.get(LAYER, []) if r.expert_id == 0)

    print(f"    After collapse : score={r4_col.starvation_score:.4f}  "
          f"detected={r4_col.starvation_detected}")
    print(f"    After recovery : score={r4_rec.starvation_score:.4f}  "
          f"detected={r4_rec.starvation_detected}")

    assert r4_col.starvation_score > r4_rec.starvation_score, \
        "Recovery should reduce starvation_score"
    print(f"    ✓ Score dropped from {r4_col.starvation_score:.4f} → "
          f"{r4_rec.starvation_score:.4f} after recovery")

    # ==================================================================
    # Scenario 5 — Report fields populated
    # ==================================================================
    print("\n  [5] Report field coverage")

    r_sample = layer_reps2[0]
    fields   = vars(r_sample)
    print(f"    Available fields: {list(fields.keys())}")

    for required in ("layer_name", "expert_id", "gradient_norm_mean",
                     "gradient_norm_std", "starvation_score",
                     "starvation_detected", "step", "n_samples"):
        assert required in fields, f"Missing field: {required}"
    print(f"    ✓ All required fields present")
    print(f"    layer_name={r_sample.layer_name!r}  expert_id={r_sample.expert_id}  "
          f"n_samples={r_sample.n_samples}")

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n  All assertions passed.")
    print(f"  Starvation scores:")
    print(f"    Healthy (norm=5.0) : {layer_reports1[0].starvation_score:.4f}")
    print(f"    Starved (norm=0.0) : {r_expert0.starvation_score:.4f}")
    print(f"    Recovered          : {r4_rec.starvation_score:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    run()
