# =============================================================================
# MoEWatch v0.2.0 — Research Paper Test Suite
# Test 10: Risk Score Fusion (Tier 1 + Tier 2 + Tier 3)
# =============================================================================
#
# Verifies RiskScoreFuser correctly combines three tier signals into a
# single per-layer risk score:
#
#   T1 = gradient starvation score   (weight 0.6)
#   T2 = 1 - normalized_entropy      (weight 0.3)
#   T3 = cross-layer spread score    (weight 0.1)
#
#   risk_score = w1*T1 + w2*T2 + w3*T3  ∈ [0, 1]
#
# Tests:
#   1. Zero risk  → all signals healthy → score ≈ 0.0, level=LOW
#   2. T2 only    → entropy collapse only → correct weighted contribution
#   3. T1 only    → gradient starvation only → heavier weight reflected
#   4. All tiers  → full fusion, dominant_signal identified correctly
#   5. T3=None    → weight redistributed to T1 and T2
#   6. Risk level thresholds: LOW<0.3 / MID<0.6 / HIGH<0.8 / CRITICAL≥0.8
#   7. Custom weights → validated and applied correctly
#
# =============================================================================

from dataclasses import dataclass, field
from typing import List, Optional

from moewatch.analyzer.risk_score import RiskScoreFuser, RiskReport, RiskLevel
from moewatch.config import WatchConfig, OutputMode

LAYER = "layers.0.gate"


def make_config() -> WatchConfig:
    return WatchConfig(output=OutputMode.SILENT)


# ---------------------------------------------------------------------------
# Minimal duck-typed tier report stubs
# (fuse() is tolerant of any object with the right attributes)
# ---------------------------------------------------------------------------

@dataclass
class FakeGradReport:
    layer_name: str = LAYER
    starvation_score: float = 0.0
    step: int = 10


@dataclass
class FakeEntropyReport:
    layer_name: str = LAYER
    normalized_entropy: float = 1.0   # 1.0 = perfectly healthy
    drift_detected: bool = False
    step: int = 10


@dataclass
class FakeCrossLayerReport:
    source_layer: Optional[str] = None
    victim_layers: List[str] = field(default_factory=list)
    spread_score: float = 0.0


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def run():
    print("=" * 60)
    print("  MoEWatch Test 10 — Risk Score Fusion (T1 + T2 + T3)")
    print("=" * 60)
    print(f"  Fusion weights: T1=0.6  T2=0.3  T3=0.1")
    print(f"  Risk levels:    LOW<0.3 | MID<0.6 | HIGH<0.8 | CRITICAL≥0.8")

    config = make_config()
    fuser  = RiskScoreFuser(config)

    # ==================================================================
    # Scenario 1 — Zero risk (all signals healthy)
    # ==================================================================
    print("\n  [1] All signals healthy → risk ≈ 0.0")

    r1 = fuser.fuse(
        gradient_report    = FakeGradReport(starvation_score=0.0),
        entropy_report     = FakeEntropyReport(normalized_entropy=1.0),
        cross_layer_report = FakeCrossLayerReport(spread_score=0.0),
    )

    print(f"    risk_score         : {r1.risk_score:.4f}  (expected ≈ 0.0)")
    print(f"    risk_level         : {r1.risk_level.name}")
    print(f"    tier1_contribution : {r1.tier1_contribution:.4f}")
    print(f"    tier2_contribution : {r1.tier2_contribution:.4f}")
    print(f"    tier3_contribution : {r1.tier3_contribution:.4f}")

    assert r1.risk_score < 0.05, f"Expected ≈ 0.0, got {r1.risk_score:.4f}"
    assert r1.risk_level == RiskLevel.LOW
    print(f"    ✓ All-healthy → risk={r1.risk_score:.4f}, level=LOW")

    # ==================================================================
    # Scenario 2 — T2 only (entropy collapse, no gradient starvation)
    # ==================================================================
    print("\n  [2] T2 only — entropy fully collapsed (entropy=0.0)")

    r2 = fuser.fuse(
        gradient_report    = FakeGradReport(starvation_score=0.0),
        entropy_report     = FakeEntropyReport(normalized_entropy=0.0),
        cross_layer_report = FakeCrossLayerReport(spread_score=0.0),
    )

    # T2 raw = 1 - 0.0 = 1.0; contribution = 0.3 * 1.0 = 0.3
    expected_t2 = 0.3
    print(f"    risk_score         : {r2.risk_score:.4f}  (expected ≈ {expected_t2:.2f})")
    print(f"    tier2_contribution : {r2.tier2_contribution:.4f}  (expected ≈ {expected_t2:.2f})")
    print(f"    dominant_signal    : {r2.dominant_signal}")

    assert abs(r2.tier2_contribution - expected_t2) < 0.05, \
        f"T2 contribution should be ≈{expected_t2}, got {r2.tier2_contribution:.4f}"
    print(f"    ✓ T2-only contribution = {r2.tier2_contribution:.4f}")

    # ==================================================================
    # Scenario 3 — T1 only (gradient starvation, healthy entropy)
    # ==================================================================
    print("\n  [3] T1 only — full gradient starvation (score=1.0)")

    r3 = fuser.fuse(
        gradient_report    = FakeGradReport(starvation_score=1.0),
        entropy_report     = FakeEntropyReport(normalized_entropy=1.0),
        cross_layer_report = FakeCrossLayerReport(spread_score=0.0),
    )

    # T1 contribution = 0.6 * 1.0 = 0.6
    expected_t1 = 0.6
    print(f"    risk_score         : {r3.risk_score:.4f}  (expected ≈ {expected_t1:.2f})")
    print(f"    tier1_contribution : {r3.tier1_contribution:.4f}  (expected ≈ {expected_t1:.2f})")
    print(f"    dominant_signal    : {r3.dominant_signal}")

    assert abs(r3.tier1_contribution - expected_t1) < 0.05, \
        f"T1 contribution should be ≈{expected_t1}, got {r3.tier1_contribution:.4f}"
    assert r3.tier1_contribution > r2.tier2_contribution, \
        "T1 weight (0.6) must produce larger contribution than T2 (0.3)"
    print(f"    ✓ T1-only contribution = {r3.tier1_contribution:.4f}  "
          f"(> T2-only {r2.tier2_contribution:.4f} ✓)")

    # ==================================================================
    # Scenario 4 — All tiers active, dominant_signal check
    # ==================================================================
    print("\n  [4] All tiers active — dominant signal identified")

    r4 = fuser.fuse(
        gradient_report    = FakeGradReport(starvation_score=0.8),
        entropy_report     = FakeEntropyReport(normalized_entropy=0.3),
        cross_layer_report = FakeCrossLayerReport(spread_score=0.5),
    )

    print(f"    risk_score         : {r4.risk_score:.4f}")
    print(f"    risk_level         : {r4.risk_level.name}")
    print(f"    tier1_contribution : {r4.tier1_contribution:.4f}  (T1=0.6×0.8={0.6*0.8:.3f})")
    print(f"    tier2_contribution : {r4.tier2_contribution:.4f}  (T2=0.3×0.7={0.3*0.7:.3f})")
    print(f"    tier3_contribution : {r4.tier3_contribution:.4f}  (T3=0.1×0.5={0.1*0.5:.3f})")
    print(f"    dominant_signal    : {r4.dominant_signal!r}  (expected 'gradient')")

    assert r4.dominant_signal == "gradient", \
        f"T1 has highest contribution, dominant should be 'gradient', got {r4.dominant_signal!r}"
    assert r4.risk_score > 0.5, \
        f"All-high signals should give risk > 0.5, got {r4.risk_score:.4f}"
    print(f"    ✓ dominant_signal='gradient' (highest weighted contributor)")

    # ==================================================================
    # Scenario 5 — T3=None: weight redistributed to T1 and T2
    # ==================================================================
    print("\n  [5] T3=None — Tier 3 weight redistributed")

    r5_with = fuser.fuse(
        gradient_report    = FakeGradReport(starvation_score=0.5),
        entropy_report     = FakeEntropyReport(normalized_entropy=0.5),
        cross_layer_report = FakeCrossLayerReport(spread_score=0.0),
    )
    r5_none = fuser.fuse(
        gradient_report    = FakeGradReport(starvation_score=0.5),
        entropy_report     = FakeEntropyReport(normalized_entropy=0.5),
        cross_layer_report = None,
    )

    print(f"    risk (T3 present, spread=0) : {r5_with.risk_score:.4f}")
    print(f"    risk (T3=None)              : {r5_none.risk_score:.4f}")
    print(f"    t1_contrib (T3=None)        : {r5_none.tier1_contribution:.4f}  "
          f"(expected > {r5_with.tier1_contribution:.4f})")

    assert r5_none.tier1_contribution >= r5_with.tier1_contribution, \
        "T1 contribution should increase when T3 weight is redistributed"
    print(f"    ✓ T3=None redistributes weight to T1/T2")

    # ==================================================================
    # Scenario 6 — Risk level thresholds
    # ==================================================================
    print("\n  [6] Risk level thresholds")

    level_cases = [
        (0.0,  0.0,  RiskLevel.LOW,      "LOW"),
        (0.35, 0.0,  RiskLevel.LOW,      "LOW"),      # 0.6×0.35 = 0.21 < 0.3
        (0.5,  0.0,  RiskLevel.MID,      "MID"),      # 0.6×0.5 = 0.30 → MID
        (0.8,  0.3,  RiskLevel.HIGH,     "HIGH"),
        (1.0,  1.0,  RiskLevel.CRITICAL, "CRITICAL"),
    ]

    print(f"    {'T1':>5}  {'T2_raw':>6}  {'score':>7}  {'level':<10}")
    print(f"    {'-'*5}  {'-'*6}  {'-'*7}  {'-'*10}")
    for t1, t2_entropy, expected_level, label in level_cases:
        r = fuser.fuse(
            gradient_report    = FakeGradReport(starvation_score=t1),
            entropy_report     = FakeEntropyReport(normalized_entropy=1.0 - t2_entropy),
            cross_layer_report = None,
        )
        match = "✓" if r.risk_level == expected_level else "✗"
        print(f"    {t1:>5.2f}  {t2_entropy:>6.2f}  {r.risk_score:>7.4f}  "
              f"{r.risk_level.name:<10} {match}")
        assert r.risk_level == expected_level, \
            f"T1={t1} T2={t2_entropy}: expected {label}, got {r.risk_level.name}"
    print(f"    ✓ All risk level thresholds correct")

    # ==================================================================
    # Scenario 7 — Custom weights
    # ==================================================================
    print("\n  [7] Custom weights (T1=0.5, T2=0.4, T3=0.1)")

    fuser_custom = RiskScoreFuser(
        config,
        weights={"tier1": 0.5, "tier2": 0.4, "tier3": 0.1}
    )
    r7 = fuser_custom.fuse(
        gradient_report    = FakeGradReport(starvation_score=1.0),
        entropy_report     = FakeEntropyReport(normalized_entropy=0.0),
        cross_layer_report = FakeCrossLayerReport(spread_score=0.0),
    )

    print(f"    tier1_contribution : {r7.tier1_contribution:.4f}  (expected 0.5×1.0=0.50)")
    print(f"    tier2_contribution : {r7.tier2_contribution:.4f}  (expected 0.4×1.0=0.40)")
    print(f"    risk_score         : {r7.risk_score:.4f}  (expected ≈ 0.90)")

    assert abs(r7.tier1_contribution - 0.5) < 0.05, \
        f"Custom T1 weight 0.5 → contribution≈0.5, got {r7.tier1_contribution:.4f}"
    assert abs(r7.tier2_contribution - 0.4) < 0.05, \
        f"Custom T2 weight 0.4 → contribution≈0.4, got {r7.tier2_contribution:.4f}"
    print(f"    ✓ Custom weights applied correctly")

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n  All assertions passed.")
    print(f"  Tier contributions (max signal):")
    print(f"    T1 alone (w=0.6): score={r3.risk_score:.4f}  level={r3.risk_level.name}")
    print(f"    T2 alone (w=0.3): score={r2.risk_score:.4f}  level={r2.risk_level.name}")
    print(f"    All tiers:        score={r4.risk_score:.4f}  level={r4.risk_level.name}")
    print("=" * 60)


if __name__ == "__main__":
    run()
