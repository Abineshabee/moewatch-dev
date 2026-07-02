# =============================================================================
# MoEWatch v0.2.0 — Research Paper Test Suite
# Test 09: CUSUM Drift Detection
# =============================================================================
#
# Verifies the CUSUMDetector and detect_change() function across six scenarios:
#
#   1. Stable signal   → no detection (cusum stays near zero)
#   2. Step-up shift   → positive CUSUM detects upward drift
#   3. Step-down shift → negative CUSUM detects downward drift
#   4. Warmup suppression → no false alert during warmup window
#   5. reset() restarts detection after a fire
#   6. is_near_threshold() early warning before full detection
#   7. detect_change() functional API on a batch series
#
# CUSUM is MoEWatch's Tier 2 drift detector — it powers entropy trend
# classification (DECLINING / STABLE / IMPROVING) in EntropyAnalyzer.
#
# =============================================================================

import math
from moewatch.analyzer.cusum import CUSUMDetector, detect_change

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def run():
    print("=" * 60)
    print("  MoEWatch Test 09 — CUSUM Drift Detection")
    print("=" * 60)

    # ==================================================================
    # Scenario 1 — Stable signal: no detection
    # ==================================================================
    print("\n  [1] Stable signal — no drift expected")

    det = CUSUMDetector(threshold=5.0, drift=0.5)
    detections = []
    for t in range(50):
        fired = det.update(0.0)   # constant zero — no drift
        if fired:
            detections.append(t)

    print(f"    n_updates     : {det.n_updates}")
    print(f"    cusum_pos     : {det.cusum_pos:.4f}  (expected 0.0)")
    print(f"    cusum_neg     : {det.cusum_neg:.4f}")
    print(f"    detections    : {detections}  (expected [])")

    assert det.n_updates == 50
    assert det.cusum_pos == 0.0, \
        f"Stable signal should keep cusum_pos=0, got {det.cusum_pos}"
    assert len(detections) == 0, \
        f"Stable signal should produce no detections, got {detections}"
    print(f"    ✓ No false detections on stable signal")

    # ==================================================================
    # Scenario 2 — Step-up shift: positive CUSUM fires
    # ==================================================================
    print("\n  [2] Step-up shift — positive drift detected")

    det2 = CUSUMDetector(threshold=5.0, drift=0.5)
    fire_step = None
    for t in range(100):
        value = 0.0 if t < 20 else 2.0   # step up at t=20
        if det2.update(value) and fire_step is None:
            fire_step = t

    print(f"    Detection step: {fire_step}  (expected > 20)")
    print(f"    cusum_pos     : {det2.cusum_pos:.4f}")
    print(f"    last_detection: {det2.last_detection_step}")

    assert fire_step is not None, "Step-up shift should trigger detection"
    assert fire_step > 20, f"Detection must occur after shift at t=20, got t={fire_step}"
    assert det2.cusum_pos > det2.threshold, \
        f"cusum_pos should exceed threshold after detection"
    print(f"    ✓ Step-up detected at t={fire_step} (delay={fire_step-20} steps after shift)")

    # ==================================================================
    # Scenario 3 — Step-down shift: negative CUSUM fires
    # ==================================================================
    print("\n  [3] Step-down shift — negative drift detected")

    det3 = CUSUMDetector(threshold=5.0, drift=0.5)
    fire_step3 = None
    for t in range(100):
        value = 0.0 if t < 20 else -2.0   # step down at t=20
        if det3.update(value) and fire_step3 is None:
            fire_step3 = t

    print(f"    Detection step: {fire_step3}  (expected > 20)")
    print(f"    cusum_neg     : {det3.cusum_neg:.4f}")

    assert fire_step3 is not None, "Step-down shift should trigger detection"
    assert fire_step3 > 20, \
        f"Detection must occur after shift at t=20, got t={fire_step3}"
    print(f"    ✓ Step-down detected at t={fire_step3}")

    # ==================================================================
    # Scenario 4 — Warmup suppression
    # ==================================================================
    print("\n  [4] Warmup suppression — no detection during warmup")

    det4 = CUSUMDetector(threshold=5.0, drift=0.5, warmup_steps=15)
    warmup_detections = []
    post_fire = None
    for t in range(60):
        # Feed large values from t=0 (would normally fire immediately)
        fired = det4.update(3.0)
        if fired:
            if t < 15:
                warmup_detections.append(t)
            elif post_fire is None:
                post_fire = t

    print(f"    Warmup detections (t<15) : {warmup_detections}  (expected [])")
    print(f"    Post-warmup fire step    : {post_fire}  (expected > 15)")

    assert len(warmup_detections) == 0, \
        f"No detection should fire during warmup, got {warmup_detections}"
    assert post_fire is not None, \
        "Detection should fire after warmup with large signal"
    assert post_fire >= 15, \
        f"Post-warmup detection must be at t>=15, got t={post_fire}"
    print(f"    ✓ Warmup suppressed {len(warmup_detections)} false alarms; "
          f"post-warmup fire at t={post_fire}")

    # ==================================================================
    # Scenario 5 — reset() restarts detection
    # ==================================================================
    print("\n  [5] reset() — restart after detection")

    det5 = CUSUMDetector(threshold=5.0, drift=0.5)
    # Build up cusum to detection
    first_fire = None
    for t in range(30):
        if det5.update(2.0) and first_fire is None:
            first_fire = t

    print(f"    First detection at t={first_fire}, cusum_pos={det5.cusum_pos:.4f}")
    assert first_fire is not None

    # Reset and verify cusum goes to zero
    det5.reset()
    print(f"    After reset(): cusum_pos={det5.cusum_pos:.4f}  cusum_neg={det5.cusum_neg:.4f}")
    assert det5.cusum_pos == 0.0, "reset() should zero cusum_pos"
    assert det5.cusum_neg == 0.0, "reset() should zero cusum_neg"

    # Can detect again after reset
    second_fire = None
    for t in range(30):
        if det5.update(2.0) and second_fire is None:
            second_fire = t
    print(f"    Second detection at t={second_fire} after reset")
    assert second_fire is not None, "Should detect again after reset()"
    print(f"    ✓ reset() clears state; re-detection works")

    # ==================================================================
    # Scenario 6 — is_near_threshold() early warning
    # ==================================================================
    print("\n  [6] is_near_threshold() — early warning at 80% of threshold")

    det6 = CUSUMDetector(threshold=10.0, drift=0.5)
    near_step  = None
    fire_step6 = None
    for t in range(100):
        fired = det6.update(1.5)
        if near_step is None and det6.is_near_threshold(margin=0.8):
            near_step = t
        if fired and fire_step6 is None:
            fire_step6 = t

    print(f"    Near-threshold warning : t={near_step}  (cusum ≥ 80% of 10.0)")
    print(f"    Full detection         : t={fire_step6}")

    assert near_step is not None, "Near-threshold warning should fire"
    if fire_step6 is not None:
        assert near_step <= fire_step6, \
            "Early warning must precede full detection"
        print(f"    ✓ Early warning fired {fire_step6 - near_step} steps before full detection")
    else:
        print(f"    ✓ Early warning fired (full threshold not reached in 100 steps)")

    # ==================================================================
    # Scenario 7 — detect_change() functional API
    # ==================================================================
    print("\n  [7] detect_change() functional API")

    # Stable then step-up
    series = [0.0] * 20 + [3.0] * 30
    detected, idx, score = detect_change(series, threshold=5.0, drift=0.5)

    print(f"    detected  : {detected}")
    print(f"    idx       : {idx}   (step where threshold crossed)")
    print(f"    score     : {score:.4f}  (cusum value at detection)")

    assert detected is True, "detect_change should find the step-up shift"
    assert idx > 20, f"Detection index must be after shift at 20, got {idx}"
    assert score >= 5.0, f"Score at detection must be >= threshold=5.0, got {score}"
    print(f"    ✓ detect_change() API works (detected at idx={idx})")

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n  All assertions passed.")
    print(f"  CUSUM step-up detection delay  : {fire_step - 20} steps")
    print(f"  CUSUM step-down detection delay: {fire_step3 - 20} steps")
    print(f"  Warmup suppression             : {det4.warmup_steps} steps")
    print(f"  detect_change() detection idx  : {idx}")
    print("=" * 60)


if __name__ == "__main__":
    run()
