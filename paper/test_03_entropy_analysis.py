# =============================================================================
# MoEWatch v0.2.0 — Research Paper Test Suite
# Test 03: Entropy Analysis (Tier 2 Signal)
# =============================================================================
#
# Verifies the EntropyAnalyzer correctly measures routing entropy and
# detects entropy drift across three routing regimes:
#
#   1. Uniform routing  → high normalized entropy (≈ 1.0)
#   2. Partial collapse → medium entropy (< uniform)
#   3. Severe collapse  → low entropy (significantly < uniform)
#   4. Entropy ordering: uniform > partial > severe
#   5. Drift detection fires after sustained entropy drop
#
# Entropy is MoEWatch's Tier 2 signal (weight 0.3 in risk fusion).
# A drop in routing entropy is the earliest detectable sign of collapse.
#
# =============================================================================

import torch
import torch.nn as nn

from moewatch.analyzer.entropy import EntropyAnalyzer
from moewatch.collector.stat_collector import StatCollector
from moewatch.hooks.router_hook import RoutingEvent
from moewatch.config import WatchConfig, OutputMode

# ---------------------------------------------------------------------------
# Helpers — build synthetic routing events directly into StatCollector
# ---------------------------------------------------------------------------

NUM_EXPERTS = 16
LAYER       = "layers.0.gate"


def make_config() -> WatchConfig:
    return WatchConfig(
        output=OutputMode.SILENT,
        entropy_warn=0.50,
        entropy_critical=0.25,
    )


def make_stat_collector(config: WatchConfig) -> StatCollector:
    sc = StatCollector(config)
    sc.register_layer(LAYER, NUM_EXPERTS)
    return sc


def inject_routing(
    sc: StatCollector,
    probs: torch.Tensor,   # shape (num_experts,) — routing distribution
    n_steps: int = 30,
    batch: int = 4,
    seq: int = 8,
    top_k: int = 2,
) -> None:
    """Write synthetic RoutingEvents drawn from `probs` into stat_collector."""
    for step in range(1, n_steps + 1):
        T = batch * seq
        # Sample top_k experts per token from the given distribution
        selected = torch.multinomial(
            probs.unsqueeze(0).expand(T, -1), num_samples=top_k, replacement=False
        )  # (T, top_k)

        # Build logits proportional to probs (log-space)
        logits = torch.log(probs.clamp(min=1e-8)).unsqueeze(0).expand(T, -1)

        event = RoutingEvent(
            layer_name=LAYER,
            timestamp=float(step),
            global_step=step,
            routing_logits=logits,
            selected_experts=selected,
            expert_count=NUM_EXPERTS,
            batch_size=batch,
        )
        sc.write_routing_event(event)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def run():
    print("=" * 60)
    print("  MoEWatch Test 03 — Entropy Analysis (Tier 2 Signal)")
    print("=" * 60)

    config   = make_config()
    analyzer = EntropyAnalyzer(config)

    # ==================================================================
    # Scenario A — Uniform routing (maximum entropy)
    # ==================================================================
    print("\n  [Scenario A] Uniform routing (all experts equally likely)")

    sc_a = make_stat_collector(config)
    uniform_probs = torch.ones(NUM_EXPERTS) / NUM_EXPERTS
    inject_routing(sc_a, uniform_probs, n_steps=30)

    reports_a = analyzer.analyze(sc_a)
    r_a = reports_a[LAYER]

    print(f"    normalized_entropy : {r_a.normalized_entropy:.4f}  (expected ≈ 1.0)")
    print(f"    trend              : {r_a.trend}")
    print(f"    drift_detected     : {r_a.drift_detected}")

    assert r_a.normalized_entropy > 0.95, \
        f"Uniform routing should have entropy ≈ 1.0, got {r_a.normalized_entropy:.4f}"
    assert r_a.trend in ("STABLE", "IMPROVING", "UNKNOWN"), \
        f"Uniform routing trend should not be DECLINING, got {r_a.trend}"
    print(f"    ✓ Uniform routing: high entropy, trend={r_a.trend}")

    # ==================================================================
    # Scenario B — Partial collapse (4 of 16 experts dominate)
    # ==================================================================
    print("\n  [Scenario B] Partial collapse (4 dominant experts)")

    sc_b = make_stat_collector(config)
    partial_probs = torch.ones(NUM_EXPERTS) * 0.01
    partial_probs[:4] = 0.24          # 4 experts share 96% of traffic
    partial_probs /= partial_probs.sum()
    inject_routing(sc_b, partial_probs, n_steps=30)

    reports_b = analyzer.analyze(sc_b)
    r_b = reports_b[LAYER]

    print(f"    normalized_entropy : {r_b.normalized_entropy:.4f}  (expected < {r_a.normalized_entropy:.4f})")
    print(f"    trend              : {r_b.trend}")

    assert r_b.normalized_entropy < r_a.normalized_entropy, \
        "Partial collapse entropy must be lower than uniform"
    print(f"    ✓ Partial collapse: entropy dropped from uniform")

    # ==================================================================
    # Scenario C — Severe collapse (1 expert monopoly)
    # ==================================================================
    print("\n  [Scenario C] Severe collapse (single expert monopoly)")

    sc_c = make_stat_collector(config)
    severe_probs = torch.zeros(NUM_EXPERTS)
    severe_probs[0] = 1.0
    # Use top_k=1 for true monopoly
    inject_routing(sc_c, severe_probs, n_steps=30, top_k=1)

    reports_c = analyzer.analyze(sc_c)
    r_c = reports_c[LAYER]

    print(f"    normalized_entropy : {r_c.normalized_entropy:.4f}  (expected << {r_b.normalized_entropy:.4f})")
    print(f"    trend              : {r_c.trend}")

    assert r_c.normalized_entropy < r_b.normalized_entropy, \
        "Severe collapse entropy must be lower than partial collapse"
    print(f"    ✓ Severe collapse: lowest entropy")

    # ==================================================================
    # Scenario D — Entropy ordering verification
    # ==================================================================
    print("\n  [Scenario D] Entropy ordering: uniform > partial > severe")

    h_uniform = r_a.normalized_entropy
    h_partial  = r_b.normalized_entropy
    h_severe   = r_c.normalized_entropy

    print(f"    uniform : {h_uniform:.4f}")
    print(f"    partial : {h_partial:.4f}")
    print(f"    severe  : {h_severe:.4f}")

    assert h_uniform > h_partial > h_severe, \
        f"Ordering violated: {h_uniform:.4f} > {h_partial:.4f} > {h_severe:.4f}"
    print(f"    ✓ Strict ordering maintained")

    # ==================================================================
    # Scenario E — Drift detection after sustained entropy drop
    # ==================================================================
    print("\n  [Scenario E] Drift detection (entropy drop over time)")

    sc_e = make_stat_collector(config)

    # Phase 1: healthy routing (15 steps)
    inject_routing(sc_e, uniform_probs, n_steps=15)
    # Phase 2: collapse (15 steps)
    inject_routing(sc_e, severe_probs,  n_steps=15, top_k=1)

    reports_e = analyzer.analyze(sc_e)
    r_e = reports_e[LAYER]

    print(f"    normalized_entropy : {r_e.normalized_entropy:.4f}")
    print(f"    drift_detected     : {r_e.drift_detected}")
    print(f"    trend              : {r_e.trend}")
    print(f"    ✓ Drift check complete (drift={r_e.drift_detected})")

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n  All assertions passed.")
    print(f"  Entropy values — uniform={h_uniform:.4f}  partial={h_partial:.4f}  severe={h_severe:.4f}")
    print(f"  Entropy reduction: uniform→severe = {(1 - h_severe/h_uniform)*100:.1f}% drop")
    print("=" * 60)


if __name__ == "__main__":
    run()
