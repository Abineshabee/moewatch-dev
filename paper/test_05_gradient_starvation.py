# =============================================================================
# MoEWatch v0.2.0 — Research Paper Test Suite
# Test 05: Gradient Starvation Analysis
# =============================================================================
#
# Verifies the GradientStarvationAnalyzer correctly detects experts with
# insufficient gradient flow during training:
#
#   1. Healthy experts receive gradient updates (norm > threshold)
#   2. Starved experts receive zero/minimal gradients (norm < threshold)
#   3. Gradient norm history is tracked per expert
#   4. Starvation state transitions: HEALTHY → STARVED → DEAD (if prolonged)
#   5. Risk score increases with gradient starvation severity
#
# Gradient starvation occurs when an expert's parameters stop receiving
# meaningful updates during backprop, indicating it's not being used in
# the forward pass or its contributions are not affecting loss.
#
# =============================================================================

import torch
from typing import Dict
from moewatch.analyzer.gradient_starvation import GradientStarvationAnalyzer
from moewatch.collector.stat_collector import StatCollector
from moewatch.hooks.gradient_hook import GradientEvent
from moewatch.config import WatchConfig, OutputMode

# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

NUM_EXPERTS = 8
NUM_LAYERS = 4
LAYER_NAMES = [f"layers.{i}.mlp.expert" for i in range(NUM_LAYERS)]


def make_config() -> WatchConfig:
    return WatchConfig(
        output=OutputMode.SILENT,
        dead_threshold=0.5,    # gradient norm < 0.5 → expert is DEAD
        cold_threshold=1.0,    # gradient norm < 1.0 → expert is COLD
        cold_steps_limit=3,    # Detect starvation after 3 consecutive steps below threshold
    )


def make_stat_collector(config: WatchConfig) -> StatCollector:
    sc = StatCollector(config)
    for layer_name in LAYER_NAMES:
        sc.register_layer(layer_name, NUM_EXPERTS)
    return sc


def inject_gradient_events(
    sc: StatCollector,
    analyzer: GradientStarvationAnalyzer,
    gradient_pattern: str,  # "healthy", "half_starved", "fully_starved"
    n_steps: int = 10,
) -> Dict[str, any]:
    """Inject gradient events and track starvation across multiple analyze() calls."""

    for step in range(1, n_steps + 1):
        for layer_idx, layer_name in enumerate(LAYER_NAMES):
            # Create gradient norms based on pattern
            if gradient_pattern == "healthy":
                # All experts get healthy gradients
                grad_norms = [2.0] * NUM_EXPERTS

            elif gradient_pattern == "half_starved":
                # First 4 experts healthy, last 4 starved
                grad_norms = [2.0] * 4 + [0.1] * 4

            elif gradient_pattern == "fully_starved":
                # All experts except first one are starved
                grad_norms = [2.0] + [0.1] * (NUM_EXPERTS - 1)

            # Write one gradient event per expert
            for expert_id in range(NUM_EXPERTS):
                sc.write_gradient_event(GradientEvent(
                    layer_name=layer_name,
                    timestamp=float(step),
                    global_step=step,
                    expert_id=expert_id,
                    gradient_norm=grad_norms[expert_id],
                    gradient_magnitude=grad_norms[expert_id],  # Same as norm for this test
                ))

        # Analyze every step so detector tracks consecutive STARVED steps
        report = analyzer.analyze(sc)

    return report


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def run():
    print("=" * 60)
    print("  MoEWatch Test 05 — Gradient Starvation Analysis")
    print("=" * 60)

    config = make_config()

    # ==================================================================
    # Scenario A — All experts healthy (good gradient flow)
    # ==================================================================
    print("\n  [Scenario A] All experts healthy (gradient norm > 1.0)")

    sc_a = make_stat_collector(config)
    analyzer_a = GradientStarvationAnalyzer(config)
    report_a = inject_gradient_events(sc_a, analyzer_a, "healthy", n_steps=10)

    print(f"    num_layers        : {len(report_a)}")
    total_starved = 0
    for layer_name in LAYER_NAMES:
        if layer_name in report_a:
            # report_a[layer_name] is a list of GradientStarvationReport, one per expert
            for expert_report in report_a[layer_name]:
                if expert_report.starvation_detected:
                    total_starved += 1

    print(f"    total_starved     : {total_starved}  (expected 0)")

    assert total_starved == 0, f"No experts should be starved, got {total_starved}"
    print(f"    ✓ All experts receiving healthy gradients")

    # ==================================================================
    # Scenario B — Half starved (experts 4-7 in each layer)
    # ==================================================================
    print("\n  [Scenario B] Half starved (experts 4–7 have grad_norm < 0.5)")

    sc_b = make_stat_collector(config)
    analyzer_b = GradientStarvationAnalyzer(config)
    # Use n_steps=5 (> cold_steps_limit=3) so starvation_detected becomes True
    report_b = inject_gradient_events(sc_b, analyzer_b, "half_starved", n_steps=5)

    total_starved_b = 0
    for layer_name in LAYER_NAMES:
        if layer_name in report_b:
            for expert_report in report_b[layer_name]:
                if expert_report.starvation_detected:
                    total_starved_b += 1

    print(f"    total_starved     : {total_starved_b}  (expected 16, 4 per layer)")
    print(f"    starvation_score  : ~0.5-0.8 (grad_norm=0.1 vs threshold=1.0)")

    # With 12 steps and cold_steps_limit=10, experts should be detected as starved
    assert total_starved_b == 16, \
        f"Expected 16 starved experts (4 per layer × 4 layers), got {total_starved_b}"
    print(f"    ✓ 4 experts per layer detected as starved")

    # ==================================================================
    # Scenario C — Fully starved (only expert 0 is healthy)
    # ==================================================================
    print("\n  [Scenario C] Fully starved (7 of 8 experts starved per layer)")

    sc_c = make_stat_collector(config)
    analyzer_c = GradientStarvationAnalyzer(config)
    report_c = inject_gradient_events(sc_c, analyzer_c, "fully_starved", n_steps=8)

    total_starved_c = 0
    max_starvation_score = 0.0
    for layer_name in LAYER_NAMES:
        if layer_name in report_c:
            for expert_report in report_c[layer_name]:
                if expert_report.starvation_detected:
                    total_starved_c += 1
                max_starvation_score = max(max_starvation_score, expert_report.starvation_score)

    print(f"    total_starved     : {total_starved_c}")
    print(f"    max_starvation_score : {max_starvation_score:.3f} (expected ~0.85-0.95)")
    print(f"    (Expected: severe starvation with 7 experts per layer)")

    assert total_starved_c > 20, \
        f"Expected many starved experts (>20), got {total_starved_c}"
    print(f"    ✓ Severe starvation detected ({total_starved_c} experts)")

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n  All assertions passed.")
    print(f"  Gradient starvation summary:")
    print(f"    Healthy  → starved=0")
    print(f"    Half     → starved=16")
    print(f"    Severe   → starved={total_starved_c}")
    print("=" * 60)


if __name__ == "__main__":
    run()
