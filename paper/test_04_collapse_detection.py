# =============================================================================
# MoEWatch v0.2.0 — Research Paper Test Suite
# Test 04: Collapse Detection (Expert Health States)
# =============================================================================
#
# Verifies the CollapseDetector correctly classifies each expert's health
# state across three regimes:
#
#   1. All experts HEALTHY when routing is uniform
#   2. Unrouted experts classified as DEAD after sustained zero load
#   3. Sporadically-routed experts classified as COLD
#   4. ExpertStatus enum values: HEALTHY / COLD / DEAD
#   5. Dead expert count matches expectation
#   6. Layer-level collapse report fields are populated
#
# CollapseDetector is MoEWatch's expert-level health signal.
# It feeds dead_expert_count into AuditReport and triggers CRITICAL alerts.
#
# =============================================================================

import torch
from typing import Dict
from moewatch.analyzer.collapse import CollapseDetector, ExpertStatus
from moewatch.collector.stat_collector import StatCollector
from moewatch.hooks.router_hook import RoutingEvent
from moewatch.config import WatchConfig, OutputMode

# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

NUM_EXPERTS = 8
LAYER       = "layers.0.gate"


def make_config() -> WatchConfig:
    return WatchConfig(
        output=OutputMode.SILENT,
        dead_threshold=0.01,   # experts below this utilization → DEAD
        cold_threshold=0.05,   # experts below this → COLD
        cold_steps_limit=10,   # promote COLD → DEAD after 10 steps (for faster testing)
    )


def make_stat_collector(config: WatchConfig) -> StatCollector:
    sc = StatCollector(config)
    sc.register_layer(LAYER, NUM_EXPERTS)
    return sc


def inject_routing(
    sc: StatCollector,
    detector: CollapseDetector,
    probs: torch.Tensor,
    n_steps: int = 40,
    batch: int = 4,
    seq: int = 16,
    top_k: int = 2,
) -> Dict[str, "LayerCollapseReport"]:
    """Inject routing events and track expert state across multiple analyze() calls.

    This is important because consecutive_cold_steps is tracked across multiple
    analyze() calls, not within a single call. Each analyze() call can increment
    the counter for experts that remain cold.
    """
    T = batch * seq
    final_report = {}

    for step in range(1, n_steps + 1):
        # Clamp probs to avoid zero-weight multinomial errors
        safe_probs = probs.clamp(min=1e-9)
        safe_probs = safe_probs / safe_probs.sum()
        selected = torch.multinomial(
            safe_probs.unsqueeze(0).expand(T, -1),
            num_samples=top_k,
            replacement=False,
        )
        logits = torch.log(safe_probs).unsqueeze(0).expand(T, -1)
        sc.write_routing_event(RoutingEvent(
            layer_name=LAYER,
            timestamp=float(step),
            global_step=step,
            routing_logits=logits,
            selected_experts=selected,
            expert_count=NUM_EXPERTS,
            batch_size=batch,
        ))

        # Call analyze() every step so detector can track consecutive_cold_steps
        final_report = detector.analyze(sc)

    return final_report


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def run():
    print("=" * 60)
    print("  MoEWatch Test 04 — Collapse Detection (Expert Health)")
    print("=" * 60)

    config = make_config()

    # ==================================================================
    # Scenario A — All experts healthy (uniform routing)
    # ==================================================================
    print("\n  [Scenario A] Uniform routing — all experts should be HEALTHY")

    sc_a      = make_stat_collector(config)
    detector_a = CollapseDetector(config)
    uniform_p = torch.ones(NUM_EXPERTS) / NUM_EXPERTS
    reports_a = inject_routing(sc_a, detector_a, uniform_p, n_steps=60)
    r_a       = reports_a[LAYER]

    print(f"    layer_name         : {r_a.layer_name}")
    print(f"    num_experts        : {len(r_a.expert_states)}")
    print(f"    dead_expert_count  : {r_a.num_dead_experts}  (expected 0)")
    print(f"    cold_expert_count  : {r_a.num_cold_experts}  (expected 0)")
    statuses_a = [r_a.expert_states[i].status.value for i in sorted(r_a.expert_states)]
    print(f"    expert statuses    : {statuses_a}")

    assert r_a.num_dead_experts == 0, \
        f"Uniform routing should have 0 dead experts, got {r_a.num_dead_experts}"
    assert r_a.num_cold_experts == 0, \
        f"Uniform routing should have 0 cold experts, got {r_a.num_cold_experts}"
    assert all(s.status == ExpertStatus.HEALTHY for s in r_a.expert_states.values()), \
        "All experts should be HEALTHY under uniform routing"
    print(f"    ✓ All {len(r_a.expert_states)} experts classified as HEALTHY")

    # ==================================================================
    # Scenario B — Dead experts (experts 4–7 receive zero tokens)
    # ==================================================================
    print("\n  [Scenario B] 4 dead experts (experts 4–7 never routed)")

    sc_b    = make_stat_collector(config)
    detector_b = CollapseDetector(config)
    dead_p  = torch.zeros(NUM_EXPERTS)
    dead_p[:4] = 0.25   # only experts 0–3 receive tokens
    reports_b = inject_routing(sc_b, detector_b, dead_p, n_steps=60, top_k=2)
    r_b       = reports_b[LAYER]

    print(f"    dead_expert_count  : {r_b.num_dead_experts}  (expected 4)")
    statuses_b = [r_b.expert_states[i].status for i in sorted(r_b.expert_states)]
    print(f"    expert statuses    : {[s.value for s in statuses_b]}")

    assert r_b.num_dead_experts == 4, \
        f"Expected 4 dead experts, got {r_b.num_dead_experts}"
    for i in range(4):
        assert r_b.expert_states[i].status == ExpertStatus.HEALTHY, \
            f"Expert {i} should be HEALTHY, got {r_b.expert_states[i].status.value}"
    for i in range(4, 8):
        assert r_b.expert_states[i].status == ExpertStatus.DEAD, \
            f"Expert {i} should be DEAD, got {r_b.expert_states[i].status.value}"
    print(f"    ✓ Experts 0–3: HEALTHY  |  Experts 4–7: DEAD")

    # ==================================================================
    # Scenario C — Cold experts (experts 4–7 barely routed)
    # ==================================================================
    print("\n  [Scenario C] Cold experts (experts 4–7 receive tiny load)")

    sc_c   = make_stat_collector(config)
    detector_c = CollapseDetector(config)
    cold_p = torch.zeros(NUM_EXPERTS)
    cold_p[:4]  = 0.24   # dominant: experts 0-3 get most traffic
    cold_p[4:]  = 0.03   # cold: experts 4-7 get minimal load (~2.78% each after norm)
    cold_p /= cold_p.sum()
    # After normalization: experts 4-7 get ~2.78% each (BELOW 5% cold_threshold)
    # So they will be classified as COLD
    # Use n_steps=8: even if expert gets zero tokens every step, max 8 COLD steps < 10 limit
    reports_c = inject_routing(sc_c, detector_c, cold_p, n_steps=8, top_k=2)
    r_c       = reports_c[LAYER]

    print(f"    dead_expert_count  : {r_c.num_dead_experts}  (expected 0)")
    print(f"    cold_expert_count  : {r_c.num_cold_experts}  (expected > 0)")
    statuses_c = [r_c.expert_states[i].status.value for i in sorted(r_c.expert_states)]
    print(f"    expert statuses    : {statuses_c}")

    assert r_c.num_dead_experts == 0, \
        f"Cold experts should not be DEAD, got dead={r_c.num_dead_experts}"
    assert r_c.num_cold_experts > 0, \
        f"Expected some COLD experts, got cold={r_c.num_cold_experts}"
    print(f"    ✓ {r_c.num_cold_experts} COLD expert(s) detected")

    # ==================================================================
    # Scenario D — ExpertStatus enum values
    # ==================================================================
    print("\n  [Scenario D] ExpertStatus enum values")

    print(f"    ExpertStatus.HEALTHY.value = {ExpertStatus.HEALTHY.value!r}")
    print(f"    ExpertStatus.COLD.value    = {ExpertStatus.COLD.value!r}")
    print(f"    ExpertStatus.DEAD.value    = {ExpertStatus.DEAD.value!r}")

    assert isinstance(ExpertStatus.HEALTHY.value, str)
    assert isinstance(ExpertStatus.COLD.value, str)
    assert isinstance(ExpertStatus.DEAD.value, str)
    print(f"    ✓ All enum values are strings (JSON-serializable)")

    # ==================================================================
    # Scenario E — Collapse report field coverage
    # ==================================================================
    print("\n  [Scenario E] LayerCollapseReport field coverage")

    fields = vars(r_b)
    print(f"    Available fields: {list(fields.keys())}")
    for required in ("layer_name", "num_dead_experts", "num_cold_experts",
                     "num_healthy_experts", "expert_states"):
        assert required in fields, f"Missing field: {required}"
    print(f"    ✓ All required report fields present")

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n  All assertions passed.")
    print(f"  Expert classification summary:")
    print(f"    Uniform   → dead=0   cold=0   healthy={len(r_a.expert_states)}")
    print(f"    Dead×4    → dead={r_b.num_dead_experts}   cold={r_b.num_cold_experts}   healthy={r_b.num_healthy_experts}")
    print(f"    Cold×4    → dead={r_c.num_dead_experts}   cold={r_c.num_cold_experts}   healthy={r_c.num_healthy_experts}")
    print("=" * 60)


if __name__ == "__main__":
    run()
