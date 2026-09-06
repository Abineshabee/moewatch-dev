# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# examples/cross_layer_walkthrough.py
# =============================================================================
#
# Project      : MoEWatch
# Version      : v0.2.0
# Description  : Standalone walkthrough of Tier 3 — cross-layer collapse
#                 propagation detection (CrossLayerCorrelation).
#
#                 Tiers 1 (gradient starvation) and 2 (entropy drift) tell
#                 you THAT a layer is collapsing. Tier 3 tells you WHERE a
#                 collapse is spreading FROM and TO: it watches every
#                 layer's entropy trajectory, finds which one is declining
#                 first ("source"), which other layers are moving in lockstep
#                 with it ("victims"), and how many steps typically separate
#                 the source's decline from a victim's — a localisation
#                 signal that neither Tier 1 nor Tier 2 can provide alone,
#                 since each of those only ever looks at one layer at a time.
#
#                 Covers:
#                   1. Why cross-layer signal matters (one-layer signals
#                      can't tell you if collapse is spreading)
#                   2. Feeding entropy directly to CrossLayerCorrelation
#                      (no model or trainer required)
#                   3. A genuine cascade: source leads, victims lag behind
#                   4. A false-alarm case: layers dip in sync but neither
#                      one is "causing" the other (no exploitable lag)
#                   5. The quiet case: independent, uncorrelated layers
#                      correctly report no source and no victims
#                   6. reset() — clearing history for one or all layers
#                   7. Live integration: this is what MoEWatch computes
#                      automatically every step once >= 2 layers exist —
#                      reading it straight off a real MoEWatch instance
#
#                 No GPU or HuggingFace Trainer required for sections 1-6;
#                 section 7 trains a tiny 2-layer MoE model on CPU.
#
# Author       : Abinesh N (@Abineshabee)
# Repository   : https://github.com/Abineshabee/MoEWatch
# License      : Apache 2.0
#
# Run
# ---
#   python examples/cross_layer_walkthrough.py
#
# =============================================================================

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from moewatch import MoEWatch, WatchConfig, OutputMode
from moewatch.analyzer.cross_layer import (
    CrossLayerCorrelation,
    _MIN_HISTORY_LENGTH,
    _CORRELATION_VICTIM_THRESHOLD,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

SEP  = "─" * 72
SEP2 = "═" * 72


def _banner(title: str) -> None:
    print(f"\n{SEP2}")
    print(f"  {title}")
    print(SEP2)


@dataclass
class FakeEntropyReport:
    """Minimal stand-in for a real ``LayerEntropyReport``.

    CrossLayerCorrelation only reads ``normalized_entropy`` and ``step``
    off whatever it's handed (see ``_ingest_entropy_reports``), so a tiny
    duck-typed object is all that's needed to exercise it directly —
    no model, hooks, or StatCollector required. In a live MoEWatch run,
    these come from ``EntropyAnalyzer.analyze()`` instead (see Section 7).
    """
    normalized_entropy: float
    step: int


def _print_report(report, label: str) -> None:
    print(f"\n  ── {label} " + "─" * max(0, 50 - len(label)))
    if not report.layer_order:
        print("    Not enough layers/history yet for correlation analysis.")
        return

    print(f"    layers analysed   : {report.layer_order}")
    print(f"    source_layer      : {report.source_layer or '(none — no clear decline)'}")
    print(f"    victim_layers     : {report.victim_layers or '(none)'}")
    print(f"    propagation_vel.  : "
          f"{f'{report.propagation_velocity:.1f} steps/layer' if report.propagation_velocity is not None else '(not computable)'}")
    print(f"    spread_score      : {report.spread_score:.3f}  "
          f"({int(len(report.victim_layers))}/{max(len(report.layer_order) - 1, 1)} layers affected)")

    print(f"\n    Correlation matrix (|r| >= {_CORRELATION_VICTIM_THRESHOLD} = victim):")
    header = "              " + "".join(f"{name:>10}" for name in report.layer_order)
    print(f"    {header}")
    for i, name in enumerate(report.layer_order):
        row = "".join(f"{report.correlation_matrix[i, j]:>10.2f}"
                       for j in range(len(report.layer_order)))
        print(f"    {name:<12}{row}")


# ── Section 1: Why this signal exists ────────────────────────────────────────

def section_1_why() -> None:
    _banner("1. Why a cross-layer signal?")
    print("""
  Tier 1 (gradient starvation) and Tier 2 (entropy drift) are both
  PER-LAYER signals — each one looks at a single layer in isolation and
  answers "is THIS layer collapsing?"

  Neither can answer a different, equally important question: in a model
  with many MoE layers, is a collapse in one layer an isolated glitch, or
  the leading edge of something spreading through the stack? That matters
  operationally — a single misbehaving layer might call for a small,
  targeted fix (e.g. AuxLoss on that one gate), while a spreading collapse
  calls for a broader response before it reaches layers that haven't
  destabilised yet.

  CrossLayerCorrelation (Tier 3) answers this by watching every layer's
  entropy trajectory over a rolling window and asking three questions:

    1. Which layer's entropy is declining fastest right now?
       → that's the "source".
    2. Which OTHER layers are moving in lockstep with the source's
       decline (high Pearson correlation)?
       → those are "victims".
    3. How many steps typically separate the source's decline from a
       victim's decline?
       → the "propagation velocity" — a rough early-warning lead time
         for layers that haven't collapsed yet but are correlated with
         one that has.

  It contributes the smallest weight (10%) to the fused risk score
  precisely because it's a LOCALISATION signal, not an earlier-warning
  one — Tiers 1 and 2 still fire first. Its value is telling you WHERE
  to intervene, not detecting collapse sooner.
""")


# ── Section 2: Exercise the analyzer directly ────────────────────────────────

def section_2_build_analyzer() -> CrossLayerCorrelation:
    _banner("2. Feeding entropy straight to CrossLayerCorrelation")
    print(f"""
  CrossLayerCorrelation.analyze() takes a dict[layer_name, entropy_report]
  every step — exactly what EntropyAnalyzer.analyze() already returns in a
  live MoEWatch run. Because it only reads two fields off each report
  (normalized_entropy, step), we can drive it directly with a tiny
  duck-typed stand-in and see exactly how it behaves, one step at a time,
  with no model, hooks, or training loop involved.

  Two thresholds matter throughout this walkthrough:
    _MIN_HISTORY_LENGTH        = {_MIN_HISTORY_LENGTH:>4}   (steps of history a layer
                                        needs before it's eligible at all)
    _CORRELATION_VICTIM_THRESHOLD = {_CORRELATION_VICTIM_THRESHOLD:.2f}   (|r| this high with the
                                        source = classified as a victim)
""")
    config = WatchConfig(output=OutputMode.SILENT)
    return CrossLayerCorrelation(config)


# ── Section 3: A genuine cascade ──────────────────────────────────────────────

def section_3_genuine_cascade(analyzer: CrossLayerCorrelation) -> None:
    _banner("3. Genuine cascade — source leads, victims lag")
    print("""
  Scenario: a 3-layer model where layer_0's entropy dips into a collapse
  and partially recovers (a Gaussian-shaped trough centred on step 18),
  layer_1 goes through the IDENTICAL trough shape 5 steps later (as if
  whatever destabilised layer_0 propagated downstream), and layer_2 stays
  healthy throughout — an unrelated, uninvolved layer in the same model.

  Why a trough shape and not a straight decline? Cross-correlation
  localises a lag by matching a distinctive FEATURE between two signals.
  Two monotonic ramps of the same slope stay highly correlated across a
  wide range of shifts — the correlation peak is broad and doesn't
  sharply pin down the true offset. A localised dip-and-recover shape,
  by contrast, gives cross-correlation something sharp to lock onto, so
  the estimated lag actually converges to the true delay we built in.

  We feed 40 steps of synthetic entropy and print the report at steps
  20 and 40 so you can watch the analyzer's picture sharpen as the full
  trough shape comes into view for both layers.
""")
    n_steps = 40
    for step in range(1, n_steps + 1):
        def trough(t: int, center: int, depth: float = 0.75, width: float = 6.0) -> float:
            return 1.0 - depth * math.exp(-((t - center) ** 2) / (2 * width ** 2))

        e0 = trough(step, center=18)          # layer_0's collapse-and-recover
        lag = 5
        e1 = trough(step, center=18 + lag)    # identical shape, 5 steps later
        e2 = 0.95 + 0.02 * math.sin(step * 1.7)   # healthy, mildly noisy, unrelated

        reports = {
            "layer_0": FakeEntropyReport(normalized_entropy=e0, step=step),
            "layer_1": FakeEntropyReport(normalized_entropy=e1, step=step),
            "layer_2": FakeEntropyReport(normalized_entropy=e2, step=step),
        }
        report = analyzer.analyze(reports)

        if step in (20, 40):
            _print_report(report, f"step {step}")

    print("""
  At step 20, layer_0's trough (centred at 18) is mostly visible but
  layer_1's (centred at 23) has barely started — the lag estimate isn't
  reliable yet with an incomplete victim trough.

  By step 40, both full troughs are visible and the analyzer correctly
  identifies layer_0 as the source (steepest decline), layer_1 as a
  victim (near-perfect correlation once both troughs are in view), and
  layer_2 as uninvolved (correlation near zero, not flagged).
  propagation_velocity should land close to the 5-step lag we built into
  the scenario (cross-correlation on discrete, noisy-ish data typically
  lands within a step or two of the true value — that's expected, not
  an error) — inferred purely from the two entropy curves, with no
  knowledge of how the data was generated.
""")


# ── Section 4: Synchronized but NOT causal ───────────────────────────────────

def section_4_false_alarm(analyzer: CrossLayerCorrelation) -> None:
    _banner("4. False-alarm case — synchronized, not cascading")
    print("""
  Scenario: two layers dip and recover at EXACTLY the same steps — e.g.
  both layers reacting to the same external shock (a batch of unusual
  inputs, a learning-rate spike) rather than one destabilising the other.

  This matters because naive "did two layers move together" detectors
  would flag this as a cascade. CrossLayerCorrelation still reports high
  correlation (the curves genuinely do move together) — but with ZERO
  lag, there's no directional "A causes B" story to tell, so
  propagation_velocity comes back as None: not computable, because the
  cross-correlation peak sits at lag 0, not a positive offset.
""")
    analyzer.reset()   # start clean — Section 3 left history behind
    n_steps = 25
    for step in range(1, n_steps + 1):
        shock = 0.35 * math.exp(-((step - 15) ** 2) / 8.0)   # shared bump
        e_a = 0.95 - shock
        e_b = 0.93 - shock   # same shape, same timing, different layer
        reports = {
            "layer_a": FakeEntropyReport(normalized_entropy=e_a, step=step),
            "layer_b": FakeEntropyReport(normalized_entropy=e_b, step=step),
        }
        report = analyzer.analyze(reports)

    _print_report(report, "final (step 25)")
    print("""
  Both layers show up as highly correlated (they moved together), and one
  may still be labelled "source" / "victim" based on which has the
  marginally steeper fitted slope — but propagation_velocity is None.
  That absence IS the signal: correlation alone never implies causality
  or spread direction, and this is what stops MoEWatch from reporting a
  confident "3 steps/layer propagation" number on a coincidence.
""")


# ── Section 5: Independent, healthy layers ───────────────────────────────────

def section_5_quiet_case(analyzer: CrossLayerCorrelation) -> None:
    _banner("5. Quiet case — independent, healthy layers")
    print("""
  Scenario: two layers with independent random noise around a healthy
  entropy baseline — no shared cause, no drift, no relationship at all.
  A well-behaved Tier 3 signal should report no source and no victims
  here, not manufacture a story out of noise.
""")
    analyzer.reset()
    torch.manual_seed(7)
    n_steps = 20
    report = None
    for step in range(1, n_steps + 1):
        e_x = 0.95 + 0.03 * float(torch.randn(1))
        e_y = 0.90 + 0.03 * float(torch.randn(1))
        reports = {
            "layer_x": FakeEntropyReport(normalized_entropy=max(0.0, min(1.0, e_x)), step=step),
            "layer_y": FakeEntropyReport(normalized_entropy=max(0.0, min(1.0, e_y)), step=step),
        }
        report = analyzer.analyze(reports)

    _print_report(report, "final (step 20)")
    print("""
  With no meaningfully negative slope on either layer, source_layer is
  None — there's nothing to call a "collapse origin" when nothing is
  collapsing. victim_layers is empty and spread_score is 0.0 as a direct
  consequence: the analyzer never gets far enough to test correlation at
  all here, because Step 4 of its algorithm (identify victims) only runs
  once a source has actually been identified.
""")


# ── Section 6: reset() ────────────────────────────────────────────────────────

def section_6_reset(analyzer: CrossLayerCorrelation) -> None:
    _banner("6. reset() — clearing history")
    print("""
  reset(layer_name) clears one layer's stored entropy window;
  reset() with no argument clears everything. Use this when you know a
  regime change makes old history misleading — e.g. right after a
  deliberate architecture change, a checkpoint reload, or (as
  Sections 3-5 just did) when moving on to a genuinely new scenario in
  the same process.
""")
    before = analyzer.get_entropy_window("layer_x")
    analyzer.reset(layer_name="layer_x")
    after = analyzer.get_entropy_window("layer_x")
    print(f"  layer_x window before reset(\"layer_x\") : {len(before)} points")
    print(f"  layer_x window after  reset(\"layer_x\") : {len(after)} points")

    analyzer.reset()
    print(f"  repr() after full reset()               : {analyzer!r}")


# ── Section 7: Live integration with a real MoEWatch run ────────────────────

class Expert(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.up   = nn.Linear(dim, dim * 2)
        self.down = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


class MoEBlock(nn.Module):
    """4-expert sparse (top_k=2) MoE block with a bias-controllable gate."""

    N_EXPERTS = 4
    TOP_K     = 2

    def __init__(self, dim: int):
        super().__init__()
        self.gate    = nn.Linear(dim, self.N_EXPERTS, bias=True)
        self.experts = nn.ModuleList([Expert(dim) for _ in range(self.N_EXPERTS)])
        nn.init.normal_(self.gate.weight, std=0.01)
        nn.init.zeros_(self.gate.bias)
        self.top_k = self.TOP_K
        self.collapse_bias = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.gate.bias.zero_()
            if self.collapse_bias != 0.0:
                self.gate.bias[0] = self.collapse_bias

        B, S, D = x.shape
        xf = x.reshape(-1, D)
        logits = self.gate(xf)
        weights, indices = torch.topk(F.softmax(logits, dim=-1), self.TOP_K, dim=-1)

        out = torch.zeros_like(xf)
        for k in range(self.TOP_K):
            for e in range(self.N_EXPERTS):
                mask = indices[:, k] == e
                if mask.any():
                    out[mask] += weights[mask, k].unsqueeze(-1) * self.experts[e](xf[mask])
        return out.reshape(B, S, D)


class TwoLayerMoE(nn.Module):
    """Two independent MoE blocks — layer_0's collapse pressure leads
    layer_1's by a fixed number of steps, modelling a genuine cascade
    inside a real (if tiny) model rather than synthetic entropy values.
    """

    def __init__(self, dim: int = 32):
        super().__init__()
        self.layer_0 = MoEBlock(dim)
        self.layer_1 = MoEBlock(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.layer_0(x)
        x = x + self.layer_1(x)
        return x

    def apply_schedule(self, step: int) -> None:
        # layer_0 ramps up starting step 10; layer_1 follows 15 steps later.
        self.layer_0.collapse_bias = min(6.0, max(0.0, (step - 10) * 0.3))
        self.layer_1.collapse_bias = min(6.0, max(0.0, (step - 25) * 0.3))


def section_7_live_integration() -> None:
    _banner("7. Live integration — reading Tier 3 off a real MoEWatch run")
    print("""
  Every live MoEWatch run already computes this signal automatically:
  step() calls cross_layer_analyzer.analyze(entropy_reports) internally
  once >= 2 layers are registered, and folds spread_score into the fused
  risk score as Tier 3 (10% weight). No opt-in is required — you only
  need >= 2 monitored layers for it to activate.

  This section trains a tiny 2-layer MoE model where layer_0's routing
  collapses starting at step 10, and layer_1 — architecturally identical
  but otherwise independent — starts an identical collapse 15 steps
  later, modelling one layer's instability spreading downstream.
""")
    torch.manual_seed(0)
    model = TwoLayerMoE(dim=32)
    config = WatchConfig(
        output=OutputMode.SILENT,
        entropy_warn=0.70,
        entropy_critical=0.40,
        intervention_enabled=False,   # isolate the Tier-3 signal; no interventions
        stats_window=60,
    )
    watch = MoEWatch(model, config)
    watch.start()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    n_steps = 60
    for step in range(1, n_steps + 1):
        model.apply_schedule(step)
        watch.pre_step(step)

        x = torch.randn(8, 12, 32, requires_grad=True)
        loss = model(x).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        watch.step(global_step=step, current_loss=float(loss.detach()))

    # Pull the Tier-3 report directly off the watcher's own analyzer —
    # this is the exact same object step() used internally all along.
    entropy_reports = watch.entropy_analyzer.analyze(watch.stat_collector)
    cross_report = watch.cross_layer_analyzer.analyze(entropy_reports)

    _print_report(cross_report, "live run, step 60")

    print(f"\n  Fused risk scores (Tier 3 folded in automatically):")
    for layer, score in watch.get_risk_summary().items():
        print(f"    {layer:<10}  {score:.4f}")

    watch.stop()
    print("""
  layer_0 should come back as the source (it started collapsing first),
  layer_1 as a victim (it repeats the same decline 15 steps later), and
  propagation_velocity should land in the neighbourhood of that 15-step
  offset — this time inferred from a real model's actual routing
  entropy, not hand-written synthetic values.
""")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(SEP2)
    print("  MoEWatch — Tier 3: Cross-Layer Collapse Propagation")
    print("  why → API → genuine cascade → false alarm → quiet case")
    print("  → reset() → live integration")
    print(SEP2)

    section_1_why()
    analyzer = section_2_build_analyzer()
    section_3_genuine_cascade(analyzer)
    section_4_false_alarm(analyzer)
    section_5_quiet_case(analyzer)
    section_6_reset(analyzer)
    section_7_live_integration()

    print(f"\n{SEP2}")
    print("  Done.")
    print("  For the other two tiers, see:")
    print("    examples/live_monitoring_walkthrough.py   (Tier 1 + Tier 2, live)")
    print("    examples/audit_walkthrough.py             (Tier 1 + Tier 2, offline)")
    print(SEP2)


if __name__ == "__main__":
    main()
