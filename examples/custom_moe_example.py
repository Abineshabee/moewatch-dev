# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# examples/custom_moe_example.py
# =============================================================================
#
# Project      : MoEWatch
# Version      : v0.2.0
# Description  : Demonstrates MoEWatch on a fully custom MoE architecture —
#                one that doesn't follow Mixtral/DeepSeek/Qwen3 naming
#                conventions. Shows how to configure MoEWatch manually when
#                auto-detection can't find the routers, and how to use every
#                intervention action type.
#
#                Custom architecture features:
#                  - Non-standard router name: "dispatch_router" (not "gate")
#                  - Hierarchical expert groups: coarse → fine routing
#                  - Variable expert count per layer (4 / 8 / 16)
#                  - Soft MoE: top-1 dispatch (not top-k sparse)
#                  - Custom config.router_modules override to teach
#                    MoEWatch where the routers are
#
#                Four demonstrations in sequence:
#                  Demo 1 — Manual router_modules override
#                  Demo 2 — All four intervention action types explained
#                  Demo 3 — Live training loop with collapse → intervention
#                  Demo 4 — Low-level API: direct StatCollector + Analyzers
#
# Usage
# -----
#   pip install moewatch torch
#   python examples/custom_moe_example.py
#
# Author       : MoEWatch Example
# License      : Apache 2.0
#
# =============================================================================

from __future__ import annotations

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from unittest.mock import MagicMock

from moewatch import MoEWatch, MoEWatchCallback
from moewatch.config import WatchConfig, OutputMode, AlertLevel
from moewatch.collector.stat_collector import StatCollector
from moewatch.collector.baseline_tracker import BaselineTracker
from moewatch.analyzer.entropy import EntropyAnalyzer
from moewatch.analyzer.collapse import CollapseDetector
from moewatch.analyzer.risk_score import RiskScoreFuser
from moewatch.analyzer.gradient_starvation import GradientStarvationReport
from moewatch.hooks.router_hook import RoutingEvent
from moewatch.intervention.actions import (
    AuxLossAction,
    RouterNoiseAction,
    ExpertDropoutAction,
    NoOpAction,
)
from moewatch.intervention.engine import InterventionEngine
from moewatch.policy.rule_policy import RulePolicy


# ---------------------------------------------------------------------------
# Custom MoE architecture — non-standard naming
# ---------------------------------------------------------------------------
# Uses "dispatch_router" instead of "gate" and variable expert counts.
# MoEWatch auto-detection won't find "dispatch_router" by default —
# Demo 1 shows how to override this with config.router_modules.
# ---------------------------------------------------------------------------

VOCAB_SIZE  = 500
HIDDEN_DIM  = 48


class CustomExpert(nn.Module):
    """Single expert FFN — any architecture works here."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.up   = nn.Linear(hidden_dim, hidden_dim * 2)
        self.down = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


class CustomMoEBlock(nn.Module):
    """
    MoE block with non-standard router name.

    Uses 'dispatch_router' instead of the conventional 'gate' — this is
    the key reason MoEWatch auto-detection won't find it. We override
    config.router_modules = ['dispatch_router'] to tell MoEWatch explicitly.
    """

    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k

        # Non-standard name — MoEWatch won't auto-detect this
        self.dispatch_router = nn.Linear(hidden_dim, num_experts, bias=False)

        self.experts = nn.ModuleList([
            CustomExpert(hidden_dim) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        flat    = x.reshape(-1, D)

        logits  = self.dispatch_router(flat)
        probs   = torch.softmax(logits, dim=-1)
        top_p, top_i = probs.topk(self.top_k, dim=-1)
        top_p   = top_p / top_p.sum(-1, keepdim=True)

        out = torch.zeros_like(flat)
        for k in range(self.top_k):
            for e in range(self.num_experts):
                mask = (top_i[:, k] == e)
                if mask.any():
                    out[mask] += top_p[mask, k:k+1] * self.experts[e](flat[mask])

        return out.reshape(B, S, D)


class CustomTransformerLayer(nn.Module):
    """Single layer: self-attn stub + custom MoE block."""

    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 1):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.moe  = CustomMoEBlock(hidden_dim, num_experts, top_k)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.moe(self.norm2(x))
        return x


class CustomMoEModel(nn.Module):
    """
    Custom MoE model with variable expert counts per layer.

    Layer structure (6 layers):
      Layer 0: 4  experts, top-1  (coarse routing)
      Layer 1: 8  experts, top-2
      Layer 2: 8  experts, top-2
      Layer 3: 16 experts, top-2  (fine-grained routing)
      Layer 4: 16 experts, top-2
      Layer 5: 8  experts, top-2

    Router names:   layers.N.moe.dispatch_router
    Expert names:   layers.N.moe.experts
    """

    LAYER_CONFIG = [
        (4,  1),   # (num_experts, top_k)
        (8,  2),
        (8,  2),
        (16, 2),
        (16, 2),
        (8,  2),
    ]

    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.embed  = nn.Embedding(VOCAB_SIZE, hidden_dim)
        self.layers = nn.ModuleList([
            CustomTransformerLayer(hidden_dim, n_exp, top_k)
            for n_exp, top_k in self.LAYER_CONFIG
        ])
        self.norm   = nn.LayerNorm(hidden_dim)
        self.head   = nn.Linear(hidden_dim, VOCAB_SIZE, bias=False)

        # Fake config for aux_loss_coef (used by AuxLossAction)
        self.config = MagicMock()
        self.config.router_aux_loss_coef = 0.001

    # ------------------------------------------------------------------
    # Collapse injection helpers
    # ------------------------------------------------------------------

    def set_collapse(self, layer_idx: int, dominant_expert: int = 0,
                     strength: float = 5.0) -> None:
        """Route all tokens in layer_idx to dominant_expert."""
        with torch.no_grad():
            router = self.layers[layer_idx].moe.dispatch_router
            router.weight.zero_()
            router.weight[dominant_expert] = strength

    def reset_layer(self, layer_idx: int) -> None:
        """Restore uniform routing in layer_idx."""
        with torch.no_grad():
            router = self.layers[layer_idx].moe.dispatch_router
            nn.init.normal_(router.weight, 0, 0.02)

    def reset_all(self) -> None:
        for i in range(len(self.layers)):
            self.reset_layer(i)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))


# ---------------------------------------------------------------------------
# Shared config factory
# ---------------------------------------------------------------------------

def make_config(router_modules_override: bool = True) -> WatchConfig:
    """Build WatchConfig for the custom architecture.

    Key setting: router_modules=['dispatch_router'] tells MoEWatch to
    look for modules named 'dispatch_router' as routers, bypassing the
    default ['gate', 'router', 'wg'] heuristic.
    """
    return WatchConfig(
        output=OutputMode.SILENT,

        # Tell MoEWatch where to find the routers
        router_modules=["dispatch_router"] if router_modules_override else None,

        # Thresholds calibrated for this small model
        entropy_warn=0.50,
        entropy_critical=0.25,
        entropy_drop_warn=0.06,
        load_imbalance_warn=3.0,
        load_imbalance_error=6.0,
        dead_threshold=0.0005,
        cold_threshold=0.002,
        cold_steps_limit=15,

        log_every=10,
        sample_every=1,

        intervention_enabled=True,
        policy_type="rule",
        intervention_cooldown=10,
        intervention_max_delta=0.05,
        loss_guard_threshold=2.0,

        reward_window_steps=15,
        baseline_min_clean_steps=5,
        baseline_exclusion_window=15,
    )


# ---------------------------------------------------------------------------
# Demo 1 — Manual router_modules override
# ---------------------------------------------------------------------------

def demo1_router_override() -> None:
    print("=" * 70)
    print("  Demo 1 — Manual router_modules Override")
    print("=" * 70)
    print()
    print("  This model uses 'dispatch_router' — MoEWatch won't auto-detect it.")
    print("  We set config.router_modules=['dispatch_router'] to override.\n")

    torch.manual_seed(0)
    model = CustomMoEModel()

    # --- Without override: should detect 0 layers ---
    config_no_override = make_config(router_modules_override=False)
    try:
        watcher_bad = MoEWatch(model, config_no_override)
        trainer_bad = MagicMock()
        trainer_bad.model = model
        watcher_bad.attach(trainer_bad)
        n = watcher_bad.num_layers_monitored
        watcher_bad.stop()
        print(f"  Without override: {n} layers monitored (expected 0)")
    except Exception as e:
        print(f"  Without override: attach() raised — {type(e).__name__}: {e}")

    # --- With override: should detect 6 layers ---
    config_override = make_config(router_modules_override=True)
    watcher_good = MoEWatch(model, config_override)
    trainer_good = MagicMock()
    trainer_good.model = model
    watcher_good.attach(trainer_good)
    n = watcher_good.num_layers_monitored
    print(f"  With override:    {n} layers monitored (expected 6)")

    # Show detected layer names
    layer_map = watcher_good.hook_manager.get_layer_map()
    print(f"\n  Detected router layers:")
    for name in sorted(layer_map.keys()):
        n_exp = model.LAYER_CONFIG[int(name.split(".")[1])][0]
        print(f"    {name:<40}  ({n_exp} experts)")

    watcher_good.stop()
    print()


# ---------------------------------------------------------------------------
# Demo 2 — All four intervention action types
# ---------------------------------------------------------------------------

def demo2_action_types() -> None:
    print("=" * 70)
    print("  Demo 2 — All Four Intervention Action Types")
    print("=" * 70)
    print()

    torch.manual_seed(1)
    model   = CustomMoEModel()
    config  = make_config()
    trainer = MagicMock()
    trainer.model = model

    watcher  = MoEWatch(model, config)
    watcher.attach(trainer)

    layer_map = watcher.hook_manager.get_layer_map()
    layer_name = sorted(layer_map.keys())[2]   # pick layers.2.moe.dispatch_router

    bt     = watcher._baseline_tracker if hasattr(watcher, '_baseline_tracker') else None
    engine = InterventionEngine(config, trainer,
                                watcher.baseline_tracker
                                if hasattr(watcher, 'baseline_tracker') else
                                BaselineTracker(config))

    actions = [
        ("AuxLossAction",
         "Increases aux_loss_coef to penalise uneven expert load.",
         AuxLossAction(layer_name=layer_name, delta=0.01)),

        ("RouterNoiseAction",
         "Injects Gaussian noise into router logits to force exploration.",
         RouterNoiseAction(layer_name=layer_name, noise_scale=0.05)),

        ("ExpertDropoutAction",
         "Increases expert dropout to reduce over-reliance on hot experts.",
         ExpertDropoutAction(layer_name=layer_name, dropout_delta=0.1)),

        ("NoOpAction",
         "Conservative baseline — monitor only, no change applied.",
         NoOpAction(layer_name=layer_name)),
    ]

    for name, description, action in actions:
        action.mark_applied(step=1)
        print(f"  [{name}]")
        print(f"    {description}")
        print(f"    action_type = {action.action_type!r}")
        print(f"    delta       = {action.delta:+.4f}")
        print(f"    applied     = step {action.applied_step}")
        print(f"    repr        = {action!r}")
        print()

    watcher.stop()


# ---------------------------------------------------------------------------
# Demo 3 — Live training loop with collapse + intervention
# ---------------------------------------------------------------------------

def demo3_training_loop() -> None:
    print("=" * 70)
    print("  Demo 3 — Live Training Loop")
    print("  Custom MoE: 6 layers × variable experts × non-standard router")
    print("=" * 70)
    print()

    torch.manual_seed(42)
    model   = CustomMoEModel()
    config  = make_config()
    trainer = MagicMock()
    trainer.model = model

    watcher  = MoEWatch(model, config)
    callback = watcher.attach(trainer)

    layer_map = watcher.hook_manager.get_layer_map()
    print(f"  Monitoring {watcher.num_layers_monitored} layers:")
    for name in sorted(layer_map.keys()):
        idx   = int(name.split(".")[1])
        n_exp = model.LAYER_CONFIG[idx][0]
        top_k = model.LAYER_CONFIG[idx][1]
        print(f"    {name:<42}  {n_exp:>2} experts  top-{top_k}")
    print()

    optimizer   = torch.optim.AdamW(model.parameters(), lr=1e-4)
    total_steps = 120

    # Collapse layers 3 and 4 (the 16-expert fine-grained layers) starting
    # at step 41, simulating a realistic partial collapse scenario where
    # only the large-expert layers are affected.
    print(f"  {'step':>5}  {'phase':<28}  {'loss':>8}  {'risk':>7}  "
          f"{'alerts':>7}  {'intv':>5}  {'aux_coef':>10}")
    print(f"  {'-'*5}  {'-'*28}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*5}  {'-'*10}")

    for step in range(1, total_steps + 1):
        # Phase control
        if step <= 30:
            model.reset_all()
            phase = "warm-up"
        elif step <= 80:
            # Inject collapse only in layers 3 and 4
            pressure = (step - 30) / 50 * 5.0
            model.set_collapse(3, dominant_expert=0, strength=pressure)
            model.set_collapse(4, dominant_expert=1, strength=pressure)
            phase = f"collapse L3+L4 (p={pressure:.1f})"
        else:
            # Recovery: ease pressure gradually
            pressure = 5.0 - (step - 80) / 40 * 4.0
            model.set_collapse(3, dominant_expert=0,
                               strength=max(pressure, 1.0))
            model.set_collapse(4, dominant_expert=1,
                               strength=max(pressure, 1.0))
            phase = f"recovery (p={max(pressure, 1.0):.1f})"

        # Forward + backward
        watcher.pre_step(step)
        input_ids = torch.randint(0, VOCAB_SIZE, (2, 12))
        labels    = torch.randint(0, VOCAB_SIZE, (2, 12))
        optimizer.zero_grad()
        logits = model(input_ids)
        loss   = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )
        loss.backward()
        optimizer.step()

        report = watcher.step(global_step=step, current_loss=loss.item())

        if step % 10 == 0:
            worst_risk = max(report.risk_scores.values(), default=0.0)
            n_alerts   = sum(
                1 for a in report.alerts
                if a.level in (AlertLevel.WARNING, AlertLevel.CRITICAL)
            )
            n_intv    = len(report.active_interventions)
            aux_coef  = model.config.router_aux_loss_coef
            print(
                f"  {step:>5}  {phase:<28}  {loss.item():>8.4f}  "
                f"{worst_risk:>7.3f}  {n_alerts:>7}  {n_intv:>5}  "
                f"{aux_coef:>10.4f}"
            )

    # Post-run summary
    print()
    risk_summary = watcher.get_risk_summary()
    if risk_summary:
        sorted_risks = sorted(risk_summary.items(), key=lambda x: -x[1])
        print(f"  Top-3 riskiest layers:")
        for name, score in sorted_risks[:3]:
            bar   = "█" * int(score * 20)
            idx   = int(name.split(".")[1])
            n_exp = model.LAYER_CONFIG[idx][0]
            print(f"    {name:<42}  {score:.4f}  {bar:<20}  ({n_exp} experts)")

    all_alerts = watcher.get_alerts(since_step=0)
    by_level: dict = {}
    for a in all_alerts:
        by_level[a.level.value] = by_level.get(a.level.value, 0) + 1
    print(f"\n  Total alerts: {len(all_alerts)}")
    for lvl, cnt in sorted(by_level.items()):
        print(f"    {lvl.upper():<10}: {cnt}")

    watcher.stop()
    print()


# ---------------------------------------------------------------------------
# Demo 4 — Low-level API: direct StatCollector + Analyzers
# ---------------------------------------------------------------------------

def demo4_low_level_api() -> None:
    print("=" * 70)
    print("  Demo 4 — Low-Level API (StatCollector + Analyzers directly)")
    print("=" * 70)
    print()
    print("  Use this pattern when you want MoEWatch's analyzers without")
    print("  attaching hooks — e.g. feeding synthetic routing distributions")
    print("  to validate threshold calibration before a real training run.\n")

    config    = make_config()
    sc        = StatCollector(config)
    bt        = BaselineTracker(config)
    entropy   = EntropyAnalyzer(config)
    collapse  = CollapseDetector(config)
    fuser     = RiskScoreFuser(config)

    layer     = "layers.3.moe.dispatch_router"
    n_experts = 16
    sc.register_layer(layer, n_experts)
    bt.register_layer(layer)

    def feed_routing(weights: torch.Tensor, step: int, n_tokens: int = 128):
        """Directly write a RoutingEvent — no model forward pass needed."""
        logits   = weights.unsqueeze(0).expand(n_tokens, -1)
        selected = torch.multinomial(logits.clamp(min=1e-6),
                                     num_samples=2, replacement=False)
        sc.write_routing_event(RoutingEvent(
            timestamp=time.time(),
            global_step=step,
            layer_name=layer,
            routing_logits=logits,
            selected_experts=selected,
            expert_count=n_experts,
            batch_size=n_tokens,
        ))
        bt.update_signal(layer, value=float(weights.max()), step=step)

    def snapshot(step: int, starvation_score: float = 0.0):
        er_map = entropy.analyze(sc)
        cr_map = collapse.analyze(sc)
        er = er_map.get(layer)
        cr = cr_map.get(layer)
        grad = GradientStarvationReport(
            layer_name=layer, expert_id=0,
            starvation_score=starvation_score, step=step,
        )
        rr = fuser.fuse(grad, er) if er else None
        return er, cr, rr

    torch.manual_seed(7)

    # --- Healthy uniform routing ---
    print("  [A] Healthy routing (16 experts, uniform)")
    w_uniform = torch.ones(n_experts) / n_experts
    for s in range(30):
        feed_routing(w_uniform, s)
    er, cr, rr = snapshot(step=29)
    print(f"    entropy={er.normalized_entropy:.4f}  "
          f"dead={cr.num_dead_experts}  "
          f"risk={rr.risk_score:.4f}  "
          f"level={rr.risk_level.value.upper()}")

    # --- Partial collapse: top-2 experts take 80% of load ---
    print("\n  [B] Partial collapse (top-2 experts dominate)")
    for s in range(30, 70):
        alpha = (s - 30) / 39
        w = torch.ones(n_experts) * 0.5
        w[0] = 0.5 + alpha * 6.0
        w[1] = 0.5 + alpha * 4.0
        w = w / w.sum()
        feed_routing(w, s)
    er, cr, rr = snapshot(step=69, starvation_score=0.35)
    print(f"    entropy={er.normalized_entropy:.4f}  "
          f"drift={er.drift_detected}  "
          f"dead={cr.num_dead_experts}  "
          f"cold={cr.num_cold_experts}  "
          f"risk={rr.risk_score:.4f}  "
          f"level={rr.risk_level.value.upper()}")

    # --- Severe collapse: single expert monopoly ---
    print("\n  [C] Severe collapse (expert 0 monopoly)")
    w_collapse = torch.zeros(n_experts)
    w_collapse[0] = 1.0
    for s in range(70, 100):
        feed_routing(w_collapse, s)
    er, cr, rr = snapshot(step=99, starvation_score=0.95)
    print(f"    entropy={er.normalized_entropy:.4f}  "
          f"drift={er.drift_detected}  "
          f"dead={cr.num_dead_experts}  "
          f"risk={rr.risk_score:.4f}  "
          f"level={rr.risk_level.value.upper()}")

    # --- Show what intervention would fire at this risk score ---
    print()
    policy = RulePolicy(config)
    from moewatch.policy.base import InterventionState
    state  = InterventionState(
        layer_name=layer,
        risk_score=rr.risk_score,
        step=99,
        intervention_history=[],
    )
    action = policy.select_action(state)
    print(f"  RulePolicy → action for risk={rr.risk_score:.4f}: "
          f"{action.action_type!r}  (delta={action.delta:+.4f})")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  MoEWatch v0.2.0 — Custom MoE Architecture Example              ║")
    print("║  Non-standard router names, variable expert counts,             ║")
    print("║  manual override, all action types, low-level API               ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    demo1_router_override()
    demo2_action_types()
    demo3_training_loop()
    demo4_low_level_api()

    print("=" * 70)
    print("  All four demos complete.")
    print("  MoEWatch works on any MoE architecture — not just Mixtral/DeepSeek.")
    print("=" * 70)


if __name__ == "__main__":
    main()
