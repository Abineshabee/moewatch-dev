"""
train_with_moewatch.py
=======================
Trains the SAME tiny MoE language model with the SAME seed, data,
optimizer, and collapse-pressure schedule as train_without_moewatch.py.
The ONLY difference: MoEWatch is attached and actively intervening.

How MoEWatch prevents dead experts:
  Layer 0 — AuxLossAction:      raises router_aux_loss_coef, adding a
                                  load-balancing penalty to the CE loss.
  Layer 1 — RouterNoiseAction:  injects Gaussian noise into gate logits.
  Layer 2 — ExpertDropoutAction: raises Dropout.p on the dominant expert,
                                  forcing the model to rely on other experts.

Why intervention_max_delta=1.5:
  The collapse bias is held at peak=2.0-2.2 permanently (a sustained
  distribution shift). RouterNoiseAction needs noise_std≥1.0 to
  counteract that. The default max_delta=0.1 is designed for HF models
  with much larger hidden dims; for HIDDEN_DIM=32 we need a larger value.

Without MoEWatch: all 3 non-dominant experts go dead (< 2% usage) at
  peak pressure (layers 1 and 2 end at entropy ≈ 0.00).
With MoEWatch: entropy stays above ~0.60 and dead-expert count stays
  at 0 throughout training.

Run:
    python train_with_moewatch.py
"""

import json
import torch
import torch.nn.functional as F

from moewatch import MoEWatch, WatchConfig, OutputMode, AlertLevel
from moewatch.intervention.actions import NoOpAction
from moewatch.policy.rule_policy import RulePolicy

from moe_model import (
    TinyMoELM, build_markov_transition, sample_batch,
    SEED, BATCH_SIZE, SEQ_LEN, TOTAL_STEPS, LR, NUM_EXPERTS,
    EVAL_BATCH_SIZE, PRESSURE_SCHEDULE,
)

DEAD_THRESHOLD = 0.02
LOG_EVERY      = 1


# Keep the original comparison setup unchanged, but make the three MoE layers
# demonstrate the three real MoEWatch intervention types in one training run.
# Layer 0 -> AuxLoss, Layer 1 -> RouterNoise, Layer 2 -> ExpertDropout.
# The trigger points are the existing pressure-window starts from moe_model.py.
class LayerInterventionPolicy(RulePolicy):
    ACTION_BY_LAYER = {
        0: "aux_loss",
        1: "router_noise",
        2: "expert_dropout",
    }
    THRESHOLD_BY_LAYER = {
        0: 0.30,
        1: 0.60,
        2: 0.80,
    }
    DOMINANT_BY_LAYER = {
        layer_idx: dominant
        for layer_idx, dominant, _start, _end, _peak in PRESSURE_SCHEDULE
    }

    def select_action(self, state):
        layer_idx = int(state.layer_id)
        action_type = self.ACTION_BY_LAYER.get(layer_idx)
        threshold = self.THRESHOLD_BY_LAYER.get(layer_idx, 1.0)

        # Keep the comparison's existing thresholds meaningful: each layer
        # is allowed to reach its assigned RulePolicy risk tier before its
        # corresponding intervention is applied.
        if action_type is None or state.risk_score < threshold:
            return NoOpAction(layer_name=state.layer_name or f"layer_{layer_idx}")

        target_layer = state.layer_name or f"layer_{layer_idx}"
        if action_type == "expert_dropout":
            target_layer = target_layer.rsplit(".gate", 1)[0]
            target_layer = (
                f"{target_layer}.experts."
                f"{self.DOMINANT_BY_LAYER[layer_idx]}"
            )

        return self._build_action(
            action_type,
            target_layer,
            max_delta=self.config.intervention_max_delta,
        )


def make_config() -> WatchConfig:
    """Intervention config calibrated for HIDDEN_DIM=32 and bias=2.0-2.2 pressure.

    intervention_max_delta=1.5:
        RouterNoiseAction is capped at noise_std = min(default, max_delta).
        At bias=2.0, noise_std=1.5 recovers entropy to ~0.80 (analytically
        confirmed). The default 0.1 is too small for this model scale and
        bias magnitude.

    entropy_warn=0.65, entropy_critical=0.40:
        WARNING fires before dead experts appear; CRITICAL fires while
        still recoverable with noise injection.

    intervention_cooldown=5:
        Allows rapid re-intervention as the bias ramps up quickly.
        Prevents cooldown from blocking the policy during a fast ramp.

    loss_guard_threshold=5.0:
        Relaxed because CE loss on this tiny model is noisy (range 3-4.5).
        A threshold of 1.5x would falsely block interventions during
        normal training spikes.

    stats_window=20:
        WatchConfig's own docstring for this field says it plainly: "the
        default of 100 suits production runs of 1,000+ steps; for shorter
        benchmarks or fast-iteration experiments, values of 20-50 give
        quicker response." This 400-step demo on a tiny model is exactly
        that "shorter benchmark" case, yet every other parameter here was
        hand-tuned for it while this one was left at the 100-step
        production default. At window=100, EntropyAnalyzer/RiskScoreFuser
        see a trailing 100-step average of the routing distribution — so
        during the ~50-100 step collapse ramp this schedule produces, the
        risk score stays deceptively low (diluted by the many still-healthy
        older steps still inside the window) long after the *current*
        routing has actually collapsed. That is what caused intervention
        to fire only twice across all 400 steps and let layers 1-2 rack up
        dead-expert steps before recovering. window=20 keeps the analyzer's
        view close to the model's actual current state.
    """
    return WatchConfig(
        output=OutputMode.SILENT,
        entropy_warn=0.65,
        entropy_critical=0.40,
        entropy_drop_warn=0.06,
        dead_threshold=0.05,
        cold_threshold=0.15,
        cold_steps_limit=10,
        log_every=10,
        sample_every=1,
        intervention_enabled=True,
        policy_type="rule",
        intervention_cooldown=5,
        intervention_max_delta=1.5,   # key: allows noise_std=1.5 to beat bias=2.0
        loss_guard_threshold=5.0,
        reward_window_steps=10,
        baseline_min_clean_steps=5,
        baseline_exclusion_window=5,
        stats_window=20,               # key: fast response for this short/tiny run
    )


def main():
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)

    trans = build_markov_transition(SEED)
    model = TinyMoELM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)

    eval_batch  = sample_batch(trans, batch_size=EVAL_BATCH_SIZE, seq_len=SEQ_LEN,
                               generator=torch.Generator().manual_seed(999))
    eval_inputs = eval_batch[:, :-1]

    config  = make_config()
    watcher = MoEWatch(model, config)
    watcher.start()
    # Keep MoEWatch's normal detection, safety checks, engine and reporting;
    # only constrain the action type by layer for this controlled comparison.
    watcher.policy = LayerInterventionPolicy(config)
    print(f"  MoEWatch auto-detected {watcher.num_layers_monitored} router layers: "
          f"{watcher._layer_order}\n")

    history = {
        "step":          [],
        "loss":          [],
        "entropy":       {n: [] for n in model.gate_names()},
        "dead_experts":  {n: [] for n in model.gate_names()},
        "risk_score":    {n: [] for n in model.gate_names()},
        "interventions": [],
    }

    print("=" * 105)
    print("  TRAINING WITH MOEWATCH  (live detection + intervention)")
    print("=" * 105)
    header = (f"  {'step':>5}  {'loss':>8}  {'phase':<12}"
              + "  ".join(f"{n.split('.')[1]:>8}-ent" for n in model.gate_names())
              + "  " + "  ".join(f"L{i}dead" for i in range(3))
              + f"  {'max_risk':>8}  {'action':<20}")
    print(header)

    for step in range(1, TOTAL_STEPS + 1):
        phase = model.apply_pressure_schedule(step)

        watcher.pre_step(step)

        batch = sample_batch(trans, BATCH_SIZE, SEQ_LEN, gen)
        inputs, targets = batch[:, :-1], batch[:, 1:]

        optimizer.zero_grad()
        logits  = model(inputs)
        ce_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        aux_coef = float(model.config.router_aux_loss_coef)
        loss = ce_loss + aux_coef * model.last_aux_loss_total
        loss.backward()
        optimizer.step()

        report = watcher.step(global_step=step, current_loss=ce_loss.item())

        applied_actions = [a.action_type for a in report.active_interventions]
        if applied_actions:
            history["interventions"].append({"step": step, "actions": applied_actions})

        if step % LOG_EVERY == 0 or step == 1:
            usage, entropy = model.expert_usage(eval_inputs)
            history["step"].append(step)
            history["loss"].append(ce_loss.item())

            row = [f"{step:>5}", f"{ce_loss.item():>8.4f}", f"{phase:<12}"]
            dead_cols = []
            for name in model.gate_names():
                ent  = entropy[name]
                dead = sum(1 for f in usage[name] if f < DEAD_THRESHOLD)
                history["entropy"][name].append(ent)
                history["dead_experts"][name].append(dead)
                history["risk_score"][name].append(report.risk_scores.get(name, 0.0))
                row.append(f"{ent:>11.3f}")
                dead_cols.append(f"{dead:>5d}")
            row.extend(dead_cols)
            max_risk   = max(report.risk_scores.values(), default=0.0)
            action_str = ",".join(applied_actions) if applied_actions else "-"
            row.append(f"{max_risk:>8.3f}")
            row.append(f"{action_str:<20}")
            print("  " + "  ".join(row))

    # --- Final report ---
    usage, entropy = model.expert_usage(eval_inputs)
    print(f"\n  Final expert usage (healthy uniform = {1/NUM_EXPERTS:.3f} each):")
    for name in model.gate_names():
        frac_str = ", ".join(f"{f:.3f}" for f in usage[name])
        dead = sum(1 for f in usage[name] if f < DEAD_THRESHOLD)
        print(f"    {name:<22} entropy={entropy[name]:.3f}  dead={dead}  "
              f"usage=[{frac_str}]")

    all_alerts = watcher.get_alerts(since_step=0)
    by_level   = {}
    for a in all_alerts:
        by_level[a.level.value] = by_level.get(a.level.value, 0) + 1
    print(f"\n  Total alerts raised      : {len(all_alerts)}")
    for lvl, cnt in sorted(by_level.items()):
        print(f"    {lvl.upper():<10}: {cnt}")
    print(f"  Total interventions      : {len(history['interventions'])}")
    print(f"  Final router_aux_loss_coef: {model.config.router_aux_loss_coef:.4f} "
          f"(started at 0.0)")

    watcher.stop()

    with open("with_moewatch_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("\n  Saved -> with_moewatch_history.json")


if __name__ == "__main__":
    main()
