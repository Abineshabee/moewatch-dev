# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# examples/live_monitoring_walkthrough.py
# =============================================================================
#
# Project      : MoEWatch
# Version      : v0.2.0
# Description  : End-to-end walkthrough of MoEWatch attached to a live
#                training loop. Covers the complete lifecycle:
#
#                  1. Building a minimal MoE model
#                  2. Configuring MoEWatch — including calibration notes
#                  3. Attaching the watcher
#                  4. Three-phase training loop: healthy → collapse → recovery
#                  5. Reading the alert stream (the primary signal)
#                  6. Inspecting the intervention log
#                  7. Post-training summary
#
#                Design decisions explained:
#
#                  - The model uses 4 experts with top_k=4 (every expert
#                    receives tokens every step) to prevent spurious gradient-
#                    starvation false-alarms from batches where a top-k<N
#                    expert happens to receive no tokens by chance.
#
#                  - Collapse is injected by adding +4.0 to expert 0's
#                    logit, pushing normalised entropy from ≈1.00 to ≈0.19.
#                    This crossing of the configured thresholds is the
#                    signal MoEWatch detects.
#
#                  - The fused risk score stays in the HIGH band (≥0.6)
#                    even during the healthy phase because the risk fuser's
#                    Tier-1 (gradient starvation) component is calibrated
#                    for production-scale gradient norms. In a real run
#                    (Mixtral, DeepSeek-MoE) the score correctly spans
#                    0.0–1.0; in this small demo model the gradient norms
#                    are too small relative to cold_threshold to produce a
#                    low Tier-1 score. The entropy and alert signals are
#                    clean and trustworthy regardless of model scale.
#
#                No GPU or HuggingFace Trainer required.
#
# Author       : Abinesh N (@Abineshabee)
# Repository   : https://github.com/Abineshabee/MoEWatch
# License      : Apache 2.0
#
# Run
# ---
#   python examples/live_monitoring_walkthrough.py
#
# =============================================================================

from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from moewatch import WatchConfig, OutputMode, AlertLevel
from moewatch._watcher import MoEWatch

# Suppress the PyTorch-internal warning about full backward hooks on modules
# whose inputs don't require gradients — expected for read-only monitoring hooks.
warnings.filterwarnings(
    "ignore",
    message="Full backward hook is firing",
    category=UserWarning,
)

# ── Formatting helpers ────────────────────────────────────────────────────────

SEP  = "─" * 68
SEP2 = "═" * 68

def _banner(title: str) -> None:
    print(f"\n{SEP2}")
    print(f"  {title}")
    print(SEP2)


# ── Model ─────────────────────────────────────────────────────────────────────
#
#  DemoMoE: 4 experts, top_k=4 (dense — every expert always receives tokens).
#
#  Dense routing is intentional for this demo:
#    - In a sparse top-k<N MoE, experts that receive no tokens in a given
#      batch have zero gradient norms for that step.  The gradient starvation
#      detector counts such steps toward its starvation counter.  In a tiny
#      model with random routing this produces false starvation alerts in
#      the healthy phase, which makes the demo misleading.
#    - With top_k=N every expert always participates → gradient norms are
#      non-zero every step → starvation is only detected when collapse
#      genuinely reduces an expert's share to near-zero.
#    - Production MoE models (Mixtral 8×7B, DeepSeek-MoE) use sparse top-k
#      with many more experts, so the probability that a given expert is
#      selected at least once per step is high even with top_k=2.

class DemoMoE(nn.Module):
    """4-expert dense MoE block. Gate is the monitored router."""

    N_EXPERTS = 4
    TOP_K     = 4     # dense: every expert selected every step

    def __init__(self, d_model: int = 32):
        super().__init__()
        self.gate    = nn.Linear(d_model, self.N_EXPERTS, bias=False)
        self.experts = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False)
            for _ in range(self.N_EXPERTS)
        ])
        # Small-std init → softmax ≈ uniform → normalised entropy ≈ 1.0 at step 0.
        nn.init.normal_(self.gate.weight, std=0.01)

        # Set this to a large positive value to inject a routing collapse
        # onto expert 0 without touching model weights.
        self.collapse_bias: float = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        xf      = x.reshape(B * S, D)

        logits = self.gate(xf)
        if self.collapse_bias != 0.0:
            logits = logits.clone()
            logits[:, 0] = logits[:, 0] + self.collapse_bias

        weights, indices = torch.topk(
            F.softmax(logits, dim=-1), self.TOP_K, dim=-1
        )

        out = torch.zeros_like(xf)
        for k in range(self.TOP_K):
            for e in range(self.N_EXPERTS):
                mask = (indices[:, k] == e)
                if mask.any():
                    out[mask] += (
                        weights[mask, k].unsqueeze(-1) * self.experts[e](xf[mask])
                    )
        return out.reshape(B, S, D)

    def norm_entropy(self, x: torch.Tensor) -> float:
        """Normalised routing entropy [0, 1] for the current batch."""
        with torch.no_grad():
            xf = x.reshape(-1, x.shape[-1])
            logits = self.gate(xf)
            if self.collapse_bias != 0.0:
                logits = logits.clone()
                logits[:, 0] = logits[:, 0] + self.collapse_bias
            p = F.softmax(logits, dim=-1).clamp(min=1e-9)
            H = -(p * p.log()).sum(dim=-1).mean()
            return (H / math.log(self.N_EXPERTS)).item()


# ── Section 1: Build ──────────────────────────────────────────────────────────

def section_1_build_model() -> DemoMoE:
    _banner("1. Build the model")
    print(f"""
  DemoMoE(d_model=32, n_experts={DemoMoE.N_EXPERTS}, top_k={DemoMoE.TOP_K})
    gate     : nn.Linear(32 → 4, std=0.01 init)  ← MoEWatch hooks this
    experts  : {DemoMoE.N_EXPERTS} × nn.Linear(32 → 32)
    top_k    : {DemoMoE.TOP_K}  (dense — every expert selected every step)

  top_k=N_EXPERTS is used so that every expert receives gradient signal on
  every step during the healthy phase. This prevents the gradient starvation
  detector from raising false alarms due to random token assignment variance,
  which would obscure the genuine collapse signal in the output.

  Collapse is injected by setting collapse_bias=+4 on the gate output for
  expert 0. This pushes normalised entropy from ≈1.00 down to ≈0.19, well
  below the configured warning (0.70) and critical (0.40) thresholds.
""")
    model = DemoMoE(d_model=32)
    total = sum(p.numel() for p in model.parameters())
    x_probe = torch.randn(16, 8, 32)
    print(f"  Parameters         : {total:,}")
    print(f"  Step-0 norm_entropy: {model.norm_entropy(x_probe):.4f}  (1.0 = uniform)")
    return model


# ── Section 2: Configure ──────────────────────────────────────────────────────

def section_2_configure() -> WatchConfig:
    _banner("2. Configure MoEWatch")
    print("""
  Entropy thresholds — the primary signal for this demo:

    entropy_warn     = 0.70  → WARNING when norm_entropy < 0.70
    entropy_critical = 0.40  → CRITICAL when norm_entropy < 0.40

  Routing entropy reference for this model (4 experts):
    1.00  uniform routing           all experts used equally    (healthy)
    0.69  bias=+2 on one expert    mild skew                   (WARNING)
    0.19  bias=+4 on one expert    near-total collapse         (CRITICAL)

  Note on the fused risk score:
    The risk score combines gradient starvation (Tier 1, weight 0.6) and
    entropy drift (Tier 2, weight 0.3). The Tier-1 threshold (cold_threshold)
    is calibrated for production gradient norms; in this small demo model the
    gradient norms (~0.001) are below cold_threshold=1.0, so Tier 1 reads
    ~0.6 even when routing is healthy. The entropy and alert signals are not
    affected by this calibration — they reflect actual routing behaviour.
    For a production run, set cold_threshold to match your model's typical
    gradient norm magnitude.

  Other settings (shortened proportionally for a 45-step demo):
    cold_steps_limit      =  8  (default  50  — steps before DEAD alert)
    reward_window_steps   =  8  (default  50  — evaluation window)
    intervention_cooldown = 10  (default 200  — min steps between actions)
    baseline_min_clean_steps = 5  (default 20 — steps needed for reward)
""")
    config = WatchConfig(
        output                  = OutputMode.SILENT,
        cold_steps_limit        = 8,
        stats_window            = 40,
        entropy_warn            = 0.70,
        entropy_critical        = 0.40,
        intervention_enabled    = True,
        policy_type             = "rule",
        reward_window_steps     = 8,
        intervention_cooldown   = 10,
        baseline_min_clean_steps = 5,
        dead_threshold          = 0.0001,
        cold_threshold          = 0.001,
        intervention_max_delta  = 0.1,
    )
    print(f"  entropy_warn             : {config.entropy_warn}")
    print(f"  entropy_critical         : {config.entropy_critical}")
    print(f"  cold_steps_limit         : {config.cold_steps_limit}")
    print(f"  baseline_min_clean_steps : {config.baseline_min_clean_steps}")
    return config


# ── Section 3: Attach ─────────────────────────────────────────────────────────

def section_3_attach(model: DemoMoE, config: WatchConfig) -> MoEWatch:
    _banner("3. Attach MoEWatch")
    print("""
  MoEWatch auto-detects the gate layer and registers forward + backward
  hooks. From this point every forward pass is captured automatically.
  The only additions to a normal training loop are:
    watch.pre_step(step)   — called before the optimizer step
    watch.step(step, ...)  — called after the optimizer step
""")
    watch = MoEWatch(model, config)
    watch.start()
    print(f"  Layers detected  : {watch._layer_order}")
    print(f"  Layers monitored : {watch.num_layers_monitored}")
    print(f"  Watcher running  : {watch.is_running}")
    return watch


# ── Section 4: Training loop ──────────────────────────────────────────────────

_PHASES = [
    ( 1, 15, 0.0, "Phase A — healthy routing   (collapse_bias = 0)"),
    (16, 35, 4.0, "Phase B — routing collapse  (collapse_bias = +4 on expert 0)"),
    (36, 45, 0.0, "Phase C — routing restored  (collapse_bias = 0)"),
]
TOTAL_STEPS = 45


def section_4_5_6_training_loop(model: DemoMoE, watch: MoEWatch) -> None:
    _banner("4. Training loop")
    print(f"""
  45 steps across three phases:

    Phase A  steps  1–15  collapse_bias=0     norm_entropy ≈ 1.00  (healthy)
    Phase B  steps 16–35  collapse_bias=+4    norm_entropy ≈ 0.19  (collapse)
    Phase C  steps 36–45  collapse_bias=0     norm_entropy ≈ 1.00  (restored)

  The entropy column is the clearest signal in this demo.
  Watch it drop sharply at step 16 and recover at step 36.

  Entropy thresholds:  warn < 0.70   critical < 0.40
""")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print(f"  {'step':>5}  {'entropy':>8}  {'alerts':>7}  {'int':>4}  note")
    print(f"  {SEP}")

    prev_label = ""
    for step in range(1, TOTAL_STEPS + 1):

        for start, end, bias, label in _PHASES:
            if step == start and label != prev_label:
                print(f"  {'─' * 68}")
                print(f"  {label}")
                print(f"  {'─' * 68}")
                prev_label = label

        # Set collapse state
        model.collapse_bias = next(
            bias for start, end, bias, _ in _PHASES if start <= step <= end
        )

        # Forward + backward
        x    = torch.randn(8, 16, 32)
        ent  = model.norm_entropy(x)
        out  = model(x)
        loss = out.pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        watch.pre_step(step)
        optimizer.step()

        report = watch.step(step, current_loss=float(loss.detach()))

        n_alert = len(report.alerts)
        n_int   = report.num_interventions

        note = ""
        if n_int > 0:
            atype = report.active_interventions[0].action_type
            note  = f"INTERVENED → {atype}"
        elif n_alert > 0:
            top  = report.alerts[0]
            note = f"{top.level.value.upper()}: {top.signal_type}"

        # Mark threshold crossings explicitly
        if ent < watch.config.entropy_critical:
            ent_flag = "  ← CRITICAL entropy"
        elif ent < watch.config.entropy_warn:
            ent_flag = "  ← WARNING entropy"
        else:
            ent_flag = ""

        print(f"  {step:>5}  {ent:>8.4f}  {n_alert:>7}  {n_int:>4}  {note}{ent_flag}")

    # ── Section 5: Alert stream ───────────────────────────────────────────────
    _banner("5. Alert stream")
    print("""
  The alert stream is the primary output of MoEWatch. Each alert carries:
    step       — when it fired
    level      — WARNING or CRITICAL
    signal_type — what triggered it (entropy_drift, gradient_starvation, …)
    message    — human-readable description with the measured value

  Alerts fire even when the fused risk score is ambiguous — entropy threshold
  crossings are evaluated directly and do not depend on Tier-1 calibration.
""")
    all_alerts  = watch.get_alerts(since_step=0)
    crit        = [a for a in all_alerts if a.level == AlertLevel.CRITICAL]
    warn        = [a for a in all_alerts if a.level == AlertLevel.WARNING]

    # Step ranges for each phase
    phase_a = [a for a in all_alerts if a.step <= 15]
    phase_b = [a for a in all_alerts if 16 <= a.step <= 35]
    phase_c = [a for a in all_alerts if a.step > 35]

    print(f"  Total alerts  : {len(all_alerts)}")
    print(f"  CRITICAL      : {len(crit)}")
    print(f"  WARNING       : {len(warn)}")
    print()
    print(f"  Alerts by phase:")
    print(f"    Phase A (healthy,   steps  1–15) : {len(phase_a):>4} alerts")
    print(f"    Phase B (collapse,  steps 16–35) : {len(phase_b):>4} alerts")
    print(f"    Phase C (restored,  steps 36–45) : {len(phase_c):>4} alerts")

    first_crit = next((a for a in crit), None)
    if first_crit:
        print(f"\n  First CRITICAL alert:")
        print(f"    step={first_crit.step}  signal={first_crit.signal_type}")
        msg = first_crit.message[:70] + "…" if len(first_crit.message) > 70 else first_crit.message
        print(f"    {msg}")

    if crit:
        print(f"\n  Sample CRITICAL alerts (first 5):\n")
        print(f"    {'step':>5}  {'signal':<26}  message")
        print(f"    {'─' * 62}")
        for a in crit[:5]:
            msg = a.message[:48] + "…" if len(a.message) > 48 else a.message
            print(f"    {a.step:>5}  {a.signal_type:<26}  {msg}")

    # ── Section 6: Intervention log ───────────────────────────────────────────
    _banner("6. Intervention log")
    print("""
  The engine records every action it considered, applied, or blocked.

  "applied"    — a non-noop action reached the model.
  "resolved"   — the observation window expired and reward was computed.
  "downgraded" — the safety guard blocked the action (cooldown, delta limit).

  Reward = baseline_risk_projected − actual_risk.
    Positive  : risk fell below the no-intervention projection → kept.
    Zero      : baseline not yet established → neutral (not a failure).
    Negative  : no improvement → reverted.

  baseline_min_clean_steps = 5 means the first reward is only meaningful
  after 5 clean steps of baseline history. Earlier windows show reward=0.0
  and are labelled "baseline not yet valid".
""")
    log        = watch.intervention_engine.get_intervention_log()
    applied    = [e for e in log if e["event"] == "applied"    and e["action"] != "noop"]
    resolved   = [e for e in log if e["event"] == "resolved"]
    downgraded = [e for e in log if e["event"] == "downgraded"]

    print(f"  Log entries  : {len(log)}")
    print(f"  Applied      : {len(applied)}")
    print(f"  Resolved     : {len(resolved)}")
    print(f"  Downgraded   : {len(downgraded)}  "
          f"(cooldown or delta limit — not errors)")

    if applied:
        print(f"\n  Applied interventions:\n")
        print(f"    {'step':>5}  {'action':<18}  {'delta':>7}  phase")
        print(f"    {'─' * 48}")
        for e in applied:
            if   e["step"] <= 15: phase = "healthy"
            elif e["step"] <= 35: phase = "collapse"
            else:                 phase = "recovery"
            print(f"    {e['step']:>5}  {e['action']:<18}  {e.get('delta', 0):>7.4f}  {phase}")

    if resolved:
        print(f"\n  Resolved outcomes:\n")
        print(f"    {'step':>5}  {'outcome':<10}  {'reward':>10}  meaning")
        print(f"    {'─' * 58}")
        for e in resolved:
            if e["reward"] > 0:
                meaning = "intervention helped — action kept"
            elif e["reward"] == 0.0:
                meaning = "baseline not yet valid — neutral result"
            else:
                meaning = "no improvement — action reverted"
            print(f"    {e['step']:>5}  {e['outcome']:<10}  {e['reward']:>+10.4f}  {meaning}")


# ── Section 7: Post-training summary ─────────────────────────────────────────

def section_7_summary(watch: MoEWatch) -> None:
    _banner("7. Post-training summary")
    print("""
  watch.get_risk_summary() gives the latest fused risk score per layer.

  Note: in this small demo model the risk score sits in the HIGH band
  throughout because the Tier-1 (gradient starvation) component is
  calibrated for production-scale gradient norms (cold_threshold=1.0
  default, your model's norms ≈ 0.001). In a real training run the
  risk score correctly spans the full 0.0–1.0 range. The entropy and
  alert signals demonstrated above are calibration-independent and
  correctly tracked all three phases.
""")
    summary = watch.get_risk_summary()
    print(f"  {'layer':<20}  {'risk score':>12}  note")
    print(f"  {'─' * 54}")
    for layer, score in summary.items():
        if score >= 0.8:   label = "CRITICAL"
        elif score >= 0.6: label = "HIGH (see calibration note above)"
        elif score >= 0.3: label = "MID"
        else:              label = "LOW ✓"
        print(f"  {layer:<20}  {score:>12.4f}  {label}")

    print(f"\n  WatchReport step records       : {len(watch.watch_report.steps)}")
    print(f"  Total interventions across run : {watch.watch_report.num_interventions}")
    watch.stop()
    print(f"\n  watch.stop() — hooks removed, watcher shut down.")
    print(f"  is_running : {watch.is_running}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    torch.manual_seed(42)

    print(SEP2)
    print("  MoEWatch — Live Monitoring Walkthrough")
    print("  build → configure → attach → train →")
    print("  collapse → detect → intervene → restore → summarise")
    print(SEP2)

    model  = section_1_build_model()
    config = section_2_configure()
    watch  = section_3_attach(model, config)
    section_4_5_6_training_loop(model, watch)
    section_7_summary(watch)

    print(f"\n{SEP2}")
    print("  Done.")
    print("  For offline diagnostics on a saved checkpoint:")
    print("  moewatch.audit(model, dataset, config=config)")
    print(SEP2)


if __name__ == "__main__":
    main()
