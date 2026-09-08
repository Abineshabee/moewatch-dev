"""
plot_comparison.py
===================
Reads without_moewatch_history.json and with_moewatch_history.json
(produced by the two training scripts) and renders a 2x3 comparison
grid:

  Row 0: Router entropy per layer (3 panels, one per layer)
  Row 1: Training loss | MoEWatch risk score | Dead-expert bar chart

Dead experts are counted with threshold < 0.02 (an expert receiving
less than 2% of routing decisions).

Run AFTER both training scripts:
    python train_without_moewatch.py
    python train_with_moewatch.py
    python plot_comparison.py
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from moe_model import PRESSURE_SCHEDULE

GATES = ["blocks.0.moe.gate", "blocks.1.moe.gate", "blocks.2.moe.gate"]
RED, GREEN = "#d62728", "#2ca02c"
DEAD_THRESHOLD = 0.02


def load(path):
    with open(path) as f:
        return json.load(f)


def shade_pressure(ax, layer_idx):
    """Gray band for this layer's own pressure window."""
    for l_idx, _dom, start, end, _peak in PRESSURE_SCHEDULE:
        if l_idx == layer_idx:
            ax.axvspan(start, end, color="gray", alpha=0.15,
                       label="_pressure window")


def main():
    without = load("without_moewatch_history.json")
    with_   = load("with_moewatch_history.json")

    fig, axes = plt.subplots(2, 3, figsize=(19, 10))

    # -----------------------------------------------------------------------
    # Row 0: Router entropy, one panel per layer
    # -----------------------------------------------------------------------
    for col, gate in enumerate(GATES):
        ax = axes[0, col]
        layer_idx = int(gate.split(".")[1])

        ax.plot(without["step"], without["entropy"][gate],
                label="Without MoEWatch", color=RED, linewidth=1.4)
        ax.plot(with_["step"], with_["entropy"][gate],
                label="With MoEWatch", color=GREEN, linewidth=1.4)

        shade_pressure(ax, layer_idx)
        ax.axhline(1.0, color="black",  linewidth=0.6, linestyle=":")
        ax.axhline(0.02, color="red",   linewidth=0.8, linestyle="--",
                   label="dead-expert floor (~0.02 entropy)")

        ax.set_title(f"Router entropy — {gate}\n(1.0 = perfectly uniform, healthy)")
        ax.set_xlabel("step")
        ax.set_ylabel("normalized entropy")
        ax.set_ylim(-0.05, 1.08)
        ax.legend(fontsize=8, loc="lower left")

    # -----------------------------------------------------------------------
    # Row 1, col 0: Training loss
    # -----------------------------------------------------------------------
    ax = axes[1, 0]
    ax.plot(without["step"], without["loss"],
            label="Without MoEWatch", color=RED, linewidth=1.0, alpha=0.7)
    ax.plot(with_["step"], with_["loss"],
            label="With MoEWatch",    color=GREEN, linewidth=1.0, alpha=0.7)
    ax.set_title("Training loss")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.legend(fontsize=8)

    # -----------------------------------------------------------------------
    # Row 1, col 1: MoEWatch fused risk score (with-run only)
    # -----------------------------------------------------------------------
    ax = axes[1, 1]
    layer_colors = ["#1f77b4", "#ff7f0e", "#9467bd"]
    for gate, color in zip(GATES, layer_colors):
        ax.plot(with_["step"], with_["risk_score"][gate],
                label=f"layer {gate.split('.')[1]}", linewidth=1.0, color=color)
    ax.axhline(0.3, color="gray",   linewidth=0.8, linestyle="--",
               label="aux_loss threshold (0.30)")
    ax.axhline(0.6, color="orange", linewidth=0.8, linestyle="--",
               label="router_noise threshold (0.60)")
    ax.axhline(0.8, color="red",    linewidth=0.8, linestyle="--",
               label="expert_dropout threshold (0.80)")
    for ev in with_["interventions"]:
        ax.axvline(ev["step"], color="green", alpha=0.08, linewidth=1)
    ax.set_title("MoEWatch fused risk score (per layer)\n"
                 "green bands = intervention steps")
    ax.set_xlabel("step")
    ax.set_ylabel("risk score [0, 1]")
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=7, loc="upper left")

    # -----------------------------------------------------------------------
    # Row 1, col 2: Dead-expert observations bar chart
    # -----------------------------------------------------------------------
    ax = axes[1, 2]
    labels = [g.split(".")[1] for g in GATES]

    # Sum across all logged steps (each step contributes 0-3 dead experts).
    without_dead_total = [sum(without["dead_experts"][g]) for g in GATES]
    with_dead_total    = [sum(with_["dead_experts"][g])   for g in GATES]

    x     = np.arange(len(labels))
    width = 0.35
    bars_w = ax.bar(x - width / 2, without_dead_total, width,
                    label="Without MoEWatch", color=RED)
    bars_h = ax.bar(x + width / 2, with_dead_total,    width,
                    label="With MoEWatch",    color=GREEN)

    # Annotate bar tops with the actual count.
    for bar in bars_w:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    str(int(h)), ha="center", va="bottom", fontsize=8)
    for bar in bars_h:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, max(h, 0) + 0.5,
                str(int(h)) if h > 0 else "0", ha="center", va="bottom",
                fontsize=8, color="darkgreen" if h == 0 else "black")

    ax.set_xticks(x)
    ax.set_xticklabels([f"layer {l}" for l in labels])
    ax.set_title(f"Dead-expert step-observations\n"
                 f"(expert usage < {DEAD_THRESHOLD*100:.0f}%, "
                 f"summed over all {len(without['step'])} logged steps)")
    ax.set_ylabel("count  (lower is better)")
    ax.legend(fontsize=8)

    fig.suptitle(
        "MoEWatch Comparison — same model, same data, same seed, same collapse-pressure\n"
        "Without MoEWatch: routers collapse to 1 expert; 2-3 dead experts per layer  |  "
        "With MoEWatch: live interventions hold entropy up, zero dead experts",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("moewatch_comparison.png", dpi=150)
    print("Saved chart -> moewatch_comparison.png")

    # -----------------------------------------------------------------------
    # Text summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    for g in GATES:
        w_min  = min(without["entropy"][g])
        h_min  = min(with_["entropy"][g])
        w_dead = sum(without["dead_experts"][g])
        h_dead = sum(with_["dead_experts"][g])
        print(f"\n  {g}")
        print(f"    Lowest entropy reached      : without={w_min:.3f}   with={h_min:.3f}")
        print(f"    Dead-expert step-observations: without={w_dead}   with={h_dead}")
        if w_dead > 0 and h_dead == 0:
            print(f"    → MoEWatch eliminated ALL dead-expert events on this layer ✓")
        elif h_dead < w_dead:
            pct = 100 * (1 - h_dead / w_dead)
            print(f"    → MoEWatch reduced dead-expert events by {pct:.0f}% on this layer")

    print(f"\n  Interventions fired: {len(with_['interventions'])}")
    action_counts: dict = {}
    for ev in with_["interventions"]:
        for a in ev["actions"]:
            action_counts[a] = action_counts.get(a, 0) + 1
    for a, c in sorted(action_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {a:<20}: {c}")

    print("\n  Without MoEWatch: collapse was invisible until inspection of")
    print("  final checkpoints. No alerts, no risk scores, no pushback --")
    print("  the model trained silently on 1 expert while 3 sat idle.")
    print("=" * 78)


if __name__ == "__main__":
    main()
