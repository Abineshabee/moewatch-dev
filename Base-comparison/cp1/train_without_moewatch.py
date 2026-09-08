"""
train_without_moewatch.py
==========================
Trains the tiny custom MoE language model with plain PyTorch --
no monitoring, no collapse detection, no intervention whatsoever.

A deterministic collapse-pressure schedule (defined in moe_model.py)
ramps the router bias up to peak=2.0-2.2 for all three layers.  At
that bias level a gate pushes ~98%+ of tokens to one expert, leaving
the other three with < 2% usage each (confirmed dead by our threshold).

Because there is no AuxLoss / RouterNoise / ExpertDropout to push back,
collapse is permanent: dead experts remain dead after pressure ends and
entropy stays low for the rest of training.

This is the "before" (broken) side of the MoEWatch comparison.

Run:
    python train_without_moewatch.py
"""

import json
import torch
import torch.nn.functional as F

from moe_model import (
    TinyMoELM, build_markov_transition, sample_batch,
    SEED, BATCH_SIZE, SEQ_LEN, TOTAL_STEPS, LR, NUM_EXPERTS,
    EVAL_BATCH_SIZE,
)

# Dead-expert threshold: expert is "dead" if it receives < 2% of tokens.
# At peak bias=2.0 the without-MoEWatch run drives all non-dominant experts
# below 1%, so this threshold cleanly separates dead from alive.
DEAD_THRESHOLD = 0.02

LOG_EVERY = 1


def main():
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)

    trans = build_markov_transition(SEED)
    model = TinyMoELM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)

    # Larger eval batch for reliable dead-expert counts (2048 routing decisions).
    eval_batch  = sample_batch(trans, batch_size=EVAL_BATCH_SIZE, seq_len=SEQ_LEN,
                               generator=torch.Generator().manual_seed(999))
    eval_inputs = eval_batch[:, :-1]

    history = {
        "step":         [],
        "loss":         [],
        "entropy":      {n: [] for n in model.gate_names()},
        "dead_experts": {n: [] for n in model.gate_names()},
    }

    print("=" * 78)
    print("  TRAINING WITHOUT MOEWATCH  (plain PyTorch, zero monitoring)")
    print("=" * 78)
    header = (f"  {'step':>5}  {'loss':>8}  {'phase':<18}"
              + "  ".join(f"{n.split('.')[1]:>8}-ent" for n in model.gate_names())
              + "  " + "  ".join(f"L{i}-dead" for i in range(3)))
    print(header)

    for step in range(1, TOTAL_STEPS + 1):
        phase = model.apply_pressure_schedule(step)

        batch = sample_batch(trans, BATCH_SIZE, SEQ_LEN, gen)
        inputs, targets = batch[:, :-1], batch[:, 1:]

        optimizer.zero_grad()
        logits = model(inputs)
        loss   = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()
        optimizer.step()

        if step % LOG_EVERY == 0 or step == 1:
            usage, entropy = model.expert_usage(eval_inputs)
            history["step"].append(step)
            history["loss"].append(loss.item())

            row  = [f"{step:>5}", f"{loss.item():>8.4f}", f"{phase:<18}"]
            dead_cols = []
            for name in model.gate_names():
                ent  = entropy[name]
                dead = sum(1 for f in usage[name] if f < DEAD_THRESHOLD)
                history["entropy"][name].append(ent)
                history["dead_experts"][name].append(dead)
                row.append(f"{ent:>11.3f}")
                dead_cols.append(f"{dead:>6d}")
            row.extend(dead_cols)
            print("  " + "  ".join(row))

    # --- Final report ---
    usage, entropy = model.expert_usage(eval_inputs)
    print(f"\n  Final expert usage (healthy uniform = {1/NUM_EXPERTS:.3f} each):")
    for name in model.gate_names():
        frac_str = ", ".join(f"{f:.3f}" for f in usage[name])
        dead = sum(1 for f in usage[name] if f < DEAD_THRESHOLD)
        print(f"    {name:<22} entropy={entropy[name]:.3f}  dead={dead}  "
              f"usage=[{frac_str}]")

    with open("without_moewatch_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("\n  Saved -> without_moewatch_history.json")


if __name__ == "__main__":
    main()
