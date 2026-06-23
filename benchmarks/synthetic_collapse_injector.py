# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# benchmarks/synthetic_collapse_injector.py
# =============================================================================
#
# Project      : MoEWatch
# Version      : v0.2.0
# Description  : Synthetic collapse injection framework. Programmatically
#                forces routing collapse at a known step by biasing router
#                weights. Used to create ground-truth collapse events for
#                lead-time and IES benchmarks.
#
#                Supports three collapse modes:
#                  - SUDDEN   — full bias applied instantly at inject_step
#                  - GRADUAL  — bias ramps linearly over ramp_steps
#                  - CASCADE  — collapse spreads layer by layer with a
#                               configurable inter-layer delay
#
#                Critical for reproducible benchmarks without waiting for
#                natural collapse events.
#
# Author       : Abinesh N (@Abineshabee)
# Repository   : https://github.com/Abineshabee/MoEWatch
# License      : Apache 2.0
#
# Public API
# ----------
#   CollapseProfile   — dataclass describing a collapse scenario
#   CollapseInjector  — attaches to a model and fires on .step(global_step)
#
# Usage
# -----
#   from benchmarks.synthetic_collapse_injector import (
#       CollapseInjector, CollapseProfile, CollapseMode,
#   )
#
#   profile = CollapseProfile(
#       mode=CollapseMode.GRADUAL,
#       inject_step=80,
#       target_layers=[0],
#       strength=12.0,
#       ramp_steps=20,
#       dominant_expert=0,
#   )
#   injector = CollapseInjector(model, profile)
#
#   for step in range(1, 201):
#       injector.step(step)          # fires automatically at inject_step
#       ...                          # forward / backward / optimizer
#
#   injector.reset()                 # undo all injected biases
#
# =============================================================================

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class CollapseMode(str, Enum):
    """Collapse injection mode.

    Attributes
    ----------
    SUDDEN:
        Full ``strength`` bias is added to the dominant expert's gate logit
        in a single step at ``inject_step``. Models an abrupt routing
        collapse with no warm-up.
    GRADUAL:
        Bias ramps linearly from 0 to ``strength`` over ``ramp_steps``
        steps starting at ``inject_step``. Models a progressive routing
        drift like those observed in long fine-tuning runs.
    CASCADE:
        Collapse is injected one layer at a time in the order given by
        ``target_layers``. Each subsequent layer is triggered
        ``cascade_delay_steps`` steps after the previous one fires at
        full ``strength``. Models cross-layer collapse propagation.
    """

    SUDDEN  = "sudden"
    GRADUAL = "gradual"
    CASCADE = "cascade"


# ---------------------------------------------------------------------------
# CollapseProfile
# ---------------------------------------------------------------------------


@dataclass
class CollapseProfile:
    """Describes a synthetic collapse scenario.

    Parameters
    ----------
    mode : CollapseMode
        Injection mode — SUDDEN, GRADUAL, or CASCADE. Default: SUDDEN.
    inject_step : int
        Global training step at which collapse injection begins.
    target_layers : list[int]
        Layer indices within ``model.layers`` to collapse. For SUDDEN and
        GRADUAL, all listed layers are collapsed simultaneously. For CASCADE,
        layers are collapsed in list order with ``cascade_delay_steps``
        between each. Default: [0].
    strength : float
        Maximum logit bias added to the dominant expert's gate output.
        Higher values produce more total routing collapse. Default: 12.0.
    dominant_expert : int
        Index of the expert that will receive the positive logit bias (and
        thus monopolise routing after injection). Default: 0.
    ramp_steps : int
        For GRADUAL mode — number of steps over which the bias ramps from
        0 to ``strength``. Ignored for SUDDEN and CASCADE. Default: 20.
    cascade_delay_steps : int
        For CASCADE mode — number of steps between successive layer
        collapses. Layer 0 fires at ``inject_step``, layer 1 fires at
        ``inject_step + cascade_delay_steps``, and so on. Ignored for
        SUDDEN and GRADUAL. Default: 10.

    Examples
    --------
    >>> p = CollapseProfile(mode=CollapseMode.SUDDEN, inject_step=80)
    >>> p.mode
    <CollapseMode.SUDDEN: 'sudden'>
    >>> p.inject_step
    80
    """

    mode:                 CollapseMode = CollapseMode.SUDDEN
    inject_step:          int          = 80
    target_layers:        List[int]    = field(default_factory=lambda: [0])
    strength:             float        = 12.0
    dominant_expert:      int          = 0
    ramp_steps:           int          = 20
    cascade_delay_steps:  int          = 10

    def __post_init__(self) -> None:
        if self.inject_step < 1:
            raise ValueError(
                f"CollapseProfile.inject_step must be >= 1, got {self.inject_step}."
            )
        if not self.target_layers:
            raise ValueError("CollapseProfile.target_layers must be non-empty.")
        if self.strength <= 0.0:
            raise ValueError(
                f"CollapseProfile.strength must be positive, got {self.strength}."
            )
        if self.dominant_expert < 0:
            raise ValueError(
                f"CollapseProfile.dominant_expert must be >= 0, "
                f"got {self.dominant_expert}."
            )
        if self.ramp_steps < 1:
            raise ValueError(
                f"CollapseProfile.ramp_steps must be >= 1, got {self.ramp_steps}."
            )
        if self.cascade_delay_steps < 1:
            raise ValueError(
                f"CollapseProfile.cascade_delay_steps must be >= 1, "
                f"got {self.cascade_delay_steps}."
            )

    # Convenience constructors --------------------------------------------------

    @classmethod
    def sudden(
        cls,
        inject_step: int,
        target_layers: Optional[List[int]] = None,
        strength: float = 12.0,
        dominant_expert: int = 0,
    ) -> "CollapseProfile":
        """Return a SUDDEN profile with sensible defaults."""
        return cls(
            mode=CollapseMode.SUDDEN,
            inject_step=inject_step,
            target_layers=target_layers or [0],
            strength=strength,
            dominant_expert=dominant_expert,
        )

    @classmethod
    def gradual(
        cls,
        inject_step: int,
        ramp_steps: int = 20,
        target_layers: Optional[List[int]] = None,
        strength: float = 12.0,
        dominant_expert: int = 0,
    ) -> "CollapseProfile":
        """Return a GRADUAL profile with sensible defaults."""
        return cls(
            mode=CollapseMode.GRADUAL,
            inject_step=inject_step,
            target_layers=target_layers or [0],
            strength=strength,
            dominant_expert=dominant_expert,
            ramp_steps=ramp_steps,
        )

    @classmethod
    def cascade(
        cls,
        inject_step: int,
        target_layers: Optional[List[int]] = None,
        cascade_delay_steps: int = 10,
        strength: float = 12.0,
        dominant_expert: int = 0,
    ) -> "CollapseProfile":
        """Return a CASCADE profile with sensible defaults."""
        layers = target_layers if target_layers is not None else [0, 1, 2]
        return cls(
            mode=CollapseMode.CASCADE,
            inject_step=inject_step,
            target_layers=layers,
            strength=strength,
            dominant_expert=dominant_expert,
            cascade_delay_steps=cascade_delay_steps,
        )

    def describe(self) -> str:
        """Return a one-line human-readable description of this profile."""
        if self.mode == CollapseMode.SUDDEN:
            return (
                f"SUDDEN collapse at step {self.inject_step} "
                f"(layers={self.target_layers}, expert={self.dominant_expert}, "
                f"strength={self.strength})"
            )
        if self.mode == CollapseMode.GRADUAL:
            return (
                f"GRADUAL collapse starting at step {self.inject_step} "
                f"over {self.ramp_steps} steps "
                f"(layers={self.target_layers}, expert={self.dominant_expert}, "
                f"strength={self.strength})"
            )
        last_step = self.inject_step + (len(self.target_layers) - 1) * self.cascade_delay_steps
        return (
            f"CASCADE collapse starting at step {self.inject_step}, "
            f"ending at step {last_step} "
            f"(layers={self.target_layers}, delay={self.cascade_delay_steps}, "
            f"expert={self.dominant_expert}, strength={self.strength})"
        )


# ---------------------------------------------------------------------------
# Layer resolver — works with both benchmark FakeQwen3MoEModel and test models
# ---------------------------------------------------------------------------


def _find_gate_module(model: nn.Module, layer_idx: int) -> Optional[nn.Module]:
    """Return the gate Linear for layer ``layer_idx``, or None if not found.

    Resolution order (most to least specific):
    1. ``model.layers[layer_idx].mlp.gate``   — benchmark _FakeQwen3MoEModel
    2. ``model.layers[layer_idx].gate``        — test FakeMoEBlock
    3. Scan ``model.named_modules()`` for any Linear whose path contains
       ``f"layers.{layer_idx}"`` and ends with ``".gate"``

    The gate is returned as an ``nn.Module`` (typically ``nn.Linear``).
    The injector writes only to a ``collapse_bias`` buffer (if present)
    or to ``gate.bias`` directly — it never touches ``gate.weight``.
    """
    layers = getattr(model, "layers", None)
    if layers is None or layer_idx >= len(layers):
        return None

    layer = layers[layer_idx]

    # Path 1: benchmark model — layers[i].mlp.gate
    mlp = getattr(layer, "mlp", None)
    if mlp is not None:
        gate = getattr(mlp, "gate", None)
        if gate is not None:
            return gate

    # Path 2: test model — layers[i].gate
    gate = getattr(layer, "gate", None)
    if gate is not None:
        return gate

    # Path 3: generic scan
    prefix = f"layers.{layer_idx}."
    for name, module in model.named_modules():
        if name.startswith(prefix) and name.endswith(".gate"):
            return module

    return None


def _find_collapse_bias_buffer(
    model: nn.Module, layer_idx: int
) -> Optional[torch.Tensor]:
    """Return the ``collapse_bias`` buffer registered on _MoEBlock, if present."""
    layers = getattr(model, "layers", None)
    if layers is None or layer_idx >= len(layers):
        return None
    layer = layers[layer_idx]
    mlp   = getattr(layer, "mlp", None)
    if mlp is None:
        return None
    buf = getattr(mlp, "collapse_bias", None)
    return buf if isinstance(buf, torch.Tensor) else None


def _n_experts_for_layer(model: nn.Module, layer_idx: int) -> int:
    """Return the expert count for layer ``layer_idx``."""
    gate = _find_gate_module(model, layer_idx)
    if gate is not None:
        out = getattr(gate, "out_features", None)
        if isinstance(out, int):
            return out
    return 0


# ---------------------------------------------------------------------------
# CollapseInjector
# ---------------------------------------------------------------------------


class CollapseInjector:
    """Programmatically injects routing collapse into a live model.

    Attaches to any ``torch.nn.Module`` whose ``model.layers`` list contains
    MoE blocks with a detectable gate linear (``layers[i].mlp.gate`` for the
    benchmark model, ``layers[i].gate`` for the test model). Fires collapse
    injection automatically when ``step()`` is called with the matching
    global step.

    The injector prefers writing to ``_MoEBlock.collapse_bias`` (a registered
    buffer on the benchmark model) when present; otherwise it creates and
    adds a ``gate.bias`` parameter. This keeps the injection:

    - **Reversible** — ``reset()`` zeros the collapse buffer or removes the
      added bias, restoring the original routing distribution exactly.
    - **Non-destructive** — gate weights are never modified.
    - **Gradient-safe** — the bias is applied as an additive constant to
      the gate's output *before* softmax, which is differentiable and
      interacts correctly with MoEWatch's backward hooks.

    Parameters
    ----------
    model : torch.nn.Module
        The model to inject collapse into. Must have a ``layers`` attribute.
    profile : CollapseProfile
        Collapse scenario specification.

    Attributes
    ----------
    model   : nn.Module        — target model
    profile : CollapseProfile  — active scenario
    active  : bool             — True once injection has started
    events  : list[dict]       — log of every injection event for audit/export
    """

    def __init__(self, model: nn.Module, profile: CollapseProfile) -> None:
        self.model   = model
        self.profile = profile
        self.active  = False
        self.events: List[Dict] = []

        # Per-layer injection state
        # _bias_current[layer_idx] = current injected bias value (float)
        self._bias_current: Dict[int, float] = {
            li: 0.0 for li in profile.target_layers
        }
        # _bias_handles[layer_idx] = True if we added a new bias param
        self._added_bias_param: Dict[int, bool] = {}

        # Validate all target layers are reachable
        for li in profile.target_layers:
            gate = _find_gate_module(model, li)
            if gate is None:
                raise ValueError(
                    f"[CollapseInjector] Cannot find a gate module for "
                    f"layer index {li}. Check that model.layers[{li}] has "
                    f"a '.mlp.gate' or '.gate' attribute."
                )
            n_exp = _n_experts_for_layer(model, li)
            if profile.dominant_expert >= n_exp:
                raise ValueError(
                    f"[CollapseInjector] dominant_expert={profile.dominant_expert} "
                    f"is out of range for layer {li} which has {n_exp} experts."
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, global_step: int) -> bool:
        """Process a training step and fire injection if warranted.

        Must be called once per training step, **before** the forward pass,
        so that the injected bias affects the current step's routing.

        Parameters
        ----------
        global_step : int
            Current global training step (1-indexed).

        Returns
        -------
        bool
            ``True`` if any injection was applied or updated this step,
            ``False`` otherwise.
        """
        mode = self.profile.mode

        if mode == CollapseMode.SUDDEN:
            return self._step_sudden(global_step)
        if mode == CollapseMode.GRADUAL:
            return self._step_gradual(global_step)
        if mode == CollapseMode.CASCADE:
            return self._step_cascade(global_step)
        return False

    def reset(self) -> None:
        """Remove all injected biases and restore original routing.

        Safe to call at any time — before, during, or after injection.
        After ``reset()``, the injector can be reused for a new run.
        """
        for li in self.profile.target_layers:
            self._set_bias(li, 0.0)
            if self._added_bias_param.get(li):
                gate = _find_gate_module(self.model, li)
                if gate is not None and hasattr(gate, "bias") and gate.bias is not None:
                    gate.bias = None   # remove the parameter we added
                self._added_bias_param[li] = False
            self._bias_current[li] = 0.0

        self.active = False
        self.events.append({"event": "reset"})

    def current_biases(self) -> Dict[int, float]:
        """Return the current injected bias per layer index."""
        return dict(self._bias_current)

    def summary(self) -> str:
        """Return a short human-readable summary of injection state."""
        lines = [
            f"CollapseInjector — {self.profile.describe()}",
            f"  active : {self.active}",
        ]
        for li, bias in self._bias_current.items():
            lines.append(f"  layer {li:>2d} : collapse_bias[{self.profile.dominant_expert}] = {bias:.4f}")
        lines.append(f"  events : {len(self.events)}")
        return "\n".join(lines)

    def export_events(self) -> List[Dict]:
        """Return a copy of the injection event log (JSON-safe)."""
        return [dict(e) for e in self.events]

    # ------------------------------------------------------------------
    # Mode implementations
    # ------------------------------------------------------------------

    def _step_sudden(self, step: int) -> bool:
        """Apply full bias instantly at inject_step."""
        if step != self.profile.inject_step:
            return False
        self.active = True
        for li in self.profile.target_layers:
            self._set_bias(li, self.profile.strength)
        self.events.append({
            "event":   "sudden",
            "step":    step,
            "layers":  list(self.profile.target_layers),
            "bias":    self.profile.strength,
        })
        return True

    def _step_gradual(self, step: int) -> bool:
        """Ramp bias linearly from 0 → strength over ramp_steps."""
        s0  = self.profile.inject_step
        s1  = s0 + self.profile.ramp_steps - 1

        if step < s0 or step > s1:
            return False

        self.active = True
        # Linear ramp: fraction completes at s1
        frac = (step - s0) / max(self.profile.ramp_steps - 1, 1)
        bias = self.profile.strength * frac

        for li in self.profile.target_layers:
            self._set_bias(li, bias)

        self.events.append({
            "event":    "gradual",
            "step":     step,
            "fraction": round(frac, 4),
            "bias":     round(bias, 4),
            "layers":   list(self.profile.target_layers),
        })
        return True

    def _step_cascade(self, step: int) -> bool:
        """Collapse layers one by one with cascade_delay_steps between each."""
        fired = False
        for order, li in enumerate(self.profile.target_layers):
            trigger = self.profile.inject_step + order * self.profile.cascade_delay_steps
            if step == trigger:
                self.active = True
                self._set_bias(li, self.profile.strength)
                self.events.append({
                    "event":       "cascade",
                    "step":        step,
                    "layer":       li,
                    "layer_order": order,
                    "bias":        self.profile.strength,
                })
                fired = True
        return fired

    # ------------------------------------------------------------------
    # Bias write — prefers collapse_bias buffer, falls back to gate.bias
    # ------------------------------------------------------------------

    def _set_bias(self, layer_idx: int, value: float) -> None:
        """Write ``value`` to the dominant expert slot for ``layer_idx``.

        Writes to:
        1. ``model.layers[layer_idx].mlp.collapse_bias[dominant_expert]``
           if the buffer exists (benchmark _FakeQwen3MoEModel).
        2. ``gate.bias[dominant_expert]`` otherwise, creating the bias
           parameter if it did not already exist (test FakeMoEBlock).
        """
        dominant = self.profile.dominant_expert

        # --- Path 1: collapse_bias buffer (benchmark model) ---
        buf = _find_collapse_bias_buffer(self.model, layer_idx)
        if buf is not None:
            with torch.no_grad():
                buf[dominant] = value
            self._bias_current[layer_idx] = value
            return

        # --- Path 2: gate.bias parameter ---
        gate = _find_gate_module(self.model, layer_idx)
        if gate is None:
            return

        n_experts = getattr(gate, "out_features", 0)

        # Create bias if absent
        if not hasattr(gate, "bias") or gate.bias is None:
            with torch.no_grad():
                gate.bias = nn.Parameter(
                    torch.zeros(n_experts, device=next(gate.parameters()).device)
                )
            self._added_bias_param[layer_idx] = True

        with torch.no_grad():
            gate.bias[dominant] = value

        self._bias_current[layer_idx] = value


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------


def make_sudden_injector(
    model: nn.Module,
    inject_step: int,
    target_layers: Optional[List[int]] = None,
    strength: float = 12.0,
    dominant_expert: int = 0,
) -> CollapseInjector:
    """Return a CollapseInjector configured for sudden collapse.

    Parameters
    ----------
    model          : model to inject into
    inject_step    : step at which collapse fires
    target_layers  : layer indices (default: [0])
    strength       : logit bias magnitude (default: 12.0)
    dominant_expert: expert to bias toward (default: 0)
    """
    return CollapseInjector(
        model,
        CollapseProfile.sudden(
            inject_step=inject_step,
            target_layers=target_layers,
            strength=strength,
            dominant_expert=dominant_expert,
        ),
    )


def make_gradual_injector(
    model: nn.Module,
    inject_step: int,
    ramp_steps: int = 20,
    target_layers: Optional[List[int]] = None,
    strength: float = 12.0,
    dominant_expert: int = 0,
) -> CollapseInjector:
    """Return a CollapseInjector configured for gradual collapse.

    Parameters
    ----------
    model          : model to inject into
    inject_step    : step at which ramp begins
    ramp_steps     : steps to reach full strength (default: 20)
    target_layers  : layer indices (default: [0])
    strength       : final logit bias magnitude (default: 12.0)
    dominant_expert: expert to bias toward (default: 0)
    """
    return CollapseInjector(
        model,
        CollapseProfile.gradual(
            inject_step=inject_step,
            ramp_steps=ramp_steps,
            target_layers=target_layers,
            strength=strength,
            dominant_expert=dominant_expert,
        ),
    )


def make_cascade_injector(
    model: nn.Module,
    inject_step: int,
    target_layers: Optional[List[int]] = None,
    cascade_delay_steps: int = 10,
    strength: float = 12.0,
    dominant_expert: int = 0,
) -> CollapseInjector:
    """Return a CollapseInjector configured for cascade collapse.

    Parameters
    ----------
    model               : model to inject into
    inject_step         : step at which first layer collapses
    target_layers       : layer indices in cascade order (default: [0, 1, 2])
    cascade_delay_steps : steps between successive layer collapses (default: 10)
    strength            : logit bias magnitude per layer (default: 12.0)
    dominant_expert     : expert to bias toward (default: 0)
    """
    return CollapseInjector(
        model,
        CollapseProfile.cascade(
            inject_step=inject_step,
            target_layers=target_layers,
            cascade_delay_steps=cascade_delay_steps,
            strength=strength,
            dominant_expert=dominant_expert,
        ),
    )


# ---------------------------------------------------------------------------
# Self-test / demo  (python benchmarks/synthetic_collapse_injector.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import torch.nn.functional as F

    print("=" * 64)
    print("  MoEWatch v0.2.0 — CollapseInjector self-test")
    print("=" * 64)

    # --- Minimal inline model so the file is self-contained ---
    _H, _E, _K = 32, 8, 2

    class _Gate(nn.Linear):
        pass

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate    = _Gate(_H, _E, bias=False)
            self.experts = nn.ModuleList([nn.Linear(_H, _H, bias=False) for _ in range(_E)])
            self.register_buffer("collapse_bias", torch.zeros(_E))

        def forward(self, x):
            logits = self.gate(x) + self.collapse_bias
            probs  = torch.softmax(logits, -1)
            return x, probs.detach().mean(0)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                type("Layer", (nn.Module,), {
                    "__init__": lambda self: (super(type(self), self).__init__() or setattr(self, "mlp", _Block())),
                    "forward":  lambda self, x: self.mlp(x),
                })() for _ in range(4)
            ])

        def forward(self, x):
            probs_all = []
            for layer in self.layers:
                x, p = layer(x)
                probs_all.append(p)
            return x, probs_all

    # ---------------------------------------------------------------
    def _entropy(probs: torch.Tensor) -> float:
        p = probs.clamp(1e-9)
        return -(p * p.log()).sum().item() / math.log(len(probs))

    def _run_demo(label: str, injector: CollapseInjector, n_steps: int = 60) -> None:
        print(f"\n  [{label}]")
        print(f"  {injector.profile.describe()}")
        print(f"  {'Step':>5}  {'L0 bias':>9}  {'L0 entropy':>11}  {'Event'}")
        print(f"  {'-'*52}")
        x = torch.randn(4, _H)
        for step in range(1, n_steps + 1):
            fired = injector.step(step)
            _, probs_all = injector.model(x)
            bias  = injector.current_biases().get(0, 0.0)
            ent   = _entropy(probs_all[0])
            evt   = "← COLLAPSE" if fired else ""
            if fired or step % 10 == 0:
                print(f"  {step:>5}  {bias:>9.3f}  {ent:>11.4f}  {evt}")
        injector.reset()
        print(f"  After reset — L0 bias: {injector.current_biases().get(0, 0.0):.3f}")

    # --- SUDDEN ---
    m = _Model()
    _run_demo("SUDDEN", make_sudden_injector(m, inject_step=20, target_layers=[0], strength=12.0))

    # --- GRADUAL ---
    m = _Model()
    _run_demo("GRADUAL", make_gradual_injector(m, inject_step=20, ramp_steps=15, target_layers=[0], strength=12.0))

    # --- CASCADE ---
    m = _Model()
    _run_demo("CASCADE", make_cascade_injector(m, inject_step=20, target_layers=[0, 1, 2], cascade_delay_steps=8, strength=12.0), n_steps=60)

    print("\n" + "=" * 64)
    print("  Self-test complete.")
    print("=" * 64)
