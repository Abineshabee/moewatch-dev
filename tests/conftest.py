# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# tests/conftest.py
# =============================================================================
#
# Project      : MoEWatch
# Version      : v0.2.0
# Description  : Shared pytest fixtures for the full MoEWatch test suite.
#
#                Provides reusable model factories, event builders,
#                pre-populated StatCollector instances, and WatchConfig
#                variants covering the four test modules:
#
#                  test_detection.py    — detect_router_modules()
#                  test_hooks.py        — HookManager attach/detach
#                  test_collector.py    — RingBuffer & StatCollector
#                  test_entropy.py      — EntropyAnalyzer
#                  (+ future modules share the same fixtures)
#
#                All heavy neural-network fixtures are parameterised with
#                small expert counts so that the suite runs entirely on CPU
#                in a few seconds without GPU access.
#
# Author       : Abinesh N (@Abineshabee)
# Repository   : https://github.com/Abineshabee/MoEWatch
# License      : Apache 2.0
#
# =============================================================================

from __future__ import annotations

import math
import time
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# MoEWatch imports
# ---------------------------------------------------------------------------
from moewatch.config import AlertLevel, OutputMode, WatchConfig
from moewatch.collector.ring_buffer import RingBuffer
from moewatch.collector.stat_collector import GradientStats, LayerStats, StatCollector
from moewatch.hooks.gradient_hook import GradientEvent
from moewatch.hooks.router_hook import RoutingEvent


# ===========================================================================
# ── Section 1: WatchConfig fixtures ────────────────────────────────────────
# ===========================================================================


@pytest.fixture(scope="session")
def default_config() -> WatchConfig:
    """Default WatchConfig with SILENT output (no terminal noise in CI)."""
    return WatchConfig(output=OutputMode.SILENT)


@pytest.fixture(scope="session")
def strict_config() -> WatchConfig:
    """WatchConfig with tightened thresholds to force alert generation in tests."""
    return WatchConfig(
        output=OutputMode.SILENT,
        dead_threshold=0.005,
        cold_threshold=0.02,
        cold_steps_limit=3,
        entropy_warn=0.6,
        entropy_critical=0.4,
        entropy_drop_warn=0.05,
        load_imbalance_warn=2.0,
        load_imbalance_error=3.5,
        log_every=1,
        sample_every=1,
    )


@pytest.fixture(scope="session")
def lenient_config() -> WatchConfig:
    """WatchConfig with relaxed thresholds — no alerts should fire on random data."""
    return WatchConfig(
        output=OutputMode.SILENT,
        dead_threshold=1e-6,
        cold_threshold=1e-5,
        cold_steps_limit=200,
        entropy_warn=0.01,
        entropy_critical=0.005,
        load_imbalance_warn=100.0,
        load_imbalance_error=200.0,
        log_every=1,
        sample_every=1,
    )


@pytest.fixture
def config(default_config: WatchConfig) -> WatchConfig:
    """Function-scoped alias for default_config — convenient for most tests."""
    return default_config


# ===========================================================================
# ── Section 2: Synthetic MoE model factories ────────────────────────────────
# ===========================================================================
#
# These factories produce minimal torch.nn.Module trees that mimic the
# structure of real MoE architectures (Mixtral / Qwen3-MoE / DeepSeek-MoE)
# at a fraction of the parameter count.  The goal is to exercise the
# detection and hooking machinery without loading real pretrained weights.
#
# Architecture sketch (for n_layers=2, n_experts=4, hidden=32):
#
#   FakeMoEModel
#   └── layers : nn.ModuleList
#       ├── [0] FakeMoEBlock
#       │   ├── gate  : nn.Linear(hidden, n_experts)   ← router detected here
#       │   └── experts : nn.ModuleList
#       │       ├── [0] nn.Linear(hidden, hidden)
#       │       ├── [1] nn.Linear(hidden, hidden)
#       │       ├── [2] nn.Linear(hidden, hidden)
#       │       └── [3] nn.Linear(hidden, hidden)
#       └── [1] FakeMoEBlock  (same structure)
#
# ===========================================================================


class FakeMoEBlock(nn.Module):
    """Minimal MoE block: linear gate + n_experts linear expert modules."""

    def __init__(self, hidden: int, n_experts: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden, n_experts, bias=False)
        self.experts = nn.ModuleList(
            [nn.Linear(hidden, hidden, bias=False) for _ in range(n_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gate(x)  # [batch, n_experts]
        probs = torch.softmax(logits, dim=-1)
        # Weighted sum of expert outputs (simplified — no top-k routing)
        out = torch.stack(
            [expert(x) for expert in self.experts], dim=-1
        )  # [batch, hidden, n_experts]
        return (out * probs.unsqueeze(1)).sum(-1)  # [batch, hidden]


class FakeMoEModel(nn.Module):
    """Stacked FakeMoEBlocks that detection and hook tests can use directly."""

    def __init__(
        self,
        n_layers: int = 2,
        n_experts: int = 4,
        hidden: int = 32,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [FakeMoEBlock(hidden, n_experts) for _ in range(n_layers)]
        )
        self.n_experts = n_experts
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class FakeNonMoEModel(nn.Module):
    """Plain MLP with no MoE structure — detection should find nothing."""

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class FakeModelWithManualRouter(nn.Module):
    """Model whose router is named 'moe_router' to match override path."""

    def __init__(self, hidden: int = 16, n_experts: int = 8) -> None:
        super().__init__()
        self.moe_router = nn.Linear(hidden, n_experts, bias=False)
        self.experts = nn.ModuleList(
            [nn.Linear(hidden, hidden, bias=False) for _ in range(n_experts)]
        )
        self.n_experts = n_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class FakeCollapsingMoEModel(nn.Module):
    """
    Model whose gate weight is frozen at near-zero except expert 0 —
    simulates a collapsed routing distribution for collapse/entropy tests.
    """

    def __init__(self, hidden: int = 32, n_experts: int = 8) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden, n_experts, bias=False)
        self.experts = nn.ModuleList(
            [nn.Linear(hidden, hidden, bias=False) for _ in range(n_experts)]
        )
        self.n_experts = n_experts
        # Bias gate toward expert 0 heavily
        with torch.no_grad():
            self.gate.weight.data.zero_()
            self.gate.weight.data[0] += 20.0  # expert 0 dominates

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gate(x)
        return x  # simplified


# ---------------------------------------------------------------------------
# pytest fixtures wrapping the factories
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def small_moe_model() -> FakeMoEModel:
    """Small FakeMoEModel (2 layers, 4 experts, hidden=32). Session-scoped."""
    return FakeMoEModel(n_layers=2, n_experts=4, hidden=32)


@pytest.fixture(scope="session")
def large_moe_model() -> FakeMoEModel:
    """Larger FakeMoEModel (4 layers, 8 experts, hidden=64). Session-scoped."""
    return FakeMoEModel(n_layers=4, n_experts=8, hidden=64)


@pytest.fixture(scope="session")
def non_moe_model() -> FakeNonMoEModel:
    """Plain MLP (no router modules). Session-scoped."""
    return FakeNonMoEModel(hidden=32)


@pytest.fixture
def moe_model() -> FakeMoEModel:
    """Fresh FakeMoEModel per test function (avoids state bleed)."""
    return FakeMoEModel(n_layers=2, n_experts=4, hidden=32)


@pytest.fixture
def collapsing_model() -> FakeCollapsingMoEModel:
    """FakeCollapsingMoEModel (biased gate → expert 0 monopoly)."""
    return FakeCollapsingMoEModel(n_experts=4, hidden=32)


@pytest.fixture
def manual_router_model() -> FakeModelWithManualRouter:
    """Model with explicit 'moe_router' attribute for override path tests."""
    return FakeModelWithManualRouter(hidden=16, n_experts=8)


# ===========================================================================
# ── Section 3: WatchConfig with router_modules override ─────────────────────
# ===========================================================================


@pytest.fixture
def config_with_override(manual_router_model: FakeModelWithManualRouter) -> WatchConfig:
    """WatchConfig that points router_modules at 'moe_router' on manual_router_model."""
    return WatchConfig(
        output=OutputMode.SILENT,
        router_modules=["moe_router"],
    )


@pytest.fixture
def config_with_bad_override() -> WatchConfig:
    """WatchConfig that references a non-existent module name — tests ValueError."""
    return WatchConfig(
        output=OutputMode.SILENT,
        router_modules=["this.does.not.exist", "also.missing"],
    )


# ===========================================================================
# ── Section 4: Event builder helpers ────────────────────────────────────────
# ===========================================================================
#
# These helpers create valid RoutingEvent / GradientEvent instances with
# sensible defaults so individual tests can override only the fields that
# matter for their assertions.
#
# ===========================================================================


def make_routing_event(
    *,
    layer_name: str = "layers.0.gate",
    n_experts: int = 4,
    batch_size: int = 8,
    global_step: int = 0,
    logits: Optional[torch.Tensor] = None,
    uniform: bool = True,
) -> RoutingEvent:
    """
    Build a synthetic RoutingEvent.

    Parameters
    ----------
    layer_name : str
        Name of the (fake) router module.
    n_experts : int
        Number of experts (determines logit tensor width).
    batch_size : int
        Number of token rows in the routing event.
    global_step : int
        Training step tag.
    logits : torch.Tensor, optional
        Pre-built logit tensor of shape ``[batch_size, n_experts]``.
        If None, random logits are generated (uniform=True → uniform
        distribution, uniform=False → biased toward expert 0).
    uniform : bool
        When ``logits`` is None, whether to use uniform logits.

    Returns
    -------
    RoutingEvent
    """
    if logits is None:
        if uniform:
            logits = torch.zeros(batch_size, n_experts)

            selected = torch.empty(
                batch_size,
                min(2, n_experts),
                dtype=torch.long,
            )

            for i in range(batch_size):
                selected[i, 0] = i % n_experts

                if n_experts > 1:
                    selected[i, 1] = (i + 1) % n_experts

        else:
            logits = torch.zeros(batch_size, n_experts)
            logits[:, 0] = 20.0

            selected = torch.zeros(
                batch_size,
                2,
                dtype=torch.long,
            )

    else:
        _, selected = torch.topk(
            logits,
            k=min(2, n_experts),
            dim=-1,
        )

    return RoutingEvent(
        timestamp=time.time(),
        global_step=global_step,
        layer_name=layer_name,
        routing_logits=logits.detach(),
        selected_experts=selected.detach(),
        expert_count=n_experts,
        batch_size=batch_size,
    )


def make_gradient_event(
    *,
    layer_name: str = "layers.0.experts",
    expert_id: int = 0,
    gradient_norm: float = 0.1,
    global_step: int = 0,
) -> GradientEvent:
    """
    Build a synthetic GradientEvent.

    Parameters
    ----------
    layer_name : str
        Expert layer name.
    expert_id : int
        Expert index (0-based).
    gradient_norm : float
        Simulated L2 gradient norm.
    global_step : int
        Training step tag.

    Returns
    -------
    GradientEvent
    """
    return GradientEvent(
        timestamp=time.time(),
        global_step=global_step,
        layer_name=layer_name,
        expert_id=expert_id,
        gradient_norm=gradient_norm,
        gradient_magnitude=gradient_norm,
    )


# Export helpers so test modules can import from conftest
pytest.make_routing_event = make_routing_event  # type: ignore[attr-defined]
pytest.make_gradient_event = make_gradient_event  # type: ignore[attr-defined]


# ===========================================================================
# ── Section 5: StatCollector fixtures ───────────────────────────────────────
# ===========================================================================


@pytest.fixture
def stat_collector(default_config: WatchConfig) -> StatCollector:
    """Fresh StatCollector with default config."""
    return StatCollector(default_config)


@pytest.fixture
def populated_collector(default_config: WatchConfig) -> StatCollector:
    """
    StatCollector pre-populated with routing events for two layers.

    Layer topology:
      - "layers.0.gate"  — 4 experts, 20 uniform routing events
      - "layers.1.gate"  — 4 experts, 20 collapsed routing events (expert 0)

    Useful for analyzer tests that need data without triggering real hooks.
    """
    collector = StatCollector(default_config)
    n_experts = 4
    batch_size = 8

    collector.register_layer("layers.0.gate", n_experts)
    collector.register_layer("layers.1.gate", n_experts)

    for step in range(20):
        # Layer 0: uniform distribution
        collector.write_routing_event(
            make_routing_event(
                layer_name="layers.0.gate",
                n_experts=n_experts,
                batch_size=batch_size,
                global_step=step,
                uniform=True,
            )
        )
        # Layer 1: collapsed distribution (expert 0 monopoly)
        collector.write_routing_event(
            make_routing_event(
                layer_name="layers.1.gate",
                n_experts=n_experts,
                batch_size=batch_size,
                global_step=step,
                uniform=False,
            )
        )

    return collector


@pytest.fixture
def collector_with_gradients(default_config: WatchConfig) -> StatCollector:
    """
    StatCollector with both routing and gradient events injected.

    Layer  "layers.0.gate" — 4 experts, 20 routing events
    Layer  "layers.0.experts" — 4 experts, each 10 gradient events
      experts 0–2 : healthy gradient norm (0.15)
      expert  3   : starved gradient norm  (0.001)
    """
    collector = StatCollector(default_config)
    n_experts = 4

    collector.register_layer("layers.0.gate", n_experts)

    for step in range(20):
        collector.write_routing_event(
            make_routing_event(
                layer_name="layers.0.gate",
                n_experts=n_experts,
                batch_size=4,
                global_step=step,
            )
        )

    # Gradient events
    for step in range(10):
        for eid in range(n_experts):
            norm = 0.001 if eid == 3 else 0.15
            collector.write_gradient_event(
                make_gradient_event(
                    layer_name="layers.0.experts",
                    expert_id=eid,
                    gradient_norm=norm,
                    global_step=step,
                )
            )

    return collector


# ===========================================================================
# ── Section 6: RingBuffer fixtures ──────────────────────────────────────────
# ===========================================================================


@pytest.fixture
def small_ring_buffer() -> RingBuffer:
    """RingBuffer with capacity=5 for boundary / eviction tests."""
    return RingBuffer(capacity=5)


@pytest.fixture
def default_ring_buffer() -> RingBuffer:
    """RingBuffer with default capacity=1000."""
    return RingBuffer()


@pytest.fixture
def prefilled_ring_buffer() -> RingBuffer:
    """RingBuffer(capacity=10) pre-filled with 10 RoutingEvents (steps 0–9)."""
    buf: RingBuffer = RingBuffer(capacity=10)
    for step in range(10):
        buf.write(make_routing_event(global_step=step))
    return buf


# ===========================================================================
# ── Section 7: Logit / distribution helpers ─────────────────────────────────
# ===========================================================================


def make_uniform_logits(batch_size: int, n_experts: int) -> torch.Tensor:
    """Return logit tensor producing a perfectly uniform routing distribution."""
    return torch.zeros(batch_size, n_experts)


def make_collapsed_logits(batch_size: int, n_experts: int, dominant: int = 0) -> torch.Tensor:
    """
    Return logit tensor producing a near-collapsed routing distribution.

    The ``dominant`` expert receives logit 20.0; all others receive 0.0.
    After softmax this yields p[dominant] ≈ 1.0.
    """
    logits = torch.zeros(batch_size, n_experts)
    logits[:, dominant] = 20.0
    return logits


def make_gradually_collapsing_logit_sequence(
    n_steps: int,
    batch_size: int,
    n_experts: int,
    final_bias: float = 15.0,
) -> List[torch.Tensor]:
    """
    Return a list of logit tensors that linearly transition from uniform
    to collapsed over ``n_steps`` steps.

    Useful for entropy trend tests (DECLINING expected).
    """
    sequence = []
    for t in range(n_steps):
        progress = t / max(n_steps - 1, 1)
        logits = torch.zeros(batch_size, n_experts)
        logits[:, 0] = final_bias * progress
        sequence.append(logits)
    return sequence


# Export helpers for direct import in test modules
pytest.make_uniform_logits = make_uniform_logits  # type: ignore[attr-defined]
pytest.make_collapsed_logits = make_collapsed_logits  # type: ignore[attr-defined]
pytest.make_gradually_collapsing_logit_sequence = make_gradually_collapsing_logit_sequence  # type: ignore[attr-defined]


# ===========================================================================
# ── Section 8: Entropy-specific fixtures ────────────────────────────────────
# ===========================================================================


@pytest.fixture
def uniform_probs_4() -> np.ndarray:
    """Perfectly uniform distribution over 4 experts."""
    return np.array([0.25, 0.25, 0.25, 0.25])


@pytest.fixture
def collapsed_probs_4() -> np.ndarray:
    """Fully collapsed distribution (expert 0 only)."""
    return np.array([1.0, 0.0, 0.0, 0.0])


@pytest.fixture
def near_collapsed_probs_8() -> np.ndarray:
    """Near-collapsed over 8 experts — expert 0 gets 95% of tokens."""
    arr = np.array([0.95, 0.05 / 7] * 1 + [0.05 / 7] * 7)
    arr = np.concatenate([[0.95], np.full(7, 0.05 / 7)])
    return arr / arr.sum()


@pytest.fixture
def entropy_stat_collector_uniform(default_config: WatchConfig) -> StatCollector:
    """StatCollector with 30 uniform routing events on layer 'layer.0.gate'."""
    collector = StatCollector(default_config)
    collector.register_layer("layer.0.gate", 4)
    for step in range(30):
        collector.write_routing_event(
            make_routing_event(
                layer_name="layer.0.gate",
                n_experts=4,
                batch_size=8,
                global_step=step,
                uniform=True,
            )
        )
    return collector


@pytest.fixture
def entropy_stat_collector_collapsed(default_config: WatchConfig) -> StatCollector:
    """StatCollector with 30 collapsed routing events on layer 'layer.0.gate'."""
    collector = StatCollector(default_config)
    collector.register_layer("layer.0.gate", 4)
    for step in range(30):
        collector.write_routing_event(
            make_routing_event(
                layer_name="layer.0.gate",
                n_experts=4,
                batch_size=8,
                global_step=step,
                uniform=False,
            )
        )
    return collector


@pytest.fixture
def entropy_stat_collector_declining(default_config: WatchConfig) -> StatCollector:
    """
    StatCollector where entropy is steadily decreasing (40 steps).

    Routing gradually collapses toward expert 0, so EntropyAnalyzer should
    classify trend as 'DECLINING' and drift_detected should eventually be True.
    """
    collector = StatCollector(default_config)
    collector.register_layer("layer.0.gate", 4)
    n_steps = 40
    for step in range(n_steps):
        progress = step / max(n_steps - 1, 1)
        logits = torch.zeros(8, 4)
        logits[:, 0] = 20.0 * progress  # linearly increases bias to expert 0
        collector.write_routing_event(
            RoutingEvent(
                timestamp=time.time(),
                global_step=step,
                layer_name="layer.0.gate",
                routing_logits=logits.detach(),
                selected_experts=torch.zeros(8, 2, dtype=torch.long),
                expert_count=4,
                batch_size=8,
            )
        )
    return collector


# ===========================================================================
# ── Section 9: Hook-testing fixtures ────────────────────────────────────────
# ===========================================================================


@pytest.fixture
def hook_manager_deps(
    moe_model: FakeMoEModel,
    default_config: WatchConfig,
) -> Tuple[FakeMoEModel, StatCollector, WatchConfig]:
    """
    Bundle of (model, stat_collector, config) ready for HookManager construction.
    Returned as a tuple so tests can unpack individually.
    """
    collector = StatCollector(default_config)
    return moe_model, collector, default_config


@pytest.fixture
def attached_hook_manager(
    hook_manager_deps: Tuple[FakeMoEModel, StatCollector, WatchConfig],
):
    """
    A HookManager that has already had attach() called.
    Yields the manager; calls detach() in teardown regardless of test outcome.
    """
    from moewatch.hooks.manager import HookManager

    model, collector, config = hook_manager_deps
    manager = HookManager(model, collector, config)
    manager.attach()
    yield manager
    manager.detach()


# ===========================================================================
# ── Section 10: Miscellaneous / utility fixtures ─────────────────────────────
# ===========================================================================


@pytest.fixture(autouse=False)
def reproducible_random() -> Generator[None, None, None]:
    """
    Set deterministic seeds for torch and numpy for the duration of a test.

    Use ``@pytest.mark.usefixtures("reproducible_random")`` on tests that
    generate random tensors and require reproducible results.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    yield
    # No teardown needed — seeds are per-process global state.


@pytest.fixture(scope="session")
def cpu_device() -> torch.device:
    """CPU device fixture — all tests run on CPU by default."""
    return torch.device("cpu")


@pytest.fixture(scope="session")
def n_experts_small() -> int:
    """Canonical small expert count used in tests (4)."""
    return 4


@pytest.fixture(scope="session")
def n_experts_large() -> int:
    """Canonical large expert count used in tests (8)."""
    return 8


@pytest.fixture(scope="session")
def hidden_dim() -> int:
    """Canonical hidden dimension for fake models (32)."""
    return 32
