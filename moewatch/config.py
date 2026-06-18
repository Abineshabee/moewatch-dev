# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# moewatch/config.py [ core ]
# =============================================================================
#
# Project      : MoEWatch
# Version      : v0.2.0
# Description  : Central configuration dataclass and enumerations.
#                Single source of truth for all tunable parameters across
#                the MoEWatch monitoring, analysis, intervention, and policy
#                subsystems. All other modules import from here.
#
# Author       : Abinesh N (@Abineshabee)
# Repository   : https://github.com/Abineshabee/MoEWatch
# License      : Apache 2.0
#
# Contents
# --------
#   OutputMode      — enum for output format (SILENT, CLI, JSON)
#   AlertLevel      — enum for alert severity (INFO, WARNING, CRITICAL)
#   WatchConfig     — central configuration dataclass with full validation
#
# Usage
# -----
#   from moewatch.config import WatchConfig, OutputMode, AlertLevel
#   config = WatchConfig(output=OutputMode.CLI, entropy_warn=0.25)
#
# =============================================================================

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class OutputMode(str, Enum):
    """Output format for MoEWatch monitoring reports.

    Attributes
    ----------
    SILENT:
        Suppress all output. Useful when the caller handles reporting
        programmatically via the returned WatchReport objects.
    CLI:
        Rich ANSI terminal dashboard. Default mode. Respects the
        NO_COLOR environment variable.
    JSON:
        Newline-delimited JSON emitted to stdout or a file. Suitable
        for log aggregators (Grafana, Splunk, W&B custom metrics).
    """

    SILENT = "silent"
    CLI = "cli"
    JSON = "json"


class AlertLevel(str, Enum):
    """Severity classification for alerts emitted by MoEWatch.

    Mirrors Python's ``logging`` module severity ladder so that alert
    levels can be compared, filtered, and mapped to log handlers without
    translation.  Levels are ordered from least to most severe:
    ``DEBUG < INFO < WARNING < ERROR < CRITICAL``.

    Attributes
    ----------
    DEBUG:
        Fine-grained diagnostic signal. Sub-threshold observation that
        does not yet require attention.  Useful for tracing routing
        dynamics during development or hyperparameter search.

        Example::

            # Expert utilisation is trending downward but still within
            # healthy bounds.
            AlertLevel.DEBUG

    INFO:
        Informational notice. No immediate action required.  Indicates
        a noteworthy routing event that is within acceptable limits.

        Example::

            # A previously cold expert recovered to HEALTHY state.
            AlertLevel.INFO

    WARNING:
        Elevated signal. Routing health is degrading; monitor closely.
        Thresholds have been crossed but the model is still recoverable
        without intervention.

        Example::

            # Normalised entropy fell below ``entropy_warn`` threshold.
            AlertLevel.WARNING

    ERROR:
        Serious degradation detected. Collapse is likely without action.
        Sits between WARNING and CRITICAL — the model is still training
        but expert routing is severely impaired.

        Example::

            # Multiple experts simultaneously entered DEAD state across
            # more than half of the router layers.
            AlertLevel.ERROR

    CRITICAL:
        Imminent collapse risk. Intervention recommended or triggered.
        MoEWatch will attempt an automatic intervention (if enabled) and
        the training run should be inspected immediately.

        Example::

            # Normalised entropy critically low; all tokens routed to a
            # single expert for the past N consecutive steps.
            AlertLevel.CRITICAL
    """

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# WatchConfig
# ---------------------------------------------------------------------------


@dataclass
class WatchConfig:
    """Central configuration container for all MoEWatch subsystems.

    All parameters have production-safe defaults that can be overridden
    at construction time. After construction, ``__post_init__`` validates
    every field and raises ``ValueError`` for any invalid combination.

    Parameters are grouped by concern:

    Signal Thresholds
        Numeric boundaries used by analyzers to classify expert health
        and trigger alerts.

    Collection & Monitoring
        Controls how frequently MoEWatch samples the model and emits
        reports during training.

    Behavior
        High-level switches for output format, color, router override,
        and intervention enable/disable.

    Policy Parameters (Phase 2 Bandit)
        Hyperparameters for the epsilon-greedy contextual bandit policy.
        Only relevant when ``policy_type == "bandit"``.

    Baseline Tracker
        Controls the intervention-conditioned counterfactual baseline
        regression window. Critical for accurate reward attribution.

    Intervention Safety
        Hard limits on intervention magnitude, cooldown, and loss spike
        detection that the SafetyGuard enforces before any action fires.

    Examples
    --------
    >>> config = WatchConfig()
    >>> config = WatchConfig(output=OutputMode.JSON, entropy_warn=0.25)
    >>> config = WatchConfig(
    ...     output=OutputMode.CLI,
    ...     intervention_enabled=True,
    ...     policy_type="bandit",
    ...     bandit_epsilon=0.15,
    ... )
    """

    # ------------------------------------------------------------------
    # SIGNAL THRESHOLDS
    # ------------------------------------------------------------------

    dead_threshold: float = 0.5
    """Gradient norm below this value → expert classified as DEAD.

    An expert is considered permanently dead when its gradient L2 norm
    falls below this threshold for ``cold_steps_limit`` consecutive steps.
    Default: 0.5 (calibrated for models with HIDDEN_DIM=64, top-k routing).
    """

    cold_threshold: float = 1.0
    """Gradient norm below this value → expert classified as COLD.

    Cold is the precursor state before DEAD. An expert that stays cold
    for ``cold_steps_limit`` steps is promoted to DEAD.
    Must be greater than ``dead_threshold``. Default: 1.0 (calibrated for models with HIDDEN_DIM=64, top-k routing).
    """

    cold_steps_limit: int = 50
    """Consecutive COLD steps before an expert is promoted to DEAD.

    Provides hysteresis to avoid false positives from transient gradient
    dips during normal training dynamics. Default: 50.
    """

    entropy_warn: float = 0.3
    """Normalized entropy below this → WARNING alert.

    Normalized entropy = H / log2(n_experts), so range is [0, 1].
    A value near 1.0 is fully uniform (healthy). Below ``entropy_warn``
    indicates narrowing router distribution. Default: 0.3.
    """

    entropy_critical: float = 0.15
    """Normalized entropy below this → CRITICAL alert.

    Must be less than ``entropy_warn``. Default: 0.15.
    """

    entropy_drop_warn: float = 0.1
    """Per-step entropy drop rate above this → WARNING.

    Detects sudden rapid narrowing of routing distribution, which is
    more dangerous than a gradual drift. Default: 0.1.
    """

    load_imbalance_warn: float = 3.0
    """Load imbalance ratio (max_load / mean_load) above this → WARNING.

    Captures token routing skew even when entropy has not yet dropped
    significantly. Default: 3.0.
    """

    load_imbalance_error: float = 5.0
    """Load imbalance ratio above this → CRITICAL alert.

    Must be greater than ``load_imbalance_warn``. Default: 5.0.
    """

    # ------------------------------------------------------------------
    # COLLECTION & MONITORING
    # ------------------------------------------------------------------

    log_every: int = 10
    """Emit reports every N training steps.

    Lower values increase output verbosity. Set to 1 for step-level
    granularity during debugging. Default: 10.
    """

    sample_every: int = 10
    """Sample gradient backward hooks every N training steps.

    Gradient hook execution adds per-step overhead. Sampling every N
    steps reduces that cost while preserving signal fidelity.
    Default: 10.
    """

    output: OutputMode = OutputMode.CLI
    """Output format for monitoring reports.

    SILENT suppresses output. CLI renders a Rich terminal dashboard.
    JSON emits newline-delimited records for log aggregation.
    Default: OutputMode.CLI.
    """

    # ------------------------------------------------------------------
    # BEHAVIOR
    # ------------------------------------------------------------------

    no_color: bool = False
    """Disable ANSI color codes in CLI output.

    Overrides the ``NO_COLOR`` environment variable. MoEWatch also
    checks the environment automatically at import time. Default: False.

    .. warning::
        The ``NO_COLOR`` environment variable follows the
        `no-color.org <https://no-color.org>`_ spec literally: presence
        of *any* non-empty value disables color, regardless of its
        content. This means ``NO_COLOR=0`` also disables color, even
        though ``"0"`` commonly means "off"/"false" in other tools.
        Only an unset variable or an empty/whitespace-only value leaves
        color enabled. To force color back on in code, construct with
        ``no_color=False`` -- but note the environment variable still
        takes precedence over this field if set to a non-empty value.
    """

    router_modules: Optional[List[str]] = None
    """Manual override for router module name patterns.

    When set, ``detect_router_modules()`` uses only these name substrings
    instead of the built-in heuristics for Mixtral, Qwen3-MoE,
    DeepSeek-MoE, and OLMoE. Useful for custom architectures.
    Must be a ``list`` of ``str`` (e.g. ``["gate", "router"]``) -- a
    bare string such as ``"gate"`` is rejected at construction time,
    since iterating over it character-by-character would silently
    produce incorrect router detection.
    Default: None (auto-detect).
    """

    intervention_enabled: bool = True
    """Enable the intervention engine.

    When False, MoEWatch runs in observation-only mode: all signals are
    computed and reported, but no actions are applied to the trainer.
    Default: True.
    """

    policy_type: str = "rule"
    """Policy implementation to use for action selection.

    ``"rule"``   — Phase 1 deterministic threshold policy. Stable,
                   explainable, ships first.
    ``"bandit"`` — Phase 2 epsilon-greedy contextual bandit. Learning
                   policy; requires experience replay and reward.
    Default: ``"rule"``.
    """

    # ------------------------------------------------------------------
    # POLICY PARAMETERS (Phase 2 Bandit)
    # ------------------------------------------------------------------

    bandit_epsilon: float = 0.1
    """Epsilon for epsilon-greedy exploration in the bandit policy.

    At each decision, the policy explores (random action) with
    probability ``epsilon``, and exploits (best known action) otherwise.
    Valid range: [0.0, 1.0]. Default: 0.1.
    """

    reward_discount_gamma: float = 0.95
    """Exponential discount factor for delayed reward computation.

    reward = Σ ΔH_counterfactual(t+k) * γ^k  for k = 1..K

    Higher γ gives more weight to later steps in the observation window.
    Valid range: (0.0, 1.0]. Default: 0.95.
    """

    reward_window_steps: int = 50
    """Number of steps in the reward observation window (K).

    The observation window defines how long after an intervention we
    wait before computing the discounted counterfactual reward.
    Default: 50.
    """

    # ------------------------------------------------------------------
    # BASELINE TRACKER
    # ------------------------------------------------------------------

    baseline_min_clean_steps: int = 20
    """Minimum clean (non-intervention) history steps for a valid baseline.

    If fewer than this many uncontaminated steps are available,
    the policy defaults to the do-nothing (NoOp) action.
    Default: 20.
    """

    baseline_exclusion_window: int = 100
    """Steps to exclude after an intervention before using baseline again.

    After any intervention on layer L, the next
    ``baseline_exclusion_window`` steps for that layer are marked as
    "intervention-influenced" and excluded from the baseline regression.
    This prevents the baseline-illusion problem. Default: 100.
    """

    # ------------------------------------------------------------------
    # INTERVENTION SAFETY
    # ------------------------------------------------------------------

    intervention_cooldown: int = 200
    """Minimum steps between consecutive interventions on the same layer.

    Enforced by SafetyGuard. Prevents rapid re-intervention before the
    previous action's effects have been observed. Default: 200.
    """

    intervention_max_delta: float = 0.1
    """Maximum magnitude change to any hyperparameter per intervention.

    Enforced by SafetyGuard. Provides a hard upper bound on how
    aggressively a single intervention can shift training dynamics.
    Default: 0.1.
    """

    loss_guard_threshold: float = 1.5
    """Loss spike factor above baseline that freezes all interventions.

    If current_loss > loss_guard_threshold * baseline_loss, all
    intervention actions are downgraded to NoOp until loss recovers.
    Default: 1.5 (50% spike above baseline).
    """

    # ------------------------------------------------------------------
    # AUDIT
    # ------------------------------------------------------------------

    audit_with_backward: bool = True
    """Run a proxy backward pass during audit() to populate gradient hooks.

    When True (default), audit() computes a sum-of-logits proxy loss and
    calls backward() after each forward pass so that Tier 1 gradient
    starvation hooks fire and gradient_results is populated. Set to False
    for routing-only audits where backward is too expensive (large models,
    CPU-only environments, or architectures where the proxy loss produces
    unstable gradients).

    This value is used as the default for the ``with_backward`` parameter
    of :func:`moewatch.audit`. Passing ``with_backward`` explicitly to
    ``audit()`` always takes precedence over this config field.
    Default: True.
    """

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate all configuration fields.

        Called automatically after dataclass construction. Raises
        ``ValueError`` with a descriptive message for any invalid value.

        Validation rules
        ----------------
        - 0.0 < entropy_critical < entropy_warn < 1.0
        - dead_threshold < cold_threshold
        - entropy_drop_warn > 0.0
        - load_imbalance_warn < load_imbalance_error
        - 0.0 <= bandit_epsilon <= 1.0
        - 0.0 < reward_discount_gamma <= 1.0
        - log_every >= 1, sample_every >= 1
        - policy_type in {"rule", "bandit"}
        - output is a valid OutputMode
        - router_modules is None or a list of str
        - NO_COLOR env var respected
        """
        # Entropy thresholds
        if not (0.0 < self.entropy_critical < self.entropy_warn < 1.0):
            raise ValueError(
                f"Entropy thresholds must satisfy "
                f"0 < entropy_critical ({self.entropy_critical}) < "
                f"entropy_warn ({self.entropy_warn}) < 1.0"
            )

        # Gradient thresholds
        if self.dead_threshold <= 0.0:
            raise ValueError(
                f"dead_threshold must be positive, got {self.dead_threshold}"
            )
        if self.cold_threshold <= self.dead_threshold:
            raise ValueError(
                f"cold_threshold ({self.cold_threshold}) must be greater than "
                f"dead_threshold ({self.dead_threshold})"
            )

        # Entropy drop
        if self.entropy_drop_warn <= 0.0:
            raise ValueError(
                f"entropy_drop_warn must be positive, got {self.entropy_drop_warn}"
            )

        # Load imbalance
        if self.load_imbalance_warn <= 1.0:
            raise ValueError(
                f"load_imbalance_warn must be > 1.0, got {self.load_imbalance_warn}"
            )
        if self.load_imbalance_error <= self.load_imbalance_warn:
            raise ValueError(
                f"load_imbalance_error ({self.load_imbalance_error}) must be "
                f"greater than load_imbalance_warn ({self.load_imbalance_warn})"
            )

        # Collection frequency
        if self.log_every < 1:
            raise ValueError(f"log_every must be >= 1, got {self.log_every}")
        if self.sample_every < 1:
            raise ValueError(f"sample_every must be >= 1, got {self.sample_every}")

        # Bandit parameters
        if not (0.0 <= self.bandit_epsilon <= 1.0):
            raise ValueError(
                f"bandit_epsilon must be in [0, 1], got {self.bandit_epsilon}"
            )
        if not (0.0 < self.reward_discount_gamma <= 1.0):
            raise ValueError(
                f"reward_discount_gamma must be in (0, 1], "
                f"got {self.reward_discount_gamma}"
            )
        if self.reward_window_steps < 1:
            raise ValueError(
                f"reward_window_steps must be >= 1, got {self.reward_window_steps}"
            )

        # Baseline tracker
        if self.baseline_min_clean_steps < 1:
            raise ValueError(
                f"baseline_min_clean_steps must be >= 1, "
                f"got {self.baseline_min_clean_steps}"
            )
        if self.baseline_exclusion_window < 0:
            raise ValueError(
                f"baseline_exclusion_window must be >= 0, "
                f"got {self.baseline_exclusion_window}"
            )

        # Intervention safety
        if self.intervention_cooldown < 0:
            raise ValueError(
                f"intervention_cooldown must be >= 0, got {self.intervention_cooldown}"
            )
        if self.intervention_max_delta <= 0.0:
            raise ValueError(
                f"intervention_max_delta must be positive, "
                f"got {self.intervention_max_delta}"
            )
        if self.loss_guard_threshold <= 1.0:
            raise ValueError(
                f"loss_guard_threshold must be > 1.0 (a multiplicative spike "
                f"factor), got {self.loss_guard_threshold}"
            )

        # audit_with_backward — must be a plain bool
        if not isinstance(self.audit_with_backward, bool):
            raise ValueError(
                f"audit_with_backward must be bool, "
                f"got {type(self.audit_with_backward).__name__!r}"
            )

        # Policy type
        if self.policy_type not in {"rule", "bandit"}:
            raise ValueError(
                f"policy_type must be 'rule' or 'bandit', got '{self.policy_type}'"
            )

        # Router modules — must be None or a list of str. A bare string
        # is a common mistake (e.g. router_modules="gate" instead of
        # router_modules=["gate"]) that would otherwise be accepted
        # silently and then iterated character-by-character downstream
        # in detect_router_modules(), producing incorrect router
        # detection with no error raised anywhere.
        if self.router_modules is not None:
            if not isinstance(self.router_modules, list) or not all(
                isinstance(item, str) for item in self.router_modules
            ):
                raise ValueError(
                    f"router_modules must be None or a list of str, "
                    f"got {self.router_modules!r}"
                )

        # Output mode — coerce string to enum if passed as plain string
        if isinstance(self.output, str):
            try:
                object.__setattr__(self, "output", OutputMode(self.output.lower()))
            except ValueError:
                valid = [m.value for m in OutputMode]
                raise ValueError(
                    f"output must be one of {valid}, got '{self.output}'"
                )

        # NO_COLOR environment variable — override no_color flag
        if os.environ.get("NO_COLOR", "").strip():
            object.__setattr__(self, "no_color", True)

    def is_color_enabled(self) -> bool:
        """Return True if ANSI colors should be rendered.

        Accounts for both the ``no_color`` field and the ``NO_COLOR``
        environment variable (set at construction time).

        Returns
        -------
        bool
            ``True`` when color output is active.
        """
        return not self.no_color

    def __repr__(self) -> str:
        return (
            f"WatchConfig("
            f"output={self.output.value!r}, "
            f"policy_type={self.policy_type!r}, "
            f"intervention_enabled={self.intervention_enabled}, "
            f"entropy_warn={self.entropy_warn}, "
            f"entropy_critical={self.entropy_critical}, "
            f"log_every={self.log_every}"
            f")"
        )
