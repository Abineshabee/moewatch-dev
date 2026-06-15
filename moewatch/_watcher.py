# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# moewatch/_watcher.py [ core ]
# =============================================================================
#
# Project      : MoEWatch
# Version      : v0.2.0
# Description  : Main MoEWatch class — live training-time MoE routing
#                collapse monitor and adaptive intervention controller.
#
#                This is the primary user-facing class. It coordinates the
#                full MoEWatch subsystem lifecycle:
#
#                  HookManager        — attach/detach forward & backward hooks
#                  StatCollector      — aggregate routing & gradient events
#                  BaselineTracker    — intervention-conditioned baselines
#                  EntropyAnalyzer    — Tier 2 entropy drift (signal)
#                  CollapseDetector   — expert health state machine
#                  GradientStarvationAnalyzer — Tier 1 gradient starvation
#                  CrossLayerCorrelation      — Tier 3 spread localization
#                  RiskScoreFuser     — weighted signal fusion → risk_score
#                  InterventionEngine — apply & measure live interventions
#                  RulePolicy         — Phase 1 deterministic action selection
#                  BanditPolicy       — Phase 2 learning action selection
#                  CLIReporter        — Rich terminal dashboard
#                  JSONReporter       — newline-delimited JSON stream
#
#                Zero weight modification guarantee: all hooks are read-only.
#                All model parameter changes go through the Trainer API only.
#
# Author       : Abinesh N (@Abineshabee)
# Repository   : https://github.com/Abineshabee/MoEWatch
# License      : Apache 2.0
#
# Classes
# -------
#   MoEWatch            — main monitoring class (context manager supported)
#   MoEWatchCallback    — HuggingFace Trainer callback (calls step())
#
# Usage
# -----
#   # Standard usage
#   config = WatchConfig(output=OutputMode.CLI)
#   watch = MoEWatch(model, config)
#   watch.attach(trainer)
#   trainer.train()
#   watch.stop()
#
#   # Context manager usage
#   with MoEWatch(model, config) as watch:
#       watch.attach(trainer)
#       trainer.train()
#
# =============================================================================

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    # Provide a minimal stub so the module can be imported even without
    # transformers installed. Actual use will fail at runtime with clear error.
    class TrainerCallback:  # type: ignore[no-redef]
        """Stub callback when transformers is not installed."""

from moewatch.config import AlertLevel, OutputMode, WatchConfig
from moewatch import Alert  # re-use canonical Alert definition

# Sub-system imports — deferred at function boundary to keep module-level
# imports fast and testable without the full dependency chain loaded.
# Each subsystem is imported lazily inside the methods that need them.

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MoEWatchCallback
# ---------------------------------------------------------------------------


class MoEWatchCallback(TrainerCallback):
    """HuggingFace Trainer callback that drives MoEWatch at each training step.

    This callback is registered with the HuggingFace ``Trainer`` via
    ``MoEWatch.attach(trainer)``. After each optimizer step the Trainer
    calls ``on_step_end``, which delegates to ``MoEWatch.step()``.

    Users do not need to instantiate this class directly; it is created
    automatically by ``MoEWatch.attach()``.

    Attributes
    ----------
    moewatch : MoEWatch
        Reference to the parent ``MoEWatch`` instance.
    """

    def __init__(self, moewatch: "MoEWatch") -> None:
        super().__init__()
        self.moewatch: "MoEWatch" = moewatch

    def on_step_end(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs,
    ) -> None:
        """Called by the Trainer after each optimizer step.

        Delegates to ``MoEWatch.step()`` with the current global step.
        Captures and logs any exceptions so that a monitoring error never
        interrupts training.

        Parameters
        ----------
        args : TrainingArguments
            HuggingFace training arguments.
        state : TrainerState
            Training state, including ``global_step`` and ``log_history``.
        control : TrainerControl
            Training control flags.
        **kwargs
            Additional keyword arguments from the Trainer (ignored).
        """
        if not self.moewatch._running:
            return
        try:
            current_loss = _extract_loss(state)
            self.moewatch.step(
                global_step=state.global_step,
                current_loss=current_loss,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "[MoEWatch] Unhandled error in step %d: %s",
                state.global_step,
                exc,
                exc_info=True,
            )

    def on_train_end(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs,
    ) -> None:
        """Called when training finishes. Stops the MoEWatch monitor."""
        if self.moewatch._running:
            self.moewatch.stop()


def _extract_loss(state: "TrainerState") -> float:
    """Extract the most recent training loss from TrainerState.

    Falls back to 0.0 if no loss is available (e.g. at step 0).

    Parameters
    ----------
    state : TrainerState
        HuggingFace TrainerState object.

    Returns
    -------
    float
        Most recent training loss, or 0.0 if unavailable.
    """
    if state.log_history:
        for entry in reversed(state.log_history):
            if "loss" in entry:
                return float(entry["loss"])
    return 0.0


# ---------------------------------------------------------------------------
# MoEWatch
# ---------------------------------------------------------------------------


class MoEWatch:
    """Live training-time MoE routing collapse monitor and intervention controller.

    Attaches to a HuggingFace Trainer and monitors MoE routing health at
    each training step. Emits structured alerts and can apply corrective
    interventions to live training hyperparameters.

    Architecture
    ------------
    The monitor runs a three-tier empirically ordered signal hierarchy:

    Tier 1 — Gradient Starvation  (50–200 steps before collapse)
        Earliest and most actionable signal. Detected via backward hooks
        on expert weight tensors.

    Tier 2 — Entropy Drift  (30–100 steps before collapse)
        Intermediate confirmation signal. Router distribution narrowing
        detected via CUSUM on per-layer normalized entropy.

    Tier 3 — Cross-Layer Correlation  (system-level spread localization)
        Identifies source layer and downstream victims as collapse spreads
        across layers.

    Risk Score Fusion
    -----------------
    risk_score = 0.6 * T1 + 0.3 * T2 + 0.1 * T3  (range 0.0 → 1.0)

    Zero Weight Modification Guarantee
    ------------------------------------
    All monitoring hooks are strictly read-only. No expert weights are
    modified by MoEWatch. Interventions operate through the Trainer API
    (aux_loss_coef, dropout, noise hooks) only.

    Parameters
    ----------
    model : torch.nn.Module
        The MoE model to monitor. Must contain detectable MoE router layers.
    config : WatchConfig, optional
        Configuration container. Uses ``WatchConfig()`` defaults if None.

    Raises
    ------
    ValueError
        If no MoE router modules can be detected in the model and no
        ``config.router_modules`` override is provided.
    ImportError
        If PyTorch or HuggingFace Transformers are not installed.

    Examples
    --------
    Standard usage:

    >>> config = WatchConfig(output=OutputMode.CLI, intervention_enabled=True)
    >>> watch = MoEWatch(model, config)
    >>> watch.attach(trainer)
    >>> trainer.train()
    >>> watch.stop()

    Context manager:

    >>> with MoEWatch(model, config) as watch:
    ...     watch.attach(trainer)
    ...     trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[WatchConfig] = None,
    ) -> None:
        self.model: nn.Module = model
        self.config: WatchConfig = config if config is not None else WatchConfig()

        # --- Internal state ---
        self._running: bool = False
        self._global_step: int = 0
        self._start_time: float = 0.0
        self._trainer: Optional[object] = None  # transformers.Trainer

        # Thread safety for alert list mutations
        self._alert_lock: threading.Lock = threading.Lock()
        self.alerts: List[Alert] = []

        # Rolling accumulator of per-step reports (populated by step()).
        from moewatch.report.watch_report import WatchReport
        self.watch_report: WatchReport = WatchReport()

        # --- Sub-system instances (initialized in start()) ---
        self.hook_manager = None
        self.stat_collector = None
        self.baseline_tracker = None
        self.entropy_analyzer = None
        self.collapse_detector = None
        self.gradient_analyzer = None
        self.cross_layer_analyzer = None
        self.risk_fuser = None
        self.intervention_engine = None
        self.policy = None
        self._intervention_gap_warned = False
        self._cli_reporter = None
        self._json_reporter = None

        # Layer metadata populated after hook attachment
        self._detected_layers: Dict[str, nn.Module] = {}
        self._layer_order: List[str] = []

        # --- Validate model has detectable routers (early check) ---
        self._validate_model()

        logger.debug(
            "[MoEWatch] Initialized. Model: %s | Config: %s",
            type(model).__name__,
            repr(self.config),
        )

    # ------------------------------------------------------------------
    # Lifecycle: start / stop / context manager
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Initialize all sub-systems and attach hooks to the model.

        This method must be called before ``attach()`` or ``step()``.
        It instantiates all analyzers, collectors, and the intervention
        engine, then registers forward and backward hooks on the detected
        MoE router modules.

        Raises
        ------
        RuntimeError
            If called while already running.
        ValueError
            If no router modules are detectable.
        """
        if self._running:
            raise RuntimeError(
                "[MoEWatch] start() called while already running."
            )

        self._start_time = time.time()

        # ---- Import sub-systems lazily ----
        from moewatch.hooks.manager import HookManager
        from moewatch.collector.stat_collector import StatCollector
        from moewatch.collector.baseline_tracker import BaselineTracker
        from moewatch.analyzer.entropy import EntropyAnalyzer
        from moewatch.analyzer.collapse import CollapseDetector
        from moewatch.analyzer.gradient_starvation import GradientStarvationAnalyzer
        from moewatch.analyzer.cross_layer import CrossLayerCorrelation
        from moewatch.analyzer.risk_score import RiskScoreFuser

        # ---- Construct sub-systems ----
        self.stat_collector = StatCollector(self.config)
        self.baseline_tracker = BaselineTracker(self.config)
        self.hook_manager = HookManager(
            model=self.model,
            stat_collector=self.stat_collector,
            config=self.config,
        )
        self.entropy_analyzer = EntropyAnalyzer(self.config)
        self.collapse_detector = CollapseDetector(self.config)
        self.gradient_analyzer = GradientStarvationAnalyzer(self.config)
        self.cross_layer_analyzer = CrossLayerCorrelation(self.config)
        self.risk_fuser = RiskScoreFuser(self.config)

        # ---- Attach hooks ----
        self.hook_manager.attach()
        self._detected_layers = self.hook_manager.get_layer_map()
        self._layer_order = list(self._detected_layers.keys())

        # ---- Register layers with baseline tracker ----
        for layer_name in self._layer_order:
            self.baseline_tracker.register_layer(layer_name)

        # ---- Construct intervention system (if enabled) ----
        # Interventions operate directly on self.model (a torch.nn.Module),
        # so no HuggingFace Trainer is required to initialise this system.
        if self.config.intervention_enabled:
            self._init_intervention_system()

        # ---- Reporters ----
        if self.config.output == OutputMode.CLI:
            from moewatch.report.cli_reporter import CLIReporter
            self._cli_reporter = CLIReporter(self.config)
        elif self.config.output == OutputMode.JSON:
            from moewatch.report.json_reporter import JSONReporter
            self._json_reporter = JSONReporter(self.config)

        self._running = True
        logger.info(
            "[MoEWatch] Started. Detected %d router layers: %s",
            len(self._layer_order),
            self._layer_order,
        )

    def stop(self) -> None:
        """Detach all hooks and finalize monitoring.

        Safe to call multiple times (idempotent). Guaranteed to execute
        hook cleanup even if called after an exception during training.

        Side Effects
        ------------
        - Removes all registered forward and backward hooks from the model.
        - Sets ``_running = False``.
        - Emits a final summary log if output is not SILENT.
        """
        if not self._running:
            return

        self._running = False

        try:
            if self.hook_manager is not None:
                self.hook_manager.detach()
                logger.debug("[MoEWatch] All hooks detached.")
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("[MoEWatch] Error during hook detachment: %s", exc)
        finally:
            elapsed = time.time() - self._start_time if self._start_time else 0.0
            logger.info(
                "[MoEWatch] Stopped. Total steps monitored: %d | "
                "Total alerts: %d | Elapsed: %.1fs",
                self._global_step,
                len(self.alerts),
                elapsed,
            )

    def attach(self, trainer: object) -> "MoEWatchCallback":
        """Register MoEWatch as a HuggingFace Trainer callback.

        Stores a reference to the trainer and registers a
        ``MoEWatchCallback`` so that ``step()`` is called after each
        training step. If ``start()`` has not yet been called, it is
        called automatically here.

        Parameters
        ----------
        trainer : transformers.Trainer
            The HuggingFace Trainer instance that will drive training.

        Returns
        -------
        MoEWatchCallback
            The callback object registered with the trainer.

        Raises
        ------
        ImportError
            If HuggingFace Transformers is not installed.
        """
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "[MoEWatch] attach() requires HuggingFace Transformers.\n"
                "Install with: pip install transformers>=4.35"
            )

        self._trainer = trainer

        if not self._running:
            self.start()
        elif self.config.intervention_enabled and self.intervention_engine is None:
            self._init_intervention_system()

        callback = MoEWatchCallback(self)
        trainer.add_callback(callback)
        logger.debug("[MoEWatch] Callback registered with Trainer.")
        return callback

    def __enter__(self) -> "MoEWatch":
        """Context manager entry. Calls ``start()``."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit. Calls ``stop()`` regardless of exceptions."""
        self.stop()

    # ------------------------------------------------------------------
    # Core monitoring loop
    # ------------------------------------------------------------------

    def pre_step(self, global_step: int) -> None:
        """Inform MoEWatch of the upcoming training step number.

        Call this *before* the forward pass (i.e. before ``model(x)``)
        so that router/gradient hooks tag the events they capture during
        that forward/backward pass with the correct ``global_step``.

        Without this call (or if called only inside :meth:`step`, which
        runs *after* the forward/backward pass), hook-captured events
        default to ``global_step=0`` because the hooks' internal step
        counter is never advanced before they fire.

        Parameters
        ----------
        global_step : int
            The training step number about to begin.
        """
        self._global_step = global_step
        if self.hook_manager is not None:
            try:
                self.hook_manager.set_global_step(global_step)
            except Exception as exc:  # pylint: disable=broad-except
                logger.debug(
                    "[MoEWatch] pre_step(): failed to propagate global_step "
                    "to hook_manager: %s",
                    exc,
                )

    def step(
        self,
        global_step: int,
        current_loss: float = 0.0,
    ) -> "StepReport":
        """Process one training step through the full analysis pipeline.

        Called by ``MoEWatchCallback.on_step_end()`` after each optimizer
        step. Runs all analyzers, fuses risk scores, selects interventions
        if enabled, and emits alerts if thresholds are crossed.

        Note
        ----
        For router/gradient hooks to tag *this* step's captured events
        with the correct ``global_step`` (rather than the previous
        step's, or ``0`` on the first step), call :meth:`pre_step` with
        the same ``global_step`` *before* the forward pass. This method
        also propagates ``global_step`` to the hook manager as a
        best-effort fallback for callers that only invoke ``step()``,
        but that only benefits events captured during the *next* step's
        forward pass.

        Parameters
        ----------
        global_step : int
            Current training step number from ``TrainerState.global_step``.
        current_loss : float, optional
            Current training loss at this step. Used by SafetyGuard for
            loss spike detection. Default 0.0.

        Returns
        -------
        StepReport
            Structured per-step snapshot with risk scores, alerts,
            interventions, and policy decisions for this step. Also
            appended to ``self.watch_report`` for rolling/aggregate
            access across steps.
        """
        from moewatch.report.watch_report import StepReport

        self._global_step = global_step

        # Best-effort: propagate to hooks so events captured during the
        # *next* forward pass carry the correct step number even if the
        # caller never calls pre_step(). Events captured during *this*
        # step's already-completed forward/backward pass cannot be fixed
        # retroactively here.
        if self.hook_manager is not None:
            try:
                self.hook_manager.set_global_step(global_step)
            except Exception as exc:  # pylint: disable=broad-except
                logger.debug(
                    "[MoEWatch] step(): failed to propagate global_step "
                    "to hook_manager: %s",
                    exc,
                )

        # ---- 1. Run all analyzers ----
        entropy_reports = {}
        collapse_reports = {}
        gradient_reports = {}
        cross_layer_report = None
        risk_reports = {}

        try:
            entropy_reports = self.entropy_analyzer.analyze(self.stat_collector)
        except Exception as exc:
            logger.debug("[MoEWatch] EntropyAnalyzer error at step %d: %s", global_step, exc)

        try:
            collapse_reports = self.collapse_detector.analyze(self.stat_collector)
        except Exception as exc:
            logger.debug("[MoEWatch] CollapseDetector error at step %d: %s", global_step, exc)

        try:
            gradient_reports = self.gradient_analyzer.analyze(self.stat_collector)
        except Exception as exc:
            logger.debug("[MoEWatch] GradientStarvationAnalyzer error at step %d: %s", global_step, exc)

        try:
            cross_layer_report = self.cross_layer_analyzer.analyze(entropy_reports)
        except Exception as exc:
            logger.debug("[MoEWatch] CrossLayerCorrelation error at step %d: %s", global_step, exc)

        # ---- 2. Update baseline tracker with latest entropy signals ----
        for layer_name, ent_report in entropy_reports.items():
            try:
                self.baseline_tracker.update_signal(
                    layer_name=layer_name,
                    value=ent_report.normalized_entropy,
                    step=global_step,
                )
            except Exception:
                pass

        # ---- 3. Fuse signals into risk scores ----
        step_risk_scores: Dict[str, float] = {}
        step_risk_levels: Dict[str, str] = {}
        dominant_signals: Dict[str, str] = {}

        for layer_name in self._layer_order:
            try:
                # Get per-layer reports (use stubs if analyzer produced no output)
                ent_report = entropy_reports.get(layer_name)
                grad_layer_reports = gradient_reports.get(layer_name, [])
                # Use the worst-case expert report (highest starvation_score)
                grad_report = (
                    max(grad_layer_reports, key=lambda r: r.starvation_score)
                    if grad_layer_reports
                    else None
                )

                if ent_report is None or grad_report is None:
                    continue

                risk_report = self.risk_fuser.fuse(
                    gradient_report=grad_report,
                    entropy_report=ent_report,
                    cross_layer_report=cross_layer_report,
                )
                risk_reports[layer_name] = risk_report
                step_risk_scores[layer_name] = risk_report.risk_score
                step_risk_levels[layer_name] = risk_report.risk_level.name
                dominant_signals[layer_name] = risk_report.dominant_signal
            except Exception as exc:
                logger.debug(
                    "[MoEWatch] Risk fusion error for layer '%s' at step %d: %s",
                    layer_name, global_step, exc,
                )

        # ---- 4. Generate alerts ----
        new_alerts: List[Alert] = []
        new_alerts.extend(
            self._generate_entropy_alerts(entropy_reports, global_step)
        )
        new_alerts.extend(
            self._generate_collapse_alerts(collapse_reports, global_step)
        )
        new_alerts.extend(
            self._generate_gradient_alerts(gradient_reports, global_step)
        )
        new_alerts.extend(
            self._generate_risk_alerts(risk_reports, global_step)
        )

        with self._alert_lock:
            self.alerts.extend(new_alerts)

        # ---- 5. Intervention: select, validate, apply ----
        applied_interventions: List = []
        policy_decisions: Dict[str, str] = {}
        counterfactual_rewards: Dict[str, float] = {}

        if (
            self.config.intervention_enabled
            and self.intervention_engine is not None
            and self.policy is not None
        ):
            applied_interventions, policy_decisions = self._run_interventions(
                step=global_step,
                risk_reports=risk_reports,
                current_loss=current_loss,
            )

            # Check expired observation windows and compute rewards
            try:
                self.intervention_engine.check_observation_windows(
                    step=global_step,
                    risk_scores=step_risk_scores,
                    policy=self.policy,
                )
            except Exception as exc:
                logger.debug(
                    "[MoEWatch] Observation window check error at step %d: %s",
                    global_step, exc,
                )
        elif self.config.intervention_enabled and not self._intervention_gap_warned:
            # intervention_enabled=True but no intervention_engine/policy is
            # active. Under normal operation start() always initialises
            # these when intervention_enabled=True, so reaching this branch
            # indicates _init_intervention_system() failed or was skipped
            # unexpectedly. Without this warning, intervention_enabled=True
            # would silently have no effect.
            self._intervention_gap_warned = True
            logger.warning(
                "[MoEWatch] intervention_enabled=True but no intervention "
                "engine is active, so no interventions will be applied. "
                "This is unexpected — start() should initialise the "
                "intervention system whenever intervention_enabled=True. "
                "Risk monitoring and alerts continue to work normally."
            )

        # ---- 6. Emit periodic reports ----
        step_report = StepReport(
            step=global_step,
            timestamp=time.time(),
            risk_scores=step_risk_scores,
            risk_levels=step_risk_levels,
            active_interventions=applied_interventions,
            policy_decisions=policy_decisions,
            counterfactual_rewards=counterfactual_rewards,
            alerts=new_alerts,
            loss=current_loss,
            dominant_signals=dominant_signals,
        )

        watch_report = self.watch_report
        watch_report.append(step_report)

        if global_step % self.config.log_every == 0:
            self._emit_report(
                watch_report=watch_report,
                global_step=global_step,
                current_loss=current_loss,
                risk_scores=step_risk_scores,
                interventions=applied_interventions,
                alerts=new_alerts,
            )

        return step_report

    # ------------------------------------------------------------------
    # Alert generation helpers
    # ------------------------------------------------------------------

    def _generate_entropy_alerts(
        self,
        entropy_reports: dict,
        step: int,
    ) -> List[Alert]:
        """Generate alerts from entropy analysis results.

        Parameters
        ----------
        entropy_reports : dict[str, LayerEntropyReport]
            Per-layer entropy analysis results.
        step : int
            Current training step.

        Returns
        -------
        list[Alert]
            Alerts generated from entropy thresholds and drift detection.
        """
        alerts: List[Alert] = []
        for layer_name, report in entropy_reports.items():
            norm_ent = report.normalized_entropy

            if norm_ent < self.config.entropy_critical:
                level = AlertLevel.CRITICAL
                msg = (
                    f"Normalized entropy critically low ({norm_ent:.4f} < "
                    f"{self.config.entropy_critical}). Imminent collapse risk."
                )
            elif norm_ent < self.config.entropy_warn:
                level = AlertLevel.WARNING
                msg = (
                    f"Normalized entropy below warning threshold "
                    f"({norm_ent:.4f} < {self.config.entropy_warn}). "
                    f"Trend: {report.trend or 'unknown'}."
                )
            elif report.drift_detected:
                level = AlertLevel.WARNING
                msg = (
                    f"Entropy drift detected by CUSUM. "
                    f"Current normalized entropy: {norm_ent:.4f}. "
                    f"Trend: {report.trend or 'unknown'}."
                )
            else:
                continue

            alerts.append(
                Alert(
                    step=step,
                    level=level,
                    layer_id=layer_name,
                    signal_type="entropy_drift",
                    message=msg,
                    metrics={"normalized_entropy": norm_ent},
                )
            )
        return alerts

    def _generate_collapse_alerts(
        self,
        collapse_reports: dict,
        step: int,
    ) -> List[Alert]:
        """Generate alerts from expert health state machine results.

        Parameters
        ----------
        collapse_reports : dict[str, LayerCollapseReport]
            Per-layer collapse detection results.
        step : int
            Current training step.

        Returns
        -------
        list[Alert]
            Alerts for dead experts and high load imbalance.
        """
        alerts: List[Alert] = []
        for layer_name, report in collapse_reports.items():
            # Dead expert alerts
            if report.num_dead_experts > 0:
                alerts.append(
                    Alert(
                        step=step,
                        level=AlertLevel.CRITICAL,
                        layer_id=layer_name,
                        signal_type="expert_dead",
                        message=(
                            f"{report.num_dead_experts} expert(s) classified as DEAD "
                            f"in layer '{layer_name}'. Routing collapse in progress."
                        ),
                        metrics={
                            "num_dead_experts": float(report.num_dead_experts),
                            "num_cold_experts": float(report.num_cold_experts),
                            "load_imbalance_ratio": report.load_imbalance_ratio,
                        },
                    )
                )

            # Load imbalance alerts
            if report.load_imbalance_ratio >= self.config.load_imbalance_error:
                alerts.append(
                    Alert(
                        step=step,
                        level=AlertLevel.CRITICAL,
                        layer_id=layer_name,
                        signal_type="load_imbalance",
                        message=(
                            f"Critical load imbalance in '{layer_name}': "
                            f"ratio={report.load_imbalance_ratio:.2f} "
                            f"(threshold={self.config.load_imbalance_error})."
                        ),
                        metrics={"load_imbalance_ratio": report.load_imbalance_ratio},
                    )
                )
            elif report.load_imbalance_ratio >= self.config.load_imbalance_warn:
                alerts.append(
                    Alert(
                        step=step,
                        level=AlertLevel.WARNING,
                        layer_id=layer_name,
                        signal_type="load_imbalance",
                        message=(
                            f"Load imbalance warning in '{layer_name}': "
                            f"ratio={report.load_imbalance_ratio:.2f} "
                            f"(threshold={self.config.load_imbalance_warn})."
                        ),
                        metrics={"load_imbalance_ratio": report.load_imbalance_ratio},
                    )
                )
        return alerts

    def _generate_gradient_alerts(
        self,
        gradient_reports: dict,
        step: int,
    ) -> List[Alert]:
        """Generate Tier 1 gradient starvation alerts.

        Parameters
        ----------
        gradient_reports : dict[str, list[GradientStarvationReport]]
            Per-layer, per-expert gradient starvation reports.
        step : int
            Current training step.

        Returns
        -------
        list[Alert]
            Alerts for experts with detected gradient starvation.
        """
        alerts: List[Alert] = []
        for layer_name, expert_reports in gradient_reports.items():
            for report in expert_reports:
                if not report.starvation_detected:
                    continue
                level = (
                    AlertLevel.CRITICAL
                    if report.gradient_norm_mean < self.config.dead_threshold
                    else AlertLevel.WARNING
                )
                alerts.append(
                    Alert(
                        step=step,
                        level=level,
                        layer_id=layer_name,
                        signal_type="gradient_starvation",
                        message=(
                            f"[Tier 1] Expert {report.expert_id} in '{layer_name}' "
                            f"is gradient-starved. "
                            f"Norm mean: {report.gradient_norm_mean:.5f} "
                            f"(starvation_score={report.starvation_score:.3f}). "
                            f"Onset step: {report.starvation_onset_step}."
                        ),
                        metrics={
                            "gradient_norm_mean": report.gradient_norm_mean,
                            "starvation_score": report.starvation_score,
                            "expert_id": float(report.expert_id),
                        },
                    )
                )
        return alerts

    def _generate_risk_alerts(
        self,
        risk_reports: dict,
        step: int,
    ) -> List[Alert]:
        """Generate fused risk score alerts (Tier 1+2+3 fusion).

        Only emits a risk alert when the risk level exceeds WARNING,
        to avoid duplicating per-signal alerts at lower risk.

        Parameters
        ----------
        risk_reports : dict[str, RiskReport]
            Per-layer fused risk reports.
        step : int
            Current training step.

        Returns
        -------
        list[Alert]
            High-level risk score alerts.
        """
        from moewatch.analyzer.risk_score import RiskLevel

        alerts: List[Alert] = []
        for layer_name, report in risk_reports.items():
            if report.risk_level not in (RiskLevel.HIGH, RiskLevel.CRITICAL):
                continue
            level = (
                AlertLevel.CRITICAL
                if report.risk_level == RiskLevel.CRITICAL
                else AlertLevel.WARNING
            )
            alerts.append(
                Alert(
                    step=step,
                    level=level,
                    layer_id=layer_name,
                    signal_type="risk_score",
                    message=(
                        f"Risk score {report.risk_score:.3f} "
                        f"[{report.risk_level.name}] in '{layer_name}'. "
                        f"Dominant signal: {report.dominant_signal}. "
                        f"T1={report.tier1_contribution:.3f}, "
                        f"T2={report.tier2_contribution:.3f}, "
                        f"T3={report.tier3_contribution:.3f}."
                    ),
                    metrics={
                        "risk_score": report.risk_score,
                        "tier1_contribution": report.tier1_contribution,
                        "tier2_contribution": report.tier2_contribution,
                        "tier3_contribution": report.tier3_contribution,
                    },
                )
            )
        return alerts

    # ------------------------------------------------------------------
    # Intervention helpers
    # ------------------------------------------------------------------

    def _init_intervention_system(self) -> None:
        """Instantiate the intervention engine and policy.

        Interventions are applied directly to ``self.model`` (a
        ``torch.nn.Module``), so this can be called from ``start()``
        regardless of whether a HuggingFace Trainer is attached.
        """
        from moewatch.intervention.engine import InterventionEngine
        from moewatch.policy.rule_policy import RulePolicy
        from moewatch.policy.bandit_policy import BanditPolicy

        self.intervention_engine = InterventionEngine(
            config=self.config,
            model=self.model,
            baseline_tracker=self.baseline_tracker,
        )

        if self.config.policy_type == "bandit":
            self.policy = BanditPolicy(self.config)
            logger.info("[MoEWatch] Using BanditPolicy (Phase 2 learning).")
        else:
            self.policy = RulePolicy(self.config)
            logger.info("[MoEWatch] Using RulePolicy (Phase 1 deterministic).")

    def _run_interventions(
        self,
        step: int,
        risk_reports: dict,
        current_loss: float,
    ) -> Tuple[List, Dict[str, str]]:
        """Select and apply interventions for at-risk layers.

        For each layer that has a valid risk report, the policy selects
        an action, the SafetyGuard validates it, and the engine applies
        it. Returns the list of actions actually applied (including NoOps)
        and the per-layer policy decision map.

        Parameters
        ----------
        step : int
            Current training step.
        risk_reports : dict[str, RiskReport]
            Per-layer fused risk reports from RiskScoreFuser.
        current_loss : float
            Current training loss for SafetyGuard loss spike detection.

        Returns
        -------
        applied_interventions : list[InterventionAction]
            All interventions applied at this step (may include NoOps).
        policy_decisions : dict[str, str]
            Per-layer action_type strings from policy selection.
        """
        from moewatch.policy.base import PolicyState

        applied_interventions = []
        policy_decisions: Dict[str, str] = {}

        step_risk_scores = {ln: rr.risk_score for ln, rr in risk_reports.items()}

        for layer_name, risk_report in risk_reports.items():
            try:
                # Build policy state
                layer_id = self._layer_order.index(layer_name) if layer_name in self._layer_order else 0
                state = PolicyState(
                    risk_score=risk_report.risk_score,
                    layer_id=layer_id,
                    training_step=step,
                    intervention_history=self._get_intervention_history(layer_name),
                    dominant_signal=risk_report.dominant_signal,
                )

                # Policy selects candidate action
                action = self.policy.select_action(state)
                # Patch layer_name to the real dotted module path so
                # action.apply(model) can resolve get_submodule() correctly.
                action.layer_name = layer_name
                policy_decisions[layer_name] = action.action_type

                # Safety validation and potential downgrade
                validated_action = self.intervention_engine.propose_intervention(
                    action=action,
                    current_loss=current_loss,
                    risk_scores=step_risk_scores,
                    layer_order=self._layer_order,
                    step=step,
                )

                # Apply validated action (may be NoOp after safety downgrade)
                if validated_action.action_type != "noop":
                    self.intervention_engine.apply_intervention(validated_action, step)
                    applied_interventions.append(validated_action)
                    # Mark baseline as intervention-influenced
                    self.baseline_tracker.mark_intervention(
                        layer_name=layer_name,
                        start_step=step,
                    )
                    logger.info(
                        "[MoEWatch] Applied '%s' on '%s' at step %d "
                        "(risk=%.3f).",
                        validated_action.action_type,
                        layer_name,
                        step,
                        risk_report.risk_score,
                    )

            except Exception as exc:  # pylint: disable=broad-except
                logger.debug(
                    "[MoEWatch] Intervention error on layer '%s' at step %d: %s",
                    layer_name, step, exc,
                )

        return applied_interventions, policy_decisions

    def _get_intervention_history(self, layer_name: str) -> List[str]:
        """Return recent action names applied to a given layer.

        Scans ``self.alerts`` for intervention-related records.

        Parameters
        ----------
        layer_name : str
            Target layer name.

        Returns
        -------
        list[str]
            Recent action type strings in chronological order.
        """
        history = []
        if self.intervention_engine is None:
            return history
        # Retrieve from engine's internal log
        for entry in self.intervention_engine._intervention_log:
            if entry.get("layer_name") == layer_name:
                history.append(entry.get("action_type", "unknown"))
        return history[-10:]  # Return last 10 actions

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def _emit_report(
        self,
        watch_report: "WatchReport",
        global_step: int,
        current_loss: float,
        risk_scores: Dict[str, float],
        interventions: List,
        alerts: List[Alert],
    ) -> None:
        """Emit a monitoring report through the configured output channel.

        Routes to ``CLIReporter``, ``JSONReporter``, or is suppressed
        depending on ``config.output``.

        Parameters
        ----------
        watch_report : WatchReport
            The current step's WatchReport.
        global_step : int
            Current training step.
        current_loss : float
            Training loss at this step.
        risk_scores : dict[str, float]
            Per-layer risk scores.
        interventions : list
            Applied intervention actions.
        alerts : list[Alert]
            New alerts since last report.
        """
        if self.config.output == OutputMode.SILENT:
            return

        if self.config.output == OutputMode.CLI and self._cli_reporter is not None:
            try:
                output = self._cli_reporter.render_dashboard(
                    watch_report=watch_report,
                    entropy_analyzer=self.entropy_analyzer,
                    collapse_detector=self.collapse_detector,
                    risk_fuser=self.risk_fuser,
                )
                print(output, flush=True)
                for alert in alerts:
                    print(self._cli_reporter.render_alert(alert), flush=True)
            except Exception as exc:
                logger.debug("[MoEWatch] CLIReporter error: %s", exc)

        elif self.config.output == OutputMode.JSON and self._json_reporter is not None:
            try:
                self._json_reporter.write_step(
                    step=global_step,
                    timestamp=time.time(),
                    risk_scores=risk_scores,
                    loss=current_loss,
                    interventions=interventions,
                    alerts=alerts,
                )
            except Exception as exc:
                logger.debug("[MoEWatch] JSONReporter error: %s", exc)

    # ------------------------------------------------------------------
    # Model validation
    # ------------------------------------------------------------------

    def _validate_model(self) -> None:
        """Perform early validation that the model has detectable routers.

        Runs the auto-detection logic against the model using the current
        config. Raises ``ValueError`` if no routers are found and no manual
        override is provided in ``config.router_modules``.

        This check runs at ``__init__`` time so the user gets an
        informative error before training starts.

        Raises
        ------
        ValueError
            If no MoE router modules could be detected.
        """
        from moewatch.hooks.detection import detect_router_modules

        detected = detect_router_modules(self.model, self.config)
        if not detected:
            if self.config.router_modules:
                raise ValueError(
                    f"[MoEWatch] No router modules found matching "
                    f"config.router_modules={self.config.router_modules!r}. "
                    "Verify that the provided module names exist in the model."
                )
            raise ValueError(
                "[MoEWatch] No MoE router modules detected in model "
                f"'{type(self.model).__name__}'. "
                "Set config.router_modules=['<your_router_name>'] to override "
                "auto-detection, or ensure the model is a Mixture-of-Experts "
                "architecture (Mixtral, Qwen3-MoE, DeepSeek-MoE, OLMoE, etc.)."
            )
        logger.debug(
            "[MoEWatch] Model validation passed. Detected %d potential routers.",
            len(detected),
        )

    # ------------------------------------------------------------------
    # Public query interface
    # ------------------------------------------------------------------

    def get_alerts(self, since_step: int = 0) -> List[Alert]:
        """Retrieve alert history since a given training step.

        Thread-safe. Returns a snapshot of the alert list filtered to
        steps greater than or equal to ``since_step``.

        Parameters
        ----------
        since_step : int, optional
            Return only alerts emitted at this step or later. Default 0
            (returns all alerts).

        Returns
        -------
        list[Alert]
            Filtered alert history in chronological order.
        """
        with self._alert_lock:
            return [a for a in self.alerts if a.step >= since_step]

    def get_risk_summary(self) -> Dict[str, float]:
        """Return the most recent risk score for each monitored layer.

        Provides a quick high-level health summary without needing to
        inspect the full WatchReport.

        Returns
        -------
        dict[str, float]
            ``{layer_name: risk_score}`` for all detected layers.
            Returns empty dict if no analysis has been run yet.
        """
        if self.risk_fuser is None:
            return {}
        summary: Dict[str, float] = {}
        for layer in self._layer_order:
            report = self.risk_fuser.get_latest_score(layer)
            if report is not None:
                summary[layer] = report.risk_score
        return summary

    @property
    def is_running(self) -> bool:
        """True if monitoring is currently active."""
        return self._running

    @property
    def num_layers_monitored(self) -> int:
        """Number of MoE router layers currently under observation."""
        return len(self._layer_order)

    def __repr__(self) -> str:
        return (
            f"MoEWatch("
            f"model={type(self.model).__name__}, "
            f"running={self._running}, "
            f"step={self._global_step}, "
            f"layers={self.num_layers_monitored}, "
            f"alerts={len(self.alerts)}, "
            f"policy={self.config.policy_type!r}"
            f")"
        )
