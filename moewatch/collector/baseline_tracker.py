# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# moewatch/collector/baseline_tracker.py [v0.2.0]
# =============================================================================
#
# Project      : MoEWatch
# Version      : v0.2.0
# Description  : Maintains an intervention-conditioned counterfactual
#                 baseline trajectory per layer. The baseline is a rolling
#                 linear regression fit ONLY on "clean" signal history —
#                 steps that fall outside any marked intervention-exclusion
#                 window for that layer.
#
#                 This is the core mechanism behind correct causal
#                 attribution of intervention effects:
#
#                   ΔH_real          = H_after - H_before            (naive, wrong)
#                   ΔH_counterfactual = H_after - H_baseline(t)       (correct)
#
#                 After any intervention on layer L at step t0, the steps
#                 [t0, t0 + baseline_exclusion_window) are marked as
#                 "intervention-influenced" and excluded from the clean
#                 history used to fit the regression. This prevents the
#                 "baseline illusion" problem, where a past intervention's
#                 effect leaks into the baseline used to evaluate a future
#                 intervention.
#
# Author       : Abinesh N (@Abineshabee)
# Repository   : https://github.com/Abineshabee/MoEWatch
# License      : Apache 2.0
#
# Contents
# --------
#   BaselineTracker — per-layer counterfactual baseline estimator
#
# Usage
# -----
#   from moewatch.collector.baseline_tracker import BaselineTracker
#
#   tracker = BaselineTracker(config)
#   tracker.register_layer("layers.5.moe_router")
#
#   tracker.update_signal("layers.5.moe_router", value=0.82, step=100)
#   ...
#   tracker.mark_intervention("layers.5.moe_router", start_step=150)
#   ...
#   delta = tracker.compute_counterfactual_delta("layers.5.moe_router", 0.91)
#
# =============================================================================

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Tuple

import numpy as np

from moewatch.config import WatchConfig

logger = logging.getLogger(__name__)


class BaselineTracker:
    """Computes intervention-conditioned counterfactual baselines per layer.

    For each registered layer, maintains:

    - A "clean history" of ``(step, value)`` pairs — signal observations
      (e.g. normalized entropy) recorded at steps that do **not** fall
      inside any active intervention-exclusion window for that layer.
    - A list of intervention-exclusion windows
      ``(start_step, end_step)``, each spanning
      ``config.baseline_exclusion_window`` steps from the step an
      intervention was applied.

    The baseline trajectory is a simple linear model
    ``y_baseline(t) = a * t + b`` fit via ordinary least squares on the
    clean history. :meth:`compute_counterfactual_delta` projects this
    model forward to the current step and returns
    ``actual_value - y_baseline(current_step)``.

    A minimum number of clean history points
    (``config.baseline_min_clean_steps``) is required before the baseline
    is considered valid; see :meth:`is_baseline_valid`.

    Parameters
    ----------
    config : WatchConfig
        Shared configuration. Uses ``config.baseline_min_clean_steps``
        and ``config.baseline_exclusion_window``.

    Attributes
    ----------
    config : WatchConfig
        Reference to the shared configuration.

    Notes
    -----
    All public methods are thread-safe via a single internal
    :class:`threading.Lock`. Per-layer history lists are bounded to
    :data:`_MAX_HISTORY_LENGTH` entries (oldest dropped first) to keep
    memory usage bounded over long training runs, mirroring the
    :class:`~moewatch.collector.ring_buffer.RingBuffer` approach used
    elsewhere in MoEWatch.
    """

    # Maximum number of (step, value) pairs retained per layer, across
    # both clean history and the raw step counter used for exclusion-
    # window pruning. Bounds memory for very long training runs.
    _MAX_HISTORY_LENGTH: int = 2000

    def __init__(self, config: WatchConfig) -> None:
        self.config: WatchConfig = config

        # layer_name -> list of (step, value) pairs, clean history only.
        self._clean_history: Dict[str, List[Tuple[int, float]]] = {}

        # layer_name -> list of (start_step, end_step) exclusion windows.
        self._intervention_windows: Dict[str, List[Tuple[int, int]]] = {}

        # layer_name -> most recent step seen via update_signal().
        self._current_step: Dict[str, int] = {}

        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_layer(self, layer_name: str) -> None:
        """Initialize baseline tracking state for a layer.

        Idempotent: calling this for an already-registered layer is a
        no-op that preserves existing history and exclusion windows.

        Parameters
        ----------
        layer_name : str
            Router module name to begin tracking.

        Returns
        -------
        None

        Notes
        -----
        Thread-safe.
        """
        with self._lock:
            self._clean_history.setdefault(layer_name, [])
            self._intervention_windows.setdefault(layer_name, [])
            self._current_step.setdefault(layer_name, 0)

        logger.debug(
            "[MoEWatch] BaselineTracker: registered layer '%s'.", layer_name
        )

    # ------------------------------------------------------------------
    # Intervention marking
    # ------------------------------------------------------------------

    def mark_intervention(self, layer_name: str, start_step: int) -> None:
        """Mark a window of steps as intervention-influenced.

        Records the half-open interval
        ``[start_step, start_step + config.baseline_exclusion_window)``
        as excluded from this layer's clean history. Any
        :meth:`update_signal` calls for steps falling inside this
        interval will be skipped (not added to clean history) — this is
        the core "baseline illusion" prevention mechanism.

        Parameters
        ----------
        layer_name : str
            Layer affected by the intervention. Auto-registered (via
            :meth:`register_layer`) if not already known.
        start_step : int
            Training step at which the intervention was applied.

        Returns
        -------
        None

        Notes
        -----
        Thread-safe. Multiple overlapping or sequential exclusion windows
        for the same layer are all retained (no merging) — overlap is
        harmless since :meth:`update_signal` only needs to check
        membership in *any* window. Calling this method does not
        retroactively remove already-recorded clean history points that
        happen to fall inside the new window (those points were recorded
        as clean *before* this intervention was known, which is the
        correct causal ordering: only future observations are affected by
        an intervention applied now).
        """
        if layer_name not in self._intervention_windows:
            self.register_layer(layer_name)

        exclusion_window = max(0, self.config.baseline_exclusion_window)
        end_step = start_step + exclusion_window

        with self._lock:
            self._intervention_windows[layer_name].append((start_step, end_step))

            # Bound the number of tracked windows to avoid unbounded
            # growth over very long runs with frequent interventions.
            windows = self._intervention_windows[layer_name]
            if len(windows) > self._MAX_HISTORY_LENGTH:
                self._intervention_windows[layer_name] = windows[
                    -self._MAX_HISTORY_LENGTH :
                ]

        logger.debug(
            "[MoEWatch] BaselineTracker: marked intervention on '%s' for "
            "steps [%d, %d).",
            layer_name,
            start_step,
            end_step,
        )

    # ------------------------------------------------------------------
    # Signal updates
    # ------------------------------------------------------------------

    def update_signal(self, layer_name: str, value: float, step: int) -> None:
        """Record a new signal observation, if the step is "clean".

        Parameters
        ----------
        layer_name : str
            Layer this observation belongs to. Auto-registered (via
            :meth:`register_layer`) if not already known.
        value : float
            Signal value at ``step`` (e.g. normalized entropy, gradient
            norm, or any scalar metric the baseline is being computed
            for).
        step : int
            Training step of this observation.

        Returns
        -------
        None

        Notes
        -----
        If ``step`` falls inside any exclusion window recorded via
        :meth:`mark_intervention` for ``layer_name``, the observation is
        **not** added to clean history (it is still tracked as the
        current step via :attr:`_current_step`, used for projection in
        :meth:`compute_counterfactual_delta`).

        Clean history is bounded to :data:`_MAX_HISTORY_LENGTH` entries
        (oldest dropped first). Thread-safe.
        """
        if layer_name not in self._clean_history:
            self.register_layer(layer_name)

        with self._lock:
            self._current_step[layer_name] = step

            if self._is_excluded(layer_name, step):
                return

            history = self._clean_history[layer_name]
            history.append((step, value))

            if len(history) > self._MAX_HISTORY_LENGTH:
                self._clean_history[layer_name] = history[
                    -self._MAX_HISTORY_LENGTH :
                ]

    # ------------------------------------------------------------------
    # Baseline computation
    # ------------------------------------------------------------------

    def compute_counterfactual_delta(
        self, layer_name: str, actual_value: float
    ) -> float:
        """Compute the counterfactual delta for ``actual_value``.

        Fits ``y = a * t + b`` via ordinary least squares on this layer's
        clean history, projects the fit to the layer's most recently
        observed step (via :attr:`_current_step`, set by
        :meth:`update_signal`), and returns
        ``actual_value - y_baseline(current_step)``.

        Parameters
        ----------
        layer_name : str
            Layer to compute the counterfactual delta for. Must have
            been registered (directly or via :meth:`update_signal`).
        actual_value : float
            The observed signal value to compare against the projected
            baseline.

        Returns
        -------
        float
            ``actual_value - baseline_projected_value``. A positive
            value indicates ``actual_value`` exceeds what the baseline
            (no-intervention) trajectory would predict — i.e. an
            improvement, when the underlying signal is "higher is
            healthier" (e.g. normalized entropy). Callers are responsible
            for interpreting sign conventions appropriate to their
            specific signal.

        Raises
        ------
        ValueError
            If :meth:`is_baseline_valid` is ``False`` for ``layer_name``
            (insufficient clean history to fit a regression).
        KeyError
            If ``layer_name`` has never been registered.

        Notes
        -----
        Regression is performed via :func:`numpy.polyfit` with degree 1.
        If the clean history contains a single distinct step value
        (degenerate, zero-variance ``t``), :func:`numpy.polyfit` may emit
        a ``RankWarning``; in that degenerate case this method instead
        falls back to a flat baseline equal to the mean of the clean
        history's values (slope ``a = 0``).
        """
        if layer_name not in self._clean_history:
            raise KeyError(
                f"[MoEWatch] BaselineTracker.compute_counterfactual_delta(): "
                f"layer '{layer_name}' is not registered."
            )

        if not self.is_baseline_valid(layer_name):
            raise ValueError(
                f"[MoEWatch] BaselineTracker.compute_counterfactual_delta(): "
                f"insufficient clean history for layer '{layer_name}' "
                f"(need >= {self.config.baseline_min_clean_steps} clean "
                f"steps)."
            )

        with self._lock:
            history = list(self._clean_history[layer_name])
            current_step = self._current_step.get(layer_name, 0)

        baseline_value = self._project_baseline(history, current_step)
        return actual_value - baseline_value

    def is_baseline_valid(self, layer_name: str) -> bool:
        """Check whether a layer has enough clean history for a baseline.

        Parameters
        ----------
        layer_name : str
            Layer to check.

        Returns
        -------
        bool
            ``True`` if the number of clean ``(step, value)`` pairs
            recorded for ``layer_name`` is ``>= config.baseline_min_clean_steps``.
            ``False`` if ``layer_name`` is unregistered or has
            insufficient clean history.

        Notes
        -----
        Thread-safe.
        """
        with self._lock:
            history = self._clean_history.get(layer_name)

        if history is None:
            return False

        return len(history) >= max(1, self.config.baseline_min_clean_steps)

    def get_baseline(self, layer_name: str, step: int) -> float:
        """Project the baseline trajectory to an arbitrary step.

        Unlike :meth:`compute_counterfactual_delta`, which projects to the
        layer's most recently observed step, this method allows projecting
        to any ``step`` — used by
        :class:`~moewatch.policy.reward.RewardComputer` to evaluate the
        baseline at each step ``k`` within a post-intervention observation
        window.

        Parameters
        ----------
        layer_name : str
            Layer to project the baseline for.
        step : int
            Training step to project the baseline trajectory to.

        Returns
        -------
        float
            ``a * step + b`` from the layer's current regression fit. If
            :meth:`is_baseline_valid` is ``False`` for ``layer_name``,
            returns ``0.0`` (no baseline available; callers such as
            :class:`RewardComputer` are documented to treat this as "no
            counterfactual correction").

        Notes
        -----
        Thread-safe.
        """
        if not self.is_baseline_valid(layer_name):
            return 0.0

        with self._lock:
            history = list(self._clean_history.get(layer_name, []))

        return self._project_baseline(history, step)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_excluded(self, layer_name: str, step: int) -> bool:
        """Check whether ``step`` falls inside any exclusion window.

        Parameters
        ----------
        layer_name : str
            Layer to check exclusion windows for.
        step : int
            Training step to test.

        Returns
        -------
        bool
            ``True`` if ``step`` is contained in any
            ``[start_step, end_step)`` window recorded for
            ``layer_name`` via :meth:`mark_intervention`.

        Notes
        -----
        Caller must already hold :attr:`_lock`. Not thread-safe on its
        own.
        """
        windows = self._intervention_windows.get(layer_name, [])
        return any(start <= step < end for start, end in windows)

    @staticmethod
    def _project_baseline(
        history: List[Tuple[int, float]], step: int
    ) -> float:
        """Fit a linear baseline on ``history`` and project to ``step``.

        Parameters
        ----------
        history : list[tuple[int, float]]
            Non-empty list of ``(step, value)`` clean-history pairs.
        step : int
            Step to project the fitted line to.

        Returns
        -------
        float
            ``a * step + b`` where ``(a, b)`` is the least-squares fit of
            ``value = a * step + b`` over ``history``. If all steps in
            ``history`` are identical (zero-variance independent
            variable, which would make ``polyfit`` ill-conditioned),
            returns the mean of the ``value`` entries instead (flat
            baseline, slope ``0``).

        Notes
        -----
        Static method; does not touch any shared state.
        """
        steps = np.array([s for s, _ in history], dtype=np.float64)
        values = np.array([v for _, v in history], dtype=np.float64)

        if np.allclose(steps, steps[0]):
            return float(values.mean())

        slope, intercept = np.polyfit(steps, values, deg=1)
        return float(slope * step + intercept)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        with self._lock:
            n_layers = len(self._clean_history)
            n_windows = sum(
                len(w) for w in self._intervention_windows.values()
            )
        return (
            f"BaselineTracker(layers={n_layers}, "
            f"active_exclusion_windows={n_windows}, "
            f"min_clean_steps={self.config.baseline_min_clean_steps})"
        )
