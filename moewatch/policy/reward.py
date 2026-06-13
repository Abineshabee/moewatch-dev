# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# moewatch/policy/reward.py [v0.2.0]
# =============================================================================
#
# Project      : MoEWatch
# Version      : v0.2.0
# Description  : Computes the discounted counterfactual reward used to
#                 evaluate completed interventions.
#
#                     reward = sum_{k=1..K} ΔH_counterfactual(t+k) * gamma^k
#
#                 where ΔH_counterfactual(t+k) = entropy_actual(t+k) -
#                 entropy_baseline(t+k), entropy_baseline is supplied by
#                 :class:`~moewatch.collector.baseline_tracker.BaselineTracker`
#                 (projected from pre-intervention "clean" history), gamma
#                 is an exponential discount factor (default 0.95), and K is
#                 the observation window length (default 50 steps).
#
#                 A positive reward indicates the intervention improved
#                 routing entropy relative to what would have happened
#                 without it; a non-positive reward indicates no
#                 improvement (or a regression).
#
# Author       : Abinesh N (@Abineshabee)
# Repository   : https://github.com/Abineshabee/MoEWatch
# License      : Apache 2.0
#
# Contents
# --------
#   RewardComputer — discounted counterfactual reward computation
#
# Usage
# -----
#   from moewatch.policy.reward import RewardComputer
#
#   reward_fn = RewardComputer(config, baseline_tracker)
#   reward = reward_fn.compute_reward(
#       layer_name="model.layers.5.block_sparse_moe.gate",
#       entropy_values=entropy_window,
#       start_step=1200,
#       end_step=1250,
#   )
#
# =============================================================================

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance
    from moewatch.collector.baseline_tracker import BaselineTracker
    from moewatch.config import WatchConfig

logger = logging.getLogger(__name__)


class RewardComputer:
    """Computes discounted counterfactual rewards for interventions.

    .. math::

        \\text{reward} = \\sum_{k=1}^{K} \\Delta H_{\\text{cf}}(t+k)
        \\cdot \\gamma^{k}

    where :math:`\\Delta H_{\\text{cf}}(t+k) = H_{\\text{actual}}(t+k) -
    H_{\\text{baseline}}(t+k)`, :math:`\\gamma` is
    ``config.reward_discount_gamma``, and :math:`K` is
    ``config.reward_window_steps``.

    A positive reward means the intervention's observed entropy trajectory
    ran *above* the counterfactual "what would have happened without
    intervention" baseline — i.e. routing stability improved relative to
    the counterfactual. This reward is the training signal consumed by
    :meth:`~moewatch.policy.base.PolicyBase.update`.

    Parameters
    ----------
    config : WatchConfig
        Configuration providing ``reward_discount_gamma`` (the exponential
        discount factor :math:`\\gamma`, typically ``0.95``) and
        ``reward_window_steps`` (the observation window :math:`K`,
        typically ``50``).
    baseline_tracker : BaselineTracker
        Source of per-layer counterfactual baseline projections, via
        :meth:`~moewatch.collector.baseline_tracker.BaselineTracker.get_baseline`.

    Attributes
    ----------
    config : WatchConfig
        See above.
    baseline_tracker : BaselineTracker
        See above.

    Notes
    -----
    This class performs no mutation of :attr:`baseline_tracker` — it only
    reads baseline projections. Marking the intervention start step (so
    that the baseline excludes the post-intervention window) is the
    responsibility of
    :meth:`~moewatch.collector.baseline_tracker.BaselineTracker.mark_intervention`,
    called by
    :class:`~moewatch.intervention.engine.InterventionEngine` at the time
    the intervention is applied.
    """

    def __init__(self, config: "WatchConfig", baseline_tracker: "BaselineTracker") -> None:
        self.config: "WatchConfig" = config
        self.baseline_tracker: "BaselineTracker" = baseline_tracker

    def compute_reward(
        self,
        layer_name: str,
        entropy_values: Sequence[float],
        start_step: int,
        end_step: int,
    ) -> float:
        """Compute the discounted counterfactual reward over an observation window.

        Parameters
        ----------
        layer_name : str
            Name of the MoE layer the intervention targeted. Passed
            through to
            :meth:`~moewatch.collector.baseline_tracker.BaselineTracker.get_baseline`
            to retrieve the counterfactual baseline trajectory for this
            layer.
        entropy_values : Sequence[float]
            Observed (actual) routing entropy values for steps
            ``start_step + 1`` through ``start_step + K`` (inclusive),
            where ``K = config.reward_window_steps``. The sequence is
            indexed from ``0``, with ``entropy_values[k - 1]``
            corresponding to step ``start_step + k``. If shorter than
            ``K`` (e.g. the run ended before the window closed), only the
            available entries contribute to the sum.
        start_step : int
            Training step at which the intervention was applied (``k=0``
            reference point; not itself included in the sum).
        end_step : int
            Training step at which the observation window closed. Used
            only for logging/diagnostics; the effective window length is
            governed by ``min(len(entropy_values), config.reward_window_steps,
            end_step - start_step)``.

        Returns
        -------
        float
            The discounted counterfactual reward
            :math:`\\sum_{k} \\Delta H_{\\text{cf}}(t+k) \\cdot \\gamma^{k}`.
            Returns ``0.0`` if :paramref:`entropy_values` is empty or
            :paramref:`end_step` does not exceed :paramref:`start_step`.

        Notes
        -----
        For each step ``t + k`` where the baseline tracker has no valid
        baseline available (e.g. fewer than the minimum number of clean
        history points have been observed for :paramref:`layer_name`),
        the counterfactual delta for that step is treated as ``0.0``
        (neither rewarding nor penalising the policy for steps where no
        causal claim can be made), per the architecture's "handle missing
        baseline" guidance.
        """
        if not entropy_values:
            logger.debug(
                "[MoEWatch] RewardComputer: empty entropy_values for layer "
                "'%s' (start_step=%d); returning reward=0.0.",
                layer_name,
                start_step,
            )
            return 0.0

        if end_step <= start_step:
            logger.debug(
                "[MoEWatch] RewardComputer: end_step (%d) <= start_step "
                "(%d) for layer '%s'; returning reward=0.0.",
                end_step,
                start_step,
                layer_name,
            )
            return 0.0

        gamma = float(self.config.reward_discount_gamma)
        max_k = min(
            len(entropy_values),
            int(self.config.reward_window_steps),
            end_step - start_step,
        )

        reward = 0.0
        for k in range(1, max_k + 1):
            actual = float(entropy_values[k - 1])

            if self.baseline_tracker.is_baseline_valid(layer_name):
                baseline = self.baseline_tracker.get_baseline(
                    layer_name, start_step + k
                )
                delta_k = actual - baseline
            else:
                delta_k = 0.0

            reward += delta_k * (gamma ** k)

        logger.debug(
            "[MoEWatch] RewardComputer: layer='%s' start_step=%d "
            "end_step=%d window=%d reward=%.6f.",
            layer_name,
            start_step,
            end_step,
            max_k,
            reward,
        )

        return float(reward)
