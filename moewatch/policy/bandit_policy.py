# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# moewatch/policy/bandit_policy.py [v0.2.0]
# =============================================================================
#
# Project      : MoEWatch
# Version      : v0.2.0
# Description  : Phase 2 contextual bandit policy. Learns, within a single
#                 training run, which intervention actions are effective at
#                 which (risk level, layer, signal) contexts.
#
#                 State representation: PolicyState.state_key(), a coarse
#                 string of the form "L{layer_id}|R{risk_bucket}|{signal}".
#                 Action space: {"noop", "aux_loss", "router_noise",
#                 "expert_dropout"}. Algorithm: epsilon-greedy exploration
#                 over a tabular Q-value estimate, updated via incremental
#                 Q-learning:
#
#                     Q(s, a) <- Q(s, a) + alpha * (reward - Q(s, a))
#
#                 with learning rate alpha = 0.1. Every (state, action,
#                 reward) observation is also appended to a
#                 PolicyMemory replay buffer for diagnostics and
#                 cross-run transfer.
#
# Author       : Abinesh N (@Abineshabee)
# Repository   : https://github.com/Abineshabee/MoEWatch
# License      : Apache 2.0
#
# Contents
# --------
#   BanditPolicy — Phase 2 epsilon-greedy contextual bandit policy
#
# Usage
# -----
#   from moewatch.policy.bandit_policy import BanditPolicy
#
#   policy = BanditPolicy(config)
#   action = policy.select_action(state)
#   ...
#   policy.update(state, action, reward)
#   policy.save_checkpoint("bandit_policy.json")
#
# =============================================================================

from __future__ import annotations

import json
import logging
import random
from typing import TYPE_CHECKING, Dict, List

from moewatch.intervention.actions import (
    AuxLossAction,
    ExpertDropoutAction,
    InterventionAction,
    NoOpAction,
    RouterNoiseAction,
)
from moewatch.policy.base import PolicyBase, PolicyState
from moewatch.policy.memory import PolicyMemory

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance
    from moewatch.config import WatchConfig

logger = logging.getLogger(__name__)


#: Action names in the bandit's fixed action space, in a stable order used
#: for Q-table initialisation and tie-breaking.
_ACTION_SPACE: tuple[str, ...] = ("noop", "aux_loss", "router_noise", "expert_dropout")

#: Q-learning step size (alpha). Each update moves Q(s, a) a fixed
#: fraction of the way toward the newly observed reward.
_LEARNING_RATE = 0.1

#: Default capacity of the experience replay buffer.
_DEFAULT_MEMORY_SIZE = 10000


class BanditPolicy(PolicyBase):
    """Phase 2 epsilon-greedy contextual bandit intervention policy.

    Learns, within a single training run, which intervention action tends
    to produce the highest reward (discounted counterfactual entropy
    improvement, see
    :class:`~moewatch.policy.reward.RewardComputer`) for a given coarse
    state context (risk bucket, layer, dominant signal).

    State
        :meth:`PolicyState.state_key` — a string of the form
        ``"L{layer_id}|R{risk_bucket}|{signal}"`` where ``risk_bucket``
        is ``risk_score`` discretised into tenths.
    Action space
        ``{"noop", "aux_loss", "router_noise", "expert_dropout"}``.
    Algorithm
        Epsilon-greedy action selection over a tabular ``Q[state][action]``
        estimate, with incremental Q-learning updates
        ``Q(s, a) <- Q(s, a) + alpha * (reward - Q(s, a))``,
        ``alpha = 0.1``.

    Parameters
    ----------
    config : WatchConfig
        Configuration providing ``bandit_epsilon`` (exploration rate,
        default ``0.1``). All other bandit hyperparameters
        (``alpha = 0.1``) are fixed module constants per the architecture
        specification.

    Attributes
    ----------
    config : WatchConfig
        See above.
    memory : PolicyMemory
        Experience replay buffer recording every ``update()`` call.

    Notes
    -----
    Epsilon is read fresh from ``self.config.bandit_epsilon`` on every
    call to :meth:`select_action`, so external code may adjust
    ``config.bandit_epsilon`` between calls (e.g. to implement
    epsilon-decay) and the change takes effect immediately without
    re-constructing the policy.
    """

    def __init__(self, config: "WatchConfig") -> None:
        self.config: "WatchConfig" = config
        self.memory: PolicyMemory = PolicyMemory(max_size=_DEFAULT_MEMORY_SIZE)

        # state_key -> action_name -> estimated Q-value.
        self._q_values: Dict[str, Dict[str, float]] = {}

        # state_key -> action_name -> visit count.
        self._action_counts: Dict[str, Dict[str, int]] = {}

        # Internal step counter, incremented on every select_action() call.
        # Reserved for future exploration-schedule use (e.g. epsilon
        # decay keyed on this counter); currently exposed for diagnostics.
        self._step: int = 0

    # ------------------------------------------------------------------
    # PolicyBase interface
    # ------------------------------------------------------------------

    def select_action(self, state: PolicyState) -> InterventionAction:
        """Select an action via epsilon-greedy exploration.

        Parameters
        ----------
        state : PolicyState
            Current per-layer state.

        Returns
        -------
        InterventionAction
            The selected action, targeting ``layer_name = f"layer_{state.layer_id}"``.

        Notes
        -----
        Algorithm:

        1. Compute ``state_key = state.state_key()``.
        2. Ensure a Q-value entry exists for every action in
           :data:`_ACTION_SPACE` under ``state_key`` (initialised to
           ``0.0`` on first visit).
        3. With probability ``config.bandit_epsilon``, **explore**: select
           a uniformly random action from :data:`_ACTION_SPACE`.
        4. Otherwise, **exploit**: select the action with the highest
           Q-value for ``state_key``. Ties are broken by
           :data:`_ACTION_SPACE` order (i.e. preferring the
           weaker/earlier-listed action, which biases ties toward more
           conservative interventions).
        5. Increment the visit count for ``(state_key, action)``.
        6. Construct and return the corresponding
           :class:`~moewatch.intervention.actions.InterventionAction`.

        The internal step counter :attr:`_step` is incremented on every
        call, regardless of whether exploration or exploitation occurred.
        """
        self._step += 1

        state_key = state.state_key()
        q_for_state = self._q_values.setdefault(
            state_key, {a: 0.0 for a in _ACTION_SPACE}
        )
        counts_for_state = self._action_counts.setdefault(
            state_key, {a: 0 for a in _ACTION_SPACE}
        )

        epsilon = float(self.config.bandit_epsilon)

        if random.random() < epsilon:
            action_name = random.choice(_ACTION_SPACE)
            mode = "explore"
        else:
            action_name = self._argmax_action(q_for_state)
            mode = "exploit"

        counts_for_state[action_name] = counts_for_state.get(action_name, 0) + 1

        layer_name = f"layer_{state.layer_id}"

        logger.debug(
            "[MoEWatch] BanditPolicy: step=%d state_key='%s' mode=%s "
            "action='%s' q=%.6f epsilon=%.4f.",
            self._step,
            state_key,
            mode,
            action_name,
            q_for_state[action_name],
            epsilon,
        )

        return self._build_action(action_name, layer_name)

    def update(
        self, state: PolicyState, action: InterventionAction, reward: float
    ) -> None:
        """Update Q-values and replay memory with an observed outcome.

        Parameters
        ----------
        state : PolicyState
            The state at the time :paramref:`action` was selected.
        action : InterventionAction
            The action that was selected for :paramref:`state`.
        reward : float
            Observed discounted counterfactual reward.

        Returns
        -------
        None

        Notes
        -----
        Side effects:

        - Appends ``(state, action.action_type, reward, state.training_step)``
          to :attr:`memory`.
        - Updates
          ``Q(state_key, action_type) <- Q(state_key, action_type) +
          alpha * (reward - Q(state_key, action_type))`` with
          ``alpha = 0.1``.
        - Increments the visit count for ``(state_key, action_type)`` if
          not already incremented by a corresponding
          :meth:`select_action` call (this keeps the count consistent even
          if :meth:`update` is called for an action chosen by a different
          policy instance, e.g. when replaying loaded memory).
        """
        state_key = state.state_key()
        action_name = action.action_type

        q_for_state = self._q_values.setdefault(
            state_key, {a: 0.0 for a in _ACTION_SPACE}
        )
        counts_for_state = self._action_counts.setdefault(
            state_key, {a: 0 for a in _ACTION_SPACE}
        )

        if action_name not in q_for_state:
            q_for_state[action_name] = 0.0
        if action_name not in counts_for_state:
            counts_for_state[action_name] = 0

        old_q = q_for_state[action_name]
        new_q = old_q + _LEARNING_RATE * (reward - old_q)
        q_for_state[action_name] = new_q

        self.memory.append(
            state=state,
            action=action_name,
            reward=reward,
            step=state.training_step,
        )

        logger.debug(
            "[MoEWatch] BanditPolicy.update: state_key='%s' action='%s' "
            "reward=%.6f Q: %.6f -> %.6f.",
            state_key,
            action_name,
            reward,
            old_q,
            new_q,
        )

    def save_checkpoint(self, path: str) -> None:
        """Save Q-values, visit counts, and replay memory to ``path`` as JSON.

        Parameters
        ----------
        path : str
            Destination file path.

        Returns
        -------
        None

        Notes
        -----
        The output schema is::

            {
                "q_values": {state_key: {action: q, ...}, ...},
                "action_counts": {state_key: {action: count, ...}, ...},
                "step": <int>,
                "memory": {"max_size": <int>, "experiences": [...]}
            }

        The ``"memory"`` block matches
        :meth:`~moewatch.policy.memory.PolicyMemory.to_json`'s schema,
        embedded inline rather than written to a separate file, so a
        single checkpoint file fully captures the policy's learned state.
        """
        # Build the memory payload using the same shape PolicyMemory.to_json
        # would produce, without requiring a second file.
        memory_payload = {
            "max_size": self.memory.max_size,
            "experiences": [
                exp.to_dict() for exp in self.memory.get_batch(len(self.memory))
            ],
        }

        payload = {
            "q_values": self._q_values,
            "action_counts": self._action_counts,
            "step": self._step,
            "memory": memory_payload,
        }

        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        except OSError as exc:
            logger.warning(
                "[MoEWatch] BanditPolicy: failed to write checkpoint to "
                "'%s' (%s).",
                path,
                exc,
            )
            raise

        logger.info(
            "[MoEWatch] BanditPolicy: saved checkpoint to '%s' "
            "(%d state(s), %d experience(s)).",
            path,
            len(self._q_values),
            len(memory_payload["experiences"]),
        )

    def load_checkpoint(self, path: str) -> None:
        """Load Q-values, visit counts, and replay memory from ``path``.

        Parameters
        ----------
        path : str
            Source file path, previously written by
            :meth:`save_checkpoint`.

        Returns
        -------
        None

        Notes
        -----
        If :paramref:`path` does not exist or contains invalid JSON, logs
        a warning and leaves the current (empty) Q-table and memory
        unchanged — a missing checkpoint at the start of a run is treated
        as "start with no prior learning" rather than an error.

        The embedded ``"memory"`` block is loaded into :attr:`memory` via
        a temporary JSON round trip
        (:meth:`~moewatch.policy.memory.PolicyMemory.from_json` reads
        from a file path), keeping
        :class:`~moewatch.policy.memory.PolicyMemory`'s file-based
        interface as the single source of truth for memory
        (de)serialisation.
        """
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "[MoEWatch] BanditPolicy: failed to load checkpoint from "
                "'%s' (%s); starting with empty policy state.",
                path,
                exc,
            )
            return

        self._q_values = {
            state_key: dict(actions)
            for state_key, actions in payload.get("q_values", {}).items()
        }
        self._action_counts = {
            state_key: dict(actions)
            for state_key, actions in payload.get("action_counts", {}).items()
        }
        self._step = int(payload.get("step", 0))

        memory_payload = payload.get("memory")
        if memory_payload is not None:
            new_memory = PolicyMemory(
                max_size=int(memory_payload.get("max_size", _DEFAULT_MEMORY_SIZE))
            )
            self._load_memory_experiences(new_memory, memory_payload.get("experiences", []))
            self.memory = new_memory

        logger.info(
            "[MoEWatch] BanditPolicy: loaded checkpoint from '%s' "
            "(%d state(s), step=%d).",
            path,
            len(self._q_values),
            self._step,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _argmax_action(q_for_state: Dict[str, float]) -> str:
        """Return the action with the highest Q-value, breaking ties by order.

        Parameters
        ----------
        q_for_state : dict[str, float]
            Mapping of action name to Q-value for a single state.

        Returns
        -------
        str
            The action name from :data:`_ACTION_SPACE` (in order) with
            the maximum Q-value. Ties favour the earlier-listed (weaker)
            action, biasing toward conservative interventions when the
            policy is indifferent.
        """
        best_action = _ACTION_SPACE[0]
        best_q = q_for_state.get(best_action, 0.0)
        for action in _ACTION_SPACE[1:]:
            q = q_for_state.get(action, 0.0)
            if q > best_q:
                best_q = q
                best_action = action
        return best_action

    @staticmethod
    def _load_memory_experiences(memory: PolicyMemory, experiences: List[dict]) -> None:
        """Populate ``memory`` from a list of serialised experience dicts.

        Parameters
        ----------
        memory : PolicyMemory
            Freshly constructed, empty memory buffer to populate.
        experiences : list[dict]
            List of dicts in the schema produced by
            :meth:`~moewatch.policy.memory.ExperienceTuple.to_dict`.

        Returns
        -------
        None

        Notes
        -----
        Malformed individual entries are skipped with a warning rather
        than aborting the whole load, matching
        :meth:`~moewatch.policy.memory.PolicyMemory.from_json`'s
        tolerance for partially-corrupt memory files.
        """
        from moewatch.policy.memory import ExperienceTuple

        for item in experiences:
            try:
                exp = ExperienceTuple.from_dict(item)
            except (TypeError, ValueError, KeyError) as exc:
                logger.warning(
                    "[MoEWatch] BanditPolicy: skipping malformed memory "
                    "entry while loading checkpoint (%s).",
                    exc,
                )
                continue
            memory.append(
                state=exp.state,
                action=exp.action,
                reward=exp.reward,
                step=exp.step,
            )

    @staticmethod
    def _build_action(action_name: str, layer_name: str) -> InterventionAction:
        """Construct the :class:`InterventionAction` for ``action_name``.

        Parameters
        ----------
        action_name : str
            One of ``"noop"``, ``"aux_loss"``, ``"router_noise"``,
            ``"expert_dropout"``.
        layer_name : str
            Target layer name passed to the action's constructor.

        Returns
        -------
        InterventionAction
            A freshly constructed action instance with default magnitude
            parameters.
        """
        if action_name == "aux_loss":
            return AuxLossAction(layer_name=layer_name)
        if action_name == "router_noise":
            return RouterNoiseAction(layer_name=layer_name)
        if action_name == "expert_dropout":
            return ExpertDropoutAction(layer_name=layer_name)
        return NoOpAction(layer_name=layer_name)
