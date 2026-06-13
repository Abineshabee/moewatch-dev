# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# moewatch/policy/memory.py [v0.2.0]
# =============================================================================
#
# Project      : MoEWatch
# Version      : v0.2.0
# Description  : Experience replay buffer for intervention policies. Stores
#                 (state, action, reward, step) tuples observed over a
#                 training run in a fixed-size circular buffer, and exposes
#                 simple statistics — per-action success rate and
#                 context-similarity lookup — used by
#                 :class:`~moewatch.policy.bandit_policy.BanditPolicy` for
#                 exploration priors and diagnostics.
#
#                 Serialisable to and from JSON (not pickle) so that memory
#                 collected in one training run can be inspected, audited,
#                 or transferred to warm-start a policy in a later run.
#
# Author       : Abinesh N (@Abineshabee)
# Repository   : https://github.com/Abineshabee/MoEWatch
# License      : Apache 2.0
#
# Contents
# --------
#   ExperienceTuple — single (state, action, reward, step) record
#   PolicyMemory     — circular replay buffer over ExperienceTuple
#
# Usage
# -----
#   from moewatch.policy.memory import PolicyMemory
#
#   memory = PolicyMemory(max_size=10000)
#   memory.append(state, action="aux_loss", reward=0.42, step=1200)
#   rate = memory.action_success_rate("aux_loss")
#   similar = memory.context_similarity(state)
#   memory.to_json("policy_memory.json")
#
# =============================================================================

from __future__ import annotations

import json
import logging
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List

from moewatch.policy.base import PolicyState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ExperienceTuple
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperienceTuple:
    """A single recorded (state, action, reward, step) experience.

    Parameters
    ----------
    state : PolicyState
        System state at the time :attr:`action` was selected.
    action : str
        Action name (e.g. ``"aux_loss"``, ``"router_noise"``,
        ``"expert_dropout"``, ``"no_op"``).
    reward : float
        Discounted counterfactual reward observed for this
        (state, action) pair.
    step : int
        Training step at which this experience was recorded.

    Notes
    -----
    Immutable: once appended to :class:`PolicyMemory`, an
    :class:`ExperienceTuple` is never mutated in place.
    """

    state: PolicyState
    action: str
    reward: float
    step: int

    def to_dict(self) -> dict:
        """Return a JSON-serialisable ``dict`` representation.

        Returns
        -------
        dict
            Mapping with keys ``"state"`` (itself a dict, via
            :meth:`PolicyState.to_dict`), ``"action"``, ``"reward"``, and
            ``"step"``.
        """
        return {
            "state": self.state.to_dict(),
            "action": self.action,
            "reward": self.reward,
            "step": self.step,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperienceTuple":
        """Reconstruct an :class:`ExperienceTuple` from :meth:`to_dict` output.

        Parameters
        ----------
        data : dict
            Mapping previously produced by :meth:`to_dict`.

        Returns
        -------
        ExperienceTuple
            Reconstructed instance.
        """
        return cls(
            state=PolicyState.from_dict(data.get("state", {})),
            action=str(data.get("action", "no_op")),
            reward=float(data.get("reward", 0.0)),
            step=int(data.get("step", 0)),
        )


# ---------------------------------------------------------------------------
# PolicyMemory
# ---------------------------------------------------------------------------


class PolicyMemory:
    """Circular experience replay buffer for intervention policies.

    Stores :class:`ExperienceTuple` records up to :attr:`max_size`,
    dropping the oldest entry once full. Maintains a per-action reward
    history for :meth:`action_success_rate`, and supports lightweight
    context-similarity lookup via :meth:`context_similarity`.

    Parameters
    ----------
    max_size : int, default=10000
        Maximum number of experience tuples retained. Must be a positive
        integer. Once exceeded, the oldest tuple is dropped for each new
        append (FIFO).

    Attributes
    ----------
    max_size : int
        See above.

    Notes
    -----
    Serialisation (:meth:`to_json` / :meth:`from_json`) uses JSON rather
    than ``pickle`` so that memory files are portable across Python
    versions, human-inspectable, and safe to load from untrusted sources
    (unlike pickle, which can execute arbitrary code on load).
    """

    def __init__(self, max_size: int = 10000) -> None:
        if max_size <= 0:
            raise ValueError(
                f"[MoEWatch] PolicyMemory: 'max_size' must be a positive "
                f"integer, got {max_size}."
            )

        self.max_size: int = int(max_size)
        self._buffer: Deque[ExperienceTuple] = deque(maxlen=self.max_size)
        self._action_rewards: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def append(
        self, state: PolicyState, action: str, reward: float, step: int
    ) -> None:
        """Add an experience tuple to the buffer.

        Parameters
        ----------
        state : PolicyState
            State at the time :paramref:`action` was selected.
        action : str
            Action name.
        reward : float
            Observed reward.
        step : int
            Training step.

        Returns
        -------
        None

        Notes
        -----
        If the buffer is at :attr:`max_size`, the oldest entry is dropped
        automatically by the underlying ``collections.deque(maxlen=...)``.
        The per-action reward history used by
        :meth:`action_success_rate` is *not* truncated when the buffer
        drops old tuples — it accumulates for the lifetime of this
        :class:`PolicyMemory` instance, since action statistics are
        intended to summarise the whole run rather than only the most
        recent window. Callers wanting a windowed success rate should
        instead iterate :attr:`_buffer` directly or construct a fresh
        :class:`PolicyMemory` periodically.
        """
        tup = ExperienceTuple(
            state=state, action=str(action), reward=float(reward), step=int(step)
        )
        self._buffer.append(tup)
        self._action_rewards.setdefault(tup.action, []).append(tup.reward)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def get_batch(self, batch_size: int) -> List[ExperienceTuple]:
        """Sample a random batch of experiences for replay.

        Parameters
        ----------
        batch_size : int
            Number of experiences to sample. If larger than the number of
            stored experiences, the entire buffer is returned (sampling
            without replacement is capped at the buffer size).

        Returns
        -------
        list[ExperienceTuple]
            Random sample of experiences, with length
            ``min(batch_size, len(self))``. Returns an empty list if the
            buffer is empty or ``batch_size <= 0``.
        """
        if batch_size <= 0 or not self._buffer:
            return []

        n = min(batch_size, len(self._buffer))
        return random.sample(list(self._buffer), n)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def action_success_rate(self, action: str) -> float:
        """Estimate the success probability of an action.

        Parameters
        ----------
        action : str
            Action name to query.

        Returns
        -------
        float
            Fraction of recorded experiences for :paramref:`action` with
            ``reward > 0``, in ``[0.0, 1.0]``. Returns ``0.0`` if
            :paramref:`action` has never been recorded (rather than
            raising or returning ``nan``), so that an unexplored action
            is treated as having no demonstrated success — a conservative
            default for exploration heuristics.
        """
        rewards = self._action_rewards.get(action)
        if not rewards:
            return 0.0

        successes = sum(1 for r in rewards if r > 0.0)
        return successes / len(rewards)

    def context_similarity(
        self, state: PolicyState, max_results: int = 10
    ) -> List[ExperienceTuple]:
        """Find past experiences with a similar context to ``state``.

        Parameters
        ----------
        state : PolicyState
            Reference state to compare against.
        max_results : int, default=10
            Maximum number of similar experiences to return, ordered from
            most to least similar.

        Returns
        -------
        list[ExperienceTuple]
            Up to :paramref:`max_results` experiences from the buffer,
            sorted by ascending L2 distance on the
            ``(risk_score, layer_id)`` feature pair relative to
            :paramref:`state`. Empty list if the buffer is empty.

        Notes
        -----
        :attr:`PolicyState.layer_id` is included in the distance metric
        without normalisation: in practice layer ids are small
        consecutive integers (0, 1, 2, ...) so a difference of one layer
        contributes comparably to a difference of ~1.0 in
        :attr:`PolicyState.risk_score` (which is itself in ``[0, 1]``).
        This intentionally treats "one layer away" as roughly as
        dissimilar as "maximally different risk", which is a reasonable
        default for a small number of MoE layers.
        """
        if not self._buffer or max_results <= 0:
            return []

        def _distance(exp: ExperienceTuple) -> float:
            d_risk = exp.state.risk_score - state.risk_score
            d_layer = exp.state.layer_id - state.layer_id
            return (d_risk * d_risk + d_layer * d_layer) ** 0.5

        ranked = sorted(self._buffer, key=_distance)
        return ranked[:max_results]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self, path: str) -> None:
        """Serialise this memory buffer to a JSON file.

        Parameters
        ----------
        path : str
            Destination file path. Overwritten if it already exists.

        Returns
        -------
        None

        Notes
        -----
        The output schema is::

            {
                "max_size": <int>,
                "experiences": [ <ExperienceTuple.to_dict()>, ... ]
            }

        Experiences are written in buffer order (oldest first).
        """
        payload = {
            "max_size": self.max_size,
            "experiences": [exp.to_dict() for exp in self._buffer],
        }

        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        except OSError as exc:
            logger.warning(
                "[MoEWatch] PolicyMemory: failed to write memory to '%s' (%s).",
                path,
                exc,
            )
            raise

    def from_json(self, path: str) -> None:
        """Load and merge experiences from a JSON file written by :meth:`to_json`.

        Parameters
        ----------
        path : str
            Source file path.

        Returns
        -------
        None

        Notes
        -----
        Replaces the current buffer contents and per-action reward
        history with those loaded from :paramref:`path`. ``max_size`` is
        *not* overwritten from the file — the buffer retains the
        ``max_size`` configured at construction time, so loading a larger
        memory file than the current buffer's capacity truncates to the
        most recent ``max_size`` experiences (oldest dropped first).

        If :paramref:`path` does not exist or contains invalid JSON, logs
        a warning and leaves the current buffer unchanged (load failure
        does not raise, so a missing or corrupt memory file at startup
        degrades to "start with empty memory" rather than crashing).
        """
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "[MoEWatch] PolicyMemory: failed to load memory from '%s' "
                "(%s); leaving current memory unchanged.",
                path,
                exc,
            )
            return

        experiences_data = payload.get("experiences", [])

        new_buffer: Deque[ExperienceTuple] = deque(maxlen=self.max_size)
        new_action_rewards: Dict[str, List[float]] = {}

        for item in experiences_data:
            try:
                exp = ExperienceTuple.from_dict(item)
            except (TypeError, ValueError, KeyError) as exc:
                logger.warning(
                    "[MoEWatch] PolicyMemory: skipping malformed experience "
                    "record in '%s' (%s).",
                    path,
                    exc,
                )
                continue

            new_buffer.append(exp)
            new_action_rewards.setdefault(exp.action, []).append(exp.reward)

        self._buffer = new_buffer
        self._action_rewards = new_action_rewards

        logger.info(
            "[MoEWatch] PolicyMemory: loaded %d experience(s) from '%s'.",
            len(self._buffer),
            path,
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return (
            f"PolicyMemory(size={len(self._buffer)}/{self.max_size}, "
            f"actions={sorted(self._action_rewards)})"
        )
