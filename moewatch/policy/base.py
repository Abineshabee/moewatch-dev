# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# moewatch/policy/base.py [v0.2.0]
# =============================================================================
#
# Project      : MoEWatch
# Version      : v0.2.0
# Description  : Abstract interface shared by all intervention-selection
#                 policies. Defines the per-decision state representation
#                 (``PolicyState``) and the contract
#                 (:class:`PolicyBase`) that both the Phase 1 deterministic
#                 :class:`~moewatch.policy.rule_policy.RulePolicy` and the
#                 Phase 2 learning
#                 :class:`~moewatch.policy.bandit_policy.BanditPolicy`
#                 implement.
#
#                 ``InterventionEngine`` and ``MoEWatch`` interact with
#                 policies exclusively through this interface, so either
#                 implementation can be swapped via
#                 ``WatchConfig.policy_type`` without any change to the
#                 calling code.
#
# Author       : Abinesh N (@Abineshabee)
# Repository   : https://github.com/Abineshabee/MoEWatch
# License      : Apache 2.0
#
# Contents
# --------
#   PolicyState  — immutable snapshot of system state at decision time
#   PolicyBase   — abstract base class for all policies
#
# Usage
# -----
#   from moewatch.policy.base import PolicyBase, PolicyState
#
#   class MyPolicy(PolicyBase):
#       def select_action(self, state: PolicyState) -> InterventionAction:
#           ...
#       def update(self, state, action, reward) -> None:
#           ...
#
# =============================================================================

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance
    from moewatch.intervention.actions import InterventionAction


# ---------------------------------------------------------------------------
# PolicyState
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyState:
    """Immutable snapshot of system state at a single decision point.

    Constructed by the
    :class:`~moewatch.intervention.engine.InterventionEngine` (typically
    from a per-layer :class:`~moewatch.analyzer.risk_score.RiskReport`) and
    passed to :meth:`PolicyBase.select_action` and
    :meth:`PolicyBase.update`. Frozen so a single instance can be safely
    reused as a dictionary key or stored in
    :class:`~moewatch.policy.memory.PolicyMemory` without risk of mutation.

    Parameters
    ----------
    risk_score : float
        Current per-layer collapse risk in ``[0.0, 1.0]``, as produced by
        :class:`~moewatch.analyzer.risk_score.RiskScoreFuser`. Higher values
        indicate a greater likelihood of imminent expert collapse.
    layer_id : int
        Zero-indexed identifier of the MoE layer this state describes.
    training_step : int
        Training step at which this state was observed.
    intervention_history : list[str]
        Recent intervention action names for this layer (most recent last),
        e.g. ``["aux_loss", "router_noise"]``. Used by policies to avoid
        oscillation and by :class:`~moewatch.policy.memory.PolicyMemory`
        for context similarity. Defaults to an empty list.
    dominant_signal : str
        Name of the risk-fusion component contributing most to
        :attr:`risk_score`. One of ``"gradient"``, ``"entropy"``, or
        ``"cross_layer"``.

    Notes
    -----
    All fields are validated in :meth:`__post_init__`:

    - :attr:`risk_score` is clamped into ``[0.0, 1.0]`` rather than raising,
      so that small numerical overshoot from upstream analyzers does not
      crash policy selection.
    - :attr:`layer_id` and :attr:`training_step` must be non-negative
      integers.
    - :attr:`dominant_signal` must be one of the three recognised signal
      names; an unrecognised value is coerced to ``"entropy"`` (the most
      general Tier 2 signal) with no error raised, to keep policy selection
      robust against future signal additions.
    """

    risk_score: float
    layer_id: int
    training_step: int
    layer_name: str = ""
    intervention_history: List[str] = field(default_factory=list)
    dominant_signal: str = "entropy"

    _VALID_SIGNALS = ("gradient", "entropy", "cross_layer")

    def __post_init__(self) -> None:
        # NOTE: object.__setattr__ is required because the dataclass is
        # frozen; this is the standard pattern for normalising fields
        # during __post_init__ on frozen dataclasses.
        clamped_risk = min(1.0, max(0.0, float(self.risk_score)))
        object.__setattr__(self, "risk_score", clamped_risk)

        layer_id = int(self.layer_id)
        if layer_id < 0:
            raise ValueError(
                f"[MoEWatch] PolicyState: 'layer_id' must be >= 0, "
                f"got {layer_id}."
            )
        object.__setattr__(self, "layer_id", layer_id)

        training_step = int(self.training_step)
        if training_step < 0:
            raise ValueError(
                f"[MoEWatch] PolicyState: 'training_step' must be >= 0, "
                f"got {training_step}."
            )
        object.__setattr__(self, "training_step", training_step)

        if self.intervention_history is None:
            object.__setattr__(self, "intervention_history", [])
        else:
            object.__setattr__(
                self, "intervention_history", list(self.intervention_history)
            )

        if self.dominant_signal not in self._VALID_SIGNALS:
            object.__setattr__(self, "dominant_signal", "entropy")

    def state_key(self) -> str:
        """Return a coarse, hashable string key for this state.

        Returns
        -------
        str
            A string of the form ``"L{layer_id}|R{risk_bucket}|{signal}"``
            where ``risk_bucket`` is :attr:`risk_score` discretised into
            tenths (``0`` through ``10``). Designed for use as a Q-table
            key by :class:`~moewatch.policy.bandit_policy.BanditPolicy`:
            coarse enough that similar states collide (enabling
            generalisation), fine enough to distinguish meaningfully
            different risk levels.

        Notes
        -----
        Intentionally excludes :attr:`training_step` and
        :attr:`intervention_history` from the key — both vary too rapidly
        to support useful generalisation in a small Q-table, but remain
        available on the full :class:`PolicyState` for richer policies
        (e.g. :class:`~moewatch.policy.memory.PolicyMemory` similarity
        search) that want them.
        """
        risk_bucket = min(10, int(self.risk_score * 10))
        return f"L{self.layer_id}|R{risk_bucket}|{self.dominant_signal}"

    def context_key(self) -> str:
        """Return a coarse, hashable string key for this state.

        Alias for :meth:`state_key`, provided for callers that refer to
        the per-layer state signature as a "context" (e.g. policy memory
        and bandit lookups keyed by observation context).

        Returns
        -------
        str
            Identical to :meth:`state_key`.
        """
        return self.state_key()

    def to_dict(self) -> dict:
        """Return a JSON-serialisable ``dict`` representation.

        Returns
        -------
        dict
            Mapping of all field names to plain Python values (``list``
            for :attr:`intervention_history`), suitable for
            ``json.dumps`` without a custom encoder.
        """
        return {
            "risk_score": self.risk_score,
            "layer_id": self.layer_id,
            "training_step": self.training_step,
            "layer_name": self.layer_name,
            "intervention_history": list(self.intervention_history),
            "dominant_signal": self.dominant_signal,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PolicyState":
        """Reconstruct a :class:`PolicyState` from :meth:`to_dict` output.

        Parameters
        ----------
        data : dict
            Mapping previously produced by :meth:`to_dict` (or an
            equivalent JSON object).

        Returns
        -------
        PolicyState
            Reconstructed, validated instance.

        Notes
        -----
        Missing keys fall back to the dataclass defaults rather than
        raising ``KeyError``, so that memory files written by future
        minor versions (which may add fields) remain loadable.
        """
        return cls(
            risk_score=data.get("risk_score", 0.0),
            layer_id=data.get("layer_id", 0),
            training_step=data.get("training_step", 0),
            layer_name=data.get("layer_name", ""),
            intervention_history=data.get("intervention_history", []),
            dominant_signal=data.get("dominant_signal", "entropy"),
        )


# ---------------------------------------------------------------------------
# PolicyBase
# ---------------------------------------------------------------------------


class PolicyBase(ABC):
    """Abstract interface for intervention-selection policies.

    A policy maps the current per-layer :class:`PolicyState` to an
    :class:`~moewatch.intervention.actions.InterventionAction`, and
    receives feedback in the form of a scalar reward once the outcome of
    that action has been observed (via
    :class:`~moewatch.policy.reward.RewardComputer`).

    Subclasses
    ----------
    - :class:`~moewatch.policy.rule_policy.RulePolicy` — Phase 1,
      deterministic threshold-based mapping. Stable, explainable, the
      default and recommended starting point.
    - :class:`~moewatch.policy.bandit_policy.BanditPolicy` — Phase 2,
      contextual bandit that adapts its action selection within a run
      based on observed rewards.

    Notes
    -----
    Both :meth:`select_action` and :meth:`update` are abstract and must be
    implemented by subclasses. :meth:`save_checkpoint` and
    :meth:`load_checkpoint` have safe no-op default implementations so that
    minimal policies (or future stateless policies) are not forced to
    implement persistence; stateful subclasses should override both.
    """

    @abstractmethod
    def select_action(self, state: "PolicyState") -> "InterventionAction":
        """Select an intervention action given the current state.

        Parameters
        ----------
        state : PolicyState
            Current per-layer system state.

        Returns
        -------
        InterventionAction
            One of
            :class:`~moewatch.intervention.actions.NoOpAction`,
            :class:`~moewatch.intervention.actions.AuxLossAction`,
            :class:`~moewatch.intervention.actions.RouterNoiseAction`, or
            :class:`~moewatch.intervention.actions.ExpertDropoutAction`,
            targeting the layer identified by ``state.layer_id``.

        Notes
        -----
        Must not have observable side effects beyond internal
        bookkeeping required for the policy's own operation (e.g.
        incrementing an internal step counter for exploration scheduling).
        Must not call ``action.apply()`` — that is the responsibility of
        :class:`~moewatch.intervention.engine.InterventionEngine`, after
        :class:`~moewatch.intervention.safety.SafetyGuard` has had a chance
        to veto or downgrade the action.
        """
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        state: "PolicyState",
        action: "InterventionAction",
        reward: float,
    ) -> None:
        """Update the policy with observed feedback.

        Parameters
        ----------
        state : PolicyState
            The state at the time :paramref:`action` was selected.
        action : InterventionAction
            The action that was selected (and, ordinarily, applied) for
            :paramref:`state`.
        reward : float
            Discounted counterfactual reward observed for this
            (state, action) pair, as computed by
            :class:`~moewatch.policy.reward.RewardComputer`. Positive
            values indicate the action improved routing stability relative
            to the counterfactual baseline; negative values indicate it
            made matters worse (or no better than doing nothing).

        Returns
        -------
        None

        Notes
        -----
        Deterministic policies (e.g. :class:`RulePolicy`) may use this
        purely for logging/diagnostics without changing future behaviour.
        Learning policies (e.g. :class:`BanditPolicy`) use this to update
        internal value estimates.
        """
        raise NotImplementedError

    def save_checkpoint(self, path: str) -> None:
        """Save policy state to ``path``.

        Parameters
        ----------
        path : str
            Destination file path.

        Returns
        -------
        None

        Notes
        -----
        Default implementation is a documented no-op: policies with no
        meaningful state to persist (e.g. :class:`RulePolicy` beyond its
        oscillation-detection history) may rely on this default. Stateful
        subclasses (e.g. :class:`BanditPolicy`) must override this to
        actually persist their state.
        """
        return None

    def load_checkpoint(self, path: str) -> None:
        """Load policy state from ``path``.

        Parameters
        ----------
        path : str
            Source file path.

        Returns
        -------
        None

        Notes
        -----
        Default implementation is a documented no-op, mirroring
        :meth:`save_checkpoint`. Stateful subclasses must override this.
        """
        return None
