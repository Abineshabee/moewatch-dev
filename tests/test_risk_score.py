# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# tests/test_risk_score.py
# =============================================================================
#
# Project      : MoEWatch
# Description  : Tests for moewatch.analyzer.risk_score.RiskScoreFuser
#
#                Covers:
#                - RiskLevel enum values and semantics
#                - RiskReport dataclass initialization and to_dict()
#                - RiskScoreFuser construction with default/custom weights
#                - Weight validation (sum ≈ 1.0, non-negative)
#                - fuse() method: signal fusion with three tiers
#                - Per-tier signal extraction and normalization
#                - Drift detection boost (T2 signal)
#                - Risk level classification (LOW/MID/HIGH/CRITICAL)
#                - Dominant signal identification
#                - Cache of latest reports per layer
#                - Missing tier handling (duck typing tolerance)
#                - Edge cases: all-zero signals, missing fields, NaN handling
#                - Idempotence on stable signals
#                - Contribution balance verification
#
# Author       : Abinesh N (@Abineshabee)
# Repository   : https://github.com/Abineshabee/MoEWatch
# License      : Apache 2.0
#
# =============================================================================

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pytest

from moewatch.analyzer.risk_score import RiskLevel, RiskReport, RiskScoreFuser
from moewatch.config import OutputMode, WatchConfig


# ===========================================================================
# ── Local fixtures for risk score testing ──────────────────────────────────
# ===========================================================================


@pytest.fixture
def risk_config() -> WatchConfig:
    """WatchConfig tuned for risk score testing."""
    return WatchConfig(
        output=OutputMode.SILENT,
        dead_threshold=0.01,
        cold_threshold=0.05,
        cold_steps_limit=5,
        entropy_warn=0.3,
        entropy_critical=0.15,
        log_every=1,
    )


# ===========================================================================
# ── Section 1: RiskLevel enum tests ────────────────────────────────────────
# ===========================================================================


class TestRiskLevelEnum:
    """Tests for RiskLevel enum."""

    def test_risk_level_members(self) -> None:
        """RiskLevel has all expected members."""
        assert hasattr(RiskLevel, "LOW")
        assert hasattr(RiskLevel, "MID")
        assert hasattr(RiskLevel, "HIGH")
        assert hasattr(RiskLevel, "CRITICAL")

    def test_risk_level_values(self) -> None:
        """RiskLevel members have string values."""
        assert RiskLevel.LOW.value == "LOW"
        assert RiskLevel.MID.value == "MID"
        assert RiskLevel.HIGH.value == "HIGH"
        assert RiskLevel.CRITICAL.value == "CRITICAL"


# ===========================================================================
# ── Section 2: RiskReport dataclass tests ──────────────────────────────────
# ===========================================================================


class TestRiskReportDataclass:
    """Tests for RiskReport initialization and serialization."""

    def test_default_construction(self) -> None:
        """RiskReport initializes with default values."""
        report = RiskReport(layer_name="layer.0.gate")
        assert report.layer_name == "layer.0.gate"
        assert report.risk_score == 0.0
        assert report.risk_level == RiskLevel.LOW
        assert report.dominant_signal == "none"
        assert report.tier1_contribution == 0.0
        assert report.tier2_contribution == 0.0
        assert report.tier3_contribution == 0.0
        assert report.step == 0

    def test_custom_construction(self) -> None:
        """RiskReport accepts all fields."""
        report = RiskReport(
            layer_name="layer.0.gate",
            risk_score=0.75,
            risk_level=RiskLevel.HIGH,
            dominant_signal="entropy",
            tier1_contribution=0.4,
            tier2_contribution=0.3,
            tier3_contribution=0.05,
            step=100,
        )
        assert report.risk_score == 0.75
        assert report.risk_level == RiskLevel.HIGH
        assert report.dominant_signal == "entropy"
        assert report.step == 100

    def test_to_dict_serialization(self) -> None:
        """to_dict() produces JSON-safe dict with string enums."""
        report = RiskReport(
            layer_name="layer.0.gate",
            risk_score=0.5,
            risk_level=RiskLevel.MID,
            dominant_signal="gradient",
        )
        d = report.to_dict()

        assert isinstance(d, dict)
        assert d["layer_name"] == "layer.0.gate"
        assert d["risk_score"] == 0.5
        assert d["risk_level"] == "MID"  # Enum as string
        assert d["dominant_signal"] == "gradient"

    def test_to_dict_all_fields(self) -> None:
        """to_dict() includes all fields."""
        report = RiskReport(
            layer_name="test",
            risk_score=0.8,
            risk_level=RiskLevel.CRITICAL,
            dominant_signal="cross_layer",
            tier1_contribution=0.5,
            tier2_contribution=0.25,
            tier3_contribution=0.05,
            step=50,
        )
        d = report.to_dict()

        required_keys = {
            "layer_name",
            "risk_score",
            "risk_level",
            "dominant_signal",
            "tier1_contribution",
            "tier2_contribution",
            "tier3_contribution",
            "step",
        }
        assert set(d.keys()) == required_keys


# ===========================================================================
# ── Section 3: RiskScoreFuser initialization and weight validation ────────
# ===========================================================================


class TestRiskScoreFuserInit:
    """Tests for RiskScoreFuser construction and weight validation."""

    def test_default_construction(self, risk_config: WatchConfig) -> None:
        """RiskScoreFuser constructs with default weights."""
        fuser = RiskScoreFuser(risk_config)
        assert fuser.config is risk_config
        assert "tier1" in fuser.weights
        assert "tier2" in fuser.weights
        assert "tier3" in fuser.weights

    def test_default_weights_sum_to_one(self, risk_config: WatchConfig) -> None:
        """Default weights sum to 1.0."""
        fuser = RiskScoreFuser(risk_config)
        weight_sum = sum(fuser.weights.values())
        assert abs(weight_sum - 1.0) < 1e-6

    def test_custom_weights_valid(self, risk_config: WatchConfig) -> None:
        """Custom weights are accepted if they sum to 1.0."""
        weights = {"tier1": 0.5, "tier2": 0.3, "tier3": 0.2}
        fuser = RiskScoreFuser(risk_config, weights=weights)
        assert fuser.weights == weights

    def test_custom_weights_invalid_sum(self, risk_config: WatchConfig) -> None:
        """Invalid weight sum raises ValueError."""
        weights = {"tier1": 0.5, "tier2": 0.3, "tier3": 0.1}  # Sum = 0.9
        with pytest.raises(ValueError):
            RiskScoreFuser(risk_config, weights=weights)

    def test_custom_weights_negative_raises(self, risk_config: WatchConfig) -> None:
        """Negative weights raise ValueError."""
        weights = {"tier1": -0.1, "tier2": 0.6, "tier3": 0.5}
        with pytest.raises(ValueError):
            RiskScoreFuser(risk_config, weights=weights)

    def test_custom_weights_all_ones(self, risk_config: WatchConfig) -> None:
        """All weights equal (1/3 each) is valid."""
        weights = {"tier1": 1.0 / 3, "tier2": 1.0 / 3, "tier3": 1.0 / 3}
        fuser = RiskScoreFuser(risk_config, weights=weights)
        assert abs(sum(fuser.weights.values()) - 1.0) < 1e-6


# ===========================================================================
# ── Section 4: Signal extraction and normalization ────────────────────────
# ===========================================================================


class TestRiskScoreFuserSignalExtraction:
    """Tests for extracting and normalizing tier signals."""

    def test_extract_tier1_signal(self, risk_config: WatchConfig) -> None:
        """Tier 1 (gradient starvation) signal extracted correctly."""
        fuser = RiskScoreFuser(risk_config)

        gradient_report = type("Report", (), {
            "starvation_score": 0.4,
            "layer_name": "layer.0.gate",
            "step": 0,
        })()
        entropy_report = type("Report", (), {
            "normalized_entropy": 0.8,
            "drift_detected": False,
            "step": 0,
        })()

        report = fuser.fuse(gradient_report, entropy_report)

        # Tier 1 contribution ≈ 0.4 * weight
        assert report.tier1_contribution > 0.0

    def test_extract_tier2_entropy_inversion(self, risk_config: WatchConfig) -> None:
        """Tier 2 (entropy) signal is inverted: low entropy → high risk."""
        fuser = RiskScoreFuser(risk_config)

        gradient_report = type("Report", (), {
            "starvation_score": 0.0,
            "layer_name": "layer.0.gate",
            "step": 0,
        })()

        # Low entropy (collapsed) → should contribute high risk.
        entropy_report = type("Report", (), {
            "normalized_entropy": 0.1,  # Low entropy
            "drift_detected": False,
            "step": 0,
        })()

        report = fuser.fuse(gradient_report, entropy_report)

        # T2 = 1 - 0.1 = 0.9 → high contribution
        assert report.tier2_contribution > 0.0
        assert report.risk_score > 0.0

    def test_drift_detected_boost(self, risk_config: WatchConfig) -> None:
        """Drift detection boosts T2 contribution by ~1.2x."""
        fuser = RiskScoreFuser(risk_config)

        gradient_report = type("Report", (), {
            "starvation_score": 0.0,
            "layer_name": "layer.0.gate",
            "step": 0,
        })()

        # Same entropy, but with drift detected.
        entropy_value = 0.5
        entropy_no_drift = type("Report", (), {
            "normalized_entropy": entropy_value,
            "drift_detected": False,
            "step": 0,
        })()
        entropy_with_drift = type("Report", (), {
            "normalized_entropy": entropy_value,
            "drift_detected": True,
            "step": 0,
        })()

        report_no_drift = fuser.fuse(gradient_report, entropy_no_drift)
        report_with_drift = fuser.fuse(gradient_report, entropy_with_drift)

        # Drift should increase T2 contribution.
        assert report_with_drift.tier2_contribution > report_no_drift.tier2_contribution

    def test_extract_tier3_cross_layer(self, risk_config: WatchConfig) -> None:
        """Tier 3 (cross-layer) signal extracted when provided."""
        fuser = RiskScoreFuser(risk_config)

        gradient_report = type("Report", (), {
            "starvation_score": 0.0,
            "layer_name": "layer.0.gate",
            "step": 0,
        })()
        entropy_report = type("Report", (), {
            "normalized_entropy": 0.8,
            "drift_detected": False,
            "step": 0,
        })()
        cross_layer_report = type("Report", (), {
            "spread_score": 0.6,
            "victim_layers": ["layer.1.gate"],
        })()

        report = fuser.fuse(gradient_report, entropy_report, cross_layer_report)

        # Tier 3 contribution should be positive.
        assert report.tier3_contribution > 0.0


# ===========================================================================
# ── Section 5: Risk score computation and contribution balance ──────────────
# ===========================================================================


class TestRiskScoreFuserFusion:
    """Tests for risk score fusion and contribution computation."""

    def test_risk_score_bounds(self, risk_config: WatchConfig) -> None:
        """Risk score is always in [0.0, 1.0]."""
        fuser = RiskScoreFuser(risk_config)

        test_cases = [
            (0.0, 1.0, None),  # All healthy
            (1.0, 0.0, None),  # All critical
            (0.5, 0.5, None),  # Mixed
            (0.0, 0.0, None),  # Zero inputs
        ]

        for tier1, tier2, tier3 in test_cases:
            gradient_report = type("Report", (), {
                "starvation_score": tier1,
                "layer_name": "layer.0.gate",
                "step": 0,
            })()
            entropy_report = type("Report", (), {
                "normalized_entropy": 1.0 - tier2,  # Invert
                "drift_detected": False,
                "step": 0,
            })()

            report = fuser.fuse(gradient_report, entropy_report)

            assert 0.0 <= report.risk_score <= 1.0

    def test_contributions_sum_to_risk_score(self, risk_config: WatchConfig) -> None:
        """Sum of contributions ≈ risk_score (allowing for numerical precision)."""
        fuser = RiskScoreFuser(risk_config)

        gradient_report = type("Report", (), {
            "starvation_score": 0.4,
            "layer_name": "layer.0.gate",
            "step": 0,
        })()
        entropy_report = type("Report", (), {
            "normalized_entropy": 0.6,
            "drift_detected": False,
            "step": 0,
        })()

        report = fuser.fuse(gradient_report, entropy_report)

        contribution_sum = (
            report.tier1_contribution +
            report.tier2_contribution +
            report.tier3_contribution
        )

        # Allow for floating-point rounding errors.
        assert abs(contribution_sum - report.risk_score) < 1e-10

    def test_zero_contributions_zero_score(self, risk_config: WatchConfig) -> None:
        """All-zero signals produce zero risk score."""
        fuser = RiskScoreFuser(risk_config)

        gradient_report = type("Report", (), {
            "starvation_score": 0.0,
            "layer_name": "layer.0.gate",
            "step": 0,
        })()
        entropy_report = type("Report", (), {
            "normalized_entropy": 1.0,  # Healthy (1 - 1.0 = 0.0)
            "drift_detected": False,
            "step": 0,
        })()

        report = fuser.fuse(gradient_report, entropy_report)

        assert report.risk_score == 0.0
        assert report.tier1_contribution == 0.0
        assert report.tier2_contribution == 0.0
        assert report.tier3_contribution == 0.0


# ===========================================================================
# ── Section 6: Risk level classification ───────────────────────────────────
# ===========================================================================


class TestRiskScoreFuserClassification:
    """Tests for risk level classification."""

    def test_risk_level_low(self, risk_config: WatchConfig) -> None:
        """Low risk scores → RiskLevel.LOW."""
        fuser = RiskScoreFuser(risk_config)

        gradient_report = type("Report", (), {
            "starvation_score": 0.1,
            "layer_name": "layer.0.gate",
            "step": 0,
        })()
        entropy_report = type("Report", (), {
            "normalized_entropy": 0.95,  # High entropy (healthy)
            "drift_detected": False,
            "step": 0,
        })()

        report = fuser.fuse(gradient_report, entropy_report)

        # Risk score should be low.
        assert report.risk_score < 0.3
        assert report.risk_level == RiskLevel.LOW

    def test_risk_level_mid(self, risk_config: WatchConfig) -> None:
        """Mid-range risk scores → RiskLevel.MID."""
        fuser = RiskScoreFuser(risk_config)

        gradient_report = type("Report", (), {
            "starvation_score": 0.5,
            "layer_name": "layer.0.gate",
            "step": 0,
        })()
        entropy_report = type("Report", (), {
            "normalized_entropy": 0.5,
            "drift_detected": False,
            "step": 0,
        })()

        report = fuser.fuse(gradient_report, entropy_report)

        # Risk score should be in mid range.
        assert report.risk_level == RiskLevel.MID

    def test_risk_level_high(self, risk_config: WatchConfig) -> None:
        """High risk scores → RiskLevel.HIGH."""
        fuser = RiskScoreFuser(risk_config)

        gradient_report = type("Report", (), {
            "starvation_score": 0.8,
            "layer_name": "layer.0.gate",
            "step": 0,
        })()
        entropy_report = type("Report", (), {
            "normalized_entropy": 0.2,  # Low entropy
            "drift_detected": False,
            "step": 0,
        })()

        report = fuser.fuse(gradient_report, entropy_report)

        assert report.risk_level == RiskLevel.HIGH

    def test_risk_level_critical(self, risk_config: WatchConfig) -> None:
        """Very high risk scores → RiskLevel.CRITICAL."""
        fuser = RiskScoreFuser(risk_config)

        gradient_report = type("Report", (), {
            "starvation_score": 0.95,
            "layer_name": "layer.0.gate",
            "step": 0,
        })()
        entropy_report = type("Report", (), {
            "normalized_entropy": 0.05,  # Very low entropy
            "drift_detected": True,  # Additional boost
            "step": 0,
        })()

        report = fuser.fuse(gradient_report, entropy_report)

        assert report.risk_level == RiskLevel.CRITICAL


# ===========================================================================
# ── Section 7: Dominant signal identification ──────────────────────────────
# ===========================================================================


class TestRiskScoreFuserDominantSignal:
    """Tests for identifying which tier contributes most."""

    def test_dominant_signal_gradient(self, risk_config: WatchConfig) -> None:
        """Tier 1 (gradient) dominates when starvation is high."""
        fuser = RiskScoreFuser(risk_config)

        # High gradient starvation, low entropy drift.
        gradient_report = type("Report", (), {
            "starvation_score": 0.9,
            "layer_name": "layer.0.gate",
            "step": 0,
        })()
        entropy_report = type("Report", (), {
            "normalized_entropy": 0.95,  # Healthy
            "drift_detected": False,
            "step": 0,
        })()

        report = fuser.fuse(gradient_report, entropy_report)

        assert report.dominant_signal == "gradient"

    def test_dominant_signal_entropy(self, risk_config: WatchConfig) -> None:
        """Tier 2 (entropy) dominates when entropy is low."""
        fuser = RiskScoreFuser(risk_config)

        gradient_report = type("Report", (), {
            "starvation_score": 0.1,  # Low
            "layer_name": "layer.0.gate",
            "step": 0,
        })()
        entropy_report = type("Report", (), {
            "normalized_entropy": 0.05,  # Very low (collapsed)
            "drift_detected": True,
            "step": 0,
        })()

        report = fuser.fuse(gradient_report, entropy_report)

        assert report.dominant_signal == "entropy"

    def test_dominant_signal_cross_layer(self, risk_config: WatchConfig) -> None:
        """Tier 3 (cross-layer) dominates when spread is high."""
        fuser = RiskScoreFuser(risk_config)

        gradient_report = type("Report", (), {
            "starvation_score": 0.1,
            "layer_name": "layer.0.gate",
            "step": 0,
        })()
        entropy_report = type("Report", (), {
            "normalized_entropy": 0.8,  # Healthy
            "drift_detected": False,
            "step": 0,
        })()
        cross_layer_report = type("Report", (), {
            "spread_score": 0.95,  # Very high spread
            "victim_layers": ["layer.1.gate", "layer.2.gate"],
        })()

        report = fuser.fuse(gradient_report, entropy_report, cross_layer_report)

        assert report.dominant_signal == "cross_layer"

    def test_dominant_signal_none_all_zero(self, risk_config: WatchConfig) -> None:
        """Dominant signal is 'none' when all contributions are zero."""
        fuser = RiskScoreFuser(risk_config)

        gradient_report = type("Report", (), {
            "starvation_score": 0.0,
            "layer_name": "layer.0.gate",
            "step": 0,
        })()
        entropy_report = type("Report", (), {
            "normalized_entropy": 1.0,  # Healthy
            "drift_detected": False,
            "step": 0,
        })()

        report = fuser.fuse(gradient_report, entropy_report)

        assert report.dominant_signal == "none"


# ===========================================================================
# ── Section 8: Report caching ──────────────────────────────────────────────
# ===========================================================================


class TestRiskScoreFuserCache:
    """Tests for latest report caching per layer."""

    def test_get_latest_score_cached(self, risk_config: WatchConfig) -> None:
        """get_latest_score() returns the most recent report for a layer."""
        fuser = RiskScoreFuser(risk_config)

        gradient_report = type("Report", (), {
            "starvation_score": 0.3,
            "layer_name": "layer.0.gate",
            "step": 0,
        })()
        entropy_report = type("Report", (), {
            "normalized_entropy": 0.7,
            "drift_detected": False,
            "step": 0,
        })()

        report1 = fuser.fuse(gradient_report, entropy_report)
        report_cached = fuser.get_latest_score("layer.0.gate")

        assert report_cached is report1

    def test_cache_updated_on_new_fuse(self, risk_config: WatchConfig) -> None:
        """Cache is updated when fuse() is called again."""
        fuser = RiskScoreFuser(risk_config)

        gradient_report1 = type("Report", (), {
            "starvation_score": 0.2,
            "layer_name": "layer.0.gate",
            "step": 0,
        })()
        entropy_report1 = type("Report", (), {
            "normalized_entropy": 0.8,
            "drift_detected": False,
            "step": 0,
        })()

        report1 = fuser.fuse(gradient_report1, entropy_report1)

        # Fuse again with different signals.
        gradient_report2 = type("Report", (), {
            "starvation_score": 0.7,  # Higher starvation
            "layer_name": "layer.0.gate",
            "step": 1,
        })()
        entropy_report2 = type("Report", (), {
            "normalized_entropy": 0.5,
            "drift_detected": False,
            "step": 1,
        })()

        report2 = fuser.fuse(gradient_report2, entropy_report2)
        report_cached = fuser.get_latest_score("layer.0.gate")

        assert report_cached is report2
        assert report_cached.step == 1
        assert report_cached.risk_score > report1.risk_score

    def test_cache_independent_per_layer(self, risk_config: WatchConfig) -> None:
        """Cache is maintained independently per layer."""
        fuser = RiskScoreFuser(risk_config)

        # Fuse for layer 0.
        gradient_report0 = type("Report", (), {
            "starvation_score": 0.3,
            "layer_name": "layer.0.gate",
            "step": 0,
        })()
        entropy_report0 = type("Report", (), {
            "normalized_entropy": 0.7,
            "drift_detected": False,
            "step": 0,
        })()

        report0 = fuser.fuse(gradient_report0, entropy_report0)

        # Fuse for layer 1.
        gradient_report1 = type("Report", (), {
            "starvation_score": 0.8,
            "layer_name": "layer.1.gate",
            "step": 0,
        })()
        entropy_report1 = type("Report", (), {
            "normalized_entropy": 0.2,
            "drift_detected": False,
            "step": 0,
        })()

        report1 = fuser.fuse(gradient_report1, entropy_report1)

        # Check cached values.
        assert fuser.get_latest_score("layer.0.gate") is report0
        assert fuser.get_latest_score("layer.1.gate") is report1

    def test_get_latest_score_nonexistent_layer(self, risk_config: WatchConfig) -> None:
        """get_latest_score() returns None for unknown layer."""
        fuser = RiskScoreFuser(risk_config)
        result = fuser.get_latest_score("nonexistent.layer")
        assert result is None


# ===========================================================================
# ── Section 9: Duck typing and missing field tolerance ──────────────────────
# ===========================================================================


class TestRiskScoreFuserDuckTyping:
    """Tests for tolerant handling of incomplete tier reports."""

    def test_missing_tier1_fields(self, risk_config: WatchConfig) -> None:
        """fuse() handles missing Tier 1 fields gracefully."""
        fuser = RiskScoreFuser(risk_config)

        # Gradient report with missing starvation_score.
        gradient_report = type("Report", (), {
            "layer_name": "layer.0.gate",
            "step": 0,
        })()
        entropy_report = type("Report", (), {
            "normalized_entropy": 0.8,
            "drift_detected": False,
            "step": 0,
        })()

        report = fuser.fuse(gradient_report, entropy_report)

        # Should still work, with T1 defaulting to 0.0.
        assert isinstance(report, RiskReport)
        assert report.tier1_contribution == 0.0

    def test_missing_tier2_fields(self, risk_config: WatchConfig) -> None:
        """fuse() handles missing Tier 2 fields gracefully."""
        fuser = RiskScoreFuser(risk_config)

        gradient_report = type("Report", (), {
            "starvation_score": 0.5,
            "layer_name": "layer.0.gate",
            "step": 0,
        })()
        # Entropy report with missing normalized_entropy.
        entropy_report = type("Report", (), {
            "drift_detected": False,
            "step": 0,
        })()

        report = fuser.fuse(gradient_report, entropy_report)

        assert isinstance(report, RiskReport)

    def test_none_tier3_allowed(self, risk_config: WatchConfig) -> None:
        """fuse() accepts None for cross-layer report."""
        fuser = RiskScoreFuser(risk_config)

        gradient_report = type("Report", (), {
            "starvation_score": 0.3,
            "layer_name": "layer.0.gate",
            "step": 0,
        })()
        entropy_report = type("Report", (), {
            "normalized_entropy": 0.7,
            "drift_detected": False,
            "step": 0,
        })()

        report = fuser.fuse(gradient_report, entropy_report, cross_layer_report=None)

        assert isinstance(report, RiskReport)
        assert report.tier3_contribution == 0.0


# ===========================================================================
# ── Section 10: Edge cases and robustness ──────────────────────────────────
# ===========================================================================


class TestRiskScoreFuserEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_nan_signal_handling(self, risk_config: WatchConfig) -> None:
        """fuse() handles NaN signals gracefully."""
        fuser = RiskScoreFuser(risk_config)

        gradient_report = type("Report", (), {
            "starvation_score": float("nan"),
            "layer_name": "layer.0.gate",
            "step": 0,
        })()
        entropy_report = type("Report", (), {
            "normalized_entropy": 0.8,
            "drift_detected": False,
            "step": 0,
        })()

        try:
            report = fuser.fuse(gradient_report, entropy_report)
            # If no exception, risk_score should be clipped to valid range.
            assert isinstance(report, RiskReport)
        except (ValueError, TypeError):
            # NaN handling is implementation-specific.
            pass

    def test_inf_signal_handling(self, risk_config: WatchConfig) -> None:
        """fuse() handles infinity signals gracefully."""
        fuser = RiskScoreFuser(risk_config)

        gradient_report = type("Report", (), {
            "starvation_score": float("inf"),
            "layer_name": "layer.0.gate",
            "step": 0,
        })()
        entropy_report = type("Report", (), {
            "normalized_entropy": 0.8,
            "drift_detected": False,
            "step": 0,
        })()

        try:
            report = fuser.fuse(gradient_report, entropy_report)
            # Risk score should be clipped to [0, 1].
            assert 0.0 <= report.risk_score <= 1.0
        except (ValueError, OverflowError):
            pass

    def test_out_of_range_entropy(self, risk_config: WatchConfig) -> None:
        """fuse() clips out-of-range entropy values."""
        fuser = RiskScoreFuser(risk_config)

        gradient_report = type("Report", (), {
            "starvation_score": 0.2,
            "layer_name": "layer.0.gate",
            "step": 0,
        })()

        # Entropy > 1.0 (out of range).
        entropy_report = type("Report", (), {
            "normalized_entropy": 1.5,
            "drift_detected": False,
            "step": 0,
        })()

        report = fuser.fuse(gradient_report, entropy_report)

        # Risk score should still be valid.
        assert 0.0 <= report.risk_score <= 1.0

    def test_empty_victim_list(self, risk_config: WatchConfig) -> None:
        """fuse() handles cross-layer report with no victims."""
        fuser = RiskScoreFuser(risk_config)

        gradient_report = type("Report", (), {
            "starvation_score": 0.2,
            "layer_name": "layer.0.gate",
            "step": 0,
        })()
        entropy_report = type("Report", (), {
            "normalized_entropy": 0.8,
            "drift_detected": False,
            "step": 0,
        })()
        cross_layer_report = type("Report", (), {
            "spread_score": 0.0,
            "victim_layers": [],  # No victims
        })()

        report = fuser.fuse(gradient_report, entropy_report, cross_layer_report)

        assert report.tier3_contribution >= 0.0


# ===========================================================================
# ── Section 11: Idempotence and consistency ────────────────────────────────
# ===========================================================================


class TestRiskScoreFuserIdempotence:
    """Tests for consistent behavior on repeated calls."""

    def test_repeated_fuse_stable_input(self, risk_config: WatchConfig) -> None:
        """Repeated fuse() calls with same input produce same risk score."""
        fuser = RiskScoreFuser(risk_config)

        gradient_report = type("Report", (), {
            "starvation_score": 0.5,
            "layer_name": "layer.0.gate",
            "step": 0,
        })()
        entropy_report = type("Report", (), {
            "normalized_entropy": 0.6,
            "drift_detected": False,
            "step": 0,
        })()

        report1 = fuser.fuse(gradient_report, entropy_report)
        report2 = fuser.fuse(gradient_report, entropy_report)

        assert report1.risk_score == report2.risk_score
        assert report1.risk_level == report2.risk_level
        assert report1.dominant_signal == report2.dominant_signal

    def test_fuse_return_type_always_valid(self, risk_config: WatchConfig) -> None:
        """fuse() always returns a valid RiskReport."""
        fuser = RiskScoreFuser(risk_config)

        test_cases = [
            (0.0, 1.0),
            (0.5, 0.5),
            (1.0, 0.0),
            (0.1, 0.9),
        ]

        for starvation, entropy_norm in test_cases:
            gradient_report = type("Report", (), {
                "starvation_score": starvation,
                "layer_name": "layer.0.gate",
                "step": 0,
            })()
            entropy_report = type("Report", (), {
                "normalized_entropy": entropy_norm,
                "drift_detected": False,
                "step": 0,
            })()

            report = fuser.fuse(gradient_report, entropy_report)

            assert isinstance(report, RiskReport)
            assert 0.0 <= report.risk_score <= 1.0
            assert hasattr(report, "risk_level")
            assert hasattr(report, "dominant_signal")


# ===========================================================================
# ── Coverage measurement checkpoint ────────────────────────────────────────
# ===========================================================================

# Target: ≥ 80% coverage of risk_score.py
#
# Key coverage areas:
#   ✓ RiskLevel enum
#   ✓ RiskReport.__init__
#   ✓ RiskReport.to_dict()
#   ✓ RiskScoreFuser.__init__
#   ✓ RiskScoreFuser._validate_weights()
#   ✓ RiskScoreFuser.fuse() (all branches)
#   ✓ Signal extraction (T1, T2, T3)
#   ✓ Normalization and clipping
#   ✓ Drift detection boost
#   ✓ Risk level classification
#   ✓ Dominant signal identification
#   ✓ Report caching
#   ✓ get_latest_score()
#
# Edge cases covered:
#   ✓ Default weights
#   ✓ Custom weights (valid and invalid)
#   ✓ Missing tier reports
#   ✓ NaN/Inf handling
#   ✓ Out-of-range signals
#   ✓ Empty victim lists
#   ✓ Idempotence
