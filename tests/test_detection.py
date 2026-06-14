# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# tests/test_detection.py
# =============================================================================
#
# Project      : MoEWatch
# Description  : Tests for moewatch.hooks.detection.detect_router_modules().
#
#                Coverage targets:
#                  - Auto-detection on FakeMoEModel (pattern "gate" in leaf)
#                  - Correct module count per layer count
#                  - Returned modules are nn.Linear with correct out_features
#                  - Non-MoE model yields empty dict (no ValueError from fn)
#                  - Manual override (config.router_modules) resolves names
#                  - Manual override with invalid names raises ValueError
#                  - Detection is read-only (model state unchanged after call)
#                  - Large model (8 experts, 4 layers)
#                  - Expert submodules are NOT returned as routers
#                  - Custom class-name hint matching (_class_matches_hint)
#                  - Returned dict keys are valid for model.get_submodule()
#
# =============================================================================

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from moewatch.config import OutputMode, WatchConfig
from moewatch.hooks.detection import (
    _auto_detect,
    _looks_like_router,
    _name_matches_pattern,
    detect_router_modules,
)

# Pull in the fake model classes registered by conftest
from conftest import (
    FakeCollapsingMoEModel,
    FakeMoEModel,
    FakeModelWithManualRouter,
    FakeNonMoEModel,
)


# ===========================================================================
# ── 1. Auto-detection on FakeMoEModel ───────────────────────────────────────
# ===========================================================================


class TestAutoDetectionFakeMoEModel:
    """Auto-detection correctly identifies gate modules in FakeMoEModel."""

    def test_returns_dict(
        self, small_moe_model: FakeMoEModel, default_config: WatchConfig
    ) -> None:
        result = detect_router_modules(small_moe_model, default_config)
        assert isinstance(result, dict), "Expected dict return type"

    def test_detects_all_gate_layers(
        self, small_moe_model: FakeMoEModel, default_config: WatchConfig
    ) -> None:
        """FakeMoEModel with 2 layers should produce exactly 2 detected routers."""
        result = detect_router_modules(small_moe_model, default_config)
        assert len(result) == 2, (
            f"Expected 2 routers for 2-layer model, got {len(result)}: {list(result.keys())}"
        )

    def test_large_model_detects_all_layers(
        self, large_moe_model: FakeMoEModel, default_config: WatchConfig
    ) -> None:
        """FakeMoEModel with 4 layers should produce exactly 4 detected routers."""
        result = detect_router_modules(large_moe_model, default_config)
        assert len(result) == 4, (
            f"Expected 4 routers for 4-layer model, got {len(result)}"
        )

    def test_keys_are_strings(
        self, small_moe_model: FakeMoEModel, default_config: WatchConfig
    ) -> None:
        result = detect_router_modules(small_moe_model, default_config)
        for key in result:
            assert isinstance(key, str), f"Key {key!r} is not a string"

    def test_values_are_nn_modules(
        self, small_moe_model: FakeMoEModel, default_config: WatchConfig
    ) -> None:
        result = detect_router_modules(small_moe_model, default_config)
        for name, module in result.items():
            assert isinstance(module, nn.Module), (
                f"Module at '{name}' is not nn.Module: {type(module)}"
            )

    def test_detected_modules_are_linear_with_correct_out_features(
        self, small_moe_model: FakeMoEModel, default_config: WatchConfig
    ) -> None:
        """Detected gate modules must be nn.Linear with out_features == n_experts."""
        result = detect_router_modules(small_moe_model, default_config)
        for name, module in result.items():
            assert isinstance(module, nn.Linear), (
                f"Router '{name}' is {type(module).__name__}, expected nn.Linear"
            )
            assert module.out_features == small_moe_model.n_experts, (
                f"Router '{name}' out_features={module.out_features}, "
                f"expected {small_moe_model.n_experts}"
            )

    def test_keys_are_resolvable_via_get_submodule(
        self, small_moe_model: FakeMoEModel, default_config: WatchConfig
    ) -> None:
        """Every key in result must be resolvable via model.get_submodule()."""
        result = detect_router_modules(small_moe_model, default_config)
        for name in result:
            resolved = small_moe_model.get_submodule(name)
            assert resolved is result[name], (
                f"get_submodule('{name}') returned a different object than detect_router_modules"
            )

    def test_gate_leaf_present_in_key(
        self, small_moe_model: FakeMoEModel, default_config: WatchConfig
    ) -> None:
        """Detected module names should end with 'gate' (the leaf pattern)."""
        result = detect_router_modules(small_moe_model, default_config)
        for name in result:
            leaf = name.rsplit(".", 1)[-1]
            assert "gate" in leaf.lower(), (
                f"Expected 'gate' in leaf of '{name}', got leaf='{leaf}'"
            )

    def test_expert_submodules_not_detected(
        self, small_moe_model: FakeMoEModel, default_config: WatchConfig
    ) -> None:
        """Expert Linear layers (inside nn.ModuleList) must NOT be returned as routers."""
        result = detect_router_modules(small_moe_model, default_config)
        for name in result:
            # Expert layers are named e.g. "layers.0.experts.0", "layers.0.experts.1"
            assert "experts" not in name, (
                f"Expert submodule '{name}' was incorrectly identified as a router"
            )

    def test_detected_modules_have_out_features_ge_2(
        self, small_moe_model: FakeMoEModel, default_config: WatchConfig
    ) -> None:
        result = detect_router_modules(small_moe_model, default_config)
        for name, module in result.items():
            out = getattr(module, "out_features", None)
            assert out is not None and out >= 2, (
                f"Router '{name}' has out_features={out}, expected >= 2"
            )


# ===========================================================================
# ── 2. Non-MoE model auto-detection ─────────────────────────────────────────
# ===========================================================================


class TestAutoDetectionNonMoEModel:
    """Detection on a plain MLP should return an empty dict."""

    def test_non_moe_returns_empty_dict(
        self, non_moe_model: FakeNonMoEModel, default_config: WatchConfig
    ) -> None:
        result = detect_router_modules(non_moe_model, default_config)
        assert result == {}, (
            f"Expected empty dict for non-MoE model, got {result}"
        )

    def test_non_moe_does_not_raise(
        self, non_moe_model: FakeNonMoEModel, default_config: WatchConfig
    ) -> None:
        """The function itself does not raise; callers decide what to do with empty."""
        try:
            detect_router_modules(non_moe_model, default_config)
        except Exception as exc:
            pytest.fail(f"detect_router_modules raised unexpectedly: {exc}")


# ===========================================================================
# ── 3. Manual override path ──────────────────────────────────────────────────
# ===========================================================================


class TestManualOverride:
    """config.router_modules overrides auto-detection and must resolve all names."""

    def test_override_resolves_moe_router(
        self,
        manual_router_model: FakeModelWithManualRouter,
        config_with_override: WatchConfig,
    ) -> None:
        result = detect_router_modules(manual_router_model, config_with_override)
        assert "moe_router" in result, (
            f"Expected 'moe_router' in result, got keys: {list(result.keys())}"
        )

    def test_override_returns_correct_module(
        self,
        manual_router_model: FakeModelWithManualRouter,
        config_with_override: WatchConfig,
    ) -> None:
        result = detect_router_modules(manual_router_model, config_with_override)
        assert result["moe_router"] is manual_router_model.moe_router

    def test_override_returns_only_specified_modules(
        self,
        manual_router_model: FakeModelWithManualRouter,
        config_with_override: WatchConfig,
    ) -> None:
        """When override is set, only the specified modules are returned."""
        result = detect_router_modules(manual_router_model, config_with_override)
        assert list(result.keys()) == ["moe_router"]

    def test_override_with_invalid_name_raises_value_error(
        self,
        small_moe_model: FakeMoEModel,
        config_with_bad_override: WatchConfig,
    ) -> None:
        with pytest.raises(ValueError, match="could not be resolved"):
            detect_router_modules(small_moe_model, config_with_bad_override)

    def test_override_error_message_includes_missing_names(
        self,
        small_moe_model: FakeMoEModel,
        config_with_bad_override: WatchConfig,
    ) -> None:
        with pytest.raises(ValueError) as exc_info:
            detect_router_modules(small_moe_model, config_with_bad_override)
        assert "this.does.not.exist" in str(exc_info.value)

    def test_override_bypasses_auto_detection_entirely(
        self,
        non_moe_model: FakeNonMoEModel,
    ) -> None:
        """Even a non-MoE model can specify a module via override."""
        # FakeNonMoEModel has 'fc1' and 'fc2' — point at fc1 (has out_features)
        config = WatchConfig(output=OutputMode.SILENT, router_modules=["fc1"])
        result = detect_router_modules(non_moe_model, config)
        assert "fc1" in result
        assert result["fc1"] is non_moe_model.fc1

    def test_empty_override_list_falls_through_to_auto_detect(
        self,
        small_moe_model: FakeMoEModel,
    ) -> None:
        """
        config.router_modules = [] (empty list) is falsy → falls back to
        auto-detection. FakeMoEModel should be auto-detected normally.
        """
        config = WatchConfig(output=OutputMode.SILENT, router_modules=[])
        result = detect_router_modules(small_moe_model, config)
        # Should auto-detect 2 gates in the 2-layer model
        assert len(result) == 2


# ===========================================================================
# ── 4. Read-only guarantee ──────────────────────────────────────────────────
# ===========================================================================


class TestReadOnlyGuarantee:
    """detect_router_modules must not modify any model parameters or attributes."""

    def test_model_parameters_unchanged(
        self, small_moe_model: FakeMoEModel, default_config: WatchConfig
    ) -> None:
        """Record parameter checksums before and after detection; they must match."""
        before = {
            name: p.data.clone()
            for name, p in small_moe_model.named_parameters()
        }
        detect_router_modules(small_moe_model, default_config)
        after = {
            name: p.data
            for name, p in small_moe_model.named_parameters()
        }
        for name in before:
            assert torch.allclose(before[name], after[name]), (
                f"Parameter '{name}' was modified by detect_router_modules()"
            )

    def test_model_named_modules_count_unchanged(
        self, small_moe_model: FakeMoEModel, default_config: WatchConfig
    ) -> None:
        before = list(small_moe_model.named_modules())
        detect_router_modules(small_moe_model, default_config)
        after = list(small_moe_model.named_modules())
        assert len(before) == len(after), (
            "detect_router_modules() added or removed modules from the model"
        )


# ===========================================================================
# ── 5. Internal helper unit tests ───────────────────────────────────────────
# ===========================================================================


class TestInternalHelpers:
    """Unit tests for the private helper functions in detection.py."""

    # ── _name_matches_pattern ───────────────────────────────────────────────

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("layers.0.gate", True),
            ("layers.0.GATE", True),          # case-insensitive via lower()
            ("layers.0.router", True),
            ("layers.0.moe_router", True),
            ("layers.0.moegate", True),
            ("layers.0.topkgate", True),
            ("layers.0.experts.0", False),     # no pattern in leaf
            ("layers.0.fc1", False),
            ("model.embed_tokens", False),
            ("", False),                       # empty name edge case → leaf=""
        ],
    )
    def test_name_matches_pattern(self, name: str, expected: bool) -> None:
        assert _name_matches_pattern(name) == expected, (
            f"_name_matches_pattern({name!r}) returned {not expected}, expected {expected}"
        )

    # ── _looks_like_router ──────────────────────────────────────────────────

    def test_looks_like_router_linear_ge_2(self) -> None:
        m = nn.Linear(32, 8)
        assert _looks_like_router(m) is True

    def test_looks_like_router_linear_exactly_2(self) -> None:
        m = nn.Linear(32, 2)
        assert _looks_like_router(m) is True

    def test_looks_like_router_linear_1_expert(self) -> None:
        """out_features=1 is not a valid router (must route to >= 2 experts)."""
        m = nn.Linear(32, 1)
        assert _looks_like_router(m) is False

    def test_looks_like_router_module_list(self) -> None:
        """ModuleList has no out_features → should return False."""
        m = nn.ModuleList([nn.Linear(32, 32) for _ in range(4)])
        assert _looks_like_router(m) is False

    def test_looks_like_router_embedding(self) -> None:
        """nn.Embedding has no out_features → False."""
        m = nn.Embedding(100, 32)
        assert _looks_like_router(m) is False

    def test_looks_like_router_relu(self) -> None:
        m = nn.ReLU()
        assert _looks_like_router(m) is False

    # ── _auto_detect ────────────────────────────────────────────────────────

    def test_auto_detect_on_empty_model(self) -> None:
        """Model with no submodules returns empty dict."""

        class Stub(nn.Module):
            def forward(self, x):
                return x

        result = _auto_detect(Stub())
        assert result == {}

    def test_auto_detect_skips_root(
        self, small_moe_model: FakeMoEModel
    ) -> None:
        """The root module (name='') should never appear in results."""
        result = _auto_detect(small_moe_model)
        assert "" not in result

    def test_auto_detect_values_are_same_objects_as_get_submodule(
        self, small_moe_model: FakeMoEModel
    ) -> None:
        result = _auto_detect(small_moe_model)
        for name, module in result.items():
            assert module is small_moe_model.get_submodule(name)


# ===========================================================================
# ── 6. Edge-case models ──────────────────────────────────────────────────────
# ===========================================================================


class TestEdgeCases:
    """Boundary / edge-case model configurations."""

    def test_single_layer_model(self, default_config: WatchConfig) -> None:
        model = FakeMoEModel(n_layers=1, n_experts=4, hidden=16)
        result = detect_router_modules(model, default_config)
        assert len(result) == 1

    def test_many_experts(self, default_config: WatchConfig) -> None:
        """Detection works correctly when n_experts is large."""
        model = FakeMoEModel(n_layers=2, n_experts=64, hidden=32)
        result = detect_router_modules(model, default_config)
        assert len(result) == 2
        for module in result.values():
            assert module.out_features == 64

    def test_two_experts_minimum(self, default_config: WatchConfig) -> None:
        """n_experts=2 is the minimum valid router; should still be detected."""
        model = FakeMoEModel(n_layers=1, n_experts=2, hidden=16)
        result = detect_router_modules(model, default_config)
        assert len(result) == 1

    def test_nested_model_wrapper(self, default_config: WatchConfig) -> None:
        """Detection works when the MoE model is wrapped inside another module."""

        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = FakeMoEModel(n_layers=2, n_experts=4, hidden=16)

            def forward(self, x):
                return self.backbone(x)

        wrapper = Wrapper()
        result = detect_router_modules(wrapper, default_config)
        # 2 gate modules inside backbone
        assert len(result) == 2
        for name in result:
            assert "backbone" in name, (
                f"Expected 'backbone' in key '{name}' for wrapped model"
            )

    def test_collapsing_model_gate_detected(
        self,
        collapsing_model: FakeCollapsingMoEModel,
        default_config: WatchConfig,
    ) -> None:
        """FakeCollapsingMoEModel (top-level gate) should be detected."""
        result = detect_router_modules(collapsing_model, default_config)
        assert len(result) == 1
        assert "gate" in list(result.keys())[0]

    def test_detect_returns_independent_dict_copy(
        self, small_moe_model: FakeMoEModel, default_config: WatchConfig
    ) -> None:
        """Mutating the returned dict must not affect subsequent calls."""
        result1 = detect_router_modules(small_moe_model, default_config)
        # Mutate result1
        result1.clear()
        result2 = detect_router_modules(small_moe_model, default_config)
        assert len(result2) == 2, "Mutating returned dict affected next call"

    def test_no_router_modules_key_is_none(
        self, small_moe_model: FakeMoEModel
    ) -> None:
        """config.router_modules=None (default) triggers auto-detection."""
        config = WatchConfig(output=OutputMode.SILENT)
        assert config.router_modules is None
        result = detect_router_modules(small_moe_model, config)
        assert len(result) == 2
