# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# moewatch/hooks/detection.py
# =============================================================================
#
# Project      : MoEWatch
# Version      : v0.2.0
# Description  : Auto-detection of MoE router (gating) modules within an
#                 arbitrary torch.nn.Module. Supports name-pattern matching
#                 tuned for Mixtral, Qwen3-MoE, DeepSeek-MoE, and OLMoE
#                 style architectures, with output-shape validation to
#                 reduce false positives. Falls back to a manual
#                 ``config.router_modules`` override for custom
#                 architectures.
#
# Author       : Abinesh N (@Abineshabee)
# Repository   : https://github.com/Abineshabee/MoEWatch
# License      : Apache 2.0
#
# Contents
# --------
#   detect_router_modules() — primary detection entry point
#
# Usage
# -----
#   from moewatch.hooks.detection import detect_router_modules
#
#   routers = detect_router_modules(model, config)
#   # routers: dict[str, torch.nn.Module]
#
# =============================================================================

from __future__ import annotations

import logging
from typing import Dict, List

import torch
import torch.nn as nn

from moewatch.config import WatchConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Name-pattern heuristics
# ---------------------------------------------------------------------------
#
# Substrings (lowercase), matched against the *leaf* (final dotted
# component) of a module's qualified name, that identify MoE
# router/gating modules across popular architectures:
#
#   Mixtral       : "...block_sparse_moe.gate"   -> leaf = "gate"
#   Qwen3-MoE     : "...mlp.gate"                 -> leaf = "gate"
#   DeepSeek-MoE  : "...mlp.gate"                 -> leaf = "gate"
#   OLMoE         : "...mlp.gate"                 -> leaf = "gate"
#
# Matching only the leaf avoids false positives on submodules nested
# under a parent whose own path contains a MoE-related substring (e.g.
# "block_sparse_moe.experts.0.w1" must NOT match merely because
# "block_sparse_moe" is an ancestor). Shape validation in
# _looks_like_router() provides a second filter.

_ROUTER_LEAF_PATTERNS: List[str] = [
    "gate",
    "router",
    "moe_router",
    "moegate",
    "topkgate",
]

# Module class name substrings that strongly suggest a gating/router layer
# even if the leaf attribute name doesn't match the patterns above
# (last-resort heuristic, still requires shape validation in
# _looks_like_router).
_ROUTER_CLASS_HINTS: List[str] = [
    "gate",
    "router",
    "moegate",
    "topkgate",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_router_modules(
    model: nn.Module,
    config: WatchConfig,
) -> Dict[str, nn.Module]:
    """Auto-detect MoE router (gating) modules within ``model``.

    Resolution order:

    1. If ``config.router_modules`` is set (non-empty list of qualified
       module-name strings), each name is resolved directly via
       :meth:`torch.nn.Module.get_submodule` and returned as the result.
       No shape validation is performed in this path, since the user has
       explicitly identified the modules.
    2. Otherwise, ``model.named_modules()`` is scanned and each module
       whose qualified name contains one of the known router name
       patterns (case-insensitive) is checked for a plausible router
       shape: it must be an ``nn.Linear`` (or expose ``in_features`` /
       ``out_features``) whose ``out_features`` corresponds to a
       plausible expert count (``>= 2``).

    Parameters
    ----------
    model : torch.nn.Module
        The model to scan for MoE router modules.
    config : WatchConfig
        Configuration object. ``config.router_modules``, if set,
        overrides auto-detection entirely.

    Returns
    -------
    dict[str, torch.nn.Module]
        Mapping of fully-qualified module name to the module instance.
        Empty dict if nothing is found (callers are expected to raise a
        descriptive ``ValueError`` in that case — see
        :class:`~moewatch._watcher.MoEWatch` and
        :func:`~moewatch._audit.audit`, which both treat an empty result
        as fatal).

    Raises
    ------
    ValueError
        If ``config.router_modules`` is set but one or more of the named
        modules cannot be resolved via ``model.get_submodule()``.

    Notes
    -----
    This function performs read-only introspection: it never calls
    ``model.forward()`` and never modifies any module attributes.
    """
    if config.router_modules:
        return _resolve_manual_override(model, config.router_modules)

    return _auto_detect(model)


# ---------------------------------------------------------------------------
# Manual override resolution
# ---------------------------------------------------------------------------


def _resolve_manual_override(
    model: nn.Module,
    router_module_names: List[str],
) -> Dict[str, nn.Module]:
    """Resolve a user-provided list of router module names.

    Each entry in ``router_module_names`` is resolved in one of two ways:

    1. **Fully-qualified path** — if the name resolves directly via
       ``model.get_submodule()`` it is used as-is.  Example:
       ``"layers.0.moe.dispatch_router"``.

    2. **Bare name / suffix** — if the name cannot be resolved directly,
       ``model.named_modules()`` is scanned and every module whose dotted
       path ends with that name is collected.  Example:
       ``"dispatch_router"`` matches ``layers.0.moe.dispatch_router``,
       ``layers.1.moe.dispatch_router``, etc.

    Parameters
    ----------
    model : torch.nn.Module
        Model to resolve submodules from.
    router_module_names : list[str]
        Fully-qualified dotted paths **or** bare module names/suffixes.

    Returns
    -------
    dict[str, torch.nn.Module]
        Mapping of fully-qualified name → module for every matched module,
        in traversal order.

    Raises
    ------
    ValueError
        If any entry cannot be resolved either as a direct path or as a
        suffix match against ``model.named_modules()``.
    """
    resolved: Dict[str, nn.Module] = {}
    unresolved: List[str] = []

    # Build full named_modules map once for suffix scanning.
    all_named: Dict[str, nn.Module] = dict(model.named_modules())

    for name in router_module_names:
        # 1. Try direct fully-qualified lookup first.
        try:
            module = model.get_submodule(name)
            resolved[name] = module
            continue
        except AttributeError:
            pass

        # 2. Fall back: collect all modules whose path ends with this name.
        suffix = f".{name}"
        matches = {
            full_name: mod
            for full_name, mod in all_named.items()
            if full_name == name or full_name.endswith(suffix)
        }

        if matches:
            resolved.update(matches)
            logger.debug(
                "[MoEWatch] detect_router_modules(): bare name %r matched "
                "%d module(s) by suffix scan: %s",
                name,
                len(matches),
                list(matches.keys()),
            )
        else:
            unresolved.append(name)

    if unresolved:
        raise ValueError(
            "[MoEWatch] detect_router_modules(): the following entries in "
            f"config.router_modules could not be resolved: {unresolved!r}. "
            "Verify that these dotted paths exist in model.named_modules()."
        )

    logger.debug(
        "[MoEWatch] detect_router_modules(): resolved %d module(s) from "
        "config.router_modules override.",
        len(resolved),
    )
    return resolved


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


def _auto_detect(model: nn.Module) -> Dict[str, nn.Module]:
    """Scan ``model.named_modules()`` for plausible MoE router modules.

    Parameters
    ----------
    model : torch.nn.Module
        Model to scan.

    Returns
    -------
    dict[str, torch.nn.Module]
        Mapping of qualified module name to module instance for every
        candidate that passes both the name-pattern check and the
        shape/heuristic validation in :func:`_looks_like_router`.
    """
    candidates: Dict[str, nn.Module] = {}

    for name, module in model.named_modules():
        if not name:
            # Skip the root module itself (named_modules yields "" for it).
            continue

        if not _name_matches_pattern(name) and not _class_matches_hint(module):
            continue

        if _looks_like_router(module):
            candidates[name] = module

    if candidates:
        logger.debug(
            "[MoEWatch] detect_router_modules(): auto-detected %d router "
            "module(s): %s",
            len(candidates),
            list(candidates.keys()),
        )
    else:
        logger.debug(
            "[MoEWatch] detect_router_modules(): auto-detection found no "
            "candidate router modules."
        )

    return candidates


def _name_matches_pattern(name: str) -> bool:
    """Check whether a module's name leaf matches a router pattern.

    Parameters
    ----------
    name : str
        Fully-qualified dotted module name (e.g.
        ``"model.layers.5.block_sparse_moe.gate"``).

    Returns
    -------
    bool
        ``True`` if the *leaf* (final dotted component) of ``name``,
        lowercased, equals or contains any of the substrings in
        :data:`_ROUTER_LEAF_PATTERNS`. Only the leaf is checked so that
        modules nested under a MoE-named ancestor (e.g. expert weight
        submodules under ``"...block_sparse_moe.experts.0.w1"``) are not
        misidentified as routers merely due to their ancestry.
    """
    leaf = name.rsplit(".", 1)[-1].lower()
    return any(pattern in leaf for pattern in _ROUTER_LEAF_PATTERNS)


def _class_matches_hint(module: nn.Module) -> bool:
    """Check whether a module's class name suggests it is a router.

    Used as a secondary signal when the dotted path itself does not
    contain a recognizable pattern (e.g. custom architectures that name
    their gating submodule something unconventional but use a
    self-descriptive class, such as ``class TopKGate(nn.Module)``).

    Parameters
    ----------
    module : torch.nn.Module
        Module instance to inspect.

    Returns
    -------
    bool
        ``True`` if the lowercased class name contains any substring in
        :data:`_ROUTER_CLASS_HINTS`.
    """
    class_name = type(module).__name__.lower()
    return any(hint in class_name for hint in _ROUTER_CLASS_HINTS)


def _looks_like_router(module: nn.Module) -> bool:
    """Validate that a module's shape is consistent with a router.

    A router/gating module is expected to be (or directly wrap) a linear
    projection from the model's hidden dimension to the number of
    experts: ``out_features`` is the expert count and must be ``>= 2``
    (a router must route to at least two experts to be meaningful).

    Parameters
    ----------
    module : torch.nn.Module
        Candidate module to validate.

    Returns
    -------
    bool
        ``True`` only if ``module`` exposes an integer ``out_features``
        attribute with value ``>= 2`` (and, if present, an integer
        ``in_features >= 1``). Modules lacking ``out_features``
        entirely — including container modules such as
        ``nn.ModuleList`` holders for experts, or composite MoE blocks
        that do not themselves expose a flat linear shape — are rejected
        here. This keeps detection precise: containers and expert
        submodules must never be mistaken for the router itself, even if
        their *name* happens to match a router pattern.
    """
    out_features = getattr(module, "out_features", None)
    if not isinstance(out_features, int) or out_features < 2:
        return False

    in_features = getattr(module, "in_features", None)
    if in_features is not None:
        if not isinstance(in_features, int) or in_features < 1:
            return False

    return True
