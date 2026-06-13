# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# moewatch/__init__.py [ core ]
# =============================================================================
#
# Project      : MoEWatch
# Version      : v0.2.0
# Description  : Public API surface for the MoEWatch library.
#
#                This module is the single entry point for all user-facing
#                imports. It performs environment checks (torch, transformers),
#                respects the NO_COLOR environment variable, and exposes the
#                complete public interface via lazy imports to keep startup
#                overhead minimal.
#
# Author       : Abinesh N (@Abineshabee)
# Repository   : https://github.com/Abineshabee/MoEWatch
# License      : Apache 2.0
#
# Public API
# ----------
#   MoEWatch            — main live monitoring class
#   MoEWatchCallback    — HuggingFace Trainer callback
#   audit               — offline post-training diagnostic function
#   WatchConfig         — central configuration dataclass
#   OutputMode          — enum: SILENT | CLI | JSON
#   AlertLevel          — enum: INFO | WARNING | CRITICAL
#   Alert               — dataclass: single alert object
#   __version__         — library version string
#
# Quick Start
# -----------
#   from moewatch import MoEWatch, WatchConfig, OutputMode
#
#   config = WatchConfig(output=OutputMode.CLI)
#   with MoEWatch(model, config) as watch:
#       watch.attach(trainer)
#       trainer.train()
#
# =============================================================================

from __future__ import annotations

import importlib
import os
import sys
from typing import TYPE_CHECKING

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

__version__: str = "0.2.0"
__author__: str = "Abinesh N"
__license__: str = "Apache 2.0"
__repository__: str = "https://github.com/Abineshabee/MoEWatch"

# ---------------------------------------------------------------------------
# Dependency availability checks
# ---------------------------------------------------------------------------
# These checks run at import time and emit clear error messages rather than
# opaque AttributeError / ModuleNotFoundError traces later.

_MISSING_DEPS: list[str] = []

try:
    import torch  # noqa: F401

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    _MISSING_DEPS.append(
        "torch — install with: pip install torch>=2.0"
    )

try:
    import transformers  # noqa: F401

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    _MISSING_DEPS.append(
        "transformers — install with: pip install transformers>=4.35"
    )

try:
    import numpy  # noqa: F401

    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False
    _MISSING_DEPS.append(
        "numpy — install with: pip install numpy"
    )

if _MISSING_DEPS:
    _missing_str = "\n  ".join(_MISSING_DEPS)
    import warnings

    warnings.warn(
        f"\n[MoEWatch] Missing required dependencies:\n  {_missing_str}\n"
        "Some functionality will be unavailable until these are installed.",
        ImportWarning,
        stacklevel=2,
    )

# ---------------------------------------------------------------------------
# NO_COLOR environment variable
# ---------------------------------------------------------------------------
# Respect the NO_COLOR convention (https://no-color.org/).
# When set to any non-empty value, all ANSI color output is suppressed.
# This is also picked up by WatchConfig.__post_init__() for per-instance
# control, but setting it here ensures the flag is visible globally.

_NO_COLOR: bool = bool(os.environ.get("NO_COLOR", "").strip())

# ---------------------------------------------------------------------------
# Config, enums, and Alert — always importable (no torch dependency)
# ---------------------------------------------------------------------------

from moewatch.config import (  # noqa: E402
    AlertLevel,
    OutputMode,
    WatchConfig,
)

# ---------------------------------------------------------------------------
# Lazy import helpers
# ---------------------------------------------------------------------------
# Heavy submodules (hooks, analyzers, intervention, policy) are deferred
# until first use. This keeps `import moewatch` fast even in environments
# where torch is available but the user hasn't yet called any monitor API.


def _require_torch(operation: str) -> None:
    """Raise ImportError with a helpful message if torch is not installed."""
    if not _TORCH_AVAILABLE:
        raise ImportError(
            f"[MoEWatch] '{operation}' requires PyTorch.\n"
            "Install with: pip install torch>=2.0"
        )


def _require_transformers(operation: str) -> None:
    """Raise ImportError with a helpful message if transformers is missing."""
    if not _TRANSFORMERS_AVAILABLE:
        raise ImportError(
            f"[MoEWatch] '{operation}' requires HuggingFace Transformers.\n"
            "Install with: pip install transformers>=4.35"
        )


# ---------------------------------------------------------------------------
# Alert dataclass — defined here (at API surface) to avoid circular imports
# ---------------------------------------------------------------------------
# Alert is used in _watcher.py, report/, and __init__.py. Defining it here
# keeps it as the canonical location while all submodules import from here.

from dataclasses import dataclass, field  # noqa: E402
from typing import Any, Dict, List  # noqa: E402


@dataclass
class Alert:
    """A single monitoring alert emitted by MoEWatch.

    Alerts represent threshold crossings or state transitions detected
    during live training monitoring or offline audit. They are accumulated
    in ``MoEWatch.alerts`` and returned in ``WatchReport.alerts``.

    Attributes
    ----------
    step : int
        Training step at which the alert was emitted.
    level : AlertLevel
        Severity classification (INFO, WARNING, CRITICAL).
    layer_id : str
        Name of the affected MoE router layer (e.g., "layers.5.moe").
    signal_type : str
        Which signal triggered the alert. One of:
        ``"gradient_starvation"``, ``"entropy_drift"``,
        ``"load_imbalance"``, ``"expert_dead"``, ``"cross_layer"``, ``"risk_score"``.
    message : str
        Human-readable description of the alert condition.
    metrics : dict[str, float]
        Supporting metrics at alert time. Keys depend on signal_type,
        e.g. ``{"gradient_norm": 0.003, "risk_score": 0.72}``.
    """

    step: int
    level: AlertLevel
    layer_id: str
    signal_type: str
    message: str
    metrics: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"[step={self.step}] [{self.level.value.upper()}] "
            f"{self.layer_id} | {self.signal_type} | {self.message}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize alert to a JSON-safe dictionary.

        Returns
        -------
        dict
            All alert fields as primitive types suitable for JSON
            serialization.
        """
        return {
            "step": self.step,
            "level": self.level.value,
            "layer_id": self.layer_id,
            "signal_type": self.signal_type,
            "message": self.message,
            "metrics": self.metrics,
        }


# ---------------------------------------------------------------------------
# MoEWatch and MoEWatchCallback — lazy top-level import
# ---------------------------------------------------------------------------
# The actual classes live in _watcher.py and are imported lazily below.
# We expose them at top level by importing them into this namespace once.

if TYPE_CHECKING:
    # For static analysis (mypy, pyright) we expose the real types.
    from moewatch._watcher import MoEWatch, MoEWatchCallback
    from moewatch._audit import audit


def __getattr__(name: str) -> Any:
    """Lazy attribute loader for heavy top-level symbols.

    Defers importing ``MoEWatch``, ``MoEWatchCallback``, and ``audit``
    until they are first accessed. This avoids importing torch and all
    submodules on ``import moewatch`` when the user may only want config
    or type annotations.

    Parameters
    ----------
    name : str
        Attribute name being accessed.

    Returns
    -------
    Any
        The requested symbol.

    Raises
    ------
    ImportError
        If required dependencies (torch, transformers) are not installed.
    AttributeError
        If the name is not part of the public API.
    """
    if name in ("MoEWatch", "MoEWatchCallback"):
        _require_torch(name)
        _require_transformers(name)
        from moewatch._watcher import MoEWatch as _MoEWatch
        from moewatch._watcher import MoEWatchCallback as _MoEWatchCallback

        # Cache in module namespace so subsequent accesses skip __getattr__
        globals()["MoEWatch"] = _MoEWatch
        globals()["MoEWatchCallback"] = _MoEWatchCallback

        return globals()[name]

    if name == "audit":
        _require_torch("audit")
        from moewatch._audit import audit as _audit

        globals()["audit"] = _audit
        return _audit

    raise AttributeError(
        f"module 'moewatch' has no attribute '{name}'. "
        f"Available public API: MoEWatch, MoEWatchCallback, audit, "
        f"WatchConfig, OutputMode, AlertLevel, Alert, __version__"
    )


# ---------------------------------------------------------------------------
# __all__ — explicit public API declaration
# ---------------------------------------------------------------------------

__all__: List[str] = [
    # Core classes
    "MoEWatch",
    "MoEWatchCallback",
    # Offline diagnostic
    "audit",
    # Configuration
    "WatchConfig",
    "OutputMode",
    "AlertLevel",
    # Data types
    "Alert",
    # Metadata
    "__version__",
    "__author__",
    "__license__",
    "__repository__",
]
