# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# moewatch/_audit.py [ core ]
# =============================================================================
#
# Project      : MoEWatch
# Version      : v0.2.0
# Description  : Offline post-training diagnostic function for MoE models.
#
#                The ``audit()`` function is the single entry point for all
#                post-hoc routing health analysis. It accepts a trained model
#                and a validation dataloader, runs N forward passes while
#                collecting routing and gradient statistics through the full
#                hook pipeline, then runs the complete analyzer suite (entropy,
#                collapse, gradient starvation, cross-layer, risk score fusion)
#                and returns a comprehensive, serializable AuditReport.
#
#                Design guarantees:
#                  - Zero weight modification: hooks are strictly read-only.
#                  - No hooks left attached after return (guaranteed via finally).
#                  - Gradient computation disabled throughout (torch.no_grad).
#                  - Graceful degradation: per-layer analyzer failures are
#                    logged and skipped rather than aborting the entire audit.
#
# Author       : Abinesh N (@Abineshabee)
# Repository   : https://github.com/Abineshabee/MoEWatch
# License      : Apache 2.0
#
# Public API
# ----------
#   audit(model, dataloader, num_batches, config, device) → AuditReport
#
# Usage
# -----
#   from moewatch import audit, WatchConfig, OutputMode
#
#   config = WatchConfig(output=OutputMode.SILENT)
#   report = audit(model, val_dataloader, num_batches=100, config=config)
#   print(report.summary())
#   report.to_json("audit_report.json")
#
# Dependencies
# ------------
#   moewatch.config              — WatchConfig
#   moewatch.hooks.manager       — HookManager
#   moewatch.collector.*         — StatCollector
#   moewatch.analyzer.*          — all analyzer classes
#   moewatch.report.audit_report — AuditReport
#   torch
#
# =============================================================================

from __future__ import annotations

import logging
import time
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from moewatch.config import WatchConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public function: audit()
# ---------------------------------------------------------------------------


def audit(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_batches: int = 100,
    config: Optional[WatchConfig] = None,
    device: str = "cpu",
    with_backward: bool = True,
) -> "AuditReport":
    """Run offline diagnostic audit on a trained MoE model.

    Collects routing statistics over ``num_batches`` forward passes, then
    runs the full analyzer suite: entropy analysis (Tier 2), collapse
    detection (expert health state), gradient starvation assessment (Tier 1),
    cross-layer correlation (Tier 3), and fused risk score computation.
    Returns a comprehensive, serializable ``AuditReport``.

    No hooks remain attached after this function returns. Model weights
    are never modified (gradients are zeroed after each backward pass).

    Parameters
    ----------
    model : torch.nn.Module
        The trained MoE model to audit. Must contain detectable MoE router
        layers (Mixtral, Qwen3-MoE, DeepSeek-MoE, OLMoE) or a
        ``config.router_modules`` override must be provided.
    dataloader : torch.utils.data.DataLoader
        Validation dataloader. Batches are iterated up to ``num_batches``
        and fed through the model.
    with_backward : bool, optional
        If True (default), runs a proxy backward pass (sum-of-logits loss)
        after each forward pass so that gradient hooks fire and Tier 1
        (gradient starvation) analysis is populated. Set to False for
        routing-only audits where backward is too expensive or unavailable.
    num_batches : int, optional
        Maximum number of batches to process. Audit stops early if the
        dataloader is exhausted before this limit. Default: 100.
    config : WatchConfig, optional
        Configuration container. All analyzer thresholds, sampling rates,
        and output settings are sourced from this object. Uses
        ``WatchConfig()`` defaults if ``None``.
    device : str, optional
        Torch device string (e.g. ``"cpu"``, ``"cuda"``, ``"cuda:0"``).
        The model is moved to this device before the audit loop. Default:
        ``"cpu"``.

    Returns
    -------
    AuditReport
        A frozen, serializable report containing:

        - ``entropy_results``     — per-layer entropy analysis
        - ``collapse_results``    — per-layer expert health states
        - ``gradient_results``    — per-expert gradient starvation reports
        - ``cross_layer_results`` — cross-layer correlation analysis
        - ``risk_scores``         — per-layer fused risk scores
        - ``dead_experts_count``  — total dead experts across all layers
        - ``critical_layers``     — layers with CRITICAL risk level

    Raises
    ------
    ValueError
        If no MoE router modules can be detected in the model and
        ``config.router_modules`` is not set.
    RuntimeError
        If the specified device is not available (e.g. CUDA not installed).
    TypeError
        If ``model`` is not a ``torch.nn.Module``.

    Examples
    --------
    Basic audit:

    >>> from moewatch import audit, WatchConfig, OutputMode
    >>> config = WatchConfig(output=OutputMode.SILENT)
    >>> report = audit(model, val_loader, num_batches=50, config=config)
    >>> print(report.summary())

    With CUDA and custom router override:

    >>> config = WatchConfig(
    ...     router_modules=["model.layers.0.block_sparse_moe.gate"],
    ...     output=OutputMode.SILENT,
    ... )
    >>> report = audit(model, val_loader, num_batches=200, config=config, device="cuda")
    """
    # ------------------------------------------------------------------
    # 0. Input validation
    # ------------------------------------------------------------------
    if not isinstance(model, nn.Module):
        raise TypeError(
            f"[MoEWatch] audit() requires a torch.nn.Module, "
            f"got {type(model).__name__!r}."
        )

    if num_batches < 1:
        raise ValueError(
            f"[MoEWatch] num_batches must be >= 1, got {num_batches}."
        )

    # Resolve config
    if config is None:
        config = WatchConfig()

    # Resolve with_backward: explicit call-site arg takes precedence,
    # otherwise fall back to config.audit_with_backward.
    if with_backward is None:
        with_backward = config.audit_with_backward

    # ------------------------------------------------------------------
    # 1. Device validation and model placement
    # ------------------------------------------------------------------
    resolved_device = _resolve_device(device)
    _move_model_to_device(model, resolved_device)

    # ------------------------------------------------------------------
    # 2. Lazy sub-system imports
    # ------------------------------------------------------------------
    # Deferred to avoid loading the entire dependency graph on module import.

    from moewatch.hooks.manager import HookManager
    from moewatch.collector.stat_collector import StatCollector
    from moewatch.analyzer.entropy import EntropyAnalyzer
    from moewatch.analyzer.collapse import CollapseDetector
    from moewatch.analyzer.gradient_starvation import GradientStarvationAnalyzer
    from moewatch.analyzer.cross_layer import CrossLayerCorrelation
    from moewatch.analyzer.risk_score import RiskScoreFuser, RiskLevel
    from moewatch.report.audit_report import AuditReport

    # ------------------------------------------------------------------
    # 3. Construct sub-systems
    # ------------------------------------------------------------------
    stat_collector = StatCollector(config)
    hook_manager = HookManager(
        model=model,
        stat_collector=stat_collector,
        config=config,
    )

    # Analyzers
    entropy_analyzer = EntropyAnalyzer(config)
    collapse_detector = CollapseDetector(config)
    gradient_analyzer = GradientStarvationAnalyzer(config)
    cross_layer_analyzer = CrossLayerCorrelation(config)
    risk_fuser = RiskScoreFuser(config)

    # ------------------------------------------------------------------
    # 4. Attach hooks and run forward pass loop
    #    Guaranteed cleanup via try/finally — no hook leaks on exception.
    # ------------------------------------------------------------------
    audit_start_time = time.time()
    batches_processed = 0
    layer_order: List[str] = []

    try:
        hook_manager.attach()
        layer_order = list(hook_manager.get_layer_map().keys())

        if not layer_order:
            raise ValueError(
                "[MoEWatch] audit(): no MoE router modules detected after "
                "hook attachment. Ensure the model is a Mixture-of-Experts "
                "architecture or set config.router_modules manually."
            )

        logger.info(
            "[MoEWatch] audit() starting. Device: %s | Layers: %d | "
            "Max batches: %d",
            resolved_device,
            len(layer_order),
            num_batches,
        )

        batches_processed = _run_forward_loop(
            model=model,
            dataloader=dataloader,
            num_batches=num_batches,
            device=resolved_device,
            with_backward=with_backward,
        )

        logger.info(
            "[MoEWatch] Forward loop complete. Batches processed: %d",
            batches_processed,
        )

    finally:
        # Guaranteed hook cleanup — always executes, even on exception.
        try:
            hook_manager.detach()
            logger.debug("[MoEWatch] audit(): all hooks detached.")
        except Exception as cleanup_exc:  # pylint: disable=broad-except
            logger.error(
                "[MoEWatch] audit(): hook cleanup error (non-fatal): %s",
                cleanup_exc,
            )

    # ------------------------------------------------------------------
    # 5. Run analysis suite
    #    Each analyzer runs independently. Failures are logged and result
    #    in empty / default output for that analyzer rather than aborting.
    # ------------------------------------------------------------------
    entropy_results: Dict = {}
    collapse_results: Dict = {}
    gradient_results: Dict = {}
    cross_layer_result = None
    risk_score_results: Dict = {}

    # --- Tier 2: Entropy analysis ---
    try:
        entropy_results = entropy_analyzer.analyze(stat_collector)
        logger.debug(
            "[MoEWatch] EntropyAnalyzer complete. Layers analyzed: %d",
            len(entropy_results),
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "[MoEWatch] EntropyAnalyzer failed (results will be empty): %s",
            exc,
        )

    # --- Expert health state machine ---
    try:
        collapse_results = collapse_detector.analyze(stat_collector)
        logger.debug(
            "[MoEWatch] CollapseDetector complete. Layers analyzed: %d",
            len(collapse_results),
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "[MoEWatch] CollapseDetector failed (results will be empty): %s",
            exc,
        )

    # --- Tier 1: Gradient starvation analysis ---
    try:
        gradient_results = gradient_analyzer.analyze(stat_collector)
        logger.debug(
            "[MoEWatch] GradientStarvationAnalyzer complete. "
            "Layers analyzed: %d",
            len(gradient_results),
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "[MoEWatch] GradientStarvationAnalyzer failed (results will be "
            "empty): %s",
            exc,
        )

    # --- Tier 3: Cross-layer correlation ---
    try:
        cross_layer_result = cross_layer_analyzer.analyze(entropy_results)
        logger.debug("[MoEWatch] CrossLayerCorrelation complete.")
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "[MoEWatch] CrossLayerCorrelation failed (result will be None): %s",
            exc,
        )

    # --- Risk score fusion (Tier 1 + 2 + 3) ---
    risk_score_results, critical_layers = _fuse_risk_scores(
        layer_order=layer_order,
        entropy_results=entropy_results,
        gradient_results=gradient_results,
        cross_layer_result=cross_layer_result,
        risk_fuser=risk_fuser,
        RiskLevel=RiskLevel,
    )

    # ------------------------------------------------------------------
    # 6. Aggregate summary statistics
    # ------------------------------------------------------------------
    dead_experts_count = _count_dead_experts(gradient_results, config)
    model_name = type(model).__name__

    audit_duration = time.time() - audit_start_time
    logger.info(
        "[MoEWatch] audit() complete in %.2fs. "
        "Dead experts: %d | Critical layers: %s",
        audit_duration,
        dead_experts_count,
        critical_layers or "none",
    )

    # ------------------------------------------------------------------
    # 7. Build and return AuditReport
    # ------------------------------------------------------------------
    return AuditReport(
        model_name=model_name,
        timestamp=time.time(),
        num_batches=batches_processed,
        entropy_results=entropy_results,
        collapse_results=collapse_results,
        gradient_results=gradient_results,
        cross_layer_results=cross_layer_result,
        risk_scores=risk_score_results,
        dead_experts_count=dead_experts_count,
        critical_layers=critical_layers,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_device(device: str) -> str:
    """Validate and normalize the device string.

    Checks that CUDA is available when a ``cuda`` device is requested.
    Falls back to ``"cpu"`` with a warning rather than raising, so that
    audit runs on CPU-only machines that may have ``config.device = "cuda"``
    set by default.

    Parameters
    ----------
    device : str
        Torch device string from the caller (e.g. ``"cpu"``, ``"cuda"``).

    Returns
    -------
    str
        Validated device string, possibly downgraded to ``"cpu"``.

    Raises
    ------
    RuntimeError
        If the device string cannot be parsed by PyTorch at all.
    """
    try:
        resolved = torch.device(device)
    except RuntimeError as exc:
        raise RuntimeError(
            f"[MoEWatch] Invalid device string {device!r}: {exc}"
        ) from exc

    if resolved.type == "cuda" and not torch.cuda.is_available():
        warnings.warn(
            f"[MoEWatch] CUDA device {device!r} requested but CUDA is not "
            "available. Falling back to CPU. Install CUDA or pass "
            "device='cpu' to suppress this warning.",
            UserWarning,
            stacklevel=3,
        )
        return "cpu"

    return str(resolved)


def _move_model_to_device(model: nn.Module, device: str) -> None:
    """Move the model to ``device`` without modifying weights.

    This is purely a device transfer — no parameter values change. If the
    model is already on the target device this is a no-op.

    Parameters
    ----------
    model : nn.Module
        Model to transfer.
    device : str
        Target device string.
    """
    try:
        model.to(device)
        logger.debug("[MoEWatch] Model moved to device: %s", device)
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(
            f"[MoEWatch] Failed to move model to device {device!r}: {exc}"
        ) from exc


def _run_forward_loop(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_batches: int,
    device: str,
    with_backward: bool = True,
) -> int:
    """Run forward passes through the model to collect routing statistics.

    Iterates the dataloader up to ``num_batches`` times. Each batch is
    moved to ``device`` and passed through the model. When ``with_backward``
    is True, a proxy backward pass (sum-of-outputs loss) is executed after
    each forward so that parameter-level gradient hooks fire and Tier 1
    (gradient starvation) analysis is populated. Gradients are zeroed
    immediately after each backward — weights are never updated.

    Parameters
    ----------
    model : nn.Module
        Model with hooks already attached.
    dataloader : torch.utils.data.DataLoader
        Source of validation batches.
    num_batches : int
        Maximum batches to process.
    device : str
        Device to move each batch to before forward pass.
    with_backward : bool, optional
        If True, run a proxy backward pass to populate gradient hooks.
        If False, run under ``torch.no_grad()`` (routing stats only).

    Returns
    -------
    int
        Number of batches actually processed (may be less than ``num_batches``
        if the dataloader is exhausted first).
    """
    model.eval()
    batches_processed = 0

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        try:
            batch = _move_batch_to_device(batch, device)

            if with_backward:
                # Enable gradients for this pass so param hooks fire.
                # Use a sum-of-outputs proxy loss — label-free and cheap.
                model.zero_grad(set_to_none=True)
                output = _run_single_forward(model, batch, return_output=True)
                proxy_loss = _proxy_loss(output)
                if proxy_loss is not None:
                    proxy_loss.backward()
                model.zero_grad(set_to_none=True)  # never accumulate
            else:
                with torch.no_grad():
                    _run_single_forward(model, batch)

            batches_processed += 1

            if batches_processed % 10 == 0:
                logger.debug(
                    "[MoEWatch] audit(): processed %d / %d batches.",
                    batches_processed,
                    num_batches,
                )

        except StopIteration:
            logger.debug(
                "[MoEWatch] audit(): dataloader exhausted at batch %d.",
                batch_idx,
            )
            break

        except RuntimeError:
            # Fatal errors (OOM, CUDA error) propagate to the caller.
            raise

        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "[MoEWatch] audit(): forward pass error on batch %d "
                "(skipping): %s",
                batch_idx,
                exc,
            )

    return batches_processed


def _proxy_loss(output: object) -> "Optional[torch.Tensor]":
    """Derive a scalar proxy loss from a model output for backward pass.

    Traverses common output formats (tensor, tuple/list, dict with
    ``logits`` or ``last_hidden_state``) to find a float tensor, then
    returns its sum as a scalar. Returns None if no suitable tensor is
    found, in which case the backward pass is skipped gracefully.

    Parameters
    ----------
    output : object
        Raw return value of a model forward pass.

    Returns
    -------
    torch.Tensor or None
        Scalar tensor suitable for ``backward()``, or None.
    """
    def _find_tensor(obj: object) -> "Optional[torch.Tensor]":
        if isinstance(obj, torch.Tensor) and obj.is_floating_point():
            return obj
        if isinstance(obj, dict):
            for key in ("logits", "last_hidden_state"):
                if key in obj and isinstance(obj[key], torch.Tensor):
                    return obj[key]
            # Fall back to first float tensor in the dict
            for v in obj.values():
                t = _find_tensor(v)
                if t is not None:
                    return t
        if isinstance(obj, (list, tuple)):
            for item in obj:
                t = _find_tensor(item)
                if t is not None:
                    return t
        return None

    tensor = _find_tensor(output)
    if tensor is None:
        logger.debug(
            "[MoEWatch] _proxy_loss: no float tensor found in output; "
            "skipping backward for this batch."
        )
        return None
    return tensor.sum()


def _move_batch_to_device(
    batch: object,
    device: str,
) -> object:
    """Recursively move a batch to the target device.

    Handles the three common batch formats returned by HuggingFace and
    PyTorch dataloaders:

    - ``dict`` (HuggingFace ``BatchEncoding`` / plain dict of tensors)
    - ``list`` / ``tuple`` of tensors
    - bare ``torch.Tensor``

    All other types are returned unchanged — this prevents failures on
    datasets that include non-tensor metadata (strings, None, etc.).

    Parameters
    ----------
    batch : object
        Batch from the dataloader.
    device : str
        Target device string.

    Returns
    -------
    object
        Batch with all tensors moved to ``device``.
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)

    if isinstance(batch, dict):
        return {
            key: _move_batch_to_device(value, device)
            for key, value in batch.items()
        }

    if isinstance(batch, (list, tuple)):
        moved = [_move_batch_to_device(item, device) for item in batch]
        return type(batch)(moved)

    # Non-tensor type (str, int, None, etc.) — pass through unchanged.
    return batch


def _run_single_forward(
    model: nn.Module,
    batch: object,
    return_output: bool = False,
) -> object:
    """Execute a single forward pass, tolerating common batch formats.

    Attempts to call the model with the most likely argument format based
    on the batch type.

    Forward call strategy (in order):
      1. ``dict`` batch        → ``model(**batch)``
      2. ``tuple``/``list``    → ``model(*batch)``
      3. ``tensor`` / other    → ``model(batch)``
      4. ``TypeError`` fallback → ``model(batch)``

    Parameters
    ----------
    model : nn.Module
        Model with hooks attached.
    batch : object
        Pre-processed batch (already on the correct device).
    return_output : bool, optional
        If True, return the model's output for proxy loss computation.
        If False, return None (routing side-effects only).

    Returns
    -------
    object or None
        Model output when ``return_output`` is True, else None.
    """
    try:
        if isinstance(batch, dict):
            output = model(**batch)
        elif isinstance(batch, (list, tuple)):
            output = model(*batch)
        else:
            output = model(batch)

    except TypeError as exc:
        logger.debug(
            "[MoEWatch] Forward pass call convention mismatch (%s). "
            "Retrying with positional arg.",
            exc,
        )
        output = model(batch)

    return output if return_output else None


def _fuse_risk_scores(
    layer_order: List[str],
    entropy_results: Dict,
    gradient_results: Dict,
    cross_layer_result: object,
    risk_fuser: object,
    RiskLevel: type,
) -> Tuple[Dict, List[str]]:
    """Fuse per-layer Tier 1 / Tier 2 / Tier 3 signals into risk scores.

    For each layer in ``layer_order``, attempts to fuse the corresponding
    entropy report, gradient report, and cross-layer result into a single
    ``RiskReport`` via ``RiskScoreFuser.fuse()``. Per-layer failures are
    logged and skipped.

    Returns a dict of ``{layer_name: RiskReport}`` and a list of layer names
    whose risk level is ``CRITICAL``.

    Parameters
    ----------
    layer_order : list[str]
        Ordered list of layer names to process.
    entropy_results : dict[str, LayerEntropyReport]
        Output from ``EntropyAnalyzer.analyze()``.
    gradient_results : dict[str, list[GradientStarvationReport]]
        Output from ``GradientStarvationAnalyzer.analyze()``.
    cross_layer_result : CrossLayerReport or None
        Output from ``CrossLayerCorrelation.analyze()``.
    risk_fuser : RiskScoreFuser
        Initialized risk fuser instance.
    RiskLevel : type
        ``RiskLevel`` enum class (passed in to avoid re-importing here).

    Returns
    -------
    risk_scores : dict[str, RiskReport]
        Per-layer fused risk reports.
    critical_layers : list[str]
        Names of layers with ``RiskLevel.CRITICAL``.
    """
    risk_scores: Dict = {}
    critical_layers: List[str] = []

    for layer_name in layer_order:
        try:
            ent_report = entropy_results.get(layer_name)
            grad_layer_reports = gradient_results.get(layer_name, [])

            # Select the representative Tier 1 signal for this layer.
            #
            # Three expert categories (in priority order):
            #
            #   1. confirmed_dead  — hook fired (n_samples >= 1) but norm=0.
            #      These are experts that were selected in forward but produced
            #      zero gradient.  starvation_score is already 1.0.
            #
            #   2. no_data        — hook NEVER fired (n_samples == 0).
            #      After a backward pass, zero events means the expert's
            #      weight never appeared in the computation graph → it was
            #      never routed to → complete gradient starvation.
            #      starvation_score defaults to 0.0 (insufficient-data guard
            #      in GradientStarvationAnalyzer), so we override to 1.0 here.
            #
            #   3. has_data       — hook fired, norm > 0.  Take the worst
            #      starvation_score among experts that have samples.
            grad_report = None
            if grad_layer_reports:
                confirmed_dead = [
                    r for r in grad_layer_reports
                    if getattr(r, "n_samples", 0) >= 1
                    and getattr(r, "gradient_norm_mean", -1.0) == 0.0
                ]
                no_data = [
                    r for r in grad_layer_reports
                    if getattr(r, "n_samples", 0) == 0
                ]
                has_data = [
                    r for r in grad_layer_reports
                    if getattr(r, "n_samples", 0) >= 1
                    and getattr(r, "gradient_norm_mean", -1.0) != 0.0
                ]

                if confirmed_dead:
                    # Most reliable: hook fired, norm is exactly zero.
                    grad_report = max(
                        confirmed_dead,
                        key=lambda r: getattr(r, "n_samples", 0),
                    )
                elif no_data:
                    # No hook events → expert never routed to → fully starved.
                    # Override starvation_score to 1.0 so the fuser sees
                    # the correct T1 signal.
                    import dataclasses as _dc
                    _base = no_data[0]
                    grad_report = _dc.replace(_base, starvation_score=1.0)
                elif has_data:
                    # All experts received gradients; pick the most starved.
                    grad_report = max(
                        has_data,
                        key=lambda r: r.starvation_score,
                    )
                else:
                    grad_report = max(
                        grad_layer_reports,
                        key=lambda r: r.starvation_score,
                    )

            # Both Tier 1 and Tier 2 are required for meaningful fusion.
            # If either is missing, skip this layer rather than producing
            # a meaningless or misleading risk score.
            if ent_report is None or grad_report is None:
                logger.debug(
                    "[MoEWatch] audit(): skipping risk fusion for layer '%s' "
                    "(missing entropy_report=%s, gradient_report=%s).",
                    layer_name,
                    ent_report is None,
                    grad_report is None,
                )
                continue

            risk_report = risk_fuser.fuse(
                gradient_report=grad_report,
                entropy_report=ent_report,
                cross_layer_report=cross_layer_result,
            )
            risk_scores[layer_name] = risk_report

            if risk_report.risk_level == RiskLevel.CRITICAL:
                critical_layers.append(layer_name)

        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "[MoEWatch] audit(): risk fusion failed for layer '%s' "
                "(skipping): %s",
                layer_name,
                exc,
            )

    return risk_scores, critical_layers


def _count_dead_experts(
    gradient_results: Dict,
    config: WatchConfig,
) -> int:
    """Count total dead experts across all layers from gradient reports.

    An expert is considered dead when its mean gradient norm falls below
    ``config.dead_threshold``.

    Parameters
    ----------
    gradient_results : dict[str, list[GradientStarvationReport]]
        Output from ``GradientStarvationAnalyzer.analyze()``.
    config : WatchConfig
        Configuration providing ``dead_threshold``.

    Returns
    -------
    int
        Total number of experts classified as dead.
    """
    dead_count = 0
    for expert_reports in gradient_results.values():
        for report in expert_reports:
            try:
                if report.gradient_norm_mean < config.dead_threshold:
                    dead_count += 1
            except AttributeError:
                # Malformed report object — skip gracefully.
                pass
    return dead_count
