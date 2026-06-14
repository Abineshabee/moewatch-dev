# MoEWatch — Project Structure

---

## `.github/`

| File | Purpose |
|------|---------|
| `.github/workflows/ci.yml` | Main CI pipeline. Runs on every push and PR. Jobs: lint (ruff), type-check (mypy), test matrix (Python 3.9/3.10/3.11, CPU-only), coverage report upload to Codecov. Fails fast on any error. |
| `.github/workflows/release.yml` | Automated PyPI release pipeline. Triggered on GitHub release tag (v*.*.*). Builds wheel + sdist via pyproject.toml, uploads to PyPI with OIDC trusted publishing. No API keys stored in repo. |
| `.github/workflows/benchmark.yml` | Weekly scheduled benchmark run on self-hosted GPU runner. Executes benchmarks/overhead_profile.py and benchmarks/lead_time_benchmark.py. Posts regression comment on PR if overhead increases >0.5%. |
| `.github/workflows/docs.yml` | Builds and deploys MkDocs documentation to GitHub Pages on every push to main. Runs mkdocs build --strict to catch broken links. |
| `.github/ISSUE_TEMPLATE/bug_report.yml` | Bug report template. Fields: model family, MoEWatch version, Python version, PyTorch version, minimal reproducible example, expected vs actual behavior. |
| `.github/ISSUE_TEMPLATE/feature_request.yml` | Feature request template. Fields: motivation, proposed API, research context (paper link if applicable). |
| `.github/ISSUE_TEMPLATE/repro_issue.yml` | Reproducibility issue template specifically for paper experiments. Tracks which experiment, which model, which seed, and what diverged from reported results. |
| `.github/PULL_REQUEST_TEMPLATE.md` | PR template. Sections: what changes, which tests added, does it affect public API, does it affect paper experiments. |

---

## `moewatch/` — Core Package

### Top-level

| File | Purpose | Exports | Deps |
|------|---------|---------|------|
| `moewatch/__init__.py` | Public API surface. Exports all public symbols. Lazy imports for fast startup. Guards for torch and transformers availability. NO_COLOR env support. | `MoEWatch`, `MoEWatchCallback`, `audit`, `WatchConfig`, `OutputMode`, `AlertLevel`, `Alert`, `__version__` | `moewatch/_watcher.py`, `moewatch/_audit.py`, `moewatch/config.py` |
| `moewatch/_watcher.py` | MoEWatch class — live training-time monitor. Manages full lifecycle: start(), stop(), attach(trainer), step(global_step). Coordinates HookManager, StatCollector, all analyzers, intervention engine, and policy. Zero weight modification guarantee. Thread-safe alert emission. Context manager support. | `MoEWatch`, `MoEWatchCallback`, `Alert` | `moewatch/config.py`, `moewatch/hooks/manager.py`, `moewatch/collector/stat_collector.py`, `moewatch/analyzer/`, `moewatch/intervention/`, `moewatch/policy/`, `moewatch/report/` |
| `moewatch/_audit.py` | audit() function — offline post-training diagnostic. Accepts model + dataloader + steps. Runs N forward passes, collects routing stats, runs full analyzer suite (entropy, collapse, gradient proxy), returns AuditReport. No hooks left attached after call. CPU-safe. | `audit` | `moewatch/config.py`, `moewatch/hooks/manager.py`, `moewatch/collector/stat_collector.py`, `moewatch/analyzer/`, `moewatch/report/audit_report.py` |
| `moewatch/config.py` | WatchConfig dataclass — all tunable parameters in one place. Fields: dead_threshold, cold_threshold, cold_steps_limit, entropy_warn, entropy_critical, entropy_drop_warn, load_imbalance_warn, load_imbalance_error, log_every, sample_every, output, no_color, router_modules, intervention_enabled, policy_type (rule/bandit), bandit_epsilon, reward_discount_gamma, reward_window_steps, baseline_min_clean_steps, baseline_exclusion_window, intervention_cooldown, intervention_max_delta, loss_guard_threshold. Validated on construction. | `WatchConfig`, `OutputMode`, `AlertLevel` | — |

---

### `moewatch/hooks/`

| File | Purpose | Exports | Deps |
|------|---------|---------|------|
| `moewatch/hooks/__init__.py` | Re-exports HookManager, RouterForwardHook, GradientStarvationHook, detect_router_modules. | `HookManager`, `RouterForwardHook`, `GradientStarvationHook`, `detect_router_modules` | — |
| `moewatch/hooks/manager.py` | HookManager — owns all hook handles. attach() registers forward + backward hooks on detected router modules. detach() cleanly removes all handles. Guaranteed no leaked hooks even on exception (uses finally). Tracks handle list for safe teardown. | `HookManager` | `moewatch/hooks/router_hook.py`, `moewatch/hooks/gradient_hook.py`, `moewatch/hooks/detection.py`, `moewatch/collector/stat_collector.py` |
| `moewatch/hooks/router_hook.py` | RouterForwardHook — forward hook on router modules. Captures output tensor (routing logits or top-k indices), infers expert count, constructs RoutingEvent with timestamp and step. Writes to StatCollector ring buffer. Designed to be on the hot forward path: minimal allocation, no Python-level loops over tokens. | `RouterForwardHook`, `RoutingEvent` | `moewatch/collector/stat_collector.py`, `moewatch/config.py` |
| `moewatch/hooks/gradient_hook.py` | *(v0.2.0)* GradientStarvationHook — backward hook on expert weight tensors (register_full_backward_hook). Captures per-expert gradient L2 norm each backward pass. Writes GradientEvent to StatCollector. Sampling rate configurable (default: every 10 steps) to minimize overhead. Core data source for Tier 1 signal. | `GradientStarvationHook`, `GradientEvent` | `moewatch/collector/stat_collector.py`, `moewatch/config.py` |
| `moewatch/hooks/detection.py` | detect_router_modules() — auto-detects MoE router layers by module name patterns (gate, router, sparse_moe, block_sparse_moe) and output shape heuristics (2D float output with dim matching n_experts). Supports Mixtral, Qwen3-MoE, DeepSeek-MoE, OLMoE, and custom architectures via WatchConfig.router_modules override. | `detect_router_modules` | `moewatch/config.py` |

---

### `moewatch/collector/`

| File | Purpose | Exports | Deps |
|------|---------|---------|------|
| `moewatch/collector/__init__.py` | Re-exports StatCollector, LayerStats, RingBuffer, GradientStats. | `StatCollector`, `LayerStats`, `RingBuffer`, `GradientStats` | — |
| `moewatch/collector/ring_buffer.py` | RingBuffer — fixed-memory circular buffer for RoutingEvent and GradientEvent streams. Configurable capacity (default 1000 events per layer). Prevents unbounded memory growth during long training runs. Thread-safe write path (GIL-protected for Python-level append). Supports windowed reads for CUSUM and baseline computation. | `RingBuffer` | — |
| `moewatch/collector/stat_collector.py` | StatCollector — aggregates RoutingEvents into LayerStats (expert token counts, utilization, load imbalance score, raw logit window) and GradientEvents into GradientStats (per-expert gradient norm history). get_all_stats() returns a snapshot dict used by all analyzers. Thread-safe read path. | `StatCollector`, `LayerStats`, `GradientStats` | `moewatch/collector/ring_buffer.py`, `moewatch/config.py` |
| `moewatch/collector/baseline_tracker.py` | *(v0.2.0)* BaselineTracker — maintains intervention-conditioned counterfactual baseline per layer. Tracks which steps are "intervention-influenced" and excludes them from baseline regression window. Computes projected no-action trajectory via rolling linear regression on clean history. Exports compute_counterfactual_delta(layer, actual_value) → float. Critical for correct attribution. | `BaselineTracker` | `moewatch/config.py`, `moewatch/collector/ring_buffer.py` |

---

### `moewatch/analyzer/`

| File | Purpose | Exports | Deps |
|------|---------|---------|------|
| `moewatch/analyzer/__init__.py` | Re-exports EntropyAnalyzer, CollapseDetector, GradientStarvationAnalyzer, CrossLayerCorrelation, RiskScoreFuser. | `EntropyAnalyzer`, `CollapseDetector`, `GradientStarvationAnalyzer`, `CrossLayerCorrelation`, `RiskScoreFuser` | — |
| `moewatch/analyzer/entropy.py` | EntropyAnalyzer — per-layer Shannon entropy. compute_entropy(probs) pure function. CUSUM detector on entropy history for drift detection (upgrade from simple recent-vs-prior). Normalized entropy against H_max = log2(n_experts). Trend: DECLINING / STABLE / IMPROVING / UNKNOWN. Source: logits (preferred) or counts (fallback). Exports LayerEntropyReport. | `EntropyAnalyzer`, `compute_entropy`, `LayerEntropyReport`, `EntropyResult` | `moewatch/collector/stat_collector.py`, `moewatch/config.py` |
| `moewatch/analyzer/collapse.py` | CollapseDetector — expert health state machine: HEALTHY / COLD / DEAD / UNKNOWN. Cold-to-dead promotion via cold_steps_limit counter. Load imbalance score (max/mean ratio). Per-expert utilization and token count. Exports LayerCollapseReport. Stateful: maintains consecutive cold counters per layer per expert. | `CollapseDetector`, `ExpertStatus`, `LayerCollapseReport`, `ExpertState` | `moewatch/collector/stat_collector.py`, `moewatch/config.py` |
| `moewatch/analyzer/gradient_starvation.py` | *(v0.2.0)* GradientStarvationAnalyzer — Tier 1 primary signal. Consumes GradientStats from collector. Computes per-expert gradient norm rolling mean. Detects starvation: expert gradient norm < starvation_threshold for N consecutive steps. Outputs GradientStarvationReport with per-expert starvation_score (0.0–1.0) and starvation_onset_step estimate. This is the earliest and most actionable signal. | `GradientStarvationAnalyzer`, `GradientStarvationReport` | `moewatch/collector/stat_collector.py`, `moewatch/config.py` |
| `moewatch/analyzer/cross_layer.py` | *(v0.2.0)* CrossLayerCorrelation — Tier 3 localization signal. Maintains sliding window of per-layer entropy sequences. Computes pairwise Pearson correlation matrix over the window. Detects: source layer (highest entropy drop rate), downstream victims (high correlation with source), propagation velocity. Exports CrossLayerReport with propagation_graph and source_layer identification. | `CrossLayerCorrelation`, `CrossLayerReport` | `moewatch/analyzer/entropy.py`, `moewatch/config.py` |
| `moewatch/analyzer/risk_score.py` | *(v0.2.0)* RiskScoreFuser — fuses three tier signals into a single per-layer collapse risk score (0.0–1.0). Weighted sum: risk = w1*gradient_signal + w2*entropy_signal + w3*spread_signal where w1 > w2 > w3. Weights are configurable and validated (must sum to 1.0). Provides risk_score, risk_level (LOW/MID/HIGH/CRITICAL), and dominant_signal for explainability. | `RiskScoreFuser`, `RiskReport`, `RiskLevel` | `moewatch/analyzer/gradient_starvation.py`, `moewatch/analyzer/entropy.py`, `moewatch/analyzer/cross_layer.py`, `moewatch/config.py` |
| `moewatch/analyzer/cusum.py` | *(v0.2.0)* CUSUM detector — cumulative sum change-point detection for entropy and gradient norm sequences. detect_change(series, threshold, drift) → (change_detected, change_point_step, cusum_value). Used by EntropyAnalyzer and GradientStarvationAnalyzer as the core statistical detector for drift. Replaces naive recent-vs-prior comparison. | `CUSUMDetector`, `detect_change` | — |

---

### `moewatch/intervention/`

| File | Purpose | Exports | Deps |
|------|---------|---------|------|
| `moewatch/intervention/__init__.py` | Re-exports InterventionEngine, all Action classes, SafetyGuard. | `InterventionEngine`, `AuxLossAction`, `RouterNoiseAction`, `ExpertDropoutAction`, `NoOpAction`, `SafetyGuard` | — |
| `moewatch/intervention/engine.py` | *(v0.2.0)* InterventionEngine — receives selected action from Policy. Validates against SafetyGuard before applying. Applies action to live training hyperparameters. Records intervention in BaselineTracker (marks steps as influenced). Schedules revert-on-degradation check after observation window. Maintains cooldown state per layer. | `InterventionEngine` | `moewatch/intervention/actions.py`, `moewatch/intervention/safety.py`, `moewatch/collector/baseline_tracker.py`, `moewatch/config.py` |
| `moewatch/intervention/actions.py` | *(v0.2.0)* Intervention action classes. AuxLossAction: adjusts aux_loss_coef by delta (soft load balancing). RouterNoiseAction: injects Gaussian noise into router logits (forces exploration). ExpertDropoutAction: temporarily increases expert dropout probability. NoOpAction: do nothing, continue monitoring. All actions are reversible and log their effect. | `AuxLossAction`, `RouterNoiseAction`, `ExpertDropoutAction`, `NoOpAction`, `InterventionAction` | `moewatch/config.py` |
| `moewatch/intervention/safety.py` | *(v0.2.0)* SafetyGuard — pre-flight checks before any intervention executes. Checks: (1) cooldown not active for this layer, (2) intervention delta within max_delta limit, (3) training loss not spiking above loss_guard_threshold, (4) neighbor layers not already under intervention. If any check fails: action downgraded to NoOp and reason logged. This is the safety backstop. | `SafetyGuard`, `SafetyCheckResult` | `moewatch/config.py` |

---

### `moewatch/policy/`

| File | Purpose | Exports | Deps |
|------|---------|---------|------|
| `moewatch/policy/__init__.py` | Re-exports PolicyBase, RulePolicy, BanditPolicy, PolicyState, PolicyMemory. | `PolicyBase`, `RulePolicy`, `BanditPolicy`, `PolicyState`, `PolicyMemory` | — |
| `moewatch/policy/base.py` | *(v0.2.0)* PolicyBase — abstract base class for all policies. select_action(state: PolicyState) → InterventionAction. update(state, action, reward) → None. save_checkpoint(path) / load_checkpoint(path). Defines the interface that RulePolicy and BanditPolicy both implement. | `PolicyBase`, `PolicyState` | `moewatch/intervention/actions.py` |
| `moewatch/policy/rule_policy.py` | *(v0.2.0)* RulePolicy — Phase 1 deterministic policy. Maps risk_score ranges to fixed actions: <0.3→NoOp, <0.6→AuxLoss, <0.8→RouterNoise, ≥0.8→ExpertDropout. Also considers intervention_history to avoid oscillation and neighbor health for cascade prevention. Explainable, no randomness, safe default. | `RulePolicy` | `moewatch/policy/base.py`, `moewatch/analyzer/risk_score.py`, `moewatch/intervention/actions.py` |
| `moewatch/policy/bandit_policy.py` | *(v0.2.0)* BanditPolicy — Phase 2 contextual bandit. State vector: (risk_score, layer_id_encoded, training_step_normalized, intervention_history_embedding). Action space: {NoOp, AuxLoss, RouterNoise, ExpertDropout}. Algorithm: epsilon-greedy (default) or Thompson Sampling (via config). Reward: discounted counterfactual ΔH over observation window. Updates linear value estimates per action-context pair. Saves policy memory to checkpoint. | `BanditPolicy` | `moewatch/policy/base.py`, `moewatch/collector/baseline_tracker.py`, `moewatch/intervention/actions.py`, `moewatch/config.py` |
| `moewatch/policy/memory.py` | *(v0.2.0)* PolicyMemory — stores (state, action, reward) tuples across the training run. Supports replay for batch value updates. Serializable to JSON for cross-run transfer. Provides action_success_rate(action) and context_similarity(state) for Thompson Sampling priors. Maximum memory size configurable to prevent unbounded growth. | `PolicyMemory`, `ExperienceTuple` | — |
| `moewatch/policy/reward.py` | *(v0.2.0)* RewardComputer — computes discounted counterfactual reward for a completed intervention. reward = Σ ΔH_counterfactual(t+k) * γ^k for k=1..K. ΔH_counterfactual from BaselineTracker. Exponential discount factor γ = WatchConfig.reward_discount_gamma (default 0.95). Handles delayed reward window K = WatchConfig.reward_window_steps (default 50). | `RewardComputer` | `moewatch/collector/baseline_tracker.py`, `moewatch/config.py` |

---

### `moewatch/report/`

| File | Purpose | Exports | Deps |
|------|---------|---------|------|
| `moewatch/report/__init__.py` | Re-exports AuditReport, WatchReport, CLIReporter, JSONReporter. | `AuditReport`, `WatchReport`, `CLIReporter`, `JSONReporter` | — |
| `moewatch/report/audit_report.py` | AuditReport — structured result from audit(). Holds: per-layer entropy results, collapse reports, gradient starvation report, cross-layer propagation report, risk scores, intervention log (if any). Methods: summary(), dead_experts(), to_json(), to_dataframe(). Immutable dataclass. | `AuditReport` | `moewatch/analyzer/` |
| `moewatch/report/watch_report.py` | *(v0.2.0)* WatchReport — live training report generated at each step() call. Includes current risk scores, active interventions, policy decisions, counterfactual reward estimates, and alert log since last call. Replaces raw Alert list for richer introspection. to_json() for programmatic consumption. | `WatchReport`, `StepReport` | `moewatch/analyzer/risk_score.py`, `moewatch/policy/`, `moewatch/intervention/` |
| `moewatch/report/cli_reporter.py` | CLIReporter — ANSI terminal output. Renders: routing health dashboard, expert utilization histograms per layer, entropy bars with trend arrows, risk score meters, active intervention log, policy memory summary. NO_COLOR fallback. Respects terminal width. | `CLIReporter` | `moewatch/report/audit_report.py`, `moewatch/report/watch_report.py` |
| `moewatch/report/json_reporter.py` | *(v0.2.0)* JSONReporter — emits newline-delimited JSON for log aggregators (Grafana, Splunk, W&B custom metrics). Every step() produces one JSON line with all metrics, risk scores, and intervention decisions. Suitable for real-time dashboards and experiment tracking pipelines. | `JSONReporter` | `moewatch/report/watch_report.py` |

---

## `tests/`

| File | What it tests |
|------|--------------|
| `tests/conftest.py` | Pytest fixtures: SyntheticMoE (tiny 2-layer MoE with 4 experts), tiny_dataloader (random token tensors), default_config (WatchConfig with low thresholds for fast test triggering), collapsed_model (pre-baked dead expert state), gradient_starved_model (pre-baked starvation state). Makes every test GPU-free and self-contained. |
| `tests/test_detection.py` | Tests detect_router_modules() against: SyntheticMoE (standard naming), mock MixtralSparseMoeBlock, custom-named routers, and models with no detectable routers (expects RuntimeError). Verifies WatchConfig.router_modules override works. |
| `tests/test_hooks.py` | Tests HookManager: attach/detach lifecycle, context manager teardown, no leaked hooks after exception. Tests RouterForwardHook fires on forward pass and writes RoutingEvent. Tests GradientStarvationHook fires on backward pass and writes GradientEvent. |
| `tests/test_collector.py` | Tests RingBuffer capacity wrapping (old events evicted). Tests StatCollector expert token count aggregation (sum equals total tokens). Tests GradientStats rolling mean computation. Tests BaselineTracker exclusion window (post-intervention steps correctly excluded from baseline regression). |
| `tests/test_entropy.py` | Tests EntropyAnalyzer: uniform distribution (max entropy, INFO), one-hot (zero entropy, ERROR), partial collapse. Tests CUSUM detector: flat series (no change), monotone decline (change detected at correct step), noisy decline (robust detection). Tests trend direction transitions. |
| `tests/test_collapse.py` | Injects synthetic expert utilization with known dead expert (0.0%) and cold expert (0.3%). Asserts ExpertStatus.state == DEAD and COLD respectively. Tests cold-to-dead promotion after cold_steps_limit consecutive cold observations. Tests load imbalance score computation. |
| `tests/test_gradient_starvation.py` | *(v0.2.0)* Tests GradientStarvationAnalyzer: normal gradient norms (no alert), artificially zeroed expert gradients (starvation detected), partial starvation (only subset of experts). Verifies starvation_onset_step estimation accuracy on synthetic series. |
| `tests/test_cross_layer.py` | *(v0.2.0)* Tests CrossLayerCorrelation: independent layer entropy sequences (low correlation), synchronized collapse sequences (high correlation, correct source identified), propagation velocity estimation. Uses synthetic entropy time series. |
| `tests/test_risk_score.py` | *(v0.2.0)* Tests RiskScoreFuser: all-healthy inputs (score near 0), all-critical inputs (score near 1), partial inputs (correct weighted fusion). Verifies risk_level thresholds and dominant_signal identification. Tests weight validation (must sum to 1.0). |
| `tests/test_baseline_tracker.py` | *(v0.2.0)* Tests BaselineTracker: clean projection accuracy on linear series, contamination exclusion (post-intervention steps correctly excluded), minimum clean history enforcement (returns None when insufficient clean steps), reset after intervention cooldown. |
| `tests/test_policy.py` | *(v0.2.0)* Tests RulePolicy: correct action selection at each risk level, oscillation prevention (same action not repeated immediately), neighbor health check. Tests BanditPolicy: action exploration (epsilon-greedy fires), policy update on positive reward, checkpoint save/load round-trip. |
| `tests/test_intervention.py` | *(v0.2.0)* Tests InterventionEngine: SafetyGuard blocks action when cooldown active, blocks when loss spike detected, blocks when delta exceeds max. Tests action application (AuxLoss changes hyperparameter, RouterNoise injects noise). Tests revert logic on degradation. |
| `tests/test_watcher.py` | Integration test for MoEWatch end-to-end. Tests attach/detach, step() with pre-baked collapsed stats, TrainerCallback step forwarding, intervention firing at correct risk threshold, alert log accumulation, WatchReport contents. |
| `tests/test_audit.py` | Integration test for audit() end-to-end on CPU. AuditReport non-empty, dead_experts() returns list, gradient starvation report populated, cross-layer report populated, to_json() is valid JSON, no hooks leaked after completion. |

---

## `benchmarks/`

| File | Purpose |
|------|---------|
| `benchmarks/overhead_profile.py` | Measures MoEWatch forward+backward hook overhead vs unmonitored baseline. Runs 500 timed iterations. Reports mean latency, p95 latency, tokens/sec, peak GPU memory delta. Target: <3% overhead on forward pass, <5% on backward. Outputs overhead_results.json. |
| `benchmarks/lead_time_benchmark.py` | Measures T_moewatch vs T_utilization vs T_entropy vs T_aux_loss across 20 training runs with different random seeds. Computes mean and std of lead-time advantage for each baseline. Core empirical proof of the paper claim. Outputs lead_time_results.json. |
| `benchmarks/ies_benchmark.py` | Measures Intervention Efficiency Score = stability_improvement / intervention_cost across all action types. Runs controlled experiments: train with MoEWatch interventions vs train without. Computes dead expert rate reduction, loss trajectory deviation, compute overhead. Outputs ies_results.json. |
| `benchmarks/synthetic_collapse_injector.py` | Synthetic collapse injection framework. Programmatically forces routing collapse at a known step by biasing router weights. Used to create ground-truth collapse events for lead-time benchmark. Supports: gradual collapse, sudden collapse, multi-layer cascade collapse. Critical for reproducible benchmarks without waiting for natural collapse. |

---

## `experiments/`

| File | Paper output |
|------|-------------|
| `experiments/exp1_lead_time_advantage.py` | Experiment 1: Lead-time advantage across model families. Trains small MoE (4 experts, 6 layers) on Wikitext-103. Lets collapse happen naturally. Records T for MoEWatch, utilization baseline, entropy baseline, aux loss baseline. Generates Table 1 of paper. |
| `experiments/exp2_signal_hierarchy.py` | Experiment 2: Validates empirical signal ordering (Tier 1 earliest, Tier 3 latest). Measures information gain of each signal at each time step before collapse. Generates Figure 2 of paper (signal lead-time distribution plot). |
| `experiments/exp3_intervention_efficacy.py` | Experiment 3: Intervention efficacy comparison. Compares: no intervention, rule policy, bandit policy, oracle policy (always optimal in hindsight). Metric: dead expert rate at end of training. Generates Table 2 and Figure 3. |
| `experiments/exp4_counterfactual_attribution.py` | Experiment 4: Validates counterfactual baseline. Shows naive reward assignment (no baseline) misattributes credit vs intervention-conditioned baseline. Compares policy learning quality with and without BaselineTracker. Generates Figure 4. |
| `experiments/exp5_overhead_analysis.py` | Experiment 5: Computational overhead analysis. Measures wall-clock overhead per model size (small, medium, large MoE). Generates overhead table for paper appendix. Validates <3% forward overhead claim. |
| `experiments/README.md` | Experiment reproduction guide. Hardware requirements, estimated runtime, how to run each experiment, expected outputs, how to regenerate paper figures from result JSON files. |

---

## `examples/`

| File | Demonstrates |
|------|-------------|
| `examples/mixtral_training_monitor.py` | Mixtral 8x7B live training monitor. Attach MoEWatch to HuggingFace Trainer, enable intervention engine, run 1000 steps, print WatchReport summary. Shows: routing health dashboard, any interventions fired, policy decisions. |
| `examples/deepseek_audit.py` | DeepSeek-MoE offline audit. Load pretrained model, run audit() on validation set, inspect AuditReport for dead experts and gradient starvation. Shows manual router module override for non-standard naming. |
| `examples/qwen3_moe_finetune.py` | Qwen3-MoE fine-tuning with MoEWatch. Shows MoEWatch during SFT fine-tuning, where collapse from domain shift is a real risk. Demonstrates intervention firing during fine-tuning, not just pretraining. |
| `examples/custom_moe_example.py` | Custom PyTorch MoE architecture integration. Shows WatchConfig(router_modules=[...]) manual override for architectures that do not match auto-detection patterns. Minimal MoE implementation included for reference. |
| `examples/json_output_pipeline.py` | JSONReporter integration example. Pipes MoEWatch JSON output to a file, then reads it with pandas for post-hoc analysis. Shows how to build a lightweight routing health dashboard using matplotlib. |

---

## `notebooks/`

| File | Purpose |
|------|---------|
| `notebooks/quickstart_colab.ipynb` | Primary Colab demo. Runs on free T4 GPU. Loads OLMoE-1B-7B, attaches MoEWatch with intervention enabled, runs 200 training steps, renders full routing health dashboard. Includes synthetic collapse demo that runs on CPU without model download. |
| `notebooks/synthetic_collapse_demo.ipynb` | CPU-only interactive demo. Creates tiny MoE, uses CollapseInjector to force collapse at step 500, shows MoEWatch detecting Tier 1 signal at step ~320 (180 steps early), intervention firing at step 350, routing recovery by step 600. Visualizes all three tier signals over time. |
| `notebooks/paper_figures.ipynb` | Notebook that reads experiment result JSON files and generates all paper figures. Fully reproducible figure generation. Figures: signal lead-time distributions, intervention efficacy comparison, overhead analysis, counterfactual attribution validation. |

---

## `docs/`

| File | Purpose |
|------|---------|
| `mkdocs.yml` | MkDocs configuration. Theme: Material for MkDocs (dark mode default). Nav structure: Getting Started, API Reference, Architecture, Research, Contributing. Plugins: mkdocstrings (auto API docs from docstrings). |
| `docs/index.md` | Documentation home page. Problem statement, solution overview, quick install, 5-line usage example, link to Colab demo. First page new users see. |
| `docs/architecture.md` | Architecture deep-dive. Explains all seven components: OBSERVE/PREDICT/ATTRIBUTE/PROPOSE/SELECT/APPLY/LEARN. Signal tier hierarchy. Counterfactual baseline design. Policy phases. Diagrams included. |
| `docs/api_reference.md` | Full API reference. Auto-generated from docstrings via mkdocstrings. Covers: MoEWatch, audit, WatchConfig, all analyzers, all intervention actions, policy classes, report classes. |
| `docs/research.md` | Research context page. Problem statement, novelty claims, related work comparison, evaluation metrics (lead-time, IES), experiment results summary, citation. |
| `docs/contributing.md` | Contributor guide. Dev setup, running tests, code style (ruff + mypy), adding new signal analyzers, adding new intervention actions, adding new model family support. Includes architecture diagram for contributors. |

---

## Root Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | PEP 517 build config. Build backend: hatchling. Dependencies: torch>=0.2.0, numpy. Optional: transformers (for HF integration). Dev extras: pytest, pytest-cov, ruff, mypy, mkdocs-material, mkdocstrings. Version: 0.2.0.0. Package metadata: homepage, license, keywords. |
| `README.md` | Primary GitHub README. Sections: one-liner, badges (PyPI, CI, coverage, license), problem statement (routing collapse), what MoEWatch does (predict+intervene+learn), 5-line usage example, architecture overview, signal tier table, installation, links to paper/docs/Colab. |
| `CONTRIBUTING.md` | Contribution guide at repo root. Quick version of docs/contributing.md. Points to full guide. Includes: how to file bugs, how to propose features, code style, PR checklist, CLA info. |
| `CITATION.cff` | Citation metadata file (Citation File Format). Allows GitHub to show "Cite this repository" button. Contains: title, authors, version, DOI (once paper is published), preferred citation format. |
| `LICENSE` | Apache 0.2.0 License. Chosen for: compatibility with PyTorch (Apache 0.2.0) and HuggingFace Transformers (Apache 0.2.0). Allows commercial use, modification, distribution, and patent use. |
| `.ruff.toml` | Ruff linter config. Rules: E, F, I (isort), UP (pyupgrade), B (flake8-bugbear). Line length: 100. Target Python: 3.9. Per-file ignores for test files. |
| `mypy.ini` | MyPy type checker config. Strict mode enabled. Ignores: missing stubs for torch (handled by torch-stubs). Per-module ignores for test files. Runs in CI on every push. |
| `.gitignore` | Standard Python gitignore. Excludes: __pycache__, *.egg-info, dist/, build/, .pytest_cache, .mypy_cache, .ruff_cache, *.json result files from benchmarks (tracked separately), model weights. |

---

> *(v0.2.0)* marks files and signals introduced in MoEWatch v0.2.0.
