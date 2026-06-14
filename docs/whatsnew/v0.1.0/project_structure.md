# MOEWATCH REPOSITORY STRUCTURE -v0.1.0

---

## Root Directory `/`

**Purpose:** Root of the repository. Contains all config files, CI workflows, and the installable package.

---

### `README.md` `[doc]`
**Path:** `README.md`
**Purpose:** The public face of the library. Covers: one-line install, quick-start snippet, feature overview, honest scope statement, and explicit acknowledgment of model-clinic. Includes a live Colab badge.

---

### `pyproject.toml` `[core]`
**Path:** `pyproject.toml`
**Purpose:** Single source of truth for packaging. Declares name, version, Python ≥3.8 constraint, dependencies (torch, transformers, numpy), optional extras (wandb, tensorboard), and entry points.
**Exports:** `[project]`, `[build-system]`, `[tool.pytest]`, `[tool.ruff]`

---

### `LICENSE` `[doc]`
**Path:** `LICENSE`
**Purpose:** Apache 2.0 — permissive license appropriate for a research-tooling library that needs broad adoption.

---

### `.gitignore` `[doc]`
**Path:** `.gitignore`
**Purpose:** Ignores `__pycache__`, `.env`, `dist/`, `build/`, `*.egg-info`, `.pytest_cache`, and local notebook checkpoints.

---

### `CHANGELOG.md` `[doc]`
**Path:** `CHANGELOG.md`
**Purpose:** Version history. Starts at v0.1.0 with the four core diagnostics. Keeps an "Unreleased" section at the top for in-progress changes.

---

### `CONTRIBUTING.md` `[doc]`
**Path:** `CONTRIBUTING.md`
**Purpose:** Onboarding guide for community contributors. Covers: dev install, running tests, adding a new router pattern, the multi-GPU contribution surface. Key for v0.3 distributed work.

---

## Directory: `.github/`

**Path:** `.github/`
**Purpose:** All GitHub-specific automation lives here — CI workflows and issue/PR templates.

### Directory: `.github/workflows/`

**Path:** `.github/workflows/`
**Purpose:** GitHub Actions workflow YAML files.

#### `ci.yml` `[ci]`
**Path:** `.github/workflows/ci.yml`
**Purpose:** Main CI pipeline. Runs on every push and PR. Steps: install deps → lint (ruff) → type-check (mypy) → pytest on CPU → upload coverage to Codecov. Matrix: Python 3.8, 3.10, 3.12.
**Exports:** `on: push / pull_request`, `jobs: lint, typecheck, test`

#### `publish.yml` `[ci]`
**Path:** `.github/workflows/publish.yml`
**Purpose:** PyPI release workflow. Triggered on version tag (v*.*.*). Builds the wheel, runs tests, then publishes via OIDC trusted publisher — no stored API key.
**Exports:** `on: push tags v*.*.*`

---

### Directory: `.github/ISSUE_TEMPLATE/`

**Path:** `.github/ISSUE_TEMPLATE/`
**Purpose:** Structured issue templates shown in the GitHub "New Issue" picker.

#### `bug_report.md` `[doc]`
**Path:** `.github/ISSUE_TEMPLATE/bug_report.md`
**Purpose:** Bug report template. Required fields: moewatch version, model name, Python version, GPU info, minimal repro snippet, actual vs expected output.

#### `feature_request.md` `[doc]`
**Path:** `.github/ISSUE_TEMPLATE/feature_request.md`
**Purpose:** Feature request template. Asks for: use case description, which model/architecture it targets, whether the contributor can implement it.

---

## Directory: `moewatch/` (Installable Package)

**Path:** `moewatch/`
**Purpose:** The installable Python package — everything under `pip install moewatch`.

### `__init__.py` `[core]`
**Path:** `moewatch/__init__.py`
**Purpose:** Public API surface. Re-exports the symbols researchers actually type. Keeps the import tree shallow — users never need to know about internal modules.
**Exports:** `audit`, `MoEWatch`, `WatchConfig`, `__version__`

### `_audit.py` `[core]`
**Path:** `moewatch/_audit.py`
**Purpose:** Implements `audit(model, dataloader, steps)`. Orchestrates: attach hooks → run N forward passes → collect stats → return AuditReport. The main entry point for offline diagnostics.
**Exports:** `audit()`

### `_watcher.py` `[core]`
**Path:** `moewatch/_watcher.py`
**Purpose:** Implements MoEWatch — the training-time monitor. Exposes `attach(trainer)` using HuggingFace TrainerCallback, and a standalone `step()` method for custom loops. Owns the alert emit logic.
**Exports:** `MoEWatch`, `MoEWatchCallback`

### `config.py` `[core]`
**Path:** `moewatch/config.py`
**Purpose:** WatchConfig dataclass. All thresholds live here: `dead_threshold`, `entropy_warn`, `entropy_critical`, `window_steps`, `sample_every`, `log_every`, output mode. Validates on construction.
**Exports:** `WatchConfig`

---

### Directory: `moewatch/hooks/`

**Path:** `moewatch/hooks/`
**Purpose:** PyTorch forward-hook machinery. Zero model modifications — reads only.

#### `__init__.py`
**Path:** `moewatch/hooks/__init__.py`
**Purpose:** Re-exports HookManager and RouterHook for the hooks sub-package.
**Exports:** `HookManager`, `RouterHook`

#### `manager.py` `[core]`
**Path:** `moewatch/hooks/manager.py`
**Purpose:** HookManager owns the hook lifecycle: `attach()`, `detach()`, and a context-manager interface. Tracks all registered hooks to guarantee clean teardown — no leaking hooks after a crash.
**Exports:** `HookManager`

#### `router_hook.py` `[core]`
**Path:** `moewatch/hooks/router_hook.py`
**Purpose:** RouterHook is the actual PyTorch forward hook. Fires on each router module forward pass, extracts routing logits, and writes a RoutingEvent to the StatCollector ring buffer.
**Exports:** `RouterHook`, `RoutingEvent`

#### `detection.py` `[core]`
**Path:** `moewatch/hooks/detection.py`
**Purpose:** Auto-detection of router modules. Two-pass strategy: (1) named pattern scan for known architectures; (2) heuristic fallback on "router"/"gate"/"sparse" in class name. Manual override via config.
**Exports:** `detect_router_modules()`

---

### Directory: `moewatch/collector/`

**Path:** `moewatch/collector/`
**Purpose:** Asynchronous stats aggregation. Decoupled from the hot forward-pass path.

#### `__init__.py`
**Path:** `moewatch/collector/__init__.py`
**Purpose:** Re-exports StatCollector and RingBuffer.
**Exports:** `StatCollector`, `RingBuffer`

#### `ring_buffer.py` `[core]`
**Path:** `moewatch/collector/ring_buffer.py`
**Purpose:** Fixed-memory circular buffer for per-step routing events. Prevents unbounded memory growth during long training runs. Configurable capacity.
**Exports:** `RingBuffer`

#### `stat_collector.py` `[core]`
**Path:** `moewatch/collector/stat_collector.py`
**Purpose:** Aggregates RoutingEvents into per-layer statistics: token counts per expert, step timestamps, raw logit tensors. CPU-side to avoid GPU stalls.
**Exports:** `StatCollector`, `LayerStats`

---

### Directory: `moewatch/analyzer/`

**Path:** `moewatch/analyzer/`
**Purpose:** Metric computation layer. Pure functions — no side effects, fully testable.

#### `__init__.py`
**Path:** `moewatch/analyzer/__init__.py`
**Purpose:** Re-exports EntropyAnalyzer and CollapseDetector.
**Exports:** `EntropyAnalyzer`, `CollapseDetector`

#### `entropy.py` `[core]`
**Path:** `moewatch/analyzer/entropy.py`
**Purpose:** Computes per-layer Shannon entropy H = -Σ p_i log(p_i). Tracks trend over window_steps, flags >20% drops as WARN, normalises against theoretical maximum.
**Exports:** `EntropyAnalyzer`, `compute_entropy()`

#### `collapse.py` `[core]`
**Path:** `moewatch/analyzer/collapse.py`
**Purpose:** CollapseDetector identifies dead and cold experts per layer. Distinguishes cold (recoverable, <500 steps unseen) from dead (confirmed, flagged ERROR). Computes load imbalance score.
**Exports:** `CollapseDetector`, `ExpertStatus`

---

### Directory: `moewatch/report/`

**Path:** `moewatch/report/`
**Purpose:** Output formatting. Separates concerns between data (AuditReport) and presentation (CLIReporter).

#### `__init__.py`
**Path:** `moewatch/report/__init__.py`
**Purpose:** Re-exports AuditReport and CLIReporter.
**Exports:** `AuditReport`, `CLIReporter`

#### `audit_report.py` `[core]`
**Path:** `moewatch/report/audit_report.py`
**Purpose:** Structured result object returned by `audit()`. Holds per-layer metrics, expert statuses, entropy scores. Methods: `summary()`, `dead_experts()`, `routing_entropy()`, `utilization()`, `to_json()`.
**Exports:** `AuditReport`

#### `cli_reporter.py` `[core]`
**Path:** `moewatch/report/cli_reporter.py`
**Purpose:** Formats AuditReport as coloured ASCII terminal output. Uses ANSI codes with NO_COLOR fallback. Renders utilization histograms, entropy bars, and alert banners.
**Exports:** `CLIReporter`

---

## Directory: `tests/`

**Path:** `tests/`
**Purpose:** Full test suite. Runs on CPU with a synthetic MoE — no GPU or real model weights required in CI.

### `conftest.py` `[test]`
**Path:** `tests/conftest.py`
**Purpose:** Pytest fixtures: a minimal SyntheticMoE model, tiny DataLoader of random tensors, and a default WatchConfig. Makes every test self-contained and GPU-free.
**Exports:** `synthetic_model`, `tiny_dataloader`, `default_config`

### `test_detection.py` `[test]`
**Path:** `tests/test_detection.py`
**Purpose:** Tests router module auto-detection against a synthetic MoE, a mock MixtralSparseMoeBlock, and a custom-named router. Verifies graceful no-op when detection fails.

### `test_hooks.py` `[test]`
**Path:** `tests/test_hooks.py`
**Purpose:** Tests HookManager lifecycle: attach, detach, context-manager teardown, and no leaked hooks after exception. Verifies RouterHook fires and writes to collector.

### `test_collector.py` `[test]`
**Path:** `tests/test_collector.py`
**Purpose:** Tests RingBuffer capacity wrapping and StatCollector aggregation. Verifies expert token counts sum to total tokens processed.

### `test_entropy.py` `[test]`
**Path:** `tests/test_entropy.py`
**Purpose:** Tests EntropyAnalyzer with known distributions: uniform (max entropy), one-hot (zero entropy), collapsed distribution. Verifies thresholds trigger at correct values.

### `test_collapse.py` `[test]`
**Path:** `tests/test_collapse.py`
**Purpose:** Injects synthetic expert utilization data with a known dead expert (0%) and cold expert (0.05%). Asserts correct ExpertStatus output.

### `test_audit.py` `[test]`
**Path:** `tests/test_audit.py`
**Purpose:** Integration test: runs `audit(synthetic_model, tiny_dataloader)` end-to-end on CPU. Asserts AuditReport is non-empty, `dead_experts()` returns a list, and `to_json()` is valid JSON.

### `test_cli_reporter.py` `[test]`
**Path:** `tests/test_cli_reporter.py`
**Purpose:** This test module ensures that CLIReporter produces accurate, readable, and reliable command-line audit reports by validating formatting helpers, report sections, expert diagnostics, recommendations, and exception-handling behavior.

### `test_watcher.py` `[test]`
**Path:** `tests/test_watcher.py`
**Purpose:** Tests MoEWatch attach/detach and TrainerCallback step hook. Verifies alerts fire at correct thresholds using pre-baked collapsed routing stats.

---

## Directory: `examples/`

**Path:** `examples/`
**Purpose:** Runnable end-to-end example scripts demonstrating moewatch integration with real-world MoE architectures. Each script is self-contained and annotated for readability.

### `transformers_example.py` `[doc]`
**Path:** `examples/transformers_example.py`
**Purpose:** Generic HuggingFace Transformers integration example. Shows how to load any MoE model via AutoModelForCausalLM, attach MoEWatch using WatchConfig defaults, run a short inference loop, and print the AuditReport summary. Serves as the canonical starting point for new users.

### `mixtral_example.py` `[doc]`
**Path:** `examples/mixtral_example.py`
**Purpose:** Mixtral 8x7B-specific walkthrough. Demonstrates auto-detection of MixtralSparseMoeBlock router layers, WatchConfig tuning for Mixtral's 8-expert topology (dead_threshold, entropy_warn), and how to read per-layer utilization from the returned AuditReport.

### `qwen3_moe_example.py` `[doc]`
**Path:** `examples/qwen3_moe_example.py`
**Purpose:** Qwen3-MoE integration example. Covers model loading with trust_remote_code, moewatch router detection for Qwen3's sparse attention-MoE hybrid, and interpreting entropy scores across the model's 128 experts per layer.

### `deepseek_example.py` `[doc]`
**Path:** `examples/deepseek_example.py`
**Purpose:** DeepSeek-MoE integration example. Demonstrates fine-grained expert monitoring for DeepSeek's shared+routed expert architecture, manual router module override via WatchConfig for architectures that use non-standard naming, and collapse detection in a model with many fine-grained experts.

---

## Directory: `benchmarks/`

**Path:** `benchmarks/`
**Purpose:** Performance benchmarking scripts that measure moewatch overhead — latency, throughput, and memory — relative to an unmonitored baseline. Results are used to validate the "< 3% overhead" claim in the README.

### `benchmark_mixtral.py` `[test]`
**Path:** `benchmarks/benchmark_mixtral.py`
**Purpose:** Benchmarks moewatch hook overhead on Mixtral 8x7B. Runs 100 warmup + 500 timed forward passes with and without hooks attached. Reports mean latency, p95 latency, tokens/sec, and peak GPU memory delta. Outputs a JSON results file for CI regression tracking.

### `benchmark_qwen_moe.py` `[test]`
**Path:** `benchmarks/benchmark_qwen_moe.py`
**Purpose:** Benchmarks moewatch overhead on Qwen3-MoE. Mirrors the Mixtral benchmark structure with Qwen-specific batch sizes and sequence lengths. Includes a multi-GPU (DataParallel) variant to validate hook correctness under device splits.

---

## Directory: `notebooks/`

**Path:** `notebooks/`
**Purpose:** Public Colab demo notebooks — both are v0.1 ship criteria.

### `quickstart_colab.ipynb` `[doc]`
**Path:** `notebooks/quickstart_colab.ipynb`
**Purpose:** Primary demo notebook. Runs on Colab free T4. Loads OLMoE-1B-7B, attaches moewatch, runs `audit()`, and renders the full report. Includes a synthetic collapse demo that does not require model download.

### `synthetic_moe_demo.ipynb` `[doc]`
**Path:** `notebooks/synthetic_moe_demo.ipynb`
**Purpose:** CPU-only demo. Creates a toy MoE, manually induces expert collapse, and shows moewatch catching it. Useful for contributors without GPU access.

---

## Directory: `docs/`

**Path:** `docs/`
**Purpose:** Documentation source. Minimal in v0.1 — README is the primary doc. Expands in v0.2 with API reference.

### `api_reference.md` `[doc]`
**Path:** `docs/api_reference.md`
**Purpose:** API reference for `audit()`, `MoEWatch`, `WatchConfig`, and `AuditReport`. Links to README and notebooks.

### `competitive_landscape.md` `[doc]`
**Path:** `docs/competitive_landscape.md`
**Purpose:** Honest write-up of the tooling gap: model-clinic, chuk-lazarus, MLXLMProbe, LibMoE, MixtureKit. Used for README positioning.
