# =============================================================================
# MoEWatch v0.2.0 — Research Paper Test Suite
# run_all_tests.py — Combined Test Runner
# =============================================================================
#
# Executes all paper test modules and prints a final summary table.
# Each test module must expose a run() function.
#
# Usage:
#   python D:\moewatch-dev\paper\run_all_tests.py
#
# =============================================================================

import importlib.util
import os
import sys
import time
import traceback

# ---------------------------------------------------------------------------
# Test registry — (id, filename, description)
# ---------------------------------------------------------------------------

PAPER_DIR = os.path.dirname(os.path.abspath(__file__))

TESTS = [
    ("01", "test_01_hook_detection.py",       "Hook Attachment & Router Detection"),
    ("02", "test_02_routing_stats.py",         "Routing Statistics Collection"),
    ("03", "test_03_entropy_analysis.py",      "Entropy Analysis (Tier 2 Signal)"),
    ("04", "test_04_collapse_detection.py",    "Collapse Detection (Expert Health)"),
    ("05", "test_05_gradient_starvation.py",   "Gradient Starvation (Tier 1 — live hooks)"),
    ("06", "test_06_risk_score.py",            "Risk Score (live watcher)"),
    ("07", "test_07_entropy.py",               "Entropy (live watcher)"),
    ("08", "test_08_interventions.py",         "Intervention Actions & Policy"),
    ("09", "test_09_cusum.py",                 "CUSUM Drift Detection"),
    ("10", "test_10_risk_fusion.py",           "Risk Score Fusion (T1 + T2 + T3)"),
    ("11", "test_11_gradient_starvation.py",   "Gradient Starvation (Tier 1 — low-level)"),
    ("12", "test_12_audit.py",                 "Offline Audit (End-to-End)"),
    ("13", "test_13_serialization.py",         "Serialization Round-Trip"),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def load_and_run(filename: str) -> tuple:
    """Import a test module and call its run() function.

    Returns (passed: bool, elapsed: float, error: str or None).
    """
    path = os.path.join(PAPER_DIR, filename)
    if not os.path.exists(path):
        return False, 0.0, f"File not found: {path}"

    spec   = importlib.util.spec_from_file_location("_test_module", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        return False, 0.0, f"Import error: {e}"

    if not hasattr(module, "run"):
        return False, 0.0, "No run() function found"

    t0 = time.perf_counter()
    try:
        module.run()
        elapsed = time.perf_counter() - t0
        return True, elapsed, None
    except Exception:
        elapsed = time.perf_counter() - t0
        return False, elapsed, traceback.format_exc()


def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  MoEWatch v0.2.0 — Research Paper Test Suite                    ║")
    print("║  Running all tests...                                            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    results = []
    total_start = time.perf_counter()

    for test_id, filename, description in TESTS:
        tag    = f"test_{test_id}"
        exists = os.path.exists(os.path.join(PAPER_DIR, filename))

        if not exists:
            print(f"  [{test_id}] SKIP  {description}  (file not found)")
            results.append((test_id, description, "SKIP", 0.0, None))
            continue

        print(f"  [{test_id}] Running: {description} ...")
        sys.stdout.flush()

        # Suppress per-test output by redirecting stdout temporarily
        import io
        old_stdout = sys.stdout
        captured   = io.StringIO()
        sys.stdout = captured

        passed, elapsed, error = load_and_run(filename)

        sys.stdout = old_stdout

        status = "PASS" if passed else "FAIL"
        results.append((test_id, description, status, elapsed, error))

        marker = "✓" if passed else "✗"
        print(f"         {marker} {status}  ({elapsed:.2f}s)")
        if error:
            # Print first 5 lines of traceback
            err_lines = error.strip().split("\n")
            for line in err_lines[-5:]:
                print(f"           {line}")

    total_elapsed = time.perf_counter() - total_start

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    passed_tests = [r for r in results if r[2] == "PASS"]
    failed_tests = [r for r in results if r[2] == "FAIL"]
    skipped      = [r for r in results if r[2] == "SKIP"]

    print()
    print("=" * 70)
    print("  MoEWatch Test Suite — Summary")
    print("=" * 70)
    print(f"  {'ID':<4}  {'Status':<6}  {'Time':>6}  Description")
    print(f"  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*45}")

    for test_id, description, status, elapsed, _ in results:
        marker = "✓" if status == "PASS" else ("—" if status == "SKIP" else "✗")
        time_str = f"{elapsed:.2f}s" if elapsed > 0 else "  —   "
        print(f"  {test_id:<4}  {marker} {status:<5}  {time_str:>6}  {description}")

    print(f"  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*45}")
    print()
    print(f"  Total     : {len(results)}")
    print(f"  Passed    : {len(passed_tests)}  ✓")
    print(f"  Failed    : {len(failed_tests)}  {'✗' if failed_tests else '—'}")
    print(f"  Skipped   : {len(skipped)}")
    print(f"  Wall time : {total_elapsed:.2f}s")
    print()

    if failed_tests:
        print("  Failed tests:")
        for test_id, description, _, _, error in failed_tests:
            print(f"    [{test_id}] {description}")
            if error:
                lines = error.strip().split("\n")
                print(f"         {lines[-1]}")
        print()

    if not failed_tests and not skipped:
        print("  ✓ All tests passed — MoEWatch v0.2.0 core verified.")
    elif not failed_tests:
        print(f"  ✓ All present tests passed ({len(skipped)} skipped).")
    else:
        print(f"  ✗ {len(failed_tests)} test(s) failed.")

    print("=" * 70)
    print()

    # Exit with non-zero code if any test failed (useful for CI)
    sys.exit(1 if failed_tests else 0)


if __name__ == "__main__":
    main()
