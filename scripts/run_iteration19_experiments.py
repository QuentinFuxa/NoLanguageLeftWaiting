#!/usr/bin/env python3
"""Iteration 19 GPU experiments -- Competition readiness validation.

Focus: Verify LongYAAL metrics, validate OmniSTEval output, test wb=1 with top_p.

Usage:
    # On A40:
    python scripts/run_iteration19_experiments.py --model /home/fuxa/HY-MT1.5-7B.Q8_0.gguf

    # Specific phase:
    python scripts/run_iteration19_experiments.py --model /path/to/model.gguf --phase 1
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RESULTS_DIR = "results/iteration19"


def run_bench(args_list, label=""):
    """Run a bench command and return the result."""
    cmd = [sys.executable, "-m", "nllw.bench"] + args_list
    print(f"\n{'='*60}")
    print(f"[{label}] Running: {' '.join(cmd)}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - t0
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    print(f"[{label}] Completed in {elapsed:.1f}s (exit code {result.returncode})")
    return result


def phase1_longyaal_validation(model_path, n_sent=20):
    """Validate LongYAAL metrics are computed and displayed correctly."""
    print("\n\n" + "="*60)
    print("PHASE 1: LongYAAL Metric Validation")
    print("="*60)

    for direction in ["en-zh", "en-de"]:
        run_bench([
            "--lang", direction,
            "--model", model_path,
            "--n", str(n_sent),
            "--comet",
        ], label=f"LongYAAL-{direction}")


def phase2_omnisteval_output(model_path, n_sent=20):
    """Generate OmniSTEval output and verify format."""
    print("\n\n" + "="*60)
    print("PHASE 2: OmniSTEval Output Validation")
    print("="*60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    for direction in ["en-zh", "en-de", "en-it", "cs-en"]:
        output_path = f"{RESULTS_DIR}/omnisteval_{direction}.jsonl"
        run_bench([
            "--lang", direction,
            "--model", model_path,
            "--n", str(n_sent),
            "--omnisteval", output_path,
        ], label=f"OmniSTEval-{direction}")

        # Validate output
        if os.path.exists(output_path):
            with open(output_path) as f:
                lines = [json.loads(l) for l in f if l.strip()]
            print(f"  [{direction}] {len(lines)} entries")
            for i, entry in enumerate(lines[:2]):
                n_words = len(entry["prediction"].split())
                n_delays = len(entry["delays"])
                ok = "OK" if n_words == n_delays else "MISMATCH"
                print(f"    Entry {i}: {n_words} words, {n_delays} delays [{ok}]")
                print(f"    Prediction: {entry['prediction'][:80]}...")


def phase3_wb1_top_p_experiment(model_path, n_sent=50):
    """Test wb=1 with top_p for latency reduction potential."""
    print("\n\n" + "="*60)
    print("PHASE 3: wb=1 with top_p (latency experiment)")
    print("="*60)

    # Compare wb=1,2,3,4 with top_p on EN-ZH
    run_bench([
        "--sweep", "wb=1,2,3,4",
        "--aggregation", "top_p",
        "--border-distance", "3",
        "--lang", "en-zh",
        "--model", model_path,
        "--n", str(n_sent),
        "--comet",
    ], label="wb-sweep-enzh")


def phase4_adaptive_top_p_confirmation(model_path, n_sent=100):
    """Confirm adaptive top_p results at 100 sentences."""
    print("\n\n" + "="*60)
    print("PHASE 4: Adaptive top_p 100-sentence confirmation")
    print("="*60)

    for direction, p_threshold in [("en-zh", 0.85), ("en-de", 0.75), ("en-it", 0.9), ("cs-en", 0.9)]:
        # Fixed optimal
        run_bench([
            "--lang", direction,
            "--model", model_path,
            "--n", str(n_sent),
            "--aggregation", "top_p",
            "--top-p-threshold", str(p_threshold),
            "--comet",
        ], label=f"fixed-{direction}")

        # Adaptive
        run_bench([
            "--lang", direction,
            "--model", model_path,
            "--n", str(n_sent),
            "--aggregation", "top_p",
            "--top-p-threshold", str(p_threshold),
            "--adaptive-top-p",
            "--comet",
        ], label=f"adaptive-{direction}")


def phase5_direction_switching(model_path, n_sent=10):
    """Test direction switching via SimulStream wrapper."""
    print("\n\n" + "="*60)
    print("PHASE 5: SimulStream Direction Switching")
    print("="*60)

    # Quick test of each direction
    for direction in ["en-zh", "en-de", "en-it", "cs-en"]:
        run_bench([
            "--lang", direction,
            "--model", model_path,
            "--n", str(n_sent),
        ], label=f"quick-{direction}")


def main():
    parser = argparse.ArgumentParser(description="Iteration 19 GPU experiments")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--phase", type=int, default=0, help="Run specific phase (0=all)")
    parser.add_argument("--n", type=int, default=0, help="Override sentence count")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    t0 = time.time()

    if args.phase == 0 or args.phase == 1:
        phase1_longyaal_validation(args.model, n_sent=args.n or 20)
    if args.phase == 0 or args.phase == 2:
        phase2_omnisteval_output(args.model, n_sent=args.n or 20)
    if args.phase == 0 or args.phase == 3:
        phase3_wb1_top_p_experiment(args.model, n_sent=args.n or 50)
    if args.phase == 0 or args.phase == 4:
        phase4_adaptive_top_p_confirmation(args.model, n_sent=args.n or 100)
    if args.phase == 0 or args.phase == 5:
        phase5_direction_switching(args.model, n_sent=args.n or 10)

    total = time.time() - t0
    print(f"\n\nTotal experiment time: {total:.0f}s ({total/60:.1f}m)")
    print(f"Results saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
