#!/usr/bin/env python3
"""Iteration 23 GPU experiments -- Confidence-adaptive wb + language-pair gen cap + iter 22 validation.

Key changes in iteration 23:
    - Confidence-adaptive word batching (novel): adjust wb from prev generation confidence
    - Language-pair-aware gen cap: tighter generation limits for compact targets
    - avg_logprob now tracked in AlignAtt-LA backend too
    - Competition is in ~7 days (April 1-15, 2026)

IMPORTANT: This script also runs the iter 22 experiments that still need validation:
    - YAAL fix (batch_first_emission_time)
    - Source-aware batching
    - Perplexity adaptive border
    - Combined features sweep

Phases:
    1. Baseline validation: all 4 directions with optimal configs (20 sent)
    2. YAAL fix validation: compare YAAL vs reference (100 sent)
    3. Confidence-adaptive wb: test wb adjustment from logprob (100 sent)
    4. Language-pair gen cap: test tighter generation limits (100 sent)
    5. Source-aware batching: function-word-aware batching (100 sent)
    6. Perplexity adaptive bd: test bd from generation confidence (100 sent)
    7. Combined features: all winning features together (100 sent)
    8. Competition OmniSTEval: generate JSONL for all directions (100 sent)

Usage:
    # Full run on A40:
    python scripts/run_iteration23_experiments.py --model /home/fuxa/HY-MT1.5-7B.Q8_0.gguf

    # Specific phase:
    python scripts/run_iteration23_experiments.py --model /path/to/model.gguf --phase 3

    # Quick mode (20 sentences instead of 100):
    python scripts/run_iteration23_experiments.py --model /path/to/model.gguf --quick
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RESULTS_DIR = "results/iteration23"
ALL_DIRECTIONS = ["en-zh", "en-de", "en-it", "cs-en"]

# Best known configs per direction (iteration 18 verified)
OPTIMAL_CONFIGS = {
    "en-zh": {"bd": 3, "wb": 4, "p": 0.85},
    "en-de": {"bd": 2, "wb": 3, "p": 0.75},
    "en-it": {"bd": 2, "wb": 3, "p": 0.9},
    "cs-en": {"bd": 3, "wb": 3, "p": 0.9},
}

# Reference results from iteration 18 (to compare against)
REFERENCE_COMET = {
    "en-zh": 0.894,
    "en-de": 0.881,
    "en-it": 0.891,
    "cs-en": 0.879,
}
REFERENCE_YAAL = {
    "en-zh": 6.09,
    "en-de": 5.45,
    "en-it": 6.76,
    "cs-en": 5.81,
}


def run_bench(args_list, label="", timeout=3600):
    """Run a bench command and return the result."""
    cmd = [sys.executable, "-m", "nllw.bench"] + args_list
    print(f"\n{'='*70}")
    print(f"[{label}] Running: {' '.join(cmd)}")
    print(f"{'='*70}")
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[{label}] TIMEOUT after {timeout}s")
        return None
    elapsed = time.time() - t0

    # Print output
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(f"[{label}] FAILED (rc={result.returncode})")
        if result.stderr:
            print(f"STDERR: {result.stderr[-500:]}")
        return None

    print(f"[{label}] Completed in {elapsed:.1f}s")
    return result.stdout


def bench_direction(model, direction, n_sent, extra_args=None, label=""):
    """Run bench for a direction with optimal config."""
    cfg = OPTIMAL_CONFIGS[direction]
    args = [
        "--model", model,
        "--lang", direction,
        "--border-distance", str(cfg["bd"]),
        "--word-batch", str(cfg["wb"]),
        "--top-p-threshold", str(cfg["p"]),
        "--aggregation", "top_p",
        "--n-sentences", str(n_sent),
        "--n-gpu-layers", "99",
        "--comet",
        "--save",
    ]
    if extra_args:
        args.extend(extra_args)
    return run_bench(args, label=f"{label} {direction}")


def phase1_baseline(model, n_sent=20):
    """Phase 1: Baseline validation -- all directions with optimal configs."""
    print("\n" + "="*70)
    print("PHASE 1: Baseline validation (smoke test)")
    print("="*70)
    for direction in ALL_DIRECTIONS:
        bench_direction(model, direction, n_sent, label="P1-baseline")


def phase2_yaal_validation(model, n_sent=100):
    """Phase 2: YAAL fix validation -- compare new YAAL vs reference."""
    print("\n" + "="*70)
    print("PHASE 2: YAAL fix validation (batch_first_emission_time)")
    print("Expected: YAAL should be lower than reference by ~(wb-1) words")
    print("="*70)
    for direction in ALL_DIRECTIONS:
        cfg = OPTIMAL_CONFIGS[direction]
        ref_yaal = REFERENCE_YAAL[direction]
        expected_improvement = cfg["wb"] - 1
        print(f"\n{direction}: reference YAAL={ref_yaal}, "
              f"expected improvement ~{expected_improvement} words")
        bench_direction(model, direction, n_sent, label="P2-yaal")


def phase3_confidence_adaptive_wb(model, n_sent=100):
    """Phase 3: Confidence-adaptive word batching (NEW in iter 23)."""
    print("\n" + "="*70)
    print("PHASE 3: Confidence-adaptive word batching")
    print("Novel: adjust word_batch from previous generation confidence")
    print("Expected: YAAL reduction (confident -> wb-1), minimal quality impact")
    print("="*70)

    for direction in ALL_DIRECTIONS:
        # Baseline (no confidence adaptive)
        bench_direction(model, direction, n_sent, label="P3-baseline")

        # With confidence adaptive wb (default thresholds)
        bench_direction(model, direction, n_sent,
                        extra_args=["--confidence-adaptive-wb"],
                        label="P3-confwb")

    # Threshold sweep on EN-ZH
    print("\n--- Confidence threshold sweep (EN-ZH) ---")
    for hi in ["-0.3", "-0.5", "-0.7", "-1.0"]:
        for lo in ["-1.5", "-2.0", "-3.0"]:
            bench_direction(model, "en-zh", n_sent,
                            extra_args=[
                                "--confidence-adaptive-wb",
                                "--confidence-wb-high", hi,
                                "--confidence-wb-low", lo,
                            ],
                            label=f"P3-sweep-hi{hi}-lo{lo}")


def phase4_language_pair_gen_cap(model, n_sent=100):
    """Phase 4: Language-pair-aware gen cap (NEW in iter 23)."""
    print("\n" + "="*70)
    print("PHASE 4: Language-pair-aware generation cap")
    print("Tighter gen limits for compact targets (ZH, JA), looser for verbose (DE)")
    print("="*70)

    for direction in ALL_DIRECTIONS:
        # Baseline
        bench_direction(model, direction, n_sent, label="P4-baseline")
        # With language pair gen cap
        bench_direction(model, direction, n_sent,
                        extra_args=["--language-pair-gen-cap"],
                        label="P4-lpgcap")


def phase5_source_aware_batching(model, n_sent=100):
    """Phase 5: Source-aware batching (from iter 21, needs validation)."""
    print("\n" + "="*70)
    print("PHASE 5: Source-aware word batching")
    print("Defer translation when batch ends on function word")
    print("="*70)

    for direction in ALL_DIRECTIONS:
        bench_direction(model, direction, n_sent, label="P5-baseline")
        bench_direction(model, direction, n_sent,
                        extra_args=["--source-aware-batch"],
                        label="P5-srcaware")


def phase6_perplexity_adaptive_bd(model, n_sent=100):
    """Phase 6: Perplexity-adaptive border (from iter 20, needs validation)."""
    print("\n" + "="*70)
    print("PHASE 6: Perplexity-adaptive border distance")
    print("Hibiki-inspired: adjust bd from generation confidence")
    print("="*70)

    for direction in ALL_DIRECTIONS:
        bench_direction(model, direction, n_sent, label="P6-baseline")
        bench_direction(model, direction, n_sent,
                        extra_args=["--perplexity-adaptive-bd"],
                        label="P6-pplbd")

    # PPL threshold sweep on EN-ZH
    print("\n--- PPL threshold sweep (EN-ZH) ---")
    for lo in ["1.5", "2.0", "3.0"]:
        for hi in ["4.0", "5.0", "6.0"]:
            bench_direction(model, "en-zh", n_sent,
                            extra_args=[
                                "--perplexity-adaptive-bd",
                                "--perplexity-bd-low", lo,
                                "--perplexity-bd-high", hi,
                            ],
                            label=f"P6-sweep-lo{lo}-hi{hi}")


def phase7_combined(model, n_sent=100):
    """Phase 7: Combined features -- all winning features together."""
    print("\n" + "="*70)
    print("PHASE 7: Combined features sweep")
    print("Test all iter 22-23 features together for competition config")
    print("="*70)

    # Combinations to test:
    # 1. confwb only
    # 2. confwb + source-aware
    # 3. confwb + pplbd
    # 4. confwb + source-aware + pplbd
    # 5. confwb + source-aware + pplbd + lpgcap
    # 6. adaptive_top_p + confwb

    combos = [
        ("confwb", ["--confidence-adaptive-wb"]),
        ("confwb+src", ["--confidence-adaptive-wb", "--source-aware-batch"]),
        ("confwb+ppl", ["--confidence-adaptive-wb", "--perplexity-adaptive-bd"]),
        ("confwb+src+ppl", [
            "--confidence-adaptive-wb", "--source-aware-batch",
            "--perplexity-adaptive-bd"
        ]),
        ("all-new", [
            "--confidence-adaptive-wb", "--source-aware-batch",
            "--perplexity-adaptive-bd", "--language-pair-gen-cap"
        ]),
        ("adaptp+confwb", [
            "--adaptive-top-p", "--confidence-adaptive-wb"
        ]),
    ]

    for direction in ALL_DIRECTIONS:
        # Baseline
        bench_direction(model, direction, n_sent, label="P7-baseline")
        for name, extra in combos:
            bench_direction(model, direction, n_sent,
                            extra_args=extra,
                            label=f"P7-{name}")


def phase8_competition_output(model, n_sent=100):
    """Phase 8: Generate competition OmniSTEval output."""
    print("\n" + "="*70)
    print("PHASE 8: Competition OmniSTEval output")
    print("Generate JSONL for all 4 directions with best config")
    print("="*70)

    os.makedirs(f"{RESULTS_DIR}/omnisteval", exist_ok=True)

    for direction in ALL_DIRECTIONS:
        output_file = f"{RESULTS_DIR}/omnisteval/{direction}.jsonl"
        bench_direction(model, direction, n_sent,
                        extra_args=[
                            "--omnisteval", output_file,
                        ],
                        label=f"P8-omnisteval-{direction}")
        if os.path.exists(output_file):
            # Validate the output
            with open(output_file) as f:
                n_lines = sum(1 for _ in f)
            print(f"  Output: {output_file} ({n_lines} entries)")


def main():
    parser = argparse.ArgumentParser(
        description="Iteration 23 GPU experiments"
    )
    parser.add_argument("--model", required=True,
                        help="Path to GGUF model file")
    parser.add_argument("--phase", type=int, default=None,
                        help="Run specific phase (1-8, default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 20 sentences instead of 100")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    n_sent = 20 if args.quick else 100

    phases = {
        1: ("Baseline validation", lambda: phase1_baseline(args.model, min(n_sent, 20))),
        2: ("YAAL fix validation", lambda: phase2_yaal_validation(args.model, n_sent)),
        3: ("Confidence-adaptive wb", lambda: phase3_confidence_adaptive_wb(args.model, n_sent)),
        4: ("Language-pair gen cap", lambda: phase4_language_pair_gen_cap(args.model, n_sent)),
        5: ("Source-aware batching", lambda: phase5_source_aware_batching(args.model, n_sent)),
        6: ("Perplexity adaptive bd", lambda: phase6_perplexity_adaptive_bd(args.model, n_sent)),
        7: ("Combined features", lambda: phase7_combined(args.model, n_sent)),
        8: ("Competition output", lambda: phase8_competition_output(args.model, n_sent)),
    }

    if args.phase:
        if args.phase not in phases:
            print(f"Unknown phase {args.phase}. Available: {sorted(phases.keys())}")
            sys.exit(1)
        name, fn = phases[args.phase]
        print(f"\nRunning Phase {args.phase}: {name}")
        fn()
    else:
        print(f"\nRunning ALL {len(phases)} phases ({n_sent} sentences each)")
        print("Estimated time: ~2-4 hours on A40")
        for num, (name, fn) in sorted(phases.items()):
            print(f"\n{'#'*70}")
            print(f"# Phase {num}: {name}")
            print(f"{'#'*70}")
            fn()

    print(f"\n{'='*70}")
    print("ALL PHASES COMPLETE")
    print(f"Results saved to: {RESULTS_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
