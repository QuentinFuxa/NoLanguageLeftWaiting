#!/usr/bin/env python3
"""Iteration 24 GPU experiments -- Entropy-gated top_p + remaining iter 22-23 validation.

Key changes in iteration 24:
    - Entropy-gated top_p (NOVEL): per-token top_p threshold modulation based on
      merged attention distribution entropy during generation. Focused attention
      -> lower threshold -> emit sooner (lower YAAL). Spread attention -> higher
      threshold -> wait longer (better quality).
    - This is PER-TOKEN within the generation loop, not per-sentence.
    - Only active when aggregation=top_p or top_p_weighted.

IMPORTANT: Competition is in ~7 days (April 1-15, 2026).

Phases:
    1. Baseline validation: all 4 directions with optimal configs (20 sent)
    2. Entropy-gated top_p: test per-token threshold modulation (100 sent)
    3. Entropy-gated + adaptive top_p combined: both gating mechanisms (100 sent)
    4. Confidence-adaptive wb (from iter 23): still needs validation (100 sent)
    5. Language-pair gen cap (from iter 23): still needs validation (100 sent)
    6. Source-aware batching (from iter 21): still needs validation (100 sent)
    7. Perplexity adaptive bd (from iter 20): still needs validation (100 sent)
    8. Combined best features: all winning features together (100 sent)
    9. Competition OmniSTEval: generate JSONL for all directions (100 sent)

Usage:
    # Full run on A40:
    python scripts/run_iteration24_experiments.py --model /home/fuxa/HY-MT1.5-7B.Q8_0.gguf

    # Specific phase:
    python scripts/run_iteration24_experiments.py --model /path/to/model.gguf --phase 2

    # Quick mode (20 sentences instead of 100):
    python scripts/run_iteration24_experiments.py --model /path/to/model.gguf --quick
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RESULTS_DIR = "results/iteration24"
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


def bench_direction(model, direction, n_sent, extra_args=None, label="",
                    xcomet=False):
    """Run bench for a direction with optimal config."""
    cfg = OPTIMAL_CONFIGS[direction]
    args = [
        "--model", model,
        "--lang", direction,
        "--border-distance", str(cfg["bd"]),
        "--word-batch", str(cfg["wb"]),
        "--aggregation", "top_p",
        "--top-p-threshold", str(cfg["p"]),
        "--n-sentences", str(n_sent),
        "--n-gpu-layers", "99",
        "--comet",
        "--save",
    ]
    if xcomet:
        args.append("--xcomet")
    if extra_args:
        args.extend(extra_args)
    return run_bench(args, label=label or f"{direction}")


def save_summary(phase, results, path):
    """Save phase summary to results dir."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"# Phase {phase} Results\n\n")
        for k, v in results.items():
            f.write(f"## {k}\n{v}\n\n")
    print(f"Summary saved to {path}")


def phase0_xcomet_baseline(model, n_sent=100):
    """Phase 0: XCOMET-XL baseline -- HIGHEST PRIORITY.

    Competition uses XCOMET-XL (Unbabel/XCOMET-XL) for quality ranking,
    NOT COMET wmt22-comet-da. We need to know our actual competition scores.
    XCOMET-XL amplifies differences ~39x, so rankings may change significantly.

    Also reports StreamLAAL (competition latency metric) and SacreBLEU.
    """
    print("\n" + "="*70)
    print("PHASE 0: XCOMET-XL BASELINE (COMPETITION METRIC)")
    print("  Quality: XCOMET-XL (Unbabel/XCOMET-XL)")
    print("  Latency: StreamLAAL")
    print("  Secondary: SacreBLEU")
    print("  Source: github.com/owaski/iwslt-2026-baselines/eval.sh")
    print("="*70)
    results = {}
    for direction in ALL_DIRECTIONS:
        # Current prompt format (NLLW custom)
        output = bench_direction(model, direction, n_sent,
                                 xcomet=True,
                                 label=f"P0-xcomet-{direction}")
        if output:
            results[f"{direction}_current"] = output

        # Official HY-MT prompt format (matches training data)
        output_official = bench_direction(model, direction, n_sent,
                                          extra_args=["--prompt-format", "hymt-official"],
                                          xcomet=True,
                                          label=f"P0-xcomet-official-{direction}")
        if output_official:
            results[f"{direction}_official_prompt"] = output_official
    save_summary(0, results, f"{RESULTS_DIR}/phase0_xcomet_baseline.md")


def phase1_baseline(model, n_sent=20):
    """Phase 1: Quick baseline validation with optimal configs (COMET only)."""
    print("\n" + "="*70)
    print("PHASE 1: Quick Baseline Validation (COMET, no XCOMET)")
    print("="*70)
    results = {}
    for direction in ALL_DIRECTIONS:
        output = bench_direction(model, direction, n_sent,
                                 label=f"P1-baseline-{direction}")
        if output:
            results[direction] = output
    save_summary(1, results, f"{RESULTS_DIR}/phase1_baseline.md")


def phase2_entropy_gated(model, n_sent=100):
    """Phase 2: Entropy-gated top_p -- per-token threshold modulation."""
    print("\n" + "="*70)
    print("PHASE 2: Entropy-Gated top_p (NOVEL)")
    print("="*70)
    results = {}
    for direction in ALL_DIRECTIONS:
        cfg = OPTIMAL_CONFIGS[direction]
        # Baseline (no gating)
        output_base = bench_direction(model, direction, n_sent,
                                       label=f"P2-base-{direction}")
        if output_base:
            results[f"{direction}_baseline"] = output_base

        # With entropy-gated top_p
        output_gated = bench_direction(model, direction, n_sent,
                                        extra_args=["--entropy-gated-top-p"],
                                        label=f"P2-entgtp-{direction}")
        if output_gated:
            results[f"{direction}_entropy_gated"] = output_gated
    save_summary(2, results, f"{RESULTS_DIR}/phase2_entropy_gated.md")


def phase3_combined_gating(model, n_sent=100):
    """Phase 3: Entropy-gated + adaptive top_p combined."""
    print("\n" + "="*70)
    print("PHASE 3: Entropy-Gated + Adaptive top_p Combined")
    print("="*70)
    results = {}
    for direction in ALL_DIRECTIONS:
        # Both gating mechanisms
        output = bench_direction(model, direction, n_sent,
                                  extra_args=[
                                      "--entropy-gated-top-p",
                                      "--adaptive-top-p",
                                  ],
                                  label=f"P3-combined-{direction}")
        if output:
            results[direction] = output
    save_summary(3, results, f"{RESULTS_DIR}/phase3_combined_gating.md")


def phase4_confidence_wb(model, n_sent=100):
    """Phase 4: Confidence-adaptive word batching (iter 23, needs validation)."""
    print("\n" + "="*70)
    print("PHASE 4: Confidence-Adaptive Word Batching")
    print("="*70)
    results = {}
    for direction in ALL_DIRECTIONS:
        output = bench_direction(model, direction, n_sent,
                                  extra_args=["--confidence-adaptive-wb"],
                                  label=f"P4-confwb-{direction}")
        if output:
            results[direction] = output

    # Threshold sweep on EN-ZH
    direction = "en-zh"
    for hi, lo in [(-0.3, -1.5), (-0.5, -2.0), (-0.7, -3.0), (-1.0, -2.0)]:
        output = bench_direction(model, direction, n_sent,
                                  extra_args=[
                                      "--confidence-adaptive-wb",
                                      "--confidence-wb-high", str(hi),
                                      "--confidence-wb-low", str(lo),
                                  ],
                                  label=f"P4-confwb-{direction}-hi{hi}-lo{lo}")
        if output:
            results[f"{direction}_hi{hi}_lo{lo}"] = output
    save_summary(4, results, f"{RESULTS_DIR}/phase4_confidence_wb.md")


def phase5_langpair_gencap(model, n_sent=100):
    """Phase 5: Language-pair generation cap (iter 23, needs validation)."""
    print("\n" + "="*70)
    print("PHASE 5: Language-Pair Generation Cap")
    print("="*70)
    results = {}
    for direction in ALL_DIRECTIONS:
        output = bench_direction(model, direction, n_sent,
                                  extra_args=["--language-pair-gen-cap"],
                                  label=f"P5-lpgcap-{direction}")
        if output:
            results[direction] = output
    save_summary(5, results, f"{RESULTS_DIR}/phase5_langpair_gencap.md")


def phase6_source_aware(model, n_sent=100):
    """Phase 6: Source-aware batching (iter 21, needs validation)."""
    print("\n" + "="*70)
    print("PHASE 6: Source-Aware Batching")
    print("="*70)
    results = {}
    for direction in ALL_DIRECTIONS:
        output = bench_direction(model, direction, n_sent,
                                  extra_args=["--source-aware-batch"],
                                  label=f"P6-srcaware-{direction}")
        if output:
            results[direction] = output
    save_summary(6, results, f"{RESULTS_DIR}/phase6_source_aware.md")


def phase7_pplbd(model, n_sent=100):
    """Phase 7: Perplexity-adaptive border distance (iter 20, needs validation)."""
    print("\n" + "="*70)
    print("PHASE 7: Perplexity-Adaptive Border Distance")
    print("="*70)
    results = {}
    for direction in ALL_DIRECTIONS:
        output = bench_direction(model, direction, n_sent,
                                  extra_args=["--perplexity-adaptive-bd"],
                                  label=f"P7-pplbd-{direction}")
        if output:
            results[direction] = output
    save_summary(7, results, f"{RESULTS_DIR}/phase7_pplbd.md")


def phase8_combined_best(model, n_sent=100):
    """Phase 8: Combined best features from all iterations."""
    print("\n" + "="*70)
    print("PHASE 8: Combined Best Features")
    print("="*70)
    results = {}

    # Combinations to test:
    combos = [
        ("entgtp", ["--entropy-gated-top-p"]),
        ("entgtp+adaptp", ["--entropy-gated-top-p", "--adaptive-top-p"]),
        ("entgtp+confwb", ["--entropy-gated-top-p", "--confidence-adaptive-wb"]),
        ("entgtp+lpgcap", ["--entropy-gated-top-p", "--language-pair-gen-cap"]),
        ("entgtp+srcaware", ["--entropy-gated-top-p", "--source-aware-batch"]),
        ("all_latency", [
            "--entropy-gated-top-p",
            "--adaptive-top-p",
            "--confidence-adaptive-wb",
        ]),
        ("all_quality", [
            "--entropy-gated-top-p",
            "--language-pair-gen-cap",
            "--source-aware-batch",
        ]),
        ("all_combined", [
            "--entropy-gated-top-p",
            "--adaptive-top-p",
            "--confidence-adaptive-wb",
            "--language-pair-gen-cap",
            "--source-aware-batch",
        ]),
    ]

    for direction in ALL_DIRECTIONS:
        for combo_name, extra_args in combos:
            output = bench_direction(model, direction, n_sent,
                                      extra_args=extra_args,
                                      label=f"P8-{combo_name}-{direction}")
            if output:
                results[f"{direction}_{combo_name}"] = output
    save_summary(8, results, f"{RESULTS_DIR}/phase8_combined.md")


def phase9_competition(model, n_sent=100):
    """Phase 9: Competition OmniSTEval output for all directions."""
    print("\n" + "="*70)
    print("PHASE 9: Competition OmniSTEval Output")
    print("="*70)
    results = {}
    for direction in ALL_DIRECTIONS:
        cfg = OPTIMAL_CONFIGS[direction]
        out_file = f"{RESULTS_DIR}/omnisteval_{direction.replace('-', '_')}.jsonl"
        args = [
            "--model", model,
            "--lang", direction,
            "--border-distance", str(cfg["bd"]),
            "--word-batch", str(cfg["wb"]),
            "--aggregation", "top_p",
            "--top-p-threshold", str(cfg["p"]),
            "--n-sentences", str(n_sent),
            "--n-gpu-layers", "99",
            "--comet",
            "--save",
            "--omnisteval", out_file,
        ]
        output = run_bench(args, label=f"P9-omnisteval-{direction}")
        if output:
            results[direction] = output
            # Check output file
            if os.path.exists(out_file):
                with open(out_file) as f:
                    lines = f.readlines()
                print(f"  OmniSTEval: {len(lines)} entries written to {out_file}")
    save_summary(9, results, f"{RESULTS_DIR}/phase9_competition.md")


def main():
    parser = argparse.ArgumentParser(description="Iteration 24 GPU experiments")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--phase", type=int, default=-1,
                        help="Run specific phase (0-9), -1=all. Phase 0 = XCOMET-XL baseline.")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 20 sentences instead of 100")
    args = parser.parse_args()

    n_sent = 20 if args.quick else 100
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"NLLW Iteration 24 GPU Experiments")
    print(f"Model: {args.model}")
    print(f"Sentences: {n_sent}")
    print(f"Phase: {'all' if args.phase == -1 else args.phase}")
    print(f"Results dir: {RESULTS_DIR}")

    phases = {
        0: ("XCOMET-XL baseline (COMPETITION METRIC)", lambda: phase0_xcomet_baseline(args.model, n_sent)),
        1: ("Quick baseline validation", lambda: phase1_baseline(args.model, min(n_sent, 20))),
        2: ("Entropy-gated top_p", lambda: phase2_entropy_gated(args.model, n_sent)),
        3: ("Combined gating", lambda: phase3_combined_gating(args.model, n_sent)),
        4: ("Confidence-adaptive wb", lambda: phase4_confidence_wb(args.model, n_sent)),
        5: ("Language-pair gen cap", lambda: phase5_langpair_gencap(args.model, n_sent)),
        6: ("Source-aware batching", lambda: phase6_source_aware(args.model, n_sent)),
        7: ("Perplexity adaptive bd", lambda: phase7_pplbd(args.model, n_sent)),
        8: ("Combined best features", lambda: phase8_combined_best(args.model, n_sent)),
        9: ("Competition OmniSTEval", lambda: phase9_competition(args.model, n_sent)),
    }

    if args.phase >= 0:
        if args.phase not in phases:
            print(f"Unknown phase {args.phase}. Valid: 0-{max(phases.keys())}")
            sys.exit(1)
        name, fn = phases[args.phase]
        print(f"\nRunning Phase {args.phase}: {name}")
        fn()
    else:
        for phase_num, (name, fn) in sorted(phases.items()):
            print(f"\n{'#'*70}")
            print(f"# Phase {phase_num}: {name}")
            print(f"{'#'*70}")
            fn()

    print(f"\n{'='*70}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results saved to {RESULTS_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
