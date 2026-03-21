#!/usr/bin/env python3
"""Iteration 25 GPU experiments -- Competition-ready evaluation + new features.

Key changes in iteration 25:
    - Generation temperature (NOVEL): low-temperature sampling to explore
      alternative translations near the greedy path. 0.1-0.3 recommended.
    - Confidence-gated token trimming (NOVEL): trim low-confidence trailing
      tokens after generation. Prevents committing hallucinated endings.
    - Competition-focused: XCOMET-XL + StreamLAAL + SacreBLEU evaluation.
    - Official HY-MT prompt format A/B test.

CRITICAL: Competition eval April 1-15, 2026. XCOMET-XL is the primary metric.

Phases:
    0. XCOMET-XL baseline: re-evaluate all 4 directions with XCOMET-XL
    1. Official HY-MT prompt: A/B test hymt vs hymt-official format
    2. Generation temperature: sweep temp=0.0,0.05,0.1,0.2,0.3
    3. Confidence trimming: sweep conftrim=-2.0,-3.0,-4.0,-5.0
    4. Entropy-gated top_p: iter 24 feature validation
    5. Confidence-adaptive wb: iter 23 feature validation
    6. Language-pair gen cap: iter 23 feature validation
    7. Combined features: test all winning features together
    8. Competition OmniSTEval: generate final submission JSONL

Usage:
    # Full run on A40:
    python scripts/run_iteration25_experiments.py --model /home/fuxa/HY-MT1.5-7B.Q8_0.gguf

    # XCOMET-XL baseline only (most urgent):
    python scripts/run_iteration25_experiments.py --model /path/to/model.gguf --phase 0

    # Quick mode (20 sentences):
    python scripts/run_iteration25_experiments.py --model /path/to/model.gguf --quick

    # Specific phases:
    python scripts/run_iteration25_experiments.py --model /path/to/model.gguf --phase 0,1,2
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RESULTS_DIR = "results/iteration25"
ALL_DIRECTIONS = ["en-zh", "en-de", "en-it", "cs-en"]

# Best known configs per direction (iteration 18 verified, COMET wmt22)
OPTIMAL_CONFIGS = {
    "en-zh": {"bd": 3, "wb": 4, "p": 0.85},
    "en-de": {"bd": 2, "wb": 3, "p": 0.75},
    "en-it": {"bd": 2, "wb": 3, "p": 0.9},
    "cs-en": {"bd": 3, "wb": 3, "p": 0.9},
}

# Reference results from iteration 18 (COMET wmt22-comet-da, 100 sentences)
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
    print(result.stdout)
    if result.stderr:
        # Only print last 20 lines of stderr to avoid noise
        err_lines = result.stderr.strip().split("\n")
        if len(err_lines) > 20:
            print(f"  ... ({len(err_lines) - 20} lines of stderr omitted)")
        for line in err_lines[-20:]:
            print(f"  stderr: {line}")
    print(f"[{label}] Completed in {elapsed:.1f}s (rc={result.returncode})")
    return result


def base_args(direction, model, n_sentences, use_xcomet=False):
    """Build base arguments for a given direction."""
    cfg = OPTIMAL_CONFIGS[direction]
    args = [
        "--lang", direction,
        "--model", model,
        "--border-distance", str(cfg["bd"]),
        "--word-batch", str(cfg["wb"]),
        "--aggregation", "top_p",
        "--top-p-threshold", str(cfg["p"]),
        "-n", str(n_sentences),
        "--n-gpu-layers", "99",
    ]
    if use_xcomet:
        args.append("--xcomet")
    else:
        args.append("--comet")
    args.append("--save")
    return args


def phase_0_xcomet_baseline(model, n_sent, results_dir):
    """Phase 0: XCOMET-XL baseline evaluation for all directions.

    This is the MOST CRITICAL phase. We need XCOMET-XL scores to know
    where we actually stand with the competition metric.
    """
    print("\n" + "="*70)
    print("PHASE 0: XCOMET-XL BASELINE (CRITICAL)")
    print("Re-evaluate all 4 directions with competition metric")
    print("="*70)

    for direction in ALL_DIRECTIONS:
        args = base_args(direction, model, n_sent, use_xcomet=True)
        run_bench(args, label=f"XCOMET baseline {direction}")


def phase_1_prompt_ab_test(model, n_sent, results_dir):
    """Phase 1: Official HY-MT prompt format A/B test.

    Compare our current prompt format vs the official HY-MT training format.
    The official format uses exact wording from training data.
    """
    print("\n" + "="*70)
    print("PHASE 1: HY-MT PROMPT A/B TEST")
    print("="*70)

    for direction in ALL_DIRECTIONS:
        # Current format
        args_current = base_args(direction, model, n_sent)
        args_current.extend(["--prompt-format", "hymt"])
        run_bench(args_current, label=f"prompt hymt {direction}")

        # Official format
        args_official = base_args(direction, model, n_sent)
        args_official.extend(["--prompt-format", "hymt-official"])
        run_bench(args_official, label=f"prompt hymt-official {direction}")


def phase_2_temperature(model, n_sent, results_dir):
    """Phase 2: Generation temperature sweep.

    Test low-temperature sampling vs greedy decoding.
    Low temperatures (0.05-0.2) can help escape suboptimal greedy paths.
    """
    print("\n" + "="*70)
    print("PHASE 2: GENERATION TEMPERATURE SWEEP")
    print("="*70)

    # Test on EN-ZH first (our strongest direction)
    direction = "en-zh"
    temps = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    for temp in temps:
        args = base_args(direction, model, n_sent)
        args.extend(["--generation-temperature", str(temp)])
        run_bench(args, label=f"temp={temp} {direction}")

    # EDT (entropy-based dynamic temperature)
    args_edt = base_args(direction, model, n_sent)
    args_edt.extend(["--entropy-dynamic-temperature", "--generation-temperature", "0.1"])
    run_bench(args_edt, label=f"EDT temp=0.1 {direction}")

    args_edt2 = base_args(direction, model, n_sent)
    args_edt2.extend(["--entropy-dynamic-temperature", "--generation-temperature", "0.2"])
    run_bench(args_edt2, label=f"EDT temp=0.2 {direction}")

    # If any temperature helps, test on all directions
    # (manual decision based on results)


def phase_3_confidence_trim(model, n_sent, results_dir):
    """Phase 3: Confidence-gated token trimming sweep.

    Test trimming trailing low-confidence tokens before commit.
    """
    print("\n" + "="*70)
    print("PHASE 3: CONFIDENCE TRIMMING SWEEP")
    print("="*70)

    direction = "en-zh"
    thresholds = [None, -2.0, -3.0, -4.0, -5.0]
    for thr in thresholds:
        args = base_args(direction, model, n_sent)
        if thr is not None:
            args.extend(["--confidence-trim", str(thr)])
        run_bench(args, label=f"conftrim={thr} {direction}")

    # Test on EN-DE (different language dynamics)
    direction = "en-de"
    for thr in [None, -3.0]:
        args = base_args(direction, model, n_sent)
        if thr is not None:
            args.extend(["--confidence-trim", str(thr)])
        run_bench(args, label=f"conftrim={thr} {direction}")


def phase_4_entropy_gated(model, n_sent, results_dir):
    """Phase 4: Entropy-gated top_p validation (from iter 24)."""
    print("\n" + "="*70)
    print("PHASE 4: ENTROPY-GATED TOP_P")
    print("="*70)

    for direction in ALL_DIRECTIONS:
        # Baseline
        args = base_args(direction, model, n_sent)
        run_bench(args, label=f"baseline {direction}")

        # Entropy-gated
        args = base_args(direction, model, n_sent)
        args.append("--entropy-gated-top-p")
        run_bench(args, label=f"entropy-gated {direction}")


def phase_5_confidence_wb(model, n_sent, results_dir):
    """Phase 5: Confidence-adaptive word batching (from iter 23)."""
    print("\n" + "="*70)
    print("PHASE 5: CONFIDENCE-ADAPTIVE WB")
    print("="*70)

    for direction in ALL_DIRECTIONS:
        args = base_args(direction, model, n_sent)
        args.append("--confidence-adaptive-wb")
        run_bench(args, label=f"confwb {direction}")


def phase_6_lang_gen_cap(model, n_sent, results_dir):
    """Phase 6: Language-pair generation cap (from iter 23)."""
    print("\n" + "="*70)
    print("PHASE 6: LANGUAGE-PAIR GEN CAP")
    print("="*70)

    for direction in ALL_DIRECTIONS:
        args = base_args(direction, model, n_sent)
        args.append("--language-pair-gen-cap")
        run_bench(args, label=f"lpgcap {direction}")


def phase_7_combined(model, n_sent, results_dir):
    """Phase 7: Combined features -- test all winning features together.

    Based on results from phases 2-6, combine the winners.
    """
    print("\n" + "="*70)
    print("PHASE 7: COMBINED FEATURES")
    print("="*70)

    combos = [
        # Conservative: just the most likely winners
        {"label": "entgtp+lpgcap",
         "flags": ["--entropy-gated-top-p", "--language-pair-gen-cap"]},
        # Aggressive: all latency features
        {"label": "entgtp+confwb+lpgcap",
         "flags": ["--entropy-gated-top-p", "--confidence-adaptive-wb",
                    "--language-pair-gen-cap"]},
        # EDT + trimming
        {"label": "edt0.1+conftrim-3",
         "flags": ["--entropy-dynamic-temperature",
                    "--generation-temperature", "0.1",
                    "--confidence-trim", "-3.0"]},
        # All features
        {"label": "all_features",
         "flags": ["--entropy-gated-top-p", "--confidence-adaptive-wb",
                    "--language-pair-gen-cap",
                    "--entropy-dynamic-temperature",
                    "--generation-temperature", "0.1",
                    "--confidence-trim", "-3.0"]},
        # Kitchen sink with perplexity adaptive bd
        {"label": "kitchen_sink",
         "flags": ["--entropy-gated-top-p", "--confidence-adaptive-wb",
                    "--language-pair-gen-cap", "--perplexity-adaptive-bd"]},
    ]

    for direction in ALL_DIRECTIONS:
        for combo in combos:
            args = base_args(direction, model, n_sent)
            args.extend(combo["flags"])
            run_bench(args, label=f"{combo['label']} {direction}")


def phase_8_competition(model, n_sent, results_dir):
    """Phase 8: Generate competition OmniSTEval output.

    Produces JSONL files for each direction in OmniSTEval format.
    """
    print("\n" + "="*70)
    print("PHASE 8: COMPETITION OMNISTEVAL OUTPUT")
    print("="*70)

    output_dir = os.path.join(results_dir, "competition_output")
    os.makedirs(output_dir, exist_ok=True)

    for direction in ALL_DIRECTIONS:
        args = base_args(direction, model, n_sent, use_xcomet=True)
        output_file = os.path.join(output_dir, f"omnisteval_{direction}.jsonl")
        args.extend(["--omnisteval", output_file])
        run_bench(args, label=f"competition {direction}", timeout=7200)


def main():
    parser = argparse.ArgumentParser(
        description="Iteration 25 competition experiments"
    )
    parser.add_argument("--model", required=True,
                        help="Path to GGUF model file")
    parser.add_argument("--phase", type=str, default=None,
                        help="Comma-separated phase numbers (0-8). None=all.")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 20 sentences instead of 100")
    parser.add_argument("--n-sentences", type=int, default=None,
                        help="Override sentence count")
    args = parser.parse_args()

    n_sent = args.n_sentences or (20 if args.quick else 100)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    phases = {
        0: ("XCOMET-XL Baseline", phase_0_xcomet_baseline),
        1: ("HY-MT Prompt A/B", phase_1_prompt_ab_test),
        2: ("Temperature Sweep", phase_2_temperature),
        3: ("Confidence Trimming", phase_3_confidence_trim),
        4: ("Entropy-Gated top_p", phase_4_entropy_gated),
        5: ("Confidence-Adaptive WB", phase_5_confidence_wb),
        6: ("Language-Pair Gen Cap", phase_6_lang_gen_cap),
        7: ("Combined Features", phase_7_combined),
        8: ("Competition Output", phase_8_competition),
    }

    if args.phase is not None:
        selected = [int(p.strip()) for p in args.phase.split(",")]
    else:
        selected = list(phases.keys())

    print(f"\nIteration 25 Experiments")
    print(f"Model: {args.model}")
    print(f"Sentences: {n_sent}")
    print(f"Phases: {selected}")
    print(f"Results: {RESULTS_DIR}")
    print(f"{'='*70}")

    t_start = time.time()
    for phase_num in selected:
        if phase_num not in phases:
            print(f"Unknown phase {phase_num}")
            continue
        name, func = phases[phase_num]
        print(f"\n>>> Starting Phase {phase_num}: {name}")
        func(args.model, n_sent, RESULTS_DIR)

    total = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"All phases completed in {total/60:.1f} minutes")
    print(f"Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
