#!/usr/bin/env python3
"""Iteration 26 GPU experiments -- Anti-LM contrastive decoding + final competition tuning.

Key changes in iteration 26:
    - Anti-LM contrastive decoding (NAACL 2024, arxiv 2311.08324):
      Subtract source-language continuation penalty from translation logits.
      Prevents hallucination and source copying. O(1) extra forward pass.
    - Competition-focused: final parameter tuning before IWSLT 2026 eval.

CRITICAL: Competition eval April 1-15, 2026. XCOMET-XL is the primary metric.

Phases:
    0. XCOMET-XL baseline: re-evaluate all 4 directions (if not done in iter 25)
    1. Anti-LM contrastive decoding: gamma sweep + per-direction validation
    2. Anti-LM + confidence trim: combined hallucination prevention
    3. Anti-LM + EDT: contrastive + dynamic temperature
    4. Best features combined: final competition config
    5. Competition OmniSTEval: generate final submission JSONL

Usage:
    # Full run on A40:
    python scripts/run_iteration26_experiments.py --model /home/fuxa/HY-MT1.5-7B.Q8_0.gguf

    # Anti-LM only:
    python scripts/run_iteration26_experiments.py --model /path/to/model.gguf --phase 1

    # Quick mode (20 sentences):
    python scripts/run_iteration26_experiments.py --model /path/to/model.gguf --quick
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

RESULTS_DIR = "results/iteration26"
ALL_DIRECTIONS = ["en-zh", "en-de", "en-it", "cs-en"]

# Best known configs per direction (iteration 18 verified, COMET wmt22)
OPTIMAL_CONFIGS = {
    "en-zh": {"bd": 3, "wb": 4, "p": 0.85},
    "en-de": {"bd": 2, "wb": 3, "p": 0.75},
    "en-it": {"bd": 2, "wb": 3, "p": 0.9},
    "cs-en": {"bd": 3, "wb": 3, "p": 0.9},
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
    """Phase 0: XCOMET-XL baseline (run if not already done in iter 25)."""
    print("\n" + "="*70)
    print("PHASE 0: XCOMET-XL BASELINE")
    print("="*70)

    for direction in ALL_DIRECTIONS:
        args = base_args(direction, model, n_sent, use_xcomet=True)
        run_bench(args, label=f"XCOMET baseline {direction}")


def phase_1_anti_lm(model, n_sent, results_dir):
    """Phase 1: Anti-LM contrastive decoding.

    Test Anti-LM with different gamma decay rates.
    gamma=0.3 is recommended by Sia et al. (NAACL 2024).
    """
    print("\n" + "="*70)
    print("PHASE 1: ANTI-LM CONTRASTIVE DECODING")
    print("="*70)

    # Gamma sweep on EN-ZH (our best direction)
    direction = "en-zh"
    gammas = [0.1, 0.2, 0.3, 0.5, 0.8]
    for gamma in gammas:
        args = base_args(direction, model, n_sent)
        args.extend(["--anti-lm", "--anti-lm-gamma", str(gamma)])
        run_bench(args, label=f"antilm gamma={gamma} {direction}")

    # Baseline comparison (no anti-LM)
    args = base_args(direction, model, n_sent)
    run_bench(args, label=f"baseline (no antilm) {direction}")

    # If gamma=0.3 works, test on all directions
    for direction in ["en-de", "en-it", "cs-en"]:
        args = base_args(direction, model, n_sent)
        args.extend(["--anti-lm", "--anti-lm-gamma", "0.3"])
        run_bench(args, label=f"antilm gamma=0.3 {direction}")


def phase_2_anti_lm_trim(model, n_sent, results_dir):
    """Phase 2: Anti-LM + confidence trimming.

    Both target hallucination prevention. Test if they compose well.
    """
    print("\n" + "="*70)
    print("PHASE 2: ANTI-LM + CONFIDENCE TRIMMING")
    print("="*70)

    for direction in ["en-zh", "en-de"]:
        # Anti-LM only
        args = base_args(direction, model, n_sent)
        args.extend(["--anti-lm"])
        run_bench(args, label=f"antilm only {direction}")

        # Confidence trim only
        args = base_args(direction, model, n_sent)
        args.extend(["--confidence-trim", "-3.0"])
        run_bench(args, label=f"conftrim only {direction}")

        # Combined
        args = base_args(direction, model, n_sent)
        args.extend(["--anti-lm", "--confidence-trim", "-3.0"])
        run_bench(args, label=f"antilm+conftrim {direction}")


def phase_3_anti_lm_edt(model, n_sent, results_dir):
    """Phase 3: Anti-LM + EDT (entropy-based dynamic temperature).

    Anti-LM modifies logits, EDT adjusts sampling temperature.
    They should compose well: Anti-LM reduces source copying,
    EDT explores alternatives more on uncertain tokens.
    """
    print("\n" + "="*70)
    print("PHASE 3: ANTI-LM + EDT")
    print("="*70)

    for direction in ["en-zh", "en-de"]:
        # Anti-LM + EDT
        args = base_args(direction, model, n_sent)
        args.extend([
            "--anti-lm",
            "--entropy-dynamic-temperature",
            "--generation-temperature", "0.1",
        ])
        run_bench(args, label=f"antilm+edt {direction}")

        # Anti-LM + EDT + confidence trim
        args = base_args(direction, model, n_sent)
        args.extend([
            "--anti-lm",
            "--entropy-dynamic-temperature",
            "--generation-temperature", "0.1",
            "--confidence-trim", "-3.0",
        ])
        run_bench(args, label=f"antilm+edt+trim {direction}")


def phase_4_best_combined(model, n_sent, results_dir):
    """Phase 4: Best features combined for competition.

    Test combinations of all promising features from iterations 20-26.
    """
    print("\n" + "="*70)
    print("PHASE 4: BEST FEATURES COMBINED")
    print("="*70)

    combos = [
        # Anti-LM + entropy-gated top_p (latency + hallucination)
        {"label": "antilm+entgtp",
         "flags": ["--anti-lm", "--entropy-gated-top-p"]},
        # Anti-LM + all latency features
        {"label": "antilm+entgtp+confwb+lpgcap",
         "flags": ["--anti-lm", "--entropy-gated-top-p",
                    "--confidence-adaptive-wb", "--language-pair-gen-cap"]},
        # Anti-LM + EDT + trim (hallucination combo)
        {"label": "antilm+edt+trim",
         "flags": ["--anti-lm", "--entropy-dynamic-temperature",
                    "--generation-temperature", "0.1",
                    "--confidence-trim", "-3.0"]},
        # Everything promising
        {"label": "full_combo",
         "flags": ["--anti-lm", "--entropy-gated-top-p",
                    "--confidence-adaptive-wb", "--language-pair-gen-cap",
                    "--confidence-trim", "-3.0"]},
        # Conservative: just anti-LM + trim (minimal changes)
        {"label": "conservative",
         "flags": ["--anti-lm", "--confidence-trim", "-4.0"]},
    ]

    for direction in ALL_DIRECTIONS:
        # Always run baseline for comparison
        args = base_args(direction, model, n_sent)
        run_bench(args, label=f"baseline {direction}")

        for combo in combos:
            args = base_args(direction, model, n_sent)
            args.extend(combo["flags"])
            run_bench(args, label=f"{combo['label']} {direction}")


def phase_5_competition(model, n_sent, results_dir):
    """Phase 5: Generate competition OmniSTEval output with XCOMET-XL."""
    print("\n" + "="*70)
    print("PHASE 5: COMPETITION OMNISTEVAL OUTPUT")
    print("="*70)

    output_dir = os.path.join(results_dir, "competition_output")
    os.makedirs(output_dir, exist_ok=True)

    for direction in ALL_DIRECTIONS:
        args = base_args(direction, model, n_sent, use_xcomet=True)
        # Add best known features (update after phase 4 results)
        args.extend(["--anti-lm"])
        output_file = os.path.join(output_dir, f"omnisteval_{direction}.jsonl")
        args.extend(["--omnisteval", output_file])
        run_bench(args, label=f"competition {direction}", timeout=7200)


def main():
    parser = argparse.ArgumentParser(
        description="Iteration 26 experiments: Anti-LM + competition tuning"
    )
    parser.add_argument("--model", required=True,
                        help="Path to GGUF model file")
    parser.add_argument("--phase", type=str, default=None,
                        help="Comma-separated phase numbers (0-5). None=all.")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 20 sentences instead of 100")
    parser.add_argument("--n-sentences", type=int, default=None,
                        help="Override sentence count")
    args = parser.parse_args()

    n_sent = args.n_sentences or (20 if args.quick else 100)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    phases = {
        0: ("XCOMET-XL Baseline", phase_0_xcomet_baseline),
        1: ("Anti-LM Contrastive Decoding", phase_1_anti_lm),
        2: ("Anti-LM + Confidence Trimming", phase_2_anti_lm_trim),
        3: ("Anti-LM + EDT", phase_3_anti_lm_edt),
        4: ("Best Combined Features", phase_4_best_combined),
        5: ("Competition Output", phase_5_competition),
    }

    if args.phase is not None:
        selected = [int(p.strip()) for p in args.phase.split(",")]
    else:
        selected = list(phases.keys())

    print(f"\nIteration 26 Experiments")
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
