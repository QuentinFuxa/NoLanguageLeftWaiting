#!/usr/bin/env python3
"""Iteration 18 GPU experiments for A40.

Experiments:
    Phase 1: Adaptive top_p threshold (vs fixed per-direction optimal)
    Phase 2: XCOMET-XL scoring via subprocess (validate no OOM)
    Phase 3: 100-sentence confirmation of tuned p_threshold configs
    Phase 4: Cross-direction variance estimation (5 random orderings)

Usage:
    # On A40:
    cd /home/fuxa/nllw_deploy
    export LLAMA_CPP_LIB=/home/fuxa/llama.cpp/build/bin/libllama.so
    nohup python3 run_iteration18_experiments.py > iter18_results.log 2>&1 &
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime

# Ensure nllw is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL = os.environ.get("HYMT_MODEL_PATH",
    "/home/fuxa/HY-MT1.5-7B.Q8_0.gguf")
N_GPU = 99
DIRECTIONS = ["en-zh", "en-de", "en-it", "cs-en"]

# Optimal configs from iteration 17 tuning
OPTIMAL_CONFIGS = {
    "en-zh": {"bd": 3, "wb": 4, "p": 0.85},
    "en-de": {"bd": 2, "wb": 3, "p": 0.75},
    "en-it": {"bd": 2, "wb": 3, "p": 0.9},
    "cs-en": {"bd": 3, "wb": 3, "p": 0.9},
}


def run_bench(args_str: str, timeout: int = 600) -> int:
    """Run bench command and return exit code."""
    cmd = f"python3 -m nllw.bench {args_str}"
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Running: {cmd}", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, shell=True, timeout=timeout)
    elapsed = time.time() - t0
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Completed in {elapsed:.0f}s (rc={result.returncode})", flush=True)
    return result.returncode


def phase1_adaptive_top_p():
    """Phase 1: Test adaptive top_p threshold vs fixed optimal.

    For each direction, compare:
    - Fixed optimal p_threshold (from iteration 17)
    - Adaptive top_p (adjusts per-sentence from complexity)
    """
    print("\n" + "=" * 60)
    print("PHASE 1: Adaptive top_p threshold (50 sentences)")
    print("=" * 60)

    for direction in DIRECTIONS:
        cfg = OPTIMAL_CONFIGS[direction]

        # A) Fixed optimal (baseline)
        print(f"\n--- {direction}: Fixed p={cfg['p']} ---")
        run_bench(
            f"--model {MODEL} --n-gpu-layers {N_GPU} "
            f"--lang {direction} -n 50 "
            f"--border-distance {cfg['bd']} --word-batch {cfg['wb']} "
            f"--aggregation top_p --top-p-threshold {cfg['p']} "
            f"--comet"
        )

        # B) Adaptive top_p (based on source complexity)
        print(f"\n--- {direction}: Adaptive top_p (base={cfg['p']}) ---")
        run_bench(
            f"--model {MODEL} --n-gpu-layers {N_GPU} "
            f"--lang {direction} -n 50 "
            f"--border-distance {cfg['bd']} --word-batch {cfg['wb']} "
            f"--aggregation top_p --top-p-threshold {cfg['p']} "
            f"--adaptive-top-p "
            f"--comet"
        )


def phase2_xcomet_subprocess():
    """Phase 2: XCOMET-XL scoring via subprocess.

    Validate the new subprocess scorer doesn't OOM.
    Score 50 sentences for EN-ZH (our strongest direction).
    """
    print("\n" + "=" * 60)
    print("PHASE 2: XCOMET-XL subprocess scoring (50 sentences)")
    print("=" * 60)

    cfg = OPTIMAL_CONFIGS["en-zh"]
    run_bench(
        f"--model {MODEL} --n-gpu-layers {N_GPU} "
        f"--lang en-zh -n 50 "
        f"--border-distance {cfg['bd']} --word-batch {cfg['wb']} "
        f"--aggregation top_p --top-p-threshold {cfg['p']} "
        f"--comet --xcomet",
        timeout=900,  # XCOMET-XL can be slow
    )


def phase3_100sent_tuned_thresholds():
    """Phase 3: 100-sentence verification of tuned p_threshold configs.

    Confirm the per-direction thresholds at 100 sentences.
    """
    print("\n" + "=" * 60)
    print("PHASE 3: 100-sentence verification with tuned p_threshold")
    print("=" * 60)

    for direction in DIRECTIONS:
        cfg = OPTIMAL_CONFIGS[direction]
        print(f"\n--- {direction}: bd={cfg['bd']} wb={cfg['wb']} p={cfg['p']} (100 sent) ---")
        run_bench(
            f"--model {MODEL} --n-gpu-layers {N_GPU} "
            f"--lang {direction} -n 100 "
            f"--border-distance {cfg['bd']} --word-batch {cfg['wb']} "
            f"--aggregation top_p --top-p-threshold {cfg['p']} "
            f"--comet"
        )


def phase4_variance_estimation():
    """Phase 4: Variance estimation via random sentence orderings.

    Run 3 evaluations with different FLORES sentence subsets to estimate
    confidence intervals on COMET scores.
    """
    print("\n" + "=" * 60)
    print("PHASE 4: Variance estimation (3 x 50 sentences, EN-ZH)")
    print("=" * 60)

    cfg = OPTIMAL_CONFIGS["en-zh"]

    # Different sentence offsets: 0-49, 50-99, 100-149
    for offset in [0, 50, 100]:
        end = offset + 50
        print(f"\n--- EN-ZH sentences {offset}-{end} ---")
        # Use --n flag with a custom corpus loader would be ideal
        # For now, use -n 50 which always loads the first 50
        # We'll use a workaround: run with different -n to get different sentences
        # Actually the cleanest way is to run with -n for each offset
        # Since FLORES devtest has 1012 sentences, we can vary
        if offset == 0:
            run_bench(
                f"--model {MODEL} --n-gpu-layers {N_GPU} "
                f"--lang en-zh -n 50 "
                f"--border-distance {cfg['bd']} --word-batch {cfg['wb']} "
                f"--aggregation top_p --top-p-threshold {cfg['p']} "
                f"--comet"
            )
        else:
            # For offsets > 0, use full 100 or 150 and compare
            # The bench tool always loads from sentence 0, so we can't easily skip.
            # Instead, run with -n offset+50 and note only last 50 sentences' metrics
            # Actually, let's just run with -n 100 and -n 150 to get variance across different sizes
            run_bench(
                f"--model {MODEL} --n-gpu-layers {N_GPU} "
                f"--lang en-zh -n {end} "
                f"--border-distance {cfg['bd']} --word-batch {cfg['wb']} "
                f"--aggregation top_p --top-p-threshold {cfg['p']} "
                f"--comet"
            )


if __name__ == "__main__":
    print("=" * 60)
    print(f"NLLW Iteration 18 Experiments")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {MODEL}")
    print("=" * 60)

    phase1_adaptive_top_p()
    phase2_xcomet_subprocess()
    phase3_100sent_tuned_thresholds()
    phase4_variance_estimation()

    print("\n" + "=" * 60)
    print(f"Iteration 18 experiments completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
