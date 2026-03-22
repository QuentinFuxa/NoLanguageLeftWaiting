#!/usr/bin/env python3
"""Iteration 27 GPU experiments -- Comprehensive XCOMET-XL evaluation + parameter re-optimization.

TWO-PHASE ARCHITECTURE:
    Phase A: Run all translations (COMET + save hypotheses). Fast, no OOM.
    Phase B: Score all saved hypotheses with XCOMET-XL in a separate process.
             No llama.cpp loaded -> full GPU available for XCOMET-XL.

This avoids the persistent XCOMET-XL OOM issue (llama.cpp CUDA allocator
doesn't release memory even after model free).

Experiment Phases:
    0. Baseline: current best configs (COMET + hypotheses saved for XCOMET)
    1. bd/wb re-optimization: sweep bd/wb
    2. top_p threshold re-optimization: sweep p
    3. Feature validation: entropy-gated, confidence-adaptive, source-aware, etc.
    4. Anti-LM validation: gamma sweep + per-direction
    5. Best combined: winning features from phases 0-4
    6. Final refinement: re-translate from scratch on is_final
    X. XCOMET scoring: score all saved hypotheses (run separately)

Usage:
    # Full run on A40 (translation + COMET):
    LLAMA_CPP_LIB=/home/fuxa/llama.cpp/build/bin/libllama.so \\
        python3 scripts/run_iteration27_experiments.py --model /home/fuxa/HY-MT1.5-7B.Q8_0.gguf --quick

    # Score all hypotheses with XCOMET-XL (no model needed, run after phase A):
    python3 scripts/run_iteration27_experiments.py --score-xcomet

    # Specific phases:
    LLAMA_CPP_LIB=/home/fuxa/llama.cpp/build/bin/libllama.so \\
        python3 scripts/run_iteration27_experiments.py --model /home/fuxa/HY-MT1.5-7B.Q8_0.gguf --phase 0,3
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Add parent directory to path for nllw imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = "results/iteration27"
HYPO_DIR = "results/iteration27/hypotheses"
ALL_DIRECTIONS = ["en-zh", "en-de", "en-it", "cs-en"]

# Best known configs per direction (iteration 18, COMET wmt22)
ITER18_CONFIGS = {
    "en-zh": {"bd": 3, "wb": 4, "p": 0.85},
    "en-de": {"bd": 2, "wb": 3, "p": 0.75},
    "en-it": {"bd": 2, "wb": 3, "p": 0.9},
    "cs-en": {"bd": 3, "wb": 3, "p": 0.9},
}


def run_bench(args_list, label="", timeout=3600, hypo_file=None):
    """Run a bench command with optional hypothesis saving."""
    cmd = [sys.executable, "-m", "nllw.bench"] + args_list
    if hypo_file:
        cmd.extend(["--save-hypotheses", hypo_file])
    print(f"\n{'='*70}")
    print(f"[{label}] Running: {' '.join(cmd)}")
    print(f"{'='*70}", flush=True)
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[{label}] TIMEOUT after {timeout}s", flush=True)
        return None
    elapsed = time.time() - t0
    # Print stderr summary (quality/latency metrics)
    if result.stderr:
        err_lines = result.stderr.strip().split("\n")
        for line in err_lines:
            if any(k in line for k in ["===", "Quality:", "Latency:", "Saved", "COMET", "BLEU"]):
                print(f"  {line.strip()}")
    print(f"[{label}] Completed in {elapsed:.1f}s (rc={result.returncode})", flush=True)
    return result


def base_args(direction, model, n_sentences, bd=None, wb=None, p=None):
    """Build base arguments for a given direction."""
    cfg = ITER18_CONFIGS[direction]
    return [
        "--lang", direction,
        "--model", model,
        "--border-distance", str(bd or cfg["bd"]),
        "--word-batch", str(wb or cfg["wb"]),
        "--aggregation", "top_p",
        "--top-p-threshold", str(p or cfg["p"]),
        "-n", str(n_sentences),
        "--n-gpu-layers", "99",
        "--comet",
        "--save",
    ]


def hypo_path(label, direction):
    """Generate hypothesis file path."""
    safe_label = label.replace(" ", "_").replace("=", "").replace(",", "_")
    return os.path.join(HYPO_DIR, f"{safe_label}_{direction}.json")


def phase_0_baseline(model, n_sent, results_dir):
    """Phase 0: Baseline with current best configs."""
    print("\n" + "="*70)
    print("PHASE 0: BASELINE (current best configs from iter 18)")
    print("="*70, flush=True)

    for direction in ALL_DIRECTIONS:
        args = base_args(direction, model, n_sent)
        hp = hypo_path("baseline", direction)
        run_bench(args, label=f"baseline {direction}", hypo_file=hp)


def phase_1_bd_wb_sweep(model, n_sent, results_dir):
    """Phase 1: Re-optimize bd/wb."""
    print("\n" + "="*70)
    print("PHASE 1: BD/WB SWEEP")
    print("="*70, flush=True)

    for direction in ALL_DIRECTIONS:
        cfg = ITER18_CONFIGS[direction]
        base_bd = cfg["bd"]
        base_wb = cfg["wb"]

        bds = sorted(set([max(1, base_bd - 1), base_bd, base_bd + 1]))
        wbs = sorted(set([max(1, base_wb - 1), base_wb, base_wb + 1, base_wb + 2]))

        for bd in bds:
            for wb in wbs:
                if bd == base_bd and wb == base_wb:
                    continue
                label = f"bd{bd}_wb{wb}"
                args = base_args(direction, model, n_sent, bd=bd, wb=wb)
                hp = hypo_path(label, direction)
                run_bench(args, label=f"{label} {direction}", hypo_file=hp)


def phase_2_topp_sweep(model, n_sent, results_dir):
    """Phase 2: Re-optimize top_p threshold."""
    print("\n" + "="*70)
    print("PHASE 2: TOP_P THRESHOLD SWEEP")
    print("="*70, flush=True)

    p_values = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    for direction in ALL_DIRECTIONS:
        cfg = ITER18_CONFIGS[direction]
        for p in p_values:
            if abs(p - cfg["p"]) < 0.001:
                continue
            label = f"p{p}"
            args = base_args(direction, model, n_sent, p=p)
            hp = hypo_path(label, direction)
            run_bench(args, label=f"{label} {direction}", hypo_file=hp)


def phase_3_features(model, n_sent, results_dir):
    """Phase 3: Individual feature validation."""
    print("\n" + "="*70)
    print("PHASE 3: FEATURE VALIDATION")
    print("="*70, flush=True)

    features = [
        ("entropy_gated", ["--entropy-gated-top-p"]),
        ("conf_wb", ["--confidence-adaptive-wb"]),
        ("lp_gencap", ["--language-pair-gen-cap"]),
        ("src_aware", ["--source-aware-batch"]),
        ("ppl_bd", ["--perplexity-adaptive-bd"]),
        ("trim3", ["--confidence-trim", "-3.0"]),
        ("trim4", ["--confidence-trim", "-4.0"]),
        ("temp01", ["--generation-temperature", "0.1"]),
        ("edt", ["--entropy-dynamic-temperature", "--generation-temperature", "0.1"]),
        ("refine", ["--final-refinement"]),
    ]

    for direction in ["en-zh", "en-de"]:
        for feat_label, feat_args in features:
            args = base_args(direction, model, n_sent)
            args.extend(feat_args)
            hp = hypo_path(feat_label, direction)
            run_bench(args, label=f"{feat_label} {direction}", hypo_file=hp)


def phase_4_anti_lm(model, n_sent, results_dir):
    """Phase 4: Anti-LM contrastive decoding."""
    print("\n" + "="*70)
    print("PHASE 4: ANTI-LM VALIDATION")
    print("="*70, flush=True)

    gammas = [0.1, 0.3, 0.5]
    for gamma in gammas:
        args = base_args("en-zh", model, n_sent)
        args.extend(["--anti-lm", "--anti-lm-gamma", str(gamma)])
        hp = hypo_path(f"antilm_g{gamma}", "en-zh")
        run_bench(args, label=f"antilm gamma={gamma} en-zh", hypo_file=hp)

    for direction in ["en-de", "en-it", "cs-en"]:
        args = base_args(direction, model, n_sent)
        args.extend(["--anti-lm", "--anti-lm-gamma", "0.3"])
        hp = hypo_path("antilm_g03", direction)
        run_bench(args, label=f"antilm gamma=0.3 {direction}", hypo_file=hp)


def phase_5_combined(model, n_sent, results_dir):
    """Phase 5: Best combined features."""
    print("\n" + "="*70)
    print("PHASE 5: BEST COMBINED FEATURES")
    print("="*70, flush=True)

    combos = [
        ("latency", ["--entropy-gated-top-p", "--confidence-adaptive-wb",
                      "--language-pair-gen-cap"]),
        ("quality", ["--anti-lm", "--confidence-trim", "-3.0"]),
        ("balanced", ["--anti-lm", "--entropy-gated-top-p",
                       "--confidence-trim", "-4.0"]),
        ("full", ["--anti-lm", "--entropy-gated-top-p",
                   "--confidence-adaptive-wb", "--language-pair-gen-cap",
                   "--confidence-trim", "-3.0"]),
        ("antilm_only", ["--anti-lm"]),
        ("refine_antilm", ["--anti-lm", "--final-refinement"]),
        ("refine_quality", ["--anti-lm", "--confidence-trim", "-3.0",
                             "--final-refinement"]),
    ]

    for direction in ALL_DIRECTIONS:
        for combo_label, combo_flags in combos:
            args = base_args(direction, model, n_sent)
            args.extend(combo_flags)
            hp = hypo_path(combo_label, direction)
            run_bench(args, label=f"{combo_label} {direction}", hypo_file=hp)


def phase_6_refinement(model, n_sent, results_dir):
    """Phase 6: Final refinement feature test."""
    print("\n" + "="*70)
    print("PHASE 6: SENTENCE-FINAL REFINEMENT")
    print("="*70, flush=True)

    for direction in ALL_DIRECTIONS:
        args = base_args(direction, model, n_sent)
        args.extend(["--final-refinement"])
        hp = hypo_path("refine_only", direction)
        run_bench(args, label=f"refinement {direction}", hypo_file=hp)


def score_all_xcomet(results_dir):
    """Score all saved hypothesis files with XCOMET-XL.

    This runs WITHOUT llama.cpp loaded, so full GPU is available.
    """
    print("\n" + "="*70)
    print("XCOMET-XL SCORING (all saved hypotheses)")
    print("="*70, flush=True)

    hypo_files = sorted(glob.glob(os.path.join(HYPO_DIR, "*.json")))
    if not hypo_files:
        print("No hypothesis files found! Run translation phases first.", flush=True)
        return

    print(f"Found {len(hypo_files)} hypothesis files to score.", flush=True)

    xcomet_results = {}
    for hf in hypo_files:
        label = Path(hf).stem
        print(f"\n--- Scoring: {label} ---", flush=True)
        t0 = time.time()

        try:
            with open(hf) as f:
                data = json.load(f)

            sources = data.get("sources", [])
            hypotheses = data.get("hypotheses", [])
            references = data.get("references", [])
            direction = data.get("direction", "unknown")
            comet = data.get("comet")

            if not sources or not hypotheses or not references:
                print(f"  Skipping {label}: missing data", flush=True)
                continue

            # Score via subprocess (no llama.cpp loaded)
            from nllw.xcomet_scorer import score_xcomet
            system_score, per_scores = score_xcomet(
                sources, hypotheses, references,
                batch_size=4,
            )

            elapsed = time.time() - t0
            xcomet_results[label] = {
                "xcomet": system_score,
                "comet": comet,
                "direction": direction,
                "n_sentences": len(sources),
            }
            print(f"  XCOMET-XL: {system_score:.4f} | COMET: {comet} | {direction} | {elapsed:.1f}s",
                  flush=True)

        except Exception as e:
            print(f"  ERROR scoring {label}: {e}", flush=True)

    # Save summary
    summary_path = os.path.join(results_dir, "xcomet_summary.json")
    with open(summary_path, "w") as f:
        json.dump(xcomet_results, f, indent=2)
    print(f"\nSaved XCOMET-XL summary to {summary_path}", flush=True)

    # Print ranking
    print(f"\n{'='*70}")
    print("XCOMET-XL RESULTS RANKING")
    print(f"{'='*70}")
    for direction in ALL_DIRECTIONS:
        dir_results = {k: v for k, v in xcomet_results.items()
                       if v.get("direction") == direction}
        if dir_results:
            print(f"\n  {direction}:")
            ranked = sorted(dir_results.items(),
                           key=lambda x: x[1]["xcomet"], reverse=True)
            for i, (label, data) in enumerate(ranked):
                marker = " *BEST*" if i == 0 else ""
                print(f"    {label}: XCOMET={data['xcomet']:.4f} COMET={data.get('comet', 'N/A')}{marker}")


def main():
    parser = argparse.ArgumentParser(
        description="Iteration 27 experiments: comprehensive evaluation"
    )
    parser.add_argument("--model", default=None,
                        help="Path to GGUF model file (not needed for --score-xcomet)")
    parser.add_argument("--phase", type=str, default=None,
                        help="Comma-separated phase numbers (0-6). None=all.")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 20 sentences instead of 100")
    parser.add_argument("--n-sentences", type=int, default=None,
                        help="Override sentence count")
    parser.add_argument("--score-xcomet", action="store_true",
                        help="Score all saved hypotheses with XCOMET-XL (no model needed)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(HYPO_DIR, exist_ok=True)

    # Phase B: XCOMET scoring only
    if args.score_xcomet:
        score_all_xcomet(RESULTS_DIR)
        return

    # Phase A: Translation + COMET
    if not args.model:
        parser.error("--model is required for translation phases")

    n_sent = args.n_sentences or (20 if args.quick else 100)

    phases = {
        0: ("Baseline", phase_0_baseline),
        1: ("BD/WB Sweep", phase_1_bd_wb_sweep),
        2: ("Top-P Sweep", phase_2_topp_sweep),
        3: ("Feature Validation", phase_3_features),
        4: ("Anti-LM Validation", phase_4_anti_lm),
        5: ("Best Combined", phase_5_combined),
        6: ("Final Refinement", phase_6_refinement),
    }

    if args.phase:
        selected = [int(p.strip()) for p in args.phase.split(",")]
    else:
        selected = list(phases.keys())

    total_t0 = time.time()
    for p in selected:
        if p not in phases:
            print(f"Unknown phase {p}, skipping")
            continue
        name, func = phases[p]
        print(f"\n{'#'*70}")
        print(f"# Phase {p}: {name}")
        print(f"# Sentences: {n_sent}")
        print(f"{'#'*70}", flush=True)
        func(args.model, n_sent, RESULTS_DIR)

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*70}")
    print(f"Translation phases completed in {total_elapsed / 60:.1f} minutes")
    print(f"Hypotheses saved to {HYPO_DIR}/")
    print(f"Run: python3 scripts/run_iteration27_experiments.py --score-xcomet")
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
