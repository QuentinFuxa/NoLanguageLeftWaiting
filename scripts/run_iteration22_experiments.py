#!/usr/bin/env python3
"""Iteration 22 GPU experiments -- YAAL fix validation + final competition tuning.

Key changes in iteration 22:
    - batch_first_emission_time: CU delays now use FIRST word in batch, not last
      This should reduce measured YAAL by ~(wb-1) * word_interval
    - avg_logprob tracking: per-step generation confidence for diagnostics
    - Competition is in ~8 days (April 1-15, 2026)

Phases:
    1. YAAL fix validation: compare old vs new YAAL on all 4 directions (20 sent)
    2. Source-aware batching: test function-word-aware batching (100 sent)
    3. Perplexity adaptive border: test bd adjustment from generation confidence (100 sent)
    4. Adaptive top_p: confirm latency reduction for competition (100 sent)
    5. Competition format: generate OmniSTEval JSONL for all directions (100 sent)
    6. Longform E2E: full recording with gold transcript -> OmniSTEval
    7. Final sweep: combined best features (100 sent)

Usage:
    # Full run on A40:
    python scripts/run_iteration22_experiments.py --model /home/fuxa/HY-MT1.5-7B.Q8_0.gguf

    # Specific phase:
    python scripts/run_iteration22_experiments.py --model /path/to/model.gguf --phase 2

    # Quick mode (20 sentences instead of 100):
    python scripts/run_iteration22_experiments.py --model /path/to/model.gguf --quick

    # With gold transcript (phase 6):
    python scripts/run_iteration22_experiments.py --model /path/to/model.gguf --phase 6 \
        --gold-jsonl /path/to/gold.jsonl
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RESULTS_DIR = "results/iteration22"
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
    print(result.stdout)
    if result.stderr:
        for line in result.stderr.split("\n"):
            if line and not any(x in line for x in ["Metal", "ggml", "llama_"]):
                print(f"  STDERR: {line}", file=sys.stderr)
    print(f"[{label}] Completed in {elapsed:.1f}s (exit code {result.returncode})")
    return result


def run_python(script, label="", timeout=3600):
    """Run an inline Python script."""
    cmd = [sys.executable, "-c", script]
    print(f"\n{'='*70}")
    print(f"[{label}] Running Python script")
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
        print(f"  STDERR: {result.stderr[:500]}", file=sys.stderr)
    print(f"[{label}] Completed in {elapsed:.1f}s")
    return result


def make_bench_args(model, direction, n_sent, cfg, extra_flags=None):
    """Build bench.py command-line arguments."""
    args = [
        "--lang", direction,
        "--border-distance", str(cfg["bd"]),
        "--word-batch", str(cfg["wb"]),
        "--aggregation", "top_p",
        "--top-p-threshold", str(cfg["p"]),
        "--n-sentences", str(n_sent),
        "--comet",
        "--save",
    ]
    if extra_flags:
        args.extend(extra_flags)
    return args


def phase_1_yaal_validation(model, n_sent=20):
    """Phase 1: Validate YAAL fix with batch_first_emission_time."""
    print("\n" + "#" * 70)
    print("# PHASE 1: YAAL Fix Validation (batch_first_emission_time)")
    print("# Compare new YAAL with reference values from iteration 18")
    print("#" * 70)

    for direction in ALL_DIRECTIONS:
        cfg = OPTIMAL_CONFIGS[direction]
        ref_yaal = REFERENCE_YAAL[direction]
        ref_comet = REFERENCE_COMET[direction]

        args = make_bench_args(model, direction, n_sent, cfg)
        result = run_bench(args, label=f"YAAL-fix-{direction}")

        if result and result.returncode == 0:
            print(f"\n  Reference (iter 18): COMET={ref_comet}, YAAL={ref_yaal}")
            print(f"  Expected: YAAL should be LOWER (batch_first_emission_time fix)")
            print(f"  COMET should be UNCHANGED (same translation, just timing)")


def phase_2_source_aware_batching(model, n_sent=100):
    """Phase 2: Test source-aware word batching."""
    print("\n" + "#" * 70)
    print("# PHASE 2: Source-Aware Word Batching")
    print("# Defer translation when batch ends on function word")
    print("#" * 70)

    for direction in ALL_DIRECTIONS:
        cfg = OPTIMAL_CONFIGS[direction]

        # Baseline (no source-aware)
        args = make_bench_args(model, direction, n_sent, cfg)
        run_bench(args, label=f"baseline-{direction}")

        # Source-aware batching
        args = make_bench_args(model, direction, n_sent, cfg,
                               extra_flags=["--source-aware-batch"])
        run_bench(args, label=f"srcaware-{direction}")


def phase_3_perplexity_adaptive(model, n_sent=100):
    """Phase 3: Test perplexity-adaptive border distance."""
    print("\n" + "#" * 70)
    print("# PHASE 3: Perplexity Adaptive Border Distance")
    print("# Hibiki-inspired: confident -> tighter bd, uncertain -> wider bd")
    print("#" * 70)

    for direction in ALL_DIRECTIONS:
        cfg = OPTIMAL_CONFIGS[direction]

        # Baseline
        args = make_bench_args(model, direction, n_sent, cfg)
        run_bench(args, label=f"baseline-{direction}")

        # Perplexity adaptive with default thresholds
        args = make_bench_args(model, direction, n_sent, cfg,
                               extra_flags=["--perplexity-adaptive-bd"])
        run_bench(args, label=f"pplbd-default-{direction}")

        # Perplexity adaptive with tuned thresholds
        for low, high in [(1.5, 4.0), (2.0, 5.0), (2.5, 6.0)]:
            args = make_bench_args(model, direction, n_sent, cfg,
                                   extra_flags=[
                                       "--perplexity-adaptive-bd",
                                       "--perplexity-bd-low", str(low),
                                       "--perplexity-bd-high", str(high),
                                   ])
            run_bench(args, label=f"pplbd-{low}-{high}-{direction}")


def phase_4_adaptive_top_p(model, n_sent=100):
    """Phase 4: Confirm adaptive top_p latency reduction."""
    print("\n" + "#" * 70)
    print("# PHASE 4: Adaptive top_p Threshold")
    print("# Per-sentence threshold from source complexity")
    print("# Expected: 6-12% YAAL reduction, <0.2% COMET cost")
    print("#" * 70)

    for direction in ALL_DIRECTIONS:
        cfg = OPTIMAL_CONFIGS[direction]

        # Fixed threshold
        args = make_bench_args(model, direction, n_sent, cfg)
        run_bench(args, label=f"fixed-topp-{direction}")

        # Adaptive threshold
        args = make_bench_args(model, direction, n_sent, cfg,
                               extra_flags=["--adaptive-top-p"])
        run_bench(args, label=f"adaptive-topp-{direction}")


def phase_5_competition_output(model, n_sent=100):
    """Phase 5: Generate competition format output."""
    print("\n" + "#" * 70)
    print("# PHASE 5: Competition OmniSTEval Output")
    print("# 100 sentences, all 4 directions, OmniSTEval JSONL")
    print("#" * 70)

    os.makedirs(f"{RESULTS_DIR}/competition", exist_ok=True)

    for direction in ALL_DIRECTIONS:
        cfg = OPTIMAL_CONFIGS[direction]
        omnisteval_path = f"{RESULTS_DIR}/competition/{direction}.jsonl"

        args = make_bench_args(model, direction, n_sent, cfg,
                               extra_flags=["--omnisteval", omnisteval_path])
        run_bench(args, label=f"competition-{direction}")

        # Validate output
        if os.path.exists(omnisteval_path):
            with open(omnisteval_path) as f:
                entries = [json.loads(line) for line in f if line.strip()]
            print(f"  OmniSTEval output: {len(entries)} entries")
            for entry in entries[:2]:
                print(f"    prediction: {entry.get('prediction', '')[:80]}...")
                print(f"    delays: {len(entry.get('delays', []))} values")


def phase_6_longform_e2e(model, gold_jsonl=None):
    """Phase 6: Full longform E2E with gold transcript."""
    print("\n" + "#" * 70)
    print("# PHASE 6: Longform E2E Pipeline")
    print("# Gold transcript -> SimulStream -> OmniSTEval JSONL")
    print("#" * 70)

    if not gold_jsonl:
        # Try to find gold transcripts
        candidates = [
            "/home/fuxa/iwslt26-sst/inputs/en/acl6060.ts/gold-jsonl/2022.acl-long.117.jsonl",
            os.path.expanduser("~/Documents/repos/iwslt26-sst/inputs/en/acl6060.ts/gold-jsonl/2022.acl-long.117.jsonl"),
        ]
        for c in candidates:
            if os.path.exists(c):
                gold_jsonl = c
                break

    if not gold_jsonl:
        print("  SKIPPING: No gold transcript found.")
        print("  Provide --gold-jsonl or place transcripts at expected paths.")
        return

    print(f"  Using gold transcript: {gold_jsonl}")
    os.makedirs(f"{RESULTS_DIR}/longform", exist_ok=True)

    for direction in ["en-zh", "en-de"]:
        cfg = OPTIMAL_CONFIGS[direction]
        output_path = f"{RESULTS_DIR}/longform/{direction}_omnisteval.jsonl"
        log_path = f"{RESULTS_DIR}/longform/{direction}_emission.log"

        script = f"""
import sys
sys.path.insert(0, '.')
from nllw.simulstream import NLLWSpeechProcessor, SimulStreamConfig, process_gold_transcript_longform
import json

config = SimulStreamConfig(
    model_path='{model}',
    direction='{direction}',
    border_distance={cfg['bd']},
    word_batch={cfg['wb']},
    aggregation='top_p',
    top_p_threshold={cfg['p']},
    longform=True,
    auto_sentence_boundary=True,
    extra_backend_params={{'n_gpu_layers': 99}},
)
proc = NLLWSpeechProcessor(config)
entry = process_gold_transcript_longform(
    proc,
    '{gold_jsonl}',
    source_name='{direction}_acl6060.wav',
    output_path='{log_path}',
)
with open('{output_path}', 'w') as f:
    f.write(json.dumps(entry, ensure_ascii=False) + '\\n')
print(f"Direction: {direction}")
print(f"Prediction length: {{len(entry['prediction'])}}")
print(f"N delays: {{len(entry['delays'])}}")
print(f"N elapsed: {{len(entry['elapsed'])}}")
print(f"Source length: {{entry['source_length']}}ms")
print(f"Preview: {{entry['prediction'][:200]}}...")
proc.close()
"""
        run_python(script, label=f"longform-{direction}", timeout=1200)


def phase_7_combined_features(model, n_sent=100):
    """Phase 7: Combined best features sweep."""
    print("\n" + "#" * 70)
    print("# PHASE 7: Combined Best Features")
    print("# Test combinations of winning features")
    print("#" * 70)

    for direction in ALL_DIRECTIONS:
        cfg = OPTIMAL_CONFIGS[direction]

        # Best baseline
        args = make_bench_args(model, direction, n_sent, cfg)
        run_bench(args, label=f"baseline-{direction}")

        # Adaptive top_p + source-aware batching
        args = make_bench_args(model, direction, n_sent, cfg,
                               extra_flags=["--adaptive-top-p", "--source-aware-batch"])
        run_bench(args, label=f"adaptp+srcaware-{direction}")

        # Adaptive top_p + perplexity adaptive bd
        args = make_bench_args(model, direction, n_sent, cfg,
                               extra_flags=["--adaptive-top-p", "--perplexity-adaptive-bd"])
        run_bench(args, label=f"adaptp+pplbd-{direction}")

        # All three combined
        args = make_bench_args(model, direction, n_sent, cfg,
                               extra_flags=[
                                   "--adaptive-top-p",
                                   "--source-aware-batch",
                                   "--perplexity-adaptive-bd",
                               ])
        run_bench(args, label=f"all-combined-{direction}")


def main():
    parser = argparse.ArgumentParser(
        description="Iteration 22 GPU experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--phase", type=int, default=0,
                        help="Run specific phase (1-7, 0=all)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 20 sentences instead of 100")
    parser.add_argument("--gold-jsonl", default=None,
                        help="Gold transcript JSONL for longform E2E (phase 6)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    n_sent = 20 if args.quick else 100

    print(f"NLLW Iteration 22 GPU Experiments")
    print(f"Model: {args.model}")
    print(f"Sentences: {n_sent}")
    print(f"Results: {RESULTS_DIR}/")
    print(f"Phase: {'all' if args.phase == 0 else args.phase}")

    phases = {
        1: ("YAAL Fix Validation", lambda: phase_1_yaal_validation(args.model, n_sent)),
        2: ("Source-Aware Batching", lambda: phase_2_source_aware_batching(args.model, n_sent)),
        3: ("Perplexity Adaptive BD", lambda: phase_3_perplexity_adaptive(args.model, n_sent)),
        4: ("Adaptive top_p", lambda: phase_4_adaptive_top_p(args.model, n_sent)),
        5: ("Competition Output", lambda: phase_5_competition_output(args.model, n_sent)),
        6: ("Longform E2E", lambda: phase_6_longform_e2e(args.model, args.gold_jsonl)),
        7: ("Combined Features", lambda: phase_7_combined_features(args.model, n_sent)),
    }

    if args.phase == 0:
        for phase_id, (name, fn) in sorted(phases.items()):
            print(f"\n{'#'*70}")
            print(f"# Starting Phase {phase_id}: {name}")
            print(f"{'#'*70}")
            fn()
    elif args.phase in phases:
        name, fn = phases[args.phase]
        print(f"\n{'#'*70}")
        print(f"# Running Phase {args.phase}: {name}")
        print(f"{'#'*70}")
        fn()
    else:
        print(f"ERROR: Invalid phase {args.phase}. Valid: 1-7 or 0 for all.")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"All phases complete. Results in {RESULTS_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
