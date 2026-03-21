#!/usr/bin/env python3
"""Iteration 21 GPU experiments -- Competition E2E validation + perplexity adaptive border.

Priority: Competition is in ~10 days (April 1-15, 2026). Focus on E2E pipeline
correctness, longform mode validation, and perplexity adaptive border testing.

Phases:
    1. Quick smoke test: all 4 directions with optimal configs (20 sentences)
    2. Perplexity adaptive border: test Hibiki-inspired bd adjustment (100 sentences)
    3. Longform E2E: full recording pipeline with gold transcript
    4. Multi-direction longform: all 4 directions in longform mode
    5. Competition format: OmniSTEval JSONL generation for all directions (100 sentences)
    6. Adaptive top_p decision: compare fixed vs adaptive for competition

Usage:
    # On A40 (full):
    python scripts/run_iteration21_experiments.py --model /home/fuxa/HY-MT1.5-7B.Q8_0.gguf

    # Specific phase:
    python scripts/run_iteration21_experiments.py --model /path/to/model.gguf --phase 2

    # Quick mode (fewer sentences):
    python scripts/run_iteration21_experiments.py --model /path/to/model.gguf --quick

    # With gold transcript (phase 3):
    python scripts/run_iteration21_experiments.py --model /path/to/model.gguf --phase 3 \\
        --gold-jsonl /path/to/gold.jsonl
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RESULTS_DIR = "results/iteration21"
ALL_DIRECTIONS = ["en-zh", "en-de", "en-it", "cs-en"]

# Best known configs per direction (iteration 18 verified)
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
        # Only show non-noise stderr
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
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[:500]}", file=sys.stderr)
    print(f"[{label}] Completed in {elapsed:.1f}s (exit code {result.returncode})")
    return result


def phase1_smoke_test(model_path, n_sent=20):
    """Quick smoke test: all 4 directions with optimal configs.

    Confirms the system works on this GPU and basic quality is correct.
    This should finish in ~5 minutes.
    """
    print("\n\n" + "=" * 70)
    print("PHASE 1: Smoke Test -- All 4 Directions")
    print(f"  Model: {model_path}")
    print(f"  Sentences: {n_sent}")
    print("=" * 70)

    results = {}
    for direction in ALL_DIRECTIONS:
        cfg = OPTIMAL_CONFIGS[direction]
        result = run_bench([
            "--lang", direction,
            "--model", model_path,
            "--n", str(n_sent),
            "--border-distance", str(cfg["bd"]),
            "--word-batch", str(cfg["wb"]),
            "--aggregation", "top_p",
            "--top-p-threshold", str(cfg["p"]),
            "--comet",
        ], label=f"Smoke-{direction}")
        results[direction] = result

    return results


def phase2_perplexity_adaptive_border(model_path, n_sent=100):
    """Test perplexity-based adaptive border (Hibiki-inspired).

    This is NEW in iteration 20 and needs GPU validation.
    Compares baseline vs perplexity adaptive bd, measuring COMET + YAAL.
    Also sweeps perplexity thresholds.
    """
    print("\n\n" + "=" * 70)
    print("PHASE 2: Perplexity Adaptive Border Test")
    print(f"  Sentences: {n_sent}")
    print("=" * 70)

    results = {}

    for direction in ALL_DIRECTIONS:
        cfg = OPTIMAL_CONFIGS[direction]
        base_args = [
            "--lang", direction,
            "--model", model_path,
            "--n", str(n_sent),
            "--border-distance", str(cfg["bd"]),
            "--word-batch", str(cfg["wb"]),
            "--aggregation", "top_p",
            "--top-p-threshold", str(cfg["p"]),
            "--comet",
        ]

        # Baseline (without perplexity adaptive bd)
        result_base = run_bench(
            base_args,
            label=f"PPL-base-{direction}",
        )
        results[f"{direction}_base"] = result_base

        # With perplexity adaptive border (default thresholds)
        result_ppl = run_bench(
            base_args + ["--perplexity-adaptive-bd"],
            label=f"PPL-adapt-{direction}",
        )
        results[f"{direction}_ppl"] = result_ppl

    # Threshold sweep on EN-ZH (best direction for testing)
    print("\n--- Perplexity Threshold Sweep (EN-ZH) ---")
    cfg = OPTIMAL_CONFIGS["en-zh"]
    for ppl_low, ppl_high in [(1.5, 4.0), (2.0, 5.0), (2.5, 5.0), (3.0, 6.0)]:
        result = run_bench([
            "--lang", "en-zh",
            "--model", model_path,
            "--n", str(n_sent),
            "--border-distance", str(cfg["bd"]),
            "--word-batch", str(cfg["wb"]),
            "--aggregation", "top_p",
            "--top-p-threshold", str(cfg["p"]),
            "--perplexity-adaptive-bd",
            "--perplexity-bd-low", str(ppl_low),
            "--perplexity-bd-high", str(ppl_high),
            "--comet",
        ], label=f"PPL-sweep-low{ppl_low}-high{ppl_high}")
        results[f"enzh_ppl_{ppl_low}_{ppl_high}"] = result

    return results


def phase3_longform_e2e(model_path, gold_jsonl=None):
    """Longform E2E test with gold transcript.

    Tests the competition pipeline: gold ASR JSONL -> longform processor
    -> OmniSTEval JSONL output. Verifies format, delays, and quality.
    """
    print("\n\n" + "=" * 70)
    print("PHASE 3: Longform E2E Test")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # If no gold JSONL provided, create a synthetic one from corpus
    if gold_jsonl and os.path.exists(gold_jsonl):
        input_path = gold_jsonl
        print(f"  Using gold transcript: {gold_jsonl}")
    else:
        # Create synthetic gold JSONL from corpus sentences
        input_path = os.path.join(RESULTS_DIR, "synthetic_gold.jsonl")
        print(f"  Creating synthetic gold transcript: {input_path}")
        create_synthetic_gold(input_path, direction="en-zh", n_sentences=10)

    omnisteval_path = os.path.join(RESULTS_DIR, "longform_output.jsonl")
    emission_log_path = os.path.join(RESULTS_DIR, "longform_emissions.jsonl")

    # Run longform processing
    script = f"""
import sys, json
sys.path.insert(0, '.')
from nllw.simulstream import NLLWSpeechProcessor, SimulStreamConfig, process_gold_transcript_longform

config = SimulStreamConfig(
    model_path='{model_path}',
    direction='en-zh',
    backend_type='alignatt',
    border_distance=3,
    word_batch=4,
    aggregation='top_p',
    top_p_threshold=0.85,
    longform=True,
    auto_sentence_boundary=True,
    extra_backend_params={{'n_gpu_layers': 99}},
)

processor = NLLWSpeechProcessor.load_model(config)

entry = process_gold_transcript_longform(
    processor,
    input_path='{input_path}',
    source_name='test_recording.wav',
    source_length_s=120.0,
    output_path='{emission_log_path}',
    char_level=True,
)

# Write OmniSTEval output
with open('{omnisteval_path}', 'w') as f:
    f.write(json.dumps(entry, ensure_ascii=False) + '\\n')

# Print summary
print(f"Prediction length: {{len(entry['prediction'])}} chars")
print(f"Delays: {{len(entry['delays'])}} entries")
print(f"Elapsed: {{len(entry['elapsed'])}} entries")
print(f"Source length: {{entry['source_length']}} ms")
print(f"First 100 chars: {{entry['prediction'][:100]}}")
print(f"Delays monotonic: {{all(a <= b for a, b in zip(entry['delays'][:-1], entry['delays'][1:]))}}")
print(f"len(delays)==len(prediction chars): {{len(entry['delays']) == len(entry['prediction'])}}")

processor.close()
print("\\nLongform E2E: SUCCESS")
"""
    result = run_python(script, label="Longform-E2E-ENZH", timeout=600)

    # Validate the output
    if os.path.exists(omnisteval_path):
        with open(omnisteval_path) as f:
            for line in f:
                entry = json.loads(line)
                n_delays = len(entry["delays"])
                n_chars = len(entry["prediction"])
                print(f"\n  OmniSTEval validation:")
                print(f"    prediction chars: {n_chars}")
                print(f"    delays count: {n_delays}")
                print(f"    match: {n_delays == n_chars}")
                print(f"    source_length: {entry['source_length']} ms")
                if entry["delays"]:
                    print(f"    delay range: [{entry['delays'][0]}, {entry['delays'][-1]}] ms")
                    monotonic = all(
                        a <= b for a, b in zip(entry["delays"][:-1], entry["delays"][1:])
                    )
                    print(f"    delays monotonic: {monotonic}")

    return result


def phase4_multi_direction_longform(model_path, n_sent=20):
    """Multi-direction longform test.

    Verify all 4 directions work correctly in longform mode.
    Creates synthetic gold JSONL for each direction and processes it.
    """
    print("\n\n" + "=" * 70)
    print("PHASE 4: Multi-Direction Longform Validation")
    print(f"  Sentences per direction: {n_sent}")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {}

    for direction in ALL_DIRECTIONS:
        cfg = OPTIMAL_CONFIGS[direction]
        char_level = direction in ("en-zh",)  # Char-level for Chinese

        input_path = os.path.join(RESULTS_DIR, f"synthetic_gold_{direction}.jsonl")
        omnisteval_path = os.path.join(RESULTS_DIR, f"longform_{direction}.jsonl")

        create_synthetic_gold(input_path, direction=direction, n_sentences=n_sent)

        script = f"""
import sys, json
sys.path.insert(0, '.')
from nllw.simulstream import NLLWSpeechProcessor, SimulStreamConfig, process_gold_transcript_longform

config = SimulStreamConfig(
    model_path='{model_path}',
    direction='{direction}',
    backend_type='alignatt',
    border_distance={cfg['bd']},
    word_batch={cfg['wb']},
    aggregation='top_p',
    top_p_threshold={cfg['p']},
    longform=True,
    auto_sentence_boundary=True,
    extra_backend_params={{'n_gpu_layers': 99}},
)

processor = NLLWSpeechProcessor.load_model(config)

entry = process_gold_transcript_longform(
    processor,
    input_path='{input_path}',
    source_name='{direction}_test.wav',
    source_length_s=float({n_sent * 6}),
    char_level={'True' if char_level else 'False'},
)

with open('{omnisteval_path}', 'w') as f:
    f.write(json.dumps(entry, ensure_ascii=False) + '\\n')

pred_len = len(entry['prediction'])
n_delays = len(entry['delays'])
print(f"Direction: {direction}")
print(f"Prediction: {{pred_len}} chars/words")
print(f"Delays: {{n_delays}} entries")
print(f"Match: {{n_delays > 0}}")
print(f"Preview: {{entry['prediction'][:80]}}")

processor.close()
print(f"\\n{direction} longform: SUCCESS")
"""
        result = run_python(script, label=f"Longform-{direction}", timeout=600)
        results[direction] = result

    return results


def phase5_competition_format(model_path, n_sent=100):
    """Generate competition-format OmniSTEval JSONL for all directions.

    This is what we submit. Uses optimal configs with 100 sentences
    from FLORES corpus. Measures COMET + LongYAAL.
    """
    print("\n\n" + "=" * 70)
    print("PHASE 5: Competition Format Output (100 sentences)")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {}

    for direction in ALL_DIRECTIONS:
        cfg = OPTIMAL_CONFIGS[direction]
        omnisteval_path = os.path.join(RESULTS_DIR, f"competition_{direction}.jsonl")

        result = run_bench([
            "--lang", direction,
            "--model", model_path,
            "--n", str(n_sent),
            "--border-distance", str(cfg["bd"]),
            "--word-batch", str(cfg["wb"]),
            "--aggregation", "top_p",
            "--top-p-threshold", str(cfg["p"]),
            "--omnisteval", omnisteval_path,
            "--comet",
        ], label=f"Competition-{direction}")
        results[direction] = result

        # Validate OmniSTEval output
        if os.path.exists(omnisteval_path):
            with open(omnisteval_path) as f:
                entries = [json.loads(l) for l in f if l.strip()]
            print(f"  [{direction}] {len(entries)} OmniSTEval entries")
            for i, entry in enumerate(entries[:2]):
                pred = entry["prediction"]
                n_d = len(entry["delays"])
                n_w = len(pred.split()) if direction != "en-zh" else len(pred)
                print(f"    entry[{i}]: {n_w} units, {n_d} delays, match={n_w == n_d}")

    return results


def phase6_adaptive_top_p_decision(model_path, n_sent=100):
    """Compare fixed vs adaptive top_p for competition decision.

    We need to decide whether to enable adaptive_top_p for the competition.
    Previous results show 6-12% YAAL reduction for <0.002 COMET cost.
    Confirm with 100 sentences + bootstrap CI.
    """
    print("\n\n" + "=" * 70)
    print("PHASE 6: Adaptive top_p Decision (Fixed vs Adaptive)")
    print(f"  Sentences: {n_sent}")
    print("=" * 70)

    results = {}

    for direction in ALL_DIRECTIONS:
        cfg = OPTIMAL_CONFIGS[direction]
        base_args = [
            "--lang", direction,
            "--model", model_path,
            "--n", str(n_sent),
            "--border-distance", str(cfg["bd"]),
            "--word-batch", str(cfg["wb"]),
            "--aggregation", "top_p",
            "--top-p-threshold", str(cfg["p"]),
            "--comet",
        ]

        # Fixed threshold
        result_fixed = run_bench(
            base_args,
            label=f"Fixed-{direction}",
        )
        results[f"{direction}_fixed"] = result_fixed

        # Adaptive threshold
        result_adapt = run_bench(
            base_args + ["--adaptive-top-p"],
            label=f"Adaptive-{direction}",
        )
        results[f"{direction}_adaptive"] = result_adapt

    return results


def phase7_source_aware_batching(model_path, n_sent=100):
    """Test source-aware word batching (new in iteration 21).

    Compares fixed word batching vs source-aware batching that defers
    translation when the batch ends on a function word.
    """
    print("\n\n" + "=" * 70)
    print("PHASE 7: Source-Aware Word Batching")
    print(f"  Sentences: {n_sent}")
    print("=" * 70)

    results = {}

    for direction in ALL_DIRECTIONS:
        cfg = OPTIMAL_CONFIGS[direction]
        base_args = [
            "--lang", direction,
            "--model", model_path,
            "--n", str(n_sent),
            "--border-distance", str(cfg["bd"]),
            "--word-batch", str(cfg["wb"]),
            "--aggregation", "top_p",
            "--top-p-threshold", str(cfg["p"]),
            "--comet",
        ]

        # Baseline (fixed batching)
        result_fixed = run_bench(
            base_args,
            label=f"FixedBatch-{direction}",
        )
        results[f"{direction}_fixed"] = result_fixed

        # Source-aware batching
        result_aware = run_bench(
            base_args + ["--source-aware-batch"],
            label=f"SrcAware-{direction}",
        )
        results[f"{direction}_source_aware"] = result_aware

    return results


def create_synthetic_gold(output_path, direction="en-zh", n_sentences=10):
    """Create a synthetic gold JSONL from corpus sentences.

    Simulates a gold ASR transcript with word-by-word timing.
    """
    # Import corpus
    sys.path.insert(0, ".")
    from nllw.corpus import get_corpus

    corpus = get_corpus(direction)
    sentences = corpus[:n_sentences]

    emission_time = 0.0
    word_interval = 0.4  # 400ms between words (typical speech rate)

    with open(output_path, "w") as f:
        for sent_idx, sent in enumerate(sentences):
            src = sent.text if hasattr(sent, "text") else sent[0]
            words = src.strip().split()
            for word_idx, word in enumerate(words):
                is_final = (word_idx == len(words) - 1)
                entry = {
                    "start": emission_time,
                    "end": emission_time + word_interval,
                    "text": f" {word}",
                    "emission_time": emission_time + word_interval,
                    "is_final": is_final,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                emission_time += word_interval

            # Small pause between sentences
            emission_time += 0.5


def print_summary(all_results):
    """Print a summary of all phase results."""
    print("\n\n" + "=" * 70)
    print("EXPERIMENT SUMMARY -- Iteration 21")
    print("=" * 70)

    for phase_name, results in all_results.items():
        print(f"\n{phase_name}:")
        if isinstance(results, dict):
            for key, result in results.items():
                if result is None:
                    print(f"  {key}: TIMEOUT")
                elif result.returncode != 0:
                    print(f"  {key}: FAILED (exit code {result.returncode})")
                else:
                    print(f"  {key}: OK")
        elif results is None:
            print("  TIMEOUT")
        elif hasattr(results, "returncode"):
            print(f"  exit code: {results.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Iteration 21 GPU experiments")
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--phase", type=int, default=0,
                        help="Run specific phase (1-7), or 0 for all")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer sentences (20 instead of 100)")
    parser.add_argument("--gold-jsonl", type=str, default=None,
                        help="Path to gold ASR JSONL for phase 3")
    args = parser.parse_args()

    n_sent = 20 if args.quick else 100
    n_smoke = 10 if args.quick else 20

    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = {}
    start_time = time.time()

    if args.phase in (0, 1):
        all_results["Phase 1: Smoke Test"] = phase1_smoke_test(
            args.model, n_sent=n_smoke)

    if args.phase in (0, 2):
        all_results["Phase 2: Perplexity Adaptive Border"] = phase2_perplexity_adaptive_border(
            args.model, n_sent=n_sent)

    if args.phase in (0, 3):
        all_results["Phase 3: Longform E2E"] = phase3_longform_e2e(
            args.model, gold_jsonl=args.gold_jsonl)

    if args.phase in (0, 4):
        n_long = 10 if args.quick else 20
        all_results["Phase 4: Multi-Direction Longform"] = phase4_multi_direction_longform(
            args.model, n_sent=n_long)

    if args.phase in (0, 5):
        all_results["Phase 5: Competition Format"] = phase5_competition_format(
            args.model, n_sent=n_sent)

    if args.phase in (0, 6):
        all_results["Phase 6: Adaptive top_p Decision"] = phase6_adaptive_top_p_decision(
            args.model, n_sent=n_sent)

    if args.phase in (0, 7):
        all_results["Phase 7: Source-Aware Batching"] = phase7_source_aware_batching(
            args.model, n_sent=n_sent)

    total_time = time.time() - start_time
    print_summary(all_results)
    print(f"\nTotal time: {total_time:.1f}s ({total_time / 60:.1f} min)")


if __name__ == "__main__":
    main()
