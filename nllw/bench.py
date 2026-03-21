"""Unified benchmarking CLI for SimulMT backends.

One-command entry point for running benchmarks, comparisons, and sweeps.

Usage:
    # Basic benchmark (requires AlignAtt backend)
    python -m nllw.bench --lang en-fr --model /path/to.gguf

    # Full corpus with COMET, save to registry
    python -m nllw.bench --suite corpus --lang en-fr --comet --save

    # Compare backends head-to-head
    python -m nllw.bench --compare alignatt full-sentence --lang en-fr --model /path/to.gguf

    # Parameter sweep
    python -m nllw.bench --sweep "bd=2,3,4 wb=2,3" --lang en-fr --model /path/to.gguf --comet

    # Export to OmniSTEval format (IWSLT submission)
    python -m nllw.bench --suite corpus --lang en-zh --omnisteval output.jsonl
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Any, Optional

from .eval import load_flores, evaluate_backend, EvalResult
from .corpus import get_corpus_as_pairs
from .backend_protocol import BackendConfig, create_backend

# Import backends to register them with the factory
import nllw.alignatt_backend  # noqa: F401 -- registers alignatt, full-sentence, eager
import nllw.alignatt_la_backend  # noqa: F401 -- registers alignatt-la
import nllw.baselines  # noqa: F401 -- registers wait-k, fixed-rate


def parse_sweep_spec(spec: str) -> Dict[str, List[Any]]:
    """Parse a sweep specification string.

    Format: "bd=2,3,4 wb=2,3 ctx=0,1"

    Maps short names to BackendConfig fields:
        bd -> border_distance
        wb -> word_batch
        ctx -> context_sentences
        topk -> top_k_heads
        entropy -> entropy_veto_threshold
    """
    SHORT_MAP = {
        "bd": "border_distance",
        "wb": "word_batch",
        "ctx": "context_sentences",
        "topk": "top_k_heads",
        "entropy": "entropy_veto_threshold",
        "agg": "aggregation",
        "topp": "top_p_threshold",
        "dynbd": "dynamic_border",
        "ssbd": "ssbd_beta",
        "mask": "display_mask_k",
        "forced": "la_forced_decode",
        "adaptive": "adaptive_ssbd",
        "twopass": "la_two_pass",
        "ams": "adaptive_aggregation",
        "tempnorm": "head_temp_normalize",
        "tempref": "head_temp_reference",
        "dynwb": "dynamic_word_batch",
        "infogain": "info_gain_threshold",
        "shiftk": "shift_k_threshold",
        "confirm": "border_confirm",
        "lsg": "lsg_kl_threshold",
        "lsgk": "lsg_k",
        "cmplx": "complexity_adaptive",
        "entchg": "entropy_change_threshold",
        "predstab": "prediction_stability",
        "cov": "coverage_threshold",
        "mono": "attention_monotonicity",
        "rep": "repetition_max_repeats",
        "attshift": "attention_shift",
        "fusion": "signal_fusion",
        "fthr": "fusion_threshold",
    }

    grid = {}
    for part in spec.strip().split():
        key, values_str = part.split("=", 1)
        full_key = SHORT_MAP.get(key, key)
        values = []
        for v in values_str.split(","):
            try:
                if "." in v:
                    values.append(float(v))
                else:
                    values.append(int(v))
            except ValueError:
                values.append(v)
        grid[full_key] = values

    return grid


def _build_base_config_dict(args) -> Dict[str, Any]:
    """Build a full config dict from CLI args. Used by all run modes.

    This centralizes ALL config parameters so that run_comparison() and
    run_sweep() get the same full config as run_benchmark().
    """
    parts = args.lang.split("-")
    tgt_lang = parts[1]
    return {
        "backend_type": args.backend,
        "model_path": args.model or "",
        "heads_path": args.heads or "",
        "n_gpu_layers": args.n_gpu_layers,
        "direction": args.lang,
        "border_distance": args.border_distance,
        "word_batch": args.word_batch,
        "context_sentences": args.context_sentences,
        "target_lang": tgt_lang,
        "n_ctx": args.n_ctx,
        "wait_k": args.wait_k,
        "entropy_veto_threshold": args.entropy_threshold,
        "aggregation": args.aggregation,
        "top_p_threshold": args.top_p_threshold,
        "dynamic_border": args.dynamic_border,
        "ssbd_beta": args.ssbd_beta,
        "la_forced_decode": args.forced_decode,
        "adaptive_ssbd": args.adaptive_ssbd,
        "la_two_pass": args.two_pass,
        "adaptive_aggregation": args.adaptive_agg,
        "head_temp_normalize": args.head_temp_norm,
        "head_temp_reference": args.head_temp_ref,
        "dynamic_word_batch": args.dynamic_wb,
        "info_gain_threshold": args.info_gain,
        "shift_k_threshold": args.shift_k,
        "border_confirm": args.border_confirm,
        "lsg_kl_threshold": args.lsg_kl,
        "lsg_k": args.lsg_k,
        "complexity_adaptive": args.complexity_adaptive,
        "entropy_change_threshold": args.entropy_change,
        "prediction_stability": args.prediction_stability,
        "coverage_threshold": args.coverage_threshold,
        "attention_monotonicity": args.attention_monotonicity,
        "repetition_max_repeats": args.repetition_halt,
        "attention_shift": args.attention_shift,
        "signal_fusion": args.signal_fusion,
        "fusion_threshold": args.fusion_threshold,
    }


def run_benchmark(args):
    """Run a single benchmark."""
    # Load corpus
    parts = args.lang.split("-")
    src_lang, tgt_lang = parts[0], parts[1]

    if args.suite == "flores":
        corpus = load_flores(src_lang, tgt_lang, n=args.n)
    elif args.suite == "flores_mini":
        corpus = load_flores(src_lang, tgt_lang, n=min(args.n, 20))
    elif args.suite == "corpus":
        corpus = get_corpus_as_pairs(args.lang, n=args.n)
        if not corpus:
            print(f"No built-in corpus for {args.lang}, falling back to FLORES", file=sys.stderr)
            corpus = load_flores(src_lang, tgt_lang, n=args.n)
    else:
        corpus = load_flores(src_lang, tgt_lang, n=args.n)

    print(f"Benchmark: {args.lang} | {len(corpus)} sentences | suite={args.suite}", file=sys.stderr)

    config = BackendConfig.from_dict(_build_base_config_dict(args))

    backend = create_backend(config)

    # Attach trace collector if requested
    trace_collector = None
    if getattr(args, 'collect_traces', None):
        from .calibrate import TraceCollector
        trace_collector = TraceCollector()
        backend.set_trace_collector(trace_collector)

    try:
        result = evaluate_backend(
            backend, corpus,
            compute_comet_score=args.comet,
            compute_xcomet_score=args.xcomet,
            trace_collector=trace_collector,
        )
    finally:
        backend.close()

    # Save traces if collected
    if trace_collector and args.collect_traces:
        from .calibrate import save_traces
        traces = trace_collector.get_traces()
        save_traces(traces, args.collect_traces)
        print(f"Saved {len(traces)} signal traces to {args.collect_traces}",
              file=sys.stderr)

    return result


def run_comparison(args):
    """Compare multiple backends on the same corpus."""
    parts = args.lang.split("-")
    src_lang, tgt_lang = parts[0], parts[1]
    corpus = load_flores(src_lang, tgt_lang, n=args.n)

    base = _build_base_config_dict(args)

    results = []
    for backend_type in args.compare:
        print(f"\n=== Backend: {backend_type} ===", file=sys.stderr)
        cfg = dict(base, backend_type=backend_type)
        config = BackendConfig.from_dict(cfg)
        backend = create_backend(config)
        try:
            result = evaluate_backend(
                backend, corpus,
                compute_comet_score=args.comet,
            )
            results.append(result)
        finally:
            backend.close()

    # Print comparison table
    print("\n=== Comparison ===")
    header = f"{'Backend':<20} {'BLEU':>6} {'COMET':>7} {'AL':>6} {'YAAL':>6} {'AP':>6} {'ms/sent':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        comet_str = f"{r.comet:.3f}" if r.comet else "  -  "
        bleu_str = f"{r.bleu:.1f}" if r.bleu else "  -  "
        print(
            f"{r.backend_type:<20} {bleu_str:>6} {comet_str:>7} "
            f"{r.avg_al:>6.2f} {r.avg_yaal:>6.2f} {r.avg_ap:>6.3f} "
            f"{r.avg_time_per_sentence_ms:>8.0f}"
        )

    return results


def run_sweep(args):
    """Run parameter sweep."""
    parts = args.lang.split("-")
    src_lang, tgt_lang = parts[0], parts[1]
    corpus = load_flores(src_lang, tgt_lang, n=args.n)

    param_grid = parse_sweep_spec(args.sweep)
    print(f"Sweep: {param_grid}", file=sys.stderr)

    base_config = _build_base_config_dict(args)

    from .eval import parameter_sweep

    def factory(config_dict):
        return create_backend(BackendConfig.from_dict(config_dict))

    results = parameter_sweep(
        factory, corpus, param_grid, base_config,
        compute_comet_score=args.comet,
    )

    # Print results table
    param_keys = list(param_grid.keys())
    print(f"\n=== Sweep Results ({len(results)} configs) ===")
    header = "  ".join(f"{k:>6}" for k in param_keys) + f" {'BLEU':>6} {'COMET':>7} {'AL':>6} {'YAAL':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        params = "  ".join(f"{r.config.get(k, '-'):>6}" for k in param_keys)
        comet_str = f"{r.comet:.3f}" if r.comet else "  -  "
        bleu_str = f"{r.bleu:.1f}" if r.bleu else "  -  "
        print(f"{params} {bleu_str:>6} {comet_str:>7} {r.avg_al:>6.2f} {r.avg_yaal:>6.2f}")

    return results


def _run_calibration(args):
    """Run fusion weight calibration pipeline.

    Two modes:
    1. --calibrate: Use synthetic traces (fast, no GPU needed, for testing)
    2. --calibrate-traces FILE: Use real traces collected from GPU runs
    """
    from .calibrate import (
        generate_synthetic_traces,
        load_traces,
        run_calibration,
        export_weights,
        print_calibration_report,
        analyze_signal_importance,
    )

    # Determine directions
    directions = [d.strip() for d in args.lang.split(",")]

    if args.calibrate_traces:
        # Load real traces
        traces = load_traces(args.calibrate_traces)
        print(f"Loaded {len(traces)} traces from {args.calibrate_traces}",
              file=sys.stderr)
    else:
        # Generate synthetic traces
        print("Generating synthetic traces for calibration demo...",
              file=sys.stderr)
        traces = []
        for d in directions:
            t = generate_synthetic_traces(
                n_sentences=args.n, direction=d, seed=42
            )
            traces.extend(t)
        print(f"  Generated {len(traces)} synthetic traces", file=sys.stderr)

    # Analyze signal importance
    print("\n--- Signal Importance Analysis ---", file=sys.stderr)
    for d in directions:
        importance = analyze_signal_importance(
            traces, d, args.border_distance
        )
        if importance:
            print(f"\n  {d}:", file=sys.stderr)
            for signal, stats in importance.items():
                print(f"    {signal:20s}: "
                      f"discriminative={stats['discriminative_power']:+.3f} "
                      f"corr={stats['correlation']:+.3f}",
                      file=sys.stderr)

    # Run calibration
    results = run_calibration(
        traces,
        directions=directions,
        border_distance=args.border_distance,
        method=args.calibrate_method,
    )
    print_calibration_report(results)

    # Export
    output_path = args.calibrate_output or args.output
    if output_path:
        export_weights(results, output_path)
        print(f"\nCalibrated weights saved to {output_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="NLLW SimulMT Benchmarking CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode
    parser.add_argument("--compare", nargs="+", help="Compare multiple backends")
    parser.add_argument("--sweep", help='Parameter sweep spec (e.g. "bd=2,3,4 wb=2,3")')

    # Data
    parser.add_argument("--lang", default="en-fr", help="Language direction")
    parser.add_argument("--suite", default="flores", choices=["flores", "flores_mini", "corpus"])
    parser.add_argument("-n", type=int, default=50, help="Number of sentences")

    # Backend
    parser.add_argument("--backend", default="alignatt")
    parser.add_argument("--model", help="Path to GGUF model")
    parser.add_argument("--heads", help="Path to head config JSON")
    parser.add_argument("--n-gpu-layers", type=int, default=0,
                        help="GPU layers to offload (0=CPU, 99=all)")

    # Parameters
    parser.add_argument("--border-distance", type=int, default=3)
    parser.add_argument("--word-batch", type=int, default=3)
    parser.add_argument("--context-sentences", type=int, default=0)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--wait-k", type=int, default=5, help="Wait-k words (for wait-k backend)")
    parser.add_argument("--entropy-threshold", type=float, default=None,
                        help="Entropy veto threshold (None=disabled)")
    parser.add_argument("--aggregation", default="ts_vote",
                        choices=["ts_vote", "softmax_mean", "entropy_weighted",
                                 "consensus", "geomean", "top_p", "top_p_weighted",
                                 "ensemble", "gaussian_kernel",
                                 "gaussian_kernel_continuous", "cumulative"],
                        help="Attention aggregation method for border detection")
    parser.add_argument("--top-p-threshold", type=float, default=0.8,
                        help="Cumulative mass threshold for top_p aggregation (0.5-0.95)")
    parser.add_argument("--dynamic-border", action="store_true",
                        help="Enable entropy-based dynamic border distance")
    parser.add_argument("--ssbd-beta", type=float, default=None,
                        help="SSBD bias (0.0=pure speculative, 0.2=recommended, None=disabled)")
    parser.add_argument("--forced-decode", action="store_true",
                        help="Enable LA forced decoding of committed prefix (CUNI approach)")
    parser.add_argument("--adaptive-ssbd", action="store_true",
                        help="Enable entropy-based adaptive SSBD beta (per-token)")
    parser.add_argument("--two-pass", action="store_true",
                        help="Enable LA two-pass catch-up (2x compute for stability)")
    parser.add_argument("--adaptive-agg", action="store_true",
                        help="Enable Adaptive Multi-Strategy (AMS) aggregation selection")
    parser.add_argument("--head-temp-norm", action="store_true",
                        help="Enable per-head temperature normalization")
    parser.add_argument("--head-temp-ref", type=float, default=1.5,
                        help="Reference entropy for head temperature normalization")
    parser.add_argument("--dynamic-wb", action="store_true",
                        help="Enable dynamic word_batch (adjust by source length)")
    parser.add_argument("--info-gain", type=float, default=None,
                        help="Info gain threshold for border modulation (0.3=recommended, None=disabled)")
    parser.add_argument("--shift-k", type=float, default=None,
                        help="Shift-k border mass threshold (0.4=recommended, None=disabled)")
    parser.add_argument("--border-confirm", type=int, default=1,
                        help="Require N consecutive border hits to stop (1=disabled, 2=recommended)")
    parser.add_argument("--lsg-kl", type=float, default=None,
                        help="LSG logit KL threshold (7.0=recommended, None=disabled)")
    parser.add_argument("--lsg-k", type=int, default=3,
                        help="LSG: number of source tokens to remove for probe (default: 3)")
    parser.add_argument("--complexity-adaptive", action="store_true",
                        help="Enable per-sentence complexity-adaptive bd/wb/gen_cap")
    parser.add_argument("--entropy-change", type=float, default=None,
                        help="REINA entropy change threshold (-0.5=recommended, None=disabled)")
    parser.add_argument("--prediction-stability", action="store_true",
                        help="Enable cross-step prediction stability border modulation")
    parser.add_argument("--coverage-threshold", type=float, default=None,
                        help="Source coverage guard threshold (0.3=recommended, None=disabled)")
    parser.add_argument("--attention-monotonicity", action="store_true",
                        help="Enable attention monotonicity-based border distance adjustment")
    parser.add_argument("--repetition-halt", type=int, default=None,
                        help="N-gram repetition halt threshold (2=recommended, None=disabled)")
    parser.add_argument("--attention-shift", action="store_true",
                        help="Enable cross-step attention shift tracking")
    parser.add_argument("--signal-fusion", action="store_true",
                        help="Enable weighted signal fusion (replaces boolean cascade)")
    parser.add_argument("--fusion-threshold", type=float, default=0.0,
                        help="Fusion decision threshold (0.0=balanced)")

    # Metrics
    parser.add_argument("--comet", action="store_true")
    parser.add_argument("--xcomet", action="store_true")

    # Output
    parser.add_argument("--save", action="store_true", help="Save results to registry")
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("--save-hypotheses", help="Save sources/hypotheses/references to JSON for offline XCOMET scoring")
    parser.add_argument("--omnisteval", help="Export to OmniSTEval JSONL format")

    # Calibration
    parser.add_argument("--calibrate", action="store_true",
                        help="Run fusion weight calibration on synthetic traces "
                             "(use --calibrate-traces with real traces for GPU calibration)")
    parser.add_argument("--calibrate-traces", type=str, default=None,
                        help="Path to collected signal traces JSON for calibration")
    parser.add_argument("--calibrate-output", type=str, default=None,
                        help="Output path for calibrated fusion weights JSON")
    parser.add_argument("--calibrate-method", type=str, default="alignment",
                        choices=["alignment", "quality"],
                        help="Labeling method for calibration (default: alignment)")
    parser.add_argument("--collect-traces", type=str, default=None,
                        help="Collect signal traces during benchmark and save to JSON "
                             "(for later calibration with --calibrate-traces)")

    args = parser.parse_args()

    # Calibration mode
    if args.calibrate or args.calibrate_traces:
        _run_calibration(args)
        return

    # Dispatch
    if args.compare:
        results = run_comparison(args)
    elif args.sweep:
        results = run_sweep(args)
    else:
        result = run_benchmark(args)
        results = [result]

    # Save
    if args.output or args.save:
        output_path = args.output or f"results_{args.lang}_{int(time.time())}.json"
        with open(output_path, "w") as f:
            json.dump(
                [r.to_dict() for r in results] if len(results) > 1 else results[0].to_dict(),
                f, indent=2, ensure_ascii=False,
            )
        print(f"Saved to {output_path}", file=sys.stderr)

    # Save hypotheses for offline XCOMET-XL scoring
    if args.save_hypotheses and len(results) >= 1:
        for r in results:
            if r.per_sentence:
                hypo_data = {
                    "direction": r.direction,
                    "backend_type": r.backend_type,
                    "n_sentences": r.n_sentences,
                    "comet": r.comet,
                    "bleu": r.bleu,
                    "sources": [ps["source"] for ps in r.per_sentence],
                    "hypotheses": [ps["hypothesis"] for ps in r.per_sentence],
                    "references": [ps["reference"] for ps in r.per_sentence],
                    "config": r.config,
                }
                with open(args.save_hypotheses, "w") as f:
                    json.dump(hypo_data, f, indent=2, ensure_ascii=False)
                print(f"Saved hypotheses to {args.save_hypotheses}", file=sys.stderr)
                break

    # OmniSTEval export
    if args.omnisteval and len(results) == 1 and results[0].per_sentence:
        from .omnisteval import eval_result_to_simuleval, write_simuleval_jsonl_file
        eval_dict = {"per_sentence": results[0].per_sentence}
        entries = eval_result_to_simuleval(
            eval_dict, source_prefix=args.lang
        )
        write_simuleval_jsonl_file(entries, args.omnisteval)


if __name__ == "__main__":
    main()
