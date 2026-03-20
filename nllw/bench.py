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
        "direction": args.lang,
        "border_distance": args.border_distance,
        "word_batch": args.word_batch,
        "context_sentences": args.context_sentences,
        "target_lang": tgt_lang,
        "n_ctx": args.n_ctx,
        "wait_k": args.wait_k,
        "entropy_veto_threshold": args.entropy_threshold,
        "aggregation": args.aggregation,
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
    try:
        result = evaluate_backend(
            backend, corpus,
            compute_comet_score=args.comet,
            compute_xcomet_score=args.xcomet,
        )
    finally:
        backend.close()

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
                                 "consensus", "geomean", "top_p", "ensemble",
                                 "gaussian_kernel", "gaussian_kernel_continuous",
                                 "cumulative"],
                        help="Attention aggregation method for border detection")
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

    # Metrics
    parser.add_argument("--comet", action="store_true")
    parser.add_argument("--xcomet", action="store_true")

    # Output
    parser.add_argument("--save", action="store_true", help="Save results to registry")
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("--omnisteval", help="Export to OmniSTEval JSONL format")

    args = parser.parse_args()

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

    # OmniSTEval export
    if args.omnisteval and len(results) == 1 and results[0].per_sentence:
        from .omnisteval import eval_result_to_omnisteval, write_jsonl_file
        eval_dict = {"per_sentence": results[0].per_sentence}
        entries = eval_result_to_omnisteval(eval_dict, talk_id_prefix=args.lang)
        write_jsonl_file(entries, args.omnisteval)


if __name__ == "__main__":
    main()
