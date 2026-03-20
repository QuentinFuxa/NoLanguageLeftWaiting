"""Unified benchmarking CLI for NLLW simultaneous translation.

One-command benchmarking that wraps eval, research, and experiment modules
into a single streamlined interface. Designed for fast iteration during
IWSLT 2026 preparation.

Usage:
    # Basic benchmark (requires web server running on :8777)
    python -m nllw.bench --lang en-fr

    # Full corpus (130 sentences) with COMET
    python -m nllw.bench --suite corpus --lang en-fr --comet

    # Compare backends head-to-head
    python -m nllw.bench --compare alignatt alignatt-la --lang en-fr

    # Parameter sweep
    python -m nllw.bench --sweep "bd=2,3,4 wb=2,3" --lang en-fr

    # Per-direction optimization
    python -m nllw.bench --sweep "bd=2,3,4 wb=1,2,3" --lang en-zh,en-de,en-it,cs-en

    # Save results to experiment registry
    python -m nllw.bench --suite corpus --lang en-fr --comet --save

    # Output as JSON
    python -m nllw.bench --lang en-fr --json

Dependencies:
    - requests (for web API communication)
    - sacrebleu (optional, for BLEU)
    - unbabel-comet (optional, for COMET/xCOMET)
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Corpus resolution
# ---------------------------------------------------------------------------

def _resolve_suite(name: str, lang_pairs: list[str] | None = None) -> list[dict]:
    """Resolve a suite name to test cases, optionally filtered by lang pairs."""
    if name == "flores_mini":
        from nllw.research import FLORES_MINI
        cases = list(FLORES_MINI)
    elif name == "default":
        from nllw.eval import DEFAULT_TEST_CASES
        cases = list(DEFAULT_TEST_CASES)
    elif name == "corpus":
        from nllw.corpus import TestCorpus
        cases = list(TestCorpus.FULL_CORPUS)
    elif Path(name).is_file():
        with open(name, encoding="utf-8") as f:
            data = json.load(f)
        cases = data if isinstance(data, list) else data.get("sentences", data)
    else:
        print(f"Unknown suite: {name!r}. Use: flores_mini, default, corpus, or a JSON file path.")
        sys.exit(1)

    if lang_pairs:
        filtered = []
        for lp in lang_pairs:
            src, tgt = lp.split("-", 1)
            filtered.extend(
                tc for tc in cases
                if tc.get("source_lang") == src and tc.get("target_lang") == tgt
            )
        if not filtered:
            available = sorted(set(
                f"{tc.get('source_lang', '?')}-{tc.get('target_lang', '?')}"
                for tc in cases
            ))
            print(f"No test cases match lang pairs {lang_pairs}. Available: {available}")
            sys.exit(1)
        return filtered

    return cases


# ---------------------------------------------------------------------------
# Sweep config parsing
# ---------------------------------------------------------------------------

def _parse_sweep(sweep_str: str) -> dict[str, list]:
    """Parse sweep string like 'bd=2,3,4 wb=2,3' into a param grid.

    Supported short names:
        bd -> border_distance
        wb -> word_batch
        ctx -> context_window
        ev -> entropy_veto_threshold
    """
    aliases = {
        "bd": "border_distance",
        "wb": "word_batch",
        "ctx": "context_window",
        "ev": "entropy_veto_threshold",
    }
    grid: dict[str, list] = {}
    for part in sweep_str.strip().split():
        if "=" not in part:
            continue
        key, vals_str = part.split("=", 1)
        key = aliases.get(key, key)
        vals = []
        for v in vals_str.split(","):
            v = v.strip()
            if not v:
                continue
            try:
                vals.append(int(v))
            except ValueError:
                try:
                    vals.append(float(v))
                except ValueError:
                    vals.append(v)
        if vals:
            grid[key] = vals
    return grid


def _expand_grid(grid: dict[str, list]) -> list[dict]:
    """Expand a parameter grid into a list of config dicts."""
    keys = list(grid.keys())
    values = list(grid.values())
    configs = []
    for combo in itertools.product(*values):
        configs.append(dict(zip(keys, combo)))
    return configs


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def _format_table(rows: list[dict], columns: list[tuple[str, str, int]]) -> str:
    """Format a list of row dicts as a clean ASCII table.

    columns: list of (key, header, width) tuples.
    """
    lines = []

    # Header
    header_parts = []
    for key, header, width in columns:
        header_parts.append(f"{header:>{width}}")
    header_line = " | ".join(header_parts)
    lines.append(header_line)
    lines.append("-" * len(header_line))

    # Rows
    for row in rows:
        parts = []
        for key, header, width in columns:
            val = row.get(key)
            if val is None:
                parts.append(f"{'N/A':>{width}}")
            elif isinstance(val, float):
                if key in ("committed_ratio",):
                    parts.append(f"{val:>{width}.1%}")
                elif key in ("bleu",):
                    parts.append(f"{val:>{width}.1f}")
                elif key in ("comet", "xcomet_xl"):
                    parts.append(f"{val:>{width}.4f}")
                elif key in ("ca_al_ms", "avg_step_ms", "total_time_ms"):
                    parts.append(f"{val:>{width}.0f}")
                elif key in ("standard_al_words",):
                    parts.append(f"{val:>{width}.2f}")
                elif key in ("realtime_ratio",):
                    parts.append(f"{val:>{width}.3f}")
                else:
                    parts.append(f"{val:>{width}.2f}")
            elif isinstance(val, str):
                parts.append(f"{val:<{width}}" if key == "label" else f"{val:>{width}}")
            else:
                parts.append(f"{str(val):>{width}}")
        lines.append(" | ".join(parts))

    return "\n".join(lines)


_RESULT_COLUMNS = [
    ("label", "Config", 28),
    ("bleu", "BLEU", 6),
    ("comet", "COMET", 7),
    ("xcomet_xl", "xCOMET", 7),
    ("committed_ratio", "Commit%", 8),
    ("ca_al_ms", "CA-AL", 6),
    ("standard_al_words", "AL(w)", 6),
    ("avg_step_ms", "ms/w", 5),
    ("n_sentences", "N", 4),
]


# ---------------------------------------------------------------------------
# Core benchmark runner (via web API)
# ---------------------------------------------------------------------------

def run_bench(
    configs: list[dict],
    test_cases: list[dict],
    *,
    web_url: str = "http://localhost:8777",
    compute_comet: bool = False,
    compute_xcomet: bool = False,
    speech_rate_wps: float = 2.5,
    verbose: bool = False,
) -> list[dict]:
    """Run benchmark suite for multiple configs over test cases.

    Each config dict can contain:
        backend_type, border_distance, word_batch, context_window,
        entropy_veto_threshold, prompt_format, label, model_path

    Returns a list of result dicts (one per config) with aggregate metrics.
    """
    import requests as _requests

    results = []

    for cfg_idx, cfg in enumerate(configs):
        label = cfg.get("label", _make_label(cfg))
        backend_type = cfg.get("backend_type", "alignatt")

        if verbose:
            print(f"\n{'='*60}")
            print(f"  [{cfg_idx + 1}/{len(configs)}] {label}")
            print(f"{'='*60}")

        # Group test cases by lang pair
        by_pair: dict[tuple[str, str], list[dict]] = {}
        for tc in test_cases:
            pair = (tc.get("source_lang", "en"), tc.get("target_lang", "fr"))
            by_pair.setdefault(pair, []).append(tc)

        all_sources = []
        all_hypotheses = []
        all_references = []
        all_committed_ratios = []
        all_ca_al = []
        all_al_words = []
        all_step_ms = []
        all_sentences = []
        total_time_start = time.perf_counter()

        for (src_lang, tgt_lang), pair_cases in by_pair.items():
            # Load backend for this lang pair
            load_body = {
                "source_lang": src_lang,
                "target_lang": tgt_lang,
                "backend_type": backend_type,
                "border_distance": cfg.get("border_distance", 3),
                "word_batch": cfg.get("word_batch", 3),
            }
            if cfg.get("model_path"):
                load_body["model_path"] = cfg["model_path"]
            if cfg.get("prompt_format"):
                load_body["prompt_format"] = cfg["prompt_format"]
            if cfg.get("entropy_veto_threshold") is not None:
                load_body["entropy_veto_threshold"] = cfg["entropy_veto_threshold"]
            if cfg.get("context_window") is not None:
                load_body["context_window"] = cfg["context_window"]
            if cfg.get("heads_path"):
                load_body["heads_path"] = cfg["heads_path"]

            # Remove None values
            load_body = {k: v for k, v in load_body.items() if v is not None}

            try:
                resp = _requests.post(
                    f"{web_url.rstrip('/')}/load", json=load_body, timeout=120
                )
                load_result = resp.json()
                if not load_result.get("ok", True) and "error" in load_result:
                    print(f"  WARN: load failed for {src_lang}-{tgt_lang}: {load_result['error']}")
                    continue
            except Exception as e:
                print(f"  ERROR: cannot connect to {web_url}: {e}")
                sys.exit(1)

            for i, tc in enumerate(pair_cases):
                source = tc["source"]
                reference = tc.get("reference", "")
                words = source.strip().split()

                if verbose:
                    tag = tc.get("tag", tc.get("category", ""))
                    print(f"  [{i+1}/{len(pair_cases)}] {src_lang}-{tgt_lang} {tag}: {source[:50]}...")

                # Reset
                _requests.post(f"{web_url.rstrip('/')}/reset", timeout=10)

                # Feed words one by one
                steps = []
                cumulative_stable = ""
                prev_committed = 0
                sent_start = time.perf_counter()

                for w_idx, word in enumerate(words):
                    chunk = word + " "
                    step_start = time.perf_counter()
                    r = _requests.post(
                        f"{web_url.rstrip('/')}/translate",
                        json={"text": chunk},
                        timeout=30,
                    )
                    step_ms = (time.perf_counter() - step_start) * 1000.0
                    data = r.json()

                    stable = data.get("stable", "")
                    committed = data.get("committed_tokens", 0)
                    new_committed = max(0, committed - prev_committed)
                    cumulative_stable += stable
                    prev_committed = committed

                    steps.append({
                        "word_index": w_idx,
                        "new_committed": new_committed,
                        "step_time_ms": round(step_ms, 2),
                    })

                # Finish
                finish_start = time.perf_counter()
                finish_resp = _requests.post(f"{web_url.rstrip('/')}/finish", timeout=60).json()
                finish_ms = (time.perf_counter() - finish_start) * 1000.0
                sent_ms = (time.perf_counter() - sent_start) * 1000.0

                remaining = finish_resp.get("remaining", "")
                full_translation = finish_resp.get("full_translation", "")
                hypothesis = full_translation.strip() if full_translation else (cumulative_stable + remaining).strip()

                # Committed ratio
                hyp_len = max(len(hypothesis), 1)
                remaining_len = len(remaining) if remaining else 0
                committed_ratio = max(0, hyp_len - remaining_len) / hyp_len

                # Compute AL metrics
                trace = {
                    "steps": steps,
                    "num_words": len(words),
                    "committed_before_finish": prev_committed,
                }
                from nllw.research import ResearchBenchmark
                ca = ResearchBenchmark.compute_aware_metrics(trace, speech_rate_wps=speech_rate_wps)

                all_sources.append(source)
                all_hypotheses.append(hypothesis)
                all_references.append(reference)
                all_committed_ratios.append(committed_ratio)
                all_ca_al.append(ca["ca_al_ms"])
                all_al_words.append(ca["standard_al_words"])
                all_step_ms.append(ca["avg_step_ms"])

                all_sentences.append({
                    "source": source,
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "tag": tc.get("tag", tc.get("category", "")),
                    "num_words": len(words),
                    "committed_ratio": round(committed_ratio, 4),
                    "ca_al_ms": ca["ca_al_ms"],
                    "standard_al_words": ca["standard_al_words"],
                    "total_time_ms": round(sent_ms, 1),
                    "steps": steps,
                })

        total_time_ms = (time.perf_counter() - total_time_start) * 1000.0

        # Corpus-level BLEU
        bleu = None
        try:
            import sacrebleu
            if all_hypotheses and all_references:
                bleu = round(sacrebleu.corpus_bleu(all_hypotheses, [all_references]).score, 1)
        except ImportError:
            pass

        # COMET / xCOMET
        comet_score = None
        xcomet_score = None
        if (compute_comet or compute_xcomet) and all_hypotheses and all_references:
            try:
                from nllw.metrics import compute_comet, compute_xcomet, comet_available
                if comet_available():
                    if compute_comet:
                        try:
                            res = compute_comet(all_sources, all_hypotheses, all_references)
                            comet_score = res.get("score")
                        except Exception as e:
                            if verbose:
                                print(f"  WARN: COMET failed: {e}")
                    if compute_xcomet:
                        try:
                            res = compute_xcomet(all_sources, all_hypotheses, all_references)
                            xcomet_score = res.get("score")
                        except Exception as e:
                            if verbose:
                                print(f"  WARN: xCOMET failed: {e}")
            except ImportError:
                if verbose:
                    print("  WARN: nllw.metrics not available for COMET")

        n = len(all_hypotheses)
        result = {
            "label": label,
            "config": {k: v for k, v in cfg.items() if k != "label"},
            "bleu": bleu,
            "comet": comet_score,
            "xcomet_xl": xcomet_score,
            "committed_ratio": round(sum(all_committed_ratios) / n, 4) if n else None,
            "ca_al_ms": round(sum(all_ca_al) / n, 1) if n else None,
            "standard_al_words": round(sum(all_al_words) / n, 3) if n else None,
            "avg_step_ms": round(sum(all_step_ms) / n, 1) if n else None,
            "total_time_ms": round(total_time_ms, 1),
            "n_sentences": n,
            "sentences": all_sentences,
        }
        results.append(result)

        # Print inline summary
        if verbose:
            print(f"\n  => BLEU={bleu}  Commit={result['committed_ratio']:.1%}  "
                  f"CA-AL={result['ca_al_ms']}ms  AL={result['standard_al_words']}w  "
                  f"({n} sentences, {total_time_ms:.0f}ms total)")

    return results


def _make_label(cfg: dict) -> str:
    """Create a short label from a config dict."""
    parts = [cfg.get("backend_type", "alignatt")]
    if cfg.get("border_distance") is not None:
        parts.append(f"bd={cfg['border_distance']}")
    if cfg.get("word_batch") is not None:
        parts.append(f"wb={cfg['word_batch']}")
    if cfg.get("context_window"):
        parts.append(f"ctx={cfg['context_window']}")
    if cfg.get("entropy_veto_threshold") is not None:
        parts.append(f"ev={cfg['entropy_veto_threshold']}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Save to experiment registry
# ---------------------------------------------------------------------------

def _save_to_registry(results: list[dict], registry_dir: str = "experiments/") -> list[Path]:
    """Save benchmark results to the experiment registry."""
    from nllw.experiment import ExperimentConfig, ExperimentResult, ExperimentRegistry

    registry = ExperimentRegistry(registry_dir)
    saved_paths = []

    for r in results:
        cfg_dict = r.get("config", {})
        config = ExperimentConfig(
            backend_type=cfg_dict.get("backend_type", "alignatt"),
            model_path=cfg_dict.get("model_path"),
            heads_path=cfg_dict.get("heads_path"),
            border_distance=cfg_dict.get("border_distance", 3),
            word_batch=cfg_dict.get("word_batch", 3),
            context_window=cfg_dict.get("context_window", 0),
            entropy_veto_threshold=cfg_dict.get("entropy_veto_threshold"),
            prompt_format=cfg_dict.get("prompt_format", "hymt"),
        )

        # Infer lang_pair from sentences
        if r.get("sentences"):
            first = r["sentences"][0]
            src = first.get("source", "").split()
            # Try to get from tag
            tag = first.get("tag", "")
            if "/" in tag:
                config.lang_pair = tag.split("/")[0]

        result = ExperimentResult(
            config=config,
            bleu=r.get("bleu"),
            comet=r.get("comet"),
            xcomet_xl=r.get("xcomet_xl"),
            committed_ratio=r.get("committed_ratio"),
            ca_al_ms=r.get("ca_al_ms"),
            al_words=r.get("standard_al_words"),
            time_per_word_ms=r.get("avg_step_ms"),
            total_time_ms=r.get("total_time_ms"),
            per_sentence=r.get("sentences", []),
        )

        path = registry.save(result)
        saved_paths.append(path)

    return saved_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m nllw.bench",
        description="NLLW Benchmark - one-command SimulMT evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m nllw.bench --lang en-fr\n"
            "  python -m nllw.bench --suite corpus --lang en-fr --comet\n"
            "  python -m nllw.bench --compare alignatt alignatt-la --lang en-fr\n"
            "  python -m nllw.bench --sweep 'bd=2,3,4 wb=2,3' --lang en-fr\n"
            "  python -m nllw.bench --sweep 'bd=2,3,4 wb=1,2,3' --lang en-zh,en-de\n"
            "  python -m nllw.bench --suite corpus --lang en-fr --comet --save\n"
        ),
    )

    # --- Suite / corpus ---
    parser.add_argument(
        "--suite", "-s",
        default="flores_mini",
        help="Test suite: flores_mini (20 sent), default (15 sent), corpus (130 sent), or JSON path. Default: flores_mini",
    )
    parser.add_argument(
        "--lang", "-l",
        default=None,
        help="Language pair(s), comma-separated. E.g. 'en-fr' or 'en-fr,en-de,en-zh'",
    )

    # --- Backend / config ---
    parser.add_argument(
        "--backend", "-b",
        default="alignatt",
        help="Backend type. Default: alignatt",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        metavar="BACKEND",
        help="Compare multiple backends head-to-head (e.g. --compare alignatt alignatt-la alignatt-kv)",
    )
    parser.add_argument(
        "--sweep",
        default=None,
        help="Parameter sweep. E.g. 'bd=2,3,4 wb=2,3' or 'bd=2,3,4 wb=2,3 ev=0.70,0.75,0.80'",
    )
    parser.add_argument(
        "--bd", type=int, default=3,
        help="Border distance (default: 3)",
    )
    parser.add_argument(
        "--wb", type=int, default=3,
        help="Word batch size (default: 3)",
    )
    parser.add_argument(
        "--ctx", type=int, default=None,
        help="Context window size",
    )
    parser.add_argument(
        "--ev", type=float, default=None,
        help="Entropy veto threshold",
    )
    parser.add_argument(
        "--prompt-format",
        default=None,
        help="Prompt format (hymt, qwen3, qwen3.5, qwen3-nothink, eurollm, custom)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model path (GGUF)",
    )

    # --- Metrics ---
    parser.add_argument(
        "--comet", action="store_true",
        help="Compute COMET (wmt22-comet-da) scores",
    )
    parser.add_argument(
        "--xcomet", action="store_true",
        help="Compute xCOMET-XL scores",
    )

    # --- Output ---
    parser.add_argument(
        "--save", action="store_true",
        help="Save results to experiment registry (experiments/)",
    )
    parser.add_argument(
        "--registry-dir",
        default="experiments/",
        help="Registry directory for --save (default: experiments/)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--omnisteval",
        default=None,
        metavar="FILE",
        help="Export to OmniSTEval JSONL format (for IWSLT submission)",
    )
    parser.add_argument(
        "--omnisteval-word-level",
        action="store_true",
        help="Use word-level delays for OmniSTEval (default: char-level for CJK)",
    )
    parser.add_argument(
        "--omnisteval-talk-id",
        default="bench",
        help="Talk/recording ID for OmniSTEval (default: bench)",
    )

    # --- Server ---
    parser.add_argument(
        "--url",
        default="http://localhost:8777",
        help="Web debug server URL (default: http://localhost:8777)",
    )
    parser.add_argument(
        "--speech-rate",
        type=float, default=2.5,
        help="Simulated speech rate in words/sec for CA-AL (default: 2.5)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose per-sentence output",
    )

    args = parser.parse_args()

    # --- Parse lang pairs ---
    lang_pairs = None
    if args.lang:
        lang_pairs = [lp.strip() for lp in args.lang.split(",")]

    # --- Resolve test cases ---
    test_cases = _resolve_suite(args.suite, lang_pairs)

    # --- Build configs ---
    configs: list[dict] = []

    if args.compare:
        # Multi-backend comparison
        for backend in args.compare:
            cfg = {
                "backend_type": backend,
                "border_distance": args.bd,
                "word_batch": args.wb,
            }
            if args.ctx is not None:
                cfg["context_window"] = args.ctx
            if args.ev is not None:
                cfg["entropy_veto_threshold"] = args.ev
            if args.model:
                cfg["model_path"] = args.model
            if args.prompt_format:
                cfg["prompt_format"] = args.prompt_format
            configs.append(cfg)

    elif args.sweep:
        # Parameter sweep
        grid = _parse_sweep(args.sweep)
        base_cfg = {"backend_type": args.backend}
        if args.model:
            base_cfg["model_path"] = args.model
        if args.prompt_format:
            base_cfg["prompt_format"] = args.prompt_format

        for combo in _expand_grid(grid):
            cfg = dict(base_cfg)
            cfg.update(combo)
            # Fill in defaults for params not in sweep
            cfg.setdefault("border_distance", args.bd)
            cfg.setdefault("word_batch", args.wb)
            if args.ctx is not None:
                cfg.setdefault("context_window", args.ctx)
            if args.ev is not None:
                cfg.setdefault("entropy_veto_threshold", args.ev)
            configs.append(cfg)

    else:
        # Single config
        cfg = {
            "backend_type": args.backend,
            "border_distance": args.bd,
            "word_batch": args.wb,
        }
        if args.ctx is not None:
            cfg["context_window"] = args.ctx
        if args.ev is not None:
            cfg["entropy_veto_threshold"] = args.ev
        if args.model:
            cfg["model_path"] = args.model
        if args.prompt_format:
            cfg["prompt_format"] = args.prompt_format
        configs.append(cfg)

    # --- Print header ---
    n_configs = len(configs)
    n_cases = len(test_cases)
    pairs_str = ", ".join(lang_pairs) if lang_pairs else "all"
    metrics = ["BLEU"]
    if args.comet:
        metrics.append("COMET")
    if args.xcomet:
        metrics.append("xCOMET-XL")

    print(f"NLLW Bench")
    print(f"  Suite:    {args.suite} ({n_cases} sentences)")
    print(f"  Langs:    {pairs_str}")
    print(f"  Configs:  {n_configs}")
    print(f"  Metrics:  {', '.join(metrics)}")
    print(f"  Server:   {args.url}")
    if n_configs <= 5:
        for i, cfg in enumerate(configs):
            print(f"  [{i+1}] {_make_label(cfg)}")
    print()

    # --- Run ---
    start = time.perf_counter()
    results = run_bench(
        configs,
        test_cases,
        web_url=args.url,
        compute_comet=args.comet,
        compute_xcomet=args.xcomet,
        speech_rate_wps=args.speech_rate,
        verbose=args.verbose,
    )
    elapsed = time.perf_counter() - start

    # --- Output ---
    if args.json:
        # Strip per-sentence detail for cleaner JSON
        compact = []
        for r in results:
            cr = {k: v for k, v in r.items() if k != "sentences"}
            compact.append(cr)
        print(json.dumps(compact, indent=2, ensure_ascii=False))
    else:
        # Clean table
        table_rows = []
        for r in results:
            table_rows.append({
                "label": r["label"],
                "bleu": r["bleu"],
                "comet": r.get("comet"),
                "xcomet_xl": r.get("xcomet_xl"),
                "committed_ratio": r["committed_ratio"],
                "ca_al_ms": r["ca_al_ms"],
                "standard_al_words": r["standard_al_words"],
                "avg_step_ms": r["avg_step_ms"],
                "n_sentences": r["n_sentences"],
            })

        # Filter columns based on available data
        active_cols = []
        for key, header, width in _RESULT_COLUMNS:
            if any(row.get(key) is not None for row in table_rows):
                active_cols.append((key, header, width))

        print(_format_table(table_rows, active_cols))
        print(f"\nCompleted in {elapsed:.1f}s")

        # Show best if multiple configs
        if len(results) > 1:
            print()
            _print_best(results)

    # --- Save ---
    if args.save:
        paths = _save_to_registry(results, args.registry_dir)
        print(f"\nSaved {len(paths)} result(s) to {args.registry_dir}")
        for p in paths:
            print(f"  {p}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")

    # --- OmniSTEval export ---
    if args.omnisteval:
        from nllw.omnisteval import bench_to_omnisteval, write_omnisteval_jsonl

        # Export first config's sentences (or all if single config)
        for i, r in enumerate(results):
            sentences = r.get("sentences", [])
            if not sentences:
                continue
            talk_id = args.omnisteval_talk_id
            if len(results) > 1:
                talk_id = f"{talk_id}.cfg{i}"
            records = bench_to_omnisteval(
                sentences,
                talk_id=talk_id,
                word_level=args.omnisteval_word_level,
                speech_rate_wps=args.speech_rate,
            )
            # Build output path (add config index if multiple)
            out_path = args.omnisteval
            if len(results) > 1:
                base = Path(out_path)
                out_path = str(base.parent / f"{base.stem}.cfg{i}{base.suffix}")
            path = write_omnisteval_jsonl(records, out_path)
            print(f"\nOmniSTEval: {len(records)} records -> {path}")


def _print_best(results: list[dict]) -> None:
    """Print best config for each metric."""
    higher_better = ["bleu", "comet", "xcomet_xl", "committed_ratio"]
    lower_better = ["ca_al_ms", "standard_al_words", "avg_step_ms"]

    print("Best per metric:")
    for metric in higher_better + lower_better:
        scored = [(r, r.get(metric)) for r in results if r.get(metric) is not None]
        if not scored:
            continue
        if metric in lower_better:
            best = min(scored, key=lambda x: x[1])
        else:
            best = max(scored, key=lambda x: x[1])
        r, val = best
        if isinstance(val, float):
            if metric in ("committed_ratio",):
                val_s = f"{val:.1%}"
            elif metric in ("comet", "xcomet_xl"):
                val_s = f"{val:.4f}"
            else:
                val_s = f"{val:.1f}"
        else:
            val_s = str(val)
        print(f"  {metric:<20} = {val_s:<12} ({r['label']})")


if __name__ == "__main__":
    main()
