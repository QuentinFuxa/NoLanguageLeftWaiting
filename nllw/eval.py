"""Evaluation harness for SimulMT backends.

Supports:
    - FLORES+ corpus loading (any language pair)
    - Streaming evaluation (word-by-word simulation)
    - Batch evaluation with parameter sweeps
    - BLEU, COMET, XCOMET-XL quality metrics
    - Full latency metrics (AL, LAAL, YAAL, AP, DAL, MaxCW)
    - JSON result export

Usage:
    python -m nllw.eval --backend web --lang en-fr --comet
    python -m nllw.eval --backend alignatt --model /path/to.gguf --lang en-zh -n 50
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

from .metrics import (
    bootstrap_confidence_interval,
    compute_all_metrics,
    compute_bleu_corpus,
    compute_comet,
    LatencyMetrics,
)
from .simulate import SimulationTrace


# ---------------------------------------------------------------------------
# FLORES+ corpus loading
# ---------------------------------------------------------------------------

# Language code -> flores_plus config name
FLORES_LANG_MAP = {
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "zh": "cmn_Hans",
    "cs": "ces_Latn",
    "ja": "jpn_Jpan",
    "ru": "rus_Cyrl",
    "ar": "arb_Arab",
    "pt": "por_Latn",
    "es": "spa_Latn",
    "nl": "nld_Latn",
    "ko": "kor_Hang",
    "tr": "tur_Latn",
    "pl": "pol_Latn",
    "uk": "ukr_Cyrl",
}


def _flores_config(lang: str) -> str:
    """Convert short language code to flores_plus config name."""
    if lang in FLORES_LANG_MAP:
        return FLORES_LANG_MAP[lang]
    # Try as-is (e.g. "eng_Latn")
    return lang


def load_flores(
    src_lang: str = "en",
    tgt_lang: str = "fr",
    split: str = "devtest",
    n: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Load parallel sentences from FLORES+.

    Args:
        src_lang: Source language code (e.g. "en", "fr", "zh")
        tgt_lang: Target language code
        split: Dataset split ("dev" or "devtest")
        n: Max number of sentences (None = all)

    Returns:
        List of {"source": str, "reference": str, "id": int}
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets required. Install: pip install datasets")

    src_config = _flores_config(src_lang)
    tgt_config = _flores_config(tgt_lang)

    src_ds = load_dataset("openlanguagedata/flores_plus", src_config, split=split)
    tgt_ds = load_dataset("openlanguagedata/flores_plus", tgt_config, split=split)

    assert len(src_ds) == len(tgt_ds), f"Mismatched: {len(src_ds)} vs {len(tgt_ds)}"

    pairs = []
    limit = n if n else len(src_ds)
    for i in range(min(limit, len(src_ds))):
        pairs.append({
            "source": src_ds[i]["text"],
            "reference": tgt_ds[i]["text"],
            "id": src_ds[i]["id"],
        })

    return pairs


# ---------------------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Result of evaluating a backend on a corpus."""
    # Config
    backend_type: str = ""
    direction: str = ""
    n_sentences: int = 0
    config: Dict[str, Any] = field(default_factory=dict)
    # Quality
    bleu: Optional[float] = None
    comet: Optional[float] = None
    xcomet: Optional[float] = None
    # Latency (averages)
    avg_al: float = 0.0
    avg_laal: float = 0.0
    avg_yaal: float = 0.0
    avg_longyaal: float = 0.0       # IWSLT 2026 PRIMARY latency metric
    avg_stream_laal: float = 0.0    # IWSLT 2026 secondary latency metric
    avg_ap: float = 0.0
    avg_dal: float = 0.0
    avg_max_cw: float = 0.0
    # Per-sentence
    per_sentence: List[Dict[str, Any]] = field(default_factory=list)
    # Timing
    total_time_s: float = 0.0
    avg_time_per_sentence_ms: float = 0.0
    # Confidence intervals (bootstrap, 95%)
    comet_ci: Optional[Tuple[float, float]] = None  # (lower, upper)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Don't include per_sentence in summary
        d.pop("per_sentence", None)
        return d

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"=== {self.backend_type} | {self.direction} | {self.n_sentences} sentences ===",
            f"  Quality: BLEU={self.bleu:.1f}" + (f" COMET={self.comet:.3f}" if self.comet else ""),
        ]
        if self.comet_ci:
            lines[-1] += f" [{self.comet_ci[0]:.3f}, {self.comet_ci[1]:.3f}]"
        if self.xcomet:
            lines[-1] += f" XCOMET={self.xcomet:.3f}"
        lines.extend([
            f"  Latency: LongYAAL={self.avg_longyaal:.2f} YAAL={self.avg_yaal:.2f} "
            f"StreamLAAL={self.avg_stream_laal:.2f} AL={self.avg_al:.2f}",
            f"  AP={self.avg_ap:.3f} DAL={self.avg_dal:.2f} MaxCW={self.avg_max_cw:.1f} | "
            f"{self.avg_time_per_sentence_ms:.0f}ms/sent",
        ])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def evaluate_backend(
    backend,
    corpus: List[Dict[str, str]],
    compute_comet_score: bool = False,
    compute_xcomet_score: bool = False,
    comet_batch_size: int = 8,
    verbose: bool = True,
    trace_collector=None,
) -> EvalResult:
    """Evaluate a SimulMT backend on a parallel corpus.

    Args:
        backend: SimulMTBackend instance
        corpus: List of {"source": str, "reference": str}
        compute_comet_score: Whether to compute COMET
        compute_xcomet_score: Whether to compute XCOMET-XL
        verbose: Print progress
        trace_collector: Optional TraceCollector for fusion calibration data

    Returns:
        EvalResult with all metrics
    """
    from .simulate import simulate_backend

    sources = []
    references = []
    hypotheses = []
    all_metrics = []
    per_sentence = []

    t0 = time.time()

    for i, item in enumerate(corpus):
        source = item["source"]
        reference = item["reference"]

        # Start trace collection for this sentence
        if trace_collector is not None:
            direction = backend.config.direction if hasattr(backend, 'config') else ""
            trace_collector.start_sentence(
                sentence_id=item.get("id", i),
                source=source,
                reference=reference,
                direction=direction,
            )

        # Simulate word-by-word
        trace = simulate_backend(backend, source, is_final_on_last=True)

        sources.append(source)
        references.append(reference)
        hypotheses.append(trace.translation)
        all_metrics.append(trace.metrics)

        per_sentence.append({
            "id": item.get("id", i),
            "source": source,
            "reference": reference,
            "hypothesis": trace.translation,
            "delays": trace.delays,
            "metrics": {
                "al": trace.metrics.al,
                "laal": trace.metrics.laal,
                "yaal": trace.metrics.yaal,
                "longyaal": trace.metrics.longyaal,
                "stream_laal": trace.metrics.stream_laal,
                "ap": trace.metrics.ap,
                "dal": trace.metrics.dal,
                "max_cw": trace.metrics.max_cw,
            },
        })

        # End trace collection for this sentence
        if trace_collector is not None:
            trace_collector.end_sentence(output_text=trace.translation)

        # Reset for next sentence
        backend.reset()

        if verbose and (i + 1) % 10 == 0:
            print(
                f"  [{i+1}/{len(corpus)}] "
                f"src: {source[:50]}... -> {trace.translation[:50]}...",
                file=sys.stderr, flush=True,
            )

    total_time = time.time() - t0

    # Aggregate latency metrics
    n = len(all_metrics)
    result = EvalResult(
        backend_type=backend.name,
        direction=backend.config.direction if hasattr(backend, 'config') else "",
        n_sentences=n,
        config=backend.config.__dict__ if hasattr(backend, 'config') else {},
        avg_al=sum(m.al for m in all_metrics) / n if n else 0,
        avg_laal=sum(m.laal for m in all_metrics) / n if n else 0,
        avg_yaal=sum(m.yaal for m in all_metrics) / n if n else 0,
        avg_longyaal=sum(m.longyaal for m in all_metrics) / n if n else 0,
        avg_stream_laal=sum(m.stream_laal for m in all_metrics) / n if n else 0,
        avg_ap=sum(m.ap for m in all_metrics) / n if n else 0,
        avg_dal=sum(m.dal for m in all_metrics) / n if n else 0,
        avg_max_cw=sum(m.max_cw for m in all_metrics) / n if n else 0,
        per_sentence=per_sentence,
        total_time_s=total_time,
        avg_time_per_sentence_ms=(total_time / n * 1000) if n else 0,
    )

    # Compute BLEU (use language-appropriate tokenizer for CJK)
    tgt_lang = None
    if hasattr(backend, 'config') and hasattr(backend.config, 'direction'):
        parts = backend.config.direction.split("-")
        if len(parts) == 2:
            tgt_lang = parts[1]
    try:
        result.bleu = compute_bleu_corpus(hypotheses, references,
                                          target_lang=tgt_lang)
    except Exception as e:
        print(f"  BLEU computation failed: {e}", file=sys.stderr)
        result.bleu = 0.0

    # Compute COMET
    if compute_comet_score:
        try:
            score, per_scores = compute_comet(sources, hypotheses, references, batch_size=comet_batch_size)
            result.comet = score
            # Bootstrap 95% confidence interval
            if per_scores and len(per_scores) >= 10:
                _, ci_lo, ci_hi = bootstrap_confidence_interval(per_scores)
                result.comet_ci = (ci_lo, ci_hi)
        except Exception as e:
            print(f"  COMET computation failed: {e}", file=sys.stderr)

    # Compute XCOMET-XL via separate process (avoids OOM).
    # XCOMET-XL (~12GB VRAM) OOMs when the translation model is still loaded.
    # Must close the backend to free GPU VRAM before launching subprocess.
    if compute_xcomet_score:
        if verbose:
            print("  Scoring XCOMET-XL via subprocess (avoids OOM)...", file=sys.stderr)
        # Close backend to free GPU VRAM -- XCOMET-XL needs it
        if hasattr(backend, 'close'):
            backend.close()
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass
        try:
            import tempfile
            from .xcomet_scorer import save_hypotheses_json, score_xcomet_subprocess

            # Save hypotheses to temp file
            fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix="nllw_eval_")
            os.close(fd)
            save_hypotheses_json(result, tmp_path)

            # Score in subprocess (GPU now free from translation model)
            scores = score_xcomet_subprocess(
                tmp_path,
                batch_size=min(comet_batch_size, 2),  # cap at 2 for A40 safety
                timeout=600,
            )
            if scores and "system_score" in scores:
                result.xcomet = scores["system_score"]
                if verbose:
                    print(f"  XCOMET-XL: {result.xcomet:.4f}", file=sys.stderr)
            else:
                print("  XCOMET-XL subprocess returned no scores", file=sys.stderr)

            # Cleanup
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception as e:
            print(f"  XCOMET-XL computation failed: {e}", file=sys.stderr)

    if verbose:
        print(result.summary(), file=sys.stderr)

    return result


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

def parameter_sweep(
    backend_factory,
    corpus: List[Dict[str, str]],
    param_grid: Dict[str, List[Any]],
    base_config: Dict[str, Any],
    compute_comet_score: bool = False,
    verbose: bool = True,
) -> List[EvalResult]:
    """Run evaluation across a grid of parameters.

    Args:
        backend_factory: Callable that takes a config dict and returns a backend
        corpus: Parallel corpus
        param_grid: e.g. {"border_distance": [2, 3, 4], "word_batch": [1, 2, 3]}
        base_config: Base configuration dict
        compute_comet_score: Whether to compute COMET for each config

    Returns:
        List of EvalResult, one per parameter combination
    """
    import itertools

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    results = []
    for combo in combinations:
        config = dict(base_config)
        for k, v in zip(keys, combo):
            config[k] = v

        if verbose:
            params_str = ", ".join(f"{k}={v}" for k, v in zip(keys, combo))
            print(f"\n--- Sweep: {params_str} ---", file=sys.stderr)

        backend = backend_factory(config)
        try:
            result = evaluate_backend(
                backend, corpus,
                compute_comet_score=compute_comet_score,
                verbose=verbose,
            )
            result.config = config
            results.append(result)
        finally:
            backend.close()

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="NLLW SimulMT Evaluation")
    parser.add_argument("--backend", default="web", choices=["web", "alignatt", "full-sentence", "eager"])
    parser.add_argument("--model", help="Path to GGUF model (for alignatt backend)")
    parser.add_argument("--heads", help="Path to head config JSON")
    parser.add_argument("--lang", default="en-fr", help="Language direction (e.g. en-fr, en-zh)")
    parser.add_argument("--prompt-format", default=None, help="Prompt format override")
    parser.add_argument("-n", type=int, default=50, help="Number of sentences")
    parser.add_argument("--border-distance", type=int, default=3)
    parser.add_argument("--word-batch", type=int, default=3)
    parser.add_argument("--comet", action="store_true", help="Compute COMET score")
    parser.add_argument("--xcomet", action="store_true", help="Compute XCOMET-XL score")
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    # Parse direction
    parts = args.lang.split("-")
    src_lang, tgt_lang = parts[0], parts[1]

    print(f"Loading FLORES+ {args.lang} ({args.n} sentences)...", file=sys.stderr)
    corpus = load_flores(src_lang, tgt_lang, n=args.n)
    print(f"Loaded {len(corpus)} sentence pairs.", file=sys.stderr)

    # Create backend
    if args.backend == "web":
        print("Web backend not yet implemented. Use --backend alignatt.", file=sys.stderr)
        sys.exit(1)

    from .backend_protocol import BackendConfig, create_backend

    config = BackendConfig(
        backend_type=args.backend,
        model_path=args.model or "",
        heads_path=args.heads or "",
        prompt_format=args.prompt_format or "hymt",
        direction=args.lang,
        border_distance=args.border_distance,
        word_batch=args.word_batch,
        target_lang=tgt_lang,
    )
    backend = create_backend(config)

    try:
        result = evaluate_backend(
            backend, corpus,
            compute_comet_score=args.comet,
            compute_xcomet_score=args.xcomet,
            verbose=args.verbose,
        )
    finally:
        backend.close()

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "summary": result.to_dict(),
                "per_sentence": result.per_sentence,
            }, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output}", file=sys.stderr)

    print(result.summary())


if __name__ == "__main__":
    main()
