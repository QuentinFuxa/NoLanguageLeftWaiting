"""Compute-aware latency metrics and research benchmark suite.

Extends the standard latency metrics with computation-aware versions
that account for actual inference time. This is critical for realistic
evaluation -- a system that takes 500ms per token has very different
real-world latency than one that takes 50ms.

Metrics:
    - CA-AL: Computation-Aware Average Lagging (includes inference time)
    - CA-YAAL: Computation-Aware YAAL (IWSLT 2026 reports both CU and CA)
    - Tokens/second: Generation throughput
    - First-token latency: Time to first translation output

Usage:
    from nllw.research import compute_ca_metrics, BenchmarkSuite
    metrics = compute_ca_metrics(delays_words, times_ms, source_length)
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .metrics import compute_yaal, compute_al, LatencyMetrics


@dataclass
class CAMetrics:
    """Computation-aware latency metrics for a single sentence."""
    # Standard (computation-unaware) metrics
    al_cu: float = 0.0
    yaal_cu: float = 0.0
    # Computation-aware metrics
    al_ca: float = 0.0
    yaal_ca: float = 0.0
    # Timing
    first_token_ms: float = 0.0  # Time to first translation token
    tokens_per_second: float = 0.0  # Generation throughput
    total_inference_ms: float = 0.0  # Total inference time
    n_tokens_generated: int = 0


def compute_ca_delays(
    delays_words: List[float],
    step_times_ms: List[float],
    words_per_second: float = 2.5,
) -> List[float]:
    """Convert computation-unaware word delays to computation-aware time delays.

    CU delays are in word counts (ideal, no compute overhead).
    CA delays add the actual inference time to each emission.

    Args:
        delays_words: CU delays (source words read when each target word emitted)
        step_times_ms: Cumulative inference time (ms) at each target word emission
        words_per_second: Source speech rate for time conversion

    Returns:
        CA delays in seconds (source time + inference time)
    """
    if not delays_words or not step_times_ms:
        return []

    ca_delays = []
    for i, (d, t) in enumerate(zip(delays_words, step_times_ms)):
        # Source arrival time (when the d-th word was spoken)
        source_time_s = d / words_per_second
        # Add inference time
        inference_time_s = t / 1000.0
        ca_delays.append(source_time_s + inference_time_s)

    return ca_delays


def compute_ca_metrics(
    delays_words: List[float],
    step_times_ms: List[float],
    source_length: int,
    target_length: int,
    words_per_second: float = 2.5,
) -> CAMetrics:
    """Compute both CU and CA metrics.

    Args:
        delays_words: CU delays in word counts
        step_times_ms: Cumulative inference time at each emission
        source_length: Number of source words
        target_length: Number of target words
        words_per_second: Source speech rate

    Returns:
        CAMetrics with both CU and CA versions
    """
    # CU metrics (standard)
    al_cu = compute_al(delays_words, source_length, target_length)
    yaal_cu = compute_yaal(delays_words, source_length, target_length)

    # CA metrics (with inference time)
    source_length_s = source_length / words_per_second
    ca_delays = compute_ca_delays(delays_words, step_times_ms, words_per_second)
    al_ca = compute_al(ca_delays, source_length_s, target_length)
    yaal_ca = compute_yaal(ca_delays, source_length_s, target_length)

    # Timing stats
    first_token_ms = step_times_ms[0] if step_times_ms else 0.0
    total_ms = step_times_ms[-1] if step_times_ms else 0.0
    n_tokens = len(step_times_ms)
    tps = (n_tokens / (total_ms / 1000.0)) if total_ms > 0 else 0.0

    return CAMetrics(
        al_cu=al_cu,
        yaal_cu=yaal_cu,
        al_ca=al_ca,
        yaal_ca=yaal_ca,
        first_token_ms=first_token_ms,
        tokens_per_second=tps,
        total_inference_ms=total_ms,
        n_tokens_generated=n_tokens,
    )


# ---------------------------------------------------------------------------
# Benchmark suite
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Result of a benchmark suite run."""
    name: str
    direction: str
    n_sentences: int = 0
    # Quality
    bleu: float = 0.0
    comet: Optional[float] = None
    # CU latency
    avg_al_cu: float = 0.0
    avg_yaal_cu: float = 0.0
    # CA latency
    avg_al_ca: float = 0.0
    avg_yaal_ca: float = 0.0
    # Throughput
    avg_tokens_per_second: float = 0.0
    avg_first_token_ms: float = 0.0
    # Config
    config: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        comet_str = f"COMET={self.comet:.3f}" if self.comet else ""
        return (
            f"{self.name:<25} | {self.direction} | "
            f"BLEU={self.bleu:5.1f} {comet_str} | "
            f"YAAL(CU)={self.avg_yaal_cu:5.2f} YAAL(CA)={self.avg_yaal_ca:5.2f} | "
            f"{self.avg_tokens_per_second:.1f} tok/s | "
            f"first={self.avg_first_token_ms:.0f}ms"
        )


class BenchmarkSuite:
    """Configurable benchmark suite for systematic evaluation.

    Runs multiple configurations on the same corpus and collects
    both CU and CA metrics for fair comparison.
    """

    def __init__(
        self,
        configs: List[Dict[str, Any]],
        direction: str = "en-zh",
        n_sentences: int = 50,
        corpus_suite: str = "flores",
    ):
        self.configs = configs
        self.direction = direction
        self.n_sentences = n_sentences
        self.corpus_suite = corpus_suite

    def run(self, compute_comet: bool = False, verbose: bool = True) -> List[BenchmarkResult]:
        """Run the benchmark suite.

        Returns list of BenchmarkResult, one per config.
        """
        from .eval import load_flores, evaluate_backend
        from .corpus import get_corpus_as_pairs
        from .backend_protocol import BackendConfig, create_backend

        # Load corpus once
        parts = self.direction.split("-")
        src_lang, tgt_lang = parts[0], parts[1]

        if self.corpus_suite == "corpus":
            corpus = get_corpus_as_pairs(self.direction, n=self.n_sentences)
            if not corpus:
                corpus = load_flores(src_lang, tgt_lang, n=self.n_sentences)
        else:
            corpus = load_flores(src_lang, tgt_lang, n=self.n_sentences)

        results = []
        for cfg in self.configs:
            full_cfg = {
                "direction": self.direction,
                "target_lang": tgt_lang,
                **cfg,
            }
            name = cfg.get("name", f"{cfg.get('backend_type', 'alignatt')}")

            if verbose:
                import sys
                print(f"\n--- {name} ---", file=sys.stderr)

            backend_cfg = BackendConfig.from_dict(full_cfg)
            backend = create_backend(backend_cfg)

            try:
                eval_result = evaluate_backend(
                    backend, corpus,
                    compute_comet_score=compute_comet,
                    verbose=verbose,
                )
            finally:
                backend.close()

            results.append(BenchmarkResult(
                name=name,
                direction=self.direction,
                n_sentences=eval_result.n_sentences,
                bleu=eval_result.bleu or 0.0,
                comet=eval_result.comet,
                avg_al_cu=eval_result.avg_al,
                avg_yaal_cu=eval_result.avg_yaal,
                avg_al_ca=eval_result.avg_al,  # TODO: wire up CA metrics
                avg_yaal_ca=eval_result.avg_yaal,  # TODO: wire up CA metrics
                avg_tokens_per_second=0.0,  # TODO
                avg_first_token_ms=0.0,  # TODO
                config=full_cfg,
            ))

        return results

    def print_results(self, results: List[BenchmarkResult]):
        """Print comparison table."""
        print("\n=== Benchmark Results ===")
        for r in results:
            print(r.summary())
