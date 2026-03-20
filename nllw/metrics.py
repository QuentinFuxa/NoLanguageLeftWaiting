"""SimulMT latency and quality metrics.

Implements the key metrics for evaluating simultaneous translation systems:

Latency metrics:
    - AL (Average Lagging): Standard latency metric from Ma et al. (2019)
    - LAAL (Length-Adaptive AL): Corrected for length differences
    - YAAL: OmniSTEval formula used as IWSLT 2026 primary latency metric
    - AP (Average Proportion): Proportion of source read before each target word
    - DAL (Differentiable AL): Smooth approximation of AL
    - MaxCW (Max Consecutive Wait): Longest streak without output

Quality metrics (wrappers):
    - BLEU (via sacrebleu)
    - COMET / XCOMET-XL (via comet)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import math


@dataclass
class LatencyMetrics:
    """Collection of latency metrics for a single sentence."""
    al: float = 0.0        # Average Lagging
    laal: float = 0.0      # Length-Adaptive AL
    yaal: float = 0.0      # OmniSTEval YAAL
    ap: float = 0.0        # Average Proportion
    dal: float = 0.0       # Differentiable AL
    max_cw: int = 0        # Max Consecutive Wait
    n_source: int = 0      # Source length (words)
    n_target: int = 0      # Target length (words)


def compute_al(delays: List[float], source_length: int, target_length: int) -> float:
    """Average Lagging (Ma et al., 2019).

    AL = (1/tau) * sum_{t=1}^{tau} [g(t) - (t-1) * |x| / |y|]
    where tau = min(t : g(t) = |x|), g(t) = delay for target word t.

    Args:
        delays: g(t) for each target word (1-indexed conceptually, 0-indexed in list)
        source_length: |x| number of source words
        target_length: |y| number of target words
    """
    if not delays or source_length == 0 or target_length == 0:
        return 0.0

    gamma = source_length / target_length
    tau = len(delays)
    for t, d in enumerate(delays):
        if d >= source_length:
            tau = t + 1
            break

    al = sum(delays[t] - t * gamma for t in range(tau)) / tau
    return al


def compute_laal(delays: List[float], source_length: int, target_length: int) -> float:
    """Length-Adaptive Average Lagging.

    Same as AL but gamma = max(|delays|, target_length) / source_length,
    which handles length mismatches better.
    """
    if not delays or source_length == 0:
        return 0.0

    gamma = max(len(delays), target_length) / source_length
    tau = len(delays)
    for t, d in enumerate(delays):
        if d >= source_length:
            tau = t + 1
            break

    laal = sum(delays[t] - t / gamma for t in range(tau)) / tau
    return laal


def compute_yaal(
    delays: List[float],
    source_length: float,
    target_length: int,
    is_longform: bool = True,
) -> float:
    """YAAL metric from OmniSTEval (IWSLT 2026 primary latency metric).

    Exact formula from omnisteval/scoring.py:YAALScorer:
        gamma = max(len(delays), target_length) / source_length
        yaal = sum(d - t/gamma for t, d in enumerate(delays) if condition) / tau

    Args:
        delays: Delay values for each target word/token
        source_length: Source length (words for word-domain, seconds for time-domain)
        target_length: Target length (words/tokens)
        is_longform: If True (default), count all delays. If False, stop at source boundary.

    Returns:
        YAAL score (lower is better, 0 = simultaneous with no lag)
    """
    if not delays or source_length == 0:
        return 0.0

    gamma = max(len(delays), target_length) / source_length
    yaal = 0.0
    tau = 0

    for t, d in enumerate(delays):
        if d >= source_length and not is_longform:
            break
        yaal += d - t / gamma
        tau += 1

    return yaal / tau if tau > 0 else 0.0


def compute_ap(delays: List[float], source_length: int, target_length: int) -> float:
    """Average Proportion.

    AP = (1/|y|) * sum_{t=1}^{|y|} g(t) / |x|
    """
    if not delays or source_length == 0 or target_length == 0:
        return 0.0
    return sum(min(d, source_length) for d in delays) / (source_length * len(delays))


def compute_dal(delays: List[float], source_length: int, target_length: int) -> float:
    """Differentiable Average Lagging (Cherry & Foster, 2019).

    Uses monotonized delays: g'(t) = max(g(t), g'(t-1) + gamma).
    """
    if not delays or source_length == 0 or target_length == 0:
        return 0.0

    gamma = source_length / target_length

    # Monotonize
    mono = [delays[0]]
    for t in range(1, len(delays)):
        mono.append(max(delays[t], mono[-1] + gamma))

    tau = len(mono)
    for t, d in enumerate(mono):
        if d >= source_length:
            tau = t + 1
            break

    dal = sum(mono[t] - t * gamma for t in range(tau)) / tau
    return dal


def compute_max_consecutive_wait(delays: List[float]) -> int:
    """Maximum number of consecutive source words read without producing output.

    Measures burstiness: high MaxCW means long silences followed by bursts.
    """
    if len(delays) <= 1:
        return 0

    max_wait = 0
    for t in range(1, len(delays)):
        wait = int(delays[t] - delays[t - 1])
        max_wait = max(max_wait, wait)
    return max_wait


def compute_all_metrics(
    delays: List[float],
    source_length: int,
    target_length: int,
    is_longform: bool = True,
) -> LatencyMetrics:
    """Compute all latency metrics at once.

    Args:
        delays: Delay values for each target word
        source_length: Number of source words
        target_length: Number of target words
        is_longform: Whether to use longform YAAL

    Returns:
        LatencyMetrics dataclass with all values
    """
    return LatencyMetrics(
        al=compute_al(delays, source_length, target_length),
        laal=compute_laal(delays, source_length, target_length),
        yaal=compute_yaal(delays, source_length, target_length, is_longform),
        ap=compute_ap(delays, source_length, target_length),
        dal=compute_dal(delays, source_length, target_length),
        max_cw=compute_max_consecutive_wait(delays),
        n_source=source_length,
        n_target=target_length,
    )


# ---------------------------------------------------------------------------
# Quality metrics (wrappers)
# ---------------------------------------------------------------------------

def compute_bleu(hypothesis: str, reference: str) -> float:
    """Compute sentence-level BLEU using sacrebleu."""
    try:
        import sacrebleu
        result = sacrebleu.sentence_bleu(hypothesis, [reference])
        return result.score
    except ImportError:
        raise ImportError("sacrebleu required for BLEU. Install: pip install sacrebleu")


def compute_bleu_corpus(hypotheses: List[str], references: List[str]) -> float:
    """Compute corpus-level BLEU."""
    try:
        import sacrebleu
        result = sacrebleu.corpus_bleu(hypotheses, [references])
        return result.score
    except ImportError:
        raise ImportError("sacrebleu required for BLEU. Install: pip install sacrebleu")


def compute_comet(
    sources: List[str],
    hypotheses: List[str],
    references: List[str],
    model_name: str = "Unbabel/wmt22-comet-da",
    batch_size: int = 8,
    gpus: int = 1,
) -> Tuple[float, List[float]]:
    """Compute COMET score.

    Args:
        sources: Source texts
        hypotheses: MT outputs
        references: Reference translations
        model_name: COMET model (wmt22-comet-da or Unbabel/XCOMET-XL)
        batch_size: Batch size for inference
        gpus: Number of GPUs (0 for CPU)

    Returns:
        (corpus_score, per_sentence_scores)
    """
    try:
        from comet import download_model, load_from_checkpoint

        model_path = download_model(model_name)
        model = load_from_checkpoint(model_path)

        data = [
            {"src": s, "mt": h, "ref": r}
            for s, h, r in zip(sources, hypotheses, references)
        ]
        output = model.predict(data, batch_size=batch_size, gpus=gpus)
        return output.system_score, output.scores
    except ImportError:
        raise ImportError(
            "comet required for COMET/XCOMET. Install: pip install unbabel-comet"
        )
