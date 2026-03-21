"""SimulMT latency and quality metrics.

Implements the key metrics for evaluating simultaneous translation systems:

Latency metrics:
    - AL (Average Lagging): Standard latency metric from Ma et al. (2019)
    - LAAL (Length-Adaptive AL): Corrected for length differences
    - YAAL: OmniSTEval formula used as IWSLT 2026 primary latency metric
    - LongYAAL: YAAL in longform mode (IWSLT 2026 primary latency metric)
    - LongYAAL_ms: Time-domain LongYAAL (milliseconds, for OmniSTEval)
    - AP (Average Proportion): Proportion of source read before each target word
    - DAL (Differentiable AL): Smooth approximation of AL
    - MaxCW (Max Consecutive Wait): Longest streak without output
    - StreamLAAL: Secondary latency metric for IWSLT 2026

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
    al: float = 0.0           # Average Lagging
    laal: float = 0.0         # Length-Adaptive AL
    yaal: float = 0.0         # OmniSTEval YAAL (shortform)
    longyaal: float = 0.0     # LongYAAL: IWSLT 2026 PRIMARY latency metric
    longyaal_ms: float = 0.0  # LongYAAL in milliseconds (for OmniSTEval output)
    stream_laal: float = 0.0  # StreamLAAL: IWSLT 2026 secondary latency metric
    ap: float = 0.0           # Average Proportion
    dal: float = 0.0          # Differentiable AL
    max_cw: int = 0           # Max Consecutive Wait
    n_source: int = 0         # Source length (words)
    n_target: int = 0         # Target length (words)


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


def compute_longyaal(
    delays: List[float],
    source_length: float,
    target_length: int,
) -> float:
    """LongYAAL -- IWSLT 2026 PRIMARY latency metric (word-count domain).

    LongYAAL = YAAL in longform mode: counts ALL target words, including those
    emitted after the source ends. This is the key metric IWSLT 2026 uses for
    ranking systems within quality tiers.

    Equivalent to compute_yaal(..., is_longform=True).

    From OmniSTEval (arXiv 2509.17349):
        gamma = max(len(delays), target_length) / source_length
        longyaal = (1/tau) * sum_{t=0}^{tau-1} [d_t - t/gamma]

    Args:
        delays: Per-target-word delay values (word-count domain: how many
                source words read before emitting target word t)
        source_length: Number of source words
        target_length: Number of target words

    Returns:
        LongYAAL score in word-count domain (lower is better)
    """
    return compute_yaal(delays, source_length, target_length, is_longform=True)


def compute_longyaal_ms(
    delays_ms: List[float],
    source_length_ms: float,
    target_length: int,
) -> float:
    """LongYAAL in milliseconds -- for OmniSTEval evaluation output.

    IWSLT 2026 evaluation uses time-domain delays (milliseconds) rather than
    word-count delays. This function computes LongYAAL directly from ms delays.

    Args:
        delays_ms: Per-target-word emission timestamps in milliseconds.
                   delays_ms[t] = time when target word t was emitted.
        source_length_ms: Total source audio duration in milliseconds.
        target_length: Number of target words.

    Returns:
        LongYAAL in milliseconds (lower is better).
        Divide by 1000 to get seconds.
    """
    if not delays_ms or source_length_ms == 0:
        return 0.0

    gamma = max(len(delays_ms), target_length) / source_length_ms
    total = 0.0
    tau = 0
    for t, d in enumerate(delays_ms):
        total += d - t / gamma
        tau += 1

    return total / tau if tau > 0 else 0.0


def compute_stream_laal(
    delays: List[float],
    source_length: float,
    target_length: int,
) -> float:
    """StreamLAAL -- IWSLT 2026 secondary latency metric.

    StreamLAAL extends LAAL for streaming scenarios by using monotonized
    delays and handling the case where target length differs from source.

    Formula from OmniSTEval:
        gamma = max(len(delays), target_length) / source_length
        Monotonize: g'(t) = max(g(t), g'(t-1) + 1/gamma)
        stream_laal = (1/tau) * sum [g'(t) - t/gamma] for t where g'(t) < source_length

    Args:
        delays: Per-target-word delay values
        source_length: Source length
        target_length: Target length

    Returns:
        StreamLAAL score (lower is better)
    """
    if not delays or source_length == 0:
        return 0.0

    gamma = max(len(delays), target_length) / source_length

    # Monotonize delays
    mono = [delays[0]]
    step = 1.0 / gamma if gamma > 0 else 0.0
    for t in range(1, len(delays)):
        mono.append(max(delays[t], mono[-1] + step))

    # Sum over tokens where monotonized delay < source_length
    total = 0.0
    tau = 0
    for t, d in enumerate(mono):
        if d >= source_length:
            break
        total += d - t / gamma
        tau += 1

    return total / tau if tau > 0 else 0.0


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
    delays_ms: Optional[List[float]] = None,
    source_length_ms: Optional[float] = None,
) -> LatencyMetrics:
    """Compute all latency metrics at once.

    Args:
        delays: Delay values for each target word (word-count domain)
        source_length: Number of source words
        target_length: Number of target words
        is_longform: Whether to use longform YAAL
        delays_ms: Optional time-domain delays (ms) for LongYAAL_ms
        source_length_ms: Optional source duration (ms) for LongYAAL_ms

    Returns:
        LatencyMetrics dataclass with all values
    """
    longyaal_ms_val = 0.0
    if delays_ms is not None and source_length_ms is not None:
        longyaal_ms_val = compute_longyaal_ms(delays_ms, source_length_ms, target_length)

    return LatencyMetrics(
        al=compute_al(delays, source_length, target_length),
        laal=compute_laal(delays, source_length, target_length),
        yaal=compute_yaal(delays, source_length, target_length, is_longform),
        longyaal=compute_longyaal(delays, source_length, target_length),
        longyaal_ms=longyaal_ms_val,
        stream_laal=compute_stream_laal(delays, source_length, target_length),
        ap=compute_ap(delays, source_length, target_length),
        dal=compute_dal(delays, source_length, target_length),
        max_cw=compute_max_consecutive_wait(delays),
        n_source=source_length,
        n_target=target_length,
    )


# ---------------------------------------------------------------------------
# Quality metrics (wrappers)
# ---------------------------------------------------------------------------

def compute_normalized_erasure(
    revision_history: List[List[int]],
) -> float:
    """Normalized Erasure (NE) -- measures output instability in re-translation.

    From Arivazhagan et al. (2020): NE quantifies how much of the output is
    revised between consecutive re-translations. Lower is better.

    NE = (1/J) * sum_{i=2}^{J} [ |o_{i-1}| - |LCP(o_i, o_{i-1})| ]

    where o_i is the i-th full translation, LCP is the longest common prefix,
    and J is the number of re-translations.

    NE < 0.2 = low revision (< 1 token revised per 5 final tokens)

    Args:
        revision_history: List of full translation token ID lists, one per
            re-translation step. Each entry is the complete translation at
            that point (not just new tokens).

    Returns:
        NE score (0 = perfectly stable, higher = more revisions)
    """
    if len(revision_history) < 2:
        return 0.0

    total_erasure = 0.0
    for i in range(1, len(revision_history)):
        prev = revision_history[i - 1]
        curr = revision_history[i]
        # LCP length
        lcp_len = 0
        for j in range(min(len(prev), len(curr))):
            if prev[j] == curr[j]:
                lcp_len = j + 1
            else:
                break
        erasure = len(prev) - lcp_len
        total_erasure += erasure

    # Normalize by number of revisions
    return total_erasure / (len(revision_history) - 1)


def compute_normalized_erasure_text(
    revision_history: List[str],
) -> float:
    """NE metric for word-level revision history (text strings).

    Same as compute_normalized_erasure but operates on word sequences.

    Args:
        revision_history: List of full translation strings, one per step.

    Returns:
        NE score (0 = perfectly stable)
    """
    if len(revision_history) < 2:
        return 0.0

    total_erasure = 0.0
    for i in range(1, len(revision_history)):
        prev_words = revision_history[i - 1].split()
        curr_words = revision_history[i].split()
        lcp_len = 0
        for j in range(min(len(prev_words), len(curr_words))):
            if prev_words[j] == curr_words[j]:
                lcp_len = j + 1
            else:
                break
        erasure = len(prev_words) - lcp_len
        total_erasure += erasure

    return total_erasure / (len(revision_history) - 1)


def bootstrap_confidence_interval(
    scores: List[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric.

    Uses the percentile method to estimate confidence intervals for the mean
    of per-sentence scores. This is the standard approach used by COMET and
    other MT evaluation tools.

    Args:
        scores: Per-sentence metric scores
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (0.95 = 95% CI)
        seed: Random seed for reproducibility

    Returns:
        (mean, lower_bound, upper_bound) -- the point estimate and CI bounds
    """
    import numpy as np
    rng = np.random.RandomState(seed)

    n = len(scores)
    if n == 0:
        return 0.0, 0.0, 0.0

    scores_arr = np.array(scores, dtype=np.float64)
    mean = float(scores_arr.mean())

    # Bootstrap: resample with replacement, compute mean of each sample
    boot_means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample = rng.choice(scores_arr, size=n, replace=True)
        boot_means[i] = sample.mean()

    # Percentile method
    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return mean, lower, upper


def paired_bootstrap_test(
    scores_a: List[float],
    scores_b: List[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Paired bootstrap significance test between two systems.

    Tests whether system A is significantly better than system B using
    the paired bootstrap resampling test (Koehn, 2004).

    Args:
        scores_a: Per-sentence scores for system A
        scores_b: Per-sentence scores for system B
        n_bootstrap: Number of bootstrap samples
        seed: Random seed

    Returns:
        (delta_mean, p_value) -- mean difference (A-B) and p-value.
        p < 0.05 means the difference is statistically significant.
    """
    import numpy as np
    rng = np.random.RandomState(seed)

    n = len(scores_a)
    if n != len(scores_b):
        raise ValueError(f"Score lists must have same length: {n} vs {len(scores_b)}")
    if n == 0:
        return 0.0, 1.0

    a = np.array(scores_a, dtype=np.float64)
    b = np.array(scores_b, dtype=np.float64)
    deltas = a - b
    observed_delta = float(deltas.mean())

    # Count how many bootstrap samples have delta <= 0 (A not better than B)
    count_not_better = 0
    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        boot_delta = deltas[indices].mean()
        if boot_delta <= 0:
            count_not_better += 1

    p_value = count_not_better / n_bootstrap
    return observed_delta, p_value


def _bleu_tokenize(target_lang: Optional[str] = None) -> str:
    """Pick sacrebleu tokenizer: 'zh' for Chinese/Japanese, default otherwise."""
    if target_lang in ("zh", "ja"):
        return "zh"
    return "13a"


def compute_bleu(hypothesis: str, reference: str,
                 target_lang: Optional[str] = None) -> float:
    """Compute sentence-level BLEU using sacrebleu."""
    try:
        import sacrebleu
        tok = _bleu_tokenize(target_lang)
        result = sacrebleu.sentence_bleu(hypothesis, [reference], tokenize=tok)
        return result.score
    except ImportError:
        raise ImportError("sacrebleu required for BLEU. Install: pip install sacrebleu")


def compute_bleu_corpus(hypotheses: List[str], references: List[str],
                        target_lang: Optional[str] = None) -> float:
    """Compute corpus-level BLEU."""
    try:
        import sacrebleu
        tok = _bleu_tokenize(target_lang)
        result = sacrebleu.corpus_bleu(hypotheses, [references], tokenize=tok)
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
