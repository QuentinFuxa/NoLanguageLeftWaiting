"""AlignAtt border detection engine for simultaneous machine translation.

Core algorithm:
    1. Source words arrive incrementally (from ASR or text)
    2. Build prompt: prefix + context + source + suffix + committed_output
    3. Generate tokens with LLM (greedy argmax)
    4. After each generated token, extract attention from top-K alignment heads
    5. TS-weighted vote: if attended source position >= n_src - border_distance -> STOP
    6. At sentence boundary: commit translation, reset, start next segment

This module contains the pure algorithmic components, independent of the
llama.cpp backend. The AlignAttBackend (in alignatt_backend.py) wires
this together with the llama backend.

Aggregation strategies (for border detection):
    - "ts_vote" (default): TS-weighted argmax vote. Original AlignAtt method.
    - "softmax_mean": Weighted average position using full attention distribution.
      Smoother than argmax -- captures multi-modal attention patterns.
    - "entropy_weighted": Weight heads by inverse attention entropy. Sharp
      attention = reliable head. Adapts per-token instead of using static TS.
    - "consensus": Only count positions where >= K heads agree. More conservative.
    - "geomean": Geometric mean of attention distributions before argmax.
      Requires agreement across heads -- single outlier head can't dominate.
    - "top_p": Cumulative threshold over joint attention. Considers spread
      of attention mass, not just the peak.

Novel aggregation strategies are an open research gap -- no published work
combines multiple strategies or compares them systematically on SimulMT.
"""

import numpy as np
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Aggregation strategies
# ---------------------------------------------------------------------------

def aggregate_ts_weighted_vote(
    src_attn: np.ndarray,
    ts_scores: List[float],
) -> int:
    """Original TS-weighted argmax vote across alignment heads.

    For each head, find the source position with maximum attention weight.
    Then do a weighted vote using each head's TS (translation score) as weight.
    Return the position that gets the highest total weight.

    Args:
        src_attn: (n_heads, n_src) attention weights over source tokens
        ts_scores: TS score for each head (higher = more reliable)

    Returns:
        The source position with the highest weighted vote
    """
    head_argmaxes = np.argmax(src_attn, axis=1)
    weighted = {}
    for h, pos in enumerate(head_argmaxes):
        pos = int(pos)
        weighted[pos] = weighted.get(pos, 0) + ts_scores[h]
    return max(weighted, key=weighted.get)


def aggregate_softmax_mean(
    src_attn: np.ndarray,
    ts_scores: List[float],
) -> float:
    """Softmax-weighted average position across all heads.

    Instead of argmax per head (which discards distribution shape), compute
    the expected source position using the full attention distribution.
    Then take the TS-weighted average across heads.

    This is smoother than argmax voting and captures multi-modal attention
    patterns (e.g., a head attending to positions 5 and 8 contributes 6.5
    rather than picking one).

    Args:
        src_attn: (n_heads, n_src) attention weights
        ts_scores: TS score per head

    Returns:
        Expected attended source position (float)
    """
    n_heads, n_src = src_attn.shape
    positions = np.arange(n_src, dtype=np.float64)

    # Expected position per head: sum(attn * pos)
    expected_pos = src_attn @ positions  # (n_heads,)

    # TS-weighted average
    ts = np.array(ts_scores, dtype=np.float64)
    ts_sum = ts.sum()
    if ts_sum < 1e-10:
        return float(np.mean(expected_pos))
    return float(np.dot(ts, expected_pos) / ts_sum)


def aggregate_entropy_weighted(
    src_attn: np.ndarray,
    ts_scores: List[float],
) -> int:
    """Entropy-weighted voting: weight heads by attention sharpness.

    Instead of using static TS scores, dynamically weight each head by the
    inverse entropy of its attention distribution. A head with sharp attention
    (low entropy) is more reliable for this specific token than a head with
    diffuse attention.

    The final weight is: w_h = ts_h * (1 / (entropy_h + eps))

    This adapts per-token: a head that's sharp on one token but diffuse
    on another will be trusted more when it's sharp.

    Args:
        src_attn: (n_heads, n_src) attention weights
        ts_scores: Base TS scores (still used as prior)

    Returns:
        Attended source position (int)
    """
    n_heads, n_src = src_attn.shape
    eps = 1e-10

    # Compute entropy per head
    entropies = np.zeros(n_heads)
    for h in range(n_heads):
        p = src_attn[h] + eps
        p = p / p.sum()
        entropies[h] = -np.sum(p * np.log(p))

    # Inverse entropy weighting (sharp = high weight)
    inv_entropy = 1.0 / (entropies + eps)

    # Combined weight: TS * inverse_entropy
    combined_w = np.array(ts_scores) * inv_entropy

    # Weighted vote (same as ts_vote but with dynamic weights)
    head_argmaxes = np.argmax(src_attn, axis=1)
    weighted = {}
    for h, pos in enumerate(head_argmaxes):
        pos = int(pos)
        weighted[pos] = weighted.get(pos, 0) + combined_w[h]
    return max(weighted, key=weighted.get)


def aggregate_consensus(
    src_attn: np.ndarray,
    ts_scores: List[float],
    min_heads: int = 3,
) -> int:
    """Consensus aggregation: only trust positions with multi-head agreement.

    A position is only considered if at least `min_heads` heads attend to it
    (or its immediate neighbors). This filters out noisy individual heads
    and only acts on strong signals.

    If no position has enough agreement, falls back to ts_vote.

    Args:
        src_attn: (n_heads, n_src) attention weights
        ts_scores: TS score per head
        min_heads: Minimum number of heads that must agree (default: 3)

    Returns:
        Attended source position (int)
    """
    n_heads, n_src = src_attn.shape
    head_argmaxes = np.argmax(src_attn, axis=1)

    # Count how many heads point at each position (with +/- 1 tolerance)
    pos_votes = {}
    pos_weights = {}
    for h, pos in enumerate(head_argmaxes):
        pos = int(pos)
        # Vote for pos and neighbors (tolerance for subword boundaries)
        for p in range(max(0, pos - 1), min(n_src, pos + 2)):
            pos_votes[p] = pos_votes.get(p, 0) + 1
            pos_weights[p] = pos_weights.get(p, 0) + ts_scores[h]

    # Filter to positions with enough agreement
    consensus_positions = {
        p: pos_weights[p] for p in pos_votes if pos_votes[p] >= min_heads
    }

    if consensus_positions:
        return max(consensus_positions, key=consensus_positions.get)

    # Fallback to standard ts_vote
    return aggregate_ts_weighted_vote(src_attn, ts_scores)


def aggregate_geomean(
    src_attn: np.ndarray,
    ts_scores: List[float],
) -> int:
    """Geometric mean of attention distributions before argmax.

    Takes the geometric mean (TS-weighted) across all heads' attention
    distributions, then argmax on the result. This naturally requires
    agreement: if one head gives 0 attention to a position, the geometric
    mean there is 0 regardless of other heads.

    More robust than arithmetic mean against outlier heads.

    Args:
        src_attn: (n_heads, n_src) attention weights
        ts_scores: TS score per head (used as power weights)

    Returns:
        Attended source position (int)
    """
    n_heads, n_src = src_attn.shape
    eps = 1e-10

    # Normalize each head's attention to sum to 1
    attn_norm = src_attn + eps
    attn_norm = attn_norm / attn_norm.sum(axis=1, keepdims=True)

    # TS-weighted geometric mean: product(attn_h ^ ts_h)
    ts = np.array(ts_scores, dtype=np.float64)
    ts_norm = ts / ts.sum()

    # log-space for numerical stability
    log_attn = np.log(attn_norm)  # (n_heads, n_src)
    weighted_log = ts_norm[:, np.newaxis] * log_attn  # (n_heads, n_src)
    joint_log = weighted_log.sum(axis=0)  # (n_src,)

    return int(np.argmax(joint_log))


def aggregate_top_p(
    src_attn: np.ndarray,
    ts_scores: List[float],
    p_threshold: float = 0.8,
) -> int:
    """Top-p aggregation: find rightmost position within cumulative threshold.

    Merge attention distributions (TS-weighted average), then find the
    rightmost source position within the top-p cumulative mass.
    This captures the attention frontier: how far right the model is looking
    with confidence p.

    Args:
        src_attn: (n_heads, n_src) attention weights
        ts_scores: TS score per head
        p_threshold: Cumulative probability threshold (default: 0.8)

    Returns:
        Rightmost position within top-p attention mass
    """
    n_heads, n_src = src_attn.shape

    # TS-weighted average distribution
    ts = np.array(ts_scores, dtype=np.float64)
    ts_sum = ts.sum()
    if ts_sum < 1e-10:
        merged = src_attn.mean(axis=0)
    else:
        merged = (ts[:, np.newaxis] * src_attn).sum(axis=0) / ts_sum

    # Normalize
    total = merged.sum()
    if total < 1e-10:
        return 0
    merged = merged / total

    # Sort by attention weight descending, accumulate
    sorted_indices = np.argsort(-merged)
    cumsum = 0.0
    top_positions = []
    for idx in sorted_indices:
        cumsum += merged[idx]
        top_positions.append(int(idx))
        if cumsum >= p_threshold:
            break

    # Return the rightmost position in the top-p set
    return max(top_positions)


def aggregate_top_p_weighted(
    src_attn: np.ndarray,
    ts_scores: List[float],
    p_threshold: float = 0.8,
) -> float:
    """Top-p weighted frontier: attention-weighted mean of top-p positions.

    Like top_p but returns a continuous position (weighted mean of top-p set
    positions by their attention mass) instead of just the rightmost.

    This captures the center-of-mass of where the model is attending within
    the top-p set. More robust than max (less sensitive to outlier positions)
    while still being more conservative than argmax (includes the frontier).

    The returned value is continuous and can be compared to border_threshold.

    Args:
        src_attn: (n_heads, n_src) attention weights
        ts_scores: TS score per head
        p_threshold: Cumulative probability threshold (default: 0.8)

    Returns:
        Attention-weighted mean position within top-p set (continuous)
    """
    n_heads, n_src = src_attn.shape

    # TS-weighted average distribution
    ts = np.array(ts_scores, dtype=np.float64)
    ts_sum = ts.sum()
    if ts_sum < 1e-10:
        merged = src_attn.mean(axis=0)
    else:
        merged = (ts[:, np.newaxis] * src_attn).sum(axis=0) / ts_sum

    # Normalize
    total = merged.sum()
    if total < 1e-10:
        return 0.0
    merged = merged / total

    # Sort by attention weight descending, accumulate
    sorted_indices = np.argsort(-merged)
    cumsum = 0.0
    top_positions = []
    top_weights = []
    for idx in sorted_indices:
        cumsum += merged[idx]
        top_positions.append(int(idx))
        top_weights.append(merged[idx])
        if cumsum >= p_threshold:
            break

    # Return weighted mean position
    positions = np.array(top_positions, dtype=np.float64)
    weights = np.array(top_weights, dtype=np.float64)
    return float(np.average(positions, weights=weights))


# ---------------------------------------------------------------------------
# Aggregation registry
# ---------------------------------------------------------------------------

def aggregate_gaussian_kernel(
    src_attn: np.ndarray,
    ts_scores: List[float],
    sigma: float = 1.5,
) -> float:
    """Gaussian kernel consensus: smooth density over attended positions.

    Novel aggregation that unifies ts_vote (sigma->0) and softmax_mean
    (sigma->inf) with a single bandwidth parameter:
        1. Each head votes for its argmax position
        2. Each vote is a Gaussian kernel centered at that position
        3. The kernels are weighted by TS scores
        4. Return the position with maximum density

    With small sigma, this approaches argmax voting (ts_vote).
    With large sigma, the density becomes uniform (like mean position).
    Intermediate sigma captures partial agreement: nearby heads reinforce
    each other, distant heads cancel out.

    Advantage over ts_vote: subword boundary tolerance (heads attending
    to adjacent subwords contribute to the same region).
    Advantage over softmax_mean: still produces a clear peak (not washed
    out by diffuse attention).

    Args:
        src_attn: (n_heads, n_src) attention weights
        ts_scores: TS score per head
        sigma: Kernel bandwidth. Recommended range: 0.5-3.0.
            0.5 = nearly argmax. 1.5 = moderate smoothing. 3.0 = wide blur.

    Returns:
        Position with maximum kernel density (float, can be fractional)
    """
    n_heads, n_src = src_attn.shape
    if n_src == 0:
        return 0.0

    head_argmaxes = np.argmax(src_attn, axis=1)
    positions = np.arange(n_src, dtype=np.float64)

    # Build kernel density: sum of TS-weighted Gaussians centered at each head's argmax
    density = np.zeros(n_src, dtype=np.float64)
    for h in range(n_heads):
        center = float(head_argmaxes[h])
        # Gaussian kernel: exp(-0.5 * ((x - center) / sigma)^2)
        kernel = np.exp(-0.5 * ((positions - center) / sigma) ** 2)
        density += ts_scores[h] * kernel

    return float(np.argmax(density))


def aggregate_gaussian_kernel_continuous(
    src_attn: np.ndarray,
    ts_scores: List[float],
    sigma: float = 1.5,
) -> float:
    """Like gaussian_kernel but uses full attention distributions, not just argmax.

    Instead of placing a Gaussian at each head's argmax, we convolve each
    head's full attention distribution with a Gaussian kernel, then take
    the TS-weighted sum. This preserves multi-modal attention patterns.

    Args:
        src_attn: (n_heads, n_src) attention weights
        ts_scores: TS score per head
        sigma: Kernel bandwidth

    Returns:
        Position with maximum smoothed density (float)
    """
    n_heads, n_src = src_attn.shape
    if n_src == 0:
        return 0.0

    positions = np.arange(n_src, dtype=np.float64)

    # Build Gaussian kernel matrix: K[i,j] = exp(-0.5 * ((i-j)/sigma)^2)
    diff = positions[:, np.newaxis] - positions[np.newaxis, :]
    kernel_matrix = np.exp(-0.5 * (diff / sigma) ** 2)

    # Convolve each head's attention with the kernel, weight by TS
    smoothed = np.zeros(n_src, dtype=np.float64)
    for h in range(n_heads):
        # attn[h] @ kernel_matrix = smoothed distribution for head h
        head_smoothed = src_attn[h] @ kernel_matrix
        smoothed += ts_scores[h] * head_smoothed

    return float(np.argmax(smoothed))


def normalize_head_temperatures(
    src_attn: np.ndarray,
    reference_entropy: float = 1.5,
) -> np.ndarray:
    """Normalize attention distributions to have uniform sharpness across heads.

    Different attention heads have different "temperatures" -- some produce
    sharp distributions (low entropy) and some produce broad ones (high entropy).
    This imbalance means sharp heads dominate argmax-based aggregation regardless
    of their actual reliability.

    This function rescales each head's attention distribution to match a target
    reference entropy, ensuring fair comparison in downstream aggregation.

    Method: For each head, compute its entropy. If entropy < reference, soften
    the distribution (raise to power < 1). If entropy > reference, sharpen it
    (raise to power > 1). The power is calibrated to approximately match the
    reference entropy.

    Args:
        src_attn: (n_heads, n_src) attention weights
        reference_entropy: Target entropy in nats (default: 1.5)
            Lower = sharper normalized attention. Higher = broader.
            1.5 nats ~ uniform over ~4.5 positions (exp(1.5)).

    Returns:
        Normalized attention weights (n_heads, n_src), same shape
    """
    n_heads, n_src = src_attn.shape
    if n_src <= 1:
        return src_attn.copy()

    eps = 1e-10
    result = np.zeros_like(src_attn)

    for h in range(n_heads):
        p = src_attn[h] + eps
        p = p / p.sum()
        head_entropy = float(-np.sum(p * np.log(p)))

        if abs(head_entropy - reference_entropy) < 0.05 or head_entropy < eps:
            result[h] = p
            continue

        # Use logits to rescale temperature.
        # Convert probs back to logits, rescale, then softmax.
        # If head_entropy > reference: need to sharpen (divide logits by T<1)
        # If head_entropy < reference: need to broaden (divide logits by T>1)
        log_p = np.log(p)
        log_p = log_p - np.mean(log_p)  # center for numerical stability

        # Binary search for temperature that gives reference entropy
        # Start with analytical estimate
        t_lo, t_hi = 0.01, 50.0
        for _ in range(30):
            t_mid = (t_lo + t_hi) / 2
            scaled = log_p / t_mid
            scaled = scaled - np.max(scaled)
            q = np.exp(scaled)
            q = q / q.sum()
            ent = float(-np.sum(q * np.log(q + eps)))
            if ent < reference_entropy:
                t_lo = t_mid  # need more temperature (broaden)
            else:
                t_hi = t_mid  # need less temperature (sharpen)

        # Apply final temperature
        t_final = (t_lo + t_hi) / 2
        scaled = log_p / t_final
        scaled = scaled - np.max(scaled)
        p_rescaled = np.exp(scaled)
        p_rescaled = p_rescaled / p_rescaled.sum()
        result[h] = p_rescaled

    return result


def select_adaptive_aggregation(
    src_attn: np.ndarray,
    ts_scores: List[float],
) -> str:
    """Adaptive Multi-Strategy (AMS): select the best aggregation method
    based on the current attention pattern.

    Novel approach: no published work on per-token aggregation selection
    for SimulMT. We analyze two signals:

    1. Head agreement ratio: fraction of heads whose argmax is within +/-1
       of the majority argmax. High agreement -> simple methods work.
       Low agreement -> need robust methods.

    2. Mean attention entropy: how diffuse the attention is across heads.
       Low entropy -> heads are confident -> trust argmax methods.
       High entropy -> heads are uncertain -> use distribution-aware methods.

    Selection logic:
        agreement >= 0.7 AND entropy <= 1.0: ts_vote (all agree, sharp)
        agreement >= 0.7 AND entropy > 1.0:  entropy_weighted (agree, but diffuse)
        agreement < 0.7  AND entropy <= 1.5: geomean (disagreement, but sharp)
        agreement < 0.7  AND entropy > 1.5:  consensus (disagreement + diffuse)

    Args:
        src_attn: (n_heads, n_src) attention weights
        ts_scores: TS score per head

    Returns:
        Name of the selected aggregation method
    """
    n_heads, n_src = src_attn.shape
    if n_heads == 0 or n_src == 0:
        return "ts_vote"

    eps = 1e-10

    # 1. Head agreement ratio
    head_argmaxes = np.argmax(src_attn, axis=1)
    # Find the most common argmax (mode)
    counts = {}
    for pos in head_argmaxes:
        pos = int(pos)
        counts[pos] = counts.get(pos, 0) + 1
    mode_pos = max(counts, key=counts.get)

    # Count heads within +/-1 of mode
    agreeing = sum(1 for pos in head_argmaxes if abs(int(pos) - mode_pos) <= 1)
    agreement_ratio = agreeing / n_heads

    # 2. Mean attention entropy (TS-weighted)
    entropies = np.zeros(n_heads)
    for h in range(n_heads):
        p = src_attn[h] + eps
        p = p / p.sum()
        entropies[h] = -np.sum(p * np.log(p))

    ts = np.array(ts_scores, dtype=np.float64)
    ts_sum = ts.sum()
    if ts_sum < eps:
        mean_entropy = float(np.mean(entropies))
    else:
        mean_entropy = float(np.dot(ts, entropies) / ts_sum)

    # 3. Selection
    if agreement_ratio >= 0.7:
        if mean_entropy <= 1.0:
            return "ts_vote"
        else:
            return "entropy_weighted"
    else:
        if mean_entropy <= 1.5:
            return "geomean"
        else:
            return "consensus"


def aggregate_ensemble(
    src_attn: np.ndarray,
    ts_scores: List[float],
    methods: Optional[List[str]] = None,
    weights: Optional[List[float]] = None,
) -> float:
    """Ensemble aggregation: weighted average of multiple methods.

    Combines outputs from multiple aggregation strategies. Each method
    returns a position; we take the weighted average.

    Novel approach: no published work combines aggregation methods for SimulMT.

    Default ensemble: ts_vote (0.4) + entropy_weighted (0.3) + geomean (0.3)
    This balances: established method + adaptive sharpness + cross-head agreement.

    Args:
        src_attn: (n_heads, n_src) attention weights
        ts_scores: TS score per head
        methods: List of method names (default: ["ts_vote", "entropy_weighted", "geomean"])
        weights: Per-method weights (default: [0.4, 0.3, 0.3])

    Returns:
        Weighted average attended position (float)
    """
    if methods is None:
        methods = ["ts_vote", "entropy_weighted", "geomean"]
    if weights is None:
        weights = [0.4, 0.3, 0.3]

    assert len(methods) == len(weights), "methods and weights must have same length"

    positions = []
    for method in methods:
        if method not in _BASE_AGGREGATION_METHODS:
            raise ValueError(f"Unknown method '{method}' in ensemble")
        pos = _BASE_AGGREGATION_METHODS[method](src_attn, ts_scores)
        positions.append(float(pos))

    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()
    return float(np.dot(w, positions))


def aggregate_cumulative_attention(
    src_attn: np.ndarray,
    ts_scores: List[float],
    lambda_threshold: float = 0.5,
) -> int:
    """Cumulative attention aggregation: DrFrattn-inspired border detection.

    Adapted from DrFrattn (EMNLP 2025, Zhao et al.). Instead of checking where
    the attention argmax falls, we compute the cumulative attention mass from
    left to right. The "attended position" is the rightmost position where
    significant attention mass still remains to the right.

    Formally: for each head, compute c_j = 1 - cumsum(attn[0:j+1]).
    The "frontier" is the last position j where c_j >= lambda_threshold,
    meaning the model still has (lambda) fraction of mass beyond position j.

    This captures the distribution SHAPE:
    - Sharp attention at pos 5: frontier is exactly at 5
    - Diffuse attention spread over 3-7: frontier depends on lambda
    - Multi-modal attention (peaks at 2 and 6): captures the rightmost mode

    Key insight: a model that splits attention 40%/40% between positions 5 and 7
    has frontier at 7, while argmax might pick either. Cumulative correctly
    identifies the generation frontier.

    Args:
        src_attn: (n_heads, n_src) attention weights
        ts_scores: TS score per head
        lambda_threshold: Remaining mass threshold (0.0-1.0, default: 0.5).
            Lower = more aggressive (earlier frontier).
            Higher = more conservative (later frontier).

    Returns:
        Attended source position (int): the attention frontier
    """
    n_heads, n_src = src_attn.shape
    if n_src == 0:
        return 0

    eps = 1e-10
    head_positions = []

    for h in range(n_heads):
        p = src_attn[h] + eps
        p = p / p.sum()
        cumsum = np.cumsum(p)
        remaining = 1.0 - cumsum  # mass remaining to the right

        # Find rightmost position where remaining mass >= lambda
        candidates = np.where(remaining >= lambda_threshold)[0]
        if len(candidates) > 0:
            head_positions.append(int(candidates[-1]))
        else:
            # All mass is concentrated at the start
            head_positions.append(0)

    # TS-weighted vote on frontier positions
    weighted = {}
    for h, pos in enumerate(head_positions):
        weighted[pos] = weighted.get(pos, 0) + ts_scores[h]
    return max(weighted, key=weighted.get)


# Base methods (before ensemble to avoid circular reference)
_BASE_AGGREGATION_METHODS = {
    "ts_vote": aggregate_ts_weighted_vote,
    "softmax_mean": aggregate_softmax_mean,
    "entropy_weighted": aggregate_entropy_weighted,
    "consensus": aggregate_consensus,
    "geomean": aggregate_geomean,
    "top_p": aggregate_top_p,
    "top_p_weighted": aggregate_top_p_weighted,
    "gaussian_kernel": aggregate_gaussian_kernel,
    "gaussian_kernel_continuous": aggregate_gaussian_kernel_continuous,
    "cumulative": aggregate_cumulative_attention,
}

_AGGREGATION_METHODS = {
    **_BASE_AGGREGATION_METHODS,
    "ensemble": aggregate_ensemble,
}


def list_aggregation_methods() -> List[str]:
    """List available aggregation method names."""
    return sorted(_AGGREGATION_METHODS.keys())


def aggregate(
    src_attn: np.ndarray,
    ts_scores: List[float],
    method: str = "ts_vote",
    **kwargs,
) -> float:
    """Unified aggregation dispatcher.

    Args:
        src_attn: (n_heads, n_src) attention weights
        ts_scores: TS score per head
        method: Aggregation method name
        **kwargs: Extra arguments forwarded to the aggregation function
            (e.g., p_threshold for top_p)

    Returns:
        Attended source position (float for softmax_mean, int for others)
    """
    if method not in _AGGREGATION_METHODS:
        raise ValueError(
            f"Unknown aggregation '{method}'. "
            f"Available: {list_aggregation_methods()}"
        )
    fn = _AGGREGATION_METHODS[method]
    # Forward kwargs that the function accepts (e.g., p_threshold for top_p)
    import inspect
    sig = inspect.signature(fn)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(src_attn, ts_scores, **valid_kwargs)


def check_border(
    src_attn: np.ndarray,
    ts_scores: List[float],
    n_src_tokens: int,
    border_distance: int,
    aggregation: str = "ts_vote",
    adaptive_aggregation: bool = False,
    head_temp_normalize: bool = False,
    head_temp_reference: float = 1.5,
    top_p_threshold: float = 0.8,
) -> bool:
    """Check if the attention has reached the border region.

    The border region starts at position (n_src - border_distance).
    When attention focuses here, it means the model is "looking at" the
    end of available source, so we should stop generating and wait for
    more source input.

    Args:
        src_attn: (n_heads, n_src) attention weights
        ts_scores: TS score per head
        n_src_tokens: Number of source tokens
        border_distance: How many tokens from the end define the border
        aggregation: Aggregation method (default: "ts_vote")
        adaptive_aggregation: If True, auto-select aggregation per token (AMS)
        head_temp_normalize: If True, normalize head temperatures before aggregation
        head_temp_reference: Reference entropy for temperature normalization
        top_p_threshold: Cumulative threshold for top_p aggregation (0.0-1.0)

    Returns:
        True if border hit (should stop generating)
    """
    border_threshold = n_src_tokens - border_distance
    if border_threshold <= 0:
        return False

    attn = src_attn
    if head_temp_normalize:
        attn = normalize_head_temperatures(attn, head_temp_reference)

    method = aggregation
    if adaptive_aggregation:
        method = select_adaptive_aggregation(attn, ts_scores)

    attended_pos = aggregate(attn, ts_scores, method=method,
                             p_threshold=top_p_threshold)
    return attended_pos >= border_threshold


def attention_entropy(src_attn: np.ndarray, ts_scores: List[float]) -> float:
    """Compute TS-weighted mean entropy of attention distributions.

    High attention entropy means heads disagree or are diffuse about where
    to attend. This signals uncertainty about source alignment.

    Args:
        src_attn: (n_heads, n_src) attention weights
        ts_scores: TS score per head

    Returns:
        Weighted mean entropy (nats). Range: [0, ln(n_src)]
    """
    n_heads, n_src = src_attn.shape
    eps = 1e-10

    entropies = np.zeros(n_heads)
    for h in range(n_heads):
        p = src_attn[h] + eps
        p = p / p.sum()
        entropies[h] = -np.sum(p * np.log(p))

    ts = np.array(ts_scores, dtype=np.float64)
    ts_sum = ts.sum()
    if ts_sum < eps:
        return float(np.mean(entropies))
    return float(np.dot(ts, entropies) / ts_sum)


def merged_attention_entropy(
    src_attn: np.ndarray,
    ts_scores: List[float],
) -> float:
    """Compute entropy of the TS-weighted MERGED attention distribution.

    Unlike attention_entropy() which averages per-head entropies, this first
    merges all heads into a single distribution (TS-weighted average) and then
    computes its entropy. This captures the ensemble's collective uncertainty
    about source position, which is what the top_p aggregation operates on.

    Low entropy = focused merged attention (model confident about source position)
    High entropy = spread merged attention (model uncertain)

    Args:
        src_attn: (n_heads, n_src) attention weights
        ts_scores: TS score per head

    Returns:
        Entropy in nats. Range: [0, ln(n_src)]
    """
    eps = 1e-10
    ts = np.array(ts_scores, dtype=np.float64)
    ts_sum = ts.sum()
    if ts_sum < eps:
        merged = src_attn.mean(axis=0)
    else:
        merged = (ts[:, np.newaxis] * src_attn).sum(axis=0) / ts_sum

    total = merged.sum()
    if total < eps:
        return 0.0
    merged = merged / total

    # Shannon entropy in nats (base e)
    merged = merged[merged > eps]
    return float(-np.sum(merged * np.log(merged)))


def entropy_gated_top_p_threshold(
    base_threshold: float,
    attention_entropy: float,
    low_entropy: float = 1.0,
    high_entropy: float = 2.5,
    low_scale: float = 0.88,
    high_scale: float = 1.08,
) -> float:
    """Adjust top_p threshold based on merged attention entropy.

    Novel technique: modulate the top_p frontier sensitivity on a per-token
    basis using the ensemble attention distribution's entropy.

    Low entropy (focused attention) -> scale DOWN threshold -> tighter frontier
    -> emit sooner (lower YAAL). The model is confident about source position,
    so we trust smaller cumulative mass for the border check.

    High entropy (spread attention) -> scale UP threshold -> wider frontier
    -> wait longer (better quality). The model is uncertain, so we require
    more cumulative mass before declaring a border hit.

    The scaling is linear between low_entropy and high_entropy, with the
    identity (scale=1.0) in the middle. This ensures smooth modulation.

    Args:
        base_threshold: Base top_p threshold (e.g. 0.85)
        attention_entropy: Entropy of merged attention (nats), from
            merged_attention_entropy()
        low_entropy: Below this -> minimum scale (default 1.0 nats)
        high_entropy: Above this -> maximum scale (default 2.5 nats)
        low_scale: Scale factor at low entropy (default 0.88, i.e. -12%)
        high_scale: Scale factor at high entropy (default 1.08, i.e. +8%)

    Returns:
        Adjusted top_p threshold, clamped to [0.5, 0.99]
    """
    if attention_entropy <= low_entropy:
        scale = low_scale
    elif attention_entropy >= high_entropy:
        scale = high_scale
    else:
        # Linear interpolation
        t = (attention_entropy - low_entropy) / (high_entropy - low_entropy)
        scale = low_scale + t * (high_scale - low_scale)

    return max(0.5, min(0.99, base_threshold * scale))


def dynamic_border_distance(
    src_attn: np.ndarray,
    ts_scores: List[float],
    base_bd: int,
    n_src_tokens: int,
    low_entropy: float = 0.5,
    high_entropy: float = 2.0,
    min_bd: int = 1,
    max_bd_delta: int = 3,
) -> int:
    """Compute dynamic border distance based on attention entropy.

    Novel approach: instead of a fixed border distance, adjust per-token
    based on how uncertain the attention is:
    - Sharp attention (low entropy) -> model knows where to look -> tighter border
    - Diffuse attention (high entropy) -> uncertain -> wider border (be cautious)

    The mapping is linear between low_entropy and high_entropy thresholds:
        ent <= low_entropy  ->  bd = base_bd - 1 (aggressive)
        ent >= high_entropy ->  bd = base_bd + max_bd_delta (conservative)
        between             ->  linear interpolation

    Args:
        src_attn: (n_heads, n_src) attention weights
        ts_scores: TS score per head
        base_bd: Base border distance (from config)
        n_src_tokens: Number of source tokens
        low_entropy: Entropy threshold for "confident" (default: 0.5 nats)
        high_entropy: Entropy threshold for "uncertain" (default: 2.0 nats)
        min_bd: Minimum border distance (default: 1)
        max_bd_delta: Maximum increase from base (default: 3)

    Returns:
        Effective border distance (always >= min_bd)
    """
    ent = attention_entropy(src_attn, ts_scores)

    if ent <= low_entropy:
        delta = -1  # slightly tighter
    elif ent >= high_entropy:
        delta = max_bd_delta
    else:
        # Linear interpolation
        ratio = (ent - low_entropy) / (high_entropy - low_entropy)
        delta = int(round(-1 + (max_bd_delta + 1) * ratio))

    effective_bd = base_bd + delta
    return max(min_bd, min(effective_bd, n_src_tokens - 1))


def check_border_dynamic(
    src_attn: np.ndarray,
    ts_scores: List[float],
    n_src_tokens: int,
    base_border_distance: int,
    aggregation: str = "ts_vote",
    low_entropy: float = 0.5,
    high_entropy: float = 2.0,
    max_bd_delta: int = 3,
    adaptive_aggregation: bool = False,
    head_temp_normalize: bool = False,
    head_temp_reference: float = 1.5,
    top_p_threshold: float = 0.8,
) -> bool:
    """Check border with dynamic border distance based on attention entropy.

    Combines aggregation-based border detection with entropy-based adaptation.
    When attention is uncertain, the border widens (more conservative).
    When attention is sharp, the border tightens (more aggressive).

    Args:
        src_attn: (n_heads, n_src) attention weights
        ts_scores: TS score per head
        n_src_tokens: Number of source tokens
        base_border_distance: Base border distance from config
        aggregation: Aggregation method (default: "ts_vote")
        low_entropy: Confident threshold
        high_entropy: Uncertain threshold
        max_bd_delta: Max border widening
        adaptive_aggregation: If True, auto-select aggregation per token (AMS)
        head_temp_normalize: If True, normalize head temperatures before aggregation
        head_temp_reference: Reference entropy for temperature normalization

    Returns:
        True if border hit (should stop generating)
    """
    attn = src_attn
    if head_temp_normalize:
        attn = normalize_head_temperatures(attn, head_temp_reference)

    effective_bd = dynamic_border_distance(
        attn, ts_scores, base_border_distance, n_src_tokens,
        low_entropy=low_entropy, high_entropy=high_entropy,
        max_bd_delta=max_bd_delta,
    )

    border_threshold = n_src_tokens - effective_bd
    if border_threshold <= 0:
        return False

    method = aggregation
    if adaptive_aggregation:
        method = select_adaptive_aggregation(attn, ts_scores)

    attended_pos = aggregate(attn, ts_scores, method=method,
                             p_threshold=top_p_threshold)
    return attended_pos >= border_threshold


def compute_entropy(logits: np.ndarray) -> float:
    """Compute entropy of a logit distribution (for entropy veto).

    High entropy = model is uncertain about next token.
    Can be used to veto low-confidence generations.

    Args:
        logits: Raw logits array (n_vocab,)

    Returns:
        Entropy in nats
    """
    # Stable softmax
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum()
    # Entropy: -sum(p * log(p)), ignoring zeros
    log_probs = np.log(probs + 1e-10)
    return float(-np.sum(probs * log_probs))


def compute_token_perplexity(logits: np.ndarray, token_id: int) -> float:
    """Compute perplexity for a specific generated token.

    Perplexity = exp(-log P(token)). Low perplexity means the model was
    confident about this token. High perplexity means uncertainty.

    Used by perplexity-based adaptive border (Hibiki-inspired):
    track running perplexity during generation to decide whether to
    widen or tighten the border distance for the next translate() call.

    Args:
        logits: Raw logits array (n_vocab,)
        token_id: The token that was actually generated

    Returns:
        Token-level perplexity (>= 1.0)
    """
    # Stable softmax
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum()
    # P(token) -> perplexity
    p = float(probs[token_id]) if token_id < len(probs) else 1e-10
    p = max(p, 1e-10)  # Guard against zero
    return float(np.exp(-np.log(p)))


def compute_generation_perplexity(
    token_perplexities: List[float],
) -> float:
    """Compute average perplexity over a sequence of generated tokens.

    Used to determine generation confidence for the entire word batch.
    Geometric mean of per-token perplexities (= exp of mean log-probs).

    Args:
        token_perplexities: Per-token perplexity values from compute_token_perplexity

    Returns:
        Average (geometric mean) perplexity. 1.0 = perfect confidence.
    """
    if not token_perplexities:
        return 1.0
    # Geometric mean via log domain
    log_ppls = [np.log(max(p, 1.0)) for p in token_perplexities]
    return float(np.exp(sum(log_ppls) / len(log_ppls)))


def perplexity_border_adjustment(
    avg_perplexity: float,
    base_bd: int,
    low_threshold: float = 2.0,
    high_threshold: float = 5.0,
) -> int:
    """Compute border distance adjustment from generation perplexity.

    Maps generation confidence to a border distance delta:
    - Very confident (ppl < low) -> bd - 1 (lower latency)
    - Confident (low <= ppl <= high) -> bd (no change)
    - Uncertain (ppl > high) -> bd + 1 (better quality)

    Hibiki-inspired: unlike entropy veto (which stops generation, a dead end),
    this adjusts the READ/WRITE policy between steps, targeting YAAL latency.

    Args:
        avg_perplexity: Average generation perplexity from compute_generation_perplexity()
        base_bd: Base border_distance from config
        low_threshold: Below this = confident -> tighten border
        high_threshold: Above this = uncertain -> widen border

    Returns:
        Adjusted border_distance (clamped to >= 1)
    """
    if avg_perplexity < low_threshold:
        return max(1, base_bd - 1)
    elif avg_perplexity > high_threshold:
        return base_bd + 1
    return base_bd


def source_lookahead_top_prob(logits: np.ndarray) -> Tuple[float, int]:
    """Compute top-token probability from raw logits.

    Used by TAF (Translation by Anticipating Future) source lookahead:
    high top-prob means the model confidently predicts the next source token,
    so translation should be deferred to accumulate more source context.

    Args:
        logits: Raw logits array (n_vocab,)

    Returns:
        (top_probability, top_token_index)
    """
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum()
    top_idx = int(np.argmax(probs))
    return float(probs[top_idx]), top_idx


def is_target_language(text: str, target_lang: str) -> bool:
    """Check if text appears to be in the target language.

    Used to validate context entries -- if the model outputted English
    instead of the target language, we don't want to inject that as context.

    Args:
        text: The text to check
        target_lang: Target language code (zh, de, ja, ar, ru, fr, it, etc.)

    Returns:
        True if text appears to be in the target language
    """
    if not text.strip():
        return False

    if target_lang == "zh":
        zh_chars = sum(1 for c in text if 0x4E00 <= ord(c) <= 0x9FFF)
        en_chars = sum(1 for c in text if ord(c) < 128 and c.isalpha())
        return zh_chars > en_chars
    elif target_lang == "ja":
        ja_chars = sum(1 for c in text if 0x3040 <= ord(c) <= 0x30FF or 0x4E00 <= ord(c) <= 0x9FFF)
        en_chars = sum(1 for c in text if ord(c) < 128 and c.isalpha())
        return ja_chars > en_chars
    elif target_lang in ("ar", "fa"):
        ar_chars = sum(1 for c in text if 0x0600 <= ord(c) <= 0x06FF)
        en_chars = sum(1 for c in text if ord(c) < 128 and c.isalpha())
        return ar_chars > en_chars
    elif target_lang == "ru":
        ru_chars = sum(1 for c in text if 0x0400 <= ord(c) <= 0x04FF)
        en_chars = sum(1 for c in text if ord(c) < 128 and c.isalpha())
        return ru_chars > en_chars
    else:
        # Latin-script languages (de, fr, it, pt, nl, tr, etc.)
        # Can't distinguish from English by script alone -- always accept
        return True


def compute_dynamic_word_batch(
    base_wb: int,
    n_source_words: int,
    short_threshold: int = 8,
    long_threshold: int = 20,
) -> int:
    """Compute dynamic word_batch based on source sentence length.

    Short sentences benefit from smaller batches (lower latency) while
    long sentences benefit from larger batches (safer, less hallucination).

    Args:
        base_wb: Base word_batch from config
        n_source_words: Number of source words seen so far
        short_threshold: Sentences shorter than this get wb-1 (default: 8)
        long_threshold: Sentences longer than this get wb+1 (default: 20)

    Returns:
        Effective word_batch (always >= 1)
    """
    if n_source_words < short_threshold:
        return max(1, base_wb - 1)
    elif n_source_words > long_threshold:
        return base_wb + 1
    return base_wb


def confidence_adaptive_word_batch(
    base_wb: int,
    prev_avg_logprob: Optional[float],
    high_threshold: float = -0.5,
    low_threshold: float = -2.0,
) -> int:
    """Adjust word_batch based on generation confidence from previous step.

    Novel: no published work on confidence-based batch size adaptation for
    simultaneous machine translation. Uses avg_logprob (from iteration 22)
    to dynamically adapt how many source words we wait for before translating.

    Confident generation (high logprob) -> model has enough context, emit sooner
    (reduce wb by 1 for lower YAAL latency).
    Uncertain generation (low logprob) -> model needs more context, wait longer
    (increase wb by 1 for better quality).

    Args:
        base_wb: Base word_batch from config (may already be adjusted by
            dynamic_word_batch or complexity_adaptive).
        prev_avg_logprob: Average log-probability from previous translate() call.
            None on first call (no adjustment). Range: typically -5.0 to 0.0.
        high_threshold: Above this -> reduce wb (confident). Default -0.5.
        low_threshold: Below this -> increase wb (uncertain). Default -2.0.

    Returns:
        Effective word_batch (always >= 1).
    """
    if prev_avg_logprob is None:
        return base_wb
    if prev_avg_logprob > high_threshold:
        return max(1, base_wb - 1)
    elif prev_avg_logprob < low_threshold:
        return base_wb + 1
    return base_wb


# Language-pair token compression ratios (target tokens / source tokens).
# These are empirical averages from FLORES with HY-MT1.5-7B subword tokenizer.
# Used to set tighter generation caps per language pair.
_LANG_PAIR_RATIOS = {
    ("en", "zh"): 0.85,   # Chinese is more compact (characters vs words)
    ("en", "de"): 1.15,   # German compounds are longer
    ("en", "it"): 1.10,   # Italian is slightly longer
    ("en", "fr"): 1.10,   # French is slightly longer
    ("en", "es"): 1.10,   # Spanish is slightly longer
    ("en", "ja"): 0.80,   # Japanese is more compact
    ("en", "ko"): 0.85,   # Korean is more compact
    ("cs", "en"): 1.10,   # Czech->English slightly expands
    ("de", "en"): 0.90,   # German->English compresses
    ("zh", "en"): 1.20,   # Chinese->English expands
}


def language_pair_gen_cap(
    n_src_tokens: int,
    src_lang: str,
    tgt_lang: str,
    min_cap: int = 3,
) -> int:
    """Compute language-pair-aware generation cap.

    Different language pairs have different output/input token ratios.
    EN->ZH produces fewer tokens (Chinese is compact), while EN->DE produces
    more (German compounds). This adjusts the per-step generation cap to
    avoid both overgeneration (waste, hallucination risk) and undergeneration
    (cut-off translations).

    Args:
        n_src_tokens: Number of source tokens in the current batch.
        src_lang: Source language code (e.g., "en", "cs").
        tgt_lang: Target language code (e.g., "zh", "de").
        min_cap: Minimum generation cap (default 3).

    Returns:
        Generation cap (number of tokens to generate at most).
    """
    ratio = _LANG_PAIR_RATIOS.get((src_lang, tgt_lang), 1.0)
    # Apply ratio with a safety margin of 1.3x (allow some overgeneration)
    cap = int(n_src_tokens * ratio * 1.3)
    return max(min_cap, cap)


# Common English function words that shouldn't end a translation unit.
# If the batch ends on one of these, wait for one more content word.
_EN_FUNCTION_WORDS = frozenset({
    # Determiners
    "the", "a", "an", "this", "that", "these", "those", "my", "your",
    "his", "her", "its", "our", "their", "some", "any", "no", "every",
    # Prepositions
    "of", "in", "to", "for", "with", "on", "at", "from", "by", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "along", "until", "upon",
    # Conjunctions
    "and", "or", "but", "nor", "yet", "so",
    # Auxiliaries / particles
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "can", "could",
    "must", "not", "n't",
    # Pronouns (when they start a clause, useful to wait for verb)
    "i", "you", "he", "she", "it", "we", "they", "who", "which", "that",
    # Relative / subordinating
    "if", "when", "while", "because", "although", "since", "unless", "whether",
})

# Czech function words (for CS-EN source side)
_CS_FUNCTION_WORDS = frozenset({
    "a", "i", "v", "na", "se", "je", "z", "s", "k", "o", "do", "za",
    "pro", "ze", "ve", "po", "od", "to", "ale", "jako", "tak", "jak",
    "ani", "nebo", "aby", "kdyz", "ze", "ktery", "jez", "ten", "ta",
})


def should_defer_batch(
    last_word: str,
    source_lang: str = "en",
    max_defer: int = 2,
    deferred_count: int = 0,
) -> bool:
    """Check if translation should be deferred because the batch ends on a function word.

    Improves translation quality by ensuring translation units don't end
    with dangling function words (e.g., "the", "of", "in") that carry
    little semantic content on their own.

    Args:
        last_word: The last source word in the current batch
        source_lang: Source language code ("en" or "cs")
        max_defer: Maximum extra words to wait (prevents unbounded latency)
        deferred_count: How many times we've already deferred this batch

    Returns:
        True if translation should be deferred (wait for one more word)
    """
    if deferred_count >= max_defer:
        return False

    word_lower = last_word.lower().strip()

    if source_lang == "cs":
        return word_lower in _CS_FUNCTION_WORDS

    # Default: English function words
    return word_lower in _EN_FUNCTION_WORDS


def compute_attention_info_gain(
    prev_attn: np.ndarray,
    curr_attn: np.ndarray,
    ts_scores: List[float],
) -> float:
    """Compute information gain between consecutive attention snapshots.

    Uses TS-weighted KL divergence KL(curr || prev) to measure how much
    the attention pattern changed. Large divergence means the model is
    processing new source information (keep generating). Small divergence
    means the source is exhausted (supports border stop).

    Inspired by LSG (arxiv 2501.00868) which uses KL(P_partial || P_full)
    for training-free read/write decisions.

    Args:
        prev_attn: (n_heads, n_src) attention from previous generation step
        curr_attn: (n_heads, n_src) attention from current generation step
        ts_scores: TS score per head

    Returns:
        TS-weighted mean KL divergence in nats. Range: [0, inf)
        Small values (< 0.3) suggest source is exhausted.
        Large values (> 1.0) suggest new source info being processed.
    """
    n_heads = curr_attn.shape[0]
    eps = 1e-10

    kl_per_head = np.zeros(n_heads)
    for h in range(n_heads):
        p = curr_attn[h] + eps
        q = prev_attn[h] + eps
        p = p / p.sum()
        q = q / q.sum()
        # KL(p || q) = sum(p * log(p / q))
        kl_per_head[h] = float(np.sum(p * np.log(p / q)))

    ts = np.array(ts_scores, dtype=np.float64)
    ts_sum = ts.sum()
    if ts_sum < eps:
        return float(np.mean(kl_per_head))
    return float(np.dot(ts, kl_per_head) / ts_sum)


def check_border_shift_k(
    src_attn: np.ndarray,
    ts_scores: List[float],
    n_src_tokens: int,
    border_distance: int,
    threshold: float = 0.4,
    head_temp_normalize: bool = False,
    head_temp_reference: float = 1.5,
) -> bool:
    """Shift-k border check: trigger stop when attention mass in border region exceeds threshold.

    Inspired by DrFrattn (EMNLP 2025) "shift-k" mechanism. Instead of checking
    whether the argmax is in the border region (binary), we measure the total
    attention MASS in the border region. This is a softer, more robust signal.

    The border region is the last `border_distance` source positions.
    If the TS-weighted attention mass in this region >= threshold, stop.

    Key insight: a model attending 30% to pos 8 and 30% to pos 9 (border)
    should stop even if argmax is at pos 8. Pure argmax misses this.

    Args:
        src_attn: (n_heads, n_src) attention weights
        ts_scores: TS score per head
        n_src_tokens: Number of source tokens
        border_distance: Border region width
        threshold: Mass threshold to trigger stop (0.0-1.0, default: 0.4)
        head_temp_normalize: If True, normalize head temperatures first
        head_temp_reference: Reference entropy for normalization

    Returns:
        True if border mass exceeds threshold (should stop)
    """
    border_start = n_src_tokens - border_distance
    if border_start <= 0:
        return False

    attn = src_attn
    if head_temp_normalize:
        attn = normalize_head_temperatures(attn, head_temp_reference)

    n_heads = attn.shape[0]
    eps = 1e-10

    # Compute TS-weighted border mass
    border_mass_per_head = np.zeros(n_heads)
    for h in range(n_heads):
        p = attn[h]
        total = p.sum()
        if total > eps:
            p = p / total
        border_mass_per_head[h] = p[border_start:].sum()

    ts = np.array(ts_scores, dtype=np.float64)
    ts_sum = ts.sum()
    if ts_sum < eps:
        weighted_mass = float(np.mean(border_mass_per_head))
    else:
        weighted_mass = float(np.dot(ts, border_mass_per_head) / ts_sum)

    return weighted_mass >= threshold


def compute_source_coverage(
    src_attn: np.ndarray,
    ts_scores: List[float],
    min_attn_threshold: float = 0.05,
) -> Tuple[float, np.ndarray]:
    """Compute source coverage: fraction of source positions well-attended.

    Novel hallucination guard for SimulMT: track how much of the source
    is "covered" by attention across alignment heads. If coverage drops
    below a threshold during generation, the model may be hallucinating
    (generating from priors rather than translating the source).

    For each source position, compute the max TS-weighted attention
    received from any head. A position is "covered" if this value exceeds
    min_attn_threshold. Coverage ratio = fraction of covered positions.

    Key insight: during faithful translation, most source positions should
    receive significant attention from at least one head. When coverage
    drops, generation is no longer grounded in the source.

    No published work on attention coverage as a hallucination guard
    for simultaneous machine translation.

    Args:
        src_attn: (n_heads, n_src) attention weights over source tokens
        ts_scores: TS score for each head (higher = more reliable)
        min_attn_threshold: Minimum attention to consider a position "covered"
            (default: 0.05 = 5% of attention mass from at least one head)

    Returns:
        (coverage_ratio, coverage_per_position):
            coverage_ratio: Fraction of source positions covered (0-1).
                1.0 = all source positions attended. < 0.3 = likely hallucinating.
            coverage_per_position: (n_src,) max TS-weighted attention per position.
    """
    n_heads, n_src = src_attn.shape
    if n_src == 0:
        return 1.0, np.array([])

    eps = 1e-10
    ts = np.array(ts_scores, dtype=np.float64)
    ts_sum = ts.sum()
    if ts_sum < eps:
        ts = np.ones(n_heads) / n_heads
    else:
        ts = ts / ts_sum

    # Normalize attention per head
    attn_norm = np.zeros_like(src_attn)
    for h in range(n_heads):
        total = src_attn[h].sum()
        if total > eps:
            attn_norm[h] = src_attn[h] / total

    # For each source position, compute TS-weighted max attention
    # across heads. A position is "covered" if any head with decent
    # TS score pays attention to it.
    # Method: TS-weighted sum of attention at each position
    coverage_per_pos = np.zeros(n_src, dtype=np.float64)
    for h in range(n_heads):
        coverage_per_pos += ts[h] * attn_norm[h]

    # A position is covered if it receives >= threshold total weighted attention
    covered = coverage_per_pos >= min_attn_threshold
    coverage_ratio = float(covered.sum()) / n_src

    return coverage_ratio, coverage_per_pos


def coverage_supports_write(
    coverage_ratio: float,
    min_coverage: float = 0.3,
) -> bool:
    """Interpret source coverage as a WRITE/hallucination signal.

    Args:
        coverage_ratio: From compute_source_coverage() (0-1)
        min_coverage: Minimum acceptable coverage ratio. Below this,
            the model is likely hallucinating. Default: 0.3.

    Returns:
        True if coverage is acceptable (supports continued generation).
        False if coverage is too low (hallucination risk, should stop).
    """
    return coverage_ratio >= min_coverage


def compute_attention_monotonicity(
    positions_history: List[float],
) -> float:
    """Compute monotonicity score of attention positions across generation steps.

    Measures how consistently attention moves forward (left-to-right) through
    the source during generation. In translation, attention should generally
    progress monotonically through the source (with some reordering allowed).

    Uses a normalized Kendall tau-like metric: count concordant vs discordant
    consecutive pairs. Score in [-1, 1]:
        1.0 = perfectly monotonic (always moves forward)
        0.0 = random movement
       -1.0 = perfectly reverse (always moves backward)

    Novel: no published work on attention monotonicity scoring for
    decoder-only LLM simultaneous translation border detection.

    Key uses:
    - Monotonic attention -> straightforward translation -> tighter border
    - Non-monotonic attention -> reordering happening -> wider border needed
    - Strongly negative -> possible hallucination loop (repeating)

    Args:
        positions_history: List of attended source positions across
            consecutive generation steps (from aggregate()).

    Returns:
        Monotonicity score in [-1, 1]. Empty or single-element returns 0.0.
    """
    n = len(positions_history)
    if n < 2:
        return 0.0

    concordant = 0
    discordant = 0
    for i in range(n - 1):
        diff = positions_history[i + 1] - positions_history[i]
        if diff > 0:
            concordant += 1
        elif diff < 0:
            discordant += 1
        # diff == 0 -> tie, neither concordant nor discordant

    total = concordant + discordant
    if total == 0:
        return 0.0  # All positions identical

    return (concordant - discordant) / total


def monotonicity_border_adjustment(
    monotonicity: float,
    base_bd: int,
    max_increase: int = 2,
) -> int:
    """Adjust border distance based on attention monotonicity.

    Monotonic attention (score > 0.5) suggests straightforward translation;
    we can use a tighter border for lower latency. Non-monotonic attention
    (score < 0) suggests reordering or confusion; widen the border for safety.

    Args:
        monotonicity: Score from compute_attention_monotonicity() [-1, 1]
        base_bd: Base border distance from config
        max_increase: Maximum border distance increase for non-monotonic (default: 2)

    Returns:
        Adjusted border distance (always >= 1)
    """
    if monotonicity >= 0.7:
        # Highly monotonic -> tighten border
        delta = -1
    elif monotonicity >= 0.3:
        # Normal -> keep base
        delta = 0
    elif monotonicity >= 0.0:
        # Mildly non-monotonic -> slight widening
        delta = 1
    else:
        # Negative monotonicity -> significant reordering or hallucination
        delta = max_increase

    return max(1, base_bd + delta)


def check_border_combined(
    src_attn: np.ndarray,
    ts_scores: List[float],
    n_src_tokens: int,
    border_distance: int,
    aggregation: str = "ts_vote",
    adaptive_aggregation: bool = False,
    head_temp_normalize: bool = False,
    head_temp_reference: float = 1.5,
    shift_k_threshold: Optional[float] = None,
    prev_attn: Optional[np.ndarray] = None,
    info_gain_threshold: Optional[float] = None,
    dynamic_border: bool = False,
    entropy_change: Optional[float] = None,
    entropy_change_threshold: Optional[float] = None,
    pred_stability_write: Optional[bool] = None,
    coverage_threshold: Optional[float] = None,
    positions_history: Optional[List[float]] = None,
    monotonicity_enabled: bool = False,
    attn_shift_write: Optional[bool] = None,
    top_p_threshold: float = 0.8,
) -> Tuple[bool, Optional[float], Optional[float]]:
    """Combined border check with all available signals.

    Combines: standard AlignAtt + shift-k mass + info gain +
    entropy change (REINA) + prediction stability (novel) +
    source coverage guard (novel) + attention monotonicity (novel).
    Returns the border decision plus diagnostic values.

    Decision logic:
        0a. Cross-step pre-filter: entropy change (REINA)
        0b. Source coverage guard: if coverage drops below threshold -> force STOP
        1. If shift_k_threshold is set and border mass >= threshold -> STOP
        2. If info_gain_threshold is set, modulate stop decisions
        3. Standard AlignAtt border check (with monotonicity-adjusted bd)
        4. Cross-step post-filter: prediction stability modulation

    Args:
        src_attn, ts_scores, n_src_tokens, border_distance: Standard border params
        aggregation, adaptive_aggregation: Aggregation params
        head_temp_normalize, head_temp_reference: Temp norm params
        shift_k_threshold: If set, enable shift-k check
        prev_attn: Previous step attention (for info gain)
        info_gain_threshold: If set, enable info gain modulation
        dynamic_border: If True, use entropy-based dynamic border
        entropy_change: Pre-computed entropy change (REINA)
        entropy_change_threshold: Threshold for entropy change
        pred_stability_write: Pre-computed prediction stability signal
        coverage_threshold: If set, enable source coverage guard. If coverage
            drops below this value, force stop (hallucination prevention).
            None=disabled, 0.3=recommended.
        positions_history: List of attended positions from generation loop
            (for monotonicity computation). None=disabled.
        monotonicity_enabled: If True and positions_history provided, adjust
            border distance based on attention monotonicity.

    Returns:
        (border_hit, info_gain_value, border_mass_value)
    """
    attn = src_attn
    if head_temp_normalize:
        attn = normalize_head_temperatures(attn, head_temp_reference)

    info_gain_val = None
    border_mass_val = None

    # 0a. Cross-step pre-filter: entropy change (REINA-inspired)
    if (entropy_change is not None and entropy_change_threshold is not None
            and entropy_change < entropy_change_threshold):
        return False, None, None

    # 0a2. Cross-step pre-filter: attention shift
    # If attention didn't shift forward, model isn't consuming source -> READ
    if attn_shift_write is not None and not attn_shift_write:
        return False, None, None

    # 0b. Source coverage guard (novel): if source coverage is too low,
    # the model is likely hallucinating -> force stop
    if coverage_threshold is not None:
        cov_ratio, _ = compute_source_coverage(attn, ts_scores)
        if not coverage_supports_write(cov_ratio, min_coverage=coverage_threshold):
            return True, info_gain_val, border_mass_val

    # 0c. Monotonicity-adjusted border distance
    effective_bd = border_distance
    if monotonicity_enabled and positions_history and len(positions_history) >= 3:
        mono_score = compute_attention_monotonicity(positions_history)
        effective_bd = monotonicity_border_adjustment(mono_score, border_distance)

    # 1. Compute info gain if available
    if prev_attn is not None and info_gain_threshold is not None:
        if prev_attn.shape[1] <= attn.shape[1]:
            padded_prev = np.zeros_like(attn)
            padded_prev[:, :prev_attn.shape[1]] = prev_attn
            info_gain_val = compute_attention_info_gain(padded_prev, attn, ts_scores)
        else:
            info_gain_val = compute_attention_info_gain(
                prev_attn[:, :attn.shape[1]], attn, ts_scores
            )

        if info_gain_val > info_gain_threshold * 3:
            return False, info_gain_val, None

    # 2. Shift-k mass check
    if shift_k_threshold is not None:
        shift_k_hit = check_border_shift_k(
            attn, ts_scores, n_src_tokens, effective_bd,
            threshold=shift_k_threshold,
            head_temp_normalize=False,
        )
        border_start = n_src_tokens - effective_bd
        if border_start > 0:
            eps = 1e-10
            n_heads = attn.shape[0]
            mass_per_head = np.zeros(n_heads)
            for h in range(n_heads):
                p = attn[h]
                total = p.sum()
                if total > eps:
                    p = p / total
                mass_per_head[h] = p[border_start:].sum()
            ts = np.array(ts_scores, dtype=np.float64)
            ts_sum = ts.sum()
            if ts_sum > eps:
                border_mass_val = float(np.dot(ts, mass_per_head) / ts_sum)
            else:
                border_mass_val = float(np.mean(mass_per_head))

        if shift_k_hit:
            if info_gain_val is not None and info_gain_val < info_gain_threshold:
                return True, info_gain_val, border_mass_val
            return True, info_gain_val, border_mass_val

    # 3. Standard AlignAtt border check
    method = aggregation
    if adaptive_aggregation:
        method = select_adaptive_aggregation(attn, ts_scores)

    if dynamic_border:
        effective_bd = dynamic_border_distance(
            attn, ts_scores, effective_bd, n_src_tokens,
        )

    border_threshold = n_src_tokens - effective_bd
    if border_threshold <= 0:
        return False, info_gain_val, border_mass_val

    attended_pos = aggregate(attn, ts_scores, method=method,
                             p_threshold=top_p_threshold)
    standard_hit = attended_pos >= border_threshold

    if standard_hit and info_gain_val is not None:
        if info_gain_val > info_gain_threshold * 1.5:
            return False, info_gain_val, border_mass_val

    # 4. Cross-step post-filter: prediction stability modulation
    if standard_hit and pred_stability_write is not None:
        if not pred_stability_write:
            return False, info_gain_val, border_mass_val

    return standard_hit, info_gain_val, border_mass_val


def compute_logit_kl(
    logits_full: np.ndarray,
    logits_reduced: np.ndarray,
) -> float:
    """Compute KL divergence between two logit distributions.

    Implements the core LSG signal (arxiv 2501.00868, AAAI 2025): compare
    output logit distributions with full source vs reduced source (last K
    source tokens removed). The KL divergence measures how much the removed
    source tokens affect the model's prediction.

    KL(P_full || P_reduced):
      - LOW KL (< delta): removing source tokens didn't change prediction
        -> the model doesn't need more source -> safe to WRITE (emit tokens)
      - HIGH KL (> delta): removing tokens changed the prediction significantly
        -> the model is still using recent source -> should READ more

    This is an orthogonal signal to attention-based border detection:
    attention measures WHERE the model looks, logit KL measures WHETHER
    looking there changed the OUTPUT.

    Args:
        logits_full: Raw logits with full source (n_vocab,)
        logits_reduced: Raw logits with reduced source (n_vocab,)

    Returns:
        KL divergence in nats. Typical range: [0, 15+]
        LSG paper recommends delta thresholds of 5.0-9.0 for 7B models.
    """
    # Stable softmax for both distributions
    lf = logits_full - np.max(logits_full)
    lr = logits_reduced - np.max(logits_reduced)

    p = np.exp(lf)
    p = p / p.sum()
    q = np.exp(lr)
    q = q / q.sum()

    eps = 1e-10
    # KL(P || Q) = sum(p * log(p / q))
    kl = float(np.sum(p * np.log((p + eps) / (q + eps))))
    return max(0.0, kl)  # guard against numerical noise


def compute_entropy_change(
    current_logits: np.ndarray,
    prev_entropy: Optional[float],
) -> Tuple[Optional[float], float]:
    """Compute change in generation entropy between translate() steps.

    Inspired by REINA (arxiv 2508.04946, AAAI 2026 Oral): use information
    gain to decide READ/WRITE. Track entropy of the first generated token
    across consecutive translate() calls. If adding a new source word
    reduced entropy significantly, the model is still learning from source
    (READ more). If entropy didn't change, source is exhausted (WRITE).

    This is a cross-step signal (between translate() calls), unlike
    attention-based checks which are within a single generation loop.

    Args:
        current_logits: Raw logits from the first generation position (n_vocab,)
        prev_entropy: Entropy from the previous translate() call. None on first call.

    Returns:
        (entropy_change, current_entropy):
            entropy_change: H_current - H_previous. Negative = entropy dropped
                (new source word was informative). None if no previous entropy.
            current_entropy: Current entropy value (to store for next call).
    """
    current_entropy = compute_entropy(current_logits)

    if prev_entropy is None:
        return None, current_entropy

    delta = current_entropy - prev_entropy
    return delta, current_entropy


def compute_prediction_stability(
    current_logits: np.ndarray,
    prev_logits: Optional[np.ndarray],
    top_k: int = 5,
) -> Tuple[Optional[float], Optional[float]]:
    """Compute prediction stability between consecutive translate() calls.

    Measures how much the model's output predictions changed when a new
    source word was added. Stable predictions indicate the model has
    enough source context (supports WRITE). Volatile predictions indicate
    the model is still adapting to new source info (supports READ).

    Two complementary metrics:
    1. Top-1 rank stability: rank of previous top-1 token in current distribution.
       Low rank (0-2) = stable, high rank (10+) = volatile.
    2. Top-K overlap: Jaccard similarity of top-K predicted tokens.
       1.0 = identical, 0.0 = no overlap.

    Novel: no published work on cross-step prediction stability for
    SimulMT border detection.

    Args:
        current_logits: Raw logits from first generation position (n_vocab,)
        prev_logits: Raw logits from previous translate() call. None on first call.
        top_k: Number of top tokens to compare (default: 5)

    Returns:
        (top1_rank_change, topk_overlap):
            top1_rank_change: Rank of previous top-1 in current distribution
                (0 = unchanged, higher = more volatile). None if no previous.
            topk_overlap: Jaccard similarity of top-K sets (0-1).
                None if no previous logits.
    """
    if prev_logits is None:
        return None, None

    # Previous top-1 token
    prev_top1 = int(np.argmax(prev_logits))

    # Current ranking: sort by descending logit
    current_ranking = np.argsort(current_logits)[::-1]

    # Find rank of prev_top1 in current distribution
    rank_positions = np.where(current_ranking == prev_top1)[0]
    top1_rank = int(rank_positions[0]) if len(rank_positions) > 0 else len(current_logits)

    # Top-K overlap (Jaccard)
    prev_topk = set(int(x) for x in np.argsort(prev_logits)[-top_k:])
    curr_topk = set(int(x) for x in current_ranking[:top_k])
    intersection = len(prev_topk & curr_topk)
    union = len(prev_topk | curr_topk)
    topk_overlap = intersection / union if union > 0 else 1.0

    return float(top1_rank), topk_overlap


def entropy_change_supports_write(
    entropy_change: Optional[float],
    threshold: float = -0.5,
) -> Optional[bool]:
    """Interpret entropy change as a READ/WRITE signal.

    Args:
        entropy_change: H_current - H_previous (from compute_entropy_change)
        threshold: Negative threshold. If delta > threshold (close to 0),
            source is exhausted -> WRITE. If delta < threshold (large drop),
            source is informative -> READ more. Default: -0.5.

    Returns:
        True if entropy change supports WRITE (source exhausted).
        False if it supports READ (source still informative).
        None if entropy change is unavailable.
    """
    if entropy_change is None:
        return None
    # delta close to zero or positive -> source didn't help -> WRITE
    # delta very negative -> source reduced uncertainty -> READ more
    return entropy_change > threshold


def prediction_stability_supports_write(
    top1_rank: Optional[float],
    topk_overlap: Optional[float],
    rank_threshold: float = 3.0,
    overlap_threshold: float = 0.4,
) -> Optional[bool]:
    """Interpret prediction stability as a READ/WRITE signal.

    Args:
        top1_rank: Rank of previous top-1 in current distribution
        topk_overlap: Jaccard similarity of top-K sets
        rank_threshold: If top1_rank <= threshold, predictions are stable (WRITE).
            Default: 3.0 (top-1 stayed in top-3).
        overlap_threshold: If topk_overlap >= threshold, predictions are stable.
            Default: 0.4 (at least 40% top-K overlap).

    Returns:
        True if predictions are stable (supports WRITE).
        False if volatile (supports READ).
        None if no stability data available.
    """
    if top1_rank is None or topk_overlap is None:
        return None
    # Both signals must agree for a strong signal
    rank_stable = top1_rank <= rank_threshold
    overlap_stable = topk_overlap >= overlap_threshold
    return rank_stable and overlap_stable


def compute_attention_shift(
    current_attn: np.ndarray,
    prev_attn: Optional[np.ndarray],
    ts_scores: List[float],
) -> Optional[float]:
    """Compute cross-step attention shift between translate() calls.

    Measures how much the model's source focus changed after adding a new
    source word. Large forward shift = model consuming new source (WRITE).
    Small/no shift = model not integrating new source (READ more).

    This is a cross-step, input-space signal -- orthogonal to:
    - Entropy change (cross-step, output-space)
    - Prediction stability (cross-step, output-space)
    - Source coverage (within-step, input-space)

    Novel: no published work on cross-step attention position shift
    as a border signal for simultaneous MT.

    Args:
        current_attn: Current step attention (n_heads, n_src)
        prev_attn: Previous translate() call's attention. None on first call.
        ts_scores: TS score per head

    Returns:
        Attention position shift (positive = forward). None if no prev_attn.
        Typical range: [-1, n_src]. Values > 0.5 indicate forward progress.
    """
    if prev_attn is None:
        return None

    ts = np.array(ts_scores, dtype=np.float64)
    ts_sum = ts.sum()
    if ts_sum < 1e-10:
        return None

    # Compute TS-weighted attended position for current and previous
    def weighted_pos(attn):
        n_heads, n_src = attn.shape
        positions = np.arange(n_src, dtype=np.float64)
        per_head_pos = np.zeros(n_heads)
        for h in range(n_heads):
            p = attn[h]
            total = p.sum()
            if total > 1e-10:
                p = p / total
                per_head_pos[h] = np.dot(p, positions)
        return float(np.dot(ts[:n_heads], per_head_pos[:n_heads]) / ts_sum)

    # Handle size mismatch (new source has more tokens)
    cur_pos = weighted_pos(current_attn)

    # For prev, only use the overlapping source region
    n_prev = prev_attn.shape[1]
    n_cur = current_attn.shape[1]
    if n_prev > 0:
        prev_pos = weighted_pos(prev_attn)
    else:
        return None

    return cur_pos - prev_pos


def attention_shift_supports_write(
    shift: Optional[float],
    min_shift: float = 0.5,
) -> Optional[bool]:
    """Interpret attention shift as READ/WRITE signal.

    Args:
        shift: Attention position change (from compute_attention_shift)
        min_shift: Minimum shift to support WRITE. Default: 0.5.

    Returns:
        True if shift supports WRITE (model consuming source).
        False if shift supports READ (model stuck, needs more).
        None if no shift data.
    """
    if shift is None:
        return None
    return shift >= min_shift


def detect_ngram_repetition(
    token_ids: List[int],
    min_n: int = 2,
    max_n: int = 4,
    max_repeats: int = 2,
) -> bool:
    """Detect n-gram repetition in generated token sequence.

    When LLMs hallucinate during generation, they often enter repetitive
    loops producing the same n-gram pattern repeatedly. This function
    detects such patterns as an early stopping signal.

    This is a within-step, output-space signal -- orthogonal to all
    attention-based signals. It measures the OUTPUT quality directly
    rather than inferring quality from attention patterns.

    Novel application: no published work on n-gram repetition detection
    as a border/halt signal in simultaneous MT with decoder-only LLMs.

    Algorithm:
        For each n in [min_n, max_n], check if the last n tokens form
        a pattern that has appeared >= max_repeats times in the recent
        token history. This catches both exact repetition (same phrase
        repeated) and degenerate loops.

    Args:
        token_ids: List of generated token IDs (chronological order)
        min_n: Minimum n-gram size to check (default: 2, bigrams)
        max_n: Maximum n-gram size to check (default: 4, 4-grams)
        max_repeats: Maximum allowed repeats before flagging (default: 2)

    Returns:
        True if repetition detected (should halt generation).
        False if no concerning repetition found.
    """
    n_tokens = len(token_ids)
    if n_tokens < min_n * (max_repeats + 1):
        return False

    for n in range(min_n, min(max_n, n_tokens // 2) + 1):
        # Extract the last n tokens as the pattern
        pattern = tuple(token_ids[-n:])

        # Count occurrences in the full sequence
        count = 0
        for i in range(n_tokens - n + 1):
            if tuple(token_ids[i:i + n]) == pattern:
                count += 1

        if count > max_repeats:
            return True

    return False


def compute_repetition_score(
    token_ids: List[int],
    window: int = 20,
) -> float:
    """Compute a continuous repetition score for the last N tokens.

    Instead of a binary flag, provides a 0-1 score indicating how
    repetitive the recent generation is. Useful for gradual modulation
    rather than hard cutoff.

    Algorithm:
        Count unique bigrams in the last `window` tokens divided by
        the total number of bigrams. Highly repetitive text will have
        few unique bigrams relative to total.

    Args:
        token_ids: List of generated token IDs
        window: Size of the lookback window (default: 20)

    Returns:
        Repetition score in [0, 1]. 0 = no repetition, 1 = fully repetitive.
        Returns 0.0 if fewer than 3 tokens generated.
    """
    if len(token_ids) < 3:
        return 0.0

    recent = token_ids[-window:]
    if len(recent) < 3:
        return 0.0

    # Count bigrams
    bigrams = [(recent[i], recent[i + 1]) for i in range(len(recent) - 1)]
    n_bigrams = len(bigrams)
    n_unique = len(set(bigrams))

    if n_bigrams <= 1:
        return 0.0

    # Score: 1 - (unique / total). 0 = all unique, 1 = all same
    return 1.0 - (n_unique / n_bigrams)


def adaptive_border_distance(
    base_bd: int,
    confidence: float,
    alpha: float = 2.0,
) -> int:
    """Compute adaptive border distance based on ASR confidence.

    Lower ASR confidence -> larger border distance (more cautious).

    Args:
        base_bd: Base border distance
        confidence: ASR confidence score (0-1)
        alpha: Scaling factor (default 2.0, range 0-4)

    Returns:
        Effective border distance (always >= 1)
    """
    return max(1, base_bd + round(alpha * (1 - confidence)))


def load_head_config(path: str) -> dict:
    """Load alignment head configuration from a JSON file.

    Expected format:
        {
            "model": "HY-MT1.5-7B",
            "direction": "en-zh",
            "token_alignment_heads": [
                {"layer": 5, "head": 12, "ts": 0.847},
                ...
            ]
        }

    Returns dict with keys: layers, heads, ts_scores, metadata
    """
    import json
    with open(path) as f:
        data = json.load(f)

    heads_list = data["token_alignment_heads"]
    return {
        "layers": [h["layer"] for h in heads_list],
        "heads": [h["head"] for h in heads_list],
        "ts_scores": [h["ts"] for h in heads_list],
        "model": data.get("model", "unknown"),
        "direction": data.get("direction", "unknown"),
        "n_heads": len(heads_list),
    }
