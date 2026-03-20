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


# Base methods (before ensemble to avoid circular reference)
_BASE_AGGREGATION_METHODS = {
    "ts_vote": aggregate_ts_weighted_vote,
    "softmax_mean": aggregate_softmax_mean,
    "entropy_weighted": aggregate_entropy_weighted,
    "consensus": aggregate_consensus,
    "geomean": aggregate_geomean,
    "top_p": aggregate_top_p,
    "gaussian_kernel": aggregate_gaussian_kernel,
    "gaussian_kernel_continuous": aggregate_gaussian_kernel_continuous,
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
) -> float:
    """Unified aggregation dispatcher.

    Args:
        src_attn: (n_heads, n_src) attention weights
        ts_scores: TS score per head
        method: Aggregation method name

    Returns:
        Attended source position (float for softmax_mean, int for others)
    """
    if method not in _AGGREGATION_METHODS:
        raise ValueError(
            f"Unknown aggregation '{method}'. "
            f"Available: {list_aggregation_methods()}"
        )
    return _AGGREGATION_METHODS[method](src_attn, ts_scores)


def check_border(
    src_attn: np.ndarray,
    ts_scores: List[float],
    n_src_tokens: int,
    border_distance: int,
    aggregation: str = "ts_vote",
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

    Returns:
        True if border hit (should stop generating)
    """
    border_threshold = n_src_tokens - border_distance
    if border_threshold <= 0:
        return False

    attended_pos = aggregate(src_attn, ts_scores, method=aggregation)
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

    Returns:
        True if border hit (should stop generating)
    """
    effective_bd = dynamic_border_distance(
        src_attn, ts_scores, base_border_distance, n_src_tokens,
        low_entropy=low_entropy, high_entropy=high_entropy,
        max_bd_delta=max_bd_delta,
    )

    border_threshold = n_src_tokens - effective_bd
    if border_threshold <= 0:
        return False

    attended_pos = aggregate(src_attn, ts_scores, method=aggregation)
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
