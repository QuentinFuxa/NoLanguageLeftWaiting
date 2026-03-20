"""Weighted signal fusion for SimulMT border detection.

Replaces the boolean cascade in check_border_combined() with a principled
weighted scoring system where each signal produces a continuous confidence
score, and the final border decision is based on their weighted sum.

Key advantages over boolean cascade:
    1. Weak signals can combine: two marginal signals together trigger a stop
       that neither would alone.
    2. Per-direction tuning: weight vectors can be optimized per language pair.
    3. Signal contributions are observable: diagnostics show which signals
       drove each decision.
    4. Order-independent: no hidden priority from if/else sequencing.

Signal taxonomy (all 8 signals):
    Within-step, position-based:
        - standard: AlignAtt argmax border check (foundation)
        - shift_k: Attention mass in border region (DrFrattn-inspired)
        - info_gain: KL divergence between consecutive attention snapshots
    Within-step, input-coverage:
        - coverage: Source position coverage ratio (hallucination guard)
    Within-step, temporal:
        - monotonicity: Attention movement regularity score
    Cross-step, output-space:
        - entropy_change: REINA-inspired entropy delta
        - pred_stability: Prediction stability index (novel)
    Cross-step, input-space:
        - attn_shift: Attention position shift between translate() calls

Each signal maps to a continuous score in [-1, +1]:
    +1 = strong WRITE confidence (model has enough source, should emit)
    -1 = strong READ confidence (model needs more source)
     0 = neutral (no information from this signal)

The fusion score is: sum(weight_i * score_i) for all enabled signals.
Decision: score >= threshold -> WRITE (border hit), else -> READ (continue).

Novel: no published work on weighted multi-signal fusion for SimulMT
border detection. Closest work is DrFrattn (EMNLP 2025) which uses a
single aggregated signal, and LSG (AAAI 2025) which checks logit KL
independently. Our approach fuses 8 orthogonal signals in a principled way.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from nllw.alignatt import (
    aggregate,
    check_border_shift_k,
    compute_attention_info_gain,
    compute_attention_monotonicity,
    compute_source_coverage,
    coverage_supports_write,
    monotonicity_border_adjustment,
    normalize_head_temperatures,
    select_adaptive_aggregation,
    dynamic_border_distance,
)


# ---------------------------------------------------------------------------
# Signal score computation
# ---------------------------------------------------------------------------

def score_standard_border(
    src_attn: np.ndarray,
    ts_scores: List[float],
    n_src_tokens: int,
    border_distance: int,
    aggregation: str = "ts_vote",
    adaptive_aggregation: bool = False,
) -> float:
    """Compute continuous WRITE score from standard AlignAtt border check.

    Instead of binary hit/miss, returns how far into the border region
    the attended position is, normalized to [-1, +1].

    Args:
        src_attn: (n_heads, n_src) attention weights
        ts_scores: TS score per head
        n_src_tokens: Number of source tokens
        border_distance: Border distance parameter
        aggregation: Aggregation method name
        adaptive_aggregation: Whether to use AMS

    Returns:
        Score in [-1, +1]. Positive = in border region (WRITE).
        Magnitude indicates how deep into border.
    """
    if n_src_tokens <= 0:
        return 0.0

    method = aggregation
    if adaptive_aggregation:
        method = select_adaptive_aggregation(src_attn, ts_scores)

    attended_pos = float(aggregate(src_attn, ts_scores, method=method))
    border_threshold = n_src_tokens - border_distance

    if border_threshold <= 0:
        return 0.0

    # Normalize: 0 at border_threshold, +1 at n_src-1, -1 at border_threshold - border_distance
    distance_from_border = attended_pos - border_threshold
    if border_distance > 0:
        score = distance_from_border / border_distance
    else:
        score = 1.0 if distance_from_border >= 0 else -1.0

    return float(np.clip(score, -1.0, 1.0))


def score_shift_k(
    src_attn: np.ndarray,
    ts_scores: List[float],
    n_src_tokens: int,
    border_distance: int,
    threshold: float = 0.4,
) -> float:
    """Compute continuous WRITE score from shift-k border mass.

    Maps border attention mass to [-1, +1] centered on threshold.

    Args:
        src_attn: (n_heads, n_src) attention weights
        ts_scores: TS score per head
        n_src_tokens: Number of source tokens
        border_distance: Border distance parameter
        threshold: Reference threshold for centering score

    Returns:
        Score in [-1, +1]. Positive = mass above threshold (WRITE).
    """
    border_start = n_src_tokens - border_distance
    if border_start <= 0 or border_start >= n_src_tokens or n_src_tokens <= 0:
        return 0.0

    eps = 1e-10
    n_heads = src_attn.shape[0]
    mass_per_head = np.zeros(n_heads)
    for h in range(n_heads):
        p = src_attn[h]
        total = p.sum()
        if total > eps:
            p = p / total
        mass_per_head[h] = p[border_start:].sum()

    ts = np.array(ts_scores, dtype=np.float64)
    ts_sum = ts.sum()
    if ts_sum > eps:
        border_mass = float(np.dot(ts, mass_per_head) / ts_sum)
    else:
        border_mass = float(np.mean(mass_per_head))

    # Center on threshold: mass > threshold -> positive, mass < threshold -> negative
    # Scale so that mass=1.0 -> score=+1, mass=0 -> score=-1
    if threshold > 0:
        score = (border_mass - threshold) / threshold
    else:
        score = border_mass * 2.0 - 1.0

    return float(np.clip(score, -1.0, 1.0))


def score_info_gain(
    src_attn: np.ndarray,
    prev_attn: Optional[np.ndarray],
    ts_scores: List[float],
    threshold: float = 0.3,
) -> float:
    """Compute continuous WRITE score from attention information gain.

    High info gain = model processing new source = READ (negative score).
    Low info gain = source exhausted = WRITE (positive score).

    Args:
        src_attn: Current attention (n_heads, n_src)
        prev_attn: Previous step attention. None -> return 0.
        ts_scores: TS scores
        threshold: Reference threshold in nats

    Returns:
        Score in [-1, +1]. Positive = low info gain (WRITE).
    """
    if prev_attn is None:
        return 0.0

    # Align sizes
    if prev_attn.shape[1] <= src_attn.shape[1]:
        padded_prev = np.zeros_like(src_attn)
        padded_prev[:, :prev_attn.shape[1]] = prev_attn
        info_gain = compute_attention_info_gain(padded_prev, src_attn, ts_scores)
    else:
        info_gain = compute_attention_info_gain(
            prev_attn[:, :src_attn.shape[1]], src_attn, ts_scores
        )

    # Invert: low info_gain -> WRITE (+1), high info_gain -> READ (-1)
    # Center on threshold
    if threshold > 0:
        score = -(info_gain - threshold) / (threshold * 2)
    else:
        score = -info_gain

    return float(np.clip(score, -1.0, 1.0))


def score_coverage(
    src_attn: np.ndarray,
    ts_scores: List[float],
    threshold: float = 0.3,
) -> float:
    """Compute continuous WRITE score from source coverage.

    Low coverage = hallucination risk = force STOP (positive score, WRITE).
    High coverage = grounded generation = allow continuation (negative score).

    Note: This signal is INVERTED from the others. When coverage is low,
    we want to STOP immediately (force WRITE/emit what we have) because
    the model is likely hallucinating. High coverage means we're safe
    to keep generating.

    Args:
        src_attn: (n_heads, n_src) attention
        ts_scores: TS scores
        threshold: Below this coverage -> force stop

    Returns:
        Score in [-1, +1]. Positive when coverage is LOW (force WRITE/stop).
    """
    cov_ratio, _ = compute_source_coverage(src_attn, ts_scores)

    # Low coverage -> force stop (WRITE): score = +1
    # High coverage -> allow generation (neutral/READ): score = -1
    if threshold > 0:
        score = -(cov_ratio - threshold) / threshold
    else:
        score = 1.0 - cov_ratio * 2.0

    return float(np.clip(score, -1.0, 1.0))


def score_monotonicity(
    positions_history: Optional[List[float]],
    border_distance: int,
) -> float:
    """Compute continuous WRITE score from attention monotonicity.

    Highly monotonic attention -> tighter border (more WRITE-inclined).
    Non-monotonic attention -> wider border (more READ-inclined).

    Args:
        positions_history: List of attended positions across generation steps
        border_distance: Base border distance (for normalization)

    Returns:
        Score in [-1, +1]. Positive = monotonic (tighter border, more WRITE).
    """
    if not positions_history or len(positions_history) < 3:
        return 0.0

    mono_score = compute_attention_monotonicity(positions_history)
    # mono_score in [-1, 1] already, but let's map:
    # >0.7 -> WRITE-inclined, <0 -> READ-inclined
    # Center around 0.3 (neutral)
    return float(np.clip((mono_score - 0.3) / 0.7, -1.0, 1.0))


def score_entropy_change(
    entropy_change: Optional[float],
    threshold: float = -0.5,
) -> float:
    """Compute continuous WRITE score from cross-step entropy change.

    Large entropy drop = model learning from new source = READ.
    Small change = source exhausted = WRITE.

    Args:
        entropy_change: H_current - H_previous. Negative = drop.
        threshold: Reference threshold (negative).

    Returns:
        Score in [-1, +1]. Positive = small change (WRITE).
    """
    if entropy_change is None:
        return 0.0

    # entropy_change > threshold (close to 0) -> WRITE (+1)
    # entropy_change < threshold (large drop) -> READ (-1)
    abs_thresh = abs(threshold) if threshold != 0 else 0.5
    score = (entropy_change - threshold) / abs_thresh

    return float(np.clip(score, -1.0, 1.0))


def score_pred_stability(
    pred_stability_write: Optional[bool],
) -> float:
    """Compute WRITE score from prediction stability.

    This is a binary signal (already interpreted), mapped to discrete scores.

    Args:
        pred_stability_write: Pre-computed stability signal.

    Returns:
        +0.7 if stable (WRITE), -0.7 if volatile (READ), 0.0 if unavailable.
    """
    if pred_stability_write is None:
        return 0.0
    return 0.7 if pred_stability_write else -0.7


def score_attn_shift(
    attn_shift_write: Optional[bool],
) -> float:
    """Compute WRITE score from cross-step attention shift.

    Binary signal mapped to discrete scores.

    Args:
        attn_shift_write: Pre-computed attention shift signal.

    Returns:
        +0.7 if shifted (WRITE), -0.7 if stuck (READ), 0.0 if unavailable.
    """
    if attn_shift_write is None:
        return 0.0
    return 0.7 if attn_shift_write else -0.7


# ---------------------------------------------------------------------------
# Weight profiles (per-direction defaults)
# ---------------------------------------------------------------------------

@dataclass
class FusionWeights:
    """Weight vector for signal fusion.

    Each weight controls how much a signal contributes to the final score.
    Weight of 0 disables a signal. Negative weights invert the signal.

    The default weights are designed to:
        1. Give standard AlignAtt the highest influence (foundation signal)
        2. Give shift-k moderate weight (soft alternative to argmax)
        3. Give cross-step signals lower weight (modulatory role)
        4. Give coverage high weight (hallucination prevention is critical)
    """
    standard: float = 1.0       # AlignAtt argmax border (foundation)
    shift_k: float = 0.6        # Border mass (DrFrattn-inspired)
    info_gain: float = 0.4      # Attention info gain
    coverage: float = 0.8       # Source coverage (hallucination guard)
    monotonicity: float = 0.3   # Attention regularity
    entropy_change: float = 0.4 # REINA entropy delta
    pred_stability: float = 0.3 # Prediction stability
    attn_shift: float = 0.3     # Attention position shift

    def as_dict(self) -> Dict[str, float]:
        """Return weights as a dictionary."""
        return {
            "standard": self.standard,
            "shift_k": self.shift_k,
            "info_gain": self.info_gain,
            "coverage": self.coverage,
            "monotonicity": self.monotonicity,
            "entropy_change": self.entropy_change,
            "pred_stability": self.pred_stability,
            "attn_shift": self.attn_shift,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "FusionWeights":
        """Create from a dict, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})

    def enabled_signals(self) -> List[str]:
        """Return list of signal names with non-zero weight."""
        return [name for name, w in self.as_dict().items() if abs(w) > 1e-6]

    def total_weight(self) -> float:
        """Sum of absolute weights (for normalization reference)."""
        return sum(abs(w) for w in self.as_dict().values())


# Pre-configured weight profiles per direction
# These are starting points -- should be optimized via grid search on FLORES
DIRECTION_WEIGHTS: Dict[str, FusionWeights] = {
    "en-zh": FusionWeights(
        standard=1.0, shift_k=0.6, info_gain=0.4,
        coverage=0.8, monotonicity=0.3,
        entropy_change=0.4, pred_stability=0.3, attn_shift=0.3,
    ),
    "en-de": FusionWeights(
        standard=1.0, shift_k=0.5, info_gain=0.4,
        coverage=0.7, monotonicity=0.4,  # DE more monotonic
        entropy_change=0.4, pred_stability=0.3, attn_shift=0.3,
    ),
    "en-it": FusionWeights(
        standard=1.0, shift_k=0.5, info_gain=0.4,
        coverage=0.7, monotonicity=0.4,
        entropy_change=0.4, pred_stability=0.3, attn_shift=0.3,
    ),
    "cs-en": FusionWeights(
        standard=1.0, shift_k=0.7, info_gain=0.3,
        coverage=0.6, monotonicity=0.2,  # CS-EN has more reordering
        entropy_change=0.5, pred_stability=0.4, attn_shift=0.3,
    ),
    "en-fr": FusionWeights(
        standard=1.0, shift_k=0.5, info_gain=0.4,
        coverage=0.7, monotonicity=0.4,
        entropy_change=0.4, pred_stability=0.3, attn_shift=0.3,
    ),
}


def get_fusion_weights(direction: str) -> FusionWeights:
    """Get fusion weights for a direction, falling back to defaults."""
    return DIRECTION_WEIGHTS.get(direction, FusionWeights())


# ---------------------------------------------------------------------------
# Fusion diagnostic
# ---------------------------------------------------------------------------

@dataclass
class FusionDiagnostic:
    """Diagnostic output from a fusion decision.

    Records each signal's raw score and weighted contribution, the final
    fusion score, and the decision. Useful for analysis and debugging.
    """
    scores: Dict[str, float] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    contributions: Dict[str, float] = field(default_factory=dict)
    fusion_score: float = 0.0
    threshold: float = 0.0
    decision: bool = False  # True = WRITE (border hit)

    def dominant_signal(self) -> Optional[str]:
        """Return the signal with the largest absolute contribution."""
        if not self.contributions:
            return None
        return max(self.contributions, key=lambda k: abs(self.contributions[k]))

    def summary(self) -> str:
        """One-line summary of the fusion decision."""
        d = "WRITE" if self.decision else "READ"
        dominant = self.dominant_signal()
        dom_str = f" (dominant: {dominant})" if dominant else ""
        return f"{d} score={self.fusion_score:.3f} thr={self.threshold:.3f}{dom_str}"


# ---------------------------------------------------------------------------
# Main fusion function
# ---------------------------------------------------------------------------

def fused_border_check(
    src_attn: np.ndarray,
    ts_scores: List[float],
    n_src_tokens: int,
    border_distance: int,
    weights: FusionWeights,
    threshold: float = 0.0,
    # Standard border params
    aggregation: str = "ts_vote",
    adaptive_aggregation: bool = False,
    head_temp_normalize: bool = False,
    head_temp_reference: float = 1.5,
    # Shift-k params
    shift_k_ref: float = 0.4,
    # Info gain params
    prev_attn: Optional[np.ndarray] = None,
    info_gain_ref: float = 0.3,
    # Dynamic border
    dynamic_border: bool = False,
    # Coverage params
    coverage_ref: float = 0.3,
    # Monotonicity params
    positions_history: Optional[List[float]] = None,
    # Cross-step signals (pre-computed)
    entropy_change: Optional[float] = None,
    entropy_change_ref: float = -0.5,
    pred_stability_write: Optional[bool] = None,
    attn_shift_write: Optional[bool] = None,
) -> Tuple[bool, FusionDiagnostic]:
    """Make a border decision by fusing all available signals.

    This replaces check_border_combined() with a weighted scoring approach.
    Each signal is computed, mapped to [-1, +1], weighted, and summed.
    The final decision is: fusion_score >= threshold -> WRITE (stop).

    Args:
        src_attn: (n_heads, n_src) attention weights
        ts_scores: TS score per alignment head
        n_src_tokens: Number of source tokens in current prompt
        border_distance: Base border distance parameter
        weights: FusionWeights instance with per-signal weights
        threshold: Decision threshold (default 0.0 = balanced)
        aggregation: Aggregation method for standard check
        adaptive_aggregation: Whether to use AMS
        head_temp_normalize: Whether to normalize head temperatures
        head_temp_reference: Reference entropy for temp normalization
        shift_k_ref: Reference threshold for shift-k scoring
        prev_attn: Previous step attention (for info gain)
        info_gain_ref: Reference threshold for info gain scoring
        dynamic_border: Whether to use entropy-based dynamic border
        coverage_ref: Reference threshold for coverage scoring
        positions_history: Attended positions history (for monotonicity)
        entropy_change: Pre-computed entropy change (REINA)
        entropy_change_ref: Reference threshold for entropy change
        pred_stability_write: Pre-computed prediction stability signal
        attn_shift_write: Pre-computed attention shift signal

    Returns:
        (border_hit, diagnostic):
            border_hit: True if fusion score exceeds threshold (WRITE).
            diagnostic: FusionDiagnostic with per-signal breakdown.
    """
    # Apply head temperature normalization if enabled
    attn = src_attn
    if head_temp_normalize:
        attn = normalize_head_temperatures(attn, head_temp_reference)

    # Apply dynamic border if enabled
    effective_bd = border_distance
    if dynamic_border and n_src_tokens > 0:
        effective_bd = dynamic_border_distance(
            attn, ts_scores, border_distance, n_src_tokens,
        )

    # Compute all signal scores
    scores: Dict[str, float] = {}
    w = weights.as_dict()

    # 1. Standard AlignAtt border
    if abs(w["standard"]) > 1e-6:
        scores["standard"] = score_standard_border(
            attn, ts_scores, n_src_tokens, effective_bd,
            aggregation=aggregation,
            adaptive_aggregation=adaptive_aggregation,
        )

    # 2. Shift-k border mass
    if abs(w["shift_k"]) > 1e-6:
        scores["shift_k"] = score_shift_k(
            attn, ts_scores, n_src_tokens, effective_bd,
            threshold=shift_k_ref,
        )

    # 3. Information gain
    if abs(w["info_gain"]) > 1e-6 and prev_attn is not None:
        scores["info_gain"] = score_info_gain(
            attn, prev_attn, ts_scores,
            threshold=info_gain_ref,
        )

    # 4. Source coverage
    if abs(w["coverage"]) > 1e-6:
        scores["coverage"] = score_coverage(
            attn, ts_scores,
            threshold=coverage_ref,
        )

    # 5. Monotonicity
    if abs(w["monotonicity"]) > 1e-6 and positions_history is not None:
        scores["monotonicity"] = score_monotonicity(
            positions_history, effective_bd,
        )

    # 6. Entropy change (cross-step)
    if abs(w["entropy_change"]) > 1e-6:
        scores["entropy_change"] = score_entropy_change(
            entropy_change, threshold=entropy_change_ref,
        )

    # 7. Prediction stability (cross-step)
    if abs(w["pred_stability"]) > 1e-6:
        scores["pred_stability"] = score_pred_stability(pred_stability_write)

    # 8. Attention shift (cross-step)
    if abs(w["attn_shift"]) > 1e-6:
        scores["attn_shift"] = score_attn_shift(attn_shift_write)

    # Compute weighted sum
    contributions = {}
    fusion_score = 0.0
    for signal_name, signal_score in scores.items():
        weight = w.get(signal_name, 0.0)
        contrib = weight * signal_score
        contributions[signal_name] = contrib
        fusion_score += contrib

    # Normalize by total active weight to keep threshold scale-independent
    active_weight = sum(abs(w[s]) for s in scores if s in w)
    if active_weight > 1e-6:
        fusion_score /= active_weight

    decision = fusion_score >= threshold

    diag = FusionDiagnostic(
        scores=scores,
        weights={s: w[s] for s in scores},
        contributions=contributions,
        fusion_score=fusion_score,
        threshold=threshold,
        decision=decision,
    )

    return decision, diag


# ---------------------------------------------------------------------------
# Fusion config for BackendConfig integration
# ---------------------------------------------------------------------------

@dataclass
class FusionConfig:
    """Configuration for signal fusion mode.

    When enabled, replaces check_border_combined() with fused_border_check().
    """
    enabled: bool = False
    threshold: float = 0.0  # Decision threshold
    weights: Optional[FusionWeights] = None  # None = use direction defaults
    # Reference thresholds for continuous signal scoring
    shift_k_ref: float = 0.4
    info_gain_ref: float = 0.3
    coverage_ref: float = 0.3
    entropy_change_ref: float = -0.5

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FusionConfig":
        """Create from dict, handling nested weights."""
        d = dict(d)  # Don't mutate original
        weights = d.pop("weights", None)
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        config = cls(**{k: v for k, v in d.items() if k in valid_keys})
        if weights is not None and isinstance(weights, dict):
            config.weights = FusionWeights.from_dict(weights)
        return config


# ---------------------------------------------------------------------------
# Grid search calibration
# ---------------------------------------------------------------------------

def calibrate_threshold(
    examples: List[Dict],
    weights: FusionWeights,
    target_write_ratio: float = 0.5,
) -> float:
    """Find optimal fusion threshold from labeled examples.

    Given a set of signal snapshots with known-good decisions, find the
    threshold that best separates READ and WRITE cases.

    Args:
        examples: List of dicts with keys:
            - "scores": Dict[str, float] -- per-signal raw scores
            - "should_write": bool -- ground truth decision
        weights: FusionWeights to use for scoring
        target_write_ratio: Target fraction of WRITE decisions (0.5 = balanced)

    Returns:
        Optimal threshold value.
    """
    if not examples:
        return 0.0

    w = weights.as_dict()
    fusion_scores = []

    for ex in examples:
        scores = ex["scores"]
        total = 0.0
        active = 0.0
        for signal, score in scores.items():
            weight = w.get(signal, 0.0)
            total += weight * score
            active += abs(weight)
        if active > 1e-6:
            total /= active
        fusion_scores.append((total, ex["should_write"]))

    # Sort by fusion score
    fusion_scores.sort(key=lambda x: x[0])

    # Find threshold that gives target_write_ratio
    n = len(fusion_scores)
    target_idx = int(n * (1 - target_write_ratio))
    target_idx = max(0, min(target_idx, n - 1))

    return fusion_scores[target_idx][0]


def grid_search_weights(
    examples: List[Dict],
    weight_grid: Optional[Dict[str, List[float]]] = None,
    metric_fn=None,
) -> Tuple[FusionWeights, float, float]:
    """Grid search over weight combinations to optimize a metric.

    Args:
        examples: List of dicts with keys:
            - "scores": Dict[str, float] -- per-signal scores
            - "should_write": bool -- ground truth
        weight_grid: Dict mapping signal names to list of weight values to try.
            Defaults to {signal: [0.0, 0.3, 0.6, 1.0]} for each signal.
        metric_fn: Function(predictions, ground_truths) -> float.
            Defaults to F1 score.

    Returns:
        (best_weights, best_threshold, best_metric)
    """
    if not examples:
        return FusionWeights(), 0.0, 0.0

    if weight_grid is None:
        weight_grid = {
            "standard": [0.6, 0.8, 1.0],
            "shift_k": [0.0, 0.3, 0.6],
            "info_gain": [0.0, 0.2, 0.4],
            "coverage": [0.0, 0.5, 0.8],
            "monotonicity": [0.0, 0.2, 0.4],
            "entropy_change": [0.0, 0.2, 0.4],
            "pred_stability": [0.0, 0.2, 0.4],
            "attn_shift": [0.0, 0.2, 0.4],
        }

    if metric_fn is None:
        metric_fn = _f1_score

    # Extract ground truth and scores
    gt = [ex["should_write"] for ex in examples]
    all_scores = [ex["scores"] for ex in examples]

    # Only search over signals present in examples
    present_signals = set()
    for s in all_scores:
        present_signals.update(s.keys())
    search_signals = [sig for sig in weight_grid if sig in present_signals]

    if not search_signals:
        return FusionWeights(), 0.0, 0.0

    best_weights = FusionWeights()
    best_threshold = 0.0
    best_metric = -1.0

    # Recursive grid search
    def search(idx, current_weights):
        nonlocal best_weights, best_threshold, best_metric

        if idx >= len(search_signals):
            # Evaluate this weight combination
            w = current_weights.copy()
            fusion_scores = []
            for scores in all_scores:
                total = 0.0
                active = 0.0
                for signal, score in scores.items():
                    weight = w.get(signal, 0.0)
                    total += weight * score
                    active += abs(weight)
                if active > 1e-6:
                    total /= active
                fusion_scores.append(total)

            # Try multiple thresholds
            sorted_scores = sorted(set(fusion_scores))
            for i in range(len(sorted_scores)):
                if i == 0:
                    thr = sorted_scores[0] - 0.01
                else:
                    thr = (sorted_scores[i - 1] + sorted_scores[i]) / 2

                preds = [s >= thr for s in fusion_scores]
                metric = metric_fn(preds, gt)
                if metric > best_metric:
                    best_metric = metric
                    best_threshold = thr
                    best_weights = FusionWeights.from_dict(w)
            return

        signal = search_signals[idx]
        for val in weight_grid[signal]:
            current_weights[signal] = val
            search(idx + 1, current_weights)

    search(0, FusionWeights().as_dict())
    return best_weights, best_threshold, best_metric


def _f1_score(predictions: List[bool], ground_truth: List[bool]) -> float:
    """Compute F1 score for binary classification."""
    tp = sum(1 for p, g in zip(predictions, ground_truth) if p and g)
    fp = sum(1 for p, g in zip(predictions, ground_truth) if p and not g)
    fn = sum(1 for p, g in zip(predictions, ground_truth) if not p and g)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall < 1e-10:
        return 0.0
    return 2 * precision * recall / (precision + recall)
