"""Source complexity estimation for adaptive SimulMT parameters.

Novel approach: estimate source sentence complexity to dynamically tune
translation parameters (border_distance, word_batch, gen_cap). No published
work combines multiple complexity signals for per-sentence parameter adaptation.

Complexity signals:
    1. Source length (word count): longer = more complex
    2. Source entropy (character diversity): more diverse = harder
    3. Subword expansion ratio: more subwords per word = morphologically complex
    4. Unknown word ratio: higher = harder for the model
    5. Numeral/entity density: more entities = translation-sensitive content

The complexity score maps to parameter adjustments:
    - Low complexity (simple, short) -> aggressive (small bd, small wb, small gen_cap)
    - High complexity (long, rare words) -> conservative (large bd, large wb, large gen_cap)
"""

import math
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ComplexityProfile:
    """Complexity analysis of a source sentence."""
    word_count: int
    char_count: int
    subword_ratio: float      # subwords per word (1.0 = no splitting)
    numeral_density: float    # fraction of tokens that are numbers
    avg_word_length: float    # mean characters per word
    complexity_score: float   # normalized 0-1 score (0=simple, 1=complex)

    # Suggested parameter adjustments
    suggested_bd_delta: int     # delta to add to border_distance
    suggested_wb_delta: int     # delta to add to word_batch
    suggested_gen_scale: float  # multiplier for generation cap


def estimate_complexity(
    source_text: str,
    subword_count: Optional[int] = None,
) -> ComplexityProfile:
    """Estimate source sentence complexity from text features.

    Args:
        source_text: The source sentence text
        subword_count: Optional number of subword tokens (from tokenizer).
            If provided, used to compute subword expansion ratio.

    Returns:
        ComplexityProfile with analysis and suggested parameter adjustments
    """
    words = source_text.split()
    word_count = len(words)
    char_count = len(source_text)

    if word_count == 0:
        return ComplexityProfile(
            word_count=0, char_count=0, subword_ratio=1.0,
            numeral_density=0.0, avg_word_length=0.0, complexity_score=0.0,
            suggested_bd_delta=0, suggested_wb_delta=0, suggested_gen_scale=1.0,
        )

    # Signal 1: Word count (longer = more complex)
    # Normalize: 5 words = 0.0, 30 words = 1.0
    length_score = max(0.0, min(1.0, (word_count - 5) / 25))

    # Signal 2: Average word length (longer words = morphologically complex)
    avg_word_len = sum(len(w) for w in words) / word_count
    # Normalize: avg 4 chars = 0.0, avg 10 chars = 1.0
    morpho_score = max(0.0, min(1.0, (avg_word_len - 4) / 6))

    # Signal 3: Subword expansion ratio
    if subword_count is not None and word_count > 0:
        subword_ratio = subword_count / word_count
    else:
        subword_ratio = 1.0  # Unknown, assume 1:1
    # Normalize: 1.0 = 0.0, 3.0 = 1.0
    subword_score = max(0.0, min(1.0, (subword_ratio - 1.0) / 2.0))

    # Signal 4: Numeral/entity density
    numeral_count = sum(1 for w in words if re.search(r'\d', w))
    numeral_density = numeral_count / word_count
    # Numerals add complexity (must be preserved exactly)
    numeral_score = min(1.0, numeral_density * 5)  # 20% numerals = 1.0

    # Signal 5: Punctuation density (complex syntax)
    punct_count = sum(1 for c in source_text if c in ',:;()[]{}"-')
    punct_density = punct_count / max(1, char_count)
    punct_score = min(1.0, punct_density * 20)  # 5% punctuation = 1.0

    # Weighted combination
    # Length is the strongest signal, followed by morphological complexity
    complexity_score = (
        0.35 * length_score
        + 0.20 * morpho_score
        + 0.20 * subword_score
        + 0.15 * numeral_score
        + 0.10 * punct_score
    )

    # Map to parameter adjustments
    # bd: simple -> -1, complex -> +2
    if complexity_score < 0.25:
        bd_delta = -1
    elif complexity_score < 0.5:
        bd_delta = 0
    elif complexity_score < 0.75:
        bd_delta = 1
    else:
        bd_delta = 2

    # wb: simple -> -1, complex -> +1
    if complexity_score < 0.3:
        wb_delta = -1
    elif complexity_score < 0.7:
        wb_delta = 0
    else:
        wb_delta = 1

    # gen_scale: simple -> 0.8x, complex -> 1.5x
    gen_scale = 0.8 + 0.7 * complexity_score

    return ComplexityProfile(
        word_count=word_count,
        char_count=char_count,
        subword_ratio=subword_ratio,
        numeral_density=numeral_density,
        avg_word_length=avg_word_len,
        complexity_score=complexity_score,
        suggested_bd_delta=bd_delta,
        suggested_wb_delta=wb_delta,
        suggested_gen_scale=gen_scale,
    )


def adaptive_params_from_complexity(
    source_text: str,
    base_bd: int = 3,
    base_wb: int = 3,
    base_gen_cap: int = 50,
    subword_count: Optional[int] = None,
) -> Tuple[int, int, int]:
    """Compute adaptive parameters from source complexity.

    Args:
        source_text: The source sentence
        base_bd: Base border distance
        base_wb: Base word batch
        base_gen_cap: Base generation cap
        subword_count: Optional subword token count

    Returns:
        (effective_bd, effective_wb, effective_gen_cap)
    """
    profile = estimate_complexity(source_text, subword_count)

    effective_bd = max(1, base_bd + profile.suggested_bd_delta)
    effective_wb = max(1, base_wb + profile.suggested_wb_delta)
    effective_gen_cap = max(10, int(base_gen_cap * profile.suggested_gen_scale))

    return effective_bd, effective_wb, effective_gen_cap


def adaptive_top_p_threshold(
    source_text: str,
    base_threshold: float = 0.8,
    subword_count: Optional[int] = None,
) -> float:
    """Compute adaptive top_p threshold based on source complexity.

    Simple/short sentences get a lower threshold (faster latency, tighter
    frontier), complex/long sentences get a higher threshold (more
    conservative, broader frontier for safety).

    The mapping is linear: complexity 0.0 -> base - 0.1, complexity 1.0 ->
    base + 0.1, capped at [0.5, 0.95].

    Novel approach: no published work on adaptive aggregation thresholds
    for simultaneous machine translation.

    Args:
        source_text: The source sentence
        base_threshold: The baseline top_p_threshold (from config)
        subword_count: Optional subword token count

    Returns:
        Adapted threshold in [0.5, 0.95]
    """
    profile = estimate_complexity(source_text, subword_count)

    # Map: complexity 0 -> base - 0.1, complexity 0.5 -> base, complexity 1 -> base + 0.1
    delta = (profile.complexity_score - 0.5) * 0.2
    threshold = base_threshold + delta

    # Clamp to valid range
    return max(0.5, min(0.95, threshold))


def classify_complexity(score: float) -> str:
    """Classify a complexity score into a human-readable category.

    Args:
        score: Complexity score (0.0-1.0)

    Returns:
        Category string: "simple", "moderate", "complex", or "very_complex"
    """
    if score < 0.25:
        return "simple"
    elif score < 0.5:
        return "moderate"
    elif score < 0.75:
        return "complex"
    else:
        return "very_complex"
