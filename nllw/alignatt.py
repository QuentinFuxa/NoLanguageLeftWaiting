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
"""

import numpy as np
from typing import List, Optional, Tuple


def aggregate_ts_weighted_vote(
    src_attn: np.ndarray,
    ts_scores: List[float],
) -> int:
    """Compute the TS-weighted attention vote across alignment heads.

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


def check_border(
    src_attn: np.ndarray,
    ts_scores: List[float],
    n_src_tokens: int,
    border_distance: int,
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

    Returns:
        True if border hit (should stop generating)
    """
    border_threshold = n_src_tokens - border_distance
    if border_threshold <= 0:
        return False

    attended_pos = aggregate_ts_weighted_vote(src_attn, ts_scores)
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
