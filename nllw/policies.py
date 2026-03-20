"""
Read/write policies for simultaneous translation.

Each policy decides when to stop generating (WRITE) and wait for more source (READ).

Policies:
- AlignAtt: Stop when attention heads focus near source boundary (default, best quality)
- WaitK: Read K words before first WRITE, then alternate READ/WRITE
- Confidence: Stop when next-token entropy exceeds threshold
- FixedRate: Generate fixed number of tokens per source word
- NoBorder: Generate freely until EOS (offline translation, upper bound)
"""

import abc
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from nllw.llama_backend import LlamaContext
from nllw.alignatt import AlignmentHead, aggregate_ts_weighted_vote


class SimulMTPolicy(abc.ABC):
    """Abstract base class for SimulMT read/write policies."""

    @abc.abstractmethod
    def should_stop(
        self,
        ctx: LlamaContext,
        step: int,
        src_start: int,
        src_end: int,
        is_final: bool,
    ) -> bool:
        """Decide whether to stop generating and READ more source.

        Args:
            ctx: The llama context (for attention/logit extraction).
            step: Current generation step within this WRITE phase.
            src_start: Start token index of source in the prompt.
            src_end: End token index of source in the prompt.
            is_final: Whether this is the final source word (never stop).

        Returns:
            True if generation should stop (READ more source).
        """
        ...


@dataclass
class AlignAttPolicy(SimulMTPolicy):
    """AlignAtt border detection: stop when attention focuses near source boundary.

    Best overall quality. Requires alignment heads to be set on context.
    """
    heads: List[AlignmentHead] = None
    border_distance: int = 3

    def __post_init__(self):
        if self.heads:
            self._ts_scores = [h.ts for h in self.heads]
            self._num_heads = len(self.heads)
        else:
            self._ts_scores = []
            self._num_heads = 0

    def should_stop(self, ctx, step, src_start, src_end, is_final):
        if is_final:
            return False  # Generate freely on final word

        num_src = src_end - src_start
        if num_src <= 0 or self._num_heads == 0:
            return False

        attn = ctx.get_attn_weights(0, self._num_heads)
        if attn is None or src_end > attn.shape[1]:
            return False

        src_attn = attn[:, src_start:src_end]
        attended_pos = aggregate_ts_weighted_vote(src_attn, self._ts_scores)
        return attended_pos >= num_src - self.border_distance


@dataclass
class WaitKPolicy(SimulMTPolicy):
    """Wait-K: Read K words before first WRITE, then generate freely after each word.

    Simple and predictable latency, but lower quality than AlignAtt.
    """
    wait_k: int = 3
    tokens_per_word: int = 5
    _words_seen: int = 0

    def reset(self):
        self._words_seen = 0

    def should_stop(self, ctx, step, src_start, src_end, is_final):
        if is_final:
            return False
        return step >= self.tokens_per_word


@dataclass
class ConfidencePolicy(SimulMTPolicy):
    """Confidence-based: stop when next-token entropy exceeds threshold.

    Higher entropy = model is uncertain = should READ more source.
    Generally worse than AlignAtt (COMET 0.468 vs 0.865).
    """
    entropy_threshold: float = 2.0
    _n_vocab: int = 0

    def should_stop(self, ctx, step, src_start, src_end, is_final):
        if is_final:
            return False

        logits = ctx.get_logits_array(-1)
        if logits is None:
            return False

        # Stable softmax + entropy
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / probs.sum()
        mask = probs > 0
        entropy = -np.sum(probs[mask] * np.log(probs[mask]))

        return entropy > self.entropy_threshold


@dataclass
class FixedRatePolicy(SimulMTPolicy):
    """Fixed-rate: generate a fixed number of tokens per source word.

    Very predictable but poor quality (COMET 0.293-0.334).
    """
    tokens_per_word: int = 2

    def should_stop(self, ctx, step, src_start, src_end, is_final):
        if is_final:
            return False
        return step >= self.tokens_per_word


@dataclass
class NoBorderPolicy(SimulMTPolicy):
    """No border: generate freely until EOS/max. Equivalent to offline translation.

    Upper bound on quality, but highest latency.
    """

    def should_stop(self, ctx, step, src_start, src_end, is_final):
        return False  # Never stop early


def create_policy(
    name: str,
    heads: Optional[List[AlignmentHead]] = None,
    **kwargs,
) -> SimulMTPolicy:
    """Factory for creating policies by name.

    Args:
        name: Policy name (attention, wait-k, confidence, fixed-rate, no-border).
        heads: Alignment heads (required for attention policy).
        **kwargs: Policy-specific arguments.

    Returns:
        Configured SimulMTPolicy instance.
    """
    if name == "attention":
        if not heads:
            raise ValueError("AlignAtt policy requires alignment heads")
        return AlignAttPolicy(
            heads=heads,
            border_distance=kwargs.get("border_distance", 3),
        )
    elif name == "wait-k":
        return WaitKPolicy(
            wait_k=kwargs.get("wait_k", 3),
            tokens_per_word=kwargs.get("tokens_per_word", 5),
        )
    elif name == "confidence":
        return ConfidencePolicy(
            entropy_threshold=kwargs.get("entropy_threshold", 2.0),
        )
    elif name == "fixed-rate":
        return FixedRatePolicy(
            tokens_per_word=kwargs.get("tokens_per_word", 2),
        )
    elif name == "no-border":
        return NoBorderPolicy()
    else:
        raise ValueError(
            f"Unknown policy: {name}. "
            f"Available: attention, wait-k, confidence, fixed-rate, no-border"
        )
