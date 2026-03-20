"""
AlignAtt border detection engine for simultaneous translation.

Implements the core AlignAtt algorithm:
1. Source words arrive incrementally
2. Build prompt: prefix + [context] + source + suffix + committed_output
3. Generate tokens with LLM (greedy argmax)
4. After each generated token, extract attention from top-K alignment heads
5. TS-weighted vote: if attended position >= source_len - border_distance -> STOP
6. At sentence boundary: commit translation, reset context, start next segment
7. KV cache reuse: only re-decode changed tokens (3-5x speedup)

Reference: "AlignAtt: Using Attention-Based Audio-Translation Alignment
for Simultaneous Speech Translation" (Polak et al., 2023)
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from nllw.llama_backend import LlamaContext, LlamaModel, init as llama_init
from nllw.prompts import PromptFormat, build_prompt, find_source_token_range
from nllw.metrics import LatencyMetrics, compute_latency_metrics


@dataclass
class AlignAttConfig:
    """Configuration for AlignAtt border detection."""
    border_distance: int = 3
    top_k_heads: int = 10
    max_new_per_step: int = 50
    max_gen_final: int = 256
    word_batch: int = 1
    context_sentences: int = 0
    segment_reset: bool = True
    n_ctx: int = 2048
    target_lang: str = "zh"


@dataclass
class AlignmentHead:
    """A single alignment head with its TS score."""
    layer: int
    head: int
    ts: float


@dataclass
class TranslationSegment:
    """Result of translating one segment."""
    source: str
    translation: str
    num_source_words: int
    num_target_tokens: int
    delays: List[int] = field(default_factory=list)
    emissions: List[dict] = field(default_factory=list)
    time_ms: float = 0.0


def load_heads(path: str, top_k: int = 10) -> List[AlignmentHead]:
    """Load alignment heads from a JSON file.

    Expected format: {"token_alignment_heads": [{"layer": L, "head": H, "ts": S}, ...]}
    """
    with open(path) as f:
        data = json.load(f)
    heads = []
    for h in data["token_alignment_heads"][:top_k]:
        heads.append(AlignmentHead(layer=h["layer"], head=h["head"], ts=h["ts"]))
    return heads


def aggregate_ts_weighted_vote(
    src_attn: np.ndarray,
    ts_scores: List[float],
) -> int:
    """Compute TS-weighted vote over attention heads to determine attended source position.

    Each head votes for its argmax source position, weighted by its TS score.
    Returns the source position with the highest weighted vote.

    Args:
        src_attn: Attention weights of shape (num_heads, num_source_tokens).
        ts_scores: TS (Token Similarity) score for each head.

    Returns:
        The source position index with highest weighted vote.
    """
    head_argmaxes = np.argmax(src_attn, axis=1)
    weighted: Dict[int, float] = {}
    for h, pos in enumerate(head_argmaxes):
        pos = int(pos)
        weighted[pos] = weighted.get(pos, 0.0) + ts_scores[h]
    return max(weighted, key=weighted.get)


def is_target_language(text: str, target_lang: str) -> bool:
    """Check if text is in the target language (not English).

    Used to filter context: don't inject English hallucinations into context.
    """
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
        # Latin-script languages (de, fr, it, etc.) can't be distinguished from English
        return True


class AlignAttEngine:
    """Stateful AlignAtt simultaneous translation engine.

    Maintains state across source word arrivals within a segment:
    - KV cache for the current prompt
    - Committed output token IDs
    - Previous segment contexts

    Supports two modes:
    1. KV cache reuse (recommended): One context per segment, delta decode
    2. Full re-decode: New context per source word (slower, simpler)

    Usage:
        engine = AlignAttEngine(model, config, heads, fmt)
        for word in asr_words:
            result = engine.feed_word(word["text"], is_final=word["is_final"])
            if result.text:
                print(result.text)
    """

    def __init__(
        self,
        model: LlamaModel,
        config: AlignAttConfig,
        heads: List[AlignmentHead],
        fmt: PromptFormat,
    ):
        self.model = model
        self.config = config
        self.heads = heads[:config.top_k_heads]
        self.fmt = fmt

        self._head_layers = [h.layer for h in self.heads]
        self._head_indices = [h.head for h in self.heads]
        self._ts_scores = [h.ts for h in self.heads]
        self._num_heads = len(self.heads)

        # Build stop token IDs
        self._stop_ids = {model.eos_id}
        for tok_str in fmt.suffix.split("\n"):
            tok_str = tok_str.strip()
            if tok_str:
                tids = model.tokenize(tok_str, add_bos=False, special=True)
                if len(tids) == 1:
                    self._stop_ids.add(tids[0])
        # Common EOS tokens across models
        for sid in [2, 151643, 151645, 107]:
            self._stop_ids.add(sid)

        self._suffix_tokens = model.tokenize(fmt.suffix, add_bos=False, special=True)

        # Segment state
        self._ctx: Optional[LlamaContext] = None
        self._prefix_tokens: Optional[List[int]] = None
        self._prev_source_tokens: List[int] = []
        self._committed_ids: List[int] = []
        self._source_words: List[str] = []
        self._prev_contexts: List[Dict[str, str]] = []
        self._batch_counter: int = 0
        self._delays: List[int] = []

    def feed_word(
        self,
        word: str,
        is_final: bool = False,
        emission_time: Optional[float] = None,
    ) -> "WordResult":
        """Feed a source word and get translation output.

        Args:
            word: The source word to add.
            is_final: Whether this is the last word of the current sentence/segment.
            emission_time: ASR emission time (for latency tracking).

        Returns:
            WordResult with new translated text and metadata.
        """
        self._source_words.append(word)
        self._batch_counter += 1

        # Word batching: skip until N words accumulated (unless final)
        if self._batch_counter < self.config.word_batch and not is_final:
            return WordResult(text="", is_final=False, border_stop=False)

        self._batch_counter = 0
        return self._translate_step(is_final)

    def _translate_step(self, is_final: bool) -> "WordResult":
        """Run one translation step with current accumulated source."""
        t0 = time.time()
        accumulated_source = " ".join(self._source_words)

        # Create context if needed (first word of segment)
        if self._ctx is None:
            self._init_segment()

        # Tokenize current source
        cur_source_tokens = self.model.tokenize(
            accumulated_source, add_bos=False, special=False,
        )

        prefix_len = len(self._prefix_tokens)

        # Find shared prefix between old and new source tokens (for KV cache delta)
        shared = 0
        for i in range(min(len(self._prev_source_tokens), len(cur_source_tokens))):
            if self._prev_source_tokens[i] == cur_source_tokens[i]:
                shared += 1
            else:
                break

        diverge_pos = prefix_len + shared

        # Remove everything from diverge_pos onwards in KV cache
        if not self._ctx.memory_seq_rm(0, diverge_pos, -1):
            # Hybrid model fallback: full clear + re-decode
            self._ctx.memory_clear()
            redecode = self._prefix_tokens + cur_source_tokens[:shared]
            if redecode:
                self._ctx.decode_batch(redecode, pos_start=0, output_last_only=True)
            diverge_pos = len(redecode)

        # New tokens to decode: remaining source + suffix + committed
        new_source_tail = cur_source_tokens[shared:]
        delta_tokens = new_source_tail + self._suffix_tokens + self._committed_ids
        total_new = len(delta_tokens)

        src_start = prefix_len
        src_end = prefix_len + len(cur_source_tokens)
        num_src_tokens = src_end - src_start

        # Safety check: context overflow
        if diverge_pos + total_new >= self.config.n_ctx - 10 or num_src_tokens <= 0:
            self._prev_source_tokens = cur_source_tokens
            return WordResult(text="", is_final=is_final, border_stop=False)

        if total_new > 0:
            self._ctx.decode_batch(delta_tokens, pos_start=diverge_pos, output_last_only=True)

        pos = diverge_pos + total_new

        # Generate new tokens with border detection
        new_ids = []
        stopped_at_border = False
        max_gen = self.config.max_gen_final if is_final else self.config.max_new_per_step

        for step in range(max_gen):
            next_tok = self._ctx.argmax_logits(-1)
            if next_tok in self._stop_ids or next_tok < 0:
                break

            self._ctx.decode_single(next_tok, pos, seq_id=0)
            pos += 1

            # Check border (skip on final -- generate freely)
            if not is_final and num_src_tokens > 0:
                attn = self._ctx.get_attn_weights(0, self._num_heads)
                if attn is not None and src_end <= attn.shape[1]:
                    src_attn = attn[:, src_start:src_end]
                    attended_pos = aggregate_ts_weighted_vote(src_attn, self._ts_scores)
                    if attended_pos >= num_src_tokens - self.config.border_distance:
                        stopped_at_border = True
                        break

            new_ids.append(next_tok)

        # Clean up exploration tokens from KV cache on border stop
        if stopped_at_border:
            cached_pos = diverge_pos + total_new + len(new_ids)
            self._ctx.memory_seq_rm(0, cached_pos, -1)

        self._prev_source_tokens = cur_source_tokens

        # Decode text (full sequence to avoid byte-fallback mojibake)
        prev_text = self.model.tokens_to_text(
            self._committed_ids, errors="ignore",
        ) if self._committed_ids else ""

        if new_ids:
            # Track delays: each new token was produced at current source word index
            word_idx = len(self._source_words) - 1
            for _ in new_ids:
                self._delays.append(word_idx)
            self._committed_ids.extend(new_ids)

        full_text = self.model.tokens_to_text(
            self._committed_ids, errors="ignore",
        ) if self._committed_ids else ""
        new_text = full_text[len(prev_text):]

        elapsed = (time.time() - t0) * 1000

        # Segment reset on final word
        if self.config.segment_reset and is_final:
            self._finish_segment(accumulated_source, full_text)

        return WordResult(
            text=new_text,
            is_final=is_final,
            border_stop=stopped_at_border,
            num_new_tokens=len(new_ids),
            time_ms=elapsed,
        )

    def _init_segment(self):
        """Initialize a new segment: create context, decode prefix."""
        ctx_arg = (
            self._prev_contexts[-self.config.context_sentences:]
            if self._prev_contexts and self.config.context_sentences > 0
            else None
        )

        # Build prefix string
        prefix_str = self.fmt.prefix
        if ctx_arg and self.fmt.context_tpl and self.fmt.context_entry:
            entries = "".join(self.fmt.context_entry.format(**c) for c in ctx_arg)
            prefix_str += self.fmt.context_tpl.format(context=entries)

        self._prefix_tokens = self.model.tokenize(prefix_str, add_bos=True, special=True)

        self._ctx = LlamaContext(
            self.model,
            n_ctx=self.config.n_ctx,
            n_batch=self.config.n_ctx,
            attn_weights=True,
        )
        self._ctx.set_attn_heads(self._head_layers, self._head_indices)

        # Decode prefix once
        self._ctx.decode_batch(self._prefix_tokens, pos_start=0, output_last_only=True)
        self._prev_source_tokens = []

    def _finish_segment(self, source: str, translation: str):
        """Finalize segment: save context, reset state for next segment."""
        if source and translation:
            if is_target_language(translation, self.config.target_lang):
                self._prev_contexts.append({
                    "source": source,
                    "translation": translation,
                })
                max_ctx = self.config.context_sentences
                if max_ctx > 0 and len(self._prev_contexts) > max_ctx:
                    self._prev_contexts = self._prev_contexts[-max_ctx:]
            else:
                self._prev_contexts = []

        self._committed_ids = []
        self._source_words = []
        self._delays = []

        if self._ctx is not None:
            self._ctx.close()
            self._ctx = None
        self._prev_source_tokens = []

    def get_segment_metrics(self) -> LatencyMetrics:
        """Get latency metrics for the current/last segment."""
        return compute_latency_metrics(
            self._delays,
            len(self._source_words),
            len(self._committed_ids),
        )

    def reset(self):
        """Full reset: clear all state including context history."""
        if self._ctx is not None:
            self._ctx.close()
            self._ctx = None
        self._prefix_tokens = None
        self._prev_source_tokens = []
        self._committed_ids = []
        self._source_words = []
        self._prev_contexts = []
        self._batch_counter = 0
        self._delays = []

    def close(self):
        """Release resources."""
        if self._ctx is not None:
            self._ctx.close()
            self._ctx = None


@dataclass
class WordResult:
    """Result of feeding one word to the AlignAtt engine."""
    text: str = ""
    is_final: bool = False
    border_stop: bool = False
    num_new_tokens: int = 0
    time_ms: float = 0.0


def stream_translate(
    model: LlamaModel,
    source_text: str,
    heads: List[AlignmentHead],
    fmt: PromptFormat,
    config: Optional[AlignAttConfig] = None,
) -> TranslationSegment:
    """Translate a complete source sentence with AlignAtt streaming simulation.

    Simulates word-by-word arrival and returns the full translation with metrics.
    Uses KV cache reuse for efficiency.

    Args:
        model: Loaded LlamaModel.
        source_text: Complete source sentence.
        heads: Alignment heads with TS scores.
        fmt: Prompt format.
        config: AlignAtt configuration (uses defaults if None).

    Returns:
        TranslationSegment with translation, delays, and timing.
    """
    if config is None:
        config = AlignAttConfig()

    engine = AlignAttEngine(model, config, heads, fmt)

    source_words = source_text.split()
    emissions = []
    t0 = time.time()

    for i, word in enumerate(source_words):
        is_final = (i == len(source_words) - 1)
        result = engine.feed_word(word, is_final=is_final)
        emissions.append({
            "src_words": i + 1,
            "new_tokens": result.num_new_tokens,
            "new_text": result.text,
            "border": result.border_stop,
        })

    elapsed = (time.time() - t0) * 1000
    translation = model.tokens_to_text(engine._committed_ids, errors="ignore") if engine._committed_ids else ""
    delays = list(engine._delays)
    n_target = len(engine._committed_ids)

    engine.close()

    return TranslationSegment(
        source=source_text,
        translation=translation,
        num_source_words=len(source_words),
        num_target_tokens=n_target,
        delays=delays,
        emissions=emissions,
        time_ms=elapsed,
    )
