"""LocalAgreement + AlignAtt hybrid backend for simultaneous translation.

LocalAgreement (LA) is an output stability policy for simultaneous MT:
    1. On each new source word, re-translate the ENTIRE source from scratch
    2. Compare the new translation with the previous one
    3. Only commit the "stable prefix" -- the longest prefix that agrees
       between consecutive translations
    4. On is_final, commit everything

This gives much more stable output than pure incremental AlignAtt at the
cost of re-translation overhead. The hybrid approach uses AlignAtt internally
for each full re-translation (border detection still applies -- it controls
how far ahead the model generates given available source).

Optimizations:
    - KV cache reuse: The prompt prefix is identical across re-translations,
      so we keep the prefix KV cache and only re-decode from the source onwards.
    - SSBD potential: Previous translation can serve as a speculative draft
      for the next re-translation (future work, see arxiv 2509.21740).
    - Token-level diff: Compare at token level, not word level, for precision.

Reference:
    - Polak et al. (IWSLT 2023/2025): AlignAtt + LocalAgreement for CUNI's
      winning SimulST system
    - Ma et al. (2019): Original re-translation approach
    - Arivazhagan et al. (2020): Re-translation with incremental decoding
"""

import time
import threading
from typing import List, Optional, Dict

import numpy as np

from . import llama_backend as ll
from .alignatt import (
    aggregate_ts_weighted_vote,
    check_border,
    check_border_dynamic,
    compute_entropy,
    is_target_language,
    load_head_config,
)
from .prompts import get_prompt_format, detect_model_family, PromptFormat
from .backend_protocol import (
    SimulMTBackend,
    BackendConfig,
    TranslationStep,
    register_backend,
)
from .alignatt_backend import _find_heads_for_model


def _longest_common_prefix_tokens(a: List[int], b: List[int]) -> int:
    """Find the length of the longest common prefix between two token lists.

    This is the core of LocalAgreement: we only commit tokens that are
    stable across consecutive re-translations.

    Args:
        a: Previous translation token IDs
        b: Current translation token IDs

    Returns:
        Number of tokens in the common prefix
    """
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _longest_common_prefix_words(a: str, b: str) -> str:
    """Find the longest common prefix at word level.

    Used as a fallback when token IDs are not available (e.g., web backend).

    Args:
        a: Previous translation text
        b: Current translation text

    Returns:
        The stable prefix text (words that agree)
    """
    words_a = a.split()
    words_b = b.split()
    n = min(len(words_a), len(words_b))
    common = 0
    for i in range(n):
        if words_a[i] == words_b[i]:
            common = i + 1
        else:
            break
    if common == 0:
        return ""
    return " ".join(words_a[:common])


@register_backend("alignatt-la")
class AlignAttLABackend(SimulMTBackend):
    """AlignAtt + LocalAgreement hybrid backend.

    Combines AlignAtt's attention-based border detection with LocalAgreement's
    output stability. On each new source word:
        1. Re-translate the full source using AlignAtt (with border detection)
        2. Compare new translation with previous
        3. Commit only the stable prefix (tokens that agree)

    This trades latency for stability -- translations don't flicker or change.

    Config params (in addition to standard AlignAtt params):
        word_batch: Re-translate every N source words (default: 1 for LA)
        la_min_agreement: Minimum agreement length to commit (default: 1 token)
    """

    def __init__(self, config: BackendConfig):
        super().__init__(config)
        self._lock = threading.Lock()

        # Load head configuration
        heads_path = config.heads_path
        if not heads_path:
            heads_path = _find_heads_for_model(config.model_path, config.direction)
        if not heads_path:
            raise ValueError(
                f"No heads config found for {config.model_path} / {config.direction}. "
                f"Provide heads_path explicitly or run head detection."
            )

        head_data = load_head_config(heads_path)
        top_k = min(config.top_k_heads, head_data["n_heads"])
        self._head_layers = head_data["layers"][:top_k]
        self._head_indices = head_data["heads"][:top_k]
        self._ts_scores = head_data["ts_scores"][:top_k]
        self._n_heads = top_k

        # Prompt format
        model_family = detect_model_family(config.model_path)
        self._prompt_fmt = get_prompt_format(model_family, config.direction)

        # Initialize llama.cpp
        ll.init()
        self._model = ll.load_model(config.model_path)
        self._vocab = ll.get_vocab(self._model)
        self._nv = ll.n_vocab(self._vocab)
        self._eos_id = ll.vocab_eos(self._vocab)

        # Build stop token set
        self._stop_ids = {self._eos_id}
        for tok_str in self._prompt_fmt.suffix.split("\n"):
            tok_str = tok_str.strip()
            if tok_str:
                tids = ll.tokenize(self._vocab, tok_str, add_bos=False, special=True)
                if len(tids) == 1:
                    self._stop_ids.add(tids[0])
        for sid in [2, 151643, 151645, 107]:
            self._stop_ids.add(sid)

        # Pre-tokenize suffix
        self._suffix_tokens = ll.tokenize(
            self._vocab, self._prompt_fmt.suffix, add_bos=False, special=True
        )

        # LA state
        self._ctx = None
        self._mem = None
        self._prefix_tokens: Optional[List[int]] = None
        self._source_words: List[str] = []
        self._prev_full_ids: List[int] = []  # Previous full translation tokens
        self._committed_ids: List[int] = []  # Actually committed (stable) tokens
        self._committed_text: str = ""       # Text of committed tokens
        self._prev_contexts: List[Dict[str, str]] = []
        self._batch_counter = 0

    def translate(self, source_word: str, is_final: bool = False,
                  emission_time: float = 0.0) -> TranslationStep:
        """Process a new source word through AlignAtt + LocalAgreement.

        Key difference from pure AlignAtt: we re-translate the full source
        each time and only commit the stable prefix.
        """
        t0 = time.time()

        with self._lock:
            self._source_words.append(source_word)
            self._batch_counter += 1

            # Word batching
            if self._batch_counter < self.config.word_batch and not is_final:
                return TranslationStep(
                    text="",
                    is_final=False,
                    source_words_seen=len(self._source_words),
                    generation_time_ms=(time.time() - t0) * 1000,
                )
            self._batch_counter = 0

            # Create context on first call per segment
            if self._ctx is None:
                self._init_segment()

            # Full re-translation from scratch
            new_full_ids = self._retranslate(is_final)

            if is_final:
                # On final, commit everything from the new translation
                new_text = self._commit_all(new_full_ids)
            else:
                # LocalAgreement: only commit the stable prefix
                new_text = self._commit_stable_prefix(new_full_ids)

            elapsed_ms = (time.time() - t0) * 1000

            # Save current translation for next comparison
            self._prev_full_ids = new_full_ids

            # Handle segment reset if final
            if is_final:
                self._handle_segment_end()

            return TranslationStep(
                text=new_text,
                is_final=is_final,
                committed_tokens=len(new_text.split()) if new_text else 0,
                stopped_at_border=not is_final and len(new_full_ids) > 0,
                source_words_seen=len(self._source_words),
                generation_time_ms=elapsed_ms,
            )

    def _retranslate(self, is_final: bool) -> List[int]:
        """Re-translate the full source from scratch using AlignAtt.

        Reuses the prefix KV cache (prompt prefix is always the same).
        Only re-decodes from source tokens onwards.

        Returns:
            Full list of generated token IDs (not just delta)
        """
        accumulated_source = " ".join(self._source_words)
        source_tokens = ll.tokenize(
            self._vocab, accumulated_source, add_bos=False, special=False
        )

        prefix_len = len(self._prefix_tokens)

        # Clear KV cache from prefix_len onwards (keep prefix)
        if not ll.memory_seq_rm(self._mem, 0, prefix_len, -1):
            # Fallback: full re-decode
            ll.memory_clear(self._mem)
            if self._prefix_tokens:
                ll.decode_batch_at(self._ctx, self._prefix_tokens, pos_start=0)

        # Decode: source + suffix (no committed tokens -- fresh translation)
        decode_tokens = source_tokens + self._suffix_tokens
        if decode_tokens:
            ll.decode_batch_at(self._ctx, decode_tokens, pos_start=prefix_len)

        pos = prefix_len + len(decode_tokens)
        src_start = prefix_len
        src_end = prefix_len + len(source_tokens)
        num_src_tokens = src_end - src_start

        # Generate tokens
        gen_ids = []
        max_gen = 256 if is_final else self.config.max_new_per_step

        for step in range(max_gen):
            next_tok = ll.argmax_logits(self._ctx, -1, self._nv)
            if next_tok in self._stop_ids or next_tok < 0:
                break

            # Entropy veto
            if self.config.entropy_veto_threshold is not None:
                logits = ll.get_logits_array(self._ctx, -1, self._nv)
                if logits is not None:
                    ent = compute_entropy(logits)
                    if ent > self.config.entropy_veto_threshold:
                        break

            ll.decode_single_at(self._ctx, next_tok, pos, seq_id=0)
            pos += 1

            # Border detection (not on final)
            if not is_final and num_src_tokens > 0:
                ctx_size = ll.n_ctx(self._ctx)
                attn = ll.get_attn_weights(
                    self._ctx, 0, self._n_heads, ctx_size
                )
                if attn is not None and src_end <= attn.shape[1]:
                    src_attn = attn[:, src_start:src_end]
                    if self.config.dynamic_border:
                        border_hit = check_border_dynamic(
                            src_attn, self._ts_scores,
                            num_src_tokens, self.config.border_distance,
                            aggregation=self.config.aggregation,
                        )
                    else:
                        border_hit = check_border(
                            src_attn, self._ts_scores,
                            num_src_tokens, self.config.border_distance,
                            aggregation=self.config.aggregation,
                        )
                    if border_hit:
                        gen_ids.append(next_tok)
                        break

            gen_ids.append(next_tok)

        # Clean up generated tokens from KV cache (we'll regenerate next time)
        ll.memory_seq_rm(self._mem, 0, prefix_len, -1)

        return gen_ids

    def _commit_stable_prefix(self, new_full_ids: List[int]) -> str:
        """Find and commit the stable prefix between old and new translations.

        The stable prefix is the longest common prefix between:
        - self._prev_full_ids (previous re-translation)
        - new_full_ids (current re-translation)

        We only commit tokens from this prefix that haven't been committed yet.

        Returns:
            Newly committed text (the delta from previously committed)
        """
        if not new_full_ids:
            return ""

        if not self._prev_full_ids:
            # First translation -- don't commit anything yet (no agreement)
            return ""

        # Find longest common prefix at token level
        stable_len = _longest_common_prefix_tokens(self._prev_full_ids, new_full_ids)

        if stable_len <= len(self._committed_ids):
            # No new stable tokens beyond what we already committed
            return ""

        # Commit the new stable tokens
        new_stable_ids = new_full_ids[len(self._committed_ids):stable_len]
        if not new_stable_ids:
            return ""

        # Decode: full committed sequence for correct text
        prev_text = self._committed_text
        self._committed_ids.extend(new_stable_ids)
        self._committed_text = ll.tokens_to_text(
            self._vocab, self._committed_ids, errors="ignore"
        )

        return self._committed_text[len(prev_text):]

    def _commit_all(self, new_full_ids: List[int]) -> str:
        """Commit all tokens from the final translation.

        Called when is_final=True -- no stability check needed.

        Returns:
            Newly committed text
        """
        if not new_full_ids:
            return ""

        prev_text = self._committed_text
        self._committed_ids = list(new_full_ids)
        self._committed_text = ll.tokens_to_text(
            self._vocab, self._committed_ids, errors="ignore"
        )

        return self._committed_text[len(prev_text):]

    def _init_segment(self):
        """Initialize a new segment: create context, decode prefix."""
        ctx_arg = None
        if self._prev_contexts and self.config.context_sentences > 0:
            ctx_arg = self._prev_contexts[-self.config.context_sentences:]

        prefix_str = self._prompt_fmt.prefix
        if ctx_arg and self._prompt_fmt.context_tpl and self._prompt_fmt.context_entry:
            entries = "".join(
                self._prompt_fmt.context_entry.format(**c) for c in ctx_arg
            )
            prefix_str += self._prompt_fmt.context_tpl.format(context=entries)

        self._prefix_tokens = ll.tokenize(
            self._vocab, prefix_str, add_bos=True, special=True
        )

        self._ctx = ll.create_context(
            self._model,
            n_ctx=self.config.n_ctx,
            n_batch=self.config.n_ctx,
            attn_weights=True,
        )
        ll.set_attn_heads(self._ctx, self._head_layers, self._head_indices)
        self._mem = ll.get_memory(self._ctx)

        # Decode prefix once (reused across re-translations)
        ll.decode_batch_at(self._ctx, self._prefix_tokens, pos_start=0)

    def _handle_segment_end(self):
        """Handle end of a sentence segment."""
        completed_source = " ".join(self._source_words)
        completed_translation = self.get_full_translation()

        if completed_source and completed_translation:
            if is_target_language(completed_translation, self.config.target_lang):
                self._prev_contexts.append({
                    "source": completed_source,
                    "translation": completed_translation,
                })
                max_ctx = self.config.context_sentences
                if max_ctx > 0 and len(self._prev_contexts) > max_ctx:
                    self._prev_contexts = self._prev_contexts[-max_ctx:]
            else:
                self._prev_contexts = []

        # Reset for next segment
        self._committed_ids = []
        self._committed_text = ""
        self._prev_full_ids = []
        self._source_words = []
        self._batch_counter = 0

        if self._ctx is not None:
            ll.free_context(self._ctx)
            self._ctx = None
            self._mem = None

    def reset(self):
        """Reset state for next segment."""
        with self._lock:
            self._handle_segment_end()

    def get_full_translation(self) -> str:
        """Get full committed translation text."""
        return self._committed_text

    def close(self):
        """Free all resources."""
        with self._lock:
            if self._ctx is not None:
                ll.free_context(self._ctx)
                self._ctx = None
            if self._model is not None:
                ll.free_model(self._model)
                self._model = None
            ll.cleanup()
