"""AlignAtt + LocalAgreement hybrid translation backend.

Re-Translation strategy: instead of permanently committing tokens on each call,
the full candidate translation is re-generated from scratch every time. A
LocalAgreement comparison between the current and previous candidate determines
which prefix is stable enough to commit.

This yields significantly higher quality because:
  1. Each translation sees the full accumulated source (no stale committed prefix).
  2. Only tokens that survive two consecutive generations are committed.
  3. The model is free to revise speculative output before it becomes permanent.

Uses composition (wraps AlignAttBackend) rather than inheritance.
"""

import threading
from typing import Optional

from nllw import llama_backend as ll
from nllw.alignatt_backend import (
    AlignAttBackend,
    _aggregate_ts_weighted_avg,
)
from nllw.timed_text import TimedText


def _common_prefix_length(a: list[int], b: list[int]) -> int:
    """Return the length of the longest common prefix between two token lists."""
    length = 0
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            break
        length = i + 1
    return length


class AlignAttLocalAgreementBackend:
    """Simultaneous translation using AlignAtt border detection + LocalAgreement.

    Instead of committing tokens directly from a single generation pass, this
    backend re-generates the *full* candidate translation from scratch on every
    call.  LocalAgreement (common-prefix between consecutive candidates) decides
    which tokens are stable.  Only the stable prefix grows monotonically;
    everything else is treated as speculative buffer.

    Conforms to the SimulMTBackend protocol (backend_protocol.py).

    Args:
        source_lang: Source language (any format: NLLB code, ISO code, or name).
        target_lang: Target language (any format: NLLB code, ISO code, or name).
        model_path: Path to the GGUF model file.
        heads_path: Path to alignment heads JSON (default: bundled universal heads).
        border_distance: How close to source end before stopping generation.
        word_batch: Number of source words to accumulate before translating.
        n_ctx: Context window size for llama.cpp.
        top_k: Number of attention heads to use.
        verbose: Print debug info.
    """

    def __init__(
        self,
        source_lang,
        target_lang,
        model_path: str = None,
        heads_path: str = None,
        prompt_format: str = "hymt",
        custom_template: str = None,
        border_distance: int = 3,
        word_batch: int = 3,
        n_ctx: int = 2048,
        top_k: int = 10,
        lora_path: str = None,
        lora_scale: float = 1.0,
        verbose: bool = False,
        **kwargs,
    ):
        # Delegate model loading and tokenizer setup to the inner backend.
        # We set word_batch=1 on the inner backend because we handle batching
        # ourselves; the inner backend is only used for its model/vocab/context.
        self._inner = AlignAttBackend(
            source_lang=source_lang,
            target_lang=target_lang,
            model_path=model_path,
            heads_path=heads_path,
            prompt_format=prompt_format,
            custom_template=custom_template,
            border_distance=border_distance,
            word_batch=1,  # we handle batching
            n_ctx=n_ctx,
            top_k=top_k,
            lora_path=lora_path,
            lora_scale=lora_scale,
            verbose=False,  # we print our own debug
            **kwargs,
        )

        # Expose key attributes from inner backend
        self.source_lang = self._inner.source_lang
        self.target_lang = self._inner.target_lang
        self.source_lang_iso = self._inner.source_lang_iso
        self.target_lang_iso = self._inner.target_lang_iso
        self.border_distance = border_distance
        self.word_batch = word_batch
        self.n_ctx = n_ctx
        self.top_k = top_k
        self.verbose = verbose

        # Provide input_buffer attribute for compatibility
        self.input_buffer = []

        # --- LocalAgreement state ---
        self._already_committed_ids: list[int] = []   # tokens confirmed stable (monotonic)
        self._previous_candidate_ids: list[int] = []   # full candidate from last call

        # --- Source accumulation state ---
        self._source_words: list[str] = []
        self._batch_counter: int = 0

        self._lock = threading.Lock()

    # --- Convenience accessors for inner backend resources ---

    @property
    def _model(self):
        return self._inner._model

    @property
    def _vocab(self):
        return self._inner._vocab

    @property
    def _nv(self):
        return self._inner._nv

    @property
    def _eos_id(self):
        return self._inner._eos_id

    @property
    def _stop_ids(self):
        return self._inner._stop_ids

    @property
    def _uncertain_ids(self):
        return self._inner._uncertain_ids

    @property
    def _sentence_end_ids(self):
        return self._inner._sentence_end_ids

    @property
    def _prompt_prefix(self):
        return self._inner._prompt_prefix

    @property
    def _prompt_suffix(self):
        return self._inner._prompt_suffix

    @property
    def _suffix_tokens(self):
        return self._inner._suffix_tokens

    @property
    def _head_layers(self):
        return self._inner._head_layers

    @property
    def _head_indices(self):
        return self._inner._head_indices

    @property
    def _ts_scores(self):
        return self._inner._ts_scores

    @property
    def _num_heads(self):
        return self._inner._num_heads

    def _ensure_context(self):
        """Ensure llama context exists (delegates to inner backend)."""
        self._inner._ensure_context()

    @property
    def _ctx(self):
        return self._inner._ctx

    @property
    def _mem(self):
        return self._inner._mem

    def _find_source_range(self, tokens):
        return self._inner._find_source_range(tokens)

    # -----------------------------------------------------------------
    # Core translate: re-translation + LocalAgreement
    # -----------------------------------------------------------------

    def translate(self, text: Optional[str | TimedText] = None):
        """Translate incrementally using re-translation + LocalAgreement.

        Returns (stable_translation, buffer) where stable is the NEW committed
        text delta and buffer is speculative.
        """
        # --- Input handling (same as AlignAttBackend) ---
        if text is None:
            return "", ""
        if isinstance(text, TimedText):
            self.input_buffer.append(text)
            raw_text = text.text or ""
        elif isinstance(text, str):
            self.input_buffer.append(TimedText(text))
            raw_text = text
        else:
            raw_text = str(text)

        new_words = raw_text.strip().split()
        if not new_words:
            return "", ""

        self._source_words.extend(new_words)
        self._batch_counter += len(new_words)

        # Word batching
        if self._batch_counter < self.word_batch:
            return "", ""
        self._batch_counter = 0

        with self._lock:
            self._ensure_context()

            # Build full prompt
            accumulated_source = " ".join(self._source_words)
            prompt = self._prompt_prefix + accumulated_source + self._prompt_suffix
            prompt_tokens = ll.tokenize(self._vocab, prompt, add_bos=True, special=True)

            # Source range for border detection
            src_start, src_end = self._find_source_range(prompt_tokens)
            n_src = max(0, src_end - src_start)

            ll.suppress_stderr()
            try:
                # ---- Re-translation: generate full candidate from scratch ----
                ll.memory_clear(self._mem)
                ll.decode_batch(self._ctx, prompt_tokens)
                logit_idx = len(prompt_tokens) - 1

                # Generation cap (same heuristics as AlignAttBackend)
                if n_src <= 6:
                    max_gen = max(2, n_src)
                else:
                    max_gen = max(6, int(n_src * 1.5))

                # Adaptive border distance
                effective_bd = max(self.border_distance, int(n_src * 0.15))
                border_threshold = n_src - effective_bd

                candidate_ids: list[int] = []
                gen_pos = len(prompt_tokens)
                consecutive_border_hits = 0

                for i in range(max_gen):
                    next_id = ll.argmax_logits(self._ctx, logit_idx, self._nv)
                    if next_id < 0 or next_id in self._stop_ids:
                        break
                    if next_id in self._uncertain_ids:
                        break

                    ll.decode_single(self._ctx, next_id, gen_pos)
                    logit_idx = 0
                    gen_pos += 1

                    # Sentence-end: include token and stop (if enough context)
                    if next_id in self._sentence_end_ids and len(candidate_ids) >= 8:
                        candidate_ids.append(next_id)
                        break

                    # Border detection (skip first few tokens to ensure progress)
                    min_gen_for_border = max(1, len(self._source_words) // 4)
                    border_hit = False
                    if len(candidate_ids) >= min_gen_for_border and n_src > 0 and border_threshold > 0:
                        raw_attn = ll.get_attn_weights(
                            self._ctx, 0, self._num_heads, ll.n_ctx(self._ctx)
                        )
                        if raw_attn is not None and raw_attn.shape[1] >= src_end:
                            src_attn = raw_attn[:, src_start:src_end]
                            if src_attn.shape[1] > 0:
                                voted_pos = _aggregate_ts_weighted_avg(src_attn, self._ts_scores)
                                if voted_pos >= border_threshold:
                                    border_hit = True

                    if border_hit:
                        consecutive_border_hits += 1
                        if consecutive_border_hits >= 2:
                            # Don't include this token — it's past the border
                            break
                    else:
                        consecutive_border_hits = 0

                    candidate_ids.append(next_id)
            finally:
                ll.restore_stderr()

            # ---- LocalAgreement: compare with previous candidate ----
            agreed_len = _common_prefix_length(self._previous_candidate_ids, candidate_ids)
            agreed_prefix = candidate_ids[:agreed_len]
            buffer_ids = candidate_ids[agreed_len:]

            # Compute the NEW committed tokens (delta over what was already committed)
            new_committed_ids = agreed_prefix[len(self._already_committed_ids):]

            # Update state
            self._already_committed_ids = list(agreed_prefix)
            self._previous_candidate_ids = list(candidate_ids)

            # Convert to text
            if new_committed_ids:
                # Compute text delta by diffing full committed text
                prev_text = ll.tokens_to_text(
                    self._vocab, self._already_committed_ids[:-len(new_committed_ids)], errors="ignore"
                ) if len(self._already_committed_ids) > len(new_committed_ids) else ""
                full_committed_text = ll.tokens_to_text(
                    self._vocab, self._already_committed_ids, errors="ignore"
                )
                stable = full_committed_text[len(prev_text):]
            else:
                stable = ""

            buffer = ""
            if buffer_ids:
                buffer = ll.tokens_to_text(self._vocab, buffer_ids, errors="ignore")

        if self.verbose:
            n_committed = len(self._already_committed_ids)
            n_candidate = len(candidate_ids)
            n_agreed = agreed_len
            print(
                f" [LA] committed={n_committed} candidate={n_candidate} agreed={n_agreed} "
                f"\033[32m{stable}\033[0m \033[35m{buffer}\033[0m"
            )

        return stable, buffer

    # -----------------------------------------------------------------
    # finish: generate freely, return everything after already_committed
    # -----------------------------------------------------------------

    def finish(self):
        """Flush remaining translation — generate freely until EOS.

        Returns the text after the already-committed prefix.
        """
        if not self._source_words:
            return ""

        with self._lock:
            self._ensure_context()

            accumulated_source = " ".join(self._source_words)
            prompt = self._prompt_prefix + accumulated_source + self._prompt_suffix
            prompt_tokens = ll.tokenize(self._vocab, prompt, add_bos=True, special=True)

            ll.suppress_stderr()
            try:
                ll.memory_clear(self._mem)
                ll.decode_batch(self._ctx, prompt_tokens)
                logit_idx = len(prompt_tokens) - 1

                # Re-decode already committed tokens to seed the context
                pos = len(prompt_tokens)
                for tid in self._already_committed_ids:
                    ll.decode_single(self._ctx, tid, pos)
                    pos += 1
                    logit_idx = 0

                # Generate until EOS — no border detection
                new_tokens: list[int] = []
                for _ in range(200):
                    next_id = ll.argmax_logits(self._ctx, logit_idx, self._nv)
                    if next_id < 0 or next_id in self._stop_ids:
                        break
                    new_tokens.append(next_id)
                    ll.decode_single(self._ctx, next_id, pos)
                    logit_idx = 0
                    pos += 1
            finally:
                ll.restore_stderr()

            # Text that was already committed
            prev_text = ll.tokens_to_text(
                self._vocab, self._already_committed_ids, errors="ignore"
            ) if self._already_committed_ids else ""

            # Full text = committed + new
            all_ids = self._already_committed_ids + new_tokens
            full_text = ll.tokens_to_text(
                self._vocab, all_ids, errors="ignore"
            ) if all_ids else ""

            remaining = full_text[len(prev_text):]
            return remaining

    # -----------------------------------------------------------------
    # reset / set_target_lang
    # -----------------------------------------------------------------

    def reset(self):
        """Reset all state for a new sentence."""
        self._inner.reset()
        self._source_words = []
        self._batch_counter = 0
        self._already_committed_ids = []
        self._previous_candidate_ids = []
        self.input_buffer = []

    def set_target_lang(self, target_lang):
        """Change the target language and reset state."""
        self._inner.set_target_lang(target_lang)
        # Sync attributes
        self.target_lang = self._inner.target_lang
        self.target_lang_iso = self._inner.target_lang_iso
        self.reset()

    def __del__(self):
        # Inner backend handles cleanup via its own __del__
        pass


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import sys

    model_path = os.environ.get("HYMT_MODEL_PATH")
    if not model_path:
        print("Set HYMT_MODEL_PATH to run the smoke test.")
        sys.exit(1)

    print("=== AlignAtt + LocalAgreement smoke test ===")
    backend = AlignAttLocalAgreementBackend(
        source_lang="en",
        target_lang="fr",
        model_path=model_path,
        verbose=True,
        word_batch=3,
    )

    # Simulate incremental source input
    words = "The weather is really nice today in Paris".split()
    all_stable = ""
    last_buffer = ""

    for w in words:
        stable, buffer = backend.translate(w + " ")
        all_stable += stable
        last_buffer = buffer
        if stable:
            print(f"  +stable: {stable!r}  buffer: {buffer!r}")

    # Finish
    remaining = backend.finish()
    all_stable += remaining
    print(f"\nFinal translation: {all_stable}")
    print("=== done ===")

    backend.reset()
