"""AlignAtt translation backend with KV cache delta decoding.

Drop-in replacement for AlignAttBackend that avoids the expensive full
re-decode on every translate() call.  Instead it:

  1. Decodes the constant prompt prefix *once* and caches it.
  2. On each translate() call, compares the new source tokens with the
     previously decoded tokens, finds the divergence point, invalidates
     only the changed suffix via ``memory_seq_rm``, and decodes only the
     new delta tokens.
  3. Falls back to a full clear + re-decode if ``memory_seq_rm`` fails.

Everything else -- border detection, entropy, min_commit, sentence-end
handling, uncertainty tokens, etc. -- is identical to AlignAttBackend.
"""

import json
import os
import threading
from typing import Optional

import numpy as np

from nllw import llama_backend as ll
from nllw.timed_text import TimedText

# Re-use shared constants and helpers from the base module
from nllw.alignatt_backend import (
    _HYMT_PROMPTS,
    _PROMPT_SUFFIX,
    _NLLB_TO_ISO,
    _resolve_lang_code,
    _default_heads_path,
    _aggregate_ts_weighted_avg,
    _auto_detect_prompt_format,
    _build_prompt_config,
    ALIGNATT_SUPPORTED_LANGUAGES,
    PROMPT_FORMATS,
)


class AlignAttKVCacheBackend:
    """Simultaneous translation backend using AlignAtt + HY-MT with KV cache reuse.

    Conforms to the SimulMTBackend protocol (backend_protocol.py).
    Identical API to AlignAttBackend; the only difference is that the prompt
    is decoded incrementally via KV cache delta decoding instead of from
    scratch on every call.

    Args:
        source_lang: Source language (any format: NLLB code, ISO code, or name).
        target_lang: Target language (any format: NLLB code, ISO code, or name).
        model_path: Path to the GGUF model file.
        heads_path: Path to alignment heads JSON (default: bundled universal heads).
        border_distance: How close to source end before stopping generation.
        word_batch: Number of source words to accumulate before translating.
        n_ctx: Context window size for llama.cpp.
        top_k: Number of attention heads to use.
        entropy_veto_threshold: Normalized entropy threshold (0-1) above which a
            token is considered uncertain and triggers a border-like stop.
            None disables the veto. Recommended value: 0.75.
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
        **kwargs,  # Accept extra kwargs for compat with TranslationBackend init patterns
    ):
        # Resolve language codes
        self.source_lang_iso = _resolve_lang_code(source_lang) or source_lang
        self.target_lang_iso = _resolve_lang_code(target_lang)

        self.source_lang = source_lang
        self.target_lang = target_lang
        self.border_distance = border_distance
        self.word_batch = word_batch
        self.n_ctx = n_ctx
        self.top_k = top_k
        self.verbose = verbose

        # Model path
        if model_path is None:
            model_path = os.environ.get("HYMT_MODEL_PATH")
        if model_path is None:
            raise ValueError(
                "AlignAtt backend requires a GGUF model path. "
                "Set model_path or HYMT_MODEL_PATH env var.\n"
                "Download: huggingface-cli download tencent/HY-MT1.5-7B-GGUF "
                "HY-MT1.5-7B-Q8_0.gguf --local-dir ."
            )
        self.model_path = model_path

        # --- Resolve prompt format ---
        if prompt_format is None or prompt_format == "auto":
            detected = _auto_detect_prompt_format(model_path)
            prompt_format = detected if detected else "hymt"

        self.prompt_format = prompt_format

        # Validate target language for hymt format
        if prompt_format == "hymt":
            if self.target_lang_iso is None or self.target_lang_iso not in _HYMT_PROMPTS:
                raise ValueError(
                    f"Unsupported target language for AlignAtt (hymt format): {target_lang}. "
                    f"Supported: {', '.join(ALIGNATT_SUPPORTED_LANGUAGES)}"
                )

        # Build prompt prefix/suffix and stop tokens for this format
        prefix, suffix, stop_strs = _build_prompt_config(
            prompt_format,
            self.source_lang_iso,
            self.target_lang_iso,
            custom_template=custom_template,
        )
        self._prompt_prefix = prefix
        self._prompt_suffix = suffix
        self._stop_token_strs = stop_strs
        self._custom_template = custom_template

        # Load alignment heads
        heads_file = heads_path or _default_heads_path()
        with open(heads_file) as f:
            data = json.load(f)
        heads = data["token_alignment_heads"][:top_k]
        self._head_layers = [h["layer"] for h in heads]
        self._head_indices = [h["head"] for h in heads]
        self._ts_scores = [h["ts"] for h in heads]
        self._num_heads = len(heads)

        # Init llama.cpp (suppress verbose logs for clean TUI)
        ll.suppress_stderr()
        ll.init()
        self._model = ll.load_model(model_path)
        ll.restore_stderr()

        # Optionally load a LoRA adapter (e.g. fine-tuned for a specific domain)
        self._lora_adapter = None
        self._lora_scale = lora_scale
        if lora_path is not None:
            if ll.has_lora_support():
                import logging
                logger = logging.getLogger(__name__)
                logger.info("Loading LoRA adapter: %s (scale=%.2f)", lora_path, lora_scale)
                self._lora_adapter = ll.load_lora(self._model, lora_path, lora_scale)
            else:
                import warnings
                warnings.warn(
                    f"LoRA adapter requested ({lora_path}) but the loaded "
                    f"libllama does not support the LoRA C API. "
                    f"The adapter will be ignored.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        self._vocab = ll.get_vocab(self._model)
        self._nv = ll.n_vocab(self._vocab)
        self._eos_id = ll.vocab_eos(self._vocab)

        # Build stop token set from format-specific stop strings
        self._stop_ids = {self._eos_id}
        for tok_str in self._stop_token_strs:
            tids = ll.tokenize(self._vocab, tok_str, add_bos=False, special=True)
            if len(tids) == 1:
                self._stop_ids.add(tids[0])

        # Tokens that signal model uncertainty -- treat as soft stop during
        # incremental generation (don't commit), but allow in finish().
        self._uncertain_ids = set()
        for tok_str in ["\u2026", "..."]:
            tids = ll.tokenize(self._vocab, tok_str, add_bos=False, special=True)
            if len(tids) == 1:
                self._uncertain_ids.add(tids[0])

        # Sentence-ending tokens -- commit them but stop generating after.
        # This prevents the model from producing multiple short sentences
        # ("Okay, fine. Let's get started.") in a single translate() call.
        self._sentence_end_ids = set()
        for tok_str in [".", "!", "?", "\u3002", "\uff01", "\uff1f"]:
            tids = ll.tokenize(self._vocab, tok_str, add_bos=False, special=True)
            if len(tids) == 1:
                self._sentence_end_ids.add(tids[0])

        # Pre-tokenize prompt prefix (with BOS) and suffix
        self._prefix_tokens = ll.tokenize(
            self._vocab, self._prompt_prefix, add_bos=True, special=True
        )
        self._suffix_tokens = ll.tokenize(
            self._vocab, self._prompt_suffix, add_bos=False, special=True
        )

        # Translation state
        self._source_words = []
        self._committed_ids = []
        self._batch_counter = 0

        # Provide input_buffer attribute for compatibility with code that checks it
        self.input_buffer = []

        # Lock to prevent concurrent access from Textual worker threads
        self._lock = threading.Lock()

        # Context (created once, reused)
        self._ctx = None
        self._mem = None

        # --- KV cache delta state ---
        # Tokens that were decoded into the KV cache on the *previous* call.
        # Layout: prefix_tokens + source_tokens + suffix_tokens + committed_ids
        # We track source_tokens separately so we can diff only that part.
        self._prev_source_tokens = []  # source tokens from last decode
        self._kv_valid_pos = 0  # number of positions valid in KV cache
        self._prefix_decoded = False  # whether the constant prefix is cached

    # --------------------------------------------------------------------- #
    #  Context management
    # --------------------------------------------------------------------- #

    def _ensure_context(self):
        """Create the llama context and decode the constant prompt prefix."""
        if self._ctx is None:
            ll.suppress_stderr()
            self._ctx = ll.create_context(
                self._model, n_ctx=self.n_ctx, n_batch=self.n_ctx
            )
            self._mem = ll.get_memory(self._ctx)
            ll.set_attn_heads(self._ctx, self._head_layers, self._head_indices)
            # Apply LoRA adapter to the new context
            if self._lora_adapter is not None:
                ll.apply_lora(self._ctx, self._lora_adapter, self._lora_scale)
            ll.restore_stderr()
            self._prefix_decoded = False

        if not self._prefix_decoded:
            # Decode the constant prompt prefix and cache it
            ll.suppress_stderr()
            try:
                ll.memory_clear(self._mem)
                ll.decode_batch_at(
                    self._ctx,
                    self._prefix_tokens,
                    pos_start=0,
                    seq_id=0,
                    output_last_only=True,
                )
            finally:
                ll.restore_stderr()
            self._kv_valid_pos = len(self._prefix_tokens)
            self._prev_source_tokens = []
            self._prefix_decoded = True

    # --------------------------------------------------------------------- #
    #  Source range helper (same as AlignAttBackend)
    # --------------------------------------------------------------------- #

    def _find_source_range(self, total_prompt_len):
        """Return (src_start, src_end) token indices within the full prompt."""
        src_start = len(self._prefix_tokens)
        src_end = total_prompt_len - len(self._suffix_tokens)
        if src_end <= src_start:
            return (0, 0)
        return (src_start, src_end)

    # --------------------------------------------------------------------- #
    #  KV cache delta helpers
    # --------------------------------------------------------------------- #

    def _find_diverge_pos(self, new_source_tokens):
        """Find the first index where new source tokens differ from previous.

        Returns the index *within the source token list* (not absolute position).
        """
        prev = self._prev_source_tokens
        min_len = min(len(prev), len(new_source_tokens))
        for i in range(min_len):
            if prev[i] != new_source_tokens[i]:
                return i
        # If one is a prefix of the other, divergence is at the shorter length
        if len(prev) != len(new_source_tokens):
            return min_len
        # Identical
        return len(new_source_tokens)

    def _decode_prompt_delta(self, source_tokens, committed_ids):
        """Decode only the changed portion of the prompt into the KV cache.

        Returns (total_prompt_len, logit_idx) or falls back to full re-decode.
        The logit_idx is the batch index to use with argmax_logits.
        """
        prefix_len = len(self._prefix_tokens)
        suffix_tokens = self._suffix_tokens
        total_prompt_tokens = self._prefix_tokens + source_tokens + suffix_tokens
        total_prompt_len = len(total_prompt_tokens)

        # Find where source tokens diverge from cached version
        diverge_idx = self._find_diverge_pos(source_tokens)

        # Absolute position where divergence starts (in the full sequence)
        diverge_abs = prefix_len + diverge_idx

        # Tokens that need to be decoded: tail of source from diverge + suffix + committed
        new_tail = source_tokens[diverge_idx:]
        delta_tokens = new_tail + suffix_tokens + committed_ids

        if not delta_tokens:
            # Nothing changed and nothing committed -- just return current state
            # KV cache is valid up to kv_valid_pos, logits from last decode
            return total_prompt_len, -1

        # Invalidate KV cache from diverge_abs onward.
        # memory_seq_rm(mem, seq_id, p0, p1) removes positions [p0, p1).
        # Use p1 = -1 to mean "to the end".
        need_full_redecode = False

        if diverge_abs < self._kv_valid_pos:
            # There are cached positions after diverge_abs that are now stale
            ok = ll.memory_seq_rm(self._mem, 0, diverge_abs, -1)
            if not ok:
                need_full_redecode = True
        elif diverge_abs > self._kv_valid_pos:
            # Gap between valid cache and where we need to decode -- shouldn't
            # happen normally, but handle defensively
            need_full_redecode = True

        if need_full_redecode:
            # Full fallback: clear everything and re-decode from scratch
            ll.memory_clear(self._mem)
            all_tokens = total_prompt_tokens + committed_ids
            ll.decode_batch_at(
                self._ctx,
                all_tokens,
                pos_start=0,
                seq_id=0,
                output_last_only=True,
            )
            self._kv_valid_pos = len(all_tokens)
            self._prev_source_tokens = list(source_tokens)
            logit_idx = len(all_tokens) - 1
            return total_prompt_len, logit_idx

        # Decode only the delta tokens starting at diverge_abs
        ll.decode_batch_at(
            self._ctx,
            delta_tokens,
            pos_start=diverge_abs,
            seq_id=0,
            output_last_only=True,
        )
        self._kv_valid_pos = diverge_abs + len(delta_tokens)
        self._prev_source_tokens = list(source_tokens)

        # logit_idx for decode_batch_at with output_last_only is n-1
        logit_idx = len(delta_tokens) - 1

        return total_prompt_len, logit_idx

    # --------------------------------------------------------------------- #
    #  translate()
    # --------------------------------------------------------------------- #

    def translate(self, text: Optional[str | TimedText] = None):
        """Translate incrementally. Returns (stable_translation, buffer).

        Matches TranslationBackend.translate() interface:
        - Accepts str or TimedText
        - Returns (stable_text, buffer_text) where stable is the new committed
          translation and buffer is speculative
        """
        # Extract text from input
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

        # Accumulate source words
        new_words = raw_text.strip().split()
        if not new_words:
            return "", ""

        self._source_words.extend(new_words)
        self._batch_counter += len(new_words)

        # Word batching: wait until we have enough words
        if self._batch_counter < self.word_batch:
            return "", ""
        self._batch_counter = 0

        with self._lock:
            self._ensure_context()

            # Tokenize source text separately (no BOS, no special)
            accumulated_source = " ".join(self._source_words)
            source_tokens = ll.tokenize(
                self._vocab, accumulated_source, add_bos=False, special=False
            )

            # Full prompt length for source range calculation
            total_prompt_len = (
                len(self._prefix_tokens) + len(source_tokens) + len(self._suffix_tokens)
            )

            # Find source token range
            src_start, src_end = self._find_source_range(total_prompt_len)
            n_src = max(0, src_end - src_start)

            # Suppress stderr during decode to avoid ggml Metal logs in TUI
            ll.suppress_stderr()
            try:
                # Delta decode: only the changed portion of source + suffix + committed
                _, logit_idx = self._decode_prompt_delta(
                    source_tokens, self._committed_ids
                )

                # If logit_idx is -1 (nothing decoded -- identical source, no committed),
                # we need to figure out where the logits are. This edge case means
                # the prompt hasn't changed at all, but we still want to generate.
                # Re-decode the last token to get logits.
                if logit_idx == -1:
                    last_pos = self._kv_valid_pos - 1
                    if self._committed_ids:
                        last_tok = self._committed_ids[-1]
                    else:
                        last_tok = self._suffix_tokens[-1]
                    # We already have this in KV cache, but need logits.
                    # Remove and re-decode the last position to get logits.
                    ll.memory_seq_rm(self._mem, 0, last_pos, -1)
                    ll.decode_single_at(self._ctx, last_tok, last_pos, seq_id=0, output=True)
                    self._kv_valid_pos = last_pos + 1
                    logit_idx = 0

                # Generate new tokens with AlignAtt border detection
                new_tokens = []
                buffer_tokens = []
                gen_pos = self._kv_valid_pos

                # Cap generation proportional to source length to prevent
                # hallucination from insufficient context.
                # Tight cap for short sources (prevents "I would" -> "Je le ferais
                # avec plaisir." hallucination), relaxed for longer sources.
                if n_src <= 6:
                    max_gen = max(2, n_src)
                else:
                    max_gen = max(6, int(n_src * 1.5))
                # Adaptive border distance: scale with source length so that
                # long sentences don't get a too-tight threshold.
                # Short (n_src<=6): use fixed border_distance (default 3)
                # Long (n_src>6): grow proportionally (~15% of source length)
                effective_bd = max(self.border_distance, int(n_src * 0.15))
                border_threshold = n_src - effective_bd
                # Guarantee at least min_commit tokens per call so translation
                # keeps up with source input during typing
                min_commit = max(1, len(self._source_words) // 4)
                consecutive_border_hits = 0

                for i in range(max_gen):
                    next_id = ll.argmax_logits(self._ctx, logit_idx, self._nv)
                    if next_id < 0 or next_id in self._stop_ids:
                        break

                    # Uncertainty tokens (...) signal the model doesn't have enough
                    # context -- treat as soft stop, don't commit
                    if next_id in self._uncertain_ids:
                        break

                    ll.decode_single(self._ctx, next_id, gen_pos)
                    logit_idx = 0  # decode_single: logits at batch index 0
                    gen_pos += 1

                    # Sentence-ending punctuation: commit and stop -- but only
                    # when enough text is committed. Prevents premature sentence
                    # boundaries from short context ("I would" -> "Je le souhaite.")
                    total_committed = len(self._committed_ids) + len(new_tokens)
                    if next_id in self._sentence_end_ids and total_committed >= 8:
                        new_tokens.append(next_id)
                        break

                    # AlignAtt border check -- skip for the first min_commit
                    # tokens to ensure translation progress each call
                    border_hit = False
                    if len(new_tokens) >= min_commit and n_src > 0 and border_threshold > 0:
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
                            buffer_tokens.append(next_id)
                            break
                    else:
                        consecutive_border_hits = 0

                    new_tokens.append(next_id)
            finally:
                ll.restore_stderr()

            # After generation: clean up exploration tokens (buffer + any
            # tokens generated beyond what we commit) from the KV cache so
            # that the next call starts with a clean state.
            n_generated = len(new_tokens) + len(buffer_tokens)
            # The KV cache now has positions up to gen_pos.  We want to keep
            # only up to the committed tokens (prompt + old committed + new committed).
            committed_end_pos = gen_pos - len(buffer_tokens)
            if buffer_tokens:
                # Remove the buffer (exploration) tokens from KV cache
                ll.memory_seq_rm(self._mem, 0, committed_end_pos, -1)
                self._kv_valid_pos = committed_end_pos
            else:
                self._kv_valid_pos = gen_pos

            # Compute stable delta
            prev_text = ll.tokens_to_text(
                self._vocab, self._committed_ids, errors="ignore"
            ) if self._committed_ids else ""
            self._committed_ids.extend(new_tokens)
            full_committed = ll.tokens_to_text(
                self._vocab, self._committed_ids, errors="ignore"
            ) if self._committed_ids else ""
            stable = full_committed[len(prev_text):]

            # Buffer text
            buffer = ""
            if buffer_tokens:
                buffer = ll.tokens_to_text(self._vocab, buffer_tokens, errors="ignore")

        if self.verbose:
            print(
                f" \033[32m{stable}\033[0m \033[35m{buffer}\033[0m"
            )

        return stable, buffer

    # --------------------------------------------------------------------- #
    #  finish()
    # --------------------------------------------------------------------- #

    def finish(self):
        """Flush remaining translation (generate until EOS).

        Returns the remaining translation text.
        """
        if not self._source_words:
            return ""

        with self._lock:
            self._ensure_context()

            # Tokenize source text separately
            accumulated_source = " ".join(self._source_words)
            source_tokens = ll.tokenize(
                self._vocab, accumulated_source, add_bos=False, special=False
            )

            ll.suppress_stderr()
            try:
                # Delta decode: source + suffix + committed
                _, logit_idx = self._decode_prompt_delta(
                    source_tokens, self._committed_ids
                )

                # Handle edge case where nothing was decoded
                if logit_idx == -1:
                    last_pos = self._kv_valid_pos - 1
                    if self._committed_ids:
                        last_tok = self._committed_ids[-1]
                    else:
                        last_tok = self._suffix_tokens[-1]
                    ll.memory_seq_rm(self._mem, 0, last_pos, -1)
                    ll.decode_single_at(self._ctx, last_tok, last_pos, seq_id=0, output=True)
                    self._kv_valid_pos = last_pos + 1
                    logit_idx = 0

                # Generate until EOS
                new_tokens = []
                gen_pos = self._kv_valid_pos
                for _ in range(200):
                    next_id = ll.argmax_logits(self._ctx, logit_idx, self._nv)
                    if next_id < 0 or next_id in self._stop_ids:
                        break
                    new_tokens.append(next_id)
                    ll.decode_single(self._ctx, next_id, gen_pos)
                    logit_idx = 0
                    gen_pos += 1
            finally:
                ll.restore_stderr()

            # Update KV valid position
            self._kv_valid_pos = gen_pos

            prev_text = ll.tokens_to_text(
                self._vocab, self._committed_ids, errors="ignore"
            ) if self._committed_ids else ""
            self._committed_ids.extend(new_tokens)
            full_text = ll.tokens_to_text(
                self._vocab, self._committed_ids, errors="ignore"
            ) if self._committed_ids else ""
            remaining = full_text[len(prev_text):]

            return remaining

    # --------------------------------------------------------------------- #
    #  reset()
    # --------------------------------------------------------------------- #

    def reset(self):
        """Reset state for a new sentence.

        Clears all KV cache state so the next translate() starts fresh.
        """
        if self._ctx is not None:
            ll.free_context(self._ctx)
            self._ctx = None
            self._mem = None
        self._source_words = []
        self._committed_ids = []
        self._batch_counter = 0
        self.input_buffer = []
        # Clear KV cache delta state
        self._prev_source_tokens = []
        self._kv_valid_pos = 0
        self._prefix_decoded = False

    # --------------------------------------------------------------------- #
    #  set_target_lang()
    # --------------------------------------------------------------------- #

    def set_target_lang(self, target_lang):
        """Change the target language and reset state."""
        iso = _resolve_lang_code(target_lang)
        if iso is None:
            raise ValueError(f"Unsupported target language: {target_lang}")
        # Rebuild prompt config for the new target language
        prefix, suffix, stop_strs = _build_prompt_config(
            self.prompt_format,
            self.source_lang_iso,
            iso,
            custom_template=self._custom_template,
        )
        self.target_lang = target_lang
        self.target_lang_iso = iso
        self._prompt_prefix = prefix
        self._prompt_suffix = suffix
        self._stop_token_strs = stop_strs
        # Rebuild stop token IDs
        self._stop_ids = {self._eos_id}
        for tok_str in self._stop_token_strs:
            tids = ll.tokenize(self._vocab, tok_str, add_bos=False, special=True)
            if len(tids) == 1:
                self._stop_ids.add(tids[0])
        # Re-tokenize prefix and suffix since the prompt changed
        self._prefix_tokens = ll.tokenize(
            self._vocab, self._prompt_prefix, add_bos=True, special=True
        )
        self._suffix_tokens = ll.tokenize(
            self._vocab, self._prompt_suffix, add_bos=False, special=True
        )
        self.reset()

    # --------------------------------------------------------------------- #
    #  cleanup
    # --------------------------------------------------------------------- #

    def __del__(self):
        if hasattr(self, "_ctx") and self._ctx is not None:
            try:
                ll.free_context(self._ctx)
            except Exception:
                pass
        if hasattr(self, "_model") and self._model is not None:
            try:
                ll.free_model(self._model)
            except Exception:
                pass
