"""Baseline translation backends for comparison against AlignAtt.

Provides simple simultaneous translation strategies commonly used as
lower-bound and upper-bound baselines in simultaneous MT research:

  WaitKBackend         — Fixed wait-k policy (read k words, then alternate)
  FullSentenceBackend  — Wait-until-end / offline translation (quality upper bound)
  EagerBackend         — Translate after every word (latency lower bound)

All backends share the same llama.cpp model and HY-MT prompt format as
AlignAttBackend, but differ in their read/write policy.  They compose
an internal AlignAttBackend for low-level llama.cpp operations (model
loading, tokenization, decoding) rather than duplicating that code.

Usage:
    from nllw.backend_protocol import create_backend

    backend = create_backend("wait-k", source_lang="en", target_lang="fr",
                             model_path="/path/to/model.gguf", k=5)
"""

import math
import threading
from typing import Optional, Tuple, Union

import numpy as np

from nllw import llama_backend as ll
from nllw.alignatt_backend import (
    AlignAttBackend,
    _HYMT_PROMPTS,
    _PROMPT_SUFFIX,
    _resolve_lang_code,
)
from nllw.backend_protocol import SimulMTBackend
from nllw.timed_text import TimedText


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(text: Optional[Union[str, TimedText]]) -> str:
    """Extract raw string from str or TimedText input."""
    if text is None:
        return ""
    if isinstance(text, TimedText):
        return text.text or ""
    return str(text)


# ---------------------------------------------------------------------------
# WaitKBackend
# ---------------------------------------------------------------------------

class WaitKBackend(SimulMTBackend):
    """Fixed wait-k simultaneous translation policy.

    Waits for *k* source words before starting to emit target tokens.
    After the initial wait, generates ``floor(tgt_src_ratio)`` target
    tokens for every new source word, using greedy decoding with no
    attention-based border detection.

    This is the standard wait-k baseline from Ma et al. (2019).

    Parameters
    ----------
    k : int
        Number of source words to wait before starting to write.
    tgt_src_ratio : float
        Expected target/source token ratio.  For each new source word
        after the initial wait, ``floor(tgt_src_ratio)`` target tokens
        are generated.
    gen_cap : int
        Hard cap on target tokens per translate() call (safety limit).
    """

    def __init__(
        self,
        source_lang: str,
        target_lang: str,
        model_path: str = None,
        heads_path: str = None,
        k: int = 3,
        tgt_src_ratio: float = 1.2,
        gen_cap: int = 50,
        n_ctx: int = 2048,
        verbose: bool = False,
        **kwargs,
    ):
        self.k = k
        self.tgt_src_ratio = tgt_src_ratio
        self.gen_cap = gen_cap
        self.verbose = verbose

        # Resolve language codes for prompt building
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.target_lang_iso = _resolve_lang_code(target_lang)
        if self.target_lang_iso is None or self.target_lang_iso not in _HYMT_PROMPTS:
            raise ValueError(
                f"Unsupported target language: {target_lang}. "
                f"Supported: {sorted(_HYMT_PROMPTS.keys())}"
            )

        # Internal AlignAttBackend for model/tokenizer/context operations.
        # We disable its border detection by using a very large border_distance
        # and word_batch=1 so it won't interfere — but we never call its
        # translate() directly; we drive generation ourselves.
        self._inner = AlignAttBackend(
            source_lang=source_lang,
            target_lang=target_lang,
            model_path=model_path,
            heads_path=heads_path,
            border_distance=9999,
            word_batch=1,
            n_ctx=n_ctx,
            verbose=False,
            **kwargs,
        )

        # Alias low-level handles from the inner backend
        self._model = self._inner._model
        self._vocab = self._inner._vocab
        self._nv = self._inner._nv
        self._eos_id = self._inner._eos_id
        self._stop_ids = self._inner._stop_ids
        self._prompt_prefix = self._inner._prompt_prefix
        self._suffix_tokens = self._inner._suffix_tokens

        # Translation state
        self._source_words: list[str] = []
        self._committed_ids: list[int] = []
        self._words_since_last_emit: int = 0
        self._started: bool = False  # Have we passed the wait-k threshold?

        self._lock = threading.Lock()
        self._ctx = None
        self._mem = None

        # Compatibility
        self.input_buffer: list = []

    def _ensure_context(self):
        if self._ctx is None:
            ll.suppress_stderr()
            # No attention weights needed — pure greedy decoding
            self._ctx = ll.create_context(
                self._model, n_ctx=self._inner.n_ctx, n_batch=self._inner.n_ctx,
            )
            self._mem = ll.get_memory(self._ctx)
            ll.restore_stderr()

    def _decode_prompt_and_committed(self):
        """Build full prompt, decode it plus committed tokens, return (prompt_tokens, pos, logit_idx)."""
        accumulated_source = " ".join(self._source_words)
        prompt = self._prompt_prefix + accumulated_source + _PROMPT_SUFFIX
        prompt_tokens = ll.tokenize(self._vocab, prompt, add_bos=True, special=True)

        ll.memory_clear(self._mem)
        ll.decode_batch(self._ctx, prompt_tokens)
        logit_idx = len(prompt_tokens) - 1

        pos = len(prompt_tokens)
        for tid in self._committed_ids:
            ll.decode_single(self._ctx, tid, pos)
            pos += 1
            logit_idx = 0

        return prompt_tokens, pos, logit_idx

    def translate(self, text: Optional[Union[str, TimedText]] = None) -> Tuple[str, str]:
        raw = _extract_text(text)
        if isinstance(text, TimedText):
            self.input_buffer.append(text)
        elif isinstance(text, str):
            self.input_buffer.append(TimedText(text))

        new_words = raw.strip().split()
        if not new_words:
            return "", ""

        self._source_words.extend(new_words)
        self._words_since_last_emit += len(new_words)

        # Wait phase: accumulate k words before starting
        if not self._started:
            if len(self._source_words) < self.k:
                return "", ""
            self._started = True
            # On first emit, generate tokens proportional to all k words
            tokens_to_gen = max(1, math.floor(len(self._source_words) * self.tgt_src_ratio))
        else:
            # After initial wait, generate proportional to new words
            tokens_to_gen = max(1, math.floor(self._words_since_last_emit * self.tgt_src_ratio))

        self._words_since_last_emit = 0
        tokens_to_gen = min(tokens_to_gen, self.gen_cap)

        with self._lock:
            self._ensure_context()

            ll.suppress_stderr()
            try:
                _, pos, logit_idx = self._decode_prompt_and_committed()

                new_tokens = []
                gen_pos = pos
                for _ in range(tokens_to_gen):
                    logits = ll.get_logits_array(self._ctx, logit_idx, self._nv)
                    if logits is None:
                        break
                    next_id = int(np.argmax(logits))
                    if next_id < 0 or next_id in self._stop_ids:
                        break
                    new_tokens.append(next_id)
                    ll.decode_single(self._ctx, next_id, gen_pos)
                    logit_idx = 0
                    gen_pos += 1
            finally:
                ll.restore_stderr()

            # Compute stable delta
            prev_text = (
                ll.tokens_to_text(self._vocab, self._committed_ids, errors="ignore")
                if self._committed_ids
                else ""
            )
            self._committed_ids.extend(new_tokens)
            full_text = (
                ll.tokens_to_text(self._vocab, self._committed_ids, errors="ignore")
                if self._committed_ids
                else ""
            )
            stable = full_text[len(prev_text):]

        if self.verbose:
            print(f"  [wait-k] k={self.k} src_words={len(self._source_words)} "
                  f"gen={len(new_tokens)} stable={stable!r}")

        return stable, ""

    def finish(self) -> str:
        if not self._source_words:
            return ""

        with self._lock:
            self._ensure_context()

            ll.suppress_stderr()
            try:
                _, pos, logit_idx = self._decode_prompt_and_committed()

                new_tokens = []
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

            prev_text = (
                ll.tokens_to_text(self._vocab, self._committed_ids, errors="ignore")
                if self._committed_ids
                else ""
            )
            self._committed_ids.extend(new_tokens)
            full_text = (
                ll.tokens_to_text(self._vocab, self._committed_ids, errors="ignore")
                if self._committed_ids
                else ""
            )
            return full_text[len(prev_text):]

    def reset(self):
        if self._ctx is not None:
            ll.free_context(self._ctx)
            self._ctx = None
            self._mem = None
        self._source_words = []
        self._committed_ids = []
        self._words_since_last_emit = 0
        self._started = False
        self.input_buffer = []

    def set_target_lang(self, target_lang: str):
        iso = _resolve_lang_code(target_lang)
        if iso is None or iso not in _HYMT_PROMPTS:
            raise ValueError(f"Unsupported target language: {target_lang}")
        self.target_lang = target_lang
        self.target_lang_iso = iso
        self._prompt_prefix = _HYMT_PROMPTS[iso]
        self._inner.set_target_lang(target_lang)
        self.reset()


# ---------------------------------------------------------------------------
# FullSentenceBackend
# ---------------------------------------------------------------------------

class FullSentenceBackend(SimulMTBackend):
    """Wait-until-end (offline) translation baseline.

    Accumulates ALL source words and only translates on ``finish()``.
    Each ``translate()`` call returns empty strings.  This represents
    the quality upper bound — the model sees the full source before
    generating any output — at the cost of maximum latency.
    """

    def __init__(
        self,
        source_lang: str,
        target_lang: str,
        model_path: str = None,
        heads_path: str = None,
        n_ctx: int = 2048,
        verbose: bool = False,
        **kwargs,
    ):
        self.verbose = verbose
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.target_lang_iso = _resolve_lang_code(target_lang)
        if self.target_lang_iso is None or self.target_lang_iso not in _HYMT_PROMPTS:
            raise ValueError(
                f"Unsupported target language: {target_lang}. "
                f"Supported: {sorted(_HYMT_PROMPTS.keys())}"
            )

        self._inner = AlignAttBackend(
            source_lang=source_lang,
            target_lang=target_lang,
            model_path=model_path,
            heads_path=heads_path,
            border_distance=9999,
            word_batch=9999,
            n_ctx=n_ctx,
            verbose=False,
            **kwargs,
        )

        self._model = self._inner._model
        self._vocab = self._inner._vocab
        self._nv = self._inner._nv
        self._eos_id = self._inner._eos_id
        self._stop_ids = self._inner._stop_ids
        self._prompt_prefix = self._inner._prompt_prefix
        self._suffix_tokens = self._inner._suffix_tokens

        self._source_words: list[str] = []
        self._lock = threading.Lock()
        self._ctx = None
        self._mem = None

        self.input_buffer: list = []

    def _ensure_context(self):
        if self._ctx is None:
            ll.suppress_stderr()
            self._ctx = ll.create_context(
                self._model, n_ctx=self._inner.n_ctx, n_batch=self._inner.n_ctx,
            )
            self._mem = ll.get_memory(self._ctx)
            ll.restore_stderr()

    def translate(self, text: Optional[Union[str, TimedText]] = None) -> Tuple[str, str]:
        raw = _extract_text(text)
        if isinstance(text, TimedText):
            self.input_buffer.append(text)
        elif isinstance(text, str):
            self.input_buffer.append(TimedText(text))

        new_words = raw.strip().split()
        if new_words:
            self._source_words.extend(new_words)

        # Never emit anything incrementally
        return "", ""

    def finish(self) -> str:
        if not self._source_words:
            return ""

        with self._lock:
            self._ensure_context()

            accumulated_source = " ".join(self._source_words)
            prompt = self._prompt_prefix + accumulated_source + _PROMPT_SUFFIX
            prompt_tokens = ll.tokenize(self._vocab, prompt, add_bos=True, special=True)

            ll.suppress_stderr()
            try:
                ll.memory_clear(self._mem)
                ll.decode_batch(self._ctx, prompt_tokens)
                logit_idx = len(prompt_tokens) - 1

                tokens = []
                pos = len(prompt_tokens)
                for _ in range(500):
                    next_id = ll.argmax_logits(self._ctx, logit_idx, self._nv)
                    if next_id < 0 or next_id in self._stop_ids:
                        break
                    tokens.append(next_id)
                    ll.decode_single(self._ctx, next_id, pos)
                    logit_idx = 0
                    pos += 1
            finally:
                ll.restore_stderr()

            result = (
                ll.tokens_to_text(self._vocab, tokens, errors="ignore")
                if tokens
                else ""
            )

        if self.verbose:
            print(f"  [full-sentence] src_words={len(self._source_words)} "
                  f"gen={len(tokens)} result={result!r}")

        return result

    def reset(self):
        if self._ctx is not None:
            ll.free_context(self._ctx)
            self._ctx = None
            self._mem = None
        self._source_words = []
        self.input_buffer = []

    def set_target_lang(self, target_lang: str):
        iso = _resolve_lang_code(target_lang)
        if iso is None or iso not in _HYMT_PROMPTS:
            raise ValueError(f"Unsupported target language: {target_lang}")
        self.target_lang = target_lang
        self.target_lang_iso = iso
        self._prompt_prefix = _HYMT_PROMPTS[iso]
        self._inner.set_target_lang(target_lang)
        self.reset()


# ---------------------------------------------------------------------------
# EagerBackend
# ---------------------------------------------------------------------------

class EagerBackend(SimulMTBackend):
    """Translate-every-word (eager) baseline.

    Generates target tokens after every single source word with no
    border detection at all.  A fixed ``gen_cap`` limits the maximum
    tokens generated per word.

    This is the latency lower bound but suffers from significant
    quality degradation due to hallucination from insufficient context.

    Parameters
    ----------
    gen_cap : int
        Maximum target tokens to generate per source word.
    """

    def __init__(
        self,
        source_lang: str,
        target_lang: str,
        model_path: str = None,
        heads_path: str = None,
        gen_cap: int = 5,
        n_ctx: int = 2048,
        verbose: bool = False,
        **kwargs,
    ):
        self.gen_cap = gen_cap
        self.verbose = verbose
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.target_lang_iso = _resolve_lang_code(target_lang)
        if self.target_lang_iso is None or self.target_lang_iso not in _HYMT_PROMPTS:
            raise ValueError(
                f"Unsupported target language: {target_lang}. "
                f"Supported: {sorted(_HYMT_PROMPTS.keys())}"
            )

        self._inner = AlignAttBackend(
            source_lang=source_lang,
            target_lang=target_lang,
            model_path=model_path,
            heads_path=heads_path,
            border_distance=9999,
            word_batch=1,
            n_ctx=n_ctx,
            verbose=False,
            **kwargs,
        )

        self._model = self._inner._model
        self._vocab = self._inner._vocab
        self._nv = self._inner._nv
        self._eos_id = self._inner._eos_id
        self._stop_ids = self._inner._stop_ids
        self._prompt_prefix = self._inner._prompt_prefix
        self._suffix_tokens = self._inner._suffix_tokens

        self._source_words: list[str] = []
        self._committed_ids: list[int] = []
        self._lock = threading.Lock()
        self._ctx = None
        self._mem = None

        self.input_buffer: list = []

    def _ensure_context(self):
        if self._ctx is None:
            ll.suppress_stderr()
            self._ctx = ll.create_context(
                self._model, n_ctx=self._inner.n_ctx, n_batch=self._inner.n_ctx,
            )
            self._mem = ll.get_memory(self._ctx)
            ll.restore_stderr()

    def _decode_prompt_and_committed(self):
        """Build full prompt, decode it plus committed tokens, return (prompt_tokens, pos, logit_idx)."""
        accumulated_source = " ".join(self._source_words)
        prompt = self._prompt_prefix + accumulated_source + _PROMPT_SUFFIX
        prompt_tokens = ll.tokenize(self._vocab, prompt, add_bos=True, special=True)

        ll.memory_clear(self._mem)
        ll.decode_batch(self._ctx, prompt_tokens)
        logit_idx = len(prompt_tokens) - 1

        pos = len(prompt_tokens)
        for tid in self._committed_ids:
            ll.decode_single(self._ctx, tid, pos)
            pos += 1
            logit_idx = 0

        return prompt_tokens, pos, logit_idx

    def translate(self, text: Optional[Union[str, TimedText]] = None) -> Tuple[str, str]:
        raw = _extract_text(text)
        if isinstance(text, TimedText):
            self.input_buffer.append(text)
        elif isinstance(text, str):
            self.input_buffer.append(TimedText(text))

        new_words = raw.strip().split()
        if not new_words:
            return "", ""

        self._source_words.extend(new_words)

        # Generate up to gen_cap tokens for each new source word
        tokens_to_gen = min(len(new_words) * self.gen_cap, 200)

        with self._lock:
            self._ensure_context()

            ll.suppress_stderr()
            try:
                _, pos, logit_idx = self._decode_prompt_and_committed()

                new_tokens = []
                gen_pos = pos
                for _ in range(tokens_to_gen):
                    logits = ll.get_logits_array(self._ctx, logit_idx, self._nv)
                    if logits is None:
                        break
                    next_id = int(np.argmax(logits))
                    if next_id < 0 or next_id in self._stop_ids:
                        break
                    new_tokens.append(next_id)
                    ll.decode_single(self._ctx, next_id, gen_pos)
                    logit_idx = 0
                    gen_pos += 1
            finally:
                ll.restore_stderr()

            # Compute stable delta
            prev_text = (
                ll.tokens_to_text(self._vocab, self._committed_ids, errors="ignore")
                if self._committed_ids
                else ""
            )
            self._committed_ids.extend(new_tokens)
            full_text = (
                ll.tokens_to_text(self._vocab, self._committed_ids, errors="ignore")
                if self._committed_ids
                else ""
            )
            stable = full_text[len(prev_text):]

        if self.verbose:
            print(f"  [eager] src_words={len(self._source_words)} "
                  f"gen={len(new_tokens)} stable={stable!r}")

        return stable, ""

    def finish(self) -> str:
        if not self._source_words:
            return ""

        with self._lock:
            self._ensure_context()

            ll.suppress_stderr()
            try:
                _, pos, logit_idx = self._decode_prompt_and_committed()

                new_tokens = []
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

            prev_text = (
                ll.tokens_to_text(self._vocab, self._committed_ids, errors="ignore")
                if self._committed_ids
                else ""
            )
            self._committed_ids.extend(new_tokens)
            full_text = (
                ll.tokens_to_text(self._vocab, self._committed_ids, errors="ignore")
                if self._committed_ids
                else ""
            )
            return full_text[len(prev_text):]

    def reset(self):
        if self._ctx is not None:
            ll.free_context(self._ctx)
            self._ctx = None
            self._mem = None
        self._source_words = []
        self._committed_ids = []
        self.input_buffer = []

    def set_target_lang(self, target_lang: str):
        iso = _resolve_lang_code(target_lang)
        if iso is None or iso not in _HYMT_PROMPTS:
            raise ValueError(f"Unsupported target language: {target_lang}")
        self.target_lang = target_lang
        self.target_lang_iso = iso
        self._prompt_prefix = _HYMT_PROMPTS[iso]
        self._inner.set_target_lang(target_lang)
        self.reset()
