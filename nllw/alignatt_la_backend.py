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
    - SSBD: Self-Speculative Biased Decoding (Zeng et al., arxiv 2509.21740).
      Previous translation serves as draft, verified in one batch forward pass.
      1.3-1.7x speedup with zero quality loss (beta=0.2 recommended).
    - Token-level diff: Compare at token level, not word level, for precision.

Reference:
    - Polak et al. (IWSLT 2023/2025): AlignAtt + LocalAgreement for CUNI's
      winning SimulST system
    - Ma et al. (2019): Original re-translation approach
    - Arivazhagan et al. (2020): Re-translation with incremental decoding
    - Zeng et al. (2025): SSBD for re-translation speedup
"""

import math
import time
import threading
from typing import List, Optional, Dict, Tuple

import numpy as np

from . import llama_backend as ll
from .alignatt import (
    aggregate_ts_weighted_vote,
    check_border,
    check_border_combined,
    check_border_dynamic,
    compute_dynamic_word_batch,
    compute_entropy,
    compute_entropy_change,
    compute_logit_kl,
    compute_prediction_stability,
    is_target_language,
    load_head_config,
    prediction_stability_supports_write,
)
from .complexity import adaptive_params_from_complexity
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


def adaptive_ssbd_beta(
    logits: np.ndarray,
    base_beta: float,
    low_entropy: float = 1.0,
    high_entropy: float = 4.0,
) -> float:
    """Compute adaptive SSBD beta based on model entropy at this position.

    Novel approach combining SSBD (Zeng et al., 2025) with entropy-modulated
    confidence (inspired by arxiv 2508.15371).

    When the model is confident (low entropy), we can be more lenient with
    draft acceptance (higher beta). When uncertain (high entropy), we should
    be stricter (lower beta) to avoid accepting wrong tokens.

    Mapping:
        entropy <= low_entropy  -> beta = base_beta * 1.5 (more lenient)
        entropy >= high_entropy -> beta = base_beta * 0.2 (stricter)
        between                 -> linear interpolation

    Args:
        logits: Raw logit array (n_vocab,)
        base_beta: Base SSBD beta (e.g. 0.2)
        low_entropy: Threshold for "confident" (default: 1.0 nats)
        high_entropy: Threshold for "uncertain" (default: 4.0 nats)

    Returns:
        Adapted beta value
    """
    # Compute entropy
    shifted = logits - np.max(logits)
    exp_l = np.exp(shifted)
    probs = exp_l / exp_l.sum()
    ent = float(-np.sum(probs * np.log(probs + 1e-10)))

    if ent <= low_entropy:
        scale = 1.5  # confident -> more lenient
    elif ent >= high_entropy:
        scale = 0.2  # uncertain -> stricter
    else:
        ratio = (ent - low_entropy) / (high_entropy - low_entropy)
        scale = 1.5 - 1.3 * ratio  # linear from 1.5 to 0.2

    return min(0.95, base_beta * scale)  # cap at 0.95 to avoid degenerate case


def ssbd_accept(logits: np.ndarray, draft_token: int, beta: float) -> bool:
    """Check if a draft token is accepted under biased speculative decoding.

    Biased probability: P'(v) = (1-beta) * P_model(v) + beta * delta(v == draft)
    Accept draft if P'(draft) >= max_{v != draft} P'(v).

    Simplification for greedy decoding:
        Accept if (1-beta)*P(draft) + beta >= (1-beta)*P(argmax)
        i.e., P(draft) >= P(argmax) - beta/(1-beta)

    With logits, we avoid full softmax by using the log-sum-exp trick:
        We accept if draft is argmax, OR if the probability gap is within threshold.

    Args:
        logits: Raw logit array of shape (n_vocab,)
        draft_token: The draft token ID to verify
        beta: Bias strength (0.0 = pure speculative, 0.2 = recommended)

    Returns:
        True if draft token is accepted
    """
    argmax_tok = int(np.argmax(logits))

    # If draft is already the greedy choice, always accept
    if argmax_tok == draft_token:
        return True

    # If no bias, reject (draft != argmax)
    if beta <= 0.0:
        return False

    # Biased acceptance: compute softmax probabilities for draft and argmax only
    # P(draft) = exp(l_draft) / Z, P(argmax) = exp(l_argmax) / Z
    # Accept if P(draft) >= P(argmax) - beta/(1-beta)
    # Equivalently: exp(l_draft - l_argmax) >= 1 - beta/(1-beta)
    # Which is: l_draft >= l_argmax + log(1 - beta/(1-beta))
    threshold = beta / (1.0 - beta)
    # Accept if P(draft)/P(argmax) >= 1 - threshold
    # i.e., exp(l_draft - l_argmax) >= 1 - threshold
    log_ratio = logits[draft_token] - logits[argmax_tok]
    prob_ratio = math.exp(min(log_ratio, 0.0))  # clamp to avoid overflow
    return prob_ratio >= 1.0 - threshold


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

        # Revision history for NE metric computation
        self._revision_history: List[List[int]] = []

        # SSBD stats
        self._ssbd_draft_tokens: int = 0    # Total draft tokens proposed
        self._ssbd_accepted_tokens: int = 0  # Total draft tokens accepted

        # Cross-step tracking for REINA entropy change + prediction stability
        self._prev_first_token_entropy: Optional[float] = None
        self._prev_first_token_logits: Optional[np.ndarray] = None
        self._current_entropy_change: Optional[float] = None
        self._current_pred_stability_write: Optional[bool] = None

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

            # Effective parameters (may be overridden by complexity)
            effective_wb = self.config.word_batch
            self._effective_bd = self.config.border_distance
            accumulated = " ".join(self._source_words)
            if self.config.complexity_adaptive and len(self._source_words) >= 3:
                self._effective_bd, effective_wb, _ = adaptive_params_from_complexity(
                    accumulated,
                    base_bd=self.config.border_distance,
                    base_wb=self.config.word_batch,
                    base_gen_cap=self.config.max_new_per_step,
                )

            # Dynamic word batching (applied on top of complexity)
            if self.config.dynamic_word_batch:
                effective_wb = compute_dynamic_word_batch(
                    effective_wb, len(self._source_words)
                )

            # Word batching
            if self._batch_counter < effective_wb and not is_final:
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
            if self.config.la_two_pass and not is_final and self._prev_full_ids:
                new_full_ids = self._retranslate_two_pass()
            else:
                new_full_ids = self._retranslate(is_final)

            # Track revision history for NE metric
            self._revision_history.append(list(new_full_ids))

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

    def _check_border(self, src_attn: np.ndarray, num_src_tokens: int) -> bool:
        """Check border with all configured enhancements (AMS, temp norm, dynamic, shift-k, info gain, REINA, stability)."""
        bd = getattr(self, '_effective_bd', self.config.border_distance)
        # Use combined check if any multi-signal feature enabled
        use_combined = (
            self.config.shift_k_threshold is not None
            or self.config.info_gain_threshold is not None
            or self.config.entropy_change_threshold is not None
            or self.config.prediction_stability
        )
        if use_combined:
            hit, _, _ = check_border_combined(
                src_attn, self._ts_scores,
                num_src_tokens, bd,
                aggregation=self.config.aggregation,
                adaptive_aggregation=self.config.adaptive_aggregation,
                head_temp_normalize=self.config.head_temp_normalize,
                head_temp_reference=self.config.head_temp_reference,
                shift_k_threshold=self.config.shift_k_threshold,
                prev_attn=getattr(self, '_prev_step_attn', None),
                info_gain_threshold=self.config.info_gain_threshold,
                dynamic_border=self.config.dynamic_border,
                entropy_change=self._current_entropy_change,
                entropy_change_threshold=self.config.entropy_change_threshold,
                pred_stability_write=self._current_pred_stability_write,
            )
            self._prev_step_attn = src_attn.copy()
            return hit

        if self.config.dynamic_border:
            return check_border_dynamic(
                src_attn, self._ts_scores,
                num_src_tokens, bd,
                aggregation=self.config.aggregation,
                adaptive_aggregation=self.config.adaptive_aggregation,
                head_temp_normalize=self.config.head_temp_normalize,
                head_temp_reference=self.config.head_temp_reference,
            )
        else:
            return check_border(
                src_attn, self._ts_scores,
                num_src_tokens, bd,
                aggregation=self.config.aggregation,
                adaptive_aggregation=self.config.adaptive_aggregation,
                head_temp_normalize=self.config.head_temp_normalize,
                head_temp_reference=self.config.head_temp_reference,
            )

    def _lsg_probe(self, last_token: int, pos: int,
                   src_end: int) -> Optional[float]:
        """LSG logit KL probe via KV cache fork (same logic as AlignAttBackend).

        Forks the KV cache, removes last K source tokens, re-decodes the last
        generated token, and compares output logit distributions.

        Returns KL divergence or None on failure.
        """
        prefix_len = len(self._prefix_tokens)
        lsg_k = min(self.config.lsg_k, src_end - prefix_len)
        if lsg_k <= 0:
            return None

        logits_full = ll.get_logits_array(self._ctx, -1, self._nv)
        if logits_full is None:
            return None

        ll.memory_seq_cp(self._mem, 0, 1, 0, pos)
        try:
            ll.memory_seq_rm(self._mem, 1, src_end - lsg_k, src_end)
            ll.memory_seq_rm(self._mem, 1, pos - 1, pos)
            ret = ll.decode_single_at(
                self._ctx, last_token, pos - 1, seq_id=1, output=True
            )
            if ret != 0:
                return None
            logits_reduced = ll.get_logits_array(self._ctx, 0, self._nv)
            if logits_reduced is None:
                return None
            return compute_logit_kl(logits_full, logits_reduced)
        finally:
            ll.memory_seq_rm(self._mem, 1, 0, -1)

    def _update_cross_step_signals(self):
        """Update REINA entropy change and prediction stability signals.

        Called once per translate() before the generation loop. Compares
        the model's first-token logits with those from the previous call
        to detect whether the new source word was informative.
        """
        use_entropy_change = self.config.entropy_change_threshold is not None
        use_pred_stability = self.config.prediction_stability
        if not (use_entropy_change or use_pred_stability):
            return

        first_logits = ll.get_logits_array(self._ctx, -1, self._nv)
        if first_logits is None:
            return

        if use_entropy_change:
            delta_h, cur_h = compute_entropy_change(
                first_logits, self._prev_first_token_entropy
            )
            self._current_entropy_change = delta_h
            self._prev_first_token_entropy = cur_h

        if use_pred_stability:
            top1_rank, topk_ovl = compute_prediction_stability(
                first_logits, self._prev_first_token_logits
            )
            self._current_pred_stability_write = (
                prediction_stability_supports_write(top1_rank, topk_ovl)
            )
            self._prev_first_token_logits = first_logits.copy()

    def _retranslate(self, is_final: bool) -> List[int]:
        """Re-translate the full source using AlignAtt.

        Strategy selection:
        1. SSBD: If enabled and we have a previous translation, uses speculative
           draft verification for 1.3-1.7x speedup.
        2. Forced decoding: If enabled and we have committed tokens, force-decode
           the committed prefix then generate continuation only.
        3. Standard: Full re-translation from scratch.

        Returns:
            Full list of generated token IDs (not just delta)
        """
        use_ssbd = (
            self.config.ssbd_beta is not None
            and self._prev_full_ids
            and not is_final  # On final, generate fully (no border detection needed)
        )

        if use_ssbd:
            return self._retranslate_ssbd()

        use_forced = (
            self.config.la_forced_decode
            and self._committed_ids
            and not is_final
        )

        if use_forced:
            return self._retranslate_forced()

        return self._retranslate_standard(is_final)

    def _retranslate_standard(self, is_final: bool) -> List[int]:
        """Standard autoregressive re-translation with border detection."""
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

        # Update cross-step signals (REINA + prediction stability)
        self._update_cross_step_signals()

        # Generate tokens
        gen_ids = []
        max_gen = 256 if is_final else self.config.max_new_per_step
        consecutive_border_hits = 0  # For border confirmation

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
                    border_hit = self._check_border(src_attn, num_src_tokens)
                    if border_hit:
                        consecutive_border_hits += 1
                        if consecutive_border_hits >= self.config.border_confirm:
                            # LSG confirmation
                            if self.config.lsg_kl_threshold is not None:
                                lsg_kl = self._lsg_probe(next_tok, pos, src_end)
                                if (lsg_kl is not None
                                        and lsg_kl > self.config.lsg_kl_threshold):
                                    consecutive_border_hits = 0
                                    gen_ids.append(next_tok)
                                    continue
                            gen_ids.append(next_tok)
                            break
                    else:
                        consecutive_border_hits = 0

            gen_ids.append(next_tok)

        # Clean up generated tokens from KV cache (we'll regenerate next time)
        ll.memory_seq_rm(self._mem, 0, prefix_len, -1)

        return gen_ids

    def _retranslate_forced(self) -> List[int]:
        """Re-translate with forced decoding of committed prefix.

        CUNI approach (Polak et al., 2025): instead of generating the full
        translation from scratch, force-decode the already-committed tokens
        first, then generate only the continuation.

        Benefits:
        - Fewer tokens to generate autoregressively (faster)
        - Model is conditioned on committed tokens (more consistent)
        - KV cache contains committed prefix (better attention context)

        Sequence decoded: source + suffix + committed_tokens (forced) -> generate new

        Returns:
            Full list of generated token IDs (committed + new generated)
        """
        accumulated_source = " ".join(self._source_words)
        source_tokens = ll.tokenize(
            self._vocab, accumulated_source, add_bos=False, special=False
        )

        prefix_len = len(self._prefix_tokens)

        # Clear KV cache from prefix_len onwards (keep prompt prefix)
        if not ll.memory_seq_rm(self._mem, 0, prefix_len, -1):
            ll.memory_clear(self._mem)
            if self._prefix_tokens:
                ll.decode_batch_at(self._ctx, self._prefix_tokens, pos_start=0)

        src_start = prefix_len
        src_end = prefix_len + len(source_tokens)
        num_src_tokens = src_end - src_start

        # Decode: source + suffix + committed_tokens (forced) in one batch
        forced_tokens = source_tokens + self._suffix_tokens + self._committed_ids
        if forced_tokens:
            ll.decode_batch_at(self._ctx, forced_tokens, pos_start=prefix_len)

        pos = prefix_len + len(forced_tokens)

        # Update cross-step signals (REINA + prediction stability)
        self._update_cross_step_signals()

        # Start with committed tokens as the base
        gen_ids = list(self._committed_ids)

        # Generate new tokens beyond the committed prefix
        max_gen = self.config.max_new_per_step - len(gen_ids)
        consecutive_border_hits = 0

        for step in range(max(0, max_gen)):
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

            # Border detection
            if num_src_tokens > 0:
                ctx_size = ll.n_ctx(self._ctx)
                attn = ll.get_attn_weights(
                    self._ctx, 0, self._n_heads, ctx_size
                )
                if attn is not None and src_end <= attn.shape[1]:
                    src_attn = attn[:, src_start:src_end]
                    border_hit = self._check_border(src_attn, num_src_tokens)
                    if border_hit:
                        consecutive_border_hits += 1
                        if consecutive_border_hits >= self.config.border_confirm:
                            # LSG confirmation
                            if self.config.lsg_kl_threshold is not None:
                                lsg_kl = self._lsg_probe(next_tok, pos, src_end)
                                if (lsg_kl is not None
                                        and lsg_kl > self.config.lsg_kl_threshold):
                                    consecutive_border_hits = 0
                                    gen_ids.append(next_tok)
                                    continue
                            gen_ids.append(next_tok)
                            break
                    else:
                        consecutive_border_hits = 0

            gen_ids.append(next_tok)

        # Clean up generated tokens from KV cache
        ll.memory_seq_rm(self._mem, 0, prefix_len, -1)

        return gen_ids

    def _retranslate_ssbd(self) -> List[int]:
        """Re-translate using SSBD: verify previous translation as draft.

        Self-Speculative Biased Decoding (Zeng et al., 2025):
        1. Decode source + suffix + draft_tokens in ONE batch forward pass
        2. At each draft position, check if model agrees (with beta bias)
        3. Accept all tokens up to first divergence
        4. Resume autoregressive generation from divergence point
        5. Apply border detection during autoregressive phase

        The batch decode is much faster than sequential token generation,
        so if most draft tokens match, we save significant compute.

        Returns:
            Full list of generated token IDs
        """
        beta = self.config.ssbd_beta
        draft_tokens = list(self._prev_full_ids)

        accumulated_source = " ".join(self._source_words)
        source_tokens = ll.tokenize(
            self._vocab, accumulated_source, add_bos=False, special=False
        )

        prefix_len = len(self._prefix_tokens)

        # Clear KV cache from prefix_len onwards (keep prefix)
        if not ll.memory_seq_rm(self._mem, 0, prefix_len, -1):
            ll.memory_clear(self._mem)
            if self._prefix_tokens:
                ll.decode_batch_at(self._ctx, self._prefix_tokens, pos_start=0)

        src_start = prefix_len
        src_end = prefix_len + len(source_tokens)
        num_src_tokens = src_end - src_start

        # === Phase 1: Batch verify draft tokens ===
        # Decode: source + suffix + draft_tokens with logits at all positions
        verify_tokens = source_tokens + self._suffix_tokens + draft_tokens
        if verify_tokens:
            ll.decode_batch_at(
                self._ctx, verify_tokens,
                pos_start=prefix_len, output_last_only=False,
            )

        # Update cross-step signals (REINA + prediction stability)
        self._update_cross_step_signals()

        # Batch index where the first draft token's prediction starts
        # (logits at this index predict what comes after source+suffix)
        verify_start = len(source_tokens) + len(self._suffix_tokens) - 1

        # Track stats
        self._ssbd_draft_tokens += len(draft_tokens)

        # Verify each draft token
        accepted = 0
        use_adaptive = self.config.adaptive_ssbd and beta > 0.0
        for j in range(len(draft_tokens)):
            batch_idx = verify_start + j
            logits = ll.get_logits_array(self._ctx, batch_idx, self._nv)
            if logits is None:
                break

            # Check if draft token in stop set
            if draft_tokens[j] in self._stop_ids:
                break

            # Adaptive beta: scale per-token based on model entropy
            effective_beta = (
                adaptive_ssbd_beta(logits, beta) if use_adaptive else beta
            )

            if ssbd_accept(logits, draft_tokens[j], effective_beta):
                accepted += 1
            else:
                break

        self._ssbd_accepted_tokens += accepted

        # === Phase 2: Prepare for continuation ===
        # Clear KV cache beyond the accepted tokens
        # Keep: prefix + source + suffix + accepted draft tokens
        accepted_end = prefix_len + len(source_tokens) + len(self._suffix_tokens) + accepted
        ll.memory_seq_rm(self._mem, 0, accepted_end, -1)

        gen_ids = list(draft_tokens[:accepted])

        # If all draft tokens accepted and we haven't hit border,
        # continue generating new tokens autoregressively
        pos = accepted_end

        # If we rejected at a position, add the correct token from divergence
        if accepted < len(draft_tokens):
            # Get the correct token at the divergence point
            # (logits at verify_start + accepted predict what should come there)
            batch_idx = verify_start + accepted
            correct_tok = ll.argmax_logits(self._ctx, batch_idx, self._nv)

            # Decode the correction token into the KV cache so Phase 3
            # can continue from fresh logits at pos
            if correct_tok not in self._stop_ids and correct_tok >= 0:
                ll.decode_single_at(self._ctx, correct_tok, pos, seq_id=0)
                pos += 1
                gen_ids.append(correct_tok)
        # else: all draft accepted -- logits at the last batch position already
        # predict the next token, so Phase 3 can proceed directly using
        # argmax_logits(ctx, -1, nv) from the batch decode

        # === Phase 3: Autoregressive continuation with border detection ===
        max_gen = self.config.max_new_per_step - len(gen_ids)
        consecutive_border_hits = 0
        for step in range(max(0, max_gen)):
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

            # Border detection
            if num_src_tokens > 0:
                ctx_size = ll.n_ctx(self._ctx)
                attn = ll.get_attn_weights(
                    self._ctx, 0, self._n_heads, ctx_size
                )
                if attn is not None and src_end <= attn.shape[1]:
                    src_attn = attn[:, src_start:src_end]
                    border_hit = self._check_border(src_attn, num_src_tokens)
                    if border_hit:
                        consecutive_border_hits += 1
                        if consecutive_border_hits >= self.config.border_confirm:
                            # LSG confirmation
                            if self.config.lsg_kl_threshold is not None:
                                lsg_kl = self._lsg_probe(next_tok, pos, src_end)
                                if (lsg_kl is not None
                                        and lsg_kl > self.config.lsg_kl_threshold):
                                    consecutive_border_hits = 0
                                    gen_ids.append(next_tok)
                                    continue
                            gen_ids.append(next_tok)
                            break
                    else:
                        consecutive_border_hits = 0

            gen_ids.append(next_tok)

        # Clean up generated tokens from KV cache
        ll.memory_seq_rm(self._mem, 0, prefix_len, -1)

        return gen_ids

    def _retranslate_two_pass(self) -> List[int]:
        """LA Two-Pass Catch-up: run two independent re-translations and keep
        the one with the longer common prefix with the previous translation.

        CUNI approach (Polak et al., IWSLT 2025): running an extra re-translation
        pass catches instability from attention drift. Between the two outputs,
        the one that's more consistent with previous output is more reliable.

        Motivation: AlignAtt border detection is stochastic -- small attention
        fluctuations can cause different translations. Two passes with different
        initialization expose this variance. Picking the more stable output
        reduces output flicker (lower NE) at the cost of 2x compute.

        The second pass uses a slightly different strategy: if SSBD was used
        for pass 1, pass 2 uses standard re-translation (or vice versa).
        This diversity increases the chance of catching instability.

        Returns:
            The more stable of the two translation token ID lists
        """
        # Pass 1: standard strategy selection
        pass1_ids = self._retranslate(is_final=False)

        # Pass 2: alternative strategy for diversity
        # If pass 1 used SSBD, pass 2 uses forced or standard
        # If pass 1 used forced or standard, pass 2 uses standard (fresh)
        saved_ssbd = self.config.ssbd_beta
        saved_forced = self.config.la_forced_decode

        # Disable SSBD/forced for pass 2 to get a "fresh" re-translation
        self.config.ssbd_beta = None
        self.config.la_forced_decode = False
        pass2_ids = self._retranslate_standard(is_final=False)

        # Restore config
        self.config.ssbd_beta = saved_ssbd
        self.config.la_forced_decode = saved_forced

        # Pick the more stable translation (longer common prefix with previous)
        prev = self._prev_full_ids
        lcp1 = _longest_common_prefix_tokens(prev, pass1_ids)
        lcp2 = _longest_common_prefix_tokens(prev, pass2_ids)

        if lcp2 > lcp1:
            return pass2_ids
        elif lcp1 > lcp2:
            return pass1_ids
        else:
            # Tie-break: prefer the longer translation (more content committed)
            return pass1_ids if len(pass1_ids) >= len(pass2_ids) else pass2_ids

    def _commit_stable_prefix(self, new_full_ids: List[int]) -> str:
        """Find and commit the stable prefix between old and new translations.

        The stable prefix is the longest common prefix between:
        - self._prev_full_ids (previous re-translation)
        - new_full_ids (current re-translation)

        We only commit tokens from this prefix that haven't been committed yet.

        With display_mask_k > 0 (display-only mask-k from SSBD paper):
        - The last k tokens of the stable prefix are hidden from display
        - But they remain in prev_full_ids for SSBD draft verification
        - This reduces NE (output flicker) while maintaining SSBD speedup
        - On is_final, all tokens are committed (mask cleared)

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

        # Apply display-only mask-k: hide last k stable tokens from display
        mask_k = self.config.display_mask_k
        if mask_k > 0:
            display_len = max(len(self._committed_ids), stable_len - mask_k)
        else:
            display_len = stable_len

        if display_len <= len(self._committed_ids):
            # No new stable tokens beyond what we already committed
            return ""

        # Commit the new stable tokens (up to display_len)
        new_stable_ids = new_full_ids[len(self._committed_ids):display_len]
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
        self._revision_history = []
        # Reset cross-step tracking
        self._prev_first_token_entropy = None
        self._prev_first_token_logits = None
        self._current_entropy_change = None
        self._current_pred_stability_write = None

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

    def get_revision_history(self) -> List[List[int]]:
        """Get the revision history for NE metric computation.

        Returns list of full translation token ID lists, one per re-translation step.
        Use with metrics.compute_normalized_erasure() to measure output stability.
        """
        return list(self._revision_history)

    def get_ssbd_stats(self) -> Dict[str, int]:
        """Get SSBD performance statistics.

        Returns:
            Dict with draft_tokens (proposed), accepted_tokens, acceptance_rate
        """
        total = self._ssbd_draft_tokens
        accepted = self._ssbd_accepted_tokens
        return {
            "draft_tokens": total,
            "accepted_tokens": accepted,
            "acceptance_rate": accepted / total if total > 0 else 0.0,
        }

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
