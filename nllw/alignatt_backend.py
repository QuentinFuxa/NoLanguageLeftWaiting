"""AlignAtt SimulMT backend with KV cache support.

This is the primary backend for simultaneous machine translation. It
implements the AlignAtt algorithm using llama.cpp with attention weight
extraction, supporting both KV-cache-reuse mode (fast) and context-recreation
mode (simple).

Architecture:
    prompt = prefix + [context] + source_words + suffix + committed_tokens
    For each new source word:
        1. Find the delta between old and new source tokens
        2. Use KV cache: only decode changed tokens (3-5x speedup)
        3. Generate tokens greedily, checking attention border after each
        4. When attention reaches border region -> STOP, wait for more source
        5. On sentence boundary (is_final) -> commit all, reset for next segment

Critical bugs fixed (from iwslt26-sst):
    - logit_idx: After decode_single, use batch index 0 for logits (not KV pos)
    - Border threshold guard: n_src - border_distance can go negative
    - Thread safety: Lock around all llama_context operations
"""

import json
import os
import sys
import time
import threading
from typing import List, Optional, Dict

import numpy as np

from . import llama_backend as ll
from .alignatt import (
    aggregate,
    aggregate_ts_weighted_vote,
    check_border,
    check_border_dynamic,
    check_border_combined,
    check_border_shift_k,
    compute_dynamic_word_batch,
    compute_entropy,
    compute_entropy_change,
    compute_logit_kl,
    compute_prediction_stability,
    entropy_change_supports_write,
    is_target_language,
    load_head_config,
    list_aggregation_methods,
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


def _find_heads_for_model(model_path: str, direction: str) -> Optional[str]:
    """Auto-discover head config file for a model + direction.

    Searches nllw/heads/configs/ for matching JSON files.
    """
    configs_dir = os.path.join(os.path.dirname(__file__), "heads", "configs")
    if not os.path.isdir(configs_dir):
        return None

    model_name = os.path.basename(model_path).lower()
    direction_str = direction.replace("-", "_")

    # Build search patterns
    patterns = []
    if "hy-mt" in model_name or "hymt" in model_name:
        if "1.8b" in model_name or "1_8b" in model_name:
            patterns.append(f"hymt1.8b_{direction_str}")
        else:
            patterns.append(f"hymt_{direction_str}")
    if "qwen3.5" in model_name or "qwen3_5" in model_name:
        for size in ["4b", "9b"]:
            if size in model_name:
                patterns.append(f"qwen3.5_{size}_{direction_str}")
        patterns.append(f"qwen3.5_4b_{direction_str}")  # default
    if "qwen3" in model_name and "qwen3.5" not in model_name:
        for size in ["4b", "8b", "14b"]:
            if size in model_name:
                patterns.append(f"qwen3_{size}_{direction_str}")
        patterns.append(f"qwen3_4b_{direction_str}")  # default
    if "eurollm" in model_name:
        patterns.append(f"{direction_str}_eurollm")
    if "tower" in model_name:
        patterns.append(f"{direction_str}_tower")
    if "gemma" in model_name or "translategemma" in model_name:
        patterns.append(f"{direction_str}_translategemma")

    for pattern in patterns:
        for fname in os.listdir(configs_dir):
            if fname.endswith(".json") and pattern in fname.lower():
                return os.path.join(configs_dir, fname)

    return None


@register_backend("alignatt")
class AlignAttBackend(SimulMTBackend):
    """AlignAtt simultaneous translation backend with KV cache reuse.

    This is the main backend. It supports:
    - KV cache delta decoding (only re-decode changed tokens)
    - Context injection from previous segments
    - Entropy veto (optional)
    - Adaptive border distance
    - Word batching
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
        # Common EOS tokens across models
        for sid in [2, 151643, 151645, 107]:
            self._stop_ids.add(sid)

        # Pre-tokenize suffix
        self._suffix_tokens = ll.tokenize(
            self._vocab, self._prompt_fmt.suffix, add_bos=False, special=True
        )

        # State
        self._ctx = None
        self._mem = None
        self._prefix_tokens = None
        self._prev_source_tokens: List[int] = []
        self._committed_ids: List[int] = []
        self._source_words: List[str] = []
        self._prev_contexts: List[Dict[str, str]] = []
        self._batch_counter = 0
        self._prev_step_attn: Optional[np.ndarray] = None  # For info gain
        # Cross-step tracking for REINA entropy change + prediction stability
        self._prev_first_token_entropy: Optional[float] = None
        self._prev_first_token_logits: Optional[np.ndarray] = None
        self._current_entropy_change: Optional[float] = None
        self._current_pred_stability_write: Optional[bool] = None
        # Attention position history for monotonicity tracking (within generation loop)
        self._gen_positions_history: List[float] = []

    def translate(self, source_word: str, is_final: bool = False,
                  emission_time: float = 0.0) -> TranslationStep:
        """Process a new source word through AlignAtt."""
        t0 = time.time()

        with self._lock:
            self._source_words.append(source_word)
            self._batch_counter += 1

            # Effective parameters (may be overridden by complexity)
            effective_bd = self.config.border_distance
            effective_wb = self.config.word_batch
            effective_gen_cap = self.config.max_new_per_step

            # Complexity-adaptive parameters: adjust bd/wb/gen per sentence
            accumulated_source = " ".join(self._source_words)
            if self.config.complexity_adaptive and len(self._source_words) >= 3:
                effective_bd, effective_wb, effective_gen_cap = (
                    adaptive_params_from_complexity(
                        accumulated_source,
                        base_bd=self.config.border_distance,
                        base_wb=self.config.word_batch,
                        base_gen_cap=self.config.max_new_per_step,
                    )
                )

            # Dynamic word batching (applied on top of complexity)
            if self.config.dynamic_word_batch:
                effective_wb = compute_dynamic_word_batch(
                    effective_wb, len(self._source_words)
                )

            # Word batching: skip until we have enough words
            if (self._batch_counter < effective_wb
                    and not is_final):
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

            # Tokenize current source
            cur_source_tokens = ll.tokenize(
                self._vocab, accumulated_source, add_bos=False, special=False
            )

            prefix_len = len(self._prefix_tokens)

            # Find shared prefix between old and new source tokens
            shared = 0
            for i in range(min(len(self._prev_source_tokens), len(cur_source_tokens))):
                if self._prev_source_tokens[i] == cur_source_tokens[i]:
                    shared += 1
                else:
                    break

            diverge_pos = prefix_len + shared

            # Remove everything from diverge_pos onwards in KV cache
            if not ll.memory_seq_rm(self._mem, 0, diverge_pos, -1):
                # Hybrid model fallback: full clear + re-decode
                ll.memory_clear(self._mem)
                redecode = self._prefix_tokens + cur_source_tokens[:shared]
                if redecode:
                    ll.decode_batch_at(self._ctx, redecode, pos_start=0)
                diverge_pos = len(redecode)

            # Decode delta: new source tail + suffix + committed tokens
            new_source_tail = cur_source_tokens[shared:]
            delta_tokens = new_source_tail + self._suffix_tokens + self._committed_ids
            total_new = len(delta_tokens)

            src_start = prefix_len
            src_end = prefix_len + len(cur_source_tokens)
            num_src_tokens = src_end - src_start

            # Safety: context overflow or no source
            if diverge_pos + total_new >= self.config.n_ctx - 10 or num_src_tokens <= 0:
                self._prev_source_tokens = cur_source_tokens
                return TranslationStep(
                    text="",
                    is_final=is_final,
                    source_words_seen=len(self._source_words),
                    generation_time_ms=(time.time() - t0) * 1000,
                )

            if total_new > 0:
                ll.decode_batch_at(self._ctx, delta_tokens, pos_start=diverge_pos)

            pos = diverge_pos + total_new

            # Cross-step signals: compute BEFORE generation loop
            # These measure how the model's state changed after adding the new source word
            use_entropy_change = self.config.entropy_change_threshold is not None
            use_pred_stability = self.config.prediction_stability
            if use_entropy_change or use_pred_stability:
                first_logits = ll.get_logits_array(self._ctx, -1, self._nv)
                if first_logits is not None:
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

            # Generate tokens
            new_ids = []
            stopped_at_border = False
            is_final_step = is_final
            max_gen = effective_gen_cap if not is_final_step else 256
            consecutive_border_hits = 0  # For border confirmation
            self._gen_positions_history = []  # Reset per translate() call

            for step in range(max_gen):
                # logit_idx=0: after decode_single, batch index is always 0
                next_tok = ll.argmax_logits(self._ctx, -1, self._nv)
                if next_tok in self._stop_ids or next_tok < 0:
                    break

                # Entropy veto
                if self.config.entropy_veto_threshold is not None:
                    logits = ll.get_logits_array(self._ctx, -1, self._nv)
                    if logits is not None:
                        ent = compute_entropy(logits)
                        if ent > self.config.entropy_veto_threshold:
                            stopped_at_border = True
                            break

                ll.decode_single_at(self._ctx, next_tok, pos, seq_id=0)
                pos += 1

                # Check border (not on final -- generate freely)
                if not is_final_step:
                    ctx_size = ll.n_ctx(self._ctx)
                    attn = ll.get_attn_weights(
                        self._ctx, 0, self._n_heads, ctx_size
                    )
                    if attn is not None and src_end <= attn.shape[1]:
                        src_attn = attn[:, src_start:src_end]

                        # Use combined check if any multi-signal feature enabled
                        use_combined = (
                            self.config.shift_k_threshold is not None
                            or self.config.info_gain_threshold is not None
                            or self.config.entropy_change_threshold is not None
                            or self.config.prediction_stability
                            or self.config.coverage_threshold is not None
                            or self.config.attention_monotonicity
                        )
                        if use_combined:
                            # Track attended position for monotonicity
                            if self.config.attention_monotonicity:
                                pos_val = float(aggregate(
                                    src_attn, self._ts_scores,
                                    method=self.config.aggregation,
                                ))
                                self._gen_positions_history.append(pos_val)

                            border_hit, _, _ = check_border_combined(
                                src_attn, self._ts_scores,
                                num_src_tokens, effective_bd,
                                aggregation=self.config.aggregation,
                                adaptive_aggregation=self.config.adaptive_aggregation,
                                head_temp_normalize=self.config.head_temp_normalize,
                                head_temp_reference=self.config.head_temp_reference,
                                shift_k_threshold=self.config.shift_k_threshold,
                                prev_attn=self._prev_step_attn,
                                info_gain_threshold=self.config.info_gain_threshold,
                                dynamic_border=self.config.dynamic_border,
                                entropy_change=self._current_entropy_change,
                                entropy_change_threshold=self.config.entropy_change_threshold,
                                pred_stability_write=self._current_pred_stability_write,
                                coverage_threshold=self.config.coverage_threshold,
                                positions_history=self._gen_positions_history if self.config.attention_monotonicity else None,
                                monotonicity_enabled=self.config.attention_monotonicity,
                            )
                            self._prev_step_attn = src_attn.copy()
                        elif self.config.dynamic_border:
                            border_hit = check_border_dynamic(
                                src_attn, self._ts_scores,
                                num_src_tokens, effective_bd,
                                aggregation=self.config.aggregation,
                                adaptive_aggregation=self.config.adaptive_aggregation,
                                head_temp_normalize=self.config.head_temp_normalize,
                                head_temp_reference=self.config.head_temp_reference,
                            )
                        else:
                            border_hit = check_border(
                                src_attn, self._ts_scores,
                                num_src_tokens, effective_bd,
                                aggregation=self.config.aggregation,
                                adaptive_aggregation=self.config.adaptive_aggregation,
                                head_temp_normalize=self.config.head_temp_normalize,
                                head_temp_reference=self.config.head_temp_reference,
                            )
                        if border_hit:
                            consecutive_border_hits += 1
                            if consecutive_border_hits >= self.config.border_confirm:
                                # LSG confirmation: if enabled, probe whether
                                # removing last K source tokens changes the
                                # output distribution. High KL = source still
                                # matters -> override border and keep generating.
                                if self.config.lsg_kl_threshold is not None:
                                    lsg_kl = self._lsg_probe(
                                        next_tok, pos, src_end
                                    )
                                    if (lsg_kl is not None
                                            and lsg_kl > self.config.lsg_kl_threshold):
                                        # Source tokens still influencing output;
                                        # override border, keep generating
                                        consecutive_border_hits = 0
                                        new_ids.append(next_tok)
                                        continue
                                stopped_at_border = True
                                break
                        else:
                            consecutive_border_hits = 0

                new_ids.append(next_tok)

            # Clean up exploration tokens from KV cache
            if stopped_at_border:
                cached_pos = diverge_pos + total_new + len(new_ids)
                ll.memory_seq_rm(self._mem, 0, cached_pos, -1)

            self._prev_source_tokens = cur_source_tokens

            # Decode text: full sequence to avoid byte-fallback mojibake
            prev_text = (
                ll.tokens_to_text(self._vocab, self._committed_ids, errors="ignore")
                if self._committed_ids else ""
            )
            if new_ids:
                self._committed_ids.extend(new_ids)
            full_text = (
                ll.tokens_to_text(self._vocab, self._committed_ids, errors="ignore")
                if self._committed_ids else ""
            )
            new_text = full_text[len(prev_text):]

            elapsed_ms = (time.time() - t0) * 1000

            # Handle segment reset if final
            if is_final:
                self._handle_segment_end()

            return TranslationStep(
                text=new_text,
                is_final=is_final,
                committed_tokens=len(new_ids),
                stopped_at_border=stopped_at_border,
                source_words_seen=len(self._source_words),
                generation_time_ms=elapsed_ms,
            )

    def _lsg_probe(self, last_token: int, pos: int,
                   src_end: int) -> Optional[float]:
        """Perform LSG logit KL probe via KV cache forking.

        Compares output logits with full source vs reduced source (last K
        source tokens removed). Returns KL divergence or None on failure.

        Method (LSG, arxiv 2501.00868):
            1. Get full-source logits (already available from main decode)
            2. Fork KV cache to seq_id=1
            3. Remove last lsg_k source token positions from the fork
            4. Remove the last decoded position from the fork
            5. Re-decode the same token on the fork (without full source)
            6. Compare logit distributions via KL divergence
            7. Clean up the fork

        The KL measures "did the last K source tokens change the prediction?"
        Low KL = source exhausted (safe to WRITE), High KL = source matters (READ).
        """
        lsg_k = min(self.config.lsg_k, src_end - len(self._prefix_tokens))
        if lsg_k <= 0:
            return None

        # 1. Get full-source logits (from the main decode chain)
        logits_full = ll.get_logits_array(self._ctx, -1, self._nv)
        if logits_full is None:
            return None

        # 2. Fork KV cache: copy seq 0 -> seq 1 for positions [0, pos)
        ll.memory_seq_cp(self._mem, 0, 1, 0, pos)

        try:
            # 3. Remove last K source tokens from the fork
            #    Source tokens occupy [prefix_len, src_end)
            #    Remove [src_end - lsg_k, src_end) from seq 1
            rm_start = src_end - lsg_k
            ll.memory_seq_rm(self._mem, 1, rm_start, src_end)

            # 4. Remove the entry at pos-1 (last decoded token) from seq 1
            #    so we can re-decode it to get fresh logits
            ll.memory_seq_rm(self._mem, 1, pos - 1, pos)

            # 5. Re-decode the same token on seq 1
            ret = ll.decode_single_at(
                self._ctx, last_token, pos - 1, seq_id=1, output=True
            )
            if ret != 0:
                return None

            # 6. Get reduced-source logits and compute KL
            logits_reduced = ll.get_logits_array(self._ctx, 0, self._nv)
            if logits_reduced is None:
                return None

            return compute_logit_kl(logits_full, logits_reduced)
        finally:
            # 7. Clean up: remove all seq 1 entries
            ll.memory_seq_rm(self._mem, 1, 0, -1)

    def _init_segment(self):
        """Initialize a new segment: create context, decode prefix."""
        ctx_arg = None
        if self._prev_contexts and self.config.context_sentences > 0:
            ctx_arg = self._prev_contexts[-self.config.context_sentences:]

        # Build prefix
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

        # Decode prefix once
        ll.decode_batch_at(self._ctx, self._prefix_tokens, pos_start=0)
        self._prev_source_tokens = []
        self._prev_step_attn = None
        # Reset cross-step tracking for new segment
        self._prev_first_token_entropy = None
        self._prev_first_token_logits = None
        self._current_entropy_change = None
        self._current_pred_stability_write = None

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
        self._source_words = []
        self._batch_counter = 0
        self._gen_positions_history = []

        if self._ctx is not None:
            ll.free_context(self._ctx)
            self._ctx = None
            self._mem = None
        self._prev_source_tokens = []

    def reset(self):
        """Reset state for next segment (called externally)."""
        with self._lock:
            self._handle_segment_end()

    def get_full_translation(self) -> str:
        """Get full committed translation text."""
        if not self._committed_ids:
            return ""
        return ll.tokens_to_text(self._vocab, self._committed_ids, errors="ignore")

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


# ---------------------------------------------------------------------------
# Baselines (registered here for convenience)
# ---------------------------------------------------------------------------

@register_backend("full-sentence")
class FullSentenceBackend(SimulMTBackend):
    """Full-sentence baseline: wait for all source, then translate.

    Quality upper bound for AlignAtt. No simultaneous aspect.
    """

    def __init__(self, config: BackendConfig):
        super().__init__(config)
        self._source_words: List[str] = []
        self._inner: Optional[AlignAttBackend] = None

    def translate(self, source_word: str, is_final: bool = False,
                  emission_time: float = 0.0) -> TranslationStep:
        self._source_words.append(source_word)
        if not is_final:
            return TranslationStep(text="", source_words_seen=len(self._source_words))

        # Translate all at once
        if self._inner is None:
            self._inner = AlignAttBackend(self.config)

        full_source = " ".join(self._source_words)
        result = self._inner.translate(full_source, is_final=True, emission_time=emission_time)
        self._source_words = []
        return result

    def reset(self):
        self._source_words = []
        if self._inner:
            self._inner.reset()

    def close(self):
        if self._inner:
            self._inner.close()


@register_backend("eager")
class EagerBackend(SimulMTBackend):
    """Eager baseline: translate after every single word.

    Latency lower bound. Uses AlignAtt internally but with border_distance=0
    so it always generates all it can.
    """

    def __init__(self, config: BackendConfig):
        eager_config = BackendConfig(**{
            **config.__dict__,
            "border_distance": 0,
            "word_batch": 1,
        })
        super().__init__(eager_config)
        self._inner = AlignAttBackend(eager_config)

    def translate(self, source_word: str, is_final: bool = False,
                  emission_time: float = 0.0) -> TranslationStep:
        return self._inner.translate(source_word, is_final, emission_time)

    def reset(self):
        self._inner.reset()

    def close(self):
        self._inner.close()
