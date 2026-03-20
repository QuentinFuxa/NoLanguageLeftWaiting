#!/usr/bin/env python3
"""
Detect token alignment heads in LLMs for AlignAtt border detection.

Algorithm:
1. Load FLORES+ dataset (100-200 source-target pairs)
2. Get word-level alignments using SimAlign (mBERT-based)
3. For each sentence: generate translation, extract attention from all heads
4. For each aligned word pair: check if head's attention argmax matches alignment
5. Compute Token Similarity (TS) score = matches / total_alignable_tokens
6. Rank heads by TS, output top-K with scores

Usage:
    python -m nllw.heads.detect \
        --model /path/to/model.gguf \
        --prompt-format hymt \
        --lang en-zh \
        -n 100 \
        -o translation_heads.json
"""

import argparse
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from nllw.llama_backend import LlamaContext, LlamaModel, init as llama_init


TS_THRESHOLD = 0.1


@dataclass
class HeadDetectionConfig:
    """Configuration for alignment head detection."""
    num_sentences: int = 100
    n_ctx: int = 2048
    max_gen: int = 256
    ts_threshold: float = 0.1
    batch_heads: int = 0  # 0 = all at once
    skip_gdn_layers: bool = False  # For Qwen3.5 hybrid models


# Prompt templates for head detection (simplified, {source} placeholder)
HEAD_DETECT_PROMPTS = {
    "hymt": {
        "template": "将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：\n\n{source}<|extra_0|>",
        "stop_strings": ["<|extra_0|>", "<|endoftext|>"],
    },
    "hymt-de": {
        "template": (
            "Translate the following text into German, please only output "
            "the translated result without additional explanation:\n\n{source}<|extra_0|>"
        ),
        "stop_strings": ["<|extra_0|>", "<|endoftext|>"],
    },
    "hymt-it": {
        "template": (
            "Translate the following text into Italian, please only output "
            "the translated result without additional explanation:\n\n{source}<|extra_0|>"
        ),
        "stop_strings": ["<|extra_0|>", "<|endoftext|>"],
    },
    "hymt-cs-en": {
        "template": (
            "Translate the following text into English, please only output "
            "the translated result without additional explanation:\n\n{source}<|extra_0|>"
        ),
        "stop_strings": ["<|extra_0|>", "<|endoftext|>"],
    },
    "qwen3": {
        "template": (
            "<|im_start|>user\n"
            "You are a professional English to Chinese translator. "
            "Produce only the Chinese translation.\n\n"
            "{source}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ),
        "stop_strings": ["<|im_end|>", "<|endoftext|>"],
    },
    "qwen3.5": {
        "template": (
            "<|im_start|>user\n"
            "You are a professional English to Chinese translator. "
            "Produce only the Chinese translation.\n\n"
            "{source}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ),
        "stop_strings": ["<|im_end|>", "<|endoftext|>"],
    },
    "eurollm": {
        "template": (
            "<|im_start|>system\nTranslate English to Chinese.<|im_end|>\n"
            "<|im_start|>user\n{source}<|im_end|>\n<|im_start|>assistant\n"
        ),
        "stop_strings": ["<|im_end|>", "<|endoftext|>"],
    },
}

LANG_CONFIGS = {
    "en-zh": {"src": ("eng", "Latn"), "tgt": ("cmn", "Hans"), "is_cjk": True},
    "en-de": {"src": ("eng", "Latn"), "tgt": ("deu", "Latn"), "is_cjk": False},
    "en-it": {"src": ("eng", "Latn"), "tgt": ("ita", "Latn"), "is_cjk": False},
    "cs-en": {"src": ("ces", "Latn"), "tgt": ("eng", "Latn"), "is_cjk": False},
    "en-fr": {"src": ("eng", "Latn"), "tgt": ("fra", "Latn"), "is_cjk": False},
    "en-ja": {"src": ("eng", "Latn"), "tgt": ("jpn", "Jpan"), "is_cjk": True},
}


def _tokens_to_word_map(token_strings: List[str]) -> Dict[int, List[int]]:
    """Map subword tokens to word indices."""
    word2tokens: Dict[int, List[int]] = defaultdict(list)
    word_idx = -1
    for i, tok in enumerate(token_strings):
        if tok.startswith(("Ġ", " ", "▁")) or word_idx == -1:
            word_idx += 1
        word2tokens[word_idx].append(i)
    return dict(word2tokens)


def _reconstruct_words(token_strings: List[str]) -> List[str]:
    """Reconstruct word list from subword tokens."""
    words = []
    current = ""
    for tok in token_strings:
        if tok.startswith(("Ġ", " ", "▁")):
            if current:
                words.append(current)
            current = tok.lstrip("Ġ ▁")
        else:
            current += tok
    if current:
        words.append(current)
    return words


def _chinese_char_to_token_map(token_strings: List[str]) -> Dict[int, List[int]]:
    """Map individual CJK characters back to token positions."""
    char2tokens: Dict[int, List[int]] = {}
    char_idx = 0
    for tok_idx, tok in enumerate(token_strings):
        clean = tok.lstrip("Ġ ▁")
        for _ in clean:
            char2tokens.setdefault(char_idx, []).append(tok_idx)
            char_idx += 1
    return char2tokens


def _reconstruct_cjk_chars(token_strings: List[str]) -> List[str]:
    """Reconstruct character-level list for CJK output."""
    text = ""
    for tok in token_strings:
        text += tok.lstrip("Ġ ▁")
    return [c for c in text if c.strip()]


def detect_alignment_heads(
    model: LlamaModel,
    prompt_format: str,
    lang: str = "en-zh",
    config: Optional[HeadDetectionConfig] = None,
    output_path: Optional[str] = None,
) -> List[dict]:
    """Detect token alignment heads in a model.

    Args:
        model: Loaded LlamaModel.
        prompt_format: Name of prompt template to use.
        lang: Language pair (e.g., "en-zh").
        config: Detection configuration.
        output_path: Path to save results JSON (optional).

    Returns:
        List of {"layer": L, "head": H, "ts": S} sorted by TS score.
    """
    if config is None:
        config = HeadDetectionConfig()

    fmt = HEAD_DETECT_PROMPTS[prompt_format]
    lang_cfg = LANG_CONFIGS[lang]
    is_cjk = lang_cfg["is_cjk"]

    # Load FLORES
    from datasets import load_dataset
    from simalign import SentenceAligner

    print(f"Loading FLORES dev data ({lang.upper()})...")
    ds = load_dataset("openlanguagedata/flores_plus", split="dev")
    src_ds = ds.filter(
        lambda x: x["iso_639_3"] == lang_cfg["src"][0] and x["iso_15924"] == lang_cfg["src"][1]
    )
    tgt_ds = ds.filter(
        lambda x: x["iso_639_3"] == lang_cfg["tgt"][0] and x["iso_15924"] == lang_cfg["tgt"][1]
    )
    src_map = {row["id"]: row["text"] for row in src_ds}
    tgt_map = {row["id"]: row["text"] for row in tgt_ds}
    common_ids = sorted(set(src_map) & set(tgt_map))
    num_sentences = min(config.num_sentences, len(common_ids))
    pair_ids = common_ids[:num_sentences]
    print(f"  {num_sentences} sentence pairs")

    # Load word aligner
    print("Loading SimAlign (mBERT)...")
    aligner = SentenceAligner(model="bert", token_type="bpe")

    # Model info
    num_layers = model.n_layer
    num_heads_per_layer = model.n_head
    eos_id = model.eos_id

    # Build stop IDs
    stop_ids = {eos_id}
    for tok_str in fmt["stop_strings"]:
        tok_ids = model.tokenize(tok_str, add_bos=False, special=True)
        if len(tok_ids) == 1:
            stop_ids.add(tok_ids[0])
    for sid in [2, 151643, 151645]:
        stop_ids.add(sid)

    # Set up head pairs to probe
    all_layers, all_head_ids = [], []
    for l in range(num_layers):
        if config.skip_gdn_layers and (l % 4 != 3):
            continue
        for h in range(num_heads_per_layer):
            all_layers.append(l)
            all_head_ids.append(h)

    total_heads = len(all_layers)
    print(f"  Probing {total_heads} heads ({num_layers} layers x {num_heads_per_layer} heads)")

    batch_size = config.batch_heads if config.batch_heads > 0 else total_heads
    n_batches = (total_heads + batch_size - 1) // batch_size

    # Score accumulation
    g = np.zeros(total_heads, dtype=np.int64)
    m = 0

    t0 = time.time()
    for idx, sid in enumerate(pair_ids):
        source_text = src_map[sid]
        template = fmt["template"]
        marker_pos = template.find("{source}")
        prefix = template[:marker_pos]
        suffix = template[marker_pos + len("{source}"):]

        prompt = prefix + source_text + suffix
        prompt_tokens = model.tokenize(prompt, add_bos=True, special=True)
        prompt_len = len(prompt_tokens)

        # Find source range
        prefix_tokens = model.tokenize(prefix, add_bos=True, special=True)
        full_tokens = model.tokenize(prefix + source_text + suffix, add_bos=True, special=True)
        suffix_tokens = model.tokenize(suffix, add_bos=False, special=True)
        src_start = len(prefix_tokens)
        src_end = len(full_tokens) - len(suffix_tokens)
        if src_end <= src_start:
            continue

        source_positions = list(range(src_start, src_end))
        source_token_strings = [
            model.token_to_piece(prompt_tokens[i]) for i in range(src_start, src_end)
        ]

        _word_aligns = None
        _src_word2tok = None
        _tgt_char2tok = None

        for batch_idx in range(n_batches):
            b_start = batch_idx * batch_size
            b_end = min(b_start + batch_size, total_heads)
            b_layers = all_layers[b_start:b_end]
            b_heads = all_head_ids[b_start:b_end]
            n_pairs_batch = b_end - b_start

            ctx = LlamaContext(model, n_ctx=config.n_ctx, attn_weights=True)
            ctx.set_attn_heads(b_layers, b_heads)
            ctx.decode_batch(prompt_tokens, pos_start=0)
            pos = prompt_len

            generated_ids = []
            step_argmaxes = []

            for step in range(config.max_gen):
                next_tok = ctx.argmax_logits(-1)
                if next_tok in stop_ids or next_tok < 0:
                    break

                generated_ids.append(next_tok)
                ctx.decode_single(next_tok, pos)
                pos += 1

                attn = ctx.get_attn_weights(0, n_pairs_batch)
                if attn is not None and src_end <= attn.shape[1]:
                    src_attn = attn[:, src_start:src_end]
                    step_argmaxes.append(np.argmax(src_attn, axis=1))
                else:
                    step_argmaxes.append(None)

            ctx.close()

            if not generated_ids:
                continue

            num_gen = len(generated_ids)
            output_tokens = [model.token_to_piece(t) for t in generated_ids]

            # Word alignments (first batch only)
            if batch_idx == 0:
                src_words = _reconstruct_words(source_token_strings)
                tgt_units = _reconstruct_cjk_chars(output_tokens) if is_cjk else _reconstruct_words(output_tokens)
                if not src_words or not tgt_units:
                    break

                try:
                    alignments = aligner.get_word_aligns(src_words, tgt_units)
                except Exception:
                    break

                _word_aligns = alignments.get("itermax", alignments.get("inter", []))
                if not _word_aligns:
                    break

                _src_word2tok = _tokens_to_word_map(source_token_strings)
                _tgt_char2tok = (
                    _chinese_char_to_token_map(output_tokens) if is_cjk
                    else _tokens_to_word_map(output_tokens)
                )

            # Score heads
            for src_widx, tgt_cidx in _word_aligns:
                if src_widx not in _src_word2tok or tgt_cidx not in _tgt_char2tok:
                    continue
                src_abs = {
                    source_positions[ti]
                    for ti in _src_word2tok[src_widx]
                    if ti < len(source_positions)
                }
                if not src_abs:
                    continue

                for tgt_step in _tgt_char2tok[tgt_cidx]:
                    if tgt_step >= num_gen or tgt_step >= len(step_argmaxes):
                        continue
                    if step_argmaxes[tgt_step] is None:
                        continue

                    if batch_idx == 0:
                        m += 1

                    for h_offset in range(n_pairs_batch):
                        abs_pos = src_start + int(step_argmaxes[tgt_step][h_offset])
                        if abs_pos in src_abs:
                            g[b_start + h_offset] += 1

        if (idx + 1) % 10 == 0 or idx == 0:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (num_sentences - idx - 1)
            print(f"  [{idx+1}/{num_sentences}] m={m} | {elapsed/(idx+1):.1f}s/sent | ETA: {eta:.0f}s")

    # Compute TS scores
    ts = g / max(m, 1)

    # Build results
    tah = []
    flat_idx = 0
    for l, h in zip(all_layers, all_head_ids):
        score = float(ts[flat_idx])
        if score > config.ts_threshold:
            tah.append({"layer": l, "head": h, "ts": round(score, 4)})
        flat_idx += 1

    tah.sort(key=lambda x: x["ts"], reverse=True)

    print(f"\n{len(tah)} alignment heads found (TS > {config.ts_threshold})")
    for entry in tah[:10]:
        print(f"  L{entry['layer']:2d} H{entry['head']:2d} : TS={entry['ts']:.4f}")

    # Save if requested
    if output_path:
        # Build TS matrix for visualization
        ts_matrix = np.zeros((num_layers, num_heads_per_layer))
        flat_idx = 0
        for l, h in zip(all_layers, all_head_ids):
            ts_matrix[l, h] = ts[flat_idx]
            flat_idx += 1

        output = {
            "model": os.path.basename(str(getattr(model, '_model_path', 'unknown'))),
            "prompt_format": prompt_format,
            "language_pair": lang,
            "num_layers": num_layers,
            "num_heads": num_heads_per_layer,
            "num_sentences": num_sentences,
            "total_alignable_tokens": int(m),
            "ts_threshold": config.ts_threshold,
            "ts_matrix": ts_matrix.tolist(),
            "token_alignment_heads": tah,
        }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved to {output_path}")

    return tah


def main():
    parser = argparse.ArgumentParser(description="Detect alignment heads via llama.cpp")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-format", required=True, choices=list(HEAD_DETECT_PROMPTS.keys()))
    parser.add_argument("--lang", default="en-zh", choices=list(LANG_CONFIGS.keys()))
    parser.add_argument("-n", type=int, default=100)
    parser.add_argument("-o", "--output", default="translation_heads.json")
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--max-gen", type=int, default=256)
    parser.add_argument("--batch-heads", type=int, default=0)
    parser.add_argument("--skip-gdn-layers", action="store_true")
    args = parser.parse_args()

    llama_init()
    model = LlamaModel(args.model, n_gpu_layers=99)

    config = HeadDetectionConfig(
        num_sentences=args.n,
        n_ctx=args.n_ctx,
        max_gen=args.max_gen,
        batch_heads=args.batch_heads,
        skip_gdn_layers=args.skip_gdn_layers,
    )

    detect_alignment_heads(
        model=model,
        prompt_format=args.prompt_format,
        lang=args.lang,
        config=config,
        output_path=args.output,
    )

    model.close()


if __name__ == "__main__":
    main()
