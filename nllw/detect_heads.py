"""Automatic alignment head detection for any GGUF model.

Detects which (layer, head) pairs in a decoder-only LLM attend to the
correct source tokens during translation. These "token alignment heads"
(TAHs) are the foundation of AlignAtt border detection.

Algorithm:
    1. Load FLORES+ parallel sentences for a given language pair
    2. For each sentence, translate with full prompt and extract attention
    3. Use SimAlign (mBERT) to get ground-truth word alignments
    4. Score each head: TS = (correct argmax predictions) / (total alignable tokens)
    5. Heads with TS > threshold (default 0.1) are token alignment heads

Usage:
    python -m nllw.detect_heads --model /path/to/model.gguf --lang en-zh -n 100
    python -m nllw.detect_heads --model /path/to/model.gguf --lang en-de --prompt-format hymt
    python -m nllw.detect_heads --model /path/to/model.gguf --lang en-zh --skip-gdn-layers

Output:
    JSON file with TS matrix and ranked token alignment heads, plus optional heatmap.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import llama_backend as ll
from .prompts import get_prompt_format, detect_model_family, PromptFormat
from .eval import load_flores

# Minimum TS score to qualify as a token alignment head
DEFAULT_TS_THRESHOLD = 0.1


# ---------------------------------------------------------------------------
# Token-to-word mapping utilities
# ---------------------------------------------------------------------------

def _tokens_to_word_map(token_strings: List[str]) -> Dict[int, List[int]]:
    """Map subword tokens to word indices.

    Words are split on leading space/Ġ/▁ markers (standard BPE conventions).
    Returns: {word_idx: [token_idx, ...]}
    """
    word2tokens: Dict[int, List[int]] = defaultdict(list)
    word_idx = -1
    for i, tok in enumerate(token_strings):
        if tok.startswith("Ġ") or tok.startswith(" ") or tok.startswith("▁") or word_idx == -1:
            word_idx += 1
        word2tokens[word_idx].append(i)
    return dict(word2tokens)


def _reconstruct_words(token_strings: List[str]) -> List[str]:
    """Reconstruct word list from BPE token strings."""
    words = []
    current = ""
    for tok in token_strings:
        if tok.startswith("Ġ") or tok.startswith(" ") or tok.startswith("▁"):
            if current:
                words.append(current)
            current = tok.lstrip("Ġ ▁")
        else:
            current += tok
    if current:
        words.append(current)
    return words


def _reconstruct_cjk_chars(token_strings: List[str]) -> List[str]:
    """Reconstruct character-level list for CJK output."""
    text = ""
    for tok in token_strings:
        clean = tok.lstrip("Ġ ▁")
        text += clean
    return [c for c in text if c.strip()]


def _cjk_char_to_token_map(token_strings: List[str]) -> Dict[int, List[int]]:
    """Map individual CJK characters back to token positions."""
    char2tokens: Dict[int, List[int]] = {}
    char_idx = 0
    for tok_idx, tok in enumerate(token_strings):
        clean = tok.lstrip("Ġ ▁")
        for _ in clean:
            char2tokens.setdefault(char_idx, []).append(tok_idx)
            char_idx += 1
    return char2tokens


# ---------------------------------------------------------------------------
# Source range detection
# ---------------------------------------------------------------------------

def _find_source_range(
    vocab, prompt_fmt: PromptFormat, source_text: str
) -> Tuple[int, int]:
    """Find the token range [start, end) of the source text within the prompt.

    Tokenizes the prefix, full prompt, and suffix separately to identify
    where source tokens start and end.
    """
    full_prompt = prompt_fmt.build_prompt(source_text)
    prefix = prompt_fmt.prefix
    suffix = prompt_fmt.suffix

    prefix_tokens = ll.tokenize(vocab, prefix, add_bos=True, special=True)
    full_tokens = ll.tokenize(vocab, full_prompt, add_bos=True, special=True)
    suffix_tokens = ll.tokenize(vocab, suffix, add_bos=False, special=True)

    src_start = len(prefix_tokens)
    src_end = len(full_tokens) - len(suffix_tokens)
    return (src_start, src_end) if src_end > src_start else (0, 0)


# ---------------------------------------------------------------------------
# Main detection algorithm
# ---------------------------------------------------------------------------

def detect_heads(
    model_path: str,
    direction: str = "en-zh",
    prompt_format: Optional[str] = None,
    n_sentences: int = 100,
    n_ctx: int = 2048,
    max_gen: int = 256,
    batch_heads: int = 0,
    skip_gdn_layers: bool = False,
    ts_threshold: float = DEFAULT_TS_THRESHOLD,
    verbose: bool = True,
) -> dict:
    """Detect token alignment heads for a GGUF model.

    Args:
        model_path: Path to GGUF model file
        direction: Language pair (e.g. "en-zh", "en-de")
        prompt_format: Model family override (auto-detected if None)
        n_sentences: Number of FLORES sentences to use
        n_ctx: Context window size
        max_gen: Max tokens to generate per sentence
        batch_heads: Process heads in batches (0 = all at once)
        skip_gdn_layers: Skip GDN/linear-attention layers (for Qwen3.5 hybrid)
        ts_threshold: Minimum TS to qualify as alignment head
        verbose: Print progress

    Returns:
        Dict with keys: model, direction, num_layers, num_heads,
        ts_matrix (2D list), token_alignment_heads (sorted by TS desc)
    """
    parts = direction.split("-")
    src_lang, tgt_lang = parts[0], parts[1]
    is_cjk = tgt_lang in ("zh", "ja", "ko")

    # Determine prompt format
    model_family = prompt_format or detect_model_family(model_path)
    fmt = get_prompt_format(model_family, direction)

    # Load FLORES parallel data
    if verbose:
        print(f"Loading FLORES+ {direction} ({n_sentences} sentences)...", file=sys.stderr)
    corpus = load_flores(src_lang, tgt_lang, n=n_sentences)
    n_sentences = len(corpus)
    if verbose:
        print(f"  Loaded {n_sentences} pairs.", file=sys.stderr)

    # Load word aligner
    if verbose:
        print("Loading word aligner (SimAlign / mBERT)...", file=sys.stderr)
    from simalign import SentenceAligner
    aligner = SentenceAligner(model="bert", token_type="bpe")

    # Initialize llama.cpp
    if verbose:
        print(f"Loading model: {model_path}", file=sys.stderr)
    ll.init()
    model = ll.load_model(model_path, n_gpu_layers=99)
    vocab = ll.get_vocab(model)
    nv = ll.n_vocab(vocab)
    eos_id = ll.vocab_eos(vocab)
    num_layers = ll.n_layer(model)
    num_heads = ll.n_head(model)

    if verbose:
        print(f"  {num_layers} layers x {num_heads} heads = {num_layers * num_heads} total",
              file=sys.stderr)

    # Build stop IDs
    stop_ids = {eos_id}
    for tok_str in fmt.extra_stop_strings:
        tok_ids = ll.tokenize(vocab, tok_str, add_bos=False, special=True)
        if len(tok_ids) == 1:
            stop_ids.add(tok_ids[0])
    # Suffix tokens as stop
    suffix_toks = ll.tokenize(vocab, fmt.suffix, add_bos=False, special=True)
    for tid in suffix_toks:
        stop_ids.add(tid)
    # Common EOS across models
    for sid in [2, 151643, 151645, 107]:
        stop_ids.add(sid)

    # Set up head pairs (handle hybrid models)
    all_layers = []
    all_head_ids = []
    for l in range(num_layers):
        if skip_gdn_layers and (l % 4 != 3):
            continue
        for h in range(num_heads):
            all_layers.append(l)
            all_head_ids.append(h)
    total_heads = len(all_layers)

    batch_size = batch_heads if batch_heads > 0 else total_heads
    n_batches = (total_heads + batch_size - 1) // batch_size

    if verbose:
        mode = f"{n_batches} batches of {batch_size}" if n_batches > 1 else f"all {total_heads} at once"
        print(f"  Processing heads: {mode}", file=sys.stderr)

    # Scoring accumulators
    g = np.zeros(total_heads, dtype=np.int64)  # correct predictions per head
    m = 0  # total alignable tokens

    t0 = time.time()

    for idx, item in enumerate(corpus):
        source_text = item["source"]

        # Tokenize prompt
        full_prompt = fmt.build_prompt(source_text)
        prompt_tokens = ll.tokenize(vocab, full_prompt, add_bos=True, special=True)
        prompt_len = len(prompt_tokens)

        # Find source token range
        src_start, src_end = _find_source_range(vocab, fmt, source_text)
        if src_end <= src_start:
            continue

        source_positions = list(range(src_start, src_end))
        source_token_strings = [
            ll.token_to_piece(vocab, prompt_tokens[i])
            for i in range(src_start, src_end)
        ]

        # Cache word alignment info across head batches
        _word_aligns = None
        _src_word2tok = None
        _tgt_char2tok = None

        for batch_idx in range(n_batches):
            b_start = batch_idx * batch_size
            b_end = min(b_start + batch_size, total_heads)
            b_layers = all_layers[b_start:b_end]
            b_heads = all_head_ids[b_start:b_end]
            n_pairs_batch = b_end - b_start

            # Create context with attention extraction
            ctx = ll.create_context(model, n_ctx=n_ctx, attn_weights=True)
            ll.set_attn_heads(ctx, b_layers, b_heads)

            # Decode prompt
            ll.decode_batch(ctx, prompt_tokens)
            pos = prompt_len

            # Generate tokens and collect per-step attention argmaxes
            generated_ids = []
            step_argmaxes = []

            for step in range(max_gen):
                next_tok = ll.argmax_logits(ctx, -1, nv)
                if next_tok in stop_ids or next_tok < 0:
                    break

                generated_ids.append(next_tok)

                ll.decode_single_at(ctx, next_tok, pos, seq_id=0)
                pos += 1

                # Extract attention weights
                ctx_size = ll.n_ctx(ctx)
                attn = ll.get_attn_weights(ctx, 0, n_pairs_batch, ctx_size)
                if attn is not None and src_end <= attn.shape[1]:
                    src_attn = attn[:, src_start:src_end]
                    argmaxes = np.argmax(src_attn, axis=1)
                    step_argmaxes.append(argmaxes)
                else:
                    step_argmaxes.append(None)

            ll.free_context(ctx)

            if not generated_ids:
                break  # skip all batches for this sentence

            num_gen = len(generated_ids)
            output_tokens = [ll.token_to_piece(vocab, t) for t in generated_ids]

            # Build word alignments (once per sentence, on first batch)
            if batch_idx == 0:
                translation = ll.tokens_to_text(vocab, generated_ids)
                src_words = _reconstruct_words(source_token_strings)
                if is_cjk:
                    tgt_units = _reconstruct_cjk_chars(output_tokens)
                else:
                    tgt_units = _reconstruct_words(output_tokens)

                if not src_words or not tgt_units:
                    break

                try:
                    alignments = aligner.get_word_aligns(src_words, tgt_units)
                except Exception:
                    break

                word_aligns = alignments.get("itermax", alignments.get("inter", []))
                if not word_aligns:
                    break

                src_word2tok = _tokens_to_word_map(source_token_strings)
                if is_cjk:
                    tgt_char2tok = _cjk_char_to_token_map(output_tokens)
                else:
                    tgt_char2tok = _tokens_to_word_map(output_tokens)

                _word_aligns = word_aligns
                _src_word2tok = src_word2tok
                _tgt_char2tok = tgt_char2tok
            else:
                word_aligns = _word_aligns
                src_word2tok = _src_word2tok
                tgt_char2tok = _tgt_char2tok

            # Score heads based on word alignments
            for src_widx, tgt_cidx in word_aligns:
                if src_widx not in src_word2tok or tgt_cidx not in tgt_char2tok:
                    continue

                src_abs_positions = set(
                    source_positions[ti]
                    for ti in src_word2tok[src_widx]
                    if ti < len(source_positions)
                )
                if not src_abs_positions:
                    continue

                for tgt_step in tgt_char2tok[tgt_cidx]:
                    if tgt_step >= num_gen or tgt_step >= len(step_argmaxes):
                        continue
                    if step_argmaxes[tgt_step] is None:
                        continue

                    if batch_idx == 0:
                        m += 1

                    argmaxes = step_argmaxes[tgt_step]
                    for h_offset in range(n_pairs_batch):
                        abs_head_idx = b_start + h_offset
                        abs_pos = src_start + int(argmaxes[h_offset])
                        if abs_pos in src_abs_positions:
                            g[abs_head_idx] += 1

        elapsed = time.time() - t0
        avg = elapsed / (idx + 1)
        eta = avg * (n_sentences - idx - 1)

        if verbose and generated_ids:
            preview = ll.tokens_to_text(vocab, generated_ids[:20])
            print(
                f"  [{idx+1}/{n_sentences}] m={m} | {preview[:60]}... | "
                f"{avg:.1f}s/sent | ETA: {eta:.0f}s",
                file=sys.stderr, flush=True,
            )

    elapsed = time.time() - t0
    if verbose:
        print(f"\nDone in {elapsed:.1f}s. Total alignable tokens: m={m}", file=sys.stderr)

    ll.free_model(model)
    ll.cleanup()

    # Compute Translation Scores
    ts = g / max(m, 1)

    # Build full TS matrix
    ts_matrix = np.zeros((num_layers, num_heads))
    probed_layers = sorted(set(all_layers))
    flat_idx = 0
    for l in probed_layers:
        for h in range(num_heads):
            ts_matrix[l, h] = ts[flat_idx]
            flat_idx += 1

    # Identify heads above threshold
    tah = []
    for l in probed_layers:
        for h in range(num_heads):
            score = ts_matrix[l, h]
            if score > ts_threshold:
                tah.append({"layer": l, "head": h, "ts": round(float(score), 4)})
    tah.sort(key=lambda x: x["ts"], reverse=True)

    if verbose:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"TOKEN ALIGNMENT HEADS (TS > {ts_threshold}): {len(tah)} / {total_heads}",
              file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        for entry in tah[:20]:
            bar = "█" * int(entry["ts"] * 50)
            print(f"  L{entry['layer']:2d} H{entry['head']:2d} : TS={entry['ts']:.4f}  {bar}",
                  file=sys.stderr)
        if len(tah) > 20:
            print(f"  ... and {len(tah) - 20} more", file=sys.stderr)

    result = {
        "model": os.path.basename(model_path),
        "prompt_format": model_family,
        "language_pair": direction,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_sentences": n_sentences,
        "total_alignable_tokens": int(m),
        "ts_threshold": ts_threshold,
        "ts_matrix": ts_matrix.tolist(),
        "token_alignment_heads": tah,
    }
    return result


# ---------------------------------------------------------------------------
# Heatmap visualization
# ---------------------------------------------------------------------------

def save_heatmap(result: dict, output_path: str):
    """Save a TS heatmap as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for heatmaps. Install: pip install matplotlib",
              file=sys.stderr)
        return

    ts_matrix = np.array(result["ts_matrix"])
    num_layers, num_heads = ts_matrix.shape
    tah = result["token_alignment_heads"]

    fig, ax = plt.subplots(
        figsize=(max(10, num_heads * 0.5), max(12, num_layers * 0.4))
    )
    im = ax.imshow(
        ts_matrix, aspect="auto", cmap="RdYlBu_r",
        vmin=0, vmax=max(0.4, ts_matrix.max()),
        interpolation="nearest",
    )
    ax.set_xlabel("Head ID", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)

    model_name = result.get("model", "").replace(".gguf", "")
    lang = result.get("language_pair", "")
    n = result.get("num_sentences", 0)
    ax.set_title(
        f"Translation Scores - {model_name} ({lang.upper()}, n={n})\n"
        f"{len(tah)} token alignment heads (TS > {result.get('ts_threshold', 0.1)})",
        fontsize=13,
    )
    ax.set_xticks(range(num_heads))
    ax.set_yticks(range(num_layers))
    plt.colorbar(im, ax=ax, label="Translation Score", shrink=0.8)

    for entry in tah:
        ax.add_patch(plt.Rectangle(
            (entry["head"] - 0.5, entry["layer"] - 0.5),
            1, 1, fill=False, edgecolor="red", linewidth=1.5,
        ))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Heatmap saved to {output_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Detect token alignment heads for a GGUF model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m nllw.detect_heads --model HY-MT1.5-7B.gguf --lang en-zh -n 100
  python -m nllw.detect_heads --model Qwen3.5-4B.gguf --lang en-de --skip-gdn-layers
  python -m nllw.detect_heads --model EuroLLM-9B.gguf --lang cs-en --prompt-format eurollm
""",
    )
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--lang", default="en-zh", help="Language pair (e.g. en-zh, en-de)")
    parser.add_argument("--prompt-format", default=None,
                        help="Model family override (auto-detected from filename)")
    parser.add_argument("-n", "--num-sentences", type=int, default=100,
                        help="Number of FLORES sentences to process")
    parser.add_argument("-o", "--output", default=None,
                        help="Output JSON file (auto-named if not set)")
    parser.add_argument("--heatmap", default=None,
                        help="Heatmap PNG path (auto-derived from output if not set)")
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--max-gen", type=int, default=256,
                        help="Max tokens to generate per sentence")
    parser.add_argument("--batch-heads", type=int, default=0,
                        help="Process heads in batches (0 = all at once)")
    parser.add_argument("--skip-gdn-layers", action="store_true",
                        help="Skip GDN/linear-attention layers (for Qwen3.5 hybrid)")
    parser.add_argument("--ts-threshold", type=float, default=DEFAULT_TS_THRESHOLD,
                        help=f"Min TS score for alignment heads (default {DEFAULT_TS_THRESHOLD})")
    args = parser.parse_args()

    # Auto-name output
    if args.output is None:
        model_name = os.path.basename(args.model).replace(".gguf", "").lower()
        model_name = model_name.replace("-", "_").replace(".", "_")
        lang_str = args.lang.replace("-", "_")
        args.output = os.path.join(
            os.path.dirname(__file__), "heads", "configs",
            f"translation_heads_{model_name}_{lang_str}.json",
        )

    if args.heatmap is None:
        args.heatmap = args.output.replace(".json", "_heatmap.png")

    result = detect_heads(
        model_path=args.model,
        direction=args.lang,
        prompt_format=args.prompt_format,
        n_sentences=args.num_sentences,
        n_ctx=args.n_ctx,
        max_gen=args.max_gen,
        batch_heads=args.batch_heads,
        skip_gdn_layers=args.skip_gdn_layers,
        ts_threshold=args.ts_threshold,
    )

    # Save JSON
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {args.output}", file=sys.stderr)

    # Save heatmap
    save_heatmap(result, args.heatmap)


if __name__ == "__main__":
    main()
