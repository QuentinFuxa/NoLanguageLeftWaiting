#!/usr/bin/env python3
"""
Evaluation harness for AlignAtt simultaneous translation.

Supports:
- FLORES dataset evaluation (sentence-level streaming)
- BLEU, COMET, XCOMET-XL scoring
- Latency metrics (AL, LAAL, AP, MaxCW)
- Parameter sweep across border_distance, word_batch, etc.
- Results export (JSON + text files for comet-score CLI)

Usage:
    python -m nllw.eval \
        --model /path/to/model.gguf \
        --heads /path/to/heads.json \
        --prompt-format hymt \
        --lang en-zh \
        --border-distance 3 \
        -n 50 \
        --output results.json
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

from nllw.alignatt import (
    AlignAttConfig,
    AlignmentHead,
    TranslationSegment,
    load_heads,
    stream_translate,
)
from nllw.llama_backend import LlamaModel, init as llama_init
from nllw.metrics import LatencyMetrics, compute_latency_metrics
from nllw.prompts import PromptFormat, get_format


LANG_CONFIGS = {
    "en-zh": {"src_lang": "eng", "src_script": "Latn", "tgt_lang": "cmn", "tgt_script": "Hans"},
    "en-de": {"src_lang": "eng", "src_script": "Latn", "tgt_lang": "deu", "tgt_script": "Latn"},
    "en-it": {"src_lang": "eng", "src_script": "Latn", "tgt_lang": "ita", "tgt_script": "Latn"},
    "cs-en": {"src_lang": "ces", "src_script": "Latn", "tgt_lang": "eng", "tgt_script": "Latn"},
    "en-fr": {"src_lang": "eng", "src_script": "Latn", "tgt_lang": "fra", "tgt_script": "Latn"},
    "en-es": {"src_lang": "eng", "src_script": "Latn", "tgt_lang": "spa", "tgt_script": "Latn"},
    "en-ja": {"src_lang": "eng", "src_script": "Latn", "tgt_lang": "jpn", "tgt_script": "Jpan"},
    "en-ru": {"src_lang": "eng", "src_script": "Latn", "tgt_lang": "rus", "tgt_script": "Cyrl"},
}


def load_flores(
    lang_pair: str,
    n: int = 50,
) -> List[Dict[str, str]]:
    """Load FLORES+ sentence pairs for evaluation.

    Returns list of {"id": ..., "source": ..., "reference": ...}.
    """
    from datasets import load_dataset

    cfg = LANG_CONFIGS[lang_pair]
    ds = load_dataset("openlanguagedata/flores_plus", split="dev")

    src_ds = ds.filter(
        lambda x: x["iso_639_3"] == cfg["src_lang"] and x["iso_15924"] == cfg["src_script"]
    )
    tgt_ds = ds.filter(
        lambda x: x["iso_639_3"] == cfg["tgt_lang"] and x["iso_15924"] == cfg["tgt_script"]
    )

    src_map = {row["id"]: row["text"] for row in src_ds}
    tgt_map = {row["id"]: row["text"] for row in tgt_ds}
    common_ids = sorted(set(src_map) & set(tgt_map))

    pairs = []
    for sid in common_ids[:n]:
        pairs.append({
            "id": int(sid),
            "source": src_map[sid],
            "reference": tgt_map[sid],
        })
    return pairs


def evaluate_streaming(
    model: LlamaModel,
    pairs: List[Dict[str, str]],
    heads: List[AlignmentHead],
    fmt: PromptFormat,
    config: AlignAttConfig,
    verbose: bool = True,
) -> Dict:
    """Run streaming evaluation on sentence pairs.

    Returns dict with summary statistics and per-sentence results.
    """
    results = []
    al_list, laal_list, ap_list, cw_list = [], [], [], []

    t0_all = time.time()

    for idx, pair in enumerate(pairs):
        t0 = time.time()

        segment = stream_translate(
            model=model,
            source_text=pair["source"],
            heads=heads,
            fmt=fmt,
            config=config,
        )

        elapsed = time.time() - t0

        metrics = compute_latency_metrics(
            segment.delays, segment.num_source_words, segment.num_target_tokens,
        )
        al_list.append(metrics.al)
        laal_list.append(metrics.laal)
        ap_list.append(metrics.ap)
        cw_list.append(metrics.max_cw)

        n_borders = sum(1 for e in segment.emissions if e.get("border", False))

        results.append({
            "id": pair["id"],
            "source": pair["source"],
            "reference": pair["reference"],
            "translation": segment.translation,
            "num_source_words": segment.num_source_words,
            "num_target_tokens": segment.num_target_tokens,
            "al": metrics.al,
            "laal": metrics.laal,
            "ap": metrics.ap,
            "max_cw": metrics.max_cw,
            "time_ms": round(elapsed * 1000),
            "border_stops": n_borders,
        })

        if verbose and ((idx + 1) % 5 == 0 or idx == 0):
            avg_al = np.mean(al_list)
            total_elapsed = time.time() - t0_all
            eta = total_elapsed / (idx + 1) * (len(pairs) - idx - 1)
            trans = segment.translation[:50] if segment.translation else "(empty)"
            print(
                f"  [{idx+1}/{len(pairs)}] AL={avg_al:.2f} borders={n_borders} "
                f"{elapsed*1000:.0f}ms | {trans}... | ETA: {eta:.0f}s",
                flush=True,
            )

    total_time = time.time() - t0_all

    al_arr = np.array(al_list)
    laal_arr = np.array(laal_list)
    ap_arr = np.array(ap_list)
    cw_arr = np.array(cw_list)

    summary = {
        "num_sentences": len(results),
        "border_distance": config.border_distance,
        "word_batch": config.word_batch,
        "num_alignment_heads": len(heads),
        "latency": {
            "avg_al": round(float(np.mean(al_arr)), 3),
            "median_al": round(float(np.median(al_arr)), 3),
            "p90_al": round(float(np.percentile(al_arr, 90)), 3),
            "avg_laal": round(float(np.mean(laal_arr)), 3),
            "avg_ap": round(float(np.mean(ap_arr)), 3),
            "avg_max_cw": round(float(np.mean(cw_arr)), 1),
            "max_max_cw": int(np.max(cw_arr)),
        },
        "avg_time_ms": round(float(np.mean([r["time_ms"] for r in results]))),
        "total_time_s": round(total_time, 1),
    }

    return {"summary": summary, "sentences": results}


def save_results(
    output_data: Dict,
    output_path: str,
    lang_pair: str = "",
    prompt_format: str = "",
):
    """Save results as JSON + text files for COMET scoring."""
    # Add metadata
    output_data["summary"]["system"] = (
        f"AlignAtt streaming (NLLW, llama.cpp KV cache, {prompt_format})"
    )
    output_data["summary"]["language_pair"] = lang_pair

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Save text files for comet-score CLI
    base = output_path.replace(".json", "")
    with open(f"{base}_hyp.txt", "w") as fh, \
         open(f"{base}_ref.txt", "w") as fr, \
         open(f"{base}_src.txt", "w") as fs:
        for r in output_data["sentences"]:
            fh.write(r["translation"].strip() + "\n")
            fr.write(r["reference"].strip() + "\n")
            fs.write(r["source"].strip() + "\n")

    print(f"\nSaved to {output_path}")
    print(f"  Hypothesis: {base}_hyp.txt")
    print(f"  Reference:  {base}_ref.txt")
    print(f"  Source:     {base}_src.txt")
    print(f"\nScore with: comet-score -s {base}_src.txt -t {base}_hyp.txt -r {base}_ref.txt")


def print_summary(summary: Dict, lang_pair: str = ""):
    """Pretty-print evaluation summary."""
    print(f"\n{'='*60}")
    print(f"AlignAtt Streaming Evaluation -- {lang_pair}")
    print(f"  Border distance: {summary['border_distance']}")
    print(f"  Word batch: {summary['word_batch']}")
    print(f"  Sentences: {summary['num_sentences']}")
    print(f"{'='*60}")
    lat = summary["latency"]
    print(f"  Avg AL:      {lat['avg_al']:.3f}")
    print(f"  Avg LAAL:    {lat['avg_laal']:.3f}")
    print(f"  Avg AP:      {lat['avg_ap']:.3f}")
    print(f"  Avg MaxCW:   {lat['avg_max_cw']:.1f}")
    print(f"  Avg time:    {summary['avg_time_ms']}ms/sentence")
    print(f"  Total:       {summary['total_time_s']}s")


def main():
    parser = argparse.ArgumentParser(
        description="AlignAtt streaming evaluation on FLORES",
    )
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--heads", required=True, help="Alignment heads JSON file")
    parser.add_argument("--prompt-format", required=True, help="Prompt format name")
    parser.add_argument("--lang", required=True, choices=list(LANG_CONFIGS.keys()),
                        help="Language pair")
    parser.add_argument("-n", type=int, default=50, help="Number of sentences")
    parser.add_argument("--border-distance", type=int, default=3)
    parser.add_argument("--word-batch", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--context-sentences", type=int, default=0)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--output", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"results_{args.lang.replace('-', '_')}_bd{args.border_distance}.json"

    # Load heads
    heads = load_heads(args.heads, top_k=args.top_k)
    print(f"Loaded {len(heads)} heads, best TS={heads[0].ts:.3f}")

    # Load FLORES
    print(f"Loading FLORES {args.lang} ({args.n} sentences)...")
    pairs = load_flores(args.lang, n=args.n)
    print(f"  {len(pairs)} sentence pairs loaded")

    # Init model
    print(f"Loading model: {args.model}")
    llama_init()
    model = LlamaModel(args.model, n_gpu_layers=99)

    # Get prompt format
    fmt = get_format(args.prompt_format)

    # Config
    config = AlignAttConfig(
        border_distance=args.border_distance,
        top_k_heads=args.top_k,
        word_batch=args.word_batch,
        context_sentences=args.context_sentences,
        n_ctx=args.n_ctx,
    )

    # Run evaluation
    print(f"\nRunning AlignAtt streaming (bd={args.border_distance}, wb={args.word_batch})...")
    output_data = evaluate_streaming(
        model=model,
        pairs=pairs,
        heads=heads,
        fmt=fmt,
        config=config,
        verbose=not args.quiet,
    )

    # Print and save
    print_summary(output_data["summary"], args.lang)
    save_results(output_data, args.output, args.lang, args.prompt_format)

    model.close()


if __name__ == "__main__":
    main()
