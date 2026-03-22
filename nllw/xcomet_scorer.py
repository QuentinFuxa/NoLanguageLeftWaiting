"""Standalone XCOMET-XL scorer for separate-process evaluation.

XCOMET-XL (~12GB VRAM) OOMs when the translation model is still loaded.
This module runs XCOMET-XL scoring in a separate process so the translation
model's VRAM is fully freed first.

Usage as CLI:
    # Score from eval result JSON (per_sentence has source/reference/hypothesis)
    python -m nllw.xcomet_scorer --input results.json --output scored.json

    # Score from hypothesis file (one per line, matching FLORES order)
    python -m nllw.xcomet_scorer --hyps hyps.txt --lang en-zh -n 100

    # Use as subprocess from Python
    from nllw.xcomet_scorer import score_xcomet_subprocess
    scores = score_xcomet_subprocess("results.json")

Design: the scorer loads ONLY comet/torch -- no llama, no backend. This
guarantees the full GPU VRAM is available for XCOMET-XL (12GB for model +
batch memory).
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple


def score_xcomet(
    sources: List[str],
    hypotheses: List[str],
    references: List[str],
    model_name: str = "Unbabel/XCOMET-XL",
    batch_size: int = 4,
    gpus: int = 1,
) -> Tuple[float, List[float]]:
    """Score translations with XCOMET-XL.

    This function should be called in a process where no other large models
    are loaded on the GPU.

    Args:
        sources: Source texts
        hypotheses: MT outputs
        references: Reference translations
        model_name: COMET model name
        batch_size: Inference batch size (lower = less VRAM)
        gpus: Number of GPUs

    Returns:
        (corpus_score, per_sentence_scores)
    """
    import torch
    torch.set_float32_matmul_precision('medium')

    from comet import download_model, load_from_checkpoint

    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)

    data = [
        {"src": s, "mt": h, "ref": r}
        for s, h, r in zip(sources, hypotheses, references)
    ]

    output = model.predict(data, batch_size=batch_size, gpus=gpus)
    return output.system_score, list(output.scores)


def score_from_eval_json(
    input_path: str,
    model_name: str = "Unbabel/XCOMET-XL",
    batch_size: int = 4,
) -> Tuple[float, List[float]]:
    """Score translations from an eval result JSON file.

    The JSON should have a "per_sentence" key with list of dicts containing
    "source", "reference", "hypothesis" keys.

    Args:
        input_path: Path to eval result JSON
        model_name: COMET model name
        batch_size: Inference batch size

    Returns:
        (corpus_score, per_sentence_scores)
    """
    with open(input_path) as f:
        data = json.load(f)

    per_sentence = data.get("per_sentence", [])
    if not per_sentence:
        raise ValueError(f"No per_sentence data in {input_path}")

    sources = [s["source"] for s in per_sentence]
    hypotheses = [s["hypothesis"] for s in per_sentence]
    references = [s["reference"] for s in per_sentence]

    return score_xcomet(sources, hypotheses, references, model_name, batch_size)


def score_xcomet_subprocess(
    input_path: str,
    model_name: str = "Unbabel/XCOMET-XL",
    batch_size: int = 4,
    output_path: Optional[str] = None,
    timeout: int = 600,
) -> Optional[Dict]:
    """Run XCOMET-XL scoring in a separate subprocess.

    This is the recommended way to score when a translation model may still
    hold GPU VRAM. The subprocess starts fresh with no GPU allocations.

    Args:
        input_path: Path to eval result JSON with per_sentence data
        model_name: COMET model name
        batch_size: Inference batch size
        output_path: Where to write scored JSON. If None, uses temp file.
        timeout: Subprocess timeout in seconds

    Returns:
        Dict with "system_score" and "per_sentence_scores", or None on failure
    """
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".json", prefix="xcomet_")
        os.close(fd)
        cleanup = True
    else:
        cleanup = False

    cmd = [
        sys.executable, "-m", "nllw.xcomet_scorer",
        "--input", input_path,
        "--output", output_path,
        "--model", model_name,
        "--batch-size", str(batch_size),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        if result.returncode != 0:
            print(f"XCOMET-XL subprocess failed (rc={result.returncode}):",
                  file=sys.stderr)
            print(result.stderr[-2000:] if result.stderr else "(no stderr)",
                  file=sys.stderr)
            return None

        with open(output_path) as f:
            scores = json.load(f)
        return scores

    except subprocess.TimeoutExpired:
        print(f"XCOMET-XL subprocess timed out after {timeout}s", file=sys.stderr)
        return None
    except Exception as e:
        print(f"XCOMET-XL subprocess error: {e}", file=sys.stderr)
        return None
    finally:
        if cleanup and os.path.exists(output_path):
            os.remove(output_path)


def save_hypotheses_json(
    eval_result,
    output_path: str,
) -> str:
    """Save evaluation results to JSON for separate XCOMET-XL scoring.

    Args:
        eval_result: EvalResult from evaluate_backend()
        output_path: Path to write JSON

    Returns:
        The output path
    """
    data = {
        "summary": eval_result.to_dict(),
        "per_sentence": eval_result.per_sentence,
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="XCOMET-XL scorer (separate process, no translation model loaded)"
    )
    parser.add_argument("--input", required=True,
                        help="Eval result JSON with per_sentence data")
    parser.add_argument("--output", help="Output JSON with scores")
    parser.add_argument("--model", default="Unbabel/XCOMET-XL",
                        help="COMET model name")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Inference batch size (lower = less VRAM)")
    parser.add_argument("--hyps", help="Hypothesis file (one per line)")
    parser.add_argument("--lang", help="Language direction for FLORES loading")
    parser.add_argument("-n", type=int, default=100,
                        help="Number of sentences (for FLORES)")

    args = parser.parse_args()

    # Mode 1: Score from eval result JSON
    if args.input and not args.hyps:
        print(f"Loading data from {args.input}...", file=sys.stderr)
        system_score, per_scores = score_from_eval_json(
            args.input, args.model, args.batch_size
        )

    # Mode 2: Score from hypothesis file + FLORES references
    elif args.hyps:
        if not args.lang:
            print("--lang required with --hyps", file=sys.stderr)
            sys.exit(1)

        parts = args.lang.split("-")
        src_lang, tgt_lang = parts[0], parts[1]

        # Load FLORES references
        from .eval import load_flores
        corpus = load_flores(src_lang, tgt_lang, n=args.n)

        # Load hypotheses
        with open(args.hyps) as f:
            hyps = [line.strip() for line in f if line.strip()]

        if len(hyps) != len(corpus):
            print(f"Mismatch: {len(hyps)} hyps vs {len(corpus)} refs",
                  file=sys.stderr)
            sys.exit(1)

        sources = [s["source"] for s in corpus]
        references = [s["reference"] for s in corpus]

        system_score, per_scores = score_xcomet(
            sources, hyps, references, args.model, args.batch_size
        )
    else:
        print("Provide --input (eval JSON) or --hyps (hypothesis file)",
              file=sys.stderr)
        sys.exit(1)

    # Print result
    print(f"\nXCOMET-XL ({args.model}): {system_score:.4f}", file=sys.stderr)
    print(f"  Per-sentence range: [{min(per_scores):.4f}, {max(per_scores):.4f}]",
          file=sys.stderr)

    # Save scores
    result = {
        "model": args.model,
        "system_score": system_score,
        "per_sentence_scores": per_scores,
        "n_sentences": len(per_scores),
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Scores saved to {args.output}", file=sys.stderr)
    else:
        # Print to stdout for piping
        json.dump(result, sys.stdout, indent=2)
        print()


if __name__ == "__main__":
    main()
