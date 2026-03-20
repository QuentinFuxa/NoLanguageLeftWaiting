"""OmniSTEval output format for IWSLT 2026 SST submission.

Converts NLLW translation traces to OmniSTEval JSONL format, which is
required for official IWSLT simultaneous speech/text translation evaluation.

OmniSTEval format (one JSON line per recording/sentence):
    {
        "source": "recording.wav",
        "prediction": "translated text...",
        "delays": [1510.0, 1510.0, ...],   // per-char/word delays in ms
        "elapsed": [1510.0, 1510.0, ...],   // same as delays
        "source_length": 695000              // audio duration in ms
    }

References:
    - OmniSTEval: https://github.com/pe-trik/OmniSTEval
    - IWSLT 2026 SST: LongYAAL (primary), XCOMET-XL (quality)
    - Install: pip install . (from cloned OmniSTEval repo)
    - Run:
        omnisteval longform \\
            --speech_segmentation MASTER.yml \\
            --ref_sentences_file REF.txt \\
            --source_sentences_file SRC.txt \\
            --hypothesis_file HYP.jsonl \\
            --hypothesis_format jsonl \\
            --output_folder EVAL_DIR/ \\
            --lang zh --char_level \\
            --comet_model Unbabel/XCOMET-XL

Usage (as library):
    from nllw.omnisteval import trace_to_omnisteval, bench_to_omnisteval

    # From a single bench trace (per-sentence result from bench.py or research.py)
    record = trace_to_omnisteval(trace, talk_id="2022.acl-long.367",
                                  source_length_s=120.5)

    # From full bench results (list of sentence results)
    records = bench_to_omnisteval(results, talk_id="demo",
                                   source_length_s=120.5)

Usage (CLI):
    python -m nllw.omnisteval traces.json --talk-id demo --source-length 120.5 -o output.jsonl
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any, Optional


def _nfkc_expand(text: str, delay_ms: float) -> tuple[list[str], list[float]]:
    """Apply NFKC normalization and expand delays for multi-char expansions.

    E.g. ligature fi (U+FB01) expands to 'f','i', both get the same delay.
    """
    chars = []
    delays = []
    for ch in text:
        # Strip replacement characters (byte-fallback tokenizer artifacts)
        if ch == "\ufffd":
            continue
        normalized = unicodedata.normalize("NFKC", ch)
        for nc in normalized:
            chars.append(nc)
            delays.append(delay_ms)
    return chars, delays


def trace_to_omnisteval(
    trace: dict,
    *,
    talk_id: str = "sentence",
    source_length_s: Optional[float] = None,
    word_level: bool = False,
    speech_rate_wps: float = 2.5,
) -> dict:
    """Convert a single benchmark trace to OmniSTEval format.

    Parameters
    ----------
    trace : dict
        A per-sentence trace from bench.py or research.py. Expected keys:
        - steps: list of step dicts with word_index, step_time_ms, new_committed
        - hypothesis: full translation string
        - num_words or source (to count words)
        Alternatively, a simpler format with just:
        - hypothesis, committed_words (list of (word, emission_time_ms) tuples)
    talk_id : str
        Recording/sentence identifier.
    source_length_s : float or None
        Total source audio duration in seconds. If None, estimated from
        word count and speech_rate_wps.
    word_level : bool
        If True, produce word-level delays (for non-CJK).
        If False, produce character-level delays (for CJK like ZH/JA).
    speech_rate_wps : float
        Assumed speech rate for estimating source_length when not provided.

    Returns
    -------
    dict
        OmniSTEval-format record with source, prediction, delays, elapsed,
        source_length keys.
    """
    hypothesis = trace.get("hypothesis", "")
    steps = trace.get("steps", [])

    # Build emission timeline from steps
    # Each step has word_index, step_time_ms, new_committed
    # We need to map committed tokens to emission times

    # Calculate cumulative wall-clock time for each step
    cumulative_time_ms = 0.0
    step_emission_times = []
    for step in steps:
        step_ms = step.get("step_time_ms", 0.0)
        word_idx = step.get("word_index", len(step_emission_times))

        # In speech mode: word arrives at word_idx / speech_rate_wps
        arrival_ms = word_idx * (1000.0 / speech_rate_wps)
        # Processing starts at max(arrival, previous finish)
        start_ms = max(arrival_ms, cumulative_time_ms)
        cumulative_time_ms = start_ms + step_ms

        step_emission_times.append({
            "word_index": word_idx,
            "emission_time_ms": cumulative_time_ms,
            "new_committed": step.get("new_committed", 0),
        })

    # Build per-character delays using NFKC normalization
    if hypothesis and step_emission_times:
        # Map each committed token to its emission time
        # We need to distribute the hypothesis characters across steps
        # based on when tokens were committed

        # Approach: use the step emission times proportionally
        # Each step that committed N tokens gets N chars of the hypothesis
        committed_chars = []
        committed_delays = []

        total_committed = sum(s["new_committed"] for s in step_emission_times)
        hyp_chars_nfkc = []
        hyp_delays_nfkc = []
        for ch in hypothesis:
            nc, nd = _nfkc_expand(ch, 0.0)
            hyp_chars_nfkc.extend(nc)

        n_chars = len(hyp_chars_nfkc)

        if total_committed > 0 and n_chars > 0:
            # Assign characters proportionally to committed tokens
            char_idx = 0
            chars_per_token = max(1, n_chars / total_committed)

            for step_info in step_emission_times:
                n_new = step_info["new_committed"]
                emission_ms = step_info["emission_time_ms"]
                # How many chars this step covers
                n_step_chars = int(round(n_new * chars_per_token))
                for _ in range(n_step_chars):
                    if char_idx < n_chars:
                        committed_delays.append(emission_ms)
                        char_idx += 1

            # Fill remaining chars with last emission time
            last_time = step_emission_times[-1]["emission_time_ms"] if step_emission_times else 0.0
            while len(committed_delays) < n_chars:
                committed_delays.append(last_time)

            # Truncate if we overassigned
            committed_delays = committed_delays[:n_chars]
        else:
            # Fallback: all characters get the last emission time
            last_time = cumulative_time_ms
            committed_delays = [last_time] * n_chars

        # Re-build with proper NFKC normalization
        all_chars = []
        all_delays = []
        delay_idx = 0
        for ch in hypothesis:
            normalized = unicodedata.normalize("NFKC", ch)
            if ch == "\ufffd":
                continue
            for nc in normalized:
                delay = committed_delays[delay_idx] if delay_idx < len(committed_delays) else committed_delays[-1]
                all_chars.append(nc)
                all_delays.append(round(delay, 1))
                delay_idx += 1

        prediction = "".join(all_chars)
    else:
        prediction = unicodedata.normalize("NFKC", hypothesis.replace("\ufffd", ""))
        all_delays = [0.0] * len(prediction)

    # Source length
    if source_length_s is not None:
        source_length_ms = source_length_s * 1000.0
    else:
        # Estimate from word count
        n_words = trace.get("num_words", len(trace.get("source", "").split()))
        source_length_ms = n_words * (1000.0 / speech_rate_wps)

    if word_level:
        # Aggregate char delays to word-level
        words = prediction.split()
        word_delays = []
        char_idx = 0
        for word in words:
            # Skip spaces
            while char_idx < len(all_chars) and all_chars[char_idx] == " ":
                char_idx += 1
            last_delay = all_delays[char_idx] if char_idx < len(all_delays) else all_delays[-1] if all_delays else 0.0
            for _ in word:
                if char_idx < len(all_delays):
                    last_delay = all_delays[char_idx]
                char_idx += 1
            word_delays.append(last_delay)

        return {
            "source": f"{talk_id}.wav",
            "prediction": prediction,
            "delays": word_delays,
            "elapsed": word_delays,
            "source_length": round(source_length_ms, 1),
        }

    return {
        "source": f"{talk_id}.wav",
        "prediction": prediction,
        "delays": all_delays,
        "elapsed": all_delays,
        "source_length": round(source_length_ms, 1),
    }


def bench_to_omnisteval(
    sentences: list[dict],
    *,
    talk_id: str = "bench",
    source_length_s: Optional[float] = None,
    word_level: bool = False,
    speech_rate_wps: float = 2.5,
) -> list[dict]:
    """Convert bench.py sentence results to OmniSTEval records.

    Parameters
    ----------
    sentences : list[dict]
        List of per-sentence results from bench.py run_bench().
        Each must have at least 'hypothesis'.
    talk_id : str
        Base talk ID. Each sentence gets {talk_id}.{index}.
    source_length_s : float or None
        Total source length. If None, estimated per-sentence.
    word_level : bool
        Word-level delays (True) or char-level (False, default for CJK).
    speech_rate_wps : float
        Assumed speech rate for estimating source_length.

    Returns
    -------
    list[dict]
        List of OmniSTEval-format records.
    """
    records = []
    for i, sent in enumerate(sentences):
        record = trace_to_omnisteval(
            sent,
            talk_id=f"{talk_id}.{i}",
            source_length_s=source_length_s,
            word_level=word_level,
            speech_rate_wps=speech_rate_wps,
        )
        records.append(record)
    return records


def write_omnisteval_jsonl(
    records: list[dict],
    output_path: str | Path,
) -> Path:
    """Write OmniSTEval records to a JSONL file."""
    path = Path(output_path)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m nllw.omnisteval",
        description="Convert NLLW bench traces to OmniSTEval JSONL format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m nllw.omnisteval traces.json --talk-id demo --source-length 120.5 -o out.jsonl\n"
            "  python -m nllw.omnisteval bench_results.json --talk-id test --word-level -o out.jsonl\n"
            "\n"
            "Then evaluate with OmniSTEval:\n"
            "  omnisteval longform --hypothesis_file out.jsonl --hypothesis_format jsonl \\\n"
            "    --speech_segmentation MASTER.yml --ref_sentences_file REF.txt \\\n"
            "    --source_sentences_file SRC.txt --output_folder eval/ \\\n"
            "    --lang zh --char_level --comet_model Unbabel/XCOMET-XL\n"
        ),
    )

    parser.add_argument(
        "input",
        help="Input JSON file (bench results or list of traces)",
    )
    parser.add_argument(
        "--talk-id", "-t",
        default="bench",
        help="Talk/recording ID (default: bench)",
    )
    parser.add_argument(
        "--source-length", "-s",
        type=float, default=None,
        help="Source audio duration in seconds",
    )
    parser.add_argument(
        "--word-level", "-w",
        action="store_true",
        help="Word-level delays (default: char-level for CJK)",
    )
    parser.add_argument(
        "--speech-rate",
        type=float, default=2.5,
        help="Speech rate in words/sec for source length estimation (default: 2.5)",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output JSONL file path",
    )

    args = parser.parse_args()

    # Load input
    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    # Handle different input formats
    if isinstance(data, list):
        # List of results (bench output) or list of sentences
        if data and "sentences" in data[0]:
            # bench.py output: list of config results, each with sentences
            sentences = data[0]["sentences"]
        else:
            sentences = data
    elif isinstance(data, dict):
        if "sentences" in data:
            sentences = data["sentences"]
        else:
            sentences = [data]
    else:
        print(f"Unexpected input format: {type(data)}", file=sys.stderr)
        sys.exit(1)

    records = bench_to_omnisteval(
        sentences,
        talk_id=args.talk_id,
        source_length_s=args.source_length,
        word_level=args.word_level,
        speech_rate_wps=args.speech_rate,
    )

    path = write_omnisteval_jsonl(records, args.output)
    print(f"Wrote {len(records)} records to {path}")

    # Print summary
    if records:
        avg_delays = sum(len(r["delays"]) for r in records) / len(records)
        print(f"  Avg delays/record: {avg_delays:.0f}")
        unit = "words" if args.word_level else "chars"
        print(f"  Delay unit: {unit}")
        if records[0].get("source_length"):
            print(f"  Source length: {records[0]['source_length']/1000:.1f}s")


if __name__ == "__main__":
    import sys
    main()
