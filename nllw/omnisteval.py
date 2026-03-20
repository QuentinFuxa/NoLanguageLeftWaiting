"""OmniSTEval JSONL output format for IWSLT 2026 submission.

Converts SimulMT evaluation traces into the official IWSLT 2026 evaluation
format used by OmniSTEval (https://github.com/pe-trik/OmniSTEval).

OmniSTEval format (one JSONL file per talk):
    Each line is a JSON object representing one emission event:
    {
        "talk_id": str,          # Talk/document identifier
        "offset": float,         # Source offset at emission time (seconds or words)
        "duration": float,       # Duration of source chunk (seconds or words)
        "emission_cu": float,    # Computation-unaware emission time (ideal)
        "emission_ca": float,    # Computation-aware emission time (with latency)
        "text": str,             # Emitted target text
        "is_eos": bool           # End of segment marker
    }

Usage:
    # From evaluation result
    python -m nllw.omnisteval traces.json --talk-id demo --source-length 120.5 -o output.jsonl

    # From bench CLI
    python -m nllw.bench --suite corpus --lang en-zh --omnisteval output.jsonl
"""

import json
import sys
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, TextIO


@dataclass
class OmniSTEvalEntry:
    """A single emission event in OmniSTEval format."""
    talk_id: str
    offset: float          # Source position at emission (cumulative words or seconds)
    duration: float        # Source chunk duration (words or seconds since last READ)
    emission_cu: float     # Computation-unaware timestamp
    emission_ca: float     # Computation-aware timestamp (includes inference time)
    text: str              # Emitted target text
    is_eos: bool = False   # End of segment

    def to_dict(self) -> dict:
        return asdict(self)


def trace_to_omnisteval(
    trace_dict: Dict[str, Any],
    talk_id: str = "talk_0",
    source_length_s: Optional[float] = None,
    words_per_second: float = 2.5,
) -> List[OmniSTEvalEntry]:
    """Convert a SimulationTrace dict to OmniSTEval entries.

    Handles two domains:
    - Word domain: delays are in word counts (default from simulate_backend)
    - Time domain: delays are in seconds (requires source_length_s)

    Args:
        trace_dict: Output of SimulationTrace.to_dict()
        talk_id: Identifier for the talk/document
        source_length_s: Total source audio length in seconds (None = word domain)
        words_per_second: Words per second for time conversion (default 2.5)

    Returns:
        List of OmniSTEvalEntry objects
    """
    actions = trace_dict.get("actions", [])
    n_source = trace_dict.get("n_source", 0)

    # Convert word indices to time if source_length_s is provided
    if source_length_s is not None and n_source > 0:
        time_per_word = source_length_s / n_source
    else:
        time_per_word = 1.0 / words_per_second if words_per_second > 0 else 0.4

    entries = []
    cumulative_src_time = 0.0
    last_read_time = 0.0
    prev_word_idx = -1

    for action in actions:
        if action["action"] == "READ":
            word_idx = action["word_idx"]
            if source_length_s is not None:
                cumulative_src_time = (word_idx + 1) * time_per_word
            else:
                cumulative_src_time = action.get("emission_time", float(word_idx))
            last_read_time = cumulative_src_time
            prev_word_idx = word_idx

        elif action["action"] == "WRITE":
            text = action.get("text", "").strip()
            if not text:
                continue

            word_idx = action.get("word_idx", prev_word_idx)
            # Duration = time since last emission
            if source_length_s is not None:
                offset = (word_idx + 1) * time_per_word
            else:
                offset = float(word_idx + 1)

            gen_time_ms = action.get("generation_time_ms", 0.0)
            gen_time_s = gen_time_ms / 1000.0

            entries.append(OmniSTEvalEntry(
                talk_id=talk_id,
                offset=round(offset, 3),
                duration=round(time_per_word, 3),
                emission_cu=round(offset, 3),
                emission_ca=round(offset + gen_time_s, 3),
                text=text,
                is_eos=False,
            ))

    # Mark last entry as EOS
    if entries:
        entries[-1].is_eos = True

    return entries


def eval_result_to_omnisteval(
    eval_result_dict: Dict[str, Any],
    talk_id_prefix: str = "talk",
    source_length_s: Optional[float] = None,
) -> List[OmniSTEvalEntry]:
    """Convert a full EvalResult (with per_sentence data) to OmniSTEval entries.

    Each sentence becomes a separate talk segment.

    Args:
        eval_result_dict: Dict with "per_sentence" key from EvalResult
        talk_id_prefix: Prefix for talk IDs
        source_length_s: Total audio length (distributed across sentences)

    Returns:
        List of OmniSTEvalEntry
    """
    per_sentence = eval_result_dict.get("per_sentence", [])
    all_entries = []

    for i, sent in enumerate(per_sentence):
        talk_id = f"{talk_id_prefix}_{i}"

        # Build a minimal trace dict from per_sentence data
        source = sent.get("source", "")
        n_source = len(source.split())
        delays = sent.get("delays", [])

        # Reconstruct WRITE actions from delays
        actions = []
        hypothesis = sent.get("hypothesis", "")
        hyp_words = hypothesis.split() if hypothesis else []

        # Group consecutive delays to reconstruct emission events
        if delays and hyp_words:
            prev_delay = -1
            current_text = []
            for t, d in enumerate(delays):
                if t < len(hyp_words):
                    if d != prev_delay and current_text:
                        actions.append({
                            "action": "WRITE",
                            "word_idx": int(prev_delay) - 1,
                            "text": " ".join(current_text),
                        })
                        current_text = []
                    # Add READ for each new source position
                    for w in range(max(0, int(prev_delay)), int(d)):
                        actions.append({
                            "action": "READ",
                            "word_idx": w,
                            "emission_time": float(w),
                        })
                    current_text.append(hyp_words[t])
                    prev_delay = d

            if current_text:
                actions.append({
                    "action": "WRITE",
                    "word_idx": int(prev_delay) - 1 if prev_delay >= 0 else 0,
                    "text": " ".join(current_text),
                })

        trace_dict = {
            "actions": actions,
            "n_source": n_source,
        }

        entries = trace_to_omnisteval(
            trace_dict, talk_id=talk_id, source_length_s=source_length_s,
        )
        all_entries.extend(entries)

    return entries


def write_jsonl(entries: List[OmniSTEvalEntry], output: TextIO):
    """Write OmniSTEval entries as JSONL to a file object."""
    for entry in entries:
        json.dump(entry.to_dict(), output, ensure_ascii=False)
        output.write("\n")


def write_jsonl_file(entries: List[OmniSTEvalEntry], output_path: str):
    """Write OmniSTEval entries as JSONL to a file path."""
    with open(output_path, "w") as f:
        write_jsonl(entries, f)
    print(f"OmniSTEval JSONL written: {output_path} ({len(entries)} entries)", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert SimulMT traces to OmniSTEval JSONL format",
    )
    parser.add_argument("input", help="Input JSON file (trace or eval result)")
    parser.add_argument("--talk-id", default="talk_0", help="Talk identifier")
    parser.add_argument("--source-length", type=float, default=None,
                        help="Source audio length in seconds (for time domain)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output JSONL file (default: stdout)")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    # Detect format: single trace vs eval result
    if "per_sentence" in data:
        entries = eval_result_to_omnisteval(
            data, talk_id_prefix=args.talk_id,
            source_length_s=args.source_length,
        )
    elif "actions" in data:
        entries = trace_to_omnisteval(
            data, talk_id=args.talk_id,
            source_length_s=args.source_length,
        )
    else:
        print("Error: input must have 'actions' (trace) or 'per_sentence' (eval result)",
              file=sys.stderr)
        sys.exit(1)

    if args.output:
        write_jsonl_file(entries, args.output)
    else:
        write_jsonl(entries, sys.stdout)


if __name__ == "__main__":
    main()
