"""OmniSTEval output format for IWSLT 2026 submission.

Converts SimulMT evaluation results into formats compatible with
OmniSTEval (https://github.com/pe-trik/OmniSTEval).

Two output formats supported:

1. **SimulEval JSONL** (PRIMARY -- what OmniSTEval expects):
   One line per segment/sentence:
   {
       "source": str,           # Source audio filename or text
       "prediction": str,       # Full hypothesis text
       "delays": [float, ...],  # Per-word CU emission times (ms)
       "elapsed": [float, ...], # Per-word CA emission times (ms)
       "source_length": float   # Source duration in ms
   }
   len(delays) == len(prediction.split()) -- CRITICAL

2. **Emission event JSONL** (legacy, for SimulStream logs):
   Multiple lines per segment, one per emission event:
   {
       "talk_id": str,
       "offset": float,
       "duration": float,
       "emission_cu": float,
       "emission_ca": float,
       "text": str,
       "is_eos": bool
   }

Usage:
    # SimulEval format (default, for OmniSTEval evaluation)
    python -m nllw.omnisteval results.json -o output.jsonl
    python -m nllw.bench --suite corpus --lang en-zh --omnisteval output.jsonl

    # Legacy emission event format
    python -m nllw.omnisteval results.json --legacy -o output.jsonl
"""

import json
import sys
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any, TextIO


# ---------------------------------------------------------------------------
# SimulEval JSONL format (what OmniSTEval actually expects)
# ---------------------------------------------------------------------------

@dataclass
class SimulEvalEntry:
    """One segment/sentence in SimulEval JSONL format.

    This is the format OmniSTEval expects for both shortform and longform
    evaluation. One line per segment, with per-word delay arrays.

    All timestamps in MILLISECONDS.
    """
    source: str = ""                     # Source audio filename or text
    prediction: str = ""                 # Full hypothesis text
    delays: List[float] = field(default_factory=list)    # CU emission times (ms), one per word
    elapsed: List[float] = field(default_factory=list)   # CA emission times (ms), one per word
    source_length: float = 0.0           # Source duration in ms
    index: Optional[int] = None          # Sequence number
    reference: str = ""                  # Reference text (optional)

    def to_dict(self) -> dict:
        d = {
            "source": self.source,
            "prediction": self.prediction,
            "delays": self.delays,
            "elapsed": self.elapsed,
            "source_length": self.source_length,
        }
        if self.index is not None:
            d["index"] = self.index
        if self.reference:
            d["reference"] = self.reference
        return d

    def validate(self) -> List[str]:
        """Check for format violations. Returns list of error strings."""
        errors = []
        n_words = len(self.prediction.split()) if self.prediction else 0
        if self.delays and len(self.delays) != n_words:
            errors.append(
                f"delays length ({len(self.delays)}) != prediction words ({n_words})"
            )
        if self.elapsed and len(self.elapsed) != n_words:
            errors.append(
                f"elapsed length ({len(self.elapsed)}) != prediction words ({n_words})"
            )
        if self.source_length <= 0:
            errors.append(f"source_length must be positive, got {self.source_length}")
        return errors


def eval_result_to_simuleval(
    eval_result_dict: Dict[str, Any],
    source_prefix: str = "segment",
    words_per_second: float = 2.5,
    source_length_ms: Optional[float] = None,
    include_reference: bool = False,
) -> List[SimulEvalEntry]:
    """Convert EvalResult to SimulEval JSONL format.

    This is the PRIMARY output format for OmniSTEval evaluation.

    Each sentence in per_sentence becomes one JSONL line with:
    - prediction: full hypothesis text
    - delays: per-word computation-unaware emission times in ms
    - elapsed: per-word computation-aware emission times in ms (same as CU if no timing)
    - source_length: source audio duration in ms

    The delays are computed from the word-level delay indices stored in
    per_sentence[i]["delays"], which are source word counts at emission time.
    These are converted to milliseconds using words_per_second.

    Args:
        eval_result_dict: Dict with "per_sentence" key from EvalResult
        source_prefix: Prefix for source identifiers
        words_per_second: Words per second for time conversion (default 2.5)
        source_length_ms: Override source length per sentence (ms). If None,
            computed from source word count * words_per_second.
        include_reference: Include reference text in output

    Returns:
        List of SimulEvalEntry, one per sentence
    """
    per_sentence = eval_result_dict.get("per_sentence", [])
    entries = []

    for i, sent in enumerate(per_sentence):
        source = sent.get("source", "")
        hypothesis = sent.get("hypothesis", "")
        reference = sent.get("reference", "")
        word_delays = sent.get("delays", [])  # Source word indices at emission

        n_source = len(source.split())
        hyp_words = hypothesis.split() if hypothesis else []
        n_hyp = len(hyp_words)

        # Compute source length in ms
        if source_length_ms is not None:
            src_len_ms = source_length_ms
        else:
            # Convert source word count to ms via words_per_second
            src_len_ms = (n_source / words_per_second) * 1000.0 if words_per_second > 0 else 0.0

        # Convert word-index delays to milliseconds
        # delays[t] = source word index when target word t was emitted
        # Convert: delay_ms[t] = delays[t] / words_per_second * 1000
        ms_per_word = (1.0 / words_per_second * 1000.0) if words_per_second > 0 else 400.0

        delays_ms = []
        elapsed_ms = []
        for t in range(min(n_hyp, len(word_delays))):
            d = word_delays[t]
            cu_ms = round(d * ms_per_word, 1)
            delays_ms.append(cu_ms)
            # For CA, add generation time if available
            gen_time_ms = sent.get("generation_time_ms", 0.0)
            ca_ms = round(cu_ms + gen_time_ms / max(n_hyp, 1), 1)
            elapsed_ms.append(ca_ms)

        # Pad if delays shorter than hypothesis (shouldn't happen, but be safe)
        while len(delays_ms) < n_hyp:
            delays_ms.append(round(src_len_ms, 1))
            elapsed_ms.append(round(src_len_ms, 1))

        entry = SimulEvalEntry(
            source=f"{source_prefix}_{i}.wav",
            prediction=hypothesis,
            delays=delays_ms,
            elapsed=elapsed_ms,
            source_length=round(src_len_ms, 1),
            index=i,
            reference=reference if include_reference else "",
        )
        entries.append(entry)

    return entries


def write_simuleval_jsonl(entries: List[SimulEvalEntry], output: TextIO):
    """Write SimulEval entries as JSONL."""
    for entry in entries:
        json.dump(entry.to_dict(), output, ensure_ascii=False)
        output.write("\n")


def write_simuleval_jsonl_file(
    entries: List[SimulEvalEntry],
    output_path: str,
    validate: bool = True,
):
    """Write SimulEval entries as JSONL to a file path.

    Args:
        entries: SimulEvalEntry list
        output_path: Output file path
        validate: If True, validate entries and warn on errors
    """
    if validate:
        for i, entry in enumerate(entries):
            errors = entry.validate()
            for err in errors:
                print(f"WARNING: Entry {i}: {err}", file=sys.stderr)

    with open(output_path, "w") as f:
        write_simuleval_jsonl(entries, f)

    n_words = sum(len(e.prediction.split()) for e in entries)
    print(
        f"OmniSTEval JSONL written: {output_path} "
        f"({len(entries)} segments, {n_words} words)",
        file=sys.stderr,
    )


def compute_longyaal_from_entries(entries: List[SimulEvalEntry]) -> Dict[str, float]:
    """Compute LongYAAL from SimulEval entries -- local validation.

    This replicates OmniSTEval's YAALScorer.score() logic to verify
    our output would produce correct LongYAAL values before submission.

    Args:
        entries: SimulEvalEntry list from eval_result_to_simuleval()

    Returns:
        Dict with per-entry and average LongYAAL values:
        {"per_entry": [...], "average": float, "n_entries": int}
    """
    from .metrics import compute_longyaal_ms

    per_entry = []
    for entry in entries:
        n_words = len(entry.prediction.split()) if entry.prediction else 0
        ly = compute_longyaal_ms(entry.delays, entry.source_length, n_words)
        per_entry.append(ly)

    avg = sum(per_entry) / len(per_entry) if per_entry else 0.0
    return {
        "per_entry": per_entry,
        "average_ms": avg,
        "average_s": avg / 1000.0,
        "n_entries": len(per_entry),
    }


# ---------------------------------------------------------------------------
# Legacy emission event format (for SimulStream compatibility)
# ---------------------------------------------------------------------------

@dataclass
class OmniSTEvalEntry:
    """A single emission event in legacy per-event format.

    This is NOT what OmniSTEval expects for evaluation. It's kept for
    SimulStream log compatibility. Use SimulEvalEntry for evaluation.
    """
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
    """Convert a SimulationTrace dict to legacy per-event entries.

    NOTE: This produces the LEGACY format (multiple lines per segment).
    For OmniSTEval evaluation, use eval_result_to_simuleval() instead.
    """
    actions = trace_dict.get("actions", [])
    n_source = trace_dict.get("n_source", 0)

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

    if entries:
        entries[-1].is_eos = True

    return entries


def eval_result_to_omnisteval(
    eval_result_dict: Dict[str, Any],
    talk_id_prefix: str = "talk",
    source_length_s: Optional[float] = None,
) -> List[OmniSTEvalEntry]:
    """Convert EvalResult to legacy per-event entries.

    NOTE: This produces the LEGACY format. For OmniSTEval, use
    eval_result_to_simuleval() instead.
    """
    per_sentence = eval_result_dict.get("per_sentence", [])
    all_entries = []

    for i, sent in enumerate(per_sentence):
        talk_id = f"{talk_id_prefix}_{i}"

        source = sent.get("source", "")
        n_source = len(source.split())
        delays = sent.get("delays", [])

        actions = []
        hypothesis = sent.get("hypothesis", "")
        hyp_words = hypothesis.split() if hypothesis else []

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

    return all_entries


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def write_jsonl(entries: List[OmniSTEvalEntry], output: TextIO):
    """Write legacy per-event entries as JSONL."""
    for entry in entries:
        json.dump(entry.to_dict(), output, ensure_ascii=False)
        output.write("\n")


def write_jsonl_file(entries: List[OmniSTEvalEntry], output_path: str):
    """Write legacy per-event entries as JSONL to a file path."""
    with open(output_path, "w") as f:
        write_jsonl(entries, f)
    print(f"OmniSTEval JSONL written: {output_path} ({len(entries)} entries)", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert SimulMT results to OmniSTEval JSONL format",
    )
    parser.add_argument("input", help="Input JSON file (eval result)")
    parser.add_argument("--talk-id", default="talk_0", help="Talk identifier / source prefix")
    parser.add_argument("--source-length", type=float, default=None,
                        help="Source audio length in seconds (for time domain)")
    parser.add_argument("--wps", type=float, default=2.5,
                        help="Words per second for ms conversion (default: 2.5)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output JSONL file (default: stdout)")
    parser.add_argument("--legacy", action="store_true",
                        help="Use legacy per-event format (NOT for OmniSTEval evaluation)")
    parser.add_argument("--include-reference", action="store_true",
                        help="Include reference translations in output")
    parser.add_argument("--word-level", action="store_true", default=True,
                        help="Word-level delays (default)")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    if args.legacy:
        # Legacy per-event format
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
            print("Error: input must have 'actions' or 'per_sentence'", file=sys.stderr)
            sys.exit(1)

        if args.output:
            write_jsonl_file(entries, args.output)
        else:
            write_jsonl(entries, sys.stdout)
    else:
        # Standard SimulEval format (for OmniSTEval)
        if "per_sentence" not in data:
            print("Error: input must have 'per_sentence' for SimulEval format", file=sys.stderr)
            sys.exit(1)

        source_length_ms = args.source_length * 1000.0 if args.source_length else None
        entries = eval_result_to_simuleval(
            data,
            source_prefix=args.talk_id,
            words_per_second=args.wps,
            source_length_ms=source_length_ms,
            include_reference=args.include_reference,
        )

        if args.output:
            write_simuleval_jsonl_file(entries, args.output)
        else:
            write_simuleval_jsonl(entries, sys.stdout)


if __name__ == "__main__":
    main()
