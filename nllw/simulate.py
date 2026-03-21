"""Policy simulation and Average Lagging computation.

Simulates simultaneous translation policies on source text:
given a full source sentence, replay the read/write decisions
that AlignAtt (or other policies) would make, and compute
the resulting delays and latency metrics.

This is useful for:
    - Offline evaluation without running the full LLM
    - Comparing policy configurations on the same data
    - Debugging border detection behavior
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from .metrics import compute_all_metrics, LatencyMetrics


@dataclass
class SimulationTrace:
    """Trace of a simulated translation session."""
    source_words: List[str]
    # Each entry: {"word_idx": int, "action": "READ"|"WRITE", "text": str, "emission_time": float}
    actions: List[Dict[str, Any]] = field(default_factory=list)
    # delays[t] = number of source words read when target word t was produced
    delays: List[float] = field(default_factory=list)
    # Full translation output
    translation: str = ""
    # Latency metrics
    metrics: Optional[LatencyMetrics] = None

    def compute_metrics(self) -> LatencyMetrics:
        """Compute latency metrics from the recorded delays."""
        n_src = len(self.source_words)
        n_tgt = len(self.delays)
        self.metrics = compute_all_metrics(self.delays, n_src, n_tgt)
        return self.metrics

    def to_dict(self) -> dict:
        return {
            "source": " ".join(self.source_words),
            "translation": self.translation,
            "delays": self.delays,
            "n_source": len(self.source_words),
            "n_target": len(self.delays),
            "metrics": {
                "al": self.metrics.al if self.metrics else None,
                "laal": self.metrics.laal if self.metrics else None,
                "yaal": self.metrics.yaal if self.metrics else None,
                "ap": self.metrics.ap if self.metrics else None,
                "dal": self.metrics.dal if self.metrics else None,
                "max_cw": self.metrics.max_cw if self.metrics else None,
            },
            "actions": self.actions,
        }


def simulate_backend(backend, source_text: str, is_final_on_last: bool = True) -> SimulationTrace:
    """Simulate a SimulMT backend on a single source sentence.

    Feeds words one at a time, records actions and delays.

    Args:
        backend: A SimulMTBackend instance
        source_text: Full source sentence
        is_final_on_last: Whether to mark the last word as is_final

    Returns:
        SimulationTrace with delays and metrics
    """
    words = source_text.strip().split()
    trace = SimulationTrace(source_words=words)
    target_word_count = 0
    collected_texts = []

    for i, word in enumerate(words):
        is_last = (i == len(words) - 1)
        is_final = is_final_on_last and is_last
        emission_time = float(i)  # Simple: 1 word per unit time

        result = backend.translate(word, is_final=is_final, emission_time=emission_time)

        trace.actions.append({
            "word_idx": i,
            "action": "READ",
            "source_word": word,
            "emission_time": emission_time,
        })

        if result.text:
            collected_texts.append(result.text)
            # Count new target words
            new_target_words = result.text.strip().split()
            for tw in new_target_words:
                trace.delays.append(float(i + 1))  # source words read so far
                target_word_count += 1

            trace.actions.append({
                "word_idx": i,
                "action": "WRITE",
                "text": result.text,
                "n_tokens": result.committed_tokens,
                "stopped_at_border": result.stopped_at_border,
            })

    # Prefer get_full_translation() for proper token-level decoding,
    # but fall back to collected step texts if the backend cleared state
    # (e.g. _handle_segment_end resets committed_ids on is_final)
    full = backend.get_full_translation()
    trace.translation = full if full else "".join(collected_texts)
    trace.compute_metrics()
    return trace


def replay_from_jsonl(jsonl_path: str) -> SimulationTrace:
    """Replay a trace from OmniSTEval-format JSONL.

    Each line: {"emission_time": float, "text": str, "is_final": bool}
    """
    trace = SimulationTrace(source_words=[])
    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line)
            text = entry.get("text", "").strip()
            etime = entry.get("emission_time", 0.0)

            if text:
                words = text.split()
                for w in words:
                    trace.delays.append(etime)

                trace.actions.append({
                    "action": "WRITE",
                    "text": text,
                    "emission_time": etime,
                })

    trace.compute_metrics()
    return trace
