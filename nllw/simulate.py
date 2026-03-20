"""Policy simulator for simultaneous translation.

Replays a pre-recorded word stream through different AlignAtt parameter
configurations via the web debug API (localhost:8777) and compares the
results.  Useful for tuning border_distance, word_batch, etc.

Usage (requires the debug server running on port 8777):

    python -m nllw.simulate "The president announced new economic reforms"
    python -m nllw.simulate --configs '{"border_distance":2}' '{"border_distance":4}' "hello world"
    python -m nllw.simulate --target-lang zh "The weather is nice today"
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import requests


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    """One step in a simulation: what happened after feeding one word batch."""

    word_idx: int
    source_so_far: str
    stable_text: str
    buffer_text: str
    committed_tokens: int
    time_ms: float


@dataclass
class SimulationTrace:
    """Full trace of replaying a sentence through a policy."""

    source_words: list[str]
    steps: list[StepRecord] = field(default_factory=list)
    full_translation: str = ""
    total_committed_before_finish: int = 0
    finish_text: str = ""
    config: dict = field(default_factory=dict)
    total_time_ms: float = 0.0

    # Populated by compute_average_lagging after the run
    average_lagging: float | None = None


# ---------------------------------------------------------------------------
# Average Lagging metric
# ---------------------------------------------------------------------------


def compute_average_lagging(trace: SimulationTrace, ref_tokens: int | None = None) -> float:
    """Compute Average Lagging (AL) from a simulation trace.

    AL = (1/tau) * sum_{i=1}^{tau} [ g(i) - (i-1) * (|x| / |y|) ]

    where:
      - |x| = number of source words
      - |y| = ref_tokens (reference translation length) or total committed tokens
      - tau  = min(|y|, number of committed token events)
      - g(i) = source word index when the i-th target token was committed

    A lower AL means lower latency.  AL=0 means simultaneous (impossible
    in practice); AL=|x| means wait-until-end.
    """
    n_source = len(trace.source_words)

    # Build g(i): for each committed token, which source word index was active
    # We derive this from the steps: each step after feeding word_idx committed
    # some number of new tokens.
    g = []  # g[i] = source word index when target token i was emitted
    prev_committed = 0
    for step in trace.steps:
        new_committed = step.committed_tokens - prev_committed
        for _ in range(max(0, new_committed)):
            g.append(step.word_idx + 1)  # 1-indexed source position
        prev_committed = step.committed_tokens

    # Include finish tokens (they are produced after all source words are read)
    if trace.finish_text:
        finish_tokens = trace.total_committed_before_finish
        # total committed after finish = all committed - what we already counted
        remaining = len(trace.full_translation.split()) - len(g)
        # Approximate: count characters in finish_text vs full_translation
        # to estimate token count.  Since we don't have exact token counts
        # for finish, use the committed count difference.
        # Actually we can compute it: total tokens after finish minus
        # total before finish.  But we only have committed_tokens from steps.
        # The finish tokens were produced with all source words available.
        pass  # finish tokens get g(i) = n_source (below)

    if not g:
        return float(n_source)  # degenerate: nothing was committed incrementally

    n_target = ref_tokens if ref_tokens is not None else len(g)
    if n_target == 0:
        return 0.0

    tau = min(n_target, len(g))
    ratio = n_source / n_target

    al = 0.0
    for i in range(1, tau + 1):
        al += g[i - 1] - (i - 1) * ratio
    al /= tau

    return al


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class SimulMTSimulator:
    """Replay a word stream through the AlignAtt backend via the web API.

    Parameters
    ----------
    base_url : str
        Base URL for the NLLW debug server (default: http://localhost:8777).
    source_lang : str
        Source language code (default: en).
    target_lang : str
        Target language code (default: fr).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8777",
        source_lang: str = "en",
        target_lang: str = "fr",
    ):
        self.base_url = base_url.rstrip("/")
        self.source_lang = source_lang
        self.target_lang = target_lang

    # -- low-level API helpers -----------------------------------------------

    def _post(self, endpoint: str, **kwargs) -> dict:
        r = requests.post(f"{self.base_url}{endpoint}", **kwargs)
        r.raise_for_status()
        return r.json()

    def _get(self, endpoint: str) -> dict:
        r = requests.get(f"{self.base_url}{endpoint}")
        r.raise_for_status()
        return r.json()

    def _load(self, config: dict) -> dict:
        body = {
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "border_distance": config.get("border_distance", 3),
            "word_batch": config.get("word_batch", 2),
        }
        model_path = config.get("model_path")
        if model_path:
            body["model_path"] = model_path
        return self._post("/load", json=body)

    def _reset(self) -> dict:
        return self._post("/reset")

    def _translate(self, text: str) -> dict:
        return self._post("/translate", json={"text": text})

    def _finish(self) -> dict:
        return self._post("/finish")

    # -- main entry points ---------------------------------------------------

    def replay_sentence(
        self,
        words: list[str],
        config: dict | None = None,
        word_batch: int = 1,
    ) -> SimulationTrace:
        """Feed *words* one by one through the backend, recording each step.

        Parameters
        ----------
        words : list[str]
            Source words to feed incrementally.
        config : dict, optional
            Backend config overrides (border_distance, word_batch for the
            backend, model_path, ...).  Passed to /load.
        word_batch : int
            How many words to feed per /translate call (simulator-side batching,
            separate from the backend's own word_batch parameter).

        Returns
        -------
        SimulationTrace
        """
        config = config or {}

        # Ensure model is loaded with this config
        status = self._get("/status")
        needs_reload = (
            not status.get("loaded")
            or status.get("source_lang") != self.source_lang
            or status.get("target_lang") != self.target_lang
            # Always reload when config specifies backend-level params
            or "border_distance" in config
            or "word_batch" in config
        )
        if needs_reload:
            result = self._load(config)
            if not result.get("ok"):
                raise RuntimeError(f"Failed to load backend: {result}")

        self._reset()

        trace = SimulationTrace(source_words=list(words), config=dict(config))
        t_total_start = time.perf_counter()

        # Feed words in batches
        for batch_start in range(0, len(words), word_batch):
            batch_end = min(batch_start + word_batch, len(words))
            chunk = " ".join(words[batch_start:batch_end])
            # Add trailing space so the backend sees word boundaries
            chunk_text = chunk + " "

            t0 = time.perf_counter()
            result = self._translate(chunk_text)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if "error" in result:
                raise RuntimeError(f"Translation error at word {batch_end}: {result['error']}")

            step = StepRecord(
                word_idx=batch_end - 1,
                source_so_far=" ".join(words[: batch_end]),
                stable_text=result.get("stable", ""),
                buffer_text=result.get("buffer", ""),
                committed_tokens=result.get("committed_tokens", 0),
                time_ms=elapsed_ms,
            )
            trace.steps.append(step)

        # Record committed count before finish
        trace.total_committed_before_finish = (
            trace.steps[-1].committed_tokens if trace.steps else 0
        )

        # Finish: flush remaining translation
        t0 = time.perf_counter()
        fin = self._finish()
        finish_ms = (time.perf_counter() - t0) * 1000

        trace.finish_text = fin.get("remaining", "")
        trace.full_translation = fin.get("full_translation", "")
        trace.total_time_ms = (time.perf_counter() - t_total_start) * 1000

        # Compute AL
        trace.average_lagging = compute_average_lagging(trace)

        return trace

    def compare_policies(
        self,
        words: list[str],
        configs: list[dict],
        word_batch: int = 1,
    ) -> list[dict]:
        """Run the same sentence through multiple configs, return comparison.

        Parameters
        ----------
        words : list[str]
            Source words.
        configs : list[dict]
            Each dict is a config to pass to replay_sentence.
        word_batch : int
            Simulator-side word batch size.

        Returns
        -------
        list[dict]
            One row per config with summary columns.
        """
        rows = []
        for cfg in configs:
            trace = self.replay_sentence(words, config=cfg, word_batch=word_batch)
            # Build incremental output log
            incremental_parts = []
            for step in trace.steps:
                if step.stable_text:
                    incremental_parts.append(step.stable_text)

            row = {
                "config": _config_label(cfg),
                "border_distance": cfg.get("border_distance", 3),
                "word_batch_backend": cfg.get("word_batch", 2),
                "word_batch_sim": word_batch,
                "full_translation": trace.full_translation,
                "committed_before_finish": trace.total_committed_before_finish,
                "finish_text": trace.finish_text,
                "avg_lagging": round(trace.average_lagging or 0.0, 2),
                "total_time_ms": round(trace.total_time_ms, 1),
                "steps": len(trace.steps),
                "incremental_output": "".join(incremental_parts),
            }
            rows.append(row)

        return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config_label(cfg: dict) -> str:
    """Human-readable label for a config dict."""
    parts = []
    for k in sorted(cfg):
        if k == "model_path":
            continue
        parts.append(f"{k}={cfg[k]}")
    return ", ".join(parts) if parts else "(default)"


def _print_trace(trace: SimulationTrace, file=sys.stdout) -> None:
    """Pretty-print a single simulation trace."""
    src = " ".join(trace.source_words)
    print(f"  Source: \"{src}\"", file=file)
    print(f"  Config: {_config_label(trace.config)}", file=file)
    print(f"  {'~' * 56}", file=file)

    committed_so_far = ""
    for step in trace.steps:
        committed_so_far += step.stable_text
        parts = []
        if step.stable_text:
            parts.append(f"stable=\"{step.stable_text}\"")
        if step.buffer_text:
            parts.append(f"buffer=\"{step.buffer_text}\"")
        parts.append(f"tokens={step.committed_tokens}")
        parts.append(f"{step.time_ms:.0f}ms")

        print(f"  [{step.word_idx + 1:2d}] src=\"{step.source_so_far}\"", file=file)
        print(f"       {' | '.join(parts)}", file=file)

    print(f"  {'~' * 56}", file=file)
    print(f"  Finish: \"{trace.finish_text}\"", file=file)
    print(f"  Full:   \"{trace.full_translation}\"", file=file)
    if trace.average_lagging is not None:
        print(f"  AL:     {trace.average_lagging:.2f}", file=file)
    print(f"  Time:   {trace.total_time_ms:.0f}ms", file=file)


def _print_comparison_table(rows: list[dict], source: str, file=sys.stdout) -> None:
    """Print a comparison table to the terminal."""
    print(f"\n{'=' * 80}", file=file)
    print(f"  Source: \"{source}\"", file=file)
    print(f"{'=' * 80}", file=file)

    # Column widths
    cfg_w = max(len(r["config"]) for r in rows)
    cfg_w = max(cfg_w, 10)
    trans_w = max(len(r["full_translation"]) for r in rows)
    trans_w = max(trans_w, 12)

    # Header
    hdr = (
        f"  {'Config':<{cfg_w}}  "
        f"{'Translation':<{trans_w}}  "
        f"{'Commit':>6}  "
        f"{'AL':>5}  "
        f"{'Time':>7}  "
        f"Finish"
    )
    print(hdr, file=file)
    print(f"  {'-' * (len(hdr) - 2)}", file=file)

    for r in rows:
        line = (
            f"  {r['config']:<{cfg_w}}  "
            f"{r['full_translation']:<{trans_w}}  "
            f"{r['committed_before_finish']:>6}  "
            f"{r['avg_lagging']:>5.2f}  "
            f"{r['total_time_ms']:>6.0f}ms  "
            f"\"{r['finish_text']}\""
        )
        print(line, file=file)

    print(file=file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Simulate simultaneous translation with different policies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python -m nllw.simulate "The president announced new economic reforms"
  python -m nllw.simulate --target-lang zh "The weather is nice today"
  python -m nllw.simulate --word-batch-sim 2 "Hello world how are you"
  python -m nllw.simulate --configs '{"border_distance":2}' '{"border_distance":5}' "test sentence"
""",
    )
    parser.add_argument(
        "sentence",
        help="Source sentence to simulate",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8777",
        help="Debug server URL (default: http://localhost:8777)",
    )
    parser.add_argument(
        "--source-lang",
        default="en",
        help="Source language (default: en)",
    )
    parser.add_argument(
        "--target-lang",
        default="fr",
        help="Target language (default: fr)",
    )
    parser.add_argument(
        "--word-batch-sim",
        type=int,
        default=1,
        help="Words per /translate call on the simulator side (default: 1)",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        help=(
            "JSON config dicts to compare. If not given, a default grid of "
            "border_distance x word_batch is used."
        ),
    )
    parser.add_argument(
        "--border-distances",
        type=int,
        nargs="+",
        default=[2, 3, 4],
        help="Border distances to sweep (default: 2 3 4)",
    )
    parser.add_argument(
        "--word-batches",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Backend word_batch values to sweep (default: 1 2 3)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full trace for each config",
    )

    args = parser.parse_args()

    words = args.sentence.strip().split()
    if not words:
        parser.error("Empty sentence")

    sim = SimulMTSimulator(
        base_url=args.base_url,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
    )

    # Build configs
    if args.configs:
        configs = [json.loads(c) for c in args.configs]
    else:
        # Default grid: border_distance x word_batch
        configs = []
        for bd, wb in itertools.product(args.border_distances, args.word_batches):
            configs.append({"border_distance": bd, "word_batch": wb})

    print(f"Simulating {len(configs)} configs for: \"{args.sentence}\"")
    print(f"  source_lang={args.source_lang}, target_lang={args.target_lang}")
    print(f"  word_batch_sim={args.word_batch_sim}")
    print()

    # Run comparison
    rows = []
    for i, cfg in enumerate(configs):
        label = _config_label(cfg)
        print(f"[{i + 1}/{len(configs)}] {label} ...", end="", flush=True)

        trace = sim.replay_sentence(words, config=cfg, word_batch=args.word_batch_sim)

        row = {
            "config": label,
            "border_distance": cfg.get("border_distance", 3),
            "word_batch_backend": cfg.get("word_batch", 2),
            "word_batch_sim": args.word_batch_sim,
            "full_translation": trace.full_translation,
            "committed_before_finish": trace.total_committed_before_finish,
            "finish_text": trace.finish_text,
            "avg_lagging": round(trace.average_lagging or 0.0, 2),
            "total_time_ms": round(trace.total_time_ms, 1),
            "steps": len(trace.steps),
        }
        rows.append(row)

        print(f" done ({trace.total_time_ms:.0f}ms, AL={trace.average_lagging:.2f})")

        if args.verbose:
            _print_trace(trace)
            print()

    # Print comparison table
    _print_comparison_table(rows, args.sentence)

    # Also dump as JSON for programmatic use
    if len(configs) > 1:
        print("JSON output:")
        print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
