"""Experiment registry and runner for the SimulMT Research Forge.

Provides structured experiment tracking, persistence, querying, comparison,
and Pareto frontier analysis for systematic parameter optimization.

Usage (as library):
    from nllw.experiment import ExperimentConfig, ExperimentRegistry, run_experiment

    config = ExperimentConfig(
        backend_type="alignatt", model_path="/path/to/model.gguf",
        border_distance=3, word_batch=3, lang_pair="en-fr",
    )
    result = run_experiment(config, corpus="flores_mini")
    registry = ExperimentRegistry()
    registry.save(result)

Usage (CLI):
    python -m nllw.experiment run config.yaml
    python -m nllw.experiment run config.yaml --corpus flores_mini --web-url http://localhost:8777
    python -m nllw.experiment results --lang en-fr --sort comet
    python -m nllw.experiment results --backend alignatt --best comet
    python -m nllw.experiment pareto --lang en-fr --quality comet --latency ca_al_ms
    python -m nllw.experiment compare --lang en-fr --top 5

Dependencies:
    - pyyaml is optional (YAML config support; falls back to JSON).
    - requests is required for web API mode (run_experiment).
    - sacrebleu is optional (BLEU scoring).
    - unbabel-comet is optional (COMET/xCOMET scoring).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import yaml as _yaml

    _YAML_AVAILABLE = True
except ImportError:
    _yaml = None
    _YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Complete configuration for a single SimulMT experiment.

    Covers backend selection, model parameters, decoding strategy, and
    evaluation corpus/language pair.
    """

    # Backend
    backend_type: str = "alignatt"
    model_path: Optional[str] = None
    heads_path: Optional[str] = None

    # Decoding parameters
    border_distance: int = 3
    word_batch: int = 3
    context_window: int = 0
    entropy_veto_threshold: Optional[float] = None

    # Prompt / model config
    prompt_format: str = "hymt"
    lora_path: Optional[str] = None
    lora_scale: float = 1.0
    top_k: Optional[int] = None
    n_ctx: Optional[int] = None

    # Evaluation scope
    corpus_name: str = "flores_mini"
    lang_pair: str = "en-fr"

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentConfig:
        """Construct from a dict, ignoring unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_yaml(self) -> str:
        """Serialise to YAML string (requires pyyaml)."""
        if not _YAML_AVAILABLE:
            raise ImportError(
                "YAML support requires pyyaml. Install with: pip install pyyaml"
            )
        return _yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> ExperimentConfig:
        """Construct from a YAML string."""
        if not _YAML_AVAILABLE:
            raise ImportError(
                "YAML support requires pyyaml. Install with: pip install pyyaml"
            )
        return cls.from_dict(_yaml.safe_load(yaml_str))

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> ExperimentConfig:
        """Load from a YAML file."""
        with open(path) as f:
            return cls.from_yaml(f.read())

    @property
    def config_hash(self) -> str:
        """Short SHA-256 hex digest (8 chars) for de-duplication.

        The hash covers all fields that affect experimental outcome
        (excludes transient metadata like corpus_name).
        """
        # Include all parameters that affect translation output
        key_fields = {
            "backend_type": self.backend_type,
            "model_path": self.model_path,
            "heads_path": self.heads_path,
            "border_distance": self.border_distance,
            "word_batch": self.word_batch,
            "context_window": self.context_window,
            "entropy_veto_threshold": self.entropy_veto_threshold,
            "prompt_format": self.prompt_format,
            "lora_path": self.lora_path,
            "lora_scale": self.lora_scale,
            "top_k": self.top_k,
            "n_ctx": self.n_ctx,
            "lang_pair": self.lang_pair,
            "corpus_name": self.corpus_name,
        }
        blob = json.dumps(key_fields, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:8]

    @property
    def source_lang(self) -> str:
        return self.lang_pair.split("-")[0]

    @property
    def target_lang(self) -> str:
        parts = self.lang_pair.split("-", 1)
        return parts[1] if len(parts) > 1 else parts[0]

    @property
    def short_label(self) -> str:
        """Human-readable short label for tables."""
        parts = [self.backend_type]
        if self.model_path:
            parts.append(Path(self.model_path).stem[:15])
        parts.append(f"bd={self.border_distance}")
        parts.append(f"wb={self.word_batch}")
        if self.context_window:
            parts.append(f"ctx={self.context_window}")
        if self.entropy_veto_threshold is not None:
            parts.append(f"ev={self.entropy_veto_threshold}")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# ExperimentResult
# ---------------------------------------------------------------------------


@dataclass
class ExperimentResult:
    """Complete result of a single experiment run."""

    config: ExperimentConfig

    # Quality metrics (all optional — depend on available scorers)
    bleu: Optional[float] = None
    comet: Optional[float] = None
    xcomet_xl: Optional[float] = None

    # Latency / behaviour metrics
    committed_ratio: Optional[float] = None
    finish_ratio: Optional[float] = None
    ca_al_ms: Optional[float] = None
    al_words: Optional[float] = None
    time_per_word_ms: Optional[float] = None
    total_time_ms: Optional[float] = None

    # Per-sentence detail (list of dicts with per-sentence scores)
    per_sentence: list[dict] = field(default_factory=list)

    # Metadata
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    hardware_tag: str = field(default_factory=lambda: _detect_hardware_tag())
    notes: str = ""

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        d = asdict(self)
        # config is already a dict via asdict, but let's ensure clean nesting
        d["config"] = self.config.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentResult:
        """Construct from a dict (e.g. loaded from JSON)."""
        config_data = d.pop("config", {})
        config = ExperimentConfig.from_dict(config_data)
        known = {f.name for f in cls.__dataclass_fields__.values()} - {"config"}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(config=config, **filtered)


# ---------------------------------------------------------------------------
# ExperimentRegistry
# ---------------------------------------------------------------------------


class ExperimentRegistry:
    """Persistent store for experiment results.

    Results are saved as individual JSON files in *registry_dir*, named
    ``{timestamp}_{config_hash}.json``.  This makes it easy to browse,
    diff, and version-control experiment history.

    Parameters
    ----------
    registry_dir : str or Path
        Directory for storing experiment result files.
        Created automatically if it does not exist.
    """

    def __init__(self, registry_dir: str | Path = "experiments/") -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, result: ExperimentResult) -> Path:
        """Persist an experiment result to disk.

        Returns the path to the saved JSON file.
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{ts}_{result.config.config_hash}.json"
        path = self.registry_dir / filename

        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        return path

    def load_all(self) -> list[ExperimentResult]:
        """Load every experiment result from the registry directory."""
        results: list[ExperimentResult] = []
        for p in sorted(self.registry_dir.glob("*.json")):
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                results.append(ExperimentResult.from_dict(data))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                import sys
                print(
                    f"Warning: skipping {p.name}: {exc}",
                    file=sys.stderr,
                )
        return results

    # ------------------------------------------------------------------
    # query / filter
    # ------------------------------------------------------------------

    def query(
        self,
        *,
        backend_type: Optional[str] = None,
        lang_pair: Optional[str] = None,
        corpus_name: Optional[str] = None,
        model_path_contains: Optional[str] = None,
        min_bleu: Optional[float] = None,
        min_comet: Optional[float] = None,
    ) -> list[ExperimentResult]:
        """Filter loaded results by various criteria.

        All filters are AND-combined.  ``None`` means "don't filter on this".
        """
        results = self.load_all()
        filtered: list[ExperimentResult] = []
        for r in results:
            if backend_type and r.config.backend_type != backend_type:
                continue
            if lang_pair and r.config.lang_pair != lang_pair:
                continue
            if corpus_name and r.config.corpus_name != corpus_name:
                continue
            if model_path_contains and (
                not r.config.model_path
                or model_path_contains not in r.config.model_path
            ):
                continue
            if min_bleu is not None and (r.bleu is None or r.bleu < min_bleu):
                continue
            if min_comet is not None and (r.comet is None or r.comet < min_comet):
                continue
            filtered.append(r)
        return filtered

    # ------------------------------------------------------------------
    # compare
    # ------------------------------------------------------------------

    @staticmethod
    def compare(results: list[ExperimentResult]) -> dict[str, Any]:
        """Build a comparison table from multiple results.

        Returns a dict with:
            - ``columns``: list of column names
            - ``rows``: list of row dicts (one per result)
            - ``best``: dict mapping metric -> index of best result
        """
        if not results:
            return {"columns": [], "rows": [], "best": {}}

        columns = [
            "label", "backend_type", "lang_pair",
            "bd", "wb", "ctx",
            "bleu", "comet", "xcomet_xl",
            "committed_ratio", "ca_al_ms", "al_words",
            "time_per_word_ms", "config_hash",
        ]

        rows: list[dict[str, Any]] = []
        for r in results:
            rows.append({
                "label": r.config.short_label,
                "backend_type": r.config.backend_type,
                "lang_pair": r.config.lang_pair,
                "bd": r.config.border_distance,
                "wb": r.config.word_batch,
                "ctx": r.config.context_window,
                "bleu": r.bleu,
                "comet": r.comet,
                "xcomet_xl": r.xcomet_xl,
                "committed_ratio": r.committed_ratio,
                "ca_al_ms": r.ca_al_ms,
                "al_words": r.al_words,
                "time_per_word_ms": r.time_per_word_ms,
                "config_hash": r.config.config_hash,
            })

        # Identify best per metric (higher is better for quality, lower for latency)
        best: dict[str, int] = {}
        _higher_better = ["bleu", "comet", "xcomet_xl", "committed_ratio"]
        _lower_better = ["ca_al_ms", "al_words", "time_per_word_ms"]

        for metric in _higher_better:
            vals = [
                (i, row[metric])
                for i, row in enumerate(rows)
                if row[metric] is not None
            ]
            if vals:
                best[metric] = max(vals, key=lambda x: x[1])[0]

        for metric in _lower_better:
            vals = [
                (i, row[metric])
                for i, row in enumerate(rows)
                if row[metric] is not None
            ]
            if vals:
                best[metric] = min(vals, key=lambda x: x[1])[0]

        return {"columns": columns, "rows": rows, "best": best}

    # ------------------------------------------------------------------
    # best_config
    # ------------------------------------------------------------------

    def best_config(
        self,
        lang_pair: str,
        metric: str = "comet",
    ) -> Optional[ExperimentConfig]:
        """Return the config that achieved the best *metric* for *lang_pair*.

        Parameters
        ----------
        lang_pair : str
            e.g. "en-fr"
        metric : str
            One of "bleu", "comet", "xcomet_xl" (higher-is-better) or
            "ca_al_ms", "al_words", "time_per_word_ms" (lower-is-better).

        Returns None if no results match.
        """
        results = self.query(lang_pair=lang_pair)
        if not results:
            return None

        lower_better = {"ca_al_ms", "al_words", "time_per_word_ms"}
        key_fn = lambda r: getattr(r, metric, None)

        scored = [(r, key_fn(r)) for r in results if key_fn(r) is not None]
        if not scored:
            return None

        if metric in lower_better:
            best_result = min(scored, key=lambda x: x[1])[0]
        else:
            best_result = max(scored, key=lambda x: x[1])[0]

        return best_result.config

    # ------------------------------------------------------------------
    # pareto_frontier
    # ------------------------------------------------------------------

    @staticmethod
    def pareto_frontier(
        results: list[ExperimentResult],
        quality: str = "comet",
        latency: str = "ca_al_ms",
    ) -> list[ExperimentResult]:
        """Compute the Pareto frontier (quality vs latency).

        A result is Pareto-optimal if no other result is strictly better
        on *both* quality (higher is better) and latency (lower is better).

        Parameters
        ----------
        results : list[ExperimentResult]
            Pool of results to filter.
        quality : str
            Quality metric attribute name (higher is better).
        latency : str
            Latency metric attribute name (lower is better).

        Returns
        -------
        list[ExperimentResult]
            Pareto-optimal results, sorted by ascending quality.
        """
        # Filter to results that have both metrics
        candidates = [
            r for r in results
            if getattr(r, quality, None) is not None
            and getattr(r, latency, None) is not None
        ]
        if not candidates:
            return []

        # Sort by quality ascending for the sweep
        candidates.sort(key=lambda r: getattr(r, quality))

        frontier: list[ExperimentResult] = []
        best_latency = float("inf")

        # Walk from highest quality down — a point is Pareto-optimal
        # if it has lower latency than all higher-quality points seen so far
        for r in reversed(candidates):
            lat = getattr(r, latency)
            if lat < best_latency:
                frontier.append(r)
                best_latency = lat

        # Return sorted by ascending quality
        frontier.sort(key=lambda r: getattr(r, quality))
        return frontier


# ---------------------------------------------------------------------------
# run_experiment — the main experiment runner
# ---------------------------------------------------------------------------


def run_experiment(
    config: ExperimentConfig,
    corpus: str | list[dict] | None = None,
    *,
    web_url: str = "http://localhost:8777",
    compute_comet: bool = False,
    compute_xcomet: bool = False,
    speech_rate_wps: float = 2.5,
    verbose: bool = False,
) -> ExperimentResult:
    """Run a complete experiment: load backend, evaluate corpus, collect metrics.

    The experiment is run via the web debug server API.  The server must be
    running at *web_url* with a model already loaded (or loadable via /load).

    Parameters
    ----------
    config : ExperimentConfig
        Full experiment configuration.
    corpus : str, list[dict], or None
        Either a built-in corpus name ("flores_mini", "default"), a list of
        test-case dicts (same format as ``eval.DEFAULT_TEST_CASES``), or
        None (uses config.corpus_name).
    web_url : str
        URL of the NLLW web debug server.
    compute_comet : bool
        Compute COMET (wmt22-comet-da) scores after evaluation.
    compute_xcomet : bool
        Compute xCOMET-XL scores after evaluation.
    speech_rate_wps : float
        Simulated speech rate for CA-AL computation.
    verbose : bool
        Print progress information.

    Returns
    -------
    ExperimentResult
        Complete result including all metrics and per-sentence detail.
    """
    # --- Resolve corpus ---
    test_cases = _resolve_corpus(corpus, config)

    # Filter by lang_pair
    src_lang = config.source_lang
    tgt_lang = config.target_lang
    filtered_cases = [
        tc for tc in test_cases
        if tc.get("source_lang", src_lang) == src_lang
        and tc.get("target_lang", tgt_lang) == tgt_lang
    ]
    if not filtered_cases:
        # If no matching cases, use all (the user may have a custom corpus)
        filtered_cases = test_cases

    # --- Load backend via web API ---
    from nllw.research import ResearchBenchmark

    rb = ResearchBenchmark(web_url)

    load_params: dict[str, Any] = {
        "model_path": config.model_path,
        "source_lang": src_lang,
        "target_lang": tgt_lang,
        "border_distance": config.border_distance,
        "word_batch": config.word_batch,
    }
    if verbose:
        print(f"Loading backend: {config.backend_type} ({config.short_label})")

    load_resp = rb.api.load(**load_params)
    if not load_resp.get("ok", True):
        raise RuntimeError(
            f"Failed to load backend via web API: {load_resp.get('error', load_resp)}"
        )

    # --- Also configure extended params via /load if supported ---
    # The server LoadRequest supports backend_type, prompt_format, entropy_veto_threshold, etc.
    # We use a raw POST to pass all config fields
    try:
        import requests as _requests

        full_load_body = {
            "model_path": config.model_path,
            "source_lang": src_lang,
            "target_lang": tgt_lang,
            "backend_type": config.backend_type,
            "prompt_format": config.prompt_format,
            "border_distance": config.border_distance,
            "word_batch": config.word_batch,
            "lora_path": config.lora_path,
            "lora_scale": config.lora_scale,
        }
        if config.heads_path:
            full_load_body["heads_path"] = config.heads_path
        if config.entropy_veto_threshold is not None:
            full_load_body["entropy_veto_threshold"] = config.entropy_veto_threshold
        # Remove None values for cleaner request
        full_load_body = {k: v for k, v in full_load_body.items() if v is not None}

        resp = _requests.post(f"{web_url.rstrip('/')}/load", json=full_load_body, timeout=120)
        load_result = resp.json()
        if not load_result.get("ok", True):
            raise RuntimeError(
                f"Full load failed: {load_result.get('error', load_result)}"
            )
    except ImportError:
        pass  # Already loaded via ResearchBenchmark above

    # --- Run benchmark ---
    cfg_dict = {
        "border_distance": config.border_distance,
        "word_batch": config.word_batch,
        "label": config.short_label,
    }
    if config.model_path:
        cfg_dict["model_path"] = config.model_path

    experiment_start = time.perf_counter()

    suite_result = rb.run_benchmark_suite(
        filtered_cases,
        [cfg_dict],
        speech_rate_wps=speech_rate_wps,
        verbose=verbose,
    )

    total_experiment_ms = (time.perf_counter() - experiment_start) * 1000.0

    # --- Extract metrics from suite result ---
    cfg_results = suite_result.get("configs", [])
    if not cfg_results:
        return ExperimentResult(
            config=config,
            notes="No results produced (empty config results)",
        )

    cfg_r = cfg_results[0]
    sentences = cfg_r.get("sentences", [])

    bleu = cfg_r.get("bleu")
    avg_committed_ratio = cfg_r.get("avg_committed_ratio", 0.0)
    avg_ca_al_ms = cfg_r.get("avg_ca_al_ms", 0.0)
    avg_al_words = cfg_r.get("avg_al_words", 0.0)
    avg_step_ms = cfg_r.get("avg_step_ms", 0.0)

    # Compute finish_ratio from per-sentence data
    finish_ratio = None
    if avg_committed_ratio is not None:
        finish_ratio = round(1.0 - avg_committed_ratio, 4)

    # Build per-sentence records
    per_sentence: list[dict] = []
    sources: list[str] = []
    hypotheses: list[str] = []
    references: list[str] = []

    for s in sentences:
        per_sentence.append({
            "tag": s.get("tag", ""),
            "source": s.get("source", ""),
            "reference": s.get("reference", ""),
            "hypothesis": s.get("hypothesis", ""),
            "committed_ratio": s.get("committed_ratio"),
            "ca_al_ms": s.get("ca_al_ms"),
            "al_words": s.get("standard_al_words"),
            "total_time_ms": s.get("total_time_ms"),
        })
        sources.append(s.get("source", ""))
        hypotheses.append(s.get("hypothesis", ""))
        if s.get("reference"):
            references.append(s["reference"])

    # --- Optional COMET / xCOMET scoring ---
    comet_score: Optional[float] = None
    xcomet_score: Optional[float] = None

    if (compute_comet or compute_xcomet) and sources and hypotheses and references:
        if len(references) == len(hypotheses):
            try:
                from nllw.metrics import (
                    compute_comet as _compute_comet,
                    compute_xcomet as _compute_xcomet,
                    comet_available,
                )

                if comet_available():
                    if compute_comet:
                        try:
                            result = _compute_comet(sources, hypotheses, references)
                            comet_score = result.get("score")
                            # Attach per-sentence COMET scores
                            comet_scores = result.get("scores", [])
                            for i, sc in enumerate(comet_scores):
                                if i < len(per_sentence):
                                    per_sentence[i]["comet"] = sc
                        except Exception as e:
                            if verbose:
                                print(f"  Warning: COMET failed: {e}")

                    if compute_xcomet:
                        try:
                            result = _compute_xcomet(sources, hypotheses, references)
                            xcomet_score = result.get("score")
                            xcomet_scores = result.get("scores", [])
                            for i, sc in enumerate(xcomet_scores):
                                if i < len(per_sentence):
                                    per_sentence[i]["xcomet_xl"] = sc
                        except Exception as e:
                            if verbose:
                                print(f"  Warning: xCOMET failed: {e}")
            except ImportError:
                if verbose:
                    print("  Warning: nllw.metrics not available for COMET scoring")

    return ExperimentResult(
        config=config,
        bleu=bleu,
        comet=comet_score,
        xcomet_xl=xcomet_score,
        committed_ratio=round(avg_committed_ratio, 4) if avg_committed_ratio else None,
        finish_ratio=finish_ratio,
        ca_al_ms=round(avg_ca_al_ms, 1) if avg_ca_al_ms else None,
        al_words=round(avg_al_words, 3) if avg_al_words else None,
        time_per_word_ms=round(avg_step_ms, 1) if avg_step_ms else None,
        total_time_ms=round(total_experiment_ms, 1),
        per_sentence=per_sentence,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_corpus(
    corpus: str | list[dict] | None,
    config: ExperimentConfig,
) -> list[dict]:
    """Resolve a corpus spec to a list of test-case dicts."""
    if isinstance(corpus, list):
        return corpus

    name = corpus or config.corpus_name

    if name == "flores_mini":
        from nllw.research import FLORES_MINI
        return list(FLORES_MINI)
    elif name == "default":
        from nllw.eval import DEFAULT_TEST_CASES
        return list(DEFAULT_TEST_CASES)
    elif name == "corpus":
        from nllw.corpus import TestCorpus
        return list(TestCorpus.FULL_CORPUS)
    elif name and os.path.isfile(name):
        # Load from JSON file
        with open(name, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "sentences" in data:
            return data["sentences"]
        else:
            raise ValueError(
                f"Corpus file {name} must be a JSON list or dict with 'sentences' key"
            )
    else:
        # Fallback to flores_mini
        from nllw.research import FLORES_MINI
        return list(FLORES_MINI)


def _detect_hardware_tag() -> str:
    """Generate a short tag describing the current hardware."""
    parts = [platform.system(), platform.machine()]
    try:
        cpu_count = os.cpu_count()
        if cpu_count:
            parts.append(f"{cpu_count}cpu")
    except Exception:
        pass
    return "-".join(parts)


# ---------------------------------------------------------------------------
# ASCII output helpers
# ---------------------------------------------------------------------------


def _print_results_table(results: list[ExperimentResult], sort_by: str = "comet") -> None:
    """Print a formatted comparison table of experiment results."""
    if not results:
        print("  No results found.")
        return

    # Sort
    lower_better = {"ca_al_ms", "al_words", "time_per_word_ms"}
    reverse = sort_by not in lower_better

    def sort_key(r: ExperimentResult) -> float:
        val = getattr(r, sort_by, None)
        if val is None:
            return float("-inf") if reverse else float("inf")
        return val

    results = sorted(results, key=sort_key, reverse=reverse)

    print()
    header = (
        f"{'#':>3}  {'Config':<30} | {'BLEU':>6} | {'COMET':>7} | "
        f"{'xCOMET':>7} | {'Commit%':>8} | {'CA-AL':>7} | "
        f"{'AL(w)':>7} | {'ms/word':>7} | {'Hash':>8}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for i, r in enumerate(results, 1):
        label = r.config.short_label[:30]
        bleu_s = f"{r.bleu:>6.1f}" if r.bleu is not None else "   N/A"
        comet_s = f"{r.comet:>7.4f}" if r.comet is not None else "    N/A"
        xcomet_s = f"{r.xcomet_xl:>7.4f}" if r.xcomet_xl is not None else "    N/A"
        commit_s = f"{r.committed_ratio:>7.1%}" if r.committed_ratio is not None else "    N/A"
        ca_s = f"{r.ca_al_ms:>5.0f}ms" if r.ca_al_ms is not None else "    N/A"
        al_s = f"{r.al_words:>7.2f}" if r.al_words is not None else "    N/A"
        tpw_s = f"{r.time_per_word_ms:>5.0f}ms" if r.time_per_word_ms is not None else "    N/A"

        print(
            f"{i:>3}  {label:<30} | {bleu_s} | {comet_s} | "
            f"{xcomet_s} | {commit_s} | {ca_s} | "
            f"{al_s} | {tpw_s} | {r.config.config_hash}"
        )

    print()


def _print_pareto_table(
    frontier: list[ExperimentResult],
    quality: str,
    latency: str,
) -> None:
    """Print the Pareto frontier as a table."""
    if not frontier:
        print("  No Pareto-optimal results found.")
        return

    print()
    print(f"  Pareto frontier: {quality} (higher=better) vs {latency} (lower=better)")
    print()

    header = (
        f"{'#':>3}  {'Config':<30} | {quality:>10} | {latency:>10} | {'Hash':>8}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for i, r in enumerate(frontier, 1):
        label = r.config.short_label[:30]
        q_val = getattr(r, quality, None)
        l_val = getattr(r, latency, None)
        q_s = f"{q_val:>10.4f}" if q_val is not None else "       N/A"
        l_s = f"{l_val:>10.1f}" if l_val is not None else "       N/A"
        print(f"{i:>3}  {label:<30} | {q_s} | {l_s} | {r.config.config_hash}")

    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m nllw.experiment",
        description="Experiment registry and runner for SimulMT Research Forge.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m nllw.experiment run config.yaml\n"
            "  python -m nllw.experiment run config.yaml --corpus flores_mini --comet\n"
            "  python -m nllw.experiment results --lang en-fr --sort comet\n"
            "  python -m nllw.experiment results --backend alignatt --best comet\n"
            "  python -m nllw.experiment pareto --lang en-fr\n"
            "  python -m nllw.experiment compare --lang en-fr --top 10\n"
        ),
    )

    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # --- run ---
    run_parser = subparsers.add_parser(
        "run", help="Run an experiment from a YAML/JSON config file"
    )
    run_parser.add_argument(
        "config_file",
        help="Path to experiment config (YAML or JSON)",
    )
    run_parser.add_argument(
        "--corpus", default=None,
        help="Corpus name or path (overrides config file)",
    )
    run_parser.add_argument(
        "--web-url", default="http://localhost:8777",
        help="Web debug server URL (default: http://localhost:8777)",
    )
    run_parser.add_argument(
        "--registry-dir", default="experiments/",
        help="Registry directory (default: experiments/)",
    )
    run_parser.add_argument(
        "--comet", action="store_true",
        help="Compute COMET scores",
    )
    run_parser.add_argument(
        "--xcomet", action="store_true",
        help="Compute xCOMET-XL scores",
    )
    run_parser.add_argument(
        "--speech-rate", type=float, default=2.5,
        help="Simulated speech rate (default: 2.5 wps)",
    )
    run_parser.add_argument(
        "--no-save", action="store_true",
        help="Don't save results to registry",
    )
    run_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output",
    )

    # --- results ---
    results_parser = subparsers.add_parser(
        "results", help="Query and display experiment results"
    )
    results_parser.add_argument(
        "--registry-dir", default="experiments/",
        help="Registry directory (default: experiments/)",
    )
    results_parser.add_argument(
        "--lang", default=None,
        help="Filter by language pair (e.g. en-fr)",
    )
    results_parser.add_argument(
        "--backend", default=None,
        help="Filter by backend type (e.g. alignatt)",
    )
    results_parser.add_argument(
        "--corpus", default=None,
        help="Filter by corpus name",
    )
    results_parser.add_argument(
        "--sort", default="comet",
        help="Sort metric (default: comet). Options: bleu, comet, xcomet_xl, "
             "ca_al_ms, al_words, time_per_word_ms, committed_ratio",
    )
    results_parser.add_argument(
        "--best", default=None, metavar="METRIC",
        help="Show only the best config for the given metric",
    )
    results_parser.add_argument(
        "--top", type=int, default=None,
        help="Show only top N results",
    )
    results_parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON",
    )

    # --- pareto ---
    pareto_parser = subparsers.add_parser(
        "pareto", help="Compute and display Pareto frontier"
    )
    pareto_parser.add_argument(
        "--registry-dir", default="experiments/",
        help="Registry directory (default: experiments/)",
    )
    pareto_parser.add_argument(
        "--lang", default=None,
        help="Filter by language pair",
    )
    pareto_parser.add_argument(
        "--backend", default=None,
        help="Filter by backend type",
    )
    pareto_parser.add_argument(
        "--quality", default="comet",
        help="Quality metric (higher=better, default: comet)",
    )
    pareto_parser.add_argument(
        "--latency", default="ca_al_ms",
        help="Latency metric (lower=better, default: ca_al_ms)",
    )
    pareto_parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON",
    )

    # --- compare ---
    compare_parser = subparsers.add_parser(
        "compare", help="Compare top experiments side-by-side"
    )
    compare_parser.add_argument(
        "--registry-dir", default="experiments/",
        help="Registry directory (default: experiments/)",
    )
    compare_parser.add_argument(
        "--lang", default=None,
        help="Filter by language pair",
    )
    compare_parser.add_argument(
        "--backend", default=None,
        help="Filter by backend type",
    )
    compare_parser.add_argument(
        "--top", type=int, default=10,
        help="Number of top results to compare (default: 10)",
    )
    compare_parser.add_argument(
        "--sort", default="comet",
        help="Sort metric (default: comet)",
    )
    compare_parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # =================================================================
    # run
    # =================================================================
    if args.command == "run":
        config_path = Path(args.config_file)
        if not config_path.exists():
            print(f"Error: config file not found: {config_path}", file=sys.stderr)
            sys.exit(1)

        # Load config (YAML or JSON)
        with open(config_path, encoding="utf-8") as f:
            raw = f.read()

        if config_path.suffix in (".yaml", ".yml"):
            config = ExperimentConfig.from_yaml(raw)
        else:
            config = ExperimentConfig.from_dict(json.loads(raw))

        # Override corpus if specified on CLI
        if args.corpus:
            config.corpus_name = args.corpus

        print(f"SimulMT Research Forge - Experiment Runner")
        print(f"  Config:   {config_path}")
        print(f"  Backend:  {config.backend_type}")
        print(f"  Model:    {config.model_path or '(server default)'}")
        print(f"  Lang:     {config.lang_pair}")
        print(f"  BD={config.border_distance}  WB={config.word_batch}  "
              f"CTX={config.context_window}")
        print(f"  Corpus:   {config.corpus_name}")
        print(f"  Server:   {args.web_url}")
        print()

        result = run_experiment(
            config,
            corpus=args.corpus,
            web_url=args.web_url,
            compute_comet=args.comet,
            compute_xcomet=args.xcomet,
            speech_rate_wps=args.speech_rate,
            verbose=args.verbose,
        )

        # Print summary
        print(f"\n--- Experiment Result ---")
        print(f"  BLEU:           {result.bleu}")
        if result.comet is not None:
            print(f"  COMET:          {result.comet:.4f}")
        if result.xcomet_xl is not None:
            print(f"  xCOMET-XL:      {result.xcomet_xl:.4f}")
        print(f"  Committed:      {result.committed_ratio:.1%}" if result.committed_ratio else "  Committed:      N/A")
        print(f"  CA-AL:          {result.ca_al_ms:.0f} ms" if result.ca_al_ms else "  CA-AL:          N/A")
        print(f"  AL (words):     {result.al_words:.2f}" if result.al_words else "  AL (words):     N/A")
        print(f"  Time/word:      {result.time_per_word_ms:.0f} ms" if result.time_per_word_ms else "  Time/word:      N/A")
        print(f"  Total time:     {result.total_time_ms:.0f} ms" if result.total_time_ms else "  Total time:     N/A")
        print(f"  Sentences:      {len(result.per_sentence)}")
        print(f"  Config hash:    {config.config_hash}")
        print(f"  Hardware:       {result.hardware_tag}")

        # Save to registry
        if not args.no_save:
            registry = ExperimentRegistry(args.registry_dir)
            path = registry.save(result)
            print(f"\n  Saved to: {path}")

    # =================================================================
    # results
    # =================================================================
    elif args.command == "results":
        registry = ExperimentRegistry(args.registry_dir)
        results = registry.query(
            backend_type=args.backend,
            lang_pair=args.lang,
            corpus_name=args.corpus,
        )

        if not results:
            print("No results found matching filters.")
            sys.exit(0)

        print(f"Found {len(results)} experiment result(s)")

        # --best: show only the single best config
        if args.best:
            best_cfg = registry.best_config(
                lang_pair=args.lang or "en-fr",
                metric=args.best,
            )
            if best_cfg:
                print(f"\nBest config for {args.best} ({args.lang or 'en-fr'}):")
                print(f"  {best_cfg.short_label}")
                print(f"  Hash: {best_cfg.config_hash}")
                if _YAML_AVAILABLE:
                    print(f"\nConfig YAML:")
                    print(best_cfg.to_yaml())
                else:
                    print(f"\nConfig JSON:")
                    print(json.dumps(best_cfg.to_dict(), indent=2))
            else:
                print(f"No results with metric '{args.best}'")
            sys.exit(0)

        if args.json:
            output = [r.to_dict() for r in results]
            print(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            if args.top:
                # Sort first, then slice
                lower_better = {"ca_al_ms", "al_words", "time_per_word_ms"}
                reverse = args.sort not in lower_better

                def sort_key(r: ExperimentResult) -> float:
                    val = getattr(r, args.sort, None)
                    if val is None:
                        return float("-inf") if reverse else float("inf")
                    return val

                results = sorted(results, key=sort_key, reverse=reverse)[:args.top]

            _print_results_table(results, sort_by=args.sort)

    # =================================================================
    # pareto
    # =================================================================
    elif args.command == "pareto":
        registry = ExperimentRegistry(args.registry_dir)
        results = registry.query(
            backend_type=args.backend,
            lang_pair=args.lang,
        )

        if not results:
            print("No results found matching filters.")
            sys.exit(0)

        frontier = ExperimentRegistry.pareto_frontier(
            results,
            quality=args.quality,
            latency=args.latency,
        )

        if args.json:
            output = [r.to_dict() for r in frontier]
            print(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            print(f"Found {len(results)} total results, "
                  f"{len(frontier)} Pareto-optimal")
            _print_pareto_table(frontier, args.quality, args.latency)

    # =================================================================
    # compare
    # =================================================================
    elif args.command == "compare":
        registry = ExperimentRegistry(args.registry_dir)
        results = registry.query(
            backend_type=args.backend,
            lang_pair=args.lang,
        )

        if not results:
            print("No results found matching filters.")
            sys.exit(0)

        # Sort and take top N
        lower_better = {"ca_al_ms", "al_words", "time_per_word_ms"}
        reverse = args.sort not in lower_better

        def sort_key(r: ExperimentResult) -> float:
            val = getattr(r, args.sort, None)
            if val is None:
                return float("-inf") if reverse else float("inf")
            return val

        results = sorted(results, key=sort_key, reverse=reverse)[:args.top]
        comparison = ExperimentRegistry.compare(results)

        if args.json:
            print(json.dumps(comparison, indent=2, ensure_ascii=False))
        else:
            print(f"Comparing top {len(results)} experiments "
                  f"(sorted by {args.sort})")
            _print_results_table(results, sort_by=args.sort)

            # Show best per metric
            if comparison["best"]:
                print("  Best per metric:")
                for metric, idx in comparison["best"].items():
                    row = comparison["rows"][idx]
                    val = row[metric]
                    val_s = f"{val:.4f}" if isinstance(val, float) else str(val)
                    print(f"    {metric:<20} = {val_s:<12} ({row['label']})")
                print()


if __name__ == "__main__":
    main()
