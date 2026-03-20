"""Experiment configuration and result registry.

Manages reproducible experiment definitions (YAML configs) and their results.
Provides tools for:
    - Defining experiments as YAML config files
    - Running experiments from configs
    - Storing and retrieving results
    - Comparing experiment outcomes

Config format (YAML):
    name: "alignatt-en-zh-bd3-wb2"
    backend:
        type: alignatt
        model: /path/to/model.gguf
        border_distance: 3
        word_batch: 2
    eval:
        direction: en-zh
        n_sentences: 100
        comet: true
        xcomet: false
    notes: "Testing bd=3 wb=2 on EN-ZH with HY-MT1.5-7B"

Usage:
    python -m nllw.experiment run config.yaml
    python -m nllw.experiment list
    python -m nllw.experiment compare config1.yaml config2.yaml
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    name: str
    backend_type: str = "alignatt"
    model_path: str = ""
    heads_path: str = ""
    direction: str = "en-zh"
    # Backend parameters
    border_distance: int = 3
    word_batch: int = 3
    context_sentences: int = 0
    top_k_heads: int = 10
    entropy_veto_threshold: Optional[float] = None
    n_ctx: int = 2048
    # Eval parameters
    n_sentences: int = 50
    corpus_suite: str = "flores"
    compute_comet: bool = False
    compute_xcomet: bool = False
    # Metadata
    notes: str = ""
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        """Create from a flat or nested dict."""
        # Handle nested YAML format
        flat = {}
        if "backend" in d:
            flat.update(d["backend"])
            if "type" in flat:
                flat["backend_type"] = flat.pop("type")
            if "model" in flat:
                flat["model_path"] = flat.pop("model")
            if "heads" in flat:
                flat["heads_path"] = flat.pop("heads")
        if "eval" in d:
            eval_cfg = d["eval"]
            if "direction" in eval_cfg:
                flat["direction"] = eval_cfg["direction"]
            if "n_sentences" in eval_cfg:
                flat["n_sentences"] = eval_cfg["n_sentences"]
            if "comet" in eval_cfg:
                flat["compute_comet"] = eval_cfg["comet"]
            if "xcomet" in eval_cfg:
                flat["compute_xcomet"] = eval_cfg["xcomet"]
            if "suite" in eval_cfg:
                flat["corpus_suite"] = eval_cfg["suite"]

        # Top-level keys
        for key in ["name", "notes", "tags"]:
            if key in d:
                flat[key] = d[key]

        # Merge remaining top-level keys (for flat format)
        for key in d:
            if key not in ("backend", "eval", "name", "notes", "tags"):
                flat.setdefault(key, d[key])

        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in flat.items() if k in valid_keys})

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load from a YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required for YAML configs. Install: pip install pyyaml")
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_backend_config(self) -> dict:
        """Convert to BackendConfig-compatible dict."""
        return {
            "backend_type": self.backend_type,
            "model_path": self.model_path,
            "heads_path": self.heads_path,
            "direction": self.direction,
            "border_distance": self.border_distance,
            "word_batch": self.word_batch,
            "context_sentences": self.context_sentences,
            "top_k_heads": self.top_k_heads,
            "entropy_veto_threshold": self.entropy_veto_threshold,
            "n_ctx": self.n_ctx,
            "target_lang": self.direction.split("-")[1],
        }


@dataclass
class ExperimentRecord:
    """A stored experiment result."""
    config: ExperimentConfig
    # Results
    bleu: float = 0.0
    comet: Optional[float] = None
    xcomet: Optional[float] = None
    avg_yaal: float = 0.0
    avg_al: float = 0.0
    avg_ap: float = 0.0
    avg_dal: float = 0.0
    avg_max_cw: float = 0.0
    n_sentences: int = 0
    total_time_s: float = 0.0
    # Meta
    timestamp: str = ""
    machine: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def summary_line(self) -> str:
        """One-line summary for tables."""
        comet_str = f"{self.comet:.3f}" if self.comet else "  -  "
        xcomet_str = f"{self.xcomet:.3f}" if self.xcomet else "  -  "
        return (
            f"{self.config.name:<30} | {self.config.direction:<6} | "
            f"BLEU={self.bleu:5.1f} | COMET={comet_str} | XCOMET={xcomet_str} | "
            f"YAAL={self.avg_yaal:5.2f} | AL={self.avg_al:5.2f} | "
            f"{self.total_time_s:6.1f}s"
        )


# ---------------------------------------------------------------------------
# Registry (file-based)
# ---------------------------------------------------------------------------

class ExperimentRegistry:
    """File-based experiment result registry.

    Stores results as individual JSON files in a directory.
    """

    def __init__(self, directory: str = "results"):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def save(self, record: ExperimentRecord) -> str:
        """Save an experiment record. Returns the file path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = record.config.name.replace(" ", "_").replace("/", "_")
        filename = f"{timestamp}_{name}.json"
        path = os.path.join(self.directory, filename)

        with open(path, "w") as f:
            json.dump(record.to_dict(), f, indent=2, ensure_ascii=False)

        return path

    def list_all(self) -> List[str]:
        """List all result files."""
        return sorted(
            f for f in os.listdir(self.directory) if f.endswith(".json")
        )

    def load(self, filename: str) -> ExperimentRecord:
        """Load a single result."""
        path = os.path.join(self.directory, filename)
        with open(path) as f:
            data = json.load(f)
        config = ExperimentConfig.from_dict(data.get("config", {}))
        return ExperimentRecord(
            config=config,
            bleu=data.get("bleu", 0.0),
            comet=data.get("comet"),
            xcomet=data.get("xcomet"),
            avg_yaal=data.get("avg_yaal", 0.0),
            avg_al=data.get("avg_al", 0.0),
            avg_ap=data.get("avg_ap", 0.0),
            avg_dal=data.get("avg_dal", 0.0),
            avg_max_cw=data.get("avg_max_cw", 0.0),
            n_sentences=data.get("n_sentences", 0),
            total_time_s=data.get("total_time_s", 0.0),
            timestamp=data.get("timestamp", ""),
            machine=data.get("machine", ""),
        )

    def load_all(self) -> List[ExperimentRecord]:
        """Load all experiment records."""
        records = []
        for f in self.list_all():
            try:
                records.append(self.load(f))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  Skipping {f}: {e}", file=sys.stderr)
        return records


# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------

def run_experiment(config: ExperimentConfig, verbose: bool = True) -> ExperimentRecord:
    """Run an experiment from config and return a record.

    Args:
        config: Experiment configuration
        verbose: Print progress

    Returns:
        ExperimentRecord with results
    """
    from .backend_protocol import BackendConfig, create_backend
    from .eval import load_flores, evaluate_backend
    from .corpus import get_corpus_as_pairs

    # Load corpus
    parts = config.direction.split("-")
    src_lang, tgt_lang = parts[0], parts[1]

    if config.corpus_suite == "corpus":
        corpus = get_corpus_as_pairs(config.direction, n=config.n_sentences)
        if not corpus:
            corpus = load_flores(src_lang, tgt_lang, n=config.n_sentences)
    else:
        corpus = load_flores(src_lang, tgt_lang, n=config.n_sentences)

    if verbose:
        print(f"Running: {config.name} ({config.direction}, {len(corpus)} sentences)",
              file=sys.stderr)

    # Create backend
    backend_cfg = BackendConfig.from_dict(config.to_backend_config())
    backend = create_backend(backend_cfg)

    try:
        result = evaluate_backend(
            backend, corpus,
            compute_comet_score=config.compute_comet,
            compute_xcomet_score=config.compute_xcomet,
            verbose=verbose,
        )
    finally:
        backend.close()

    import socket
    record = ExperimentRecord(
        config=config,
        bleu=result.bleu or 0.0,
        comet=result.comet,
        xcomet=result.xcomet,
        avg_yaal=result.avg_yaal,
        avg_al=result.avg_al,
        avg_ap=result.avg_ap,
        avg_dal=result.avg_dal,
        avg_max_cw=result.avg_max_cw,
        n_sentences=result.n_sentences,
        total_time_s=result.total_time_s,
        timestamp=datetime.now().isoformat(),
        machine=socket.gethostname(),
    )

    if verbose:
        print(record.summary_line(), file=sys.stderr)

    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="NLLW Experiment Manager")
    sub = parser.add_subparsers(dest="command")

    # Run
    run_p = sub.add_parser("run", help="Run an experiment from config")
    run_p.add_argument("config", help="YAML config file")
    run_p.add_argument("--save", action="store_true", help="Save results to registry")
    run_p.add_argument("--results-dir", default="results", help="Results directory")

    # List
    list_p = sub.add_parser("list", help="List experiment results")
    list_p.add_argument("--results-dir", default="results")

    # Compare
    cmp_p = sub.add_parser("compare", help="Compare experiments")
    cmp_p.add_argument("configs", nargs="+", help="YAML config files")
    cmp_p.add_argument("--results-dir", default="results")

    args = parser.parse_args()

    if args.command == "run":
        config = ExperimentConfig.from_yaml(args.config)
        record = run_experiment(config)
        if args.save:
            registry = ExperimentRegistry(args.results_dir)
            path = registry.save(record)
            print(f"Saved to {path}", file=sys.stderr)

    elif args.command == "list":
        registry = ExperimentRegistry(args.results_dir)
        records = registry.load_all()
        if not records:
            print("No experiments found.", file=sys.stderr)
            return
        for r in records:
            print(r.summary_line())

    elif args.command == "compare":
        records = []
        for cfg_path in args.configs:
            config = ExperimentConfig.from_yaml(cfg_path)
            record = run_experiment(config)
            records.append(record)
        print("\n=== Comparison ===")
        for r in records:
            print(r.summary_line())

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
