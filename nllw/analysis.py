"""Result analysis, Pareto frontier, and report generation for SimulMT experiments.

Provides tools for:
    - Loading and comparing experiment results
    - Computing Pareto frontiers (quality vs latency tradeoffs)
    - Identifying edge cases and failure patterns
    - Generating human-readable reports

Usage:
    from nllw.analysis import ParetoAnalyzer, load_results
    results = load_results("results/*.json")
    analyzer = ParetoAnalyzer(results)
    frontier = analyzer.pareto_frontier(quality_key="comet", latency_key="avg_yaal")
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class ExperimentResult:
    """A single experiment result for analysis."""
    name: str
    config: Dict[str, Any]
    # Quality
    bleu: float = 0.0
    comet: Optional[float] = None
    xcomet: Optional[float] = None
    # Latency
    avg_al: float = 0.0
    avg_laal: float = 0.0
    avg_yaal: float = 0.0
    avg_ap: float = 0.0
    avg_dal: float = 0.0
    avg_max_cw: float = 0.0
    # Meta
    n_sentences: int = 0
    direction: str = ""
    backend_type: str = ""
    avg_time_ms: float = 0.0
    # Per-sentence (optional)
    per_sentence: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_eval_dict(cls, d: Dict[str, Any], name: str = "") -> "ExperimentResult":
        """Create from an EvalResult.to_dict() output."""
        return cls(
            name=name or d.get("backend_type", "unknown"),
            config=d.get("config", {}),
            bleu=d.get("bleu", 0.0),
            comet=d.get("comet"),
            xcomet=d.get("xcomet"),
            avg_al=d.get("avg_al", 0.0),
            avg_laal=d.get("avg_laal", 0.0),
            avg_yaal=d.get("avg_yaal", 0.0),
            avg_ap=d.get("avg_ap", 0.0),
            avg_dal=d.get("avg_dal", 0.0),
            avg_max_cw=d.get("avg_max_cw", 0.0),
            n_sentences=d.get("n_sentences", 0),
            direction=d.get("direction", ""),
            backend_type=d.get("backend_type", ""),
            avg_time_ms=d.get("avg_time_per_sentence_ms", 0.0),
        )

    @property
    def quality(self) -> float:
        """Best available quality metric (prefer xcomet > comet > bleu/100)."""
        if self.xcomet is not None:
            return self.xcomet
        if self.comet is not None:
            return self.comet
        return self.bleu / 100.0

    @property
    def latency(self) -> float:
        """Primary latency metric (YAAL, IWSLT 2026 standard)."""
        return self.avg_yaal

    def config_str(self) -> str:
        """Short config description."""
        parts = []
        for key in ["border_distance", "word_batch", "context_sentences", "entropy_veto_threshold"]:
            if key in self.config:
                short = {"border_distance": "bd", "word_batch": "wb",
                         "context_sentences": "ctx", "entropy_veto_threshold": "ent"}
                parts.append(f"{short.get(key, key)}={self.config[key]}")
        return " ".join(parts) if parts else self.name


def load_results(path: str) -> List[ExperimentResult]:
    """Load experiment results from a JSON file.

    Handles both single result dicts and arrays of results.
    """
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return [ExperimentResult.from_eval_dict(d, name=f"{os.path.basename(path)}_{i}")
                for i, d in enumerate(data)]
    elif isinstance(data, dict):
        if "per_sentence" in data and "summary" in data:
            return [ExperimentResult.from_eval_dict(data["summary"], name=os.path.basename(path))]
        return [ExperimentResult.from_eval_dict(data, name=os.path.basename(path))]
    return []


def load_results_dir(directory: str) -> List[ExperimentResult]:
    """Load all JSON result files from a directory."""
    results = []
    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".json"):
            try:
                results.extend(load_results(os.path.join(directory, fname)))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  Skipping {fname}: {e}")
    return results


# ---------------------------------------------------------------------------
# Pareto analysis
# ---------------------------------------------------------------------------

class ParetoAnalyzer:
    """Analyze quality-latency Pareto frontiers across experiments."""

    def __init__(self, results: List[ExperimentResult]):
        self.results = results

    def pareto_frontier(
        self,
        quality_key: str = "quality",
        latency_key: str = "latency",
        higher_quality_better: bool = True,
        lower_latency_better: bool = True,
    ) -> List[ExperimentResult]:
        """Compute the Pareto frontier.

        A result is Pareto-optimal if no other result is strictly better
        on both quality and latency.

        Args:
            quality_key: Attribute name for quality (default uses .quality property)
            latency_key: Attribute name for latency (default uses .latency property)
            higher_quality_better: Whether higher quality is better
            lower_latency_better: Whether lower latency is better

        Returns:
            List of Pareto-optimal results, sorted by latency
        """
        def get_quality(r):
            return getattr(r, quality_key) if hasattr(r, quality_key) else r.quality

        def get_latency(r):
            return getattr(r, latency_key) if hasattr(r, latency_key) else r.latency

        frontier = []
        for r in self.results:
            q = get_quality(r)
            l = get_latency(r)
            if q is None or l is None:
                continue

            dominated = False
            for other in self.results:
                oq = get_quality(other)
                ol = get_latency(other)
                if oq is None or ol is None:
                    continue

                better_q = (oq > q) if higher_quality_better else (oq < q)
                better_l = (ol < l) if lower_latency_better else (ol > l)
                eq_q = abs(oq - q) < 1e-6
                eq_l = abs(ol - l) < 1e-6

                if (better_q and (better_l or eq_l)) or (better_l and (better_q or eq_q)):
                    if not (eq_q and eq_l):
                        dominated = True
                        break

            if not dominated:
                frontier.append(r)

        frontier.sort(key=lambda r: get_latency(r))
        return frontier

    def best_by(self, key: str, maximize: bool = True) -> Optional[ExperimentResult]:
        """Find the result with the best value for a given metric."""
        valid = [r for r in self.results if getattr(r, key, None) is not None]
        if not valid:
            return None
        return max(valid, key=lambda r: getattr(r, key)) if maximize else \
               min(valid, key=lambda r: getattr(r, key))

    def filter_direction(self, direction: str) -> "ParetoAnalyzer":
        """Filter results to a specific language direction."""
        return ParetoAnalyzer([r for r in self.results if r.direction == direction])

    def filter_backend(self, backend_type: str) -> "ParetoAnalyzer":
        """Filter results to a specific backend type."""
        return ParetoAnalyzer([r for r in self.results if r.backend_type == backend_type])


# ---------------------------------------------------------------------------
# Edge case analysis
# ---------------------------------------------------------------------------

def find_edge_cases(
    results: List[ExperimentResult],
    quality_threshold: float = 0.5,
    latency_threshold: float = 10.0,
) -> Dict[str, List[ExperimentResult]]:
    """Identify problematic results for further investigation.

    Returns dict with categories:
        - low_quality: Quality below threshold
        - high_latency: Latency above threshold
        - quality_outliers: Per-sentence quality much worse than average
    """
    categories = {
        "low_quality": [],
        "high_latency": [],
        "low_commit_rate": [],
    }

    for r in results:
        if r.quality < quality_threshold:
            categories["low_quality"].append(r)
        if r.latency > latency_threshold:
            categories["high_latency"].append(r)

    return categories


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_comparison_table(
    results: List[ExperimentResult],
    include_per_direction: bool = True,
) -> str:
    """Generate a markdown comparison table of results."""
    if not results:
        return "No results to display."

    lines = []
    lines.append("| Config | Direction | BLEU | COMET | XCOMET | YAAL | AL | AP | ms/sent |")
    lines.append("|--------|-----------|------|-------|--------|------|----|----|---------|")

    for r in sorted(results, key=lambda x: (x.direction, -x.quality)):
        comet_str = f"{r.comet:.3f}" if r.comet is not None else "-"
        xcomet_str = f"{r.xcomet:.3f}" if r.xcomet is not None else "-"
        lines.append(
            f"| {r.config_str():<20} | {r.direction:<9} | "
            f"{r.bleu:5.1f} | {comet_str:>7} | {xcomet_str:>7} | "
            f"{r.avg_yaal:5.2f} | {r.avg_al:5.2f} | {r.avg_ap:5.3f} | "
            f"{r.avg_time_ms:7.0f} |"
        )

    return "\n".join(lines)


def generate_pareto_report(
    results: List[ExperimentResult],
    title: str = "Pareto Analysis",
) -> str:
    """Generate a Pareto analysis report in markdown."""
    analyzer = ParetoAnalyzer(results)

    lines = [f"# {title}\n"]

    # Overall stats
    lines.append(f"Total experiments: {len(results)}")
    directions = sorted(set(r.direction for r in results))
    lines.append(f"Directions: {', '.join(directions)}")
    lines.append("")

    # Per-direction analysis
    for direction in directions:
        dir_results = [r for r in results if r.direction == direction]
        dir_analyzer = ParetoAnalyzer(dir_results)
        frontier = dir_analyzer.pareto_frontier()

        lines.append(f"## {direction.upper()} ({len(dir_results)} experiments)\n")
        lines.append(f"### Pareto frontier ({len(frontier)} points)\n")
        lines.append("| Config | Quality | YAAL | BLEU | ms/sent |")
        lines.append("|--------|---------|------|------|---------|")
        for r in frontier:
            lines.append(
                f"| {r.config_str():<25} | {r.quality:.3f} | "
                f"{r.avg_yaal:5.2f} | {r.bleu:5.1f} | {r.avg_time_ms:7.0f} |"
            )
        lines.append("")

        # Best configs
        best_quality = dir_analyzer.best_by("quality", maximize=True)
        best_latency = dir_analyzer.best_by("avg_yaal", maximize=False)
        if best_quality:
            lines.append(f"**Best quality**: {best_quality.config_str()} "
                         f"(quality={best_quality.quality:.3f}, YAAL={best_quality.avg_yaal:.2f})")
        if best_latency:
            lines.append(f"**Best latency**: {best_latency.config_str()} "
                         f"(quality={best_latency.quality:.3f}, YAAL={best_latency.avg_yaal:.2f})")
        lines.append("")

    return "\n".join(lines)
