"""Pareto frontier analysis and edge case diagnostics for NLLW experiments.

Provides:
  - compute_pareto_frontier: identify quality-latency Pareto-optimal configs
  - format_comparison_table: ASCII ranked table with Pareto markers
  - format_latex_table: LaTeX table for paper submission
  - analyze_edge_cases: per-category quality breakdown from sentence traces
  - generate_report: full Markdown report (tables, ASCII Pareto plot, recommendations)
  - CLI: python -m nllw.analysis --results experiments/ --output docs/research/

All output is ASCII — no matplotlib dependency required.

Usage (library):
    from nllw.analysis import compute_pareto_frontier, generate_report
    pareto = compute_pareto_frontier(results, quality_metric="comet")
    generate_report(results, corpus_name="flores_mini", output_dir="reports/")

Usage (CLI):
    python -m nllw.analysis --results experiments/ --output docs/research/
    python -m nllw.analysis --results experiments/ --pareto --quality comet --latency ca_al_ms
"""

from __future__ import annotations

import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Pareto frontier computation
# ---------------------------------------------------------------------------


def compute_pareto_frontier(
    results: list[dict],
    quality_metric: str = "comet",
    latency_metric: str = "ca_al_ms",
) -> list[dict]:
    """Identify Pareto-optimal points from a list of experiment results.

    A point is Pareto-optimal if no other point is **strictly better** on both
    metrics (higher quality AND lower latency).

    Parameters
    ----------
    results : list[dict]
        Each dict represents one experiment configuration. Must contain
        keys for *quality_metric* and *latency_metric*. Common shapes:
        - Output of ``ResearchBenchmark.run_benchmark_suite()["configs"]``
          (keys: bleu, avg_committed_ratio, avg_ca_al_ms, avg_al_words, ...)
        - Output of ``SimulMTEvaluator.run_parameter_sweep()``
          (keys: bleu, avg_committed_ratio, avg_time_per_word_ms, ...)
        - Raw experiment registry dicts with arbitrary metric keys.

        Metric value aliases handled automatically:
        - ``"ca_al_ms"`` looks up ``"ca_al_ms"`` then ``"avg_ca_al_ms"``
        - ``"al_words"`` looks up ``"al_words"`` then ``"avg_al_words"``
          then ``"standard_al_words"``
        - ``"committed_ratio"`` looks up ``"committed_ratio"`` then
          ``"avg_committed_ratio"``
    quality_metric : str
        Higher is better. Default: ``"comet"``.
    latency_metric : str
        Lower is better. Default: ``"ca_al_ms"``.

    Returns
    -------
    list[dict]
        Pareto-optimal points, each containing:
        - ``config``: the original result dict
        - ``quality_score``: extracted quality value
        - ``latency_score``: extracted latency value
        - ``quality_metric``: name of the quality metric used
        - ``latency_metric``: name of the latency metric used
        Sorted by quality_score descending.
    """
    # Extract metric values with alias resolution
    points: list[dict] = []
    for r in results:
        q = _resolve_metric(r, quality_metric)
        l = _resolve_metric(r, latency_metric)
        if q is None or l is None:
            continue
        points.append({
            "config": r,
            "quality_score": q,
            "latency_score": l,
            "quality_metric": quality_metric,
            "latency_metric": latency_metric,
        })

    if not points:
        return []

    # Filter non-dominated solutions
    # A point p is dominated if there exists another point q where
    # q.quality >= p.quality AND q.latency <= p.latency AND at least one strict.
    pareto: list[dict] = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            if (
                q["quality_score"] >= p["quality_score"]
                and q["latency_score"] <= p["latency_score"]
                and (
                    q["quality_score"] > p["quality_score"]
                    or q["latency_score"] < p["latency_score"]
                )
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(p)

    # Sort by quality descending
    pareto.sort(key=lambda p: p["quality_score"], reverse=True)
    return pareto


def _resolve_metric(result: dict, metric: str) -> Optional[float]:
    """Extract a metric value from a result dict, trying common aliases."""
    ALIASES: dict[str, list[str]] = {
        "ca_al_ms": ["ca_al_ms", "avg_ca_al_ms"],
        "al_words": ["al_words", "avg_al_words", "standard_al_words"],
        "committed_ratio": ["committed_ratio", "avg_committed_ratio"],
        "step_ms": ["step_ms", "avg_step_ms", "avg_time_per_word_ms"],
    }

    # Try the exact key first
    if metric in result and result[metric] is not None:
        return float(result[metric])

    # Try aliases
    for alias in ALIASES.get(metric, []):
        if alias in result and result[alias] is not None:
            return float(result[alias])

    return None


def _get_label(result: dict) -> str:
    """Extract a human-readable label from a result dict."""
    if "label" in result:
        return str(result["label"])
    # Build from config params
    parts = []
    for key in ["border_distance", "word_batch", "backend_type", "context_window"]:
        if key in result:
            short = {"border_distance": "bd", "word_batch": "wb",
                     "backend_type": "be", "context_window": "ctx"}.get(key, key)
            parts.append(f"{short}={result[key]}")
    if "config" in result and isinstance(result["config"], dict):
        for key in ["border_distance", "word_batch", "backend_type", "context_window"]:
            if key in result["config"]:
                short = {"border_distance": "bd", "word_batch": "wb",
                         "backend_type": "be", "context_window": "ctx"}.get(key, key)
                parts.append(f"{short}={result['config'][key]}")
    return " ".join(parts) if parts else "config"


def _is_pareto_optimal(result: dict, pareto_configs: list[dict]) -> bool:
    """Check if a result dict is in the Pareto frontier."""
    for p in pareto_configs:
        if p["config"] is result:
            return True
    return False


# ---------------------------------------------------------------------------
# ASCII comparison table
# ---------------------------------------------------------------------------


def format_comparison_table(
    results: list[dict],
    metrics: Optional[list[str]] = None,
    *,
    sort_by: Optional[str] = None,
    pareto_quality: str = "comet",
    pareto_latency: str = "ca_al_ms",
    max_label_width: int = 28,
) -> str:
    """Format results as an ASCII comparison table.

    Parameters
    ----------
    results : list[dict]
        Experiment result dicts (from sweep, benchmark suite, or registry).
    metrics : list[str] or None
        Metrics to display as columns. Default:
        ``["bleu", "comet", "committed_ratio", "ca_al_ms"]``.
    sort_by : str or None
        Sort rows by this metric (descending for quality metrics,
        ascending for latency). Defaults to first metric in *metrics*.
    pareto_quality, pareto_latency : str
        Used to compute Pareto frontier. Rows on the frontier are marked
        with ``*`` in the first column.
    max_label_width : int
        Maximum width for the config label column.

    Returns
    -------
    str
        Multi-line ASCII table string.
    """
    if not results:
        return "  (no results)\n"

    if metrics is None:
        metrics = ["bleu", "comet", "committed_ratio", "ca_al_ms"]

    if sort_by is None:
        sort_by = metrics[0]

    # Compute Pareto frontier
    pareto = compute_pareto_frontier(results, pareto_quality, pareto_latency)
    pareto_ids = {id(p["config"]) for p in pareto}

    # Sort results
    LOWER_IS_BETTER = {"ca_al_ms", "avg_ca_al_ms", "al_words", "avg_al_words",
                       "step_ms", "avg_step_ms", "avg_time_per_word_ms",
                       "total_time_ms", "realtime_ratio"}
    reverse = sort_by not in LOWER_IS_BETTER

    def sort_key(r: dict) -> float:
        v = _resolve_metric(r, sort_by)
        if v is None:
            return float("-inf") if reverse else float("inf")
        return v

    sorted_results = sorted(results, key=sort_key, reverse=reverse)

    # Format columns
    METRIC_FORMATS: dict[str, dict] = {
        "bleu":              {"header": "BLEU",      "fmt": "{:>7.2f}", "width": 7},
        "comet":             {"header": "COMET",     "fmt": "{:>7.4f}", "width": 7},
        "xcomet_xl":         {"header": "xCOMET",    "fmt": "{:>7.4f}", "width": 7},
        "committed_ratio":   {"header": "Commit%",   "fmt": "{:>7.1%}", "width": 7},
        "ca_al_ms":          {"header": "CA-AL(ms)", "fmt": "{:>9.0f}", "width": 9},
        "al_words":          {"header": "AL(words)", "fmt": "{:>9.2f}", "width": 9},
        "step_ms":           {"header": "Step(ms)",  "fmt": "{:>8.0f}", "width": 8},
        "realtime_ratio":    {"header": "RT ratio",  "fmt": "{:>8.2f}", "width": 8},
        "total_time_ms":     {"header": "Total(ms)", "fmt": "{:>9.0f}", "width": 9},
    }

    lines: list[str] = []

    # Header
    hdr_parts = [f"{'Config':<{max_label_width}} "]
    for m in metrics:
        info = METRIC_FORMATS.get(m, {"header": m, "width": max(len(m), 7)})
        hdr_parts.append(f"| {info['header']:>{info['width']}} ")
    hdr_parts.append("| P")
    header = "".join(hdr_parts)
    lines.append(header)
    lines.append("-" * len(header))

    # Rows
    for r in sorted_results:
        label = _get_label(r)
        if len(label) > max_label_width:
            label = label[: max_label_width - 2] + ".."
        row_parts = [f"{label:<{max_label_width}} "]

        for m in metrics:
            info = METRIC_FORMATS.get(m, {"fmt": "{:>7.3f}", "width": max(len(m), 7)})
            v = _resolve_metric(r, m)
            if v is not None:
                cell = info["fmt"].format(v)
            else:
                cell = f"{'N/A':>{info['width']}}"
            row_parts.append(f"| {cell} ")

        is_pareto = id(r) in pareto_ids
        row_parts.append(f"| {'*' if is_pareto else ' '}")
        lines.append("".join(row_parts))

    lines.append("")
    lines.append(f"  P = Pareto-optimal ({pareto_quality} vs {pareto_latency})")
    lines.append(f"  Sorted by: {sort_by} ({'desc' if reverse else 'asc'})")
    lines.append(f"  {len(pareto)} Pareto-optimal configs out of {len(results)} total")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------


def format_latex_table(
    results: list[dict],
    *,
    metrics: Optional[list[str]] = None,
    caption: str = "Simultaneous translation results",
    label: str = "tab:simulmt-results",
    pareto_quality: str = "comet",
    pareto_latency: str = "ca_al_ms",
) -> str:
    """Format results as a LaTeX table for paper submission.

    Standard SimulMT format: System | BLEU | COMET | AL | LAAL (CA-AL).

    Parameters
    ----------
    results : list[dict]
        Experiment result dicts.
    metrics : list[str] or None
        Default: ``["bleu", "comet", "al_words", "ca_al_ms"]``.
    caption, label : str
        LaTeX caption and label.
    pareto_quality, pareto_latency : str
        Pareto-optimal rows are bolded.

    Returns
    -------
    str
        Complete LaTeX table environment string.
    """
    if metrics is None:
        metrics = ["bleu", "comet", "al_words", "ca_al_ms"]

    pareto = compute_pareto_frontier(results, pareto_quality, pareto_latency)
    pareto_ids = {id(p["config"]) for p in pareto}

    LATEX_HEADERS: dict[str, str] = {
        "bleu": "BLEU",
        "comet": "COMET",
        "xcomet_xl": "xCOMET-XL",
        "committed_ratio": "Commit\\%",
        "ca_al_ms": "LAAL (ms)",
        "al_words": "AL (words)",
        "step_ms": "Step (ms)",
    }

    LATEX_FMTS: dict[str, str] = {
        "bleu": "{:.2f}",
        "comet": "{:.4f}",
        "xcomet_xl": "{:.4f}",
        "committed_ratio": "{:.1%}",
        "ca_al_ms": "{:.0f}",
        "al_words": "{:.2f}",
        "step_ms": "{:.0f}",
    }

    # Sort by first quality metric descending
    LOWER_IS_BETTER = {"ca_al_ms", "al_words", "step_ms"}
    primary = metrics[0]
    rev = primary not in LOWER_IS_BETTER

    def sort_key(r: dict) -> float:
        v = _resolve_metric(r, primary)
        return v if v is not None else (float("-inf") if rev else float("inf"))

    sorted_results = sorted(results, key=sort_key, reverse=rev)

    # Build LaTeX
    n_cols = 1 + len(metrics)
    col_spec = "l" + "r" * len(metrics)

    lines: list[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header row
    hdrs = ["System"] + [LATEX_HEADERS.get(m, m) for m in metrics]
    lines.append(" & ".join(hdrs) + " \\\\")
    lines.append("\\midrule")

    # Data rows
    for r in sorted_results:
        is_pareto = id(r) in pareto_ids
        label_str = _get_label(r).replace("_", "\\_").replace("%", "\\%")
        if is_pareto:
            label_str = f"\\textbf{{{label_str}}}"

        cells = [label_str]
        for m in metrics:
            v = _resolve_metric(r, m)
            if v is not None:
                fmt = LATEX_FMTS.get(m, "{:.3f}")
                cell = fmt.format(v)
                if is_pareto:
                    cell = f"\\textbf{{{cell}}}"
            else:
                cell = "---"
            cells.append(cell)
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Edge case analysis
# ---------------------------------------------------------------------------


def analyze_edge_cases(
    traces: list[dict],
    corpus: list[dict],
) -> dict:
    """Analyze per-category quality breakdown from sentence-level traces.

    Parameters
    ----------
    traces : list[dict]
        Per-sentence trace dicts. Each should contain at minimum:
        - ``tag``: category tag (e.g. ``"en-fr/short"``, ``"fr-en/pronoun_ambiguity"``)
        - ``committed_ratio``: float 0..1
        - ``hypothesis``: translated text

        Accepted shapes:
        - Entries from ``ResearchBenchmark.run_benchmark_suite()``
          ``configs[i]["sentences"]``
        - Entries from ``SimulMTEvaluator.evaluate_corpus()["sentences"]``

        Optional keys used if present:
        - ``steps``: list of per-word step dicts (for step-level analysis)
        - ``ca_al_ms``, ``standard_al_words``: latency metrics
        - ``bleu``, ``comet``: per-sentence quality scores
    corpus : list[dict]
        The test corpus used (for reference texts and category metadata).
        Each entry should have ``tag``, ``source``, ``reference``, and
        optionally ``category``.

    Returns
    -------
    dict with keys:
        - ``categories``: dict mapping category name to breakdown dict:
            - ``count``: number of sentences in this category
            - ``avg_committed_ratio``: mean committed ratio
            - ``avg_quality``: mean of best available quality metric
            - ``failure_rate``: fraction of sentences with committed_ratio < 0.3
            - ``avg_ca_al_ms``: mean CA-AL (if available)
            - ``sentences``: list of per-sentence summaries
            - ``quality_metric``: which metric was used for avg_quality
        - ``lang_pairs``: dict mapping lang pair (e.g. "en-fr") to same structure
        - ``worst_sentences``: top 10 sentences by lowest committed_ratio
        - ``step_anomalies``: list of steps flagged as problematic
            (e.g. zero new_committed despite new input, high step times)
    """
    # Build reference map from corpus
    corpus_by_tag: dict[str, dict] = {}
    for entry in corpus:
        tag = entry.get("tag", "")
        if tag:
            corpus_by_tag[tag] = entry

    # Pair traces with corpus entries and categorize
    categorized: dict[str, list[dict]] = {}
    lang_pair_groups: dict[str, list[dict]] = {}
    all_annotated: list[dict] = []

    for i, trace in enumerate(traces):
        tag = trace.get("tag", "")

        # Extract category from tag: "en-fr/short" -> "short", "en-fr/3" -> "general"
        lang_pair, _, cat_part = tag.partition("/")
        if cat_part and not cat_part.isdigit():
            category = cat_part
        else:
            # Check corpus entry for explicit category
            corpus_entry = corpus_by_tag.get(tag, {})
            category = corpus_entry.get("category", "general")

        # Get quality score (prefer comet > bleu > committed_ratio)
        quality_score, quality_metric = _best_quality_score(trace)

        # Detect failure: very low committed ratio
        committed = trace.get("committed_ratio",
                              trace.get("avg_committed_ratio", 0.0))
        is_failure = committed < 0.3

        annotated = {
            "index": i,
            "tag": tag,
            "category": category,
            "lang_pair": lang_pair,
            "source": trace.get("source", ""),
            "reference": trace.get("reference", ""),
            "hypothesis": trace.get("hypothesis", ""),
            "committed_ratio": committed,
            "quality_score": quality_score,
            "quality_metric": quality_metric,
            "is_failure": is_failure,
            "ca_al_ms": trace.get("ca_al_ms", trace.get("avg_ca_al_ms")),
            "al_words": trace.get("standard_al_words", trace.get("avg_al_words")),
        }
        all_annotated.append(annotated)

        # Group by category
        categorized.setdefault(category, []).append(annotated)
        if lang_pair:
            lang_pair_groups.setdefault(lang_pair, []).append(annotated)

    # Step-level anomaly detection
    step_anomalies = _detect_step_anomalies(traces)

    # Build per-category summaries
    categories_summary = {}
    for cat, items in sorted(categorized.items()):
        categories_summary[cat] = _summarize_group(items)

    lang_pairs_summary = {}
    for lp, items in sorted(lang_pair_groups.items()):
        lang_pairs_summary[lp] = _summarize_group(items)

    # Worst sentences (lowest committed_ratio)
    worst = sorted(all_annotated, key=lambda x: x["committed_ratio"])[:10]

    return {
        "categories": categories_summary,
        "lang_pairs": lang_pairs_summary,
        "worst_sentences": worst,
        "step_anomalies": step_anomalies,
        "total_sentences": len(traces),
        "total_failures": sum(1 for a in all_annotated if a["is_failure"]),
    }


def _best_quality_score(trace: dict) -> tuple[float, str]:
    """Extract the best available quality score from a trace."""
    for metric in ["comet", "xcomet_xl", "bleu"]:
        v = trace.get(metric)
        if v is not None:
            return (float(v), metric)
    # Fall back to committed_ratio as proxy
    cr = trace.get("committed_ratio", trace.get("avg_committed_ratio", 0.0))
    return (float(cr), "committed_ratio")


def _summarize_group(items: list[dict]) -> dict:
    """Compute summary statistics for a group of annotated traces."""
    n = len(items)
    if n == 0:
        return {"count": 0, "avg_committed_ratio": 0.0, "avg_quality": 0.0,
                "failure_rate": 0.0, "avg_ca_al_ms": None, "sentences": [],
                "quality_metric": "N/A"}

    committed_ratios = [it["committed_ratio"] for it in items]
    quality_scores = [it["quality_score"] for it in items]
    failures = sum(1 for it in items if it["is_failure"])

    ca_al_values = [it["ca_al_ms"] for it in items if it["ca_al_ms"] is not None]
    al_words_values = [it["al_words"] for it in items if it["al_words"] is not None]

    quality_metric = items[0]["quality_metric"] if items else "N/A"

    return {
        "count": n,
        "avg_committed_ratio": round(statistics.mean(committed_ratios), 4),
        "avg_quality": round(statistics.mean(quality_scores), 4),
        "failure_rate": round(failures / n, 4),
        "avg_ca_al_ms": round(statistics.mean(ca_al_values), 1) if ca_al_values else None,
        "avg_al_words": round(statistics.mean(al_words_values), 3) if al_words_values else None,
        "std_committed_ratio": round(statistics.stdev(committed_ratios), 4) if n > 1 else 0.0,
        "std_quality": round(statistics.stdev(quality_scores), 4) if n > 1 else 0.0,
        "quality_metric": quality_metric,
        "sentences": items,
    }


def _detect_step_anomalies(traces: list[dict]) -> list[dict]:
    """Scan per-step data for anomalies (stalls, high latency, etc.)."""
    anomalies: list[dict] = []

    for t_idx, trace in enumerate(traces):
        steps = trace.get("steps", [])
        if not steps:
            continue

        tag = trace.get("tag", f"sentence_{t_idx}")

        # Compute median step time for this sentence
        step_times = [s.get("step_time_ms", 0) for s in steps]
        median_time = statistics.median(step_times) if step_times else 0

        consecutive_zero = 0
        for s_idx, step in enumerate(steps):
            new_committed = step.get("new_committed", 0)
            step_ms = step.get("step_time_ms", 0)

            # Flag: no output produced for 3+ consecutive words
            if new_committed == 0:
                consecutive_zero += 1
            else:
                consecutive_zero = 0

            if consecutive_zero >= 3 and s_idx == len(steps) - 1 or (
                consecutive_zero == 3
            ):
                anomalies.append({
                    "type": "output_stall",
                    "tag": tag,
                    "step_index": s_idx,
                    "word": step.get("word", ""),
                    "consecutive_zero_steps": consecutive_zero,
                    "detail": (f"No new committed tokens for {consecutive_zero} "
                               f"consecutive words ending at step {s_idx}"),
                })

            # Flag: step latency > 3x median (and > 100ms absolute)
            if median_time > 0 and step_ms > max(3 * median_time, 100):
                anomalies.append({
                    "type": "high_latency",
                    "tag": tag,
                    "step_index": s_idx,
                    "word": step.get("word", ""),
                    "step_time_ms": step_ms,
                    "median_step_ms": round(median_time, 1),
                    "detail": (f"Step {s_idx} took {step_ms:.0f}ms "
                               f"(median: {median_time:.0f}ms)"),
                })

    return anomalies


# ---------------------------------------------------------------------------
# ASCII Pareto plot
# ---------------------------------------------------------------------------


def format_pareto_ascii(
    results: list[dict],
    quality_metric: str = "comet",
    latency_metric: str = "ca_al_ms",
    *,
    width: int = 60,
    height: int = 20,
) -> str:
    """Render a Pareto frontier as ASCII art.

    X-axis: latency (lower is better, left is better)
    Y-axis: quality (higher is better, top is better)
    Pareto-optimal points marked with ``*``, others with ``o``.

    Parameters
    ----------
    width, height : int
        Character dimensions of the plot area.

    Returns
    -------
    str
        Multi-line ASCII plot.
    """
    pareto = compute_pareto_frontier(results, quality_metric, latency_metric)
    pareto_ids = {id(p["config"]) for p in pareto}

    # Extract all plottable points
    points: list[dict] = []
    for r in results:
        q = _resolve_metric(r, quality_metric)
        l = _resolve_metric(r, latency_metric)
        if q is None or l is None:
            continue
        points.append({
            "label": _get_label(r),
            "quality": q,
            "latency": l,
            "is_pareto": id(r) in pareto_ids,
        })

    if not points:
        return "  (no plottable data)\n"

    # Determine axis ranges (with 5% padding)
    q_values = [p["quality"] for p in points]
    l_values = [p["latency"] for p in points]

    q_min, q_max = min(q_values), max(q_values)
    l_min, l_max = min(l_values), max(l_values)

    # Add padding
    q_range = q_max - q_min if q_max > q_min else 0.01
    l_range = l_max - l_min if l_max > l_min else 1.0
    q_min -= q_range * 0.05
    q_max += q_range * 0.05
    l_min -= l_range * 0.05
    l_max += l_range * 0.05
    q_range = q_max - q_min
    l_range = l_max - l_min

    # Build character grid
    grid = [[" " for _ in range(width)] for _ in range(height)]

    # Place points (latency on X, quality on Y)
    placed: list[dict] = []
    for p in points:
        x = int((p["latency"] - l_min) / l_range * (width - 1))
        y = int((p["quality"] - q_min) / q_range * (height - 1))
        # Flip Y so higher quality is at top
        y = height - 1 - y
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        char = "*" if p["is_pareto"] else "o"
        grid[y][x] = char
        placed.append({"x": x, "y": y, "label": p["label"], "is_pareto": p["is_pareto"]})

    # Draw the Pareto frontier line (connect Pareto points)
    pareto_points_sorted = sorted(
        [p for p in placed if p["is_pareto"]],
        key=lambda p: p["x"],
    )
    for i in range(len(pareto_points_sorted) - 1):
        p1 = pareto_points_sorted[i]
        p2 = pareto_points_sorted[i + 1]
        # Simple line drawing between consecutive Pareto points
        dx = p2["x"] - p1["x"]
        dy = p2["y"] - p1["y"]
        steps = max(abs(dx), abs(dy), 1)
        for s in range(1, steps):
            ix = p1["x"] + int(dx * s / steps)
            iy = p1["y"] + int(dy * s / steps)
            if 0 <= ix < width and 0 <= iy < height:
                if grid[iy][ix] == " ":
                    grid[iy][ix] = "."

    # Render
    lines: list[str] = []
    lines.append(f"  Pareto Frontier: {quality_metric} (Y, higher=better) "
                 f"vs {latency_metric} (X, lower=better)")
    lines.append("")

    # Y-axis labels (show ~4 ticks)
    y_ticks = 4
    for row_idx in range(height):
        # Y-axis label every (height / y_ticks) rows
        if row_idx % max(height // y_ticks, 1) == 0:
            q_val = q_max - (row_idx / max(height - 1, 1)) * q_range
            y_label = f"{q_val:>8.3f}"
        else:
            y_label = "        "

        border = "|" if row_idx == 0 or row_idx == height - 1 else "|"
        lines.append(f"  {y_label} {border}{''.join(grid[row_idx])}|")

    # X-axis border
    lines.append(f"  {'':>8} +{'-' * width}+")

    # X-axis labels
    x_label_line = f"  {'':>8}  "
    x_ticks = 4
    for t in range(x_ticks + 1):
        pos = int(t * (width - 1) / x_ticks)
        l_val = l_min + t * l_range / x_ticks
        tick_str = f"{l_val:.0f}"
        # Pad to position
        while len(x_label_line) < 12 + pos:
            x_label_line += " "
        x_label_line += tick_str
    lines.append(x_label_line)

    # Legend
    lines.append("")
    lines.append(f"  {'':>8}   * = Pareto-optimal   o = dominated   . = frontier line")

    # Label the Pareto points
    if pareto_points_sorted:
        lines.append("")
        lines.append("  Pareto-optimal configs:")
        for i, p in enumerate(pareto_points_sorted):
            lines.append(f"    {i + 1}. {p['label']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(
    results: list[dict],
    corpus_name: str,
    output_dir: str,
    *,
    corpus: Optional[list[dict]] = None,
    traces: Optional[list[dict]] = None,
    quality_metric: str = "comet",
    latency_metric: str = "ca_al_ms",
) -> str:
    """Generate a full Markdown analysis report.

    Parameters
    ----------
    results : list[dict]
        Experiment result dicts (one per config). Should contain at minimum
        a quality metric and latency metric. For per-sentence analysis,
        should also contain a ``"sentences"`` key with trace lists.
    corpus_name : str
        Name of the test corpus (for the report title).
    output_dir : str
        Directory to write the report to.
    corpus : list[dict] or None
        The test corpus (for edge case analysis). If None, edge case
        analysis is skipped.
    traces : list[dict] or None
        Explicit per-sentence traces. If None, traces are extracted from
        the first result's ``"sentences"`` key.
    quality_metric, latency_metric : str
        Metrics for Pareto analysis.

    Returns
    -------
    str
        Path to the generated report file.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "report.md")

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    sections: list[str] = []

    # Title
    sections.append(f"# SimulMT Analysis Report: {corpus_name}")
    sections.append("")
    sections.append(f"Generated: {timestamp}")
    sections.append(f"Configs evaluated: {len(results)}")
    sections.append("")

    # --- Section 1: Ranked comparison table ---
    sections.append("## Ranked Configuration Comparison")
    sections.append("")
    sections.append("```")
    sections.append(format_comparison_table(
        results,
        sort_by=quality_metric,
        pareto_quality=quality_metric,
        pareto_latency=latency_metric,
    ))
    sections.append("```")
    sections.append("")

    # --- Section 2: Pareto frontier ---
    pareto = compute_pareto_frontier(results, quality_metric, latency_metric)
    sections.append("## Pareto Frontier")
    sections.append("")
    sections.append(f"Quality metric: `{quality_metric}` (higher is better)")
    sections.append(f"Latency metric: `{latency_metric}` (lower is better)")
    sections.append(f"Pareto-optimal configs: {len(pareto)}/{len(results)}")
    sections.append("")

    sections.append("```")
    sections.append(format_pareto_ascii(
        results,
        quality_metric=quality_metric,
        latency_metric=latency_metric,
    ))
    sections.append("```")
    sections.append("")

    # Pareto point details
    if pareto:
        sections.append("### Pareto-Optimal Configurations")
        sections.append("")
        for i, p in enumerate(pareto):
            label = _get_label(p["config"])
            sections.append(
                f"{i + 1}. **{label}** -- "
                f"{quality_metric}={p['quality_score']:.4f}, "
                f"{latency_metric}={p['latency_score']:.0f}"
            )
        sections.append("")

    # --- Section 3: LaTeX table ---
    sections.append("## LaTeX Table (for paper submission)")
    sections.append("")
    sections.append("```latex")
    sections.append(format_latex_table(
        results,
        caption=f"SimulMT results on {corpus_name}",
        pareto_quality=quality_metric,
        pareto_latency=latency_metric,
    ))
    sections.append("```")
    sections.append("")

    # --- Section 4: Per-language-pair breakdown ---
    lang_pair_results = _group_by_lang_pair(results)
    if lang_pair_results:
        sections.append("## Per-Language-Pair Analysis")
        sections.append("")
        for lp, lp_results in sorted(lang_pair_results.items()):
            sections.append(f"### {lp}")
            sections.append("")
            lp_pareto = compute_pareto_frontier(lp_results, quality_metric, latency_metric)
            if lp_pareto:
                best = lp_pareto[0]
                sections.append(
                    f"Best config: **{_get_label(best['config'])}** "
                    f"({quality_metric}={best['quality_score']:.4f}, "
                    f"{latency_metric}={best['latency_score']:.0f})"
                )
            sections.append("")
            sections.append("```")
            sections.append(format_comparison_table(
                lp_results,
                sort_by=quality_metric,
                pareto_quality=quality_metric,
                pareto_latency=latency_metric,
            ))
            sections.append("```")
            sections.append("")

    # --- Section 5: Edge case analysis ---
    if traces or _has_sentences(results):
        actual_traces = traces
        if actual_traces is None:
            actual_traces = _extract_traces(results)
        if corpus is None:
            corpus = []

        if actual_traces:
            edge_analysis = analyze_edge_cases(actual_traces, corpus)
            sections.append("## Edge Case Analysis")
            sections.append("")
            sections.append(
                f"Total sentences: {edge_analysis['total_sentences']}, "
                f"Failures (commit < 30%): {edge_analysis['total_failures']}"
            )
            sections.append("")

            # Per-category table
            cats = edge_analysis["categories"]
            if cats:
                sections.append("### Per-Category Breakdown")
                sections.append("")
                sections.append("```")
                cat_lines = _format_category_table(cats)
                sections.append(cat_lines)
                sections.append("```")
                sections.append("")

            # Per-lang-pair table
            lps = edge_analysis["lang_pairs"]
            if lps:
                sections.append("### Per-Language-Pair Breakdown")
                sections.append("")
                sections.append("```")
                sections.append(_format_category_table(lps))
                sections.append("```")
                sections.append("")

            # Worst sentences
            worst = edge_analysis["worst_sentences"]
            if worst:
                sections.append("### Worst Performing Sentences")
                sections.append("")
                for w in worst[:5]:
                    sections.append(
                        f"- **{w['tag']}** (commit={w['committed_ratio']:.1%}): "
                        f"`{w['source'][:60]}{'...' if len(w.get('source', '')) > 60 else ''}`"
                    )
                sections.append("")

            # Step anomalies
            anomalies = edge_analysis["step_anomalies"]
            if anomalies:
                sections.append("### Step-Level Anomalies")
                sections.append("")
                sections.append(f"Total anomalies detected: {len(anomalies)}")
                sections.append("")
                # Group by type
                by_type: dict[str, list] = {}
                for a in anomalies:
                    by_type.setdefault(a["type"], []).append(a)
                for atype, alist in sorted(by_type.items()):
                    sections.append(f"**{atype}** ({len(alist)} occurrences):")
                    for a in alist[:5]:
                        sections.append(f"  - {a['tag']}: {a['detail']}")
                    if len(alist) > 5:
                        sections.append(f"  - ... and {len(alist) - 5} more")
                    sections.append("")

    # --- Section 6: Best config recommendations ---
    sections.append("## Recommendations")
    sections.append("")
    if pareto:
        # Recommend: best quality, best latency, balanced
        best_quality = max(pareto, key=lambda p: p["quality_score"])
        best_latency = min(pareto, key=lambda p: p["latency_score"])

        sections.append("### Best Quality")
        sections.append(
            f"- **{_get_label(best_quality['config'])}**: "
            f"{quality_metric}={best_quality['quality_score']:.4f}, "
            f"{latency_metric}={best_quality['latency_score']:.0f}"
        )
        sections.append("")

        sections.append("### Lowest Latency")
        sections.append(
            f"- **{_get_label(best_latency['config'])}**: "
            f"{quality_metric}={best_latency['quality_score']:.4f}, "
            f"{latency_metric}={best_latency['latency_score']:.0f}"
        )
        sections.append("")

        if len(pareto) >= 3:
            # Balanced: minimize normalized distance to ideal point
            q_range = best_quality["quality_score"] - min(p["quality_score"] for p in pareto)
            l_range = max(p["latency_score"] for p in pareto) - best_latency["latency_score"]
            q_range = q_range if q_range > 0 else 1.0
            l_range = l_range if l_range > 0 else 1.0

            def balanced_score(p: dict) -> float:
                q_norm = (best_quality["quality_score"] - p["quality_score"]) / q_range
                l_norm = (p["latency_score"] - best_latency["latency_score"]) / l_range
                return math.sqrt(q_norm ** 2 + l_norm ** 2)

            balanced = min(pareto, key=balanced_score)
            sections.append("### Balanced (quality-latency tradeoff)")
            sections.append(
                f"- **{_get_label(balanced['config'])}**: "
                f"{quality_metric}={balanced['quality_score']:.4f}, "
                f"{latency_metric}={balanced['latency_score']:.0f}"
            )
            sections.append("")

        # Per-language-pair recommendations
        if lang_pair_results:
            sections.append("### Best Config per Language Pair")
            sections.append("")
            for lp, lp_results in sorted(lang_pair_results.items()):
                lp_pareto = compute_pareto_frontier(lp_results, quality_metric, latency_metric)
                if lp_pareto:
                    best = lp_pareto[0]
                    sections.append(
                        f"- **{lp}**: {_get_label(best['config'])} "
                        f"({quality_metric}={best['quality_score']:.4f})"
                    )
            sections.append("")
    else:
        sections.append("No Pareto-optimal configs found (insufficient metric data).")
        sections.append("")

    # Write
    report_content = "\n".join(sections)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    # Also save raw results as JSON
    json_path = os.path.join(output_dir, "results.json")
    _save_results_json(results, json_path)

    return report_path


def _group_by_lang_pair(results: list[dict]) -> dict[str, list[dict]]:
    """Group results by language pair using per-sentence tags.

    Builds per-lang-pair result dicts with aggregated metrics for each
    config, so Pareto analysis can run per language pair.
    """
    lang_pair_data: dict[str, dict[str, list]] = {}  # lp -> label -> sentences

    for r in results:
        label = _get_label(r)
        sentences = r.get("sentences", [])
        for s in sentences:
            tag = s.get("tag", "")
            lp = tag.split("/")[0] if "/" in tag else ""
            if not lp:
                continue
            lang_pair_data.setdefault(lp, {}).setdefault(label, []).append(s)

    # Build per-LP result dicts
    grouped: dict[str, list[dict]] = {}
    for lp, labels in lang_pair_data.items():
        lp_results: list[dict] = []
        for label_str, sents in labels.items():
            n = len(sents)
            if n == 0:
                continue

            # Find the original result dict for this label to carry config
            original = None
            for r in results:
                if _get_label(r) == label_str:
                    original = r
                    break

            cr_vals = [s.get("committed_ratio", 0) for s in sents]
            ca_vals = [s.get("ca_al_ms", 0) for s in sents if s.get("ca_al_ms") is not None]
            al_vals = [s.get("standard_al_words", 0) for s in sents if s.get("standard_al_words") is not None]

            lp_result: dict[str, Any] = {
                "label": label_str,
                "lang_pair": lp,
                "num_sentences": n,
                "avg_committed_ratio": round(statistics.mean(cr_vals), 4) if cr_vals else 0,
                "avg_ca_al_ms": round(statistics.mean(ca_vals), 1) if ca_vals else 0,
                "avg_al_words": round(statistics.mean(al_vals), 3) if al_vals else 0,
                "sentences": sents,
            }
            # Carry over config and corpus-level metrics
            if original:
                lp_result["config"] = original.get("config", {})
                for key in ["bleu", "comet", "xcomet_xl"]:
                    if key in original:
                        lp_result[key] = original[key]
            lp_results.append(lp_result)

        if lp_results:
            grouped[lp] = lp_results

    return grouped


def _has_sentences(results: list[dict]) -> bool:
    """Check if any result has per-sentence traces."""
    return any(r.get("sentences") for r in results)


def _extract_traces(results: list[dict]) -> list[dict]:
    """Extract per-sentence traces from the first result that has them."""
    for r in results:
        sents = r.get("sentences")
        if sents:
            return sents
    return []


def _format_category_table(categories: dict[str, dict]) -> str:
    """Format category breakdown as an ASCII table."""
    if not categories:
        return "  (no categories)\n"

    lines: list[str] = []
    header = (
        f"{'Category':<24} | {'Count':>5} | {'Commit%':>8} | "
        f"{'Quality':>8} | {'Fail%':>6} | {'CA-AL':>9}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for cat, summary in sorted(categories.items()):
        count = summary["count"]
        cr = summary["avg_committed_ratio"]
        qual = summary["avg_quality"]
        fail = summary["failure_rate"]
        ca = summary.get("avg_ca_al_ms")
        ca_str = f"{ca:>9.0f}" if ca is not None else f"{'N/A':>9}"
        lines.append(
            f"{cat:<24} | {count:>5} | {cr:>7.1%} | "
            f"{qual:>8.4f} | {fail:>5.1%} | {ca_str}"
        )

    return "\n".join(lines)


def _save_results_json(results: list[dict], path: str) -> None:
    """Save results to JSON, stripping non-serializable objects."""

    def _clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        if isinstance(obj, (int, str, bool, type(None))):
            return obj
        return str(obj)

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_clean(results), f, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        pass  # Non-critical: skip JSON export on serialization error


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for ``python -m nllw.analysis``."""
    import argparse
    import glob as glob_mod

    parser = argparse.ArgumentParser(
        description="Pareto frontier analysis and reporting for NLLW experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m nllw.analysis --results experiments/\n"
            "  python -m nllw.analysis --results experiments/ --output docs/research/\n"
            "  python -m nllw.analysis --results results.json --pareto\n"
            "  python -m nllw.analysis --results experiments/ --table --sort comet\n"
            "  python -m nllw.analysis --results experiments/ --latex\n"
            "  python -m nllw.analysis --results experiments/ --edge-cases --corpus nllw/data/test_corpus.json\n"
        ),
    )
    parser.add_argument(
        "--results", required=True,
        help="Path to results JSON file or directory of JSON result files.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory for report (default: print to stdout).",
    )
    parser.add_argument(
        "--corpus", default=None,
        help="Path to test corpus JSON (for edge case analysis).",
    )
    parser.add_argument(
        "--quality", default="comet",
        help="Quality metric for Pareto analysis (default: comet).",
    )
    parser.add_argument(
        "--latency", default="ca_al_ms",
        help="Latency metric for Pareto analysis (default: ca_al_ms).",
    )
    parser.add_argument(
        "--sort", default=None,
        help="Sort table by this metric.",
    )
    parser.add_argument(
        "--pareto", action="store_true",
        help="Print Pareto frontier (ASCII plot).",
    )
    parser.add_argument(
        "--table", action="store_true",
        help="Print comparison table.",
    )
    parser.add_argument(
        "--latex", action="store_true",
        help="Print LaTeX table.",
    )
    parser.add_argument(
        "--edge-cases", action="store_true",
        help="Run edge case analysis.",
    )

    args = parser.parse_args()

    # Load results
    results = _load_results(args.results)
    if not results:
        print("No results found. Check --results path.")
        return

    print(f"Loaded {len(results)} experiment result(s).")

    # Load corpus if specified
    corpus: list[dict] = []
    if args.corpus:
        try:
            with open(args.corpus, "r", encoding="utf-8") as f:
                corpus = json.load(f)
            print(f"Loaded corpus: {len(corpus)} entries.")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: could not load corpus: {e}")

    # Generate full report
    if args.output:
        report_path = generate_report(
            results,
            corpus_name=os.path.basename(args.results),
            output_dir=args.output,
            corpus=corpus if corpus else None,
            quality_metric=args.quality,
            latency_metric=args.latency,
        )
        print(f"Report written to: {report_path}")
        return

    # Individual outputs
    any_output = False

    if args.table or (not args.pareto and not args.latex and not args.edge_cases):
        print()
        print(format_comparison_table(
            results,
            sort_by=args.sort or args.quality,
            pareto_quality=args.quality,
            pareto_latency=args.latency,
        ))
        any_output = True

    if args.pareto:
        print()
        print(format_pareto_ascii(
            results,
            quality_metric=args.quality,
            latency_metric=args.latency,
        ))
        any_output = True

    if args.latex:
        print()
        print(format_latex_table(
            results,
            pareto_quality=args.quality,
            pareto_latency=args.latency,
        ))
        any_output = True

    if args.edge_cases:
        traces = _extract_traces(results)
        if traces:
            edge = analyze_edge_cases(traces, corpus)
            print()
            print(f"Edge Case Analysis ({edge['total_sentences']} sentences, "
                  f"{edge['total_failures']} failures)")
            print()
            if edge["categories"]:
                print("Per-Category Breakdown:")
                print(_format_category_table(edge["categories"]))
                print()
            if edge["lang_pairs"]:
                print("Per-Language-Pair Breakdown:")
                print(_format_category_table(edge["lang_pairs"]))
                print()
            if edge["worst_sentences"]:
                print("Worst Performing Sentences:")
                for w in edge["worst_sentences"][:5]:
                    print(f"  {w['tag']:<24} commit={w['committed_ratio']:.1%}  "
                          f"{w['source'][:50]}")
                print()
            if edge["step_anomalies"]:
                print(f"Step Anomalies: {len(edge['step_anomalies'])} detected")
                for a in edge["step_anomalies"][:10]:
                    print(f"  [{a['type']}] {a['tag']}: {a['detail']}")
                print()
        else:
            print("No per-sentence traces found in results (needed for edge case analysis).")
        any_output = True


def _load_results(path: str) -> list[dict]:
    """Load experiment results from a JSON file or directory of JSON files."""
    import glob as glob_mod

    p = Path(path)

    if p.is_file():
        return _load_json_results(str(p))

    if p.is_dir():
        results: list[dict] = []
        patterns = [str(p / "*.json"), str(p / "**/*.json")]
        seen_files: set[str] = set()
        for pattern in patterns:
            for fpath in glob_mod.glob(pattern, recursive=True):
                real = os.path.realpath(fpath)
                if real in seen_files:
                    continue
                seen_files.add(real)
                loaded = _load_json_results(fpath)
                results.extend(loaded)
        return results

    return []


def _load_json_results(filepath: str) -> list[dict]:
    """Load a single JSON file, handling different result shapes.

    Supports:
    - List of result dicts (direct output of sweep or benchmark suite)
    - Dict with "configs" key (output of run_benchmark_suite)
    - Single result dict (wraps in list)
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        # run_benchmark_suite output
        if "configs" in data and isinstance(data["configs"], list):
            return data["configs"]
        # Single result
        return [data]

    return []


if __name__ == "__main__":
    main()
