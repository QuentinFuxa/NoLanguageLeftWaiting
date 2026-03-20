"""Cross-lingual alignment head transfer analysis.

Validates whether alignment heads detected for one language pair can be
reused for another. The ICLR 2026 "Translation Heads" paper suggests that
alignment heads are sparse, universal, and consistent across languages.

If confirmed, this eliminates the need for per-pair head detection (expensive:
requires FLORES + SimAlign + model inference for each new pair).

Analysis approach:
    1. Load head configs for the same model across different language pairs
    2. Compute overlap metrics between head sets:
       - Jaccard similarity: |intersection| / |union| of (layer, head) pairs
       - Rank correlation: Spearman correlation of TS scores
       - Top-K overlap: how many of the top-K heads are shared
    3. Estimate quality impact: if we use heads from pair A on pair B,
       how much TS score mass do we lose?

Usage:
    python -m nllw.head_transfer --model qwen3_4b
    python -m nllw.head_transfer --model qwen3_4b --top-k 10
    python -m nllw.head_transfer --all
"""

import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np

from .alignatt import load_head_config


def discover_configs(configs_dir: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """Discover all head configs, grouped by model.

    Returns:
        {model_key: {direction: file_path, ...}, ...}
    """
    if configs_dir is None:
        configs_dir = os.path.join(os.path.dirname(__file__), "heads", "configs")

    if not os.path.isdir(configs_dir):
        return {}

    model_configs = defaultdict(dict)

    for fname in sorted(os.listdir(configs_dir)):
        if not fname.endswith(".json"):
            continue

        path = os.path.join(configs_dir, fname)
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        model = data.get("model", "unknown")
        direction = data.get("language_pair", data.get("direction", "unknown"))

        # Normalize model name to a key
        model_key = model.lower().replace("/", "_").replace("-", "_")

        model_configs[model_key][direction] = path

    return dict(model_configs)


def head_set(config_path: str, top_k: int = 10) -> set:
    """Extract top-K (layer, head) pairs from a config file.

    Args:
        config_path: Path to head config JSON
        top_k: Number of top heads to consider

    Returns:
        Set of (layer, head) tuples
    """
    data = load_head_config(config_path)
    n = min(top_k, data["n_heads"])
    return {(data["layers"][i], data["heads"][i]) for i in range(n)}


def head_scores_dict(config_path: str) -> Dict[Tuple[int, int], float]:
    """Extract all (layer, head) -> TS score mappings.

    Args:
        config_path: Path to head config JSON

    Returns:
        Dict mapping (layer, head) to TS score
    """
    data = load_head_config(config_path)
    scores = {}
    for i in range(data["n_heads"]):
        key = (data["layers"][i], data["heads"][i])
        scores[key] = data["ts_scores"][i]
    return scores


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets.

    J(A, B) = |A intersect B| / |A union B|
    """
    if not set_a and not set_b:
        return 1.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def top_k_overlap(set_a: set, set_b: set) -> float:
    """Fraction of heads in set_a that also appear in set_b."""
    if not set_a:
        return 0.0
    return len(set_a & set_b) / len(set_a)


def ts_score_correlation(
    scores_a: Dict[Tuple[int, int], float],
    scores_b: Dict[Tuple[int, int], float],
) -> float:
    """Spearman rank correlation of TS scores for shared heads.

    Only computes over heads that appear in both configs.

    Returns:
        Correlation coefficient (-1 to 1), or 0.0 if fewer than 3 shared heads
    """
    shared_keys = set(scores_a.keys()) & set(scores_b.keys())
    if len(shared_keys) < 3:
        return 0.0

    vals_a = [scores_a[k] for k in sorted(shared_keys)]
    vals_b = [scores_b[k] for k in sorted(shared_keys)]

    # Spearman rank correlation
    ranks_a = np.argsort(np.argsort(vals_a)).astype(float)
    ranks_b = np.argsort(np.argsort(vals_b)).astype(float)

    n = len(ranks_a)
    d = ranks_a - ranks_b
    rho = 1 - (6 * np.sum(d ** 2)) / (n * (n ** 2 - 1))
    return float(rho)


def transferred_ts_mass(
    source_config: str,
    target_config: str,
    top_k: int = 10,
) -> float:
    """Estimate quality impact of using source heads on target language pair.

    Computes what fraction of the target pair's top-K TS mass is captured
    by the source pair's top-K heads.

    A value of 0.9 means using source heads captures 90% of the target
    pair's alignment quality.

    Args:
        source_config: Path to source head config (what we'd reuse)
        target_config: Path to target head config (ground truth)
        top_k: Number of heads to consider

    Returns:
        Fraction of target TS mass captured (0.0 to 1.0)
    """
    source_heads = head_set(source_config, top_k)
    target_scores = head_scores_dict(target_config)

    # Total TS mass of target's top-K
    target_data = load_head_config(target_config)
    n = min(top_k, target_data["n_heads"])
    total_target_mass = sum(target_data["ts_scores"][:n])

    if total_target_mass <= 0:
        return 0.0

    # TS mass of target heads that overlap with source
    captured_mass = sum(
        target_scores.get(h, 0.0) for h in source_heads
    )

    return min(1.0, captured_mass / total_target_mass)


def analyze_model_transfer(
    model_configs: Dict[str, str],
    top_k: int = 10,
) -> List[Dict]:
    """Analyze cross-lingual head transfer for a single model.

    Args:
        model_configs: {direction: config_path} for one model
        top_k: Number of top heads to consider

    Returns:
        List of pairwise analysis results
    """
    directions = sorted(model_configs.keys())
    if len(directions) < 2:
        return []

    results = []
    for i, dir_a in enumerate(directions):
        for dir_b in directions[i + 1:]:
            path_a = model_configs[dir_a]
            path_b = model_configs[dir_b]

            set_a = head_set(path_a, top_k)
            set_b = head_set(path_b, top_k)

            scores_a = head_scores_dict(path_a)
            scores_b = head_scores_dict(path_b)

            result = {
                "pair_a": dir_a,
                "pair_b": dir_b,
                "top_k": top_k,
                "jaccard": jaccard_similarity(set_a, set_b),
                "overlap_a_in_b": top_k_overlap(set_a, set_b),
                "overlap_b_in_a": top_k_overlap(set_b, set_a),
                "ts_correlation": ts_score_correlation(scores_a, scores_b),
                "ts_mass_a_on_b": transferred_ts_mass(path_a, path_b, top_k),
                "ts_mass_b_on_a": transferred_ts_mass(path_b, path_a, top_k),
                "shared_heads": sorted(set_a & set_b),
                "only_a": sorted(set_a - set_b),
                "only_b": sorted(set_b - set_a),
            }
            results.append(result)

    return results


def print_transfer_report(
    model_key: str,
    model_configs: Dict[str, str],
    top_k: int = 10,
):
    """Print a formatted cross-lingual transfer report for one model."""
    results = analyze_model_transfer(model_configs, top_k)
    if not results:
        print(f"  {model_key}: need >= 2 language pairs for transfer analysis")
        return

    print(f"\n{'='*70}")
    print(f"  Model: {model_key} | top_k={top_k} | {len(model_configs)} directions")
    print(f"  Directions: {', '.join(sorted(model_configs.keys()))}")
    print(f"{'='*70}")

    header = (
        f"{'Pair A':>8} -> {'Pair B':<8}  "
        f"{'Jaccard':>7}  {'Overlap':>7}  {'TS Corr':>7}  {'TS Mass':>7}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['pair_a']:>8} -> {r['pair_b']:<8}  "
            f"{r['jaccard']:>7.3f}  "
            f"{r['overlap_a_in_b']:>7.3f}  "
            f"{r['ts_correlation']:>7.3f}  "
            f"{r['ts_mass_a_on_b']:>7.3f}"
        )
        print(
            f"{r['pair_b']:>8} -> {r['pair_a']:<8}  "
            f"{'':>7}  "
            f"{r['overlap_b_in_a']:>7.3f}  "
            f"{'':>7}  "
            f"{r['ts_mass_b_on_a']:>7.3f}"
        )

    # Summary
    jaccards = [r["jaccard"] for r in results]
    overlaps = [r["overlap_a_in_b"] for r in results] + [r["overlap_b_in_a"] for r in results]
    ts_masses = [r["ts_mass_a_on_b"] for r in results] + [r["ts_mass_b_on_a"] for r in results]

    print(f"\n  Summary:")
    print(f"    Mean Jaccard:     {np.mean(jaccards):.3f} (1.0 = identical head sets)")
    print(f"    Mean Overlap:     {np.mean(overlaps):.3f} (1.0 = full reuse)")
    print(f"    Mean TS Mass:     {np.mean(ts_masses):.3f} (1.0 = no quality loss)")
    print(f"    Min TS Mass:      {min(ts_masses):.3f} (worst-case transfer)")

    # Verdict
    mean_mass = np.mean(ts_masses)
    if mean_mass >= 0.9:
        verdict = "EXCELLENT: heads are highly transferable (>90% TS mass)"
    elif mean_mass >= 0.7:
        verdict = "GOOD: heads are moderately transferable (>70% TS mass)"
    elif mean_mass >= 0.5:
        verdict = "FAIR: partial transfer possible, per-pair detection recommended"
    else:
        verdict = "POOR: heads are language-pair-specific, no transfer"

    print(f"    Verdict:          {verdict}")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Cross-lingual alignment head transfer analysis"
    )
    parser.add_argument("--model", help="Model key to analyze (e.g., qwen3_4b)")
    parser.add_argument("--all", action="store_true", help="Analyze all models")
    parser.add_argument("--top-k", type=int, default=10, help="Top K heads to compare")
    parser.add_argument("--configs-dir", help="Override configs directory")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    all_configs = discover_configs(args.configs_dir)

    if not all_configs:
        print("No head configs found.", file=sys.stderr)
        sys.exit(1)

    if args.model:
        # Find matching model
        matches = {k: v for k, v in all_configs.items() if args.model.lower() in k}
        if not matches:
            print(f"No configs found for model '{args.model}'.", file=sys.stderr)
            print(f"Available models: {', '.join(sorted(all_configs.keys()))}", file=sys.stderr)
            sys.exit(1)
        target_configs = matches
    elif args.all:
        target_configs = all_configs
    else:
        # Default: analyze all models with >= 2 directions
        target_configs = {k: v for k, v in all_configs.items() if len(v) >= 2}

    if args.json:
        all_results = {}
        for model_key, model_cfgs in sorted(target_configs.items()):
            results = analyze_model_transfer(model_cfgs, args.top_k)
            if results:
                all_results[model_key] = results
        print(json.dumps(all_results, indent=2, default=str))
    else:
        print("Cross-Lingual Alignment Head Transfer Analysis")
        print(f"(Top-K = {args.top_k})")
        for model_key in sorted(target_configs.keys()):
            print_transfer_report(model_key, target_configs[model_key], args.top_k)


if __name__ == "__main__":
    main()
