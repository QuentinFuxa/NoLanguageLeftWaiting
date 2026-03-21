"""Fusion weight calibration pipeline for SimulMT border detection.

Bridges the gap between the fusion framework (fusion.py) and optimal performance
by providing tools to:
    1. Collect signal traces during translation (SignalTrace)
    2. Generate labeled border decisions from reference translations (label_borders)
    3. Optimize fusion weights via grid search on labeled data (calibrate_direction)
    4. Export optimized weights per direction (run_calibration)

The pipeline works in two phases:
    Phase 1 (GPU): Run translations with trace collection enabled.
        For each sentence, record all signal scores at every generation step.
    Phase 2 (CPU): Load traces, label border decisions, optimize weights.
        Uses word alignment heuristics to determine ground-truth borders.

Usage:
    # Phase 1: Collect traces (on GPU with model loaded)
    collector = TraceCollector()
    for sentence in corpus:
        backend.translate(word, trace_callback=collector.record_step)
    traces = collector.get_traces()

    # Phase 2: Calibrate (can run anywhere)
    optimal_weights, threshold, f1 = calibrate_direction(
        traces, direction="en-zh"
    )

    # CLI: Full pipeline
    python -m nllw.calibrate --traces traces.json --direction en-zh
    python -m nllw.calibrate --traces traces.json --all-directions --output weights.json

Novel: No published work on data-driven fusion weight calibration for SimulMT
border detection. Closest: AliBaStr-MT (Apple) trains a separate classifier,
but uses 6M params. Our approach optimizes weights over existing signals (zero
extra params, zero extra compute at inference).
"""

import json
import math
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from nllw.fusion import (
    FusionWeights,
    FusionDiagnostic,
    FusionConfig,
    calibrate_threshold,
    grid_search_weights,
    get_fusion_weights,
    DIRECTION_WEIGHTS,
)


# ---------------------------------------------------------------------------
# Signal trace data structures
# ---------------------------------------------------------------------------

@dataclass
class SignalSnapshot:
    """Signal scores at a single generation step.

    Records the raw scores from all active signals, plus context about
    what the model was doing at this step. Used as training data for
    fusion weight calibration.
    """
    # Step context
    step_idx: int = 0                   # Generation step within translate() call
    source_words_seen: int = 0          # Number of source words read so far
    tokens_generated: int = 0           # Tokens generated so far in this step
    # Signal scores (from fusion scorers, range [-1, +1])
    scores: Dict[str, float] = field(default_factory=dict)
    # Border decision that was actually made
    actual_decision: bool = False       # True = WRITE (border hit)
    # Fusion diagnostic (if fusion was enabled)
    fusion_score: Optional[float] = None
    fusion_threshold: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SignalSnapshot":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


@dataclass
class SentenceTrace:
    """Complete signal trace for one sentence translation.

    Contains all snapshots from translating a single sentence, plus
    metadata about the source and output quality.
    """
    sentence_id: int = 0
    direction: str = "en-zh"
    source_text: str = ""
    reference_text: str = ""
    output_text: str = ""
    n_source_words: int = 0
    # Quality metrics (if available)
    bleu: Optional[float] = None
    comet: Optional[float] = None
    # All signal snapshots during translation
    snapshots: List[SignalSnapshot] = field(default_factory=list)
    # Border decisions timeline: (source_pos, was_write, emitted_text)
    border_timeline: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["snapshots"] = [s.to_dict() for s in self.snapshots]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SentenceTrace":
        snapshots = [SignalSnapshot.from_dict(s) for s in d.get("snapshots", [])]
        d = {k: v for k, v in d.items() if k != "snapshots"}
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        obj = cls(**{k: v for k, v in d.items() if k in valid_keys})
        obj.snapshots = snapshots
        return obj


# ---------------------------------------------------------------------------
# Trace collector (integrates with backend)
# ---------------------------------------------------------------------------

class TraceCollector:
    """Collects signal traces during translation.

    Designed to be wired into a backend via callback. The backend calls
    record_step() at each generation step with the current signal scores.

    Usage:
        collector = TraceCollector()
        collector.start_sentence(sentence_id=0, source="Hello world",
                                 reference="Bonjour le monde", direction="en-fr")

        # During translation (called by backend):
        collector.record_step(
            source_words_seen=1,
            tokens_generated=3,
            scores={"standard": 0.5, "shift_k": 0.2},
            actual_decision=False,
        )

        collector.end_sentence(output_text="Bonjour")
        traces = collector.get_traces()
    """

    def __init__(self):
        self._traces: List[SentenceTrace] = []
        self._current: Optional[SentenceTrace] = None
        self._step_idx = 0

    def start_sentence(
        self,
        sentence_id: int,
        source: str,
        reference: str = "",
        direction: str = "en-zh",
    ):
        """Begin tracing a new sentence translation."""
        if self._current is not None:
            self._traces.append(self._current)
        self._current = SentenceTrace(
            sentence_id=sentence_id,
            direction=direction,
            source_text=source,
            reference_text=reference,
            n_source_words=len(source.split()),
        )
        self._step_idx = 0

    def record_step(
        self,
        source_words_seen: int,
        tokens_generated: int,
        scores: Dict[str, float],
        actual_decision: bool,
        fusion_score: Optional[float] = None,
        fusion_threshold: Optional[float] = None,
    ):
        """Record signal scores at a generation step.

        Called by the backend during the generation loop, after computing
        all signal scores and making the border decision.
        """
        if self._current is None:
            return

        snapshot = SignalSnapshot(
            step_idx=self._step_idx,
            source_words_seen=source_words_seen,
            tokens_generated=tokens_generated,
            scores=dict(scores),
            actual_decision=actual_decision,
            fusion_score=fusion_score,
            fusion_threshold=fusion_threshold,
        )
        self._current.snapshots.append(snapshot)
        self._step_idx += 1

    def record_border_event(
        self,
        source_pos: int,
        was_write: bool,
        emitted_text: str = "",
    ):
        """Record a border decision event."""
        if self._current is None:
            return
        self._current.border_timeline.append({
            "source_pos": source_pos,
            "was_write": was_write,
            "emitted_text": emitted_text,
        })

    def end_sentence(
        self,
        output_text: str = "",
        bleu: Optional[float] = None,
        comet: Optional[float] = None,
    ):
        """Finalize the current sentence trace."""
        if self._current is not None:
            self._current.output_text = output_text
            self._current.bleu = bleu
            self._current.comet = comet
            self._traces.append(self._current)
            self._current = None

    def get_traces(self) -> List[SentenceTrace]:
        """Get all collected traces (finalizes any in-progress sentence)."""
        if self._current is not None:
            self._traces.append(self._current)
            self._current = None
        return list(self._traces)

    def clear(self):
        """Reset the collector."""
        self._traces = []
        self._current = None
        self._step_idx = 0


# ---------------------------------------------------------------------------
# Border labeling from reference translations
# ---------------------------------------------------------------------------

def _monotonic_alignment(n_source: int, n_target: int) -> List[int]:
    """Generate a monotonic word alignment.

    Maps each target word to the source word it most likely aligns to,
    assuming roughly monotonic translation. This is a simple heuristic
    that works well for similar-order language pairs (en-fr, en-de, en-it).

    For reordering pairs (en-zh, en-ja), the alignment is less accurate
    but still provides useful signal for calibration.

    Args:
        n_source: Number of source words
        n_target: Number of target words

    Returns:
        List of source positions (one per target word).
    """
    if n_target == 0 or n_source == 0:
        return []

    alignment = []
    for j in range(n_target):
        # Linear interpolation: target word j aligns to source position
        # approximately at (j / n_target) * n_source
        src_pos = int((j + 0.5) * n_source / n_target)
        src_pos = min(src_pos, n_source - 1)
        alignment.append(src_pos)

    return alignment


def _reorder_aware_alignment(
    n_source: int,
    n_target: int,
    direction: str = "en-zh",
) -> List[int]:
    """Generate alignment with reordering awareness.

    For language pairs with significant reordering (en-zh, en-ja),
    uses a more conservative alignment that accounts for the possibility
    that later target words may depend on earlier source words.

    The key insight: for reordering pairs, we should be MORE conservative
    about when to emit (higher source position requirements).
    """
    base_alignment = _monotonic_alignment(n_source, n_target)

    if not base_alignment:
        return base_alignment

    # For reordering pairs, shift alignment forward (require more source)
    reordering_pairs = {"en-zh", "en-ja", "en-ko", "en-ar", "en-he"}
    direction_key = direction.lower().replace("_", "-")

    if direction_key in reordering_pairs:
        # Add a lookahead buffer: each target word may depend on source
        # words 1-2 positions ahead of the monotonic estimate
        lookahead = max(1, n_source // (n_target + 1))
        adjusted = []
        for pos in base_alignment:
            adjusted_pos = min(pos + lookahead, n_source - 1)
            adjusted.append(adjusted_pos)
        return adjusted

    return base_alignment


def label_borders_from_alignment(
    n_source: int,
    reference: str,
    direction: str = "en-zh",
    border_distance: int = 3,
) -> List[Dict[str, Any]]:
    """Generate labeled border decisions from reference alignment.

    For each source position, determines whether a WRITE decision would
    be correct based on word alignment with the reference.

    The labeling logic:
        - At source position i, target words whose max aligned source
          position <= i are "safe" to emit
        - If there are new safe target words → should_write = True
        - Otherwise → should_write = False

    Args:
        n_source: Number of source words
        reference: Reference translation text
        direction: Translation direction
        border_distance: Border distance parameter (affects labeling)

    Returns:
        List of labeled examples, one per source position:
        [{"source_pos": int, "should_write": bool, "n_safe_targets": int}]
    """
    ref_words = reference.split()
    n_target = len(ref_words)

    if n_source == 0 or n_target == 0:
        return []

    # Get alignment
    alignment = _reorder_aware_alignment(n_source, n_target, direction)

    # For each source position, count how many target words are "safe"
    labels = []
    prev_safe = 0

    for src_pos in range(n_source):
        # Target words that are safe to emit at this source position
        n_safe = sum(1 for a in alignment if a <= src_pos)

        # Should we write? Yes if there are new safe targets since last check
        has_new_safe = n_safe > prev_safe

        # But also consider border_distance: if we're far from the end of
        # the safe region, we should wait (be conservative)
        # This mimics what AlignAtt would do with this border_distance
        is_near_border = (src_pos >= n_source - border_distance)

        labels.append({
            "source_pos": src_pos,
            "should_write": has_new_safe or is_near_border,
            "n_safe_targets": n_safe,
            "n_new_safe": n_safe - prev_safe,
        })

        if has_new_safe:
            prev_safe = n_safe

    return labels


def label_traces_from_quality(
    traces: List[SentenceTrace],
    quality_threshold: float = 0.5,
) -> List[Dict]:
    """Label trace snapshots using output quality as ground truth.

    Simpler approach than alignment-based labeling: uses the final
    translation quality (BLEU/COMET) to determine if border decisions
    were correct.

    Heuristic:
        - For high-quality translations (COMET > threshold):
          the actual decisions were probably good → use them as labels
        - For low-quality translations:
          the actual decisions were probably bad → flip them

    This is noisy but requires no alignment computation.

    Args:
        traces: List of sentence traces with quality metrics
        quality_threshold: COMET score threshold for "good" translations

    Returns:
        Labeled examples for grid_search_weights()
    """
    examples = []

    for trace in traces:
        quality = trace.comet if trace.comet is not None else trace.bleu
        if quality is None:
            continue

        is_good = quality > quality_threshold

        for snapshot in trace.snapshots:
            if not snapshot.scores:
                continue

            # For good translations, actual decisions are correct
            # For bad translations, actual decisions are incorrect
            should_write = snapshot.actual_decision if is_good else not snapshot.actual_decision

            examples.append({
                "scores": dict(snapshot.scores),
                "should_write": should_write,
                "quality": quality,
                "sentence_id": trace.sentence_id,
            })

    return examples


def label_traces_from_timeline(
    traces: List[SentenceTrace],
    border_distance: int = 3,
) -> List[Dict]:
    """Label trace snapshots using alignment-based border timeline.

    Combines signal scores from traces with alignment-based labels.
    For each sentence, generates alignment labels and matches them
    with the signal snapshots.

    Args:
        traces: Sentence traces with snapshots containing signal scores
        border_distance: Border distance for alignment labeling

    Returns:
        Labeled examples compatible with grid_search_weights()
    """
    examples = []

    for trace in traces:
        if not trace.reference_text or not trace.snapshots:
            continue

        # Generate alignment-based labels
        labels = label_borders_from_alignment(
            n_source=trace.n_source_words,
            reference=trace.reference_text,
            direction=trace.direction,
            border_distance=border_distance,
        )

        # Build label lookup by source position
        label_by_pos = {l["source_pos"]: l for l in labels}

        for snapshot in trace.snapshots:
            if not snapshot.scores:
                continue

            # Find the label for this snapshot's source position
            src_pos = snapshot.source_words_seen - 1  # 0-indexed
            label = label_by_pos.get(src_pos)
            if label is None:
                continue

            examples.append({
                "scores": dict(snapshot.scores),
                "should_write": label["should_write"],
                "source_pos": src_pos,
                "sentence_id": trace.sentence_id,
            })

    return examples


# ---------------------------------------------------------------------------
# Calibration pipeline
# ---------------------------------------------------------------------------

def calibrate_direction(
    traces: List[SentenceTrace],
    direction: str = "en-zh",
    border_distance: int = 3,
    method: str = "alignment",
    quality_threshold: float = 0.5,
    weight_grid: Optional[Dict[str, List[float]]] = None,
) -> Tuple[FusionWeights, float, float]:
    """Calibrate fusion weights for a specific direction.

    Runs the full calibration pipeline:
    1. Filter traces for this direction
    2. Generate labeled examples (alignment or quality-based)
    3. Grid search over weight combinations
    4. Return optimal weights + threshold + metric score

    Args:
        traces: All collected traces
        direction: Direction to calibrate (e.g., "en-zh")
        border_distance: Border distance for alignment labeling
        method: Labeling method ("alignment" or "quality")
        quality_threshold: COMET threshold for quality-based labeling
        weight_grid: Custom weight grid for search (None = defaults)

    Returns:
        (optimal_weights, optimal_threshold, best_f1)
    """
    # Filter traces for this direction
    dir_traces = [t for t in traces if t.direction == direction]
    if not dir_traces:
        return get_fusion_weights(direction), 0.0, 0.0

    # Generate labeled examples
    if method == "alignment":
        examples = label_traces_from_timeline(dir_traces, border_distance)
    elif method == "quality":
        examples = label_traces_from_quality(dir_traces, quality_threshold)
    else:
        raise ValueError(f"Unknown labeling method: {method}")

    if not examples:
        return get_fusion_weights(direction), 0.0, 0.0

    # Run grid search
    best_weights, best_threshold, best_f1 = grid_search_weights(
        examples, weight_grid=weight_grid
    )

    return best_weights, best_threshold, best_f1


def run_calibration(
    traces: List[SentenceTrace],
    directions: Optional[List[str]] = None,
    border_distance: int = 3,
    method: str = "alignment",
    quality_threshold: float = 0.5,
    weight_grid: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run calibration for multiple directions.

    Args:
        traces: All collected traces
        directions: List of directions to calibrate (None = auto-detect)
        border_distance: Border distance for alignment labeling
        method: Labeling method ("alignment" or "quality")
        quality_threshold: COMET threshold for quality-based labeling
        weight_grid: Custom weight grid for search

    Returns:
        Dict mapping direction -> {"weights": FusionWeights, "threshold": float, "f1": float}
    """
    if directions is None:
        # Auto-detect directions from traces
        directions = sorted(set(t.direction for t in traces))

    results = {}
    for direction in directions:
        weights, threshold, f1 = calibrate_direction(
            traces, direction, border_distance, method,
            quality_threshold, weight_grid,
        )
        results[direction] = {
            "weights": weights,
            "threshold": threshold,
            "f1": f1,
            "n_traces": sum(1 for t in traces if t.direction == direction),
            "n_examples": sum(
                len(t.snapshots) for t in traces if t.direction == direction
            ),
        }

    return results


# ---------------------------------------------------------------------------
# Synthetic trace generation (for testing without GPU)
# ---------------------------------------------------------------------------

def generate_synthetic_traces(
    n_sentences: int = 50,
    direction: str = "en-zh",
    n_source_words: int = 12,
    border_distance: int = 3,
    seed: int = 42,
) -> List[SentenceTrace]:
    """Generate synthetic traces for testing the calibration pipeline.

    Creates realistic-looking signal traces without running an actual
    translation model. Useful for:
    - Unit testing the calibration pipeline
    - Prototyping weight optimization strategies
    - Validating the labeling heuristics

    The synthetic signals model the expected behavior:
    - Standard border score increases as we approach the end of source
    - Coverage score correlates with how much source we've read
    - Entropy change is large after first few words, then diminishes
    - Shift-k correlates with standard border but with noise

    Args:
        n_sentences: Number of synthetic sentences
        direction: Translation direction
        n_source_words: Average source sentence length
        border_distance: Border distance parameter
        seed: Random seed for reproducibility

    Returns:
        List of SentenceTrace with synthetic snapshots
    """
    rng = np.random.RandomState(seed)
    traces = []

    for sent_idx in range(n_sentences):
        # Vary sentence length around the mean
        n_src = max(3, int(rng.normal(n_source_words, 3)))
        n_tgt = max(2, int(n_src * rng.uniform(0.8, 1.4)))

        # Generate fake reference
        ref_words = [f"w{j}" for j in range(n_tgt)]
        reference = " ".join(ref_words)

        trace = SentenceTrace(
            sentence_id=sent_idx,
            direction=direction,
            source_text=" ".join([f"s{i}" for i in range(n_src)]),
            reference_text=reference,
            n_source_words=n_src,
        )

        # Generate snapshots: one per source word, simulating the
        # generation loop where we read words and check borders
        for src_pos in range(n_src):
            progress = (src_pos + 1) / n_src  # 0 to 1

            # Standard border: increases with progress
            border_threshold = n_src - border_distance
            if border_threshold > 0:
                standard_score = (src_pos - border_threshold) / max(border_distance, 1)
                standard_score = np.clip(standard_score + rng.normal(0, 0.15), -1, 1)
            else:
                standard_score = 0.0

            # Shift-k: correlated with standard but noisier
            shift_k_score = standard_score * 0.8 + rng.normal(0, 0.2)
            shift_k_score = np.clip(shift_k_score, -1, 1)

            # Info gain: large early (model learning), small late (exhausted)
            info_gain_base = -0.5 + progress
            info_gain_score = np.clip(info_gain_base + rng.normal(0, 0.2), -1, 1)

            # Coverage: starts high, may drop if hallucinating
            coverage_base = 0.3 - 0.2 * progress
            if rng.random() < 0.05:  # 5% chance of hallucination
                coverage_base = -0.8
            coverage_score = np.clip(coverage_base + rng.normal(0, 0.15), -1, 1)

            # Monotonicity: generally positive, occasionally drops
            mono_base = 0.2 if progress > 0.3 else -0.3
            mono_score = np.clip(mono_base + rng.normal(0, 0.2), -1, 1)

            # Entropy change: large negative early, small late
            entropy_base = -0.3 + 0.6 * progress
            entropy_score = np.clip(entropy_base + rng.normal(0, 0.25), -1, 1)

            # Prediction stability: unstable early, stable late
            pred_base = -0.5 + progress
            pred_score = np.clip(pred_base + rng.normal(0, 0.2), -1, 1)

            # Attention shift: large early (consuming), small late (stuck)
            shift_base = 0.3 - 0.5 * progress
            attn_shift_score = np.clip(shift_base + rng.normal(0, 0.2), -1, 1)

            scores = {
                "standard": float(standard_score),
                "shift_k": float(shift_k_score),
                "info_gain": float(info_gain_score),
                "coverage": float(coverage_score),
                "monotonicity": float(mono_score),
                "entropy_change": float(entropy_score),
                "pred_stability": float(pred_score),
                "attn_shift": float(attn_shift_score),
            }

            # Simulate border decision (based on standard + noise)
            actual_decision = standard_score > 0 and rng.random() > 0.3

            snapshot = SignalSnapshot(
                step_idx=src_pos,
                source_words_seen=src_pos + 1,
                tokens_generated=max(0, int(progress * n_tgt * 0.5)),
                scores=scores,
                actual_decision=actual_decision,
            )
            trace.snapshots.append(snapshot)

        trace.output_text = " ".join(ref_words[:int(n_tgt * 0.8)])
        traces.append(trace)

    return traces


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_traces(traces: List[SentenceTrace], path: str):
    """Save traces to a JSON file."""
    data = [t.to_dict() for t in traces]
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=_NumpyEncoder)


def load_traces(path: str) -> List[SentenceTrace]:
    """Load traces from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return [SentenceTrace.from_dict(d) for d in data]


def export_weights(
    results: Dict[str, Dict[str, Any]],
    path: str,
):
    """Export calibrated weights to a JSON file.

    The output format can be loaded directly into FusionConfig.

    Args:
        results: Output of run_calibration()
        path: Output file path
    """
    output = {}
    for direction, result in results.items():
        weights = result["weights"]
        output[direction] = {
            "weights": weights.as_dict(),
            "threshold": result["threshold"],
            "f1": result["f1"],
            "n_traces": result["n_traces"],
            "n_examples": result["n_examples"],
        }

    with open(path, "w") as f:
        json.dump(output, f, indent=2)


def load_calibrated_weights(path: str) -> Dict[str, Tuple[FusionWeights, float]]:
    """Load calibrated weights from a JSON file.

    Returns:
        Dict mapping direction -> (FusionWeights, threshold)
    """
    with open(path) as f:
        data = json.load(f)

    result = {}
    for direction, d in data.items():
        weights = FusionWeights.from_dict(d["weights"])
        threshold = d.get("threshold", 0.0)
        result[direction] = (weights, threshold)

    return result


# ---------------------------------------------------------------------------
# Analysis utilities
# ---------------------------------------------------------------------------

def analyze_signal_importance(
    traces: List[SentenceTrace],
    direction: str = "en-zh",
    border_distance: int = 3,
) -> Dict[str, Dict[str, float]]:
    """Analyze the importance of each signal for border detection.

    Computes per-signal statistics:
    - Mean score when should_write=True vs False
    - Discriminative power (score difference between WRITE and READ cases)
    - Correlation with ground truth

    Useful for understanding which signals matter most for a direction,
    and for pruning signals that don't contribute.

    Args:
        traces: Sentence traces
        direction: Direction to analyze
        border_distance: Border distance for labeling

    Returns:
        Dict mapping signal name -> stats dict
    """
    dir_traces = [t for t in traces if t.direction == direction]
    examples = label_traces_from_timeline(dir_traces, border_distance)

    if not examples:
        return {}

    # Collect scores by signal and label
    signal_scores: Dict[str, Dict[str, List[float]]] = {}

    for ex in examples:
        label = "write" if ex["should_write"] else "read"
        for signal, score in ex["scores"].items():
            if signal not in signal_scores:
                signal_scores[signal] = {"write": [], "read": []}
            signal_scores[signal][label].append(score)

    # Compute statistics
    results = {}
    for signal, scores_by_label in signal_scores.items():
        write_scores = scores_by_label["write"]
        read_scores = scores_by_label["read"]

        mean_write = np.mean(write_scores) if write_scores else 0.0
        mean_read = np.mean(read_scores) if read_scores else 0.0
        discriminative_power = float(mean_write - mean_read)

        # Correlation with ground truth (point-biserial)
        all_scores = write_scores + read_scores
        all_labels = [1.0] * len(write_scores) + [0.0] * len(read_scores)
        if len(set(all_labels)) > 1 and len(all_scores) > 1:
            correlation = float(np.corrcoef(all_scores, all_labels)[0, 1])
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0

        results[signal] = {
            "mean_write": float(mean_write),
            "mean_read": float(mean_read),
            "discriminative_power": discriminative_power,
            "correlation": correlation,
            "n_write": len(write_scores),
            "n_read": len(read_scores),
        }

    # Sort by discriminative power
    results = dict(sorted(
        results.items(),
        key=lambda x: abs(x[1]["discriminative_power"]),
        reverse=True,
    ))

    return results


def print_calibration_report(
    results: Dict[str, Dict[str, Any]],
    baseline_weights: Optional[Dict[str, FusionWeights]] = None,
):
    """Print a human-readable calibration report.

    Args:
        results: Output of run_calibration()
        baseline_weights: Optional baseline weights for comparison
    """
    if baseline_weights is None:
        baseline_weights = DIRECTION_WEIGHTS

    print("=" * 70)
    print("FUSION WEIGHT CALIBRATION REPORT")
    print("=" * 70)

    for direction, result in results.items():
        weights = result["weights"]
        threshold = result["threshold"]
        f1 = result["f1"]
        n_traces = result["n_traces"]
        n_examples = result["n_examples"]

        print(f"\n--- {direction} ---")
        print(f"  Sentences: {n_traces}, Examples: {n_examples}")
        print(f"  Best F1: {f1:.3f}")
        print(f"  Optimal threshold: {threshold:.3f}")
        print(f"  Calibrated weights:")

        baseline = baseline_weights.get(direction, FusionWeights())
        baseline_d = baseline.as_dict()
        weights_d = weights.as_dict()

        for signal in sorted(weights_d.keys()):
            w = weights_d[signal]
            bw = baseline_d.get(signal, 0.0)
            delta = w - bw
            delta_str = f" ({delta:+.2f})" if abs(delta) > 0.01 else ""
            print(f"    {signal:20s}: {w:.2f}{delta_str}")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """CLI entry point for calibration.

    Usage:
        # Run calibration on collected traces
        python -m nllw.calibrate --traces traces.json --direction en-zh

        # All directions
        python -m nllw.calibrate --traces traces.json --all-directions

        # Export weights
        python -m nllw.calibrate --traces traces.json --all-directions --output weights.json

        # Demo with synthetic data
        python -m nllw.calibrate --demo --direction en-zh

        # Analyze signal importance
        python -m nllw.calibrate --traces traces.json --analyze --direction en-zh
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Calibrate fusion weights for SimulMT border detection"
    )
    parser.add_argument("--traces", type=str, help="Path to traces JSON file")
    parser.add_argument("--direction", type=str, default="en-zh",
                        help="Direction to calibrate (default: en-zh)")
    parser.add_argument("--all-directions", action="store_true",
                        help="Calibrate all directions found in traces")
    parser.add_argument("--output", "-o", type=str,
                        help="Output path for calibrated weights JSON")
    parser.add_argument("--method", type=str, default="alignment",
                        choices=["alignment", "quality"],
                        help="Labeling method (default: alignment)")
    parser.add_argument("--border-distance", type=int, default=3,
                        help="Border distance for alignment labeling")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo with synthetic data")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze signal importance (no calibration)")
    parser.add_argument("--n-synthetic", type=int, default=100,
                        help="Number of synthetic sentences for demo")

    args = parser.parse_args()

    if args.demo:
        print("Generating synthetic traces...")
        directions = ["en-zh", "en-de", "en-it", "cs-en", "en-fr"]
        all_traces = []
        for d in directions:
            traces = generate_synthetic_traces(
                n_sentences=args.n_synthetic, direction=d
            )
            all_traces.extend(traces)
            print(f"  {d}: {len(traces)} sentences, "
                  f"{sum(len(t.snapshots) for t in traces)} snapshots")

        if args.analyze:
            for d in directions:
                print(f"\n--- Signal importance: {d} ---")
                importance = analyze_signal_importance(
                    all_traces, d, args.border_distance
                )
                for signal, stats in importance.items():
                    print(f"  {signal:20s}: discriminative={stats['discriminative_power']:+.3f} "
                          f"corr={stats['correlation']:+.3f}")
            return

        print("\nRunning calibration...")
        results = run_calibration(
            all_traces,
            directions=directions,
            border_distance=args.border_distance,
            method=args.method,
        )
        print_calibration_report(results)

        if args.output:
            export_weights(results, args.output)
            print(f"\nWeights saved to {args.output}")

        return

    if not args.traces:
        parser.error("--traces is required (or use --demo)")

    traces = load_traces(args.traces)
    print(f"Loaded {len(traces)} traces from {args.traces}")

    if args.analyze:
        dirs = sorted(set(t.direction for t in traces)) if args.all_directions else [args.direction]
        for d in dirs:
            print(f"\n--- Signal importance: {d} ---")
            importance = analyze_signal_importance(traces, d, args.border_distance)
            for signal, stats in importance.items():
                print(f"  {signal:20s}: discriminative={stats['discriminative_power']:+.3f} "
                      f"corr={stats['correlation']:+.3f} "
                      f"(W:{stats['n_write']}, R:{stats['n_read']})")
        return

    if args.all_directions:
        directions = None  # auto-detect
    else:
        directions = [args.direction]

    results = run_calibration(
        traces,
        directions=directions,
        border_distance=args.border_distance,
        method=args.method,
    )
    print_calibration_report(results)

    if args.output:
        export_weights(results, args.output)
        print(f"\nWeights saved to {args.output}")


if __name__ == "__main__":
    main()
