"""Tests for nllw/calibrate.py -- Fusion weight calibration pipeline."""

import json
import os
import sys
import tempfile

import numpy as np
import pytest

from nllw.calibrate import (
    SignalSnapshot,
    SentenceTrace,
    TraceCollector,
    calibrate_direction,
    run_calibration,
    generate_synthetic_traces,
    save_traces,
    load_traces,
    export_weights,
    load_calibrated_weights,
    analyze_signal_importance,
    label_borders_from_alignment,
    label_traces_from_quality,
    label_traces_from_timeline,
    _monotonic_alignment,
    _reorder_aware_alignment,
)
from nllw.fusion import FusionWeights


# ---------------------------------------------------------------------------
# SignalSnapshot tests
# ---------------------------------------------------------------------------

class TestSignalSnapshot:
    def test_default_values(self):
        snap = SignalSnapshot()
        assert snap.step_idx == 0
        assert snap.source_words_seen == 0
        assert snap.scores == {}
        assert snap.actual_decision is False

    def test_with_scores(self):
        snap = SignalSnapshot(
            step_idx=3,
            source_words_seen=5,
            tokens_generated=7,
            scores={"standard": 0.5, "shift_k": -0.2},
            actual_decision=True,
        )
        assert snap.scores["standard"] == 0.5
        assert snap.actual_decision is True

    def test_to_dict_roundtrip(self):
        snap = SignalSnapshot(
            step_idx=2,
            scores={"standard": 0.8, "coverage": -0.3},
            actual_decision=True,
            fusion_score=0.45,
        )
        d = snap.to_dict()
        snap2 = SignalSnapshot.from_dict(d)
        assert snap2.step_idx == 2
        assert snap2.scores == snap.scores
        assert snap2.actual_decision is True
        assert snap2.fusion_score == 0.45

    def test_from_dict_ignores_unknown_keys(self):
        d = {"step_idx": 1, "unknown_field": "foo", "scores": {}}
        snap = SignalSnapshot.from_dict(d)
        assert snap.step_idx == 1


# ---------------------------------------------------------------------------
# SentenceTrace tests
# ---------------------------------------------------------------------------

class TestSentenceTrace:
    def test_default_values(self):
        trace = SentenceTrace()
        assert trace.sentence_id == 0
        assert trace.snapshots == []
        assert trace.border_timeline == []

    def test_with_snapshots(self):
        trace = SentenceTrace(
            sentence_id=5,
            direction="en-fr",
            source_text="Hello world",
            n_source_words=2,
            snapshots=[
                SignalSnapshot(step_idx=0, scores={"standard": 0.1}),
                SignalSnapshot(step_idx=1, scores={"standard": 0.8}),
            ],
        )
        assert len(trace.snapshots) == 2
        assert trace.n_source_words == 2

    def test_to_dict_roundtrip(self):
        trace = SentenceTrace(
            sentence_id=3,
            direction="en-zh",
            source_text="Test sentence",
            reference_text="Test reference",
            output_text="Test output",
            n_source_words=2,
            bleu=0.45,
            comet=0.82,
            snapshots=[
                SignalSnapshot(step_idx=0, scores={"standard": -0.5}),
            ],
            border_timeline=[
                {"source_pos": 0, "was_write": False, "emitted_text": ""},
            ],
        )
        d = trace.to_dict()
        trace2 = SentenceTrace.from_dict(d)
        assert trace2.sentence_id == 3
        assert trace2.direction == "en-zh"
        assert trace2.bleu == 0.45
        assert len(trace2.snapshots) == 1
        assert trace2.snapshots[0].scores["standard"] == -0.5


# ---------------------------------------------------------------------------
# TraceCollector tests
# ---------------------------------------------------------------------------

class TestTraceCollector:
    def test_basic_workflow(self):
        collector = TraceCollector()
        collector.start_sentence(0, "Hello world", "Bonjour le monde", "en-fr")
        collector.record_step(
            source_words_seen=1, tokens_generated=0,
            scores={"standard": -0.5}, actual_decision=False,
        )
        collector.record_step(
            source_words_seen=2, tokens_generated=3,
            scores={"standard": 0.8}, actual_decision=True,
        )
        collector.end_sentence(output_text="Bonjour le monde")

        traces = collector.get_traces()
        assert len(traces) == 1
        assert len(traces[0].snapshots) == 2
        assert traces[0].output_text == "Bonjour le monde"

    def test_multiple_sentences(self):
        collector = TraceCollector()
        for i in range(3):
            collector.start_sentence(i, f"Source {i}", f"Target {i}")
            collector.record_step(1, 0, {"standard": 0.1 * i}, False)
            collector.end_sentence(output_text=f"Output {i}")

        traces = collector.get_traces()
        assert len(traces) == 3

    def test_auto_finalize_on_new_sentence(self):
        collector = TraceCollector()
        collector.start_sentence(0, "First", "Premier")
        collector.record_step(1, 0, {"standard": 0.1}, False)
        # Start new sentence without calling end_sentence
        collector.start_sentence(1, "Second", "Deuxieme")
        collector.record_step(1, 0, {"standard": 0.2}, False)
        collector.end_sentence()

        traces = collector.get_traces()
        assert len(traces) == 2

    def test_record_border_event(self):
        collector = TraceCollector()
        collector.start_sentence(0, "Test", "Test")
        collector.record_border_event(0, False, "")
        collector.record_border_event(1, True, "Translated")
        collector.end_sentence()

        traces = collector.get_traces()
        assert len(traces[0].border_timeline) == 2
        assert traces[0].border_timeline[1]["was_write"] is True

    def test_clear(self):
        collector = TraceCollector()
        collector.start_sentence(0, "Test", "Test")
        collector.end_sentence()
        assert len(collector.get_traces()) == 1

        collector.clear()
        assert len(collector.get_traces()) == 0

    def test_record_without_start_is_noop(self):
        collector = TraceCollector()
        collector.record_step(1, 0, {"standard": 0.5}, False)
        assert len(collector.get_traces()) == 0

    def test_get_traces_finalizes_current(self):
        collector = TraceCollector()
        collector.start_sentence(0, "Test", "Test")
        collector.record_step(1, 0, {"standard": 0.5}, False)
        # Don't call end_sentence - get_traces should finalize
        traces = collector.get_traces()
        assert len(traces) == 1


# ---------------------------------------------------------------------------
# Alignment tests
# ---------------------------------------------------------------------------

class TestAlignment:
    def test_monotonic_alignment_equal_length(self):
        alignment = _monotonic_alignment(5, 5)
        assert len(alignment) == 5
        assert alignment == [0, 1, 2, 3, 4]

    def test_monotonic_alignment_more_target(self):
        alignment = _monotonic_alignment(3, 6)
        assert len(alignment) == 6
        # Each target word should map to a valid source position
        for pos in alignment:
            assert 0 <= pos < 3

    def test_monotonic_alignment_more_source(self):
        alignment = _monotonic_alignment(6, 3)
        assert len(alignment) == 3
        for pos in alignment:
            assert 0 <= pos < 6
        # Should be monotonically non-decreasing
        for i in range(len(alignment) - 1):
            assert alignment[i] <= alignment[i + 1]

    def test_monotonic_alignment_empty(self):
        assert _monotonic_alignment(0, 5) == []
        assert _monotonic_alignment(5, 0) == []

    def test_reorder_aware_for_monotonic_pair(self):
        # For en-fr (monotonic), should be same as monotonic
        mono = _monotonic_alignment(5, 5)
        reorder = _reorder_aware_alignment(5, 5, "en-fr")
        assert mono == reorder

    def test_reorder_aware_for_reordering_pair(self):
        # For en-zh (reordering), should shift forward
        mono = _monotonic_alignment(10, 10)
        reorder = _reorder_aware_alignment(10, 10, "en-zh")
        # Reorder positions should be >= monotonic positions
        for m, r in zip(mono, reorder):
            assert r >= m

    def test_reorder_aware_stays_within_bounds(self):
        reorder = _reorder_aware_alignment(5, 5, "en-zh")
        for pos in reorder:
            assert 0 <= pos < 5


# ---------------------------------------------------------------------------
# Border labeling tests
# ---------------------------------------------------------------------------

class TestBorderLabeling:
    def test_label_borders_basic(self):
        labels = label_borders_from_alignment(
            n_source=5, reference="word1 word2 word3 word4 word5",
            direction="en-fr", border_distance=2,
        )
        assert len(labels) == 5
        # Last position should always be should_write (near border)
        assert labels[-1]["should_write"] is True

    def test_label_borders_empty(self):
        assert label_borders_from_alignment(0, "test") == []
        assert label_borders_from_alignment(5, "") == []

    def test_label_borders_all_have_keys(self):
        labels = label_borders_from_alignment(
            n_source=8, reference="a b c d e f g h",
            direction="en-de", border_distance=3,
        )
        for label in labels:
            assert "source_pos" in label
            assert "should_write" in label
            assert "n_safe_targets" in label
            assert "n_new_safe" in label

    def test_label_borders_monotonic_increasing_safe(self):
        labels = label_borders_from_alignment(
            n_source=10, reference="a b c d e f g h i j",
            direction="en-fr", border_distance=3,
        )
        # n_safe_targets should be monotonically non-decreasing
        prev = 0
        for label in labels:
            assert label["n_safe_targets"] >= prev
            prev = label["n_safe_targets"]

    def test_label_from_quality_good_translation(self):
        traces = [SentenceTrace(
            sentence_id=0, direction="en-zh",
            comet=0.85,
            snapshots=[
                SignalSnapshot(scores={"standard": 0.5}, actual_decision=True),
                SignalSnapshot(scores={"standard": -0.3}, actual_decision=False),
            ],
        )]
        examples = label_traces_from_quality(traces, quality_threshold=0.5)
        assert len(examples) == 2
        # Good translation: actual decisions used as labels
        assert examples[0]["should_write"] is True
        assert examples[1]["should_write"] is False

    def test_label_from_quality_bad_translation(self):
        traces = [SentenceTrace(
            sentence_id=0, direction="en-zh",
            comet=0.3,
            snapshots=[
                SignalSnapshot(scores={"standard": 0.5}, actual_decision=True),
                SignalSnapshot(scores={"standard": -0.3}, actual_decision=False),
            ],
        )]
        examples = label_traces_from_quality(traces, quality_threshold=0.5)
        assert len(examples) == 2
        # Bad translation: decisions are flipped
        assert examples[0]["should_write"] is False
        assert examples[1]["should_write"] is True

    def test_label_from_quality_no_metrics(self):
        traces = [SentenceTrace(
            sentence_id=0,
            snapshots=[
                SignalSnapshot(scores={"standard": 0.5}, actual_decision=True),
            ],
        )]
        examples = label_traces_from_quality(traces)
        assert len(examples) == 0  # No quality metric -> skip

    def test_label_from_timeline(self):
        traces = [SentenceTrace(
            sentence_id=0,
            direction="en-fr",
            source_text="hello world foo",
            reference_text="bonjour monde bar",
            n_source_words=3,
            snapshots=[
                SignalSnapshot(
                    step_idx=0, source_words_seen=1,
                    scores={"standard": -0.5}, actual_decision=False,
                ),
                SignalSnapshot(
                    step_idx=1, source_words_seen=2,
                    scores={"standard": 0.2}, actual_decision=False,
                ),
                SignalSnapshot(
                    step_idx=2, source_words_seen=3,
                    scores={"standard": 0.8}, actual_decision=True,
                ),
            ],
        )]
        examples = label_traces_from_timeline(traces, border_distance=2)
        assert len(examples) == 3
        # All examples should have scores and should_write
        for ex in examples:
            assert "scores" in ex
            assert "should_write" in ex

    def test_label_from_timeline_no_reference(self):
        traces = [SentenceTrace(
            sentence_id=0,
            reference_text="",
            snapshots=[
                SignalSnapshot(scores={"standard": 0.5}),
            ],
        )]
        examples = label_traces_from_timeline(traces)
        assert len(examples) == 0


# ---------------------------------------------------------------------------
# Synthetic trace generation tests
# ---------------------------------------------------------------------------

class TestSyntheticTraces:
    def test_generates_correct_count(self):
        traces = generate_synthetic_traces(n_sentences=20)
        assert len(traces) == 20

    def test_all_traces_have_snapshots(self):
        traces = generate_synthetic_traces(n_sentences=10)
        for trace in traces:
            assert len(trace.snapshots) > 0
            assert trace.n_source_words > 0
            assert trace.direction == "en-zh"

    def test_signal_scores_in_range(self):
        traces = generate_synthetic_traces(n_sentences=50)
        for trace in traces:
            for snap in trace.snapshots:
                for signal, score in snap.scores.items():
                    assert -1.0 <= score <= 1.0, \
                        f"Signal {signal} out of range: {score}"

    def test_all_8_signals_present(self):
        traces = generate_synthetic_traces(n_sentences=10)
        expected_signals = {
            "standard", "shift_k", "info_gain", "coverage",
            "monotonicity", "entropy_change", "pred_stability", "attn_shift",
        }
        for trace in traces:
            for snap in trace.snapshots:
                assert set(snap.scores.keys()) == expected_signals

    def test_custom_direction(self):
        traces = generate_synthetic_traces(n_sentences=5, direction="en-de")
        for trace in traces:
            assert trace.direction == "en-de"

    def test_reproducibility(self):
        t1 = generate_synthetic_traces(n_sentences=5, seed=123)
        t2 = generate_synthetic_traces(n_sentences=5, seed=123)
        for a, b in zip(t1, t2):
            assert a.n_source_words == b.n_source_words
            for sa, sb in zip(a.snapshots, b.snapshots):
                assert sa.scores == sb.scores

    def test_different_seeds_different_results(self):
        t1 = generate_synthetic_traces(n_sentences=5, seed=1)
        t2 = generate_synthetic_traces(n_sentences=5, seed=2)
        # At least one should differ
        any_diff = False
        for a, b in zip(t1, t2):
            if a.n_source_words != b.n_source_words:
                any_diff = True
                break
        assert any_diff


# ---------------------------------------------------------------------------
# I/O tests
# ---------------------------------------------------------------------------

class TestIO:
    def test_save_load_roundtrip(self, tmp_path):
        traces = generate_synthetic_traces(n_sentences=5)
        path = str(tmp_path / "traces.json")
        save_traces(traces, path)
        loaded = load_traces(path)
        assert len(loaded) == 5
        for orig, loaded_t in zip(traces, loaded):
            assert orig.sentence_id == loaded_t.sentence_id
            assert orig.direction == loaded_t.direction
            assert len(orig.snapshots) == len(loaded_t.snapshots)

    def test_export_load_weights(self, tmp_path):
        results = {
            "en-zh": {
                "weights": FusionWeights(standard=1.0, shift_k=0.6),
                "threshold": 0.15,
                "f1": 0.82,
                "n_traces": 50,
                "n_examples": 500,
            },
        }
        path = str(tmp_path / "weights.json")
        export_weights(results, path)
        loaded = load_calibrated_weights(path)
        assert "en-zh" in loaded
        weights, threshold = loaded["en-zh"]
        assert weights.standard == 1.0
        assert weights.shift_k == 0.6
        assert threshold == 0.15


# ---------------------------------------------------------------------------
# Calibration pipeline tests
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_calibrate_direction_synthetic(self):
        traces = generate_synthetic_traces(
            n_sentences=50, direction="en-zh", seed=42
        )
        weights, threshold, f1 = calibrate_direction(
            traces, direction="en-zh", method="alignment"
        )
        # Should return valid FusionWeights
        assert isinstance(weights, FusionWeights)
        assert isinstance(threshold, float)
        assert 0.0 <= f1 <= 1.0

    def test_calibrate_direction_no_traces(self):
        weights, threshold, f1 = calibrate_direction(
            [], direction="en-zh"
        )
        assert f1 == 0.0  # No data

    def test_calibrate_direction_wrong_direction(self):
        traces = generate_synthetic_traces(n_sentences=10, direction="en-fr")
        weights, threshold, f1 = calibrate_direction(
            traces, direction="en-zh"  # No en-zh traces
        )
        assert f1 == 0.0

    def test_calibrate_quality_method(self):
        traces = generate_synthetic_traces(n_sentences=30, direction="en-zh")
        # Add quality metrics
        for trace in traces:
            trace.comet = np.random.uniform(0.3, 0.9)

        weights, threshold, f1 = calibrate_direction(
            traces, direction="en-zh", method="quality",
            quality_threshold=0.5,
        )
        assert isinstance(weights, FusionWeights)

    def test_run_calibration_multi_direction(self):
        all_traces = []
        for d in ["en-zh", "en-de", "en-fr"]:
            traces = generate_synthetic_traces(
                n_sentences=20, direction=d
            )
            all_traces.extend(traces)

        results = run_calibration(all_traces)
        assert "en-zh" in results
        assert "en-de" in results
        assert "en-fr" in results
        for d, result in results.items():
            assert "weights" in result
            assert "threshold" in result
            assert "f1" in result

    def test_run_calibration_specific_directions(self):
        all_traces = []
        for d in ["en-zh", "en-de", "en-fr"]:
            traces = generate_synthetic_traces(n_sentences=20, direction=d)
            all_traces.extend(traces)

        results = run_calibration(all_traces, directions=["en-zh"])
        assert "en-zh" in results
        assert "en-de" not in results

    def test_calibrate_invalid_method(self):
        traces = generate_synthetic_traces(n_sentences=5)
        with pytest.raises(ValueError, match="Unknown labeling method"):
            calibrate_direction(traces, method="invalid")


# ---------------------------------------------------------------------------
# Signal importance analysis tests
# ---------------------------------------------------------------------------

class TestSignalAnalysis:
    def test_analyze_signal_importance(self):
        traces = generate_synthetic_traces(
            n_sentences=50, direction="en-zh", seed=42
        )
        importance = analyze_signal_importance(traces, "en-zh")
        assert len(importance) > 0
        for signal, stats in importance.items():
            assert "mean_write" in stats
            assert "mean_read" in stats
            assert "discriminative_power" in stats
            assert "correlation" in stats
            assert "n_write" in stats
            assert "n_read" in stats

    def test_analyze_empty_traces(self):
        importance = analyze_signal_importance([], "en-zh")
        assert importance == {}

    def test_analyze_wrong_direction(self):
        traces = generate_synthetic_traces(n_sentences=10, direction="en-fr")
        importance = analyze_signal_importance(traces, "en-zh")
        assert importance == {}

    def test_standard_signal_is_discriminative(self):
        """Standard border should be the most discriminative signal in synthetic data."""
        traces = generate_synthetic_traces(
            n_sentences=100, direction="en-zh", seed=42
        )
        importance = analyze_signal_importance(traces, "en-zh")
        assert "standard" in importance
        # Standard should have positive discriminative power
        # (higher score for WRITE than READ)
        assert importance["standard"]["discriminative_power"] > 0


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------

class TestCLI:
    def test_demo_mode(self):
        """Test that --demo runs without error."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "nllw.calibrate", "--demo",
             "--n-synthetic", "10"],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        assert result.returncode == 0
        assert "CALIBRATION REPORT" in result.stdout

    def test_demo_with_analyze(self):
        """Test --demo --analyze."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "nllw.calibrate", "--demo", "--analyze",
             "--n-synthetic", "10"],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        assert result.returncode == 0
        assert "discriminative" in result.stdout

    def test_demo_with_output(self, tmp_path):
        """Test --demo --output exports weights."""
        import subprocess
        output_path = str(tmp_path / "weights.json")
        result = subprocess.run(
            [sys.executable, "-m", "nllw.calibrate", "--demo",
             "--n-synthetic", "10", "--output", output_path],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        assert result.returncode == 0
        assert os.path.exists(output_path)
        with open(output_path) as f:
            data = json.load(f)
        assert "en-zh" in data


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_word_sentence(self):
        labels = label_borders_from_alignment(
            n_source=1, reference="bonjour",
            direction="en-fr", border_distance=3,
        )
        assert len(labels) == 1
        assert labels[0]["should_write"] is True  # Only word, must emit

    def test_very_long_sentence(self):
        traces = generate_synthetic_traces(
            n_sentences=3, n_source_words=50, seed=42
        )
        for trace in traces:
            assert trace.n_source_words > 0
            assert len(trace.snapshots) == trace.n_source_words

    def test_zero_source_words(self):
        labels = label_borders_from_alignment(
            n_source=0, reference="test", direction="en-fr",
        )
        assert labels == []

    def test_snapshot_with_fusion_diagnostics(self):
        snap = SignalSnapshot(
            scores={"standard": 0.5, "shift_k": 0.3},
            fusion_score=0.42,
            fusion_threshold=0.1,
        )
        d = snap.to_dict()
        snap2 = SignalSnapshot.from_dict(d)
        assert snap2.fusion_score == 0.42
        assert snap2.fusion_threshold == 0.1

    def test_trace_collector_with_fusion_info(self):
        collector = TraceCollector()
        collector.start_sentence(0, "Test", "Test")
        collector.record_step(
            source_words_seen=1, tokens_generated=0,
            scores={"standard": 0.5},
            actual_decision=True,
            fusion_score=0.42,
            fusion_threshold=0.1,
        )
        collector.end_sentence()
        traces = collector.get_traces()
        assert traces[0].snapshots[0].fusion_score == 0.42

    def test_alignment_single_target_word(self):
        alignment = _monotonic_alignment(5, 1)
        assert len(alignment) == 1
        assert 0 <= alignment[0] < 5

    def test_alignment_single_source_word(self):
        alignment = _monotonic_alignment(1, 5)
        assert len(alignment) == 5
        assert all(pos == 0 for pos in alignment)
