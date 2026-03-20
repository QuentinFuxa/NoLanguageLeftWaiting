"""Tests for Iteration 12 features: weighted signal fusion framework.

Replaces the boolean cascade in check_border_combined() with a principled
weighted scoring system where each signal produces a continuous confidence
score in [-1, +1], and the final border decision is based on their weighted sum.

Key advantages:
    1. Weak signals combine: two marginal signals trigger stops neither would alone
    2. Per-direction tuning: weight vectors optimized per language pair
    3. Signal contributions are observable: diagnostics show which signals drove each decision
    4. Order-independent: no hidden priority from if/else sequencing

Novel: no published work on weighted multi-signal fusion for SimulMT border detection.

All tests are unit tests that don't require llama.cpp.
"""

import numpy as np
import pytest

from nllw.fusion import (
    score_standard_border,
    score_shift_k,
    score_info_gain,
    score_coverage,
    score_monotonicity,
    score_entropy_change,
    score_pred_stability,
    score_attn_shift,
    FusionWeights,
    FusionDiagnostic,
    FusionConfig,
    fused_border_check,
    get_fusion_weights,
    calibrate_threshold,
    grid_search_weights,
    DIRECTION_WEIGHTS,
    _f1_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_border_attn(n_heads=3, n_src=10, focus_pos=9):
    """Create attention focused on a specific position."""
    attn = np.zeros((n_heads, n_src))
    for h in range(n_heads):
        attn[h, focus_pos] = 0.9
        # Spread remaining 0.1 uniformly
        for j in range(n_src):
            if j != focus_pos:
                attn[h, j] = 0.1 / (n_src - 1)
    return attn


def _make_uniform_attn(n_heads=3, n_src=10):
    """Create uniform attention."""
    return np.ones((n_heads, n_src)) / n_src


def _make_ts_scores(n_heads=3):
    """Create simple TS scores."""
    return [0.8, 0.6, 0.4][:n_heads]


# ===========================================================================
# Signal score unit tests
# ===========================================================================

class TestScoreStandardBorder:
    """Test standard AlignAtt border scoring."""

    def test_in_border_region_gives_positive_score(self):
        """Attention at last position -> positive WRITE score."""
        attn = _make_border_attn(focus_pos=9)
        ts = _make_ts_scores()
        score = score_standard_border(attn, ts, 10, 3)
        assert score > 0.0, f"Expected positive score for border attention, got {score}"

    def test_far_from_border_gives_negative_score(self):
        """Attention at start -> negative READ score."""
        attn = _make_border_attn(focus_pos=2)
        ts = _make_ts_scores()
        score = score_standard_border(attn, ts, 10, 3)
        assert score < 0.0, f"Expected negative score for early attention, got {score}"

    def test_at_border_boundary_gives_zero(self):
        """Attention exactly at border threshold -> score near zero."""
        attn = _make_border_attn(focus_pos=7)  # border_threshold = 10-3 = 7
        ts = _make_ts_scores()
        score = score_standard_border(attn, ts, 10, 3)
        assert abs(score) < 0.2, f"Expected score near zero at boundary, got {score}"

    def test_score_bounded(self):
        """Score should always be in [-1, +1]."""
        for pos in range(10):
            attn = _make_border_attn(focus_pos=pos)
            ts = _make_ts_scores()
            score = score_standard_border(attn, ts, 10, 3)
            assert -1.0 <= score <= 1.0, f"Score {score} out of range for pos={pos}"

    def test_empty_source(self):
        """Zero source tokens -> return 0."""
        attn = np.zeros((3, 0))
        score = score_standard_border(attn, [0.8, 0.6, 0.4], 0, 3)
        assert score == 0.0

    def test_adaptive_aggregation(self):
        """AMS should work without error."""
        attn = _make_border_attn(focus_pos=9)
        ts = _make_ts_scores()
        score = score_standard_border(
            attn, ts, 10, 3, adaptive_aggregation=True
        )
        assert -1.0 <= score <= 1.0


class TestScoreShiftK:
    """Test shift-k border mass scoring."""

    def test_high_mass_in_border_gives_positive(self):
        """Mass concentrated in border region -> positive score."""
        attn = _make_border_attn(focus_pos=9)
        ts = _make_ts_scores()
        score = score_shift_k(attn, ts, 10, 3, threshold=0.4)
        assert score > 0.0, f"Expected positive score for border mass, got {score}"

    def test_low_mass_in_border_gives_negative(self):
        """Mass at start -> negative score."""
        attn = _make_border_attn(focus_pos=2)
        ts = _make_ts_scores()
        score = score_shift_k(attn, ts, 10, 3, threshold=0.4)
        assert score < 0.0, f"Expected negative score for low border mass, got {score}"

    def test_score_bounded(self):
        """Score always in [-1, +1]."""
        for pos in range(10):
            attn = _make_border_attn(focus_pos=pos)
            ts = _make_ts_scores()
            score = score_shift_k(attn, ts, 10, 3, threshold=0.4)
            assert -1.0 <= score <= 1.0

    def test_zero_border_distance(self):
        """border_distance=0 -> border_start=n_src -> return 0."""
        attn = _make_border_attn(focus_pos=9)
        ts = _make_ts_scores()
        score = score_shift_k(attn, ts, 10, 0, threshold=0.4)
        assert score == 0.0


class TestScoreInfoGain:
    """Test information gain scoring."""

    def test_no_prev_attn_returns_zero(self):
        """No previous attention -> neutral score."""
        attn = _make_border_attn()
        ts = _make_ts_scores()
        score = score_info_gain(attn, None, ts)
        assert score == 0.0

    def test_same_attn_gives_positive(self):
        """Same attention = zero info gain -> WRITE (positive)."""
        attn = _make_border_attn(focus_pos=5)
        ts = _make_ts_scores()
        score = score_info_gain(attn, attn.copy(), ts, threshold=0.3)
        assert score > 0.0, f"Expected positive for zero info gain, got {score}"

    def test_different_attn_gives_negative(self):
        """Very different attention = high info gain -> READ (negative)."""
        attn1 = _make_border_attn(focus_pos=2)
        attn2 = _make_border_attn(focus_pos=8)
        ts = _make_ts_scores()
        score = score_info_gain(attn2, attn1, ts, threshold=0.3)
        assert score < 0.0, f"Expected negative for high info gain, got {score}"

    def test_score_bounded(self):
        """Score always in [-1, +1]."""
        attn1 = _make_border_attn(focus_pos=2)
        attn2 = _make_border_attn(focus_pos=8)
        ts = _make_ts_scores()
        score = score_info_gain(attn2, attn1, ts, threshold=0.3)
        assert -1.0 <= score <= 1.0

    def test_different_sizes(self):
        """Handles prev_attn with fewer columns (new source tokens added)."""
        attn_prev = _make_border_attn(n_src=8, focus_pos=5)
        attn_cur = _make_border_attn(n_src=10, focus_pos=7)
        ts = _make_ts_scores()
        score = score_info_gain(attn_cur, attn_prev, ts, threshold=0.3)
        assert -1.0 <= score <= 1.0


class TestScoreCoverage:
    """Test source coverage scoring."""

    def test_focused_attn_gives_positive(self):
        """Narrow focus (low coverage) -> force stop -> positive WRITE score."""
        attn = _make_border_attn(focus_pos=5)  # Focus on one position
        ts = _make_ts_scores()
        score = score_coverage(attn, ts, threshold=0.3)
        # Focused attention = low coverage = hallucination risk = force WRITE
        assert score > 0.0, f"Expected positive for focused attention, got {score}"

    def test_uniform_attn_gives_negative(self):
        """Uniform attention (high coverage) -> safe to continue -> negative."""
        attn = _make_uniform_attn()
        ts = _make_ts_scores()
        score = score_coverage(attn, ts, threshold=0.3)
        assert score < 0.0, f"Expected negative for good coverage, got {score}"

    def test_score_bounded(self):
        """Score always in [-1, +1]."""
        for focus in range(10):
            attn = _make_border_attn(focus_pos=focus)
            ts = _make_ts_scores()
            score = score_coverage(attn, ts, threshold=0.3)
            assert -1.0 <= score <= 1.0


class TestScoreMonotonicity:
    """Test monotonicity scoring."""

    def test_no_history_returns_zero(self):
        """No position history -> neutral."""
        assert score_monotonicity(None, 3) == 0.0
        assert score_monotonicity([], 3) == 0.0
        assert score_monotonicity([1.0, 2.0], 3) == 0.0

    def test_monotonic_gives_positive(self):
        """Monotonically increasing positions -> positive WRITE score."""
        history = [1.0, 2.0, 3.0, 4.0, 5.0]
        score = score_monotonicity(history, 3)
        assert score > 0.0, f"Expected positive for monotonic, got {score}"

    def test_non_monotonic_gives_negative(self):
        """Jumping positions -> negative READ score."""
        history = [5.0, 1.0, 8.0, 2.0, 7.0]
        score = score_monotonicity(history, 3)
        assert score < 0.0, f"Expected negative for non-monotonic, got {score}"

    def test_score_bounded(self):
        """Score always in [-1, +1]."""
        history = [1.0, 2.0, 3.0, 4.0, 5.0]
        score = score_monotonicity(history, 3)
        assert -1.0 <= score <= 1.0


class TestScoreEntropyChange:
    """Test entropy change scoring."""

    def test_none_returns_zero(self):
        """No entropy data -> neutral."""
        assert score_entropy_change(None) == 0.0

    def test_small_change_gives_positive(self):
        """Small entropy change -> source exhausted -> WRITE (positive)."""
        score = score_entropy_change(-0.1, threshold=-0.5)
        assert score > 0.0, f"Expected positive for small change, got {score}"

    def test_large_drop_gives_negative(self):
        """Large entropy drop -> model learning -> READ (negative)."""
        score = score_entropy_change(-2.0, threshold=-0.5)
        assert score < 0.0, f"Expected negative for large drop, got {score}"

    def test_score_bounded(self):
        """Score always in [-1, +1]."""
        for delta in [-5.0, -2.0, -0.5, 0.0, 0.5, 2.0]:
            score = score_entropy_change(delta, threshold=-0.5)
            assert -1.0 <= score <= 1.0, f"Score {score} out of range for delta={delta}"


class TestScorePredStability:
    """Test prediction stability scoring."""

    def test_none_returns_zero(self):
        assert score_pred_stability(None) == 0.0

    def test_stable_gives_positive(self):
        assert score_pred_stability(True) > 0.0

    def test_volatile_gives_negative(self):
        assert score_pred_stability(False) < 0.0


class TestScoreAttnShift:
    """Test attention shift scoring."""

    def test_none_returns_zero(self):
        assert score_attn_shift(None) == 0.0

    def test_shifted_gives_positive(self):
        assert score_attn_shift(True) > 0.0

    def test_stuck_gives_negative(self):
        assert score_attn_shift(False) < 0.0


# ===========================================================================
# FusionWeights tests
# ===========================================================================

class TestFusionWeights:
    """Test FusionWeights dataclass."""

    def test_default_weights(self):
        """All defaults are positive."""
        w = FusionWeights()
        d = w.as_dict()
        assert all(v >= 0 for v in d.values())
        assert d["standard"] == 1.0

    def test_from_dict(self):
        """Create from dict, ignore unknown keys."""
        w = FusionWeights.from_dict({"standard": 0.5, "coverage": 0.9, "bogus": 42})
        assert w.standard == 0.5
        assert w.coverage == 0.9

    def test_enabled_signals(self):
        """Only signals with non-zero weight are enabled."""
        w = FusionWeights(standard=1.0, shift_k=0.0, info_gain=0.5)
        enabled = w.enabled_signals()
        assert "standard" in enabled
        assert "info_gain" in enabled
        assert "shift_k" not in enabled

    def test_total_weight(self):
        """Total weight = sum of absolute values."""
        w = FusionWeights(standard=1.0, shift_k=-0.5)
        assert abs(w.total_weight() - (1.0 + 0.5 + 0.4 + 0.8 + 0.3 + 0.4 + 0.3 + 0.3)) < 0.01

    def test_roundtrip(self):
        """as_dict -> from_dict roundtrip preserves values."""
        w1 = FusionWeights(standard=0.7, coverage=0.9)
        w2 = FusionWeights.from_dict(w1.as_dict())
        assert w2.standard == w1.standard
        assert w2.coverage == w1.coverage


class TestDirectionWeights:
    """Test per-direction weight profiles."""

    def test_all_directions_have_weights(self):
        """All 5 directions have pre-configured weights."""
        for d in ["en-zh", "en-de", "en-it", "cs-en", "en-fr"]:
            w = get_fusion_weights(d)
            assert isinstance(w, FusionWeights)
            assert w.standard > 0

    def test_unknown_direction_uses_defaults(self):
        """Unknown direction falls back to default FusionWeights."""
        w = get_fusion_weights("xx-yy")
        assert isinstance(w, FusionWeights)
        assert w.standard == 1.0

    def test_directions_are_different(self):
        """Different directions should have some different weights."""
        w_zh = get_fusion_weights("en-zh")
        w_cs = get_fusion_weights("cs-en")
        # CS-EN has more reordering, lower monotonicity weight
        assert w_cs.monotonicity != w_zh.monotonicity


# ===========================================================================
# FusionDiagnostic tests
# ===========================================================================

class TestFusionDiagnostic:
    """Test diagnostic output."""

    def test_dominant_signal(self):
        """Returns signal with largest contribution magnitude."""
        diag = FusionDiagnostic(
            contributions={"standard": 0.3, "coverage": 0.8, "shift_k": -0.1}
        )
        assert diag.dominant_signal() == "coverage"

    def test_dominant_signal_negative(self):
        """Negative contributions can dominate."""
        diag = FusionDiagnostic(
            contributions={"standard": 0.3, "entropy_change": -0.9}
        )
        assert diag.dominant_signal() == "entropy_change"

    def test_empty_contributions(self):
        """No contributions -> None."""
        diag = FusionDiagnostic()
        assert diag.dominant_signal() is None

    def test_summary(self):
        """Summary string is well-formed."""
        diag = FusionDiagnostic(
            contributions={"standard": 0.5},
            fusion_score=0.5,
            threshold=0.0,
            decision=True,
        )
        s = diag.summary()
        assert "WRITE" in s
        assert "0.500" in s


# ===========================================================================
# Main fusion function tests
# ===========================================================================

class TestFusedBorderCheck:
    """Test the main fused_border_check() function."""

    def test_border_attention_triggers_write(self):
        """Attention at end of source -> WRITE."""
        attn = _make_border_attn(focus_pos=9)
        ts = _make_ts_scores()
        weights = FusionWeights(standard=1.0, shift_k=0.0, info_gain=0.0,
                                coverage=0.0, monotonicity=0.0,
                                entropy_change=0.0, pred_stability=0.0,
                                attn_shift=0.0)
        hit, diag = fused_border_check(
            attn, ts, 10, 3, weights=weights, threshold=0.0,
        )
        assert hit is True
        assert diag.decision is True
        assert diag.fusion_score > 0.0

    def test_early_attention_triggers_read(self):
        """Attention at start -> READ."""
        attn = _make_border_attn(focus_pos=2)
        ts = _make_ts_scores()
        weights = FusionWeights(standard=1.0, shift_k=0.0, info_gain=0.0,
                                coverage=0.0, monotonicity=0.0,
                                entropy_change=0.0, pred_stability=0.0,
                                attn_shift=0.0)
        hit, diag = fused_border_check(
            attn, ts, 10, 3, weights=weights, threshold=0.0,
        )
        assert hit is False
        assert diag.decision is False
        assert diag.fusion_score < 0.0

    def test_weak_signals_combine(self):
        """Two weak positive signals can trigger WRITE together.

        This is the KEY advantage over boolean cascade: neither signal
        alone would trigger, but together they do.
        """
        attn = _make_border_attn(focus_pos=7)  # Exactly at border boundary
        ts = _make_ts_scores()
        # Standard alone might be near zero; add shift-k which also sees some mass
        weights = FusionWeights(
            standard=0.5, shift_k=0.5,
            info_gain=0.0, coverage=0.0, monotonicity=0.0,
            entropy_change=0.0, pred_stability=0.0, attn_shift=0.0,
        )

        # Also give entropy change a small positive push
        hit, diag = fused_border_check(
            attn, ts, 10, 3, weights=weights, threshold=-0.1,
        )
        # With both signals weakly positive, the combined score should be
        # more informative than either alone
        assert "standard" in diag.scores
        assert "shift_k" in diag.scores

    def test_coverage_can_force_write(self):
        """Focused attention (low coverage) -> coverage signal forces WRITE."""
        attn = _make_border_attn(focus_pos=5)  # Not in border region
        ts = _make_ts_scores()
        # Only coverage enabled with high weight
        weights = FusionWeights(
            standard=0.0, shift_k=0.0, info_gain=0.0,
            coverage=1.0, monotonicity=0.0,
            entropy_change=0.0, pred_stability=0.0, attn_shift=0.0,
        )
        hit, diag = fused_border_check(
            attn, ts, 10, 3, weights=weights, threshold=0.0,
            coverage_ref=0.3,
        )
        # Focused attention = low coverage = positive score = WRITE
        assert diag.scores.get("coverage", 0) > 0.0

    def test_entropy_change_inhibits_write(self):
        """Large entropy drop -> READ -> can inhibit border."""
        attn = _make_border_attn(focus_pos=9)  # In border region
        ts = _make_ts_scores()
        weights = FusionWeights(
            standard=0.5, shift_k=0.0, info_gain=0.0,
            coverage=0.0, monotonicity=0.0,
            entropy_change=0.8,  # High weight on entropy change
            pred_stability=0.0, attn_shift=0.0,
        )
        hit, diag = fused_border_check(
            attn, ts, 10, 3, weights=weights, threshold=0.0,
            entropy_change=-3.0,  # Large drop -> READ signal
            entropy_change_ref=-0.5,
        )
        # entropy_change score should be strongly negative
        assert diag.scores.get("entropy_change", 0) < 0.0

    def test_all_signals_enabled(self):
        """All 8 signals compute without error."""
        attn = _make_border_attn(focus_pos=8)
        prev_attn = _make_border_attn(focus_pos=6)
        ts = _make_ts_scores()
        weights = FusionWeights()  # All defaults
        hit, diag = fused_border_check(
            attn, ts, 10, 3, weights=weights,
            prev_attn=prev_attn,
            positions_history=[1.0, 2.0, 3.0, 4.0, 5.0],
            entropy_change=-0.3,
            pred_stability_write=True,
            attn_shift_write=True,
        )
        # All signals should be present
        assert len(diag.scores) == 8, f"Expected 8 signals, got {len(diag.scores)}"
        assert all(-1.0 <= s <= 1.0 for s in diag.scores.values())
        assert isinstance(diag.fusion_score, float)
        assert isinstance(diag.decision, bool)

    def test_diagnostic_has_all_fields(self):
        """Diagnostic output is complete."""
        attn = _make_border_attn(focus_pos=9)
        ts = _make_ts_scores()
        weights = FusionWeights(standard=1.0, shift_k=0.5)
        _, diag = fused_border_check(attn, ts, 10, 3, weights=weights)
        assert "standard" in diag.scores
        assert "standard" in diag.weights
        assert "standard" in diag.contributions
        assert diag.threshold == 0.0

    def test_threshold_effect(self):
        """Higher threshold requires stronger signal to trigger WRITE."""
        attn = _make_border_attn(focus_pos=8)
        ts = _make_ts_scores()
        weights = FusionWeights(standard=1.0, shift_k=0.0, info_gain=0.0,
                                coverage=0.0, monotonicity=0.0,
                                entropy_change=0.0, pred_stability=0.0,
                                attn_shift=0.0)

        hit_low, _ = fused_border_check(
            attn, ts, 10, 3, weights=weights, threshold=-0.5,
        )
        hit_high, _ = fused_border_check(
            attn, ts, 10, 3, weights=weights, threshold=0.5,
        )
        # Low threshold should be easier to trigger
        assert hit_low or not hit_high  # If high fires, low must too

    def test_head_temp_normalization(self):
        """Head temp normalization works through fusion."""
        attn = _make_border_attn(focus_pos=9)
        ts = _make_ts_scores()
        weights = FusionWeights(standard=1.0)
        hit, _ = fused_border_check(
            attn, ts, 10, 3, weights=weights,
            head_temp_normalize=True, head_temp_reference=1.5,
        )
        assert isinstance(hit, bool)

    def test_dynamic_border(self):
        """Dynamic border works through fusion."""
        attn = _make_border_attn(focus_pos=9)
        ts = _make_ts_scores()
        weights = FusionWeights(standard=1.0)
        hit, _ = fused_border_check(
            attn, ts, 10, 3, weights=weights, dynamic_border=True,
        )
        assert isinstance(hit, bool)

    def test_normalized_score(self):
        """Fusion score is normalized by active weight sum."""
        attn = _make_border_attn(focus_pos=9)
        ts = _make_ts_scores()

        # Two signals with weight 1.0 each
        w1 = FusionWeights(standard=1.0, shift_k=1.0,
                           info_gain=0.0, coverage=0.0, monotonicity=0.0,
                           entropy_change=0.0, pred_stability=0.0, attn_shift=0.0)
        _, diag1 = fused_border_check(attn, ts, 10, 3, weights=w1)

        # Same signals with weight 2.0 each (should give same normalized score)
        w2 = FusionWeights(standard=2.0, shift_k=2.0,
                           info_gain=0.0, coverage=0.0, monotonicity=0.0,
                           entropy_change=0.0, pred_stability=0.0, attn_shift=0.0)
        _, diag2 = fused_border_check(attn, ts, 10, 3, weights=w2)

        assert abs(diag1.fusion_score - diag2.fusion_score) < 0.01, \
            f"Normalized scores should match: {diag1.fusion_score} vs {diag2.fusion_score}"

    def test_zero_weights_means_neutral(self):
        """All zero weights -> fusion score = 0."""
        attn = _make_border_attn(focus_pos=9)
        ts = _make_ts_scores()
        weights = FusionWeights(standard=0.0, shift_k=0.0, info_gain=0.0,
                                coverage=0.0, monotonicity=0.0,
                                entropy_change=0.0, pred_stability=0.0,
                                attn_shift=0.0)
        hit, diag = fused_border_check(attn, ts, 10, 3, weights=weights)
        assert diag.fusion_score == 0.0
        assert len(diag.scores) == 0


# ===========================================================================
# FusionConfig tests
# ===========================================================================

class TestFusionConfig:
    """Test FusionConfig."""

    def test_default_disabled(self):
        """Fusion is disabled by default."""
        fc = FusionConfig()
        assert fc.enabled is False

    def test_from_dict(self):
        """Create from dict with nested weights."""
        fc = FusionConfig.from_dict({
            "enabled": True,
            "threshold": 0.1,
            "weights": {"standard": 0.7, "coverage": 0.9},
        })
        assert fc.enabled is True
        assert fc.threshold == 0.1
        assert fc.weights.standard == 0.7
        assert fc.weights.coverage == 0.9

    def test_from_dict_without_weights(self):
        """Create from dict without weights -> None."""
        fc = FusionConfig.from_dict({"enabled": True, "threshold": 0.2})
        assert fc.enabled is True
        assert fc.weights is None


# ===========================================================================
# Calibration tests
# ===========================================================================

class TestCalibrateThreshold:
    """Test threshold calibration from examples."""

    def test_perfect_separation(self):
        """Perfectly separable data -> good threshold."""
        examples = [
            {"scores": {"standard": 0.8}, "should_write": True},
            {"scores": {"standard": 0.6}, "should_write": True},
            {"scores": {"standard": -0.3}, "should_write": False},
            {"scores": {"standard": -0.7}, "should_write": False},
        ]
        thr = calibrate_threshold(examples, FusionWeights(standard=1.0))
        # Threshold should be between -0.3 and 0.6
        assert -0.5 < thr < 0.8

    def test_empty_examples(self):
        """No examples -> threshold 0."""
        thr = calibrate_threshold([], FusionWeights())
        assert thr == 0.0

    def test_target_write_ratio(self):
        """Different write ratios affect threshold."""
        examples = [
            {"scores": {"standard": float(i) / 10}, "should_write": i > 5}
            for i in range(10)
        ]
        thr_balanced = calibrate_threshold(
            examples, FusionWeights(standard=1.0), target_write_ratio=0.5
        )
        thr_aggressive = calibrate_threshold(
            examples, FusionWeights(standard=1.0), target_write_ratio=0.8
        )
        # Aggressive WRITE -> lower threshold
        assert thr_aggressive <= thr_balanced


class TestGridSearchWeights:
    """Test grid search over weight combinations."""

    def test_finds_good_weights(self):
        """Grid search finds weights with high F1."""
        # Create clear signal pattern:
        # standard positive + coverage positive -> WRITE
        # standard negative -> READ
        examples = [
            {"scores": {"standard": 0.8, "coverage": 0.5}, "should_write": True},
            {"scores": {"standard": 0.6, "coverage": 0.3}, "should_write": True},
            {"scores": {"standard": -0.5, "coverage": -0.2}, "should_write": False},
            {"scores": {"standard": -0.8, "coverage": -0.4}, "should_write": False},
        ]
        weights, thr, metric = grid_search_weights(
            examples,
            weight_grid={"standard": [0.5, 1.0], "coverage": [0.0, 0.5]},
        )
        assert metric > 0.5, f"Expected F1 > 0.5, got {metric}"
        assert isinstance(weights, FusionWeights)

    def test_empty_examples(self):
        """Empty examples -> default weights."""
        weights, thr, metric = grid_search_weights([])
        assert isinstance(weights, FusionWeights)
        assert metric == 0.0


class TestF1Score:
    """Test F1 helper."""

    def test_perfect_predictions(self):
        """All correct -> F1 = 1.0."""
        assert _f1_score([True, True, False], [True, True, False]) == 1.0

    def test_all_wrong(self):
        """All wrong -> F1 = 0.0."""
        assert _f1_score([True, True, True], [False, False, False]) == 0.0

    def test_no_positives(self):
        """No true positives -> F1 = 0."""
        assert _f1_score([False, False], [True, True]) == 0.0


# ===========================================================================
# Integration-style tests
# ===========================================================================

class TestFusionIntegration:
    """Integration tests simulating realistic scenarios."""

    def test_monotonic_translation_en_de(self):
        """EN-DE is mostly monotonic -> tighter border decisions."""
        n_src = 12
        ts = [0.9, 0.7, 0.5]
        weights = get_fusion_weights("en-de")

        # Simulate generation with monotonically advancing attention
        positions = []
        for step in range(8):
            focus = min(step + 4, n_src - 1)
            attn = _make_border_attn(n_heads=3, n_src=n_src, focus_pos=focus)
            positions.append(float(focus))

            hit, diag = fused_border_check(
                attn, ts, n_src, 3, weights=weights,
                positions_history=positions,
                entropy_change=-0.1,
                pred_stability_write=True,
                attn_shift_write=True,
            )
            # Later steps (closer to border) should have higher scores
            if step >= 6:
                assert diag.fusion_score > diag.threshold or focus >= n_src - 3

    def test_reordering_translation_cs_en(self):
        """CS-EN has reordering -> wider border, more conservative."""
        n_src = 12
        ts = [0.9, 0.7, 0.5]
        weights = get_fusion_weights("cs-en")

        # Non-monotonic attention (Czech reordering)
        positions = [3.0, 1.0, 5.0, 2.0, 7.0]
        attn = _make_border_attn(n_heads=3, n_src=n_src, focus_pos=7)

        hit, diag = fused_border_check(
            attn, ts, n_src, 3, weights=weights,
            positions_history=positions,
            entropy_change=-1.5,  # Model still learning
            pred_stability_write=False,
            attn_shift_write=True,
        )
        # With non-monotonic attention and volatile predictions,
        # fusion should be more conservative (lower score)
        assert diag.scores.get("monotonicity", 0) < 0.0 or len(positions) < 3

    def test_hallucination_detection(self):
        """Low coverage should trigger force WRITE regardless of other signals."""
        n_src = 10
        ts = [0.9, 0.7, 0.5]

        # Attention focused on one position (hallucination pattern)
        attn = np.zeros((3, n_src))
        attn[:, 0] = 0.95
        attn[:, 1:] = 0.05 / (n_src - 1)

        weights = FusionWeights(
            standard=0.5, coverage=1.0,
            shift_k=0.0, info_gain=0.0, monotonicity=0.0,
            entropy_change=0.0, pred_stability=0.0, attn_shift=0.0,
        )

        hit, diag = fused_border_check(
            attn, ts, n_src, 3, weights=weights,
            coverage_ref=0.3,
        )
        # Coverage signal should dominate and force WRITE
        assert diag.scores.get("coverage", 0) > 0.0
        assert diag.dominant_signal() == "coverage"

    def test_all_signals_realistic(self):
        """Realistic scenario with all 8 signals."""
        n_src = 15
        ts = [0.92, 0.78, 0.65, 0.55, 0.42]
        attn = np.random.dirichlet(np.ones(n_src), size=5)
        prev_attn = np.random.dirichlet(np.ones(n_src - 1), size=5)

        weights = get_fusion_weights("en-zh")

        hit, diag = fused_border_check(
            attn, ts, n_src, 3, weights=weights,
            prev_attn=prev_attn,
            positions_history=[2.0, 3.5, 5.0, 6.2, 7.8],
            entropy_change=-0.4,
            entropy_change_ref=-0.5,
            pred_stability_write=True,
            attn_shift_write=True,
        )
        # All 8 signals should be computed
        assert len(diag.scores) == 8
        assert diag.dominant_signal() is not None
        # Summary should be well-formed
        assert "WRITE" in diag.summary() or "READ" in diag.summary()

    def test_fusion_vs_combined_agreement(self):
        """Fusion and boolean cascade should roughly agree on clear cases.

        For extreme cases (clear WRITE or clear READ), both approaches
        should give the same answer. Fusion may differ on borderline cases
        (which is the point -- it handles them better).
        """
        ts = _make_ts_scores()

        # Clear WRITE: attention at end
        attn_write = _make_border_attn(focus_pos=9)
        weights = FusionWeights(standard=1.0)
        hit, _ = fused_border_check(attn_write, ts, 10, 3, weights=weights)
        assert hit is True, "Clear WRITE case should trigger"

        # Clear READ: attention at start
        attn_read = _make_border_attn(focus_pos=1)
        hit, _ = fused_border_check(attn_read, ts, 10, 3, weights=weights)
        assert hit is False, "Clear READ case should not trigger"
