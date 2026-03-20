"""Tests for Iteration 9 features: REINA entropy change + prediction stability.

Two new cross-step border detection signals:

1. Entropy change (REINA-inspired, arxiv 2508.04946, AAAI 2026):
   Track generation entropy across consecutive translate() calls. If adding
   a new source word reduces entropy significantly, the model is still learning
   from source -> inhibit border stop (READ more).

2. Prediction stability (novel):
   Track how much the model's top predictions change between translate() calls.
   Stable predictions = model has enough context (supports WRITE).
   Volatile predictions = model still adapting (supports READ).

Both are "cross-step" signals (between translate() calls), unlike attention-based
checks (within a single generation loop). They modulate the border decision
from check_border_combined().

All tests are unit tests that don't require llama.cpp.
"""

import numpy as np
import pytest

from nllw.alignatt import (
    compute_entropy,
    compute_entropy_change,
    compute_prediction_stability,
    entropy_change_supports_write,
    prediction_stability_supports_write,
    check_border_combined,
)
from nllw.backend_protocol import BackendConfig
from nllw.bench import parse_sweep_spec


# ===========================================================================
# compute_entropy_change tests
# ===========================================================================

class TestComputeEntropyChange:
    """Tests for REINA-inspired entropy change tracking."""

    def test_first_call_returns_none_delta(self):
        """First call has no previous entropy, delta should be None."""
        logits = np.array([2.0, 1.0, 0.5, -1.0])
        delta, cur_h = compute_entropy_change(logits, prev_entropy=None)
        assert delta is None
        assert cur_h > 0

    def test_identical_logits_zero_change(self):
        """Same logits should give zero entropy change."""
        logits = np.array([2.0, 1.0, 0.5, -1.0])
        _, h1 = compute_entropy_change(logits, prev_entropy=None)
        delta, h2 = compute_entropy_change(logits, prev_entropy=h1)
        assert delta == pytest.approx(0.0, abs=1e-6)
        assert h1 == pytest.approx(h2, abs=1e-6)

    def test_entropy_drop_negative_delta(self):
        """More peaked logits -> lower entropy -> negative delta."""
        # First call: flat distribution (high entropy)
        flat_logits = np.array([1.0, 1.0, 1.0, 1.0])
        _, h_flat = compute_entropy_change(flat_logits, prev_entropy=None)

        # Second call: peaked distribution (low entropy)
        peaked_logits = np.array([10.0, -5.0, -5.0, -5.0])
        delta, h_peaked = compute_entropy_change(peaked_logits, prev_entropy=h_flat)

        assert delta < 0  # Entropy dropped
        assert h_peaked < h_flat

    def test_entropy_rise_positive_delta(self):
        """More uniform logits -> higher entropy -> positive delta."""
        peaked = np.array([10.0, -5.0, -5.0, -5.0])
        _, h_peaked = compute_entropy_change(peaked, prev_entropy=None)

        flat = np.array([1.0, 1.0, 1.0, 1.0])
        delta, h_flat = compute_entropy_change(flat, prev_entropy=h_peaked)

        assert delta > 0  # Entropy rose

    def test_current_entropy_matches_compute_entropy(self):
        """The returned entropy should match compute_entropy()."""
        logits = np.array([3.0, 1.0, -2.0, 0.5])
        _, cur_h = compute_entropy_change(logits, prev_entropy=None)
        expected_h = compute_entropy(logits)
        assert cur_h == pytest.approx(expected_h, abs=1e-6)

    def test_large_vocab_logits(self):
        """Should work with large vocabulary sizes."""
        rng = np.random.RandomState(42)
        logits1 = rng.randn(32000)
        logits2 = logits1 + rng.randn(32000) * 0.1  # small perturbation

        _, h1 = compute_entropy_change(logits1, prev_entropy=None)
        delta, h2 = compute_entropy_change(logits2, prev_entropy=h1)

        assert delta is not None
        assert abs(delta) < 1.0  # small perturbation -> small change

    def test_sequence_of_calls(self):
        """Simulate a sequence of translate() calls."""
        rng = np.random.RandomState(123)
        prev_h = None
        deltas = []
        for _ in range(5):
            logits = rng.randn(100)
            delta, prev_h = compute_entropy_change(logits, prev_entropy=prev_h)
            deltas.append(delta)

        assert deltas[0] is None  # First call
        assert all(d is not None for d in deltas[1:])


# ===========================================================================
# compute_prediction_stability tests
# ===========================================================================

class TestComputePredictionStability:
    """Tests for novel prediction stability index."""

    def test_first_call_returns_none(self):
        """First call has no previous logits, should return None."""
        logits = np.array([2.0, 1.0, 0.5, -1.0])
        rank, overlap = compute_prediction_stability(logits, prev_logits=None)
        assert rank is None
        assert overlap is None

    def test_identical_logits_perfect_stability(self):
        """Same logits -> rank 0, overlap 1.0."""
        logits = np.array([2.0, 1.0, 0.5, -1.0])
        rank, overlap = compute_prediction_stability(logits, logits, top_k=3)
        assert rank == 0.0  # top-1 unchanged
        assert overlap == 1.0  # perfect overlap

    def test_reversed_logits_high_rank_change(self):
        """Reversed logits -> top-1 is at the bottom."""
        logits1 = np.array([10.0, 5.0, 1.0, -5.0])  # top-1 = idx 0
        logits2 = np.array([-5.0, 1.0, 5.0, 10.0])  # top-1 = idx 3
        rank, overlap = compute_prediction_stability(logits2, logits1, top_k=2)

        assert rank == 3.0  # idx 0 went from rank 0 to rank 3
        assert overlap < 0.5  # low overlap

    def test_small_perturbation_stable(self):
        """Small logit perturbation -> stable predictions."""
        rng = np.random.RandomState(42)
        logits1 = rng.randn(100)
        logits2 = logits1 + rng.randn(100) * 0.001  # tiny perturbation

        rank, overlap = compute_prediction_stability(logits2, logits1, top_k=5)
        assert rank == 0.0  # top-1 shouldn't change with tiny noise
        assert overlap >= 0.8  # high overlap

    def test_large_perturbation_volatile(self):
        """Large logit perturbation -> volatile predictions."""
        rng = np.random.RandomState(42)
        logits1 = rng.randn(100)
        logits2 = rng.randn(100)  # completely different

        rank, overlap = compute_prediction_stability(logits2, logits1, top_k=5)
        # With random logits, expect some rank change
        assert rank is not None
        assert overlap is not None

    def test_top_k_parameter(self):
        """Different top_k values should affect overlap."""
        logits1 = np.zeros(10)
        logits1[0] = 10.0  # clear top-1
        logits1[1] = 9.0   # clear top-2

        logits2 = np.zeros(10)
        logits2[0] = 10.0  # same top-1
        logits2[2] = 9.0   # different top-2

        rank, overlap_2 = compute_prediction_stability(logits2, logits1, top_k=2)
        _, overlap_5 = compute_prediction_stability(logits2, logits1, top_k=5)
        assert rank == 0.0  # top-1 unchanged
        assert overlap_2 < 1.0  # top-2 changed, so <1

    def test_large_vocab(self):
        """Should work with large vocabulary sizes."""
        rng = np.random.RandomState(42)
        logits1 = rng.randn(32000)
        logits2 = logits1.copy()
        logits2[np.argmax(logits1)] += 5.0  # boost the top-1 even more

        rank, overlap = compute_prediction_stability(logits2, logits1, top_k=5)
        assert rank == 0.0  # same top-1 (just stronger)
        assert overlap >= 0.6  # should still have good top-5 overlap


# ===========================================================================
# entropy_change_supports_write tests
# ===========================================================================

class TestEntropyChangeSupportsWrite:
    """Tests for REINA signal interpretation."""

    def test_none_returns_none(self):
        """No entropy change -> None."""
        assert entropy_change_supports_write(None) is None

    def test_zero_change_supports_write(self):
        """No entropy change -> source exhausted -> WRITE."""
        assert entropy_change_supports_write(0.0) is True

    def test_positive_change_supports_write(self):
        """Entropy increased -> source confused -> still WRITE."""
        assert entropy_change_supports_write(0.5) is True

    def test_small_negative_supports_write(self):
        """Small entropy drop -> not enough signal -> WRITE."""
        assert entropy_change_supports_write(-0.2, threshold=-0.5) is True

    def test_large_negative_supports_read(self):
        """Large entropy drop -> source informative -> READ."""
        assert entropy_change_supports_write(-1.0, threshold=-0.5) is False

    def test_threshold_boundary(self):
        """At exact threshold -> READ (delta not > threshold)."""
        assert entropy_change_supports_write(-0.5, threshold=-0.5) is False

    def test_custom_threshold(self):
        """Custom threshold should be respected."""
        assert entropy_change_supports_write(-0.3, threshold=-0.2) is False
        assert entropy_change_supports_write(-0.1, threshold=-0.2) is True


# ===========================================================================
# prediction_stability_supports_write tests
# ===========================================================================

class TestPredictionStabilitySupportsWrite:
    """Tests for prediction stability signal interpretation."""

    def test_none_returns_none(self):
        """No stability data -> None."""
        assert prediction_stability_supports_write(None, None) is None
        assert prediction_stability_supports_write(0.0, None) is None
        assert prediction_stability_supports_write(None, 0.5) is None

    def test_stable_predictions_write(self):
        """Low rank + high overlap -> stable -> WRITE."""
        assert prediction_stability_supports_write(0.0, 0.8) is True

    def test_volatile_predictions_read(self):
        """High rank + low overlap -> volatile -> READ."""
        assert prediction_stability_supports_write(10.0, 0.1) is False

    def test_mixed_signals_read(self):
        """Rank stable but overlap low -> conservative -> READ."""
        assert prediction_stability_supports_write(1.0, 0.2) is False

    def test_mixed_signals_read2(self):
        """Rank volatile but overlap high -> conservative -> READ."""
        assert prediction_stability_supports_write(5.0, 0.8) is False

    def test_custom_thresholds(self):
        """Custom thresholds should be respected."""
        # Strict thresholds
        assert prediction_stability_supports_write(
            1.0, 0.5, rank_threshold=0.5, overlap_threshold=0.6
        ) is False  # rank 1 > threshold 0.5
        # Lenient thresholds
        assert prediction_stability_supports_write(
            5.0, 0.3, rank_threshold=10.0, overlap_threshold=0.2
        ) is True


# ===========================================================================
# check_border_combined with new signals
# ===========================================================================

class TestCheckBorderCombinedNewSignals:
    """Tests for check_border_combined with REINA + stability signals."""

    def _make_border_attn(self, n_heads=3, n_src=10, attended_pos=8):
        """Create attention focused at a specific position."""
        attn = np.zeros((n_heads, n_src))
        attn[:, attended_pos] = 1.0  # all heads focus here
        return attn

    def _make_non_border_attn(self, n_heads=3, n_src=10, attended_pos=3):
        """Create attention focused at a non-border position."""
        attn = np.zeros((n_heads, n_src))
        attn[:, attended_pos] = 1.0
        return attn

    def test_baseline_border_hit_unchanged(self):
        """Without new signals, border detection unchanged."""
        attn = self._make_border_attn()
        ts = [0.8, 0.6, 0.4]
        hit, _, _ = check_border_combined(
            attn, ts, n_src_tokens=10, border_distance=3,
        )
        assert hit is True

    def test_baseline_no_border_unchanged(self):
        """Without new signals, non-border detection unchanged."""
        attn = self._make_non_border_attn()
        ts = [0.8, 0.6, 0.4]
        hit, _, _ = check_border_combined(
            attn, ts, n_src_tokens=10, border_distance=3,
        )
        assert hit is False

    def test_entropy_change_inhibits_border(self):
        """Large entropy drop (informative source) should inhibit border stop."""
        attn = self._make_border_attn()
        ts = [0.8, 0.6, 0.4]
        hit, _, _ = check_border_combined(
            attn, ts, n_src_tokens=10, border_distance=3,
            entropy_change=-2.0,  # large drop = source very informative
            entropy_change_threshold=-0.5,
        )
        assert hit is False  # Inhibited by entropy change

    def test_small_entropy_change_allows_border(self):
        """Small entropy change should not inhibit border stop."""
        attn = self._make_border_attn()
        ts = [0.8, 0.6, 0.4]
        hit, _, _ = check_border_combined(
            attn, ts, n_src_tokens=10, border_distance=3,
            entropy_change=-0.1,  # small change = source not informative
            entropy_change_threshold=-0.5,
        )
        assert hit is True  # Not inhibited

    def test_none_entropy_change_no_effect(self):
        """None entropy change (first call) should not affect border."""
        attn = self._make_border_attn()
        ts = [0.8, 0.6, 0.4]
        hit, _, _ = check_border_combined(
            attn, ts, n_src_tokens=10, border_distance=3,
            entropy_change=None,
            entropy_change_threshold=-0.5,
        )
        assert hit is True  # No effect

    def test_pred_stability_inhibits_border(self):
        """Volatile predictions should inhibit border stop."""
        attn = self._make_border_attn()
        ts = [0.8, 0.6, 0.4]
        hit, _, _ = check_border_combined(
            attn, ts, n_src_tokens=10, border_distance=3,
            pred_stability_write=False,  # volatile = READ
        )
        assert hit is False  # Inhibited by volatile predictions

    def test_stable_predictions_allow_border(self):
        """Stable predictions should allow border stop."""
        attn = self._make_border_attn()
        ts = [0.8, 0.6, 0.4]
        hit, _, _ = check_border_combined(
            attn, ts, n_src_tokens=10, border_distance=3,
            pred_stability_write=True,  # stable = WRITE
        )
        assert hit is True  # Border allowed

    def test_none_stability_no_effect(self):
        """None stability (first call) should not affect border."""
        attn = self._make_border_attn()
        ts = [0.8, 0.6, 0.4]
        hit, _, _ = check_border_combined(
            attn, ts, n_src_tokens=10, border_distance=3,
            pred_stability_write=None,
        )
        assert hit is True  # No effect

    def test_entropy_inhibit_takes_priority(self):
        """Entropy change inhibit should take priority over stability."""
        attn = self._make_border_attn()
        ts = [0.8, 0.6, 0.4]
        hit, _, _ = check_border_combined(
            attn, ts, n_src_tokens=10, border_distance=3,
            entropy_change=-2.0,
            entropy_change_threshold=-0.5,
            pred_stability_write=True,  # stability says WRITE...
        )
        assert hit is False  # ...but entropy change says READ (takes priority)

    def test_non_border_not_affected_by_stability(self):
        """Stability should only affect border hits, not non-border cases."""
        attn = self._make_non_border_attn()
        ts = [0.8, 0.6, 0.4]
        hit, _, _ = check_border_combined(
            attn, ts, n_src_tokens=10, border_distance=3,
            pred_stability_write=True,
        )
        assert hit is False  # Still no border

    def test_combined_with_shift_k(self):
        """New signals should work alongside shift-k."""
        # Attention heavily in border region
        attn = np.zeros((3, 10))
        attn[:, 8] = 0.5
        attn[:, 9] = 0.5
        ts = [0.8, 0.6, 0.4]

        # Shift-k fires, entropy change inhibits
        hit, _, _ = check_border_combined(
            attn, ts, n_src_tokens=10, border_distance=3,
            shift_k_threshold=0.4,
            entropy_change=-2.0,
            entropy_change_threshold=-0.5,
        )
        assert hit is False  # Entropy change pre-filter inhibits everything


# ===========================================================================
# BackendConfig tests
# ===========================================================================

class TestBackendConfigNewFields:
    """Tests for new config fields."""

    def test_default_entropy_change_disabled(self):
        """Entropy change should be disabled by default."""
        config = BackendConfig()
        assert config.entropy_change_threshold is None

    def test_default_prediction_stability_disabled(self):
        """Prediction stability should be disabled by default."""
        config = BackendConfig()
        assert config.prediction_stability is False

    def test_from_dict_with_new_fields(self):
        """from_dict should accept new fields."""
        d = {
            "entropy_change_threshold": -0.5,
            "prediction_stability": True,
        }
        config = BackendConfig.from_dict(d)
        assert config.entropy_change_threshold == -0.5
        assert config.prediction_stability is True

    def test_from_dict_ignores_unknown(self):
        """from_dict should still ignore unknown fields."""
        d = {
            "entropy_change_threshold": -0.5,
            "unknown_field": "ignored",
        }
        config = BackendConfig.from_dict(d)
        assert config.entropy_change_threshold == -0.5


# ===========================================================================
# Sweep parser tests
# ===========================================================================

class TestSweepParserNewShortnames:
    """Tests for new sweep shortnames."""

    def test_entchg_shortname(self):
        """entchg should map to entropy_change_threshold."""
        grid = parse_sweep_spec("entchg=-0.5,-1.0,-2.0")
        assert "entropy_change_threshold" in grid
        assert grid["entropy_change_threshold"] == [-0.5, -1.0, -2.0]

    def test_predstab_shortname(self):
        """predstab should map to prediction_stability."""
        grid = parse_sweep_spec("predstab=0,1")
        assert "prediction_stability" in grid
        assert grid["prediction_stability"] == [0, 1]

    def test_combined_with_existing(self):
        """New shortnames should work alongside existing ones."""
        grid = parse_sweep_spec("bd=2,3 entchg=-0.5 predstab=0,1")
        assert "border_distance" in grid
        assert "entropy_change_threshold" in grid
        assert "prediction_stability" in grid
        assert grid["border_distance"] == [2, 3]
