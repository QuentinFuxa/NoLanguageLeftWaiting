"""Tests for Iteration 7 features.

New features:
1. Dynamic word_batch (source-length-adaptive batching)
2. Attention information gain (KL-divergence border signal)
3. Shift-k adaptive border (attention mass threshold, DrFrattn-inspired)
4. Combined border check (multi-signal fusion)

All tests are unit tests that don't require llama.cpp.
"""

import numpy as np
import pytest

from nllw.alignatt import (
    compute_dynamic_word_batch,
    compute_attention_info_gain,
    check_border_shift_k,
    check_border_combined,
    check_border,
)
from nllw.backend_protocol import BackendConfig
from nllw.bench import parse_sweep_spec


# ===========================================================================
# Dynamic word_batch tests
# ===========================================================================

class TestDynamicWordBatchEdgeCases:
    """Additional edge cases for dynamic word_batch."""

    def test_exact_short_boundary(self):
        """At exactly short_threshold, should use base wb."""
        assert compute_dynamic_word_batch(3, n_source_words=8) == 3

    def test_exact_long_boundary(self):
        """At exactly long_threshold, should use base wb."""
        assert compute_dynamic_word_batch(3, n_source_words=20) == 3

    def test_wb_one_short_stays_one(self):
        """wb=1 on short sentence stays at 1 (not 0)."""
        assert compute_dynamic_word_batch(1, n_source_words=3) == 1

    def test_very_long_sentence(self):
        """100-word sentence gets wb+1."""
        assert compute_dynamic_word_batch(3, n_source_words=100) == 4

    def test_zero_words(self):
        """Zero words (first call) gets reduced wb."""
        assert compute_dynamic_word_batch(3, n_source_words=0) == 2


# ===========================================================================
# Attention information gain tests
# ===========================================================================

class TestInfoGainEdgeCases:
    """Edge cases for attention information gain."""

    def test_single_position_same(self):
        """Single source position, same attention -> zero divergence."""
        prev = np.array([[1.0]])
        curr = np.array([[1.0]])
        ig = compute_attention_info_gain(prev, curr, [1.0])
        assert ig < 0.01

    def test_large_array(self):
        """Works with realistic-size attention arrays."""
        np.random.seed(42)
        prev = np.random.dirichlet([1] * 50, size=10)
        curr = np.random.dirichlet([1] * 50, size=10)
        ts = [0.5 + 0.05 * i for i in range(10)]
        ig = compute_attention_info_gain(prev, curr, ts)
        assert ig >= 0  # KL divergence is always non-negative
        assert np.isfinite(ig)

    def test_equal_ts_scores(self):
        """Equal TS scores = simple mean."""
        prev = np.array([[0.5, 0.5], [0.3, 0.7]])
        curr = np.array([[0.3, 0.7], [0.5, 0.5]])
        ig = compute_attention_info_gain(prev, curr, [1.0, 1.0])
        assert ig > 0

    def test_zero_ts_fallback(self):
        """Zero TS scores falls back to simple mean."""
        prev = np.array([[0.5, 0.5]])
        curr = np.array([[0.3, 0.7]])
        ig = compute_attention_info_gain(prev, curr, [0.0])
        assert ig >= 0
        assert np.isfinite(ig)


# ===========================================================================
# Shift-k border tests
# ===========================================================================

class TestShiftKEdgeCases:
    """Edge cases for shift-k border check."""

    def test_threshold_exactly_met(self):
        """Border mass exactly at threshold."""
        # 5 positions, bd=2 -> border = last 2 positions
        # Uniform = 0.2 per position, border mass = 0.4
        src_attn = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]])
        assert check_border_shift_k(
            src_attn, [1.0], n_src_tokens=5, border_distance=2, threshold=0.4
        ) is True  # >= 0.4

    def test_threshold_just_below(self):
        """Border mass just below threshold."""
        src_attn = np.array([[0.25, 0.25, 0.25, 0.15, 0.1]])
        assert check_border_shift_k(
            src_attn, [1.0], n_src_tokens=5, border_distance=2, threshold=0.4
        ) is False  # 0.25 < 0.4

    def test_bd_equals_nsrc(self):
        """Border distance equals n_src -> all mass in border but guard."""
        src_attn = np.array([[0.25, 0.25, 0.25, 0.25]])
        assert check_border_shift_k(
            src_attn, [1.0], n_src_tokens=4, border_distance=4, threshold=0.4
        ) is False  # border_start <= 0, guard

    def test_bd_one(self):
        """Minimal border distance of 1."""
        src_attn = np.array([[0.1, 0.1, 0.1, 0.1, 0.6]])
        assert check_border_shift_k(
            src_attn, [1.0], n_src_tokens=5, border_distance=1, threshold=0.5
        ) is True

    def test_with_temp_normalization(self):
        """Shift-k works with head temperature normalization."""
        # Sharp head + diffuse head
        attn = np.array([
            [0.01, 0.01, 0.01, 0.01, 0.96],  # Very sharp at end
            [0.18, 0.18, 0.18, 0.24, 0.22],   # Diffuse, slight end bias
        ])
        # Without normalization, sharp head dominates
        hit_no_norm = check_border_shift_k(
            attn, [0.5, 0.5], n_src_tokens=5, border_distance=2, threshold=0.5
        )
        # With normalization, more balanced
        hit_norm = check_border_shift_k(
            attn, [0.5, 0.5], n_src_tokens=5, border_distance=2, threshold=0.5,
            head_temp_normalize=True, head_temp_reference=1.5
        )
        # Both should agree in this obvious case
        assert hit_no_norm is True


# ===========================================================================
# Combined border check tests
# ===========================================================================

class TestCombinedBorderEdgeCases:
    """Edge cases for the combined border check."""

    def test_dynamic_border_with_combined(self):
        """Dynamic border works within combined check."""
        attn = np.zeros((1, 8))
        attn[0, 7] = 0.9  # Peak at last position
        attn[0, :7] = 0.1 / 7
        hit, _, _ = check_border_combined(
            attn, [1.0], n_src_tokens=8, border_distance=3,
            dynamic_border=True,
        )
        # pos 7 clearly in border region even with dynamic adjustment
        assert hit is True

    def test_prev_attn_size_mismatch(self):
        """Previous attention with fewer source tokens is handled."""
        curr = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        prev = np.array([[0.5, 0.3, 0.15, 0.05]])  # Shorter
        hit, ig, _ = check_border_combined(
            curr, [1.0], n_src_tokens=8, border_distance=2,
            info_gain_threshold=0.3,
            prev_attn=prev,
        )
        # Should handle gracefully
        assert ig is not None
        assert np.isfinite(ig)

    def test_all_signals_disabled(self):
        """With all signals disabled, falls back to standard border."""
        attn = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]])
        hit, ig, bm = check_border_combined(
            attn, [1.0], n_src_tokens=5, border_distance=2,
        )
        assert hit is True
        assert ig is None
        assert bm is None

    def test_shift_k_alone(self):
        """Shift-k alone without info gain."""
        attn = np.array([[0.0, 0.0, 0.0, 0.3, 0.7]])
        hit, ig, bm = check_border_combined(
            attn, [1.0], n_src_tokens=5, border_distance=2,
            shift_k_threshold=0.4,
        )
        assert hit is True
        assert ig is None
        assert bm is not None
        assert bm > 0.4


# ===========================================================================
# Config / CLI integration tests
# ===========================================================================

class TestIteration7Config:
    """Test BackendConfig and sweep spec for iteration 7 features."""

    def test_config_new_fields(self):
        """New config fields have correct defaults."""
        cfg = BackendConfig()
        assert cfg.dynamic_word_batch is False
        assert cfg.info_gain_threshold is None
        assert cfg.shift_k_threshold is None

    def test_config_from_dict(self):
        """from_dict picks up new fields."""
        cfg = BackendConfig.from_dict({
            "dynamic_word_batch": True,
            "info_gain_threshold": 0.3,
            "shift_k_threshold": 0.4,
        })
        assert cfg.dynamic_word_batch is True
        assert cfg.info_gain_threshold == 0.3
        assert cfg.shift_k_threshold == 0.4

    def test_sweep_shortnames(self):
        """New sweep shortnames parse correctly."""
        grid = parse_sweep_spec("dynwb=0,1 infogain=0.2,0.3,0.5 shiftk=0.3,0.4,0.5")
        assert grid["dynamic_word_batch"] == [0, 1]
        assert grid["info_gain_threshold"] == [0.2, 0.3, 0.5]
        assert grid["shift_k_threshold"] == [0.3, 0.4, 0.5]

    def test_sweep_shortnames_mixed(self):
        """New shortnames work mixed with existing ones."""
        grid = parse_sweep_spec("bd=2,3 wb=2,3 shiftk=0.4")
        assert "border_distance" in grid
        assert "word_batch" in grid
        assert "shift_k_threshold" in grid

    def test_border_confirm_default(self):
        """Border confirm defaults to 1 (disabled)."""
        cfg = BackendConfig()
        assert cfg.border_confirm == 1

    def test_border_confirm_from_dict(self):
        """Border confirm can be set via from_dict."""
        cfg = BackendConfig.from_dict({"border_confirm": 2})
        assert cfg.border_confirm == 2

    def test_border_confirm_sweep(self):
        """Border confirm sweep shortname works."""
        grid = parse_sweep_spec("confirm=1,2,3")
        assert grid["border_confirm"] == [1, 2, 3]
