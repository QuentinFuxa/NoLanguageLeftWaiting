"""Tests for AlignAtt core algorithm."""

import numpy as np
import pytest
from nllw.alignatt import (
    aggregate,
    aggregate_cumulative_attention,
    aggregate_ts_weighted_vote,
    check_border,
    check_border_combined,
    check_border_shift_k,
    compute_attention_info_gain,
    compute_dynamic_word_batch,
    compute_entropy,
    source_lookahead_top_prob,
    is_target_language,
    adaptive_border_distance,
)


class TestTSWeightedVote:
    def test_single_head(self):
        """Single head, attention peaks at position 3."""
        src_attn = np.array([[0.1, 0.1, 0.1, 0.7]])  # shape (1, 4)
        ts_scores = [1.0]
        assert aggregate_ts_weighted_vote(src_attn, ts_scores) == 3

    def test_two_heads_agree(self):
        """Two heads both point to position 2."""
        src_attn = np.array([
            [0.1, 0.1, 0.8, 0.0],
            [0.0, 0.1, 0.9, 0.0],
        ])
        ts_scores = [0.8, 0.7]
        assert aggregate_ts_weighted_vote(src_attn, ts_scores) == 2

    def test_two_heads_disagree(self):
        """Two heads disagree; higher TS wins."""
        src_attn = np.array([
            [0.1, 0.9, 0.0, 0.0],  # points to 1
            [0.0, 0.0, 0.0, 1.0],  # points to 3
        ])
        ts_scores = [0.3, 0.9]  # head 1 has higher TS
        assert aggregate_ts_weighted_vote(src_attn, ts_scores) == 3


class TestCheckBorder:
    def test_no_border(self):
        """Attention at start of source -> no border."""
        src_attn = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])
        ts_scores = [1.0]
        assert check_border(src_attn, ts_scores, n_src_tokens=5, border_distance=2) is False

    def test_border_hit(self):
        """Attention at end of source -> border hit."""
        src_attn = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]])
        ts_scores = [1.0]
        assert check_border(src_attn, ts_scores, n_src_tokens=5, border_distance=2) is True

    def test_border_distance_threshold(self):
        """Position 3 with 5 tokens and bd=2: threshold=3, pos=3 >= 3 -> border."""
        src_attn = np.array([[0.0, 0.0, 0.0, 1.0, 0.0]])
        ts_scores = [1.0]
        assert check_border(src_attn, ts_scores, n_src_tokens=5, border_distance=2) is True

    def test_negative_threshold_guard(self):
        """When n_src < border_distance, guard returns False."""
        src_attn = np.array([[1.0, 0.0]])
        ts_scores = [1.0]
        assert check_border(src_attn, ts_scores, n_src_tokens=2, border_distance=5) is False


class TestEntropy:
    def test_uniform(self):
        """Uniform distribution has high entropy."""
        logits = np.zeros(100)  # equal logits -> uniform
        ent = compute_entropy(logits)
        assert ent > 4.0  # ln(100) ~ 4.6

    def test_peaked(self):
        """Peaked distribution has low entropy."""
        logits = np.zeros(100)
        logits[0] = 100.0  # very peaked
        ent = compute_entropy(logits)
        assert ent < 0.01


class TestSourceLookahead:
    def test_confident_prediction(self):
        """Very peaked logits -> high top probability."""
        logits = np.zeros(1000)
        logits[42] = 50.0
        prob, idx = source_lookahead_top_prob(logits)
        assert idx == 42
        assert prob > 0.99

    def test_uniform_prediction(self):
        """Uniform logits -> low top probability."""
        logits = np.zeros(1000)
        prob, idx = source_lookahead_top_prob(logits)
        assert prob < 0.01


class TestIsTargetLanguage:
    def test_chinese(self):
        assert is_target_language("这是中文测试", "zh") is True
        assert is_target_language("This is English", "zh") is False

    def test_japanese(self):
        assert is_target_language("これはテストです", "ja") is True

    def test_arabic(self):
        assert is_target_language("هذا اختبار", "ar") is True

    def test_latin_always_true(self):
        """Latin-script languages can't be distinguished from English."""
        assert is_target_language("Ceci est un test", "fr") is True
        assert is_target_language("Hello world", "fr") is True

    def test_empty(self):
        assert is_target_language("", "zh") is False
        assert is_target_language("   ", "zh") is False


class TestAdaptiveBorderDistance:
    def test_high_confidence(self):
        """High confidence -> no adjustment."""
        assert adaptive_border_distance(3, confidence=1.0) == 3

    def test_low_confidence(self):
        """Low confidence -> larger border."""
        bd = adaptive_border_distance(3, confidence=0.0, alpha=2.0)
        assert bd == 5  # 3 + round(2.0 * 1.0) = 5

    def test_minimum_one(self):
        """Border distance never goes below 1."""
        assert adaptive_border_distance(0, confidence=1.0) >= 1


class TestDynamicWordBatch:
    def test_short_sentence_reduces_wb(self):
        """Short sentences (< 8 words) get wb - 1."""
        assert compute_dynamic_word_batch(3, n_source_words=4) == 2
        assert compute_dynamic_word_batch(3, n_source_words=7) == 2

    def test_medium_sentence_keeps_wb(self):
        """Medium sentences (8-20 words) keep base wb."""
        assert compute_dynamic_word_batch(3, n_source_words=10) == 3
        assert compute_dynamic_word_batch(3, n_source_words=15) == 3
        assert compute_dynamic_word_batch(3, n_source_words=20) == 3

    def test_long_sentence_increases_wb(self):
        """Long sentences (> 20 words) get wb + 1."""
        assert compute_dynamic_word_batch(3, n_source_words=25) == 4
        assert compute_dynamic_word_batch(2, n_source_words=30) == 3

    def test_minimum_wb_is_one(self):
        """Word batch never goes below 1."""
        assert compute_dynamic_word_batch(1, n_source_words=3) >= 1

    def test_custom_thresholds(self):
        """Custom short/long thresholds work."""
        assert compute_dynamic_word_batch(3, n_source_words=3, short_threshold=5) == 2
        assert compute_dynamic_word_batch(3, n_source_words=12, long_threshold=10) == 4


class TestAttentionInfoGain:
    def test_identical_attention_zero_kl(self):
        """Identical attention distributions have near-zero info gain."""
        attn = np.array([[0.1, 0.2, 0.3, 0.4]])
        ig = compute_attention_info_gain(attn, attn, [1.0])
        assert ig < 0.01

    def test_different_attention_high_kl(self):
        """Very different attention distributions have high info gain."""
        prev = np.array([[0.9, 0.05, 0.03, 0.02]])
        curr = np.array([[0.02, 0.03, 0.05, 0.9]])
        ig = compute_attention_info_gain(prev, curr, [1.0])
        assert ig > 1.0  # Large shift

    def test_slight_shift_small_kl(self):
        """Slight attention shift gives small but nonzero info gain."""
        prev = np.array([[0.1, 0.7, 0.15, 0.05]])
        curr = np.array([[0.05, 0.6, 0.25, 0.1]])
        ig = compute_attention_info_gain(prev, curr, [1.0])
        assert 0.01 < ig < 1.0

    def test_multi_head_ts_weighting(self):
        """TS weighting affects the result -- high-TS head dominates."""
        prev = np.array([
            [0.8, 0.1, 0.05, 0.05],  # head 0: stable
            [0.9, 0.05, 0.03, 0.02],  # head 1: shifts a lot
        ])
        curr = np.array([
            [0.75, 0.15, 0.05, 0.05],  # head 0: barely moved
            [0.02, 0.03, 0.05, 0.9],   # head 1: shifted completely
        ])
        # Head 1 has high TS -> should dominate
        ig_high_ts1 = compute_attention_info_gain(prev, curr, [0.1, 0.9])
        ig_high_ts0 = compute_attention_info_gain(prev, curr, [0.9, 0.1])
        assert ig_high_ts1 > ig_high_ts0  # When shifting head has high TS, IG is higher


class TestShiftKBorder:
    def test_no_mass_in_border(self):
        """All attention at start -> no border."""
        src_attn = np.array([[0.8, 0.15, 0.03, 0.01, 0.01]])
        assert check_border_shift_k(
            src_attn, [1.0], n_src_tokens=5, border_distance=2, threshold=0.4
        ) is False

    def test_high_mass_in_border(self):
        """Most attention at end -> border hit."""
        src_attn = np.array([[0.05, 0.05, 0.05, 0.35, 0.5]])
        assert check_border_shift_k(
            src_attn, [1.0], n_src_tokens=5, border_distance=2, threshold=0.4
        ) is True

    def test_spread_mass_below_threshold(self):
        """Attention spread evenly, border mass below threshold."""
        src_attn = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]])
        # Border region = last 2 positions = 0.4 mass exactly
        assert check_border_shift_k(
            src_attn, [1.0], n_src_tokens=5, border_distance=2, threshold=0.5
        ) is False

    def test_multi_head_weighted(self):
        """Multiple heads, TS-weighted border mass."""
        src_attn = np.array([
            [0.0, 0.0, 0.0, 0.0, 1.0],  # head 0: all mass in border
            [1.0, 0.0, 0.0, 0.0, 0.0],  # head 1: no mass in border
        ])
        # Only head 0 has high TS -> border hit
        assert check_border_shift_k(
            src_attn, [0.9, 0.1], n_src_tokens=5, border_distance=2, threshold=0.4
        ) is True
        # Only head 1 has high TS -> no border
        assert check_border_shift_k(
            src_attn, [0.1, 0.9], n_src_tokens=5, border_distance=2, threshold=0.4
        ) is False

    def test_negative_border_guard(self):
        """Small n_src with large bd returns False."""
        src_attn = np.array([[0.5, 0.5]])
        assert check_border_shift_k(
            src_attn, [1.0], n_src_tokens=2, border_distance=5, threshold=0.4
        ) is False


class TestCombinedBorderCheck:
    def _make_attn(self, peaks):
        """Helper: create attention with peaks at given positions."""
        n_src = 8
        attn = np.zeros((len(peaks), n_src))
        for h, p in enumerate(peaks):
            attn[h, p] = 0.8
            for i in range(n_src):
                if i != p:
                    attn[h, i] = 0.2 / (n_src - 1)
        return attn

    def test_combined_standard_border(self):
        """Without shift-k or info-gain, falls back to standard."""
        attn = self._make_attn([7])  # Peak at end -> border
        hit, ig, bm = check_border_combined(
            attn, [1.0], n_src_tokens=8, border_distance=2
        )
        assert hit is True
        assert ig is None  # Not computed
        assert bm is None

    def test_shift_k_fires(self):
        """Shift-k detects border mass."""
        attn = self._make_attn([7])
        hit, ig, bm = check_border_combined(
            attn, [1.0], n_src_tokens=8, border_distance=2,
            shift_k_threshold=0.3,
        )
        assert hit is True
        assert bm is not None
        assert bm > 0.3

    def test_info_gain_inhibits_stop(self):
        """High info gain inhibits border stop."""
        attn = self._make_attn([7])  # Would normally hit border
        # Create prev_attn that's very different (high info gain)
        prev_attn = self._make_attn([0])
        hit, ig, _ = check_border_combined(
            attn, [1.0], n_src_tokens=8, border_distance=2,
            info_gain_threshold=0.3,
            prev_attn=prev_attn,
        )
        assert ig is not None
        assert ig > 0.3 * 3  # Very high info gain
        assert hit is False  # Inhibited

    def test_info_gain_reinforces_stop(self):
        """Low info gain reinforces border stop with shift-k."""
        attn = self._make_attn([7])
        prev_attn = attn.copy()  # Same attention -> low info gain
        hit, ig, bm = check_border_combined(
            attn, [1.0], n_src_tokens=8, border_distance=2,
            shift_k_threshold=0.3,
            info_gain_threshold=0.3,
            prev_attn=prev_attn,
        )
        assert hit is True
        assert ig is not None
        assert ig < 0.3  # Low info gain

    def test_no_border_when_attending_start(self):
        """Attention at start -> no border with any method."""
        attn = self._make_attn([0])
        hit, _, _ = check_border_combined(
            attn, [1.0], n_src_tokens=8, border_distance=2,
            shift_k_threshold=0.3,
        )
        assert hit is False


class TestCumulativeAttention:
    """Tests for DrFrattn-inspired cumulative attention aggregation."""

    def test_sharp_attention_at_start(self):
        """Sharp attention at position 0 -> frontier at 0."""
        attn = np.array([[0.95, 0.02, 0.01, 0.01, 0.01]])
        pos = aggregate_cumulative_attention(attn, [1.0], lambda_threshold=0.5)
        assert pos == 0  # Almost all mass at start, nothing remaining

    def test_sharp_attention_at_end(self):
        """Sharp attention at last position -> frontier at end."""
        attn = np.array([[0.01, 0.01, 0.01, 0.01, 0.96]])
        pos = aggregate_cumulative_attention(attn, [1.0], lambda_threshold=0.5)
        assert pos == 3 or pos == 4  # Frontier near end

    def test_uniform_attention(self):
        """Uniform attention -> frontier depends on lambda."""
        attn = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]])
        # With lambda=0.5, frontier should be around position 2
        # cumsum = [0.2, 0.4, 0.6, 0.8, 1.0], remaining = [0.8, 0.6, 0.4, 0.2, 0.0]
        # remaining >= 0.5 at positions 0, 1 -> frontier = 1
        pos = aggregate_cumulative_attention(attn, [1.0], lambda_threshold=0.5)
        assert pos == 1

    def test_lambda_sensitivity(self):
        """Lower lambda = more aggressive (later frontier)."""
        attn = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]])
        pos_conservative = aggregate_cumulative_attention(attn, [1.0], lambda_threshold=0.5)
        pos_aggressive = aggregate_cumulative_attention(attn, [1.0], lambda_threshold=0.2)
        assert pos_aggressive >= pos_conservative  # Lower lambda = later frontier

    def test_multi_head_voting(self):
        """Multiple heads vote on frontier position."""
        attn = np.array([
            [0.0, 0.0, 0.0, 0.9, 0.1],  # head 0: sharp at 3
            [0.0, 0.0, 0.8, 0.1, 0.1],  # head 1: sharp at 2
        ])
        # Head 0 (higher TS) should win
        pos = aggregate_cumulative_attention(attn, [0.9, 0.3], lambda_threshold=0.3)
        assert pos >= 2  # Frontier at 2 or 3

    def test_split_attention_captures_frontier(self):
        """Split attention between two positions captures rightmost."""
        attn = np.array([[0.0, 0.0, 0.4, 0.0, 0.4, 0.2]])
        # cumsum = [0, 0, 0.4, 0.4, 0.8, 1.0], remaining = [1.0, 1.0, 0.6, 0.6, 0.2, 0.0]
        # remaining >= 0.3 at positions 0,1,2,3 -> frontier = 3
        pos = aggregate_cumulative_attention(attn, [1.0], lambda_threshold=0.3)
        assert pos >= 2  # Captures that mass extends past position 2

    def test_registered_in_aggregation(self):
        """Cumulative is registered and usable via aggregate()."""
        attn = np.array([[0.1, 0.2, 0.3, 0.4]])
        pos = aggregate(attn, [1.0], method="cumulative")
        assert isinstance(pos, (int, float))

    def test_empty_source(self):
        """Empty source returns 0."""
        attn = np.zeros((1, 0))
        pos = aggregate_cumulative_attention(attn, [1.0])
        assert pos == 0


# ---------------------------------------------------------------------------
# Perplexity-based adaptive border tests (Hibiki-inspired)
# ---------------------------------------------------------------------------

class TestTokenPerplexity:
    """Test compute_token_perplexity()."""

    def test_confident_token(self):
        """High probability token -> low perplexity."""
        from nllw.alignatt import compute_token_perplexity
        # Logits that produce P(token_0) ~ 0.95
        logits = np.array([5.0, 0.0, 0.0, 0.0])
        ppl = compute_token_perplexity(logits, 0)
        assert ppl < 2.0  # Very confident

    def test_uncertain_token(self):
        """Low probability token -> high perplexity."""
        from nllw.alignatt import compute_token_perplexity
        # Uniform logits
        logits = np.array([1.0, 1.0, 1.0, 1.0])
        ppl = compute_token_perplexity(logits, 0)
        assert ppl > 3.0  # 4-way uniform = ppl 4

    def test_perplexity_minimum(self):
        """Perplexity is always >= 1.0 for the argmax token."""
        from nllw.alignatt import compute_token_perplexity
        logits = np.array([100.0, 0.0])
        ppl = compute_token_perplexity(logits, 0)
        assert ppl >= 1.0


class TestGenerationPerplexity:
    """Test compute_generation_perplexity()."""

    def test_single_token(self):
        from nllw.alignatt import compute_generation_perplexity
        ppl = compute_generation_perplexity([2.5])
        assert ppl == pytest.approx(2.5, abs=0.01)

    def test_multiple_tokens(self):
        from nllw.alignatt import compute_generation_perplexity
        # Geometric mean of [2, 8] = sqrt(16) = 4
        ppl = compute_generation_perplexity([2.0, 8.0])
        assert ppl == pytest.approx(4.0, abs=0.01)

    def test_empty(self):
        from nllw.alignatt import compute_generation_perplexity
        assert compute_generation_perplexity([]) == 1.0


class TestPerplexityBorderAdjustment:
    """Test perplexity_border_adjustment()."""

    def test_confident_tightens(self):
        """Low perplexity -> tighten border (bd-1)."""
        from nllw.alignatt import perplexity_border_adjustment
        bd = perplexity_border_adjustment(1.5, base_bd=3, low_threshold=2.0)
        assert bd == 2  # 3 - 1

    def test_uncertain_widens(self):
        """High perplexity -> widen border (bd+1)."""
        from nllw.alignatt import perplexity_border_adjustment
        bd = perplexity_border_adjustment(6.0, base_bd=3, high_threshold=5.0)
        assert bd == 4  # 3 + 1

    def test_normal_no_change(self):
        """Middle perplexity -> no change."""
        from nllw.alignatt import perplexity_border_adjustment
        bd = perplexity_border_adjustment(3.5, base_bd=3, low_threshold=2.0, high_threshold=5.0)
        assert bd == 3

    def test_minimum_bd_1(self):
        """Border distance never goes below 1."""
        from nllw.alignatt import perplexity_border_adjustment
        bd = perplexity_border_adjustment(0.5, base_bd=1, low_threshold=2.0)
        assert bd == 1  # Can't go below 1

    def test_config_field_exists(self):
        """BackendConfig has perplexity_adaptive_bd field."""
        from nllw.backend_protocol import BackendConfig
        config = BackendConfig()
        assert hasattr(config, 'perplexity_adaptive_bd')
        assert config.perplexity_adaptive_bd is False
        assert config.perplexity_bd_low == 2.0
        assert config.perplexity_bd_high == 5.0
