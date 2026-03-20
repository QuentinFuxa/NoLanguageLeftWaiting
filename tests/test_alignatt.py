"""Tests for AlignAtt core algorithm."""

import numpy as np
import pytest
from nllw.alignatt import (
    aggregate_ts_weighted_vote,
    check_border,
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
