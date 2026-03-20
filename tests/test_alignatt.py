"""Tests for AlignAtt core algorithms (no GPU required)."""

import numpy as np
import pytest

from nllw.alignatt import (
    AlignAttConfig,
    AlignmentHead,
    aggregate_ts_weighted_vote,
    is_target_language,
)


class TestTSWeightedVote:
    def test_single_head(self):
        """Single head votes for position with highest attention."""
        src_attn = np.array([[0.1, 0.2, 0.7]])  # 1 head, 3 src tokens
        ts_scores = [1.0]
        assert aggregate_ts_weighted_vote(src_attn, ts_scores) == 2

    def test_two_heads_agree(self):
        """Two heads both attend to same position."""
        src_attn = np.array([
            [0.1, 0.1, 0.8],  # head 0 -> pos 2
            [0.1, 0.2, 0.7],  # head 1 -> pos 2
        ])
        ts_scores = [0.5, 0.3]
        assert aggregate_ts_weighted_vote(src_attn, ts_scores) == 2

    def test_weighted_disagreement(self):
        """Heads disagree; higher-TS head should win."""
        src_attn = np.array([
            [0.9, 0.05, 0.05],  # head 0 -> pos 0 (TS=0.8)
            [0.1, 0.1, 0.8],    # head 1 -> pos 2 (TS=0.2)
        ])
        ts_scores = [0.8, 0.2]
        assert aggregate_ts_weighted_vote(src_attn, ts_scores) == 0

    def test_close_scores(self):
        """Multiple heads with close TS scores."""
        src_attn = np.array([
            [0.1, 0.8, 0.1],  # -> pos 1
            [0.1, 0.1, 0.8],  # -> pos 2
            [0.1, 0.7, 0.2],  # -> pos 1
        ])
        ts_scores = [0.4, 0.3, 0.35]
        # pos 1: 0.4 + 0.35 = 0.75, pos 2: 0.3
        assert aggregate_ts_weighted_vote(src_attn, ts_scores) == 1


class TestIsTargetLanguage:
    def test_chinese_text(self):
        assert is_target_language("这是一个测试", "zh") is True

    def test_english_text_zh_target(self):
        assert is_target_language("This is a test", "zh") is False

    def test_mixed_text_zh(self):
        # More Chinese than English
        assert is_target_language("这是一个test测试", "zh") is True

    def test_german_text(self):
        # Latin-script: always returns True (can't distinguish)
        assert is_target_language("Das ist ein Test", "de") is True

    def test_english_text_de_target(self):
        assert is_target_language("This is a test", "de") is True

    def test_japanese_text(self):
        assert is_target_language("これはテストです", "ja") is True

    def test_russian_text(self):
        assert is_target_language("Это тест", "ru") is True

    def test_arabic_text(self):
        assert is_target_language("هذا اختبار", "ar") is True


class TestAlignAttConfig:
    def test_defaults(self):
        config = AlignAttConfig()
        assert config.border_distance == 3
        assert config.top_k_heads == 10
        assert config.word_batch == 1
        assert config.context_sentences == 0
        assert config.segment_reset is True

    def test_custom(self):
        config = AlignAttConfig(border_distance=4, word_batch=3, target_lang="de")
        assert config.border_distance == 4
        assert config.word_batch == 3
        assert config.target_lang == "de"


class TestAlignmentHead:
    def test_creation(self):
        head = AlignmentHead(layer=7, head=21, ts=0.737)
        assert head.layer == 7
        assert head.head == 21
        assert head.ts == pytest.approx(0.737)
