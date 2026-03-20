"""Tests for head detection module (unit tests only, no GPU required)."""

import pytest
from nllw.detect_heads import (
    _tokens_to_word_map,
    _reconstruct_words,
    _reconstruct_cjk_chars,
    _cjk_char_to_token_map,
    DEFAULT_TS_THRESHOLD,
)


class TestTokenToWordMap:
    def test_basic_bpe(self):
        """Standard BPE tokens with space prefix."""
        tokens = ["The", " quick", " brown", " fox"]
        mapping = _tokens_to_word_map(tokens)
        assert len(mapping) == 4
        assert mapping[0] == [0]  # "The"
        assert mapping[1] == [1]  # " quick"

    def test_subword_tokens(self):
        """Subword splits map to same word."""
        tokens = ["un", "believ", "Ġable"]
        mapping = _tokens_to_word_map(tokens)
        # "un" + "believ" = word 0, "Ġable" = word 1
        assert mapping[0] == [0, 1]
        assert mapping[1] == [2]

    def test_sentencepiece_prefix(self):
        """SentencePiece ▁ prefix."""
        tokens = ["▁Hello", "▁world"]
        mapping = _tokens_to_word_map(tokens)
        assert len(mapping) == 2


class TestReconstructWords:
    def test_basic(self):
        tokens = ["Ġhello", "Ġworld"]
        words = _reconstruct_words(tokens)
        assert words == ["hello", "world"]

    def test_subwords(self):
        tokens = ["Ġhel", "lo", "Ġworld"]
        words = _reconstruct_words(tokens)
        assert words == ["hello", "world"]

    def test_empty(self):
        assert _reconstruct_words([]) == []


class TestCJKReconstruction:
    def test_chinese_chars(self):
        tokens = ["Ġ\u4f60", "\u597d", "Ġ\u4e16\u754c"]
        chars = _reconstruct_cjk_chars(tokens)
        assert "\u4f60" in chars
        assert "\u597d" in chars
        assert "\u4e16" in chars
        assert "\u754c" in chars

    def test_char_to_token_map(self):
        tokens = ["\u4f60\u597d", "\u4e16\u754c"]
        mapping = _cjk_char_to_token_map(tokens)
        # char 0 (\u4f60) -> token 0
        # char 1 (\u597d) -> token 0
        # char 2 (\u4e16) -> token 1
        # char 3 (\u754c) -> token 1
        assert mapping[0] == [0]
        assert mapping[1] == [0]
        assert mapping[2] == [1]
        assert mapping[3] == [1]


class TestThreshold:
    def test_default_threshold(self):
        assert DEFAULT_TS_THRESHOLD == 0.1
