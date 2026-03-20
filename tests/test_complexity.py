"""Tests for source complexity estimation."""

import pytest

from nllw.complexity import (
    estimate_complexity,
    adaptive_params_from_complexity,
    classify_complexity,
    ComplexityProfile,
)


class TestEstimateComplexity:
    def test_empty_text(self):
        """Empty text has zero complexity."""
        profile = estimate_complexity("")
        assert profile.word_count == 0
        assert profile.complexity_score == 0.0

    def test_simple_short_sentence(self):
        """Short simple sentence has low complexity."""
        profile = estimate_complexity("Hello world")
        assert profile.word_count == 2
        assert profile.complexity_score < 0.3
        assert profile.suggested_bd_delta <= 0
        assert profile.suggested_wb_delta <= 0

    def test_long_complex_sentence(self):
        """Long sentence with many words has high complexity."""
        text = (
            "The unprecedented geopolitical ramifications of the multilateral "
            "trade agreements negotiated during the intergovernmental conference "
            "have significantly impacted the socioeconomic infrastructure of "
            "participating nations throughout the developing world"
        )
        profile = estimate_complexity(text)
        assert profile.word_count > 20
        assert profile.complexity_score > 0.4
        assert profile.suggested_bd_delta >= 0

    def test_numeral_rich_text(self):
        """Text with many numbers has higher complexity."""
        text = "In 2024 the GDP was $3.2 trillion with 15% growth and 42 million jobs"
        profile = estimate_complexity(text)
        assert profile.numeral_density > 0.1
        # Numbers add complexity (must be preserved)
        assert profile.complexity_score > 0.2

    def test_short_vs_long(self):
        """Longer sentences have higher complexity."""
        short = estimate_complexity("The cat sat")
        long = estimate_complexity(
            "The quick brown fox jumped over the lazy dog and then ran "
            "across the field to the barn where it found some chickens"
        )
        assert long.complexity_score > short.complexity_score

    def test_subword_ratio_increases_complexity(self):
        """Higher subword ratio means more complex."""
        text = "Hello world test"
        simple = estimate_complexity(text, subword_count=3)
        complex = estimate_complexity(text, subword_count=9)
        assert complex.complexity_score > simple.complexity_score

    def test_avg_word_length(self):
        """Long words increase complexity."""
        short_words = estimate_complexity("the cat sat on a mat")
        long_words = estimate_complexity("internationalization characterization")
        assert long_words.avg_word_length > short_words.avg_word_length

    def test_complexity_score_bounded(self):
        """Complexity score is always between 0 and 1."""
        cases = [
            "",
            "Hi",
            "The cat sat",
            "A" * 1000,
            "123 456 789 012 345 678 901 234 567 890" * 5,
        ]
        for text in cases:
            profile = estimate_complexity(text)
            assert 0.0 <= profile.complexity_score <= 1.0

    def test_punctuation_adds_complexity(self):
        """Heavy punctuation indicates complex syntax."""
        simple = estimate_complexity("The president announced reforms today")
        complex = estimate_complexity('The president (who was, of course, skeptical) announced "reforms" -- today')
        assert complex.complexity_score >= simple.complexity_score


class TestAdaptiveParams:
    def test_simple_sentence_reduces_params(self):
        """Simple sentence gets reduced parameters."""
        bd, wb, gen = adaptive_params_from_complexity(
            "Hello world", base_bd=3, base_wb=3, base_gen_cap=50
        )
        assert bd <= 3  # bd might be reduced
        assert wb <= 3  # wb might be reduced
        assert gen <= 50  # gen_cap might be reduced

    def test_complex_sentence_increases_params(self):
        """Complex sentence gets increased parameters."""
        text = (
            "The unprecedented geopolitical ramifications of the multilateral "
            "trade agreements negotiated during the intergovernmental conference "
            "have significantly impacted the socioeconomic infrastructure"
        )
        bd, wb, gen = adaptive_params_from_complexity(
            text, base_bd=3, base_wb=3, base_gen_cap=50
        )
        assert bd >= 3  # bd increased or same
        assert gen >= 50  # gen_cap increased

    def test_minimum_values(self):
        """Parameters never go below minimum."""
        bd, wb, gen = adaptive_params_from_complexity(
            "Hi", base_bd=1, base_wb=1, base_gen_cap=10
        )
        assert bd >= 1
        assert wb >= 1
        assert gen >= 10

    def test_subword_count_effect(self):
        """Subword count affects parameters."""
        text = "Hello world test sentence here"
        _, _, gen_low = adaptive_params_from_complexity(text, subword_count=5)
        _, _, gen_high = adaptive_params_from_complexity(text, subword_count=15)
        assert gen_high >= gen_low


class TestClassifyComplexity:
    def test_simple(self):
        assert classify_complexity(0.1) == "simple"

    def test_moderate(self):
        assert classify_complexity(0.35) == "moderate"

    def test_complex(self):
        assert classify_complexity(0.6) == "complex"

    def test_very_complex(self):
        assert classify_complexity(0.8) == "very_complex"

    def test_boundaries(self):
        assert classify_complexity(0.0) == "simple"
        assert classify_complexity(0.25) == "moderate"
        assert classify_complexity(0.5) == "complex"
        assert classify_complexity(0.75) == "very_complex"
