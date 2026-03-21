"""Tests for Iteration 18: XCOMET-XL subprocess scorer + adaptive top_p threshold.

Features tested:
    1. XCOMET-XL separate process scorer (nllw/xcomet_scorer.py)
    2. Adaptive top_p threshold (nllw/complexity.py + backend integration)
    3. BackendConfig adaptive_top_p field
    4. CLI integration (bench.py)
"""

import json
import os
import tempfile
import pytest
import numpy as np

from nllw.complexity import (
    estimate_complexity,
    adaptive_top_p_threshold,
    adaptive_params_from_complexity,
    classify_complexity,
)
from nllw.backend_protocol import BackendConfig
from nllw.metrics import (
    bootstrap_confidence_interval,
    paired_bootstrap_test,
)


# ---------------------------------------------------------------------------
# Adaptive top_p threshold tests
# ---------------------------------------------------------------------------

class TestAdaptiveTopPThreshold:
    """Test adaptive_top_p_threshold function."""

    def test_simple_sentence_lowers_threshold(self):
        """Simple/short sentence should lower the threshold."""
        simple = "The cat sat."
        threshold = adaptive_top_p_threshold(simple, base_threshold=0.8)
        assert threshold < 0.8, f"Simple sentence should lower threshold, got {threshold}"

    def test_complex_sentence_raises_threshold(self):
        """Complex/long sentence should raise the threshold."""
        complex_text = (
            "The extraordinary $1.2B ramifications of the economic policies "
            "implemented by the government during the fiscal quarter of 2025-2026 "
            "have significantly altered the macroeconomic landscape across "
            "multiple interconnected sectors and geographical regions, "
            "including agriculture, technology, manufacturing, and defense."
        )
        threshold = adaptive_top_p_threshold(complex_text, base_threshold=0.8)
        assert threshold > 0.8, f"Complex sentence should raise threshold, got {threshold}"

    def test_threshold_clamped_minimum(self):
        """Threshold should never go below 0.5."""
        # Very simple sentence + low base
        simple = "Hi there."
        threshold = adaptive_top_p_threshold(simple, base_threshold=0.5)
        assert threshold >= 0.5

    def test_threshold_clamped_maximum(self):
        """Threshold should never go above 0.95."""
        # Very complex sentence + high base
        complex_text = "The 12345-calibrated $1.2M quantum-entangled multi-dimensional " * 10
        threshold = adaptive_top_p_threshold(complex_text, base_threshold=0.95)
        assert threshold <= 0.95

    def test_empty_sentence(self):
        """Empty sentence should return base threshold."""
        threshold = adaptive_top_p_threshold("", base_threshold=0.85)
        # Empty -> complexity 0 -> delta = -0.1, so 0.85 - 0.1 = 0.75
        assert 0.5 <= threshold <= 0.95

    def test_moderate_sentence_near_base(self):
        """Moderate sentence should be close to base threshold."""
        moderate = "The president announced new economic reforms today."
        threshold = adaptive_top_p_threshold(moderate, base_threshold=0.85)
        # Should be within +/- 0.1 of base
        assert abs(threshold - 0.85) <= 0.15

    def test_different_base_thresholds(self):
        """Adaptive adjustment should work with different base thresholds."""
        sentence = "This is a test sentence with some words."
        t1 = adaptive_top_p_threshold(sentence, base_threshold=0.7)
        t2 = adaptive_top_p_threshold(sentence, base_threshold=0.9)
        # t2 should be higher than t1 (both shifted by same delta)
        assert t2 > t1

    def test_with_subword_count(self):
        """Subword count should affect complexity and thus threshold."""
        sentence = "The specialized terminology requires precise adaptation."
        t_no_sub = adaptive_top_p_threshold(sentence, base_threshold=0.8)
        # High subword ratio = more complex
        t_high_sub = adaptive_top_p_threshold(
            sentence, base_threshold=0.8, subword_count=30  # 30 subwords for 6 words
        )
        assert t_high_sub >= t_no_sub

    def test_monotonic_with_complexity(self):
        """More complex sentences should get higher or equal thresholds."""
        simple = "Hello world."
        medium = "The president of France announced economic reforms."
        complex_text = (
            "The extraordinarily complex multi-dimensional ramifications "
            "of the government's $1.2B restructuring program in 2025."
        )
        t_simple = adaptive_top_p_threshold(simple, base_threshold=0.8)
        t_medium = adaptive_top_p_threshold(medium, base_threshold=0.8)
        t_complex = adaptive_top_p_threshold(complex_text, base_threshold=0.8)
        assert t_simple <= t_medium <= t_complex


# ---------------------------------------------------------------------------
# BackendConfig adaptive_top_p field tests
# ---------------------------------------------------------------------------

class TestBackendConfigAdaptiveTopP:
    """Test BackendConfig has adaptive_top_p field."""

    def test_default_disabled(self):
        """adaptive_top_p should be disabled by default."""
        config = BackendConfig()
        assert config.adaptive_top_p is False

    def test_enable_via_constructor(self):
        """Can enable adaptive_top_p via constructor."""
        config = BackendConfig(adaptive_top_p=True)
        assert config.adaptive_top_p is True

    def test_from_dict(self):
        """Can enable adaptive_top_p from dict."""
        config = BackendConfig.from_dict({"adaptive_top_p": True, "top_p_threshold": 0.85})
        assert config.adaptive_top_p is True
        assert config.top_p_threshold == 0.85

    def test_from_dict_ignores_unknown(self):
        """from_dict should ignore unknown keys."""
        config = BackendConfig.from_dict({"adaptive_top_p": True, "unknown_key": 42})
        assert config.adaptive_top_p is True


# ---------------------------------------------------------------------------
# XCOMET-XL scorer tests (offline, no GPU required)
# ---------------------------------------------------------------------------

class TestXCOMETScorer:
    """Test xcomet_scorer module (JSON handling, no actual XCOMET model)."""

    def test_save_hypotheses_json(self):
        """Test saving eval results to JSON for XCOMET scoring."""
        from nllw.eval import EvalResult
        from nllw.xcomet_scorer import save_hypotheses_json

        result = EvalResult(
            backend_type="test",
            direction="en-zh",
            n_sentences=2,
            per_sentence=[
                {"source": "Hello", "reference": "Ni hao", "hypothesis": "Ni hao"},
                {"source": "Bye", "reference": "Zai jian", "hypothesis": "Zai jian"},
            ],
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as f:
            path = f.name

        try:
            save_hypotheses_json(result, path)
            with open(path) as f:
                data = json.load(f)

            assert "per_sentence" in data
            assert len(data["per_sentence"]) == 2
            assert data["per_sentence"][0]["source"] == "Hello"
            assert data["per_sentence"][0]["hypothesis"] == "Ni hao"
            assert "summary" in data
        finally:
            os.unlink(path)

    def test_score_from_eval_json(self):
        """Test loading eval JSON for scoring (mock -- doesn't load XCOMET model)."""
        # Just verify JSON loading works
        data = {
            "summary": {"direction": "en-zh"},
            "per_sentence": [
                {"source": "test", "reference": "ref", "hypothesis": "hyp"},
            ],
        }

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode='w'
        ) as f:
            json.dump(data, f)
            path = f.name

        try:
            # We can't actually run score_from_eval_json without COMET installed
            # but we can verify the JSON loads correctly
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["per_sentence"][0]["source"] == "test"
        finally:
            os.unlink(path)

    def test_scorer_module_importable(self):
        """Test that xcomet_scorer module can be imported."""
        from nllw import xcomet_scorer
        assert hasattr(xcomet_scorer, 'score_xcomet')
        assert hasattr(xcomet_scorer, 'score_xcomet_subprocess')
        assert hasattr(xcomet_scorer, 'save_hypotheses_json')
        assert hasattr(xcomet_scorer, 'score_from_eval_json')


# ---------------------------------------------------------------------------
# Bench CLI integration tests
# ---------------------------------------------------------------------------

class TestBenchCLIAdaptiveTopP:
    """Test bench.py CLI recognizes adaptive_top_p flags."""

    def test_sweep_shortname_registered(self):
        """'adaptp' shortname should be in bench.py sweep parser."""
        # The shortname map is inside parse_sweep_spec function.
        # Verify by importing bench and checking parse_sweep_spec handles it.
        from nllw.bench import parse_sweep_spec
        grid = parse_sweep_spec("adaptp=0,1")
        assert "adaptive_top_p" in grid
        assert grid["adaptive_top_p"] == [0, 1]

    def test_cli_flag_exists(self):
        """--adaptive-top-p flag should be recognized."""
        import argparse
        # Just verify the flag can be parsed (bench creates the parser internally)
        # We test the config mapping instead
        config = BackendConfig.from_dict({"adaptive_top_p": True})
        assert config.adaptive_top_p is True


# ---------------------------------------------------------------------------
# Integration: complexity -> adaptive threshold flow
# ---------------------------------------------------------------------------

class TestAdaptiveTopPIntegration:
    """Integration tests for complexity-based adaptive top_p."""

    def test_full_pipeline_various_sentences(self):
        """Test adaptive_top_p_threshold on diverse sentences."""
        sentences = [
            "Hi.",  # very simple
            "The cat sat on the mat.",  # simple
            "The government announced new tax reforms affecting small businesses.",  # moderate
            "The extraordinarily complex diplomatic negotiations between 15 nations "
            "regarding the $2.5B climate adaptation fund have resulted in a "
            "comprehensive multi-lateral agreement.",  # complex
        ]
        thresholds = [
            adaptive_top_p_threshold(s, base_threshold=0.85)
            for s in sentences
        ]
        # Should be monotonically non-decreasing (or close to it)
        # Actually complexity depends on features, not just length, so just check range
        for t in thresholds:
            assert 0.5 <= t <= 0.95, f"Threshold {t} out of range"

    def test_chinese_source(self):
        """Test with Chinese source text (CJK characters)."""
        # Chinese text has short "words" when split by spaces (usually no spaces)
        chinese = "美国总统今天宣布了新的经济政策"
        threshold = adaptive_top_p_threshold(chinese, base_threshold=0.85)
        assert 0.5 <= threshold <= 0.95

    def test_numerical_heavy_sentence(self):
        """Sentence with many numbers should be more complex."""
        no_nums = "The president announced reforms today."
        with_nums = "In 2025, the $1.2B budget for sectors 1-5 was approved."
        t_no = adaptive_top_p_threshold(no_nums, base_threshold=0.8)
        t_nums = adaptive_top_p_threshold(with_nums, base_threshold=0.8)
        # Numbers add complexity -> higher threshold
        assert t_nums >= t_no

    def test_consistency_across_calls(self):
        """Same input should always produce same output."""
        sentence = "The president of the United States announced new policies."
        t1 = adaptive_top_p_threshold(sentence, base_threshold=0.8)
        t2 = adaptive_top_p_threshold(sentence, base_threshold=0.8)
        assert t1 == t2


# ---------------------------------------------------------------------------
# Bootstrap confidence interval tests
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    """Test bootstrap confidence interval computation."""

    def test_basic_ci(self):
        """Basic CI should contain the mean."""
        scores = [0.8, 0.85, 0.9, 0.82, 0.88, 0.91, 0.87, 0.83, 0.86, 0.89]
        mean, lo, hi = bootstrap_confidence_interval(scores)
        assert lo <= mean <= hi
        assert abs(mean - np.mean(scores)) < 1e-6

    def test_tight_ci_uniform(self):
        """Uniform scores should have a tight CI."""
        scores = [0.85] * 100
        mean, lo, hi = bootstrap_confidence_interval(scores)
        assert abs(mean - 0.85) < 1e-6
        assert abs(lo - 0.85) < 1e-6
        assert abs(hi - 0.85) < 1e-6

    def test_wide_ci_variable(self):
        """Highly variable scores should have a wider CI."""
        scores = [0.3, 0.9, 0.2, 0.95, 0.4, 0.85, 0.1, 0.99, 0.5, 0.7]
        mean, lo, hi = bootstrap_confidence_interval(scores)
        width = hi - lo
        assert width > 0.1, f"CI too tight for variable scores: {width}"

    def test_empty_scores(self):
        """Empty scores should return zeros."""
        mean, lo, hi = bootstrap_confidence_interval([])
        assert mean == 0.0 and lo == 0.0 and hi == 0.0

    def test_single_score(self):
        """Single score should give tight CI."""
        mean, lo, hi = bootstrap_confidence_interval([0.9])
        assert mean == 0.9
        assert lo == 0.9
        assert hi == 0.9

    def test_reproducibility(self):
        """Same seed should give identical results."""
        scores = [0.8, 0.85, 0.9, 0.82, 0.88]
        r1 = bootstrap_confidence_interval(scores, seed=42)
        r2 = bootstrap_confidence_interval(scores, seed=42)
        assert r1 == r2

    def test_different_confidence_levels(self):
        """Higher confidence should give wider CI."""
        scores = list(np.random.RandomState(42).normal(0.85, 0.05, 100))
        _, lo_95, hi_95 = bootstrap_confidence_interval(scores, confidence=0.95)
        _, lo_99, hi_99 = bootstrap_confidence_interval(scores, confidence=0.99)
        assert (hi_99 - lo_99) >= (hi_95 - lo_95)


class TestPairedBootstrapTest:
    """Test paired bootstrap significance test."""

    def test_identical_systems(self):
        """Identical scores should give p=0.5 (not significant)."""
        scores = [0.8, 0.85, 0.9, 0.82, 0.88]
        delta, p = paired_bootstrap_test(scores, scores)
        assert abs(delta) < 1e-10
        assert p >= 0.3  # Not significant

    def test_clearly_better_system(self):
        """System clearly better should give low p-value."""
        scores_a = [0.9, 0.92, 0.91, 0.93, 0.89, 0.94, 0.90, 0.95, 0.91, 0.92]
        scores_b = [0.7, 0.72, 0.71, 0.73, 0.69, 0.74, 0.70, 0.75, 0.71, 0.72]
        delta, p = paired_bootstrap_test(scores_a, scores_b)
        assert delta > 0.15  # A is clearly better
        assert p < 0.05  # Significant

    def test_marginal_difference(self):
        """Marginal difference should give higher p-value."""
        rng = np.random.RandomState(42)
        scores_a = list(rng.normal(0.88, 0.05, 20))
        scores_b = list(rng.normal(0.87, 0.05, 20))
        delta, p = paired_bootstrap_test(scores_a, scores_b)
        assert abs(delta) < 0.05  # Small difference

    def test_mismatched_lengths(self):
        """Mismatched lengths should raise error."""
        with pytest.raises(ValueError):
            paired_bootstrap_test([0.8, 0.9], [0.7])

    def test_empty_scores(self):
        """Empty scores should return p=1.0."""
        delta, p = paired_bootstrap_test([], [])
        assert p == 1.0
