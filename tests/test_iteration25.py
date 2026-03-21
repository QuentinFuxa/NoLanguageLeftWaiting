"""Tests for iteration 25: Generation temperature + Confidence-gated token trimming."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSampleWithTemperature(unittest.TestCase):
    """Tests for sample_with_temperature()."""

    def test_zero_temperature_is_greedy(self):
        """Temperature 0.0 should return argmax."""
        from nllw.alignatt import sample_with_temperature
        logits = np.array([1.0, 5.0, 2.0, 3.0])
        result = sample_with_temperature(logits, temperature=0.0)
        self.assertEqual(result, 1)  # argmax

    def test_very_low_temperature_concentrates(self):
        """Very low temperature should almost always pick the argmax."""
        from nllw.alignatt import sample_with_temperature
        logits = np.array([1.0, 10.0, 2.0, 3.0])
        # Sample 100 times at very low temperature
        np.random.seed(42)
        results = [sample_with_temperature(logits, 0.01) for _ in range(100)]
        # Should pick index 1 (almost) every time
        self.assertEqual(results.count(1), 100)

    def test_high_temperature_distributes(self):
        """High temperature should produce diverse samples."""
        from nllw.alignatt import sample_with_temperature
        logits = np.array([1.0, 2.0, 1.5, 1.8])
        np.random.seed(42)
        results = [sample_with_temperature(logits, 2.0) for _ in range(200)]
        # Should see multiple distinct values
        unique = set(results)
        self.assertGreater(len(unique), 2)

    def test_returns_valid_index(self):
        """Result should be a valid index into the logits array."""
        from nllw.alignatt import sample_with_temperature
        logits = np.random.randn(32000)
        np.random.seed(42)
        result = sample_with_temperature(logits, 0.3)
        self.assertGreaterEqual(result, 0)
        self.assertLess(result, len(logits))

    def test_negative_logits_work(self):
        """Should handle negative logits correctly."""
        from nllw.alignatt import sample_with_temperature
        logits = np.array([-5.0, -1.0, -3.0, -2.0])
        result = sample_with_temperature(logits, 0.0)
        self.assertEqual(result, 1)  # -1.0 is the max


class TestTrimLowConfidenceTokens(unittest.TestCase):
    """Tests for trim_low_confidence_tokens()."""

    def test_no_trimming_all_confident(self):
        """All tokens above threshold -> no trimming."""
        from nllw.alignatt import trim_low_confidence_tokens
        ids = [10, 20, 30, 40]
        logprobs = [-0.5, -1.0, -0.8, -1.2]
        result = trim_low_confidence_tokens(ids, logprobs, threshold=-3.0)
        self.assertEqual(result, ids)

    def test_trim_trailing_low_confidence(self):
        """Last two tokens below threshold -> trim them."""
        from nllw.alignatt import trim_low_confidence_tokens
        ids = [10, 20, 30, 40]
        logprobs = [-0.5, -1.0, -5.0, -6.0]
        result = trim_low_confidence_tokens(ids, logprobs, threshold=-3.0)
        self.assertEqual(result, [10, 20])

    def test_trim_single_trailing(self):
        """Only last token below threshold -> trim it."""
        from nllw.alignatt import trim_low_confidence_tokens
        ids = [10, 20, 30]
        logprobs = [-1.0, -0.5, -4.0]
        result = trim_low_confidence_tokens(ids, logprobs, threshold=-3.0)
        self.assertEqual(result, [10, 20])

    def test_min_keep_prevents_empty(self):
        """All tokens below threshold -> keep min_keep."""
        from nllw.alignatt import trim_low_confidence_tokens
        ids = [10, 20, 30]
        logprobs = [-5.0, -4.0, -6.0]
        result = trim_low_confidence_tokens(ids, logprobs, threshold=-3.0, min_keep=1)
        self.assertEqual(len(result), 1)

    def test_min_keep_two(self):
        """All below threshold, min_keep=2 -> keep first 2."""
        from nllw.alignatt import trim_low_confidence_tokens
        ids = [10, 20, 30]
        logprobs = [-5.0, -4.0, -6.0]
        result = trim_low_confidence_tokens(ids, logprobs, threshold=-3.0, min_keep=2)
        self.assertEqual(len(result), 2)

    def test_empty_input(self):
        """Empty input -> empty output."""
        from nllw.alignatt import trim_low_confidence_tokens
        result = trim_low_confidence_tokens([], [], threshold=-3.0)
        self.assertEqual(result, [])

    def test_mismatched_lengths(self):
        """Mismatched lengths -> return original (no trim)."""
        from nllw.alignatt import trim_low_confidence_tokens
        ids = [10, 20, 30]
        logprobs = [-1.0, -2.0]  # Different length
        result = trim_low_confidence_tokens(ids, logprobs, threshold=-3.0)
        self.assertEqual(result, ids)

    def test_middle_dip_no_trim(self):
        """Low confidence in middle but confident at end -> no trim."""
        from nllw.alignatt import trim_low_confidence_tokens
        ids = [10, 20, 30, 40]
        logprobs = [-1.0, -5.0, -0.5, -1.0]
        result = trim_low_confidence_tokens(ids, logprobs, threshold=-3.0)
        self.assertEqual(result, ids)

    def test_exact_threshold_no_trim(self):
        """Token at exact threshold -> keep it."""
        from nllw.alignatt import trim_low_confidence_tokens
        ids = [10, 20]
        logprobs = [-1.0, -3.0]
        result = trim_low_confidence_tokens(ids, logprobs, threshold=-3.0)
        self.assertEqual(result, ids)

    def test_single_token_confident(self):
        """Single confident token -> keep it."""
        from nllw.alignatt import trim_low_confidence_tokens
        ids = [42]
        logprobs = [-1.0]
        result = trim_low_confidence_tokens(ids, logprobs, threshold=-3.0)
        self.assertEqual(result, [42])

    def test_single_token_low_confidence(self):
        """Single low-confidence token -> keep it (min_keep=1)."""
        from nllw.alignatt import trim_low_confidence_tokens
        ids = [42]
        logprobs = [-5.0]
        result = trim_low_confidence_tokens(ids, logprobs, threshold=-3.0)
        self.assertEqual(result, [42])


class TestBackendConfigNewFields(unittest.TestCase):
    """Tests for new BackendConfig fields."""

    def test_generation_temperature_default(self):
        """Default generation_temperature should be 0.0 (greedy)."""
        from nllw.backend_protocol import BackendConfig
        config = BackendConfig()
        self.assertEqual(config.generation_temperature, 0.0)

    def test_generation_temperature_from_dict(self):
        """generation_temperature should be settable from dict."""
        from nllw.backend_protocol import BackendConfig
        config = BackendConfig.from_dict({"generation_temperature": 0.2})
        self.assertAlmostEqual(config.generation_temperature, 0.2)

    def test_confidence_trim_default(self):
        """Default confidence_trim_threshold should be None."""
        from nllw.backend_protocol import BackendConfig
        config = BackendConfig()
        self.assertIsNone(config.confidence_trim_threshold)

    def test_confidence_trim_from_dict(self):
        """confidence_trim_threshold should be settable from dict."""
        from nllw.backend_protocol import BackendConfig
        config = BackendConfig.from_dict({"confidence_trim_threshold": -3.0})
        self.assertAlmostEqual(config.confidence_trim_threshold, -3.0)


class TestBenchSweepShortnames(unittest.TestCase):
    """Tests for new sweep shortnames."""

    def test_temp_shortname(self):
        """'temp' shortname should map to generation_temperature."""
        from nllw.bench import parse_sweep_spec
        grid = parse_sweep_spec("temp=0.0,0.1,0.2")
        self.assertIn("generation_temperature", grid)
        self.assertEqual(len(grid["generation_temperature"]), 3)
        self.assertAlmostEqual(grid["generation_temperature"][0], 0.0)
        self.assertAlmostEqual(grid["generation_temperature"][1], 0.1)
        self.assertAlmostEqual(grid["generation_temperature"][2], 0.2)

    def test_conftrim_shortname(self):
        """'conftrim' shortname should map to confidence_trim_threshold."""
        from nllw.bench import parse_sweep_spec
        grid = parse_sweep_spec("conftrim=-2.0,-3.0,-4.0")
        self.assertIn("confidence_trim_threshold", grid)
        self.assertEqual(len(grid["confidence_trim_threshold"]), 3)
        self.assertAlmostEqual(grid["confidence_trim_threshold"][0], -2.0)
        self.assertAlmostEqual(grid["confidence_trim_threshold"][1], -3.0)
        self.assertAlmostEqual(grid["confidence_trim_threshold"][2], -4.0)


class TestTemperatureEdgeCases(unittest.TestCase):
    """Edge cases for temperature sampling."""

    def test_large_logits_stability(self):
        """Very large logits should not cause overflow."""
        from nllw.alignatt import sample_with_temperature
        logits = np.array([100.0, 200.0, 150.0])
        result = sample_with_temperature(logits, 0.1)
        self.assertEqual(result, 1)  # 200 is clearly the max

    def test_identical_logits(self):
        """All identical logits should produce uniform-ish sampling."""
        from nllw.alignatt import sample_with_temperature
        logits = np.ones(10) * 5.0
        np.random.seed(42)
        results = [sample_with_temperature(logits, 1.0) for _ in range(500)]
        unique = set(results)
        # Should see most values at least once
        self.assertGreater(len(unique), 5)

    def test_deterministic_with_seed(self):
        """Same seed should produce same results."""
        from nllw.alignatt import sample_with_temperature
        logits = np.array([1.0, 2.0, 3.0, 4.0])
        np.random.seed(123)
        r1 = sample_with_temperature(logits, 0.5)
        np.random.seed(123)
        r2 = sample_with_temperature(logits, 0.5)
        self.assertEqual(r1, r2)


class TestConfidenceTrimEdgeCases(unittest.TestCase):
    """Edge cases for confidence trimming."""

    def test_all_tokens_exactly_at_threshold(self):
        """All tokens at exact threshold -> keep all."""
        from nllw.alignatt import trim_low_confidence_tokens
        ids = [1, 2, 3]
        logprobs = [-3.0, -3.0, -3.0]
        result = trim_low_confidence_tokens(ids, logprobs, threshold=-3.0)
        self.assertEqual(result, ids)

    def test_very_negative_logprobs(self):
        """Very negative logprobs should trigger trimming."""
        from nllw.alignatt import trim_low_confidence_tokens
        ids = [1, 2, 3, 4, 5]
        logprobs = [-1.0, -2.0, -15.0, -20.0, -25.0]
        result = trim_low_confidence_tokens(ids, logprobs, threshold=-3.0)
        self.assertEqual(result, [1, 2])

    def test_gradual_degradation(self):
        """Gradual confidence degradation should trim at the right point."""
        from nllw.alignatt import trim_low_confidence_tokens
        ids = [1, 2, 3, 4, 5]
        logprobs = [-0.5, -1.5, -2.5, -3.5, -4.5]
        result = trim_low_confidence_tokens(ids, logprobs, threshold=-3.0)
        self.assertEqual(result, [1, 2, 3])  # -2.5 >= -3.0, -3.5 < -3.0

    def test_large_min_keep(self):
        """min_keep larger than trimmed result should override."""
        from nllw.alignatt import trim_low_confidence_tokens
        ids = [1, 2, 3, 4, 5]
        logprobs = [-1.0, -5.0, -5.0, -5.0, -5.0]
        result = trim_low_confidence_tokens(ids, logprobs, threshold=-3.0, min_keep=3)
        self.assertEqual(len(result), 3)


class TestEntropyBasedDynamicTemperature(unittest.TestCase):
    """Tests for entropy_based_dynamic_temperature() - EDT."""

    def test_confident_low_entropy_low_temp(self):
        """Confident distribution (low entropy) -> low temperature."""
        from nllw.alignatt import entropy_based_dynamic_temperature
        # Very peaked distribution -> low entropy -> low temp
        logits = np.zeros(1000)
        logits[42] = 20.0  # One token dominates
        temp = entropy_based_dynamic_temperature(logits, base_temperature=0.1)
        self.assertLess(temp, 0.2)  # Should be low

    def test_uncertain_high_entropy_higher_temp(self):
        """Uncertain distribution (high entropy) -> higher temperature."""
        from nllw.alignatt import entropy_based_dynamic_temperature
        # Flat distribution -> high entropy -> higher temp
        logits = np.ones(1000) + np.random.randn(1000) * 0.01
        temp = entropy_based_dynamic_temperature(logits, base_temperature=0.1)
        # Should be higher than the confident case
        self.assertGreater(temp, 0.01)

    def test_temperature_within_bounds(self):
        """Temperature should always be within [min, max]."""
        from nllw.alignatt import entropy_based_dynamic_temperature
        # Test various distributions
        for _ in range(20):
            logits = np.random.randn(32000)
            temp = entropy_based_dynamic_temperature(
                logits, base_temperature=0.1,
                min_temperature=0.01, max_temperature=0.5
            )
            self.assertGreaterEqual(temp, 0.01)
            self.assertLessEqual(temp, 0.5)

    def test_zero_entropy_returns_min(self):
        """Zero entropy (degenerate) -> minimum temperature."""
        from nllw.alignatt import entropy_based_dynamic_temperature
        # All probability on one token
        logits = np.full(100, -100.0)
        logits[0] = 100.0
        temp = entropy_based_dynamic_temperature(logits, base_temperature=0.1)
        # Should be at or near minimum
        self.assertLessEqual(temp, 0.1)

    def test_different_base_temperatures(self):
        """Base temperature should scale the output proportionally."""
        from nllw.alignatt import entropy_based_dynamic_temperature
        logits = np.random.randn(1000)
        temp1 = entropy_based_dynamic_temperature(logits, base_temperature=0.1)
        temp2 = entropy_based_dynamic_temperature(logits, base_temperature=0.2)
        # Higher base -> higher output (not necessarily 2x due to clamping)
        self.assertGreaterEqual(temp2, temp1)

    def test_medium_entropy_intermediate_temp(self):
        """Medium entropy -> intermediate temperature."""
        from nllw.alignatt import entropy_based_dynamic_temperature
        # Distribution with moderate uncertainty: ~10 equally likely tokens
        # This gives entropy ~log(10) = 2.3 nats which is in the mid range
        logits = np.full(1000, -20.0)
        logits[:10] = 0.0  # 10 equal tokens
        temp = entropy_based_dynamic_temperature(
            logits, base_temperature=0.1,
            min_temperature=0.01, max_temperature=0.5
        )
        # Should be between min and max (entropy around 2.3 nats)
        self.assertGreater(temp, 0.01)
        self.assertLessEqual(temp, 0.5)


class TestEDTBackendConfig(unittest.TestCase):
    """Tests for EDT config field."""

    def test_edt_default_disabled(self):
        """EDT should be disabled by default."""
        from nllw.backend_protocol import BackendConfig
        cfg = BackendConfig()
        self.assertFalse(cfg.entropy_dynamic_temperature)

    def test_edt_from_dict(self):
        """EDT should be settable from dict."""
        from nllw.backend_protocol import BackendConfig
        cfg = BackendConfig.from_dict({"entropy_dynamic_temperature": True})
        self.assertTrue(cfg.entropy_dynamic_temperature)


class TestEDTSweepShortname(unittest.TestCase):
    """Tests for EDT sweep shortname."""

    def test_edt_shortname(self):
        """'edt' shortname should map to entropy_dynamic_temperature."""
        from nllw.bench import parse_sweep_spec
        grid = parse_sweep_spec("edt=0,1")
        self.assertIn("entropy_dynamic_temperature", grid)
        self.assertEqual(grid["entropy_dynamic_temperature"], [0, 1])


if __name__ == "__main__":
    unittest.main()
