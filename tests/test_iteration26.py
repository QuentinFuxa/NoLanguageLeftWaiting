"""Tests for iteration 26: Anti-LM contrastive decoding + beam search research."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestApplyAntiLmPenalty(unittest.TestCase):
    """Tests for apply_anti_lm_penalty()."""

    def test_step_zero_full_penalty(self):
        """At step 0, penalty is gamma^0 = 1.0 (full strength)."""
        from nllw.alignatt import apply_anti_lm_penalty
        logits = np.array([5.0, 3.0, 1.0, 0.0])
        anti_lm = np.array([4.0, 1.0, 0.5, 0.0])
        result = apply_anti_lm_penalty(logits, anti_lm, step=0, gamma=0.3)
        # At step 0: gamma^0 = 1.0, so result = logits - 1.0 * anti_lm
        np.testing.assert_array_almost_equal(
            result, np.array([1.0, 2.0, 0.5, 0.0])
        )

    def test_step_one_decayed_penalty(self):
        """At step 1, penalty is gamma^1 = gamma."""
        from nllw.alignatt import apply_anti_lm_penalty
        logits = np.array([5.0, 3.0, 1.0, 0.0])
        anti_lm = np.array([4.0, 1.0, 0.5, 0.0])
        gamma = 0.3
        result = apply_anti_lm_penalty(logits, anti_lm, step=1, gamma=gamma)
        expected = logits - gamma * anti_lm
        np.testing.assert_array_almost_equal(result, expected)

    def test_high_step_negligible_penalty(self):
        """At high steps, penalty decays to near zero."""
        from nllw.alignatt import apply_anti_lm_penalty
        logits = np.array([5.0, 3.0, 1.0, 0.0])
        anti_lm = np.array([10.0, 10.0, 10.0, 10.0])
        result = apply_anti_lm_penalty(logits, anti_lm, step=20, gamma=0.3)
        # gamma^20 = 0.3^20 ~= 3.5e-11, essentially zero
        np.testing.assert_array_almost_equal(result, logits, decimal=5)

    def test_gamma_zero_no_penalty(self):
        """Gamma <= 0 should return original logits (no penalty)."""
        from nllw.alignatt import apply_anti_lm_penalty
        logits = np.array([5.0, 3.0, 1.0])
        anti_lm = np.array([10.0, 10.0, 10.0])
        result = apply_anti_lm_penalty(logits, anti_lm, step=0, gamma=0.0)
        np.testing.assert_array_almost_equal(result, logits)

    def test_gamma_one_constant_penalty(self):
        """Gamma = 1.0 means no decay (constant penalty at all steps)."""
        from nllw.alignatt import apply_anti_lm_penalty
        logits = np.array([5.0, 3.0])
        anti_lm = np.array([2.0, 1.0])
        r0 = apply_anti_lm_penalty(logits, anti_lm, step=0, gamma=1.0)
        r5 = apply_anti_lm_penalty(logits, anti_lm, step=5, gamma=1.0)
        np.testing.assert_array_almost_equal(r0, r5)

    def test_changes_argmax(self):
        """Anti-LM can change which token has highest logit."""
        from nllw.alignatt import apply_anti_lm_penalty
        # Without penalty, token 0 wins (logit 5.0)
        logits = np.array([5.0, 4.5, 1.0])
        # But token 0 is strongly predicted by anti-LM (source continuation)
        anti_lm = np.array([4.0, 0.1, 0.1])
        result = apply_anti_lm_penalty(logits, anti_lm, step=0, gamma=1.0)
        # Token 0: 5.0 - 4.0 = 1.0; Token 1: 4.5 - 0.1 = 4.4
        self.assertEqual(np.argmax(result), 1)

    def test_preserves_shape(self):
        """Output should have same shape as input logits."""
        from nllw.alignatt import apply_anti_lm_penalty
        logits = np.random.randn(32000)
        anti_lm = np.random.randn(32000)
        result = apply_anti_lm_penalty(logits, anti_lm, step=0, gamma=0.3)
        self.assertEqual(result.shape, logits.shape)

    def test_negative_step_no_penalty(self):
        """Negative step should return original logits."""
        from nllw.alignatt import apply_anti_lm_penalty
        logits = np.array([5.0, 3.0])
        anti_lm = np.array([10.0, 10.0])
        result = apply_anti_lm_penalty(logits, anti_lm, step=-1, gamma=0.3)
        np.testing.assert_array_almost_equal(result, logits)


class TestComputeAntiLmLogProbs(unittest.TestCase):
    """Tests for compute_anti_lm_log_probs()."""

    def test_sums_to_zero(self):
        """Log-probabilities should satisfy log-sum-exp = 0 (probabilities sum to 1)."""
        from nllw.alignatt import compute_anti_lm_log_probs
        logits = np.array([2.0, 1.0, 0.5, -1.0])
        log_probs = compute_anti_lm_log_probs(logits)
        # exp(log_probs) should sum to 1
        probs = np.exp(log_probs)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=6)

    def test_preserves_ranking(self):
        """Log-probabilities should preserve the ordering of logits."""
        from nllw.alignatt import compute_anti_lm_log_probs
        logits = np.array([5.0, 3.0, 1.0, 0.5])
        log_probs = compute_anti_lm_log_probs(logits)
        # Ranking should be preserved
        self.assertEqual(np.argmax(log_probs), np.argmax(logits))
        sorted_indices = np.argsort(logits)[::-1]
        sorted_lp = np.argsort(log_probs)[::-1]
        np.testing.assert_array_equal(sorted_indices, sorted_lp)

    def test_all_negative(self):
        """All log-probabilities should be <= 0 (probabilities <= 1)."""
        from nllw.alignatt import compute_anti_lm_log_probs
        logits = np.random.randn(1000)
        log_probs = compute_anti_lm_log_probs(logits)
        self.assertTrue(np.all(log_probs <= 0.0 + 1e-10))

    def test_numerical_stability(self):
        """Should handle large logit values without overflow."""
        from nllw.alignatt import compute_anti_lm_log_probs
        logits = np.array([1000.0, 999.0, 998.0, 0.0])
        log_probs = compute_anti_lm_log_probs(logits)
        self.assertTrue(np.all(np.isfinite(log_probs)))
        probs = np.exp(log_probs)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)

    def test_uniform_logits(self):
        """Uniform logits should give uniform log-probabilities."""
        from nllw.alignatt import compute_anti_lm_log_probs
        logits = np.ones(4) * 3.0
        log_probs = compute_anti_lm_log_probs(logits)
        expected = np.log(0.25)  # = log(1/4)
        np.testing.assert_array_almost_equal(log_probs, np.full(4, expected))

    def test_large_vocab(self):
        """Should work with realistic vocabulary size."""
        from nllw.alignatt import compute_anti_lm_log_probs
        logits = np.random.randn(151936)  # HY-MT vocab size
        log_probs = compute_anti_lm_log_probs(logits)
        self.assertEqual(log_probs.shape, logits.shape)
        self.assertTrue(np.all(np.isfinite(log_probs)))


class TestAntiLmConfig(unittest.TestCase):
    """Tests for Anti-LM configuration in BackendConfig."""

    def test_default_disabled(self):
        """Anti-LM should be disabled by default."""
        from nllw.backend_protocol import BackendConfig
        cfg = BackendConfig()
        self.assertFalse(cfg.anti_lm)
        self.assertEqual(cfg.anti_lm_gamma, 0.3)

    def test_from_dict(self):
        """Should be creatable from dict."""
        from nllw.backend_protocol import BackendConfig
        cfg = BackendConfig.from_dict({
            "anti_lm": True,
            "anti_lm_gamma": 0.5,
        })
        self.assertTrue(cfg.anti_lm)
        self.assertEqual(cfg.anti_lm_gamma, 0.5)

    def test_from_dict_ignores_unknown(self):
        """Should ignore unknown keys."""
        from nllw.backend_protocol import BackendConfig
        cfg = BackendConfig.from_dict({
            "anti_lm": True,
            "unknown_field": 42,
        })
        self.assertTrue(cfg.anti_lm)


class TestAntiLmIntegration(unittest.TestCase):
    """Integration tests for Anti-LM in the generation pipeline."""

    def test_anti_lm_penalty_reduces_source_tokens(self):
        """Anti-LM penalty should reduce logits of source-language tokens."""
        from nllw.alignatt import apply_anti_lm_penalty, compute_anti_lm_log_probs
        # Simulate: token 0 is a source-language token (high anti-LM prob)
        # token 1 is a target-language token (low anti-LM prob)
        anti_lm_logits = np.zeros(10)
        anti_lm_logits[0] = 10.0  # Source token dominates
        anti_lm_logits[1] = -5.0  # Target token unlikely in source
        anti_lm_log_probs = compute_anti_lm_log_probs(anti_lm_logits)

        # Translation logits: both tokens have similar scores
        trans_logits = np.zeros(10)
        trans_logits[0] = 3.0  # Slightly prefers source token
        trans_logits[1] = 2.8  # Target token close behind

        adjusted = apply_anti_lm_penalty(
            trans_logits, anti_lm_log_probs, step=0, gamma=1.0
        )
        # Source token should be penalized more than target token
        # Token 0: 3.0 - anti_lm_log_probs[0] (near 0, highest log-prob)
        # Token 1: 2.8 - anti_lm_log_probs[1] (very negative, low log-prob)
        # So token 1 should now be preferred
        self.assertGreater(adjusted[1], adjusted[0])

    def test_anti_lm_with_edt_composable(self):
        """Anti-LM should compose with EDT (applied before temperature)."""
        from nllw.alignatt import (
            apply_anti_lm_penalty,
            compute_anti_lm_log_probs,
            entropy_based_dynamic_temperature,
            sample_with_temperature,
        )
        logits = np.random.randn(100)
        anti_lm = compute_anti_lm_log_probs(np.random.randn(100))
        # Step 1: apply anti-LM
        adjusted = apply_anti_lm_penalty(logits, anti_lm, step=0, gamma=0.3)
        # Step 2: compute EDT temperature from adjusted logits
        temp = entropy_based_dynamic_temperature(adjusted, base_temperature=0.1)
        self.assertGreater(temp, 0)
        # Step 3: sample with temperature
        tok = sample_with_temperature(adjusted, temp)
        self.assertGreaterEqual(tok, 0)
        self.assertLess(tok, 100)

    def test_anti_lm_decay_sequence(self):
        """Verify decay follows gamma^step pattern."""
        from nllw.alignatt import apply_anti_lm_penalty
        logits = np.array([10.0, 0.0])
        anti_lm = np.array([5.0, 0.0])
        gamma = 0.3
        penalties = []
        for step in range(5):
            result = apply_anti_lm_penalty(logits, anti_lm, step, gamma)
            penalty = logits[0] - result[0]  # Penalty applied to token 0
            penalties.append(penalty)
        # Verify exponential decay
        for i in range(len(penalties)):
            expected = 5.0 * (gamma ** i)
            self.assertAlmostEqual(penalties[i], expected, places=6)


class TestAntiLmSweepShortnames(unittest.TestCase):
    """Tests for Anti-LM sweep shortnames in bench.py."""

    def test_sweep_parser_knows_antilm(self):
        """Sweep parser should recognize 'antilm' shortname."""
        from nllw.bench import parse_sweep_spec
        grid = parse_sweep_spec("antilm=0,1")
        self.assertIn("anti_lm", grid)
        self.assertEqual(grid["anti_lm"], [0, 1])

    def test_sweep_parser_knows_gamma(self):
        """Sweep parser should recognize 'almgamma' shortname."""
        from nllw.bench import parse_sweep_spec
        grid = parse_sweep_spec("almgamma=0.1,0.3,0.5")
        self.assertIn("anti_lm_gamma", grid)
        self.assertEqual(len(grid["anti_lm_gamma"]), 3)
        self.assertAlmostEqual(grid["anti_lm_gamma"][0], 0.1)
        self.assertAlmostEqual(grid["anti_lm_gamma"][1], 0.3)
        self.assertAlmostEqual(grid["anti_lm_gamma"][2], 0.5)


if __name__ == "__main__":
    unittest.main()
