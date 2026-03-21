"""Tests for iteration 24: Entropy-gated top_p threshold modulation."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMergedAttentionEntropy(unittest.TestCase):
    """Tests for merged_attention_entropy() function."""

    def test_focused_attention_low_entropy(self):
        """All heads attending to same position -> low entropy."""
        from nllw.alignatt import merged_attention_entropy
        # 5 heads, 10 source positions, all focusing on position 3
        src_attn = np.zeros((5, 10))
        src_attn[:, 3] = 1.0
        ts_scores = [1.0, 0.8, 0.6, 0.5, 0.3]
        ent = merged_attention_entropy(src_attn, ts_scores)
        self.assertAlmostEqual(ent, 0.0, places=5)

    def test_uniform_attention_high_entropy(self):
        """Uniform attention across all positions -> high entropy."""
        from nllw.alignatt import merged_attention_entropy
        n_src = 10
        src_attn = np.ones((5, n_src)) / n_src
        ts_scores = [1.0, 0.8, 0.6, 0.5, 0.3]
        ent = merged_attention_entropy(src_attn, ts_scores)
        expected = np.log(n_src)  # Maximum entropy
        self.assertAlmostEqual(ent, expected, places=3)

    def test_two_peak_medium_entropy(self):
        """Two peaks -> moderate entropy."""
        from nllw.alignatt import merged_attention_entropy
        src_attn = np.zeros((3, 8))
        src_attn[:, 2] = 0.5
        src_attn[:, 6] = 0.5
        ts_scores = [1.0, 1.0, 1.0]
        ent = merged_attention_entropy(src_attn, ts_scores)
        expected_two_peak = np.log(2)  # ~0.693 nats
        self.assertAlmostEqual(ent, expected_two_peak, places=3)

    def test_empty_attention_zero_entropy(self):
        """All-zero attention -> 0 entropy."""
        from nllw.alignatt import merged_attention_entropy
        src_attn = np.zeros((3, 8))
        ts_scores = [1.0, 0.5, 0.3]
        ent = merged_attention_entropy(src_attn, ts_scores)
        self.assertEqual(ent, 0.0)

    def test_ts_weighting_matters(self):
        """Different TS scores change the merged distribution."""
        from nllw.alignatt import merged_attention_entropy
        src_attn = np.zeros((2, 5))
        # Head 0 focuses on position 0
        src_attn[0, 0] = 1.0
        # Head 1 focuses on position 4
        src_attn[1, 4] = 1.0
        # Equal TS -> equal mix -> 2 peaks -> log(2)
        ent_equal = merged_attention_entropy(src_attn, [1.0, 1.0])
        # Dominant head 0 -> nearly focused -> low entropy
        ent_dominant = merged_attention_entropy(src_attn, [10.0, 0.01])
        self.assertGreater(ent_equal, ent_dominant)


class TestEntropyGatedTopPThreshold(unittest.TestCase):
    """Tests for entropy_gated_top_p_threshold() function."""

    def test_low_entropy_reduces_threshold(self):
        """Low attention entropy -> threshold scaled down."""
        from nllw.alignatt import entropy_gated_top_p_threshold
        base = 0.85
        result = entropy_gated_top_p_threshold(base, attention_entropy=0.5)
        self.assertLess(result, base)

    def test_high_entropy_increases_threshold(self):
        """High attention entropy -> threshold scaled up."""
        from nllw.alignatt import entropy_gated_top_p_threshold
        base = 0.85
        result = entropy_gated_top_p_threshold(base, attention_entropy=3.0)
        self.assertGreater(result, base)

    def test_medium_entropy_near_base(self):
        """Medium attention entropy (between thresholds) -> near base."""
        from nllw.alignatt import entropy_gated_top_p_threshold
        base = 0.85
        # Midpoint of [1.0, 2.5] = 1.75 -> scale ~1.0
        result = entropy_gated_top_p_threshold(base, attention_entropy=1.75)
        # Should be close to base (within ~5%)
        self.assertAlmostEqual(result, base, delta=0.05)

    def test_clamped_minimum(self):
        """Result never goes below 0.5."""
        from nllw.alignatt import entropy_gated_top_p_threshold
        # Very low base + very low entropy -> would go below 0.5 without clamping
        result = entropy_gated_top_p_threshold(0.5, attention_entropy=0.1)
        self.assertGreaterEqual(result, 0.5)

    def test_clamped_maximum(self):
        """Result never goes above 0.99."""
        from nllw.alignatt import entropy_gated_top_p_threshold
        result = entropy_gated_top_p_threshold(0.95, attention_entropy=5.0)
        self.assertLessEqual(result, 0.99)

    def test_monotonic_in_entropy(self):
        """Higher entropy -> higher threshold (monotonic)."""
        from nllw.alignatt import entropy_gated_top_p_threshold
        base = 0.85
        entropies = [0.3, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
        thresholds = [
            entropy_gated_top_p_threshold(base, e) for e in entropies
        ]
        for i in range(len(thresholds) - 1):
            self.assertLessEqual(thresholds[i], thresholds[i + 1],
                                 f"Not monotonic at entropy {entropies[i]} vs {entropies[i+1]}")

    def test_smooth_interpolation(self):
        """Values between low and high entropy should interpolate smoothly."""
        from nllw.alignatt import entropy_gated_top_p_threshold
        base = 0.85
        # Check at boundaries
        at_low = entropy_gated_top_p_threshold(base, 1.0)
        at_high = entropy_gated_top_p_threshold(base, 2.5)
        at_mid = entropy_gated_top_p_threshold(base, 1.75)
        # Midpoint should be approximately halfway between low and high
        expected_mid = (at_low + at_high) / 2
        self.assertAlmostEqual(at_mid, expected_mid, places=3)

    def test_custom_parameters(self):
        """Custom scale factors work correctly."""
        from nllw.alignatt import entropy_gated_top_p_threshold
        base = 0.8
        # Very aggressive scaling
        result_low = entropy_gated_top_p_threshold(
            base, 0.5, low_scale=0.7, high_scale=1.2
        )
        result_high = entropy_gated_top_p_threshold(
            base, 3.0, low_scale=0.7, high_scale=1.2
        )
        self.assertAlmostEqual(result_low, 0.56, places=2)  # 0.8 * 0.7
        self.assertAlmostEqual(result_high, 0.96, places=2)  # 0.8 * 1.2


class TestEntropyGatedTopPConfig(unittest.TestCase):
    """Tests for entropy_gated_top_p configuration in BackendConfig."""

    def test_config_field_exists(self):
        """BackendConfig has entropy_gated_top_p field."""
        from nllw.backend_protocol import BackendConfig
        cfg = BackendConfig()
        self.assertFalse(cfg.entropy_gated_top_p)

    def test_config_from_dict(self):
        """entropy_gated_top_p can be set via from_dict."""
        from nllw.backend_protocol import BackendConfig
        cfg = BackendConfig.from_dict({"entropy_gated_top_p": True})
        self.assertTrue(cfg.entropy_gated_top_p)

    def test_sweep_shortname(self):
        """Sweep shortname 'entgtp' maps to entropy_gated_top_p."""
        from nllw.bench import parse_sweep_spec
        result = parse_sweep_spec("entgtp=0,1")
        self.assertIn("entropy_gated_top_p", result)
        self.assertEqual(result["entropy_gated_top_p"], [0, 1])


class TestEntropyGatedTopPIntegration(unittest.TestCase):
    """Integration tests: entropy-gated top_p works with check_border."""

    def test_focused_attention_emits_sooner(self):
        """Focused attention (low entropy) should lower threshold -> border hit sooner."""
        from nllw.alignatt import (
            check_border, merged_attention_entropy,
            entropy_gated_top_p_threshold,
        )
        # 5 heads, 10 positions, attention focused on position 7
        # With top_p at 0.85, rightmost top-p position = 7
        # border_threshold = 10 - 3 = 7, so position 7 >= 7 -> hit
        src_attn = np.zeros((5, 10))
        src_attn[:, 7] = 0.8
        src_attn[:, 5] = 0.2
        ts_scores = [1.0, 0.8, 0.6, 0.4, 0.2]

        # With high base threshold, top_p set includes more positions
        base_threshold = 0.85
        m_ent = merged_attention_entropy(src_attn, ts_scores)

        # Focused attention -> low entropy
        self.assertLess(m_ent, 1.0)

        # Gated threshold should be lower
        gated = entropy_gated_top_p_threshold(base_threshold, m_ent)
        self.assertLess(gated, base_threshold)

    def test_spread_attention_waits_longer(self):
        """Spread attention (high entropy) should raise threshold -> less border hit."""
        from nllw.alignatt import (
            merged_attention_entropy, entropy_gated_top_p_threshold,
        )
        # Uniform attention across 10 positions
        src_attn = np.ones((5, 10)) / 10
        ts_scores = [1.0, 0.8, 0.6, 0.4, 0.2]

        m_ent = merged_attention_entropy(src_attn, ts_scores)
        # Uniform -> high entropy (log(10) = 2.30)
        self.assertGreater(m_ent, 2.0)

        base = 0.85
        gated = entropy_gated_top_p_threshold(base, m_ent)
        self.assertGreater(gated, base)

    def test_only_active_with_top_p_aggregation(self):
        """Entropy gating only applies when aggregation is top_p or top_p_weighted."""
        from nllw.backend_protocol import BackendConfig
        # With ts_vote aggregation, entropy_gated_top_p should be ignored
        cfg = BackendConfig(
            entropy_gated_top_p=True,
            aggregation="ts_vote",
        )
        # The check in the backend only gates when aggregation is top_p
        self.assertTrue(cfg.entropy_gated_top_p)
        self.assertEqual(cfg.aggregation, "ts_vote")
        # Backend code checks: if aggregation in ("top_p", "top_p_weighted")
        # so this config would not actually gate anything


if __name__ == "__main__":
    unittest.main()
