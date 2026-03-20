"""Tests for Iteration 8 features: LSG logit KL divergence.

LSG (Look, See, and Generate, arxiv 2501.00868, AAAI 2025):
Training-free read/write policy using KL divergence between output logit
distributions with full vs reduced source. Integrated as a border
confirmation signal in AlignAtt and AlignAtt-LA backends.

All tests are unit tests that don't require llama.cpp.
"""

import numpy as np
import pytest

from nllw.alignatt import compute_logit_kl
from nllw.backend_protocol import BackendConfig
from nllw.bench import parse_sweep_spec
from nllw.complexity import adaptive_params_from_complexity, estimate_complexity


# ===========================================================================
# compute_logit_kl tests
# ===========================================================================

class TestComputeLogitKL:
    """Tests for KL divergence between logit distributions."""

    def test_identical_distributions_zero_kl(self):
        """Identical logits should give KL = 0."""
        logits = np.array([2.0, 1.0, 0.5, -1.0, -3.0])
        kl = compute_logit_kl(logits, logits)
        assert kl == pytest.approx(0.0, abs=1e-6)

    def test_similar_distributions_low_kl(self):
        """Similar logits should give low KL."""
        logits_full = np.array([2.0, 1.0, 0.5, -1.0, -3.0])
        logits_reduced = np.array([2.1, 0.9, 0.6, -1.1, -2.9])
        kl = compute_logit_kl(logits_full, logits_reduced)
        assert 0 <= kl < 0.5

    def test_different_distributions_high_kl(self):
        """Very different logits should give high KL."""
        logits_full = np.array([10.0, -5.0, -5.0, -5.0, -5.0])
        logits_reduced = np.array([-5.0, -5.0, -5.0, -5.0, 10.0])
        kl = compute_logit_kl(logits_full, logits_reduced)
        assert kl > 5.0

    def test_kl_non_negative(self):
        """KL divergence is always non-negative."""
        rng = np.random.RandomState(42)
        for _ in range(20):
            logits_a = rng.randn(100)
            logits_b = rng.randn(100)
            kl = compute_logit_kl(logits_a, logits_b)
            assert kl >= 0.0

    def test_kl_asymmetric(self):
        """KL(P||Q) != KL(Q||P) in general."""
        logits_p = np.array([5.0, 1.0, -2.0])
        logits_q = np.array([1.0, 5.0, -2.0])
        kl_pq = compute_logit_kl(logits_p, logits_q)
        kl_qp = compute_logit_kl(logits_q, logits_p)
        # Both should be positive but not necessarily equal
        assert kl_pq > 0
        assert kl_qp > 0

    def test_uniform_distributions(self):
        """Uniform logits should give zero KL."""
        logits = np.zeros(1000)
        kl = compute_logit_kl(logits, logits)
        assert kl == pytest.approx(0.0, abs=1e-6)

    def test_large_vocab(self):
        """Should work with large vocabulary sizes (32k+)."""
        rng = np.random.RandomState(123)
        logits_full = rng.randn(32000)
        logits_reduced = logits_full + rng.randn(32000) * 0.1  # small perturbation
        kl = compute_logit_kl(logits_full, logits_reduced)
        assert 0 <= kl < 2.0  # small perturbation -> low KL

    def test_peaked_vs_flat(self):
        """Very peaked distribution vs flat should give high KL."""
        logits_peaked = np.full(100, -10.0)
        logits_peaked[0] = 10.0  # strong peak at position 0
        logits_flat = np.zeros(100)  # uniform
        kl = compute_logit_kl(logits_peaked, logits_flat)
        assert kl > 2.0

    def test_single_element(self):
        """Single-element vocab should give zero KL."""
        kl = compute_logit_kl(np.array([5.0]), np.array([3.0]))
        assert kl == pytest.approx(0.0, abs=1e-6)

    def test_numerical_stability_extreme_logits(self):
        """Should handle extreme logit values without overflow/NaN."""
        logits_full = np.array([100.0, -100.0, 0.0])
        logits_reduced = np.array([0.0, -100.0, 100.0])
        kl = compute_logit_kl(logits_full, logits_reduced)
        assert np.isfinite(kl)
        assert kl > 0


class TestComputeLogitKLThresholds:
    """Tests verifying KL divergence at various threshold levels."""

    def test_small_shift_below_lsg_threshold(self):
        """Small logit shifts should be below typical LSG thresholds (5-9)."""
        rng = np.random.RandomState(42)
        base = rng.randn(10000)
        # Add tiny noise
        shifted = base + rng.randn(10000) * 0.01
        kl = compute_logit_kl(base, shifted)
        assert kl < 5.0  # Well below LSG threshold

    def test_argmax_swap_high_kl(self):
        """Swapping the argmax token should produce high KL."""
        logits_full = np.array([10.0, 5.0, 0.0, -5.0])
        # Swap: token 2 is now argmax instead of token 0
        logits_reduced = np.array([0.0, 5.0, 10.0, -5.0])
        kl = compute_logit_kl(logits_full, logits_reduced)
        assert kl > 5.0  # Above typical threshold


# ===========================================================================
# BackendConfig LSG fields tests
# ===========================================================================

class TestBackendConfigLSG:
    """Tests for LSG config fields in BackendConfig."""

    def test_default_lsg_disabled(self):
        """LSG should be disabled by default."""
        config = BackendConfig()
        assert config.lsg_kl_threshold is None
        assert config.lsg_k == 3

    def test_lsg_from_dict(self):
        """LSG config should load from dict."""
        config = BackendConfig.from_dict({
            "lsg_kl_threshold": 7.0,
            "lsg_k": 5,
        })
        assert config.lsg_kl_threshold == 7.0
        assert config.lsg_k == 5

    def test_lsg_from_dict_unknown_keys_ignored(self):
        """Unknown keys should be silently ignored."""
        config = BackendConfig.from_dict({
            "lsg_kl_threshold": 7.0,
            "unknown_key": 42,
        })
        assert config.lsg_kl_threshold == 7.0

    def test_lsg_none_disabled(self):
        """Explicit None should disable LSG."""
        config = BackendConfig.from_dict({
            "lsg_kl_threshold": None,
        })
        assert config.lsg_kl_threshold is None


# ===========================================================================
# Sweep shortname tests
# ===========================================================================

class TestSweepShortnames:
    """Tests for LSG sweep shortnames in bench.py."""

    def test_lsg_shortname(self):
        """'lsg' shortname maps to lsg_kl_threshold."""
        grid = parse_sweep_spec("lsg=5.0,7.0,9.0")
        assert "lsg_kl_threshold" in grid
        assert grid["lsg_kl_threshold"] == [5.0, 7.0, 9.0]

    def test_lsgk_shortname(self):
        """'lsgk' shortname maps to lsg_k."""
        grid = parse_sweep_spec("lsgk=1,3,5")
        assert "lsg_k" in grid
        assert grid["lsg_k"] == [1, 3, 5]

    def test_lsg_combined_sweep(self):
        """LSG + other params in combined sweep."""
        grid = parse_sweep_spec("bd=2,3 lsg=5.0,7.0")
        assert "border_distance" in grid
        assert "lsg_kl_threshold" in grid
        assert grid["border_distance"] == [2, 3]
        assert grid["lsg_kl_threshold"] == [5.0, 7.0]

    def test_lsg_with_lsgk_sweep(self):
        """LSG threshold + k in same sweep."""
        grid = parse_sweep_spec("lsg=7.0 lsgk=1,3,5")
        assert grid["lsg_kl_threshold"] == [7.0]
        assert grid["lsg_k"] == [1, 3, 5]


# ===========================================================================
# Integration-style tests (no llama.cpp required)
# ===========================================================================

class TestLSGSignalInterpretation:
    """Tests verifying the semantic interpretation of LSG KL values.

    These tests document the expected behavior:
    - Low KL (< threshold): source exhausted, border stop confirmed
    - High KL (> threshold): source still matters, override border stop
    """

    def test_low_kl_confirms_border(self):
        """Low KL means source is exhausted -> border should stop."""
        # Simulate: removing source tokens barely changes output
        logits_full = np.array([8.0, 2.0, -1.0, -3.0, -5.0])
        logits_reduced = np.array([7.8, 2.1, -0.9, -3.1, -4.8])
        kl = compute_logit_kl(logits_full, logits_reduced)

        threshold = 7.0  # LSG recommended
        assert kl < threshold  # Should confirm border (let it stop)

    def test_high_kl_overrides_border(self):
        """High KL means source matters -> border should be overridden."""
        # Simulate: removing source tokens dramatically changes output
        logits_full = np.array([12.0, -5.0, -5.0, -5.0, -5.0])
        logits_reduced = np.array([-5.0, -5.0, 12.0, -5.0, -5.0])
        kl = compute_logit_kl(logits_full, logits_reduced)

        threshold = 7.0  # LSG recommended
        assert kl > threshold  # Should override border (keep generating)

    def test_threshold_sensitivity(self):
        """Verify behavior at different threshold levels."""
        logits_full = np.array([5.0, 1.0, -2.0, -4.0])
        logits_reduced = np.array([1.0, 5.0, -2.0, -4.0])
        kl = compute_logit_kl(logits_full, logits_reduced)

        # KL should be moderate (top tokens swapped but similar magnitude)
        assert 0.5 < kl < 15.0

    def test_monotonic_with_perturbation_scale(self):
        """KL should increase with larger logit perturbations."""
        rng = np.random.RandomState(42)
        base = rng.randn(1000)

        kls = []
        for scale in [0.01, 0.1, 0.5, 1.0, 2.0]:
            perturbed = base + rng.randn(1000) * scale
            kl = compute_logit_kl(base, perturbed)
            kls.append(kl)

        # Should be monotonically increasing (or near-monotonic)
        for i in range(len(kls) - 1):
            assert kls[i] <= kls[i + 1] + 0.1  # small tolerance


class TestLSGEdgeCases:
    """Edge cases for LSG integration."""

    def test_lsg_config_with_all_features(self):
        """LSG should coexist with all other features."""
        config = BackendConfig.from_dict({
            "lsg_kl_threshold": 7.0,
            "lsg_k": 3,
            "shift_k_threshold": 0.4,
            "info_gain_threshold": 0.3,
            "border_confirm": 2,
            "dynamic_word_batch": True,
            "adaptive_aggregation": True,
            "head_temp_normalize": True,
        })
        assert config.lsg_kl_threshold == 7.0
        assert config.shift_k_threshold == 0.4
        assert config.border_confirm == 2

    def test_lsg_k_values(self):
        """Various lsg_k values should be accepted."""
        for k in [1, 2, 3, 4, 5, 10]:
            config = BackendConfig.from_dict({"lsg_k": k})
            assert config.lsg_k == k

    def test_lsg_threshold_values(self):
        """Various threshold values should be accepted."""
        for t in [0.5, 1.0, 3.0, 5.0, 7.0, 9.0, 15.0]:
            config = BackendConfig.from_dict({"lsg_kl_threshold": t})
            assert config.lsg_kl_threshold == t


# ===========================================================================
# Complexity-adaptive parameter tests
# ===========================================================================

class TestComplexityAdaptiveConfig:
    """Tests for complexity_adaptive config field."""

    def test_default_disabled(self):
        """Complexity adaptive should be disabled by default."""
        config = BackendConfig()
        assert config.complexity_adaptive is False

    def test_from_dict(self):
        """Should load from dict."""
        config = BackendConfig.from_dict({"complexity_adaptive": True})
        assert config.complexity_adaptive is True

    def test_sweep_shortname(self):
        """'cmplx' shortname maps to complexity_adaptive."""
        grid = parse_sweep_spec("cmplx=0,1")
        assert "complexity_adaptive" in grid
        assert grid["complexity_adaptive"] == [0, 1]


class TestComplexityAdaptiveParams:
    """Tests for adaptive parameter computation from source complexity."""

    def test_simple_sentence_reduces_bd(self):
        """Simple short sentence should reduce border distance."""
        bd, wb, gen = adaptive_params_from_complexity(
            "hello world", base_bd=3, base_wb=3, base_gen_cap=50
        )
        assert bd <= 3  # Should be reduced (simple sentence)
        assert bd >= 1  # Minimum guard

    def test_complex_sentence_increases_bd(self):
        """Complex long sentence should increase border distance."""
        complex_text = (
            "The president of the European Commission announced 42 "
            "unprecedented emergency economic reforms targeting 15.7% "
            "GDP growth redistribution across 27 member states, including "
            "debt-restructuring programs worth approximately 3.2 trillion euros"
        )
        bd, wb, gen = adaptive_params_from_complexity(
            complex_text, base_bd=3, base_wb=3, base_gen_cap=50
        )
        assert bd >= 3  # Should be same or higher

    def test_minimum_guards(self):
        """Parameters should never go below minimums."""
        bd, wb, gen = adaptive_params_from_complexity(
            "hi", base_bd=1, base_wb=1, base_gen_cap=10
        )
        assert bd >= 1
        assert wb >= 1
        assert gen >= 10

    def test_numerals_increase_complexity(self):
        """Sentences with numerals should score higher complexity."""
        plain = estimate_complexity("the cat sat on the mat today")
        numeric = estimate_complexity("the 42 cats and 17 dogs at 3:45 PM")
        assert numeric.complexity_score >= plain.complexity_score

    def test_empty_string(self):
        """Empty string should return base params."""
        bd, wb, gen = adaptive_params_from_complexity(
            "", base_bd=3, base_wb=3, base_gen_cap=50
        )
        assert bd == 3
        assert wb == 3

    def test_complexity_with_all_features(self):
        """Complexity adaptive should coexist with LSG and other features."""
        config = BackendConfig.from_dict({
            "complexity_adaptive": True,
            "lsg_kl_threshold": 7.0,
            "dynamic_word_batch": True,
            "shift_k_threshold": 0.4,
        })
        assert config.complexity_adaptive is True
        assert config.lsg_kl_threshold == 7.0
        assert config.dynamic_word_batch is True
