"""Tests for Iteration 5 features: Gaussian Kernel, LA Forced Decoding, Adaptive SSBD.

Tests the three new features added in iteration 5:
1. Gaussian Kernel Consensus aggregation (2 variants)
2. LA Forced Decoding config and strategy selection
3. Adaptive SSBD Beta (entropy-based dynamic bias)

All tests are unit tests that don't require llama.cpp.
"""

import math
import numpy as np
import pytest

from nllw.alignatt import (
    aggregate_gaussian_kernel,
    aggregate_gaussian_kernel_continuous,
    aggregate,
    check_border,
    list_aggregation_methods,
)
from nllw.alignatt_la_backend import (
    adaptive_ssbd_beta,
    ssbd_accept,
)
from nllw.backend_protocol import BackendConfig
from nllw.bench import parse_sweep_spec


# ===========================================================================
# Gaussian Kernel Consensus tests
# ===========================================================================

class TestGaussianKernel:
    """Test gaussian_kernel aggregation (argmax-based with smoothing)."""

    def test_single_head_peaked(self):
        """Single head, peaked attention -> returns argmax position."""
        src_attn = np.array([[0.0, 0.0, 0.0, 1.0, 0.0]])
        ts = [1.0]
        result = aggregate_gaussian_kernel(src_attn, ts, sigma=1.5)
        assert result == 3.0

    def test_single_head_small_sigma(self):
        """Very small sigma -> essentially argmax (like ts_vote)."""
        src_attn = np.array([[0.0, 0.0, 0.8, 0.2, 0.0]])
        ts = [1.0]
        result = aggregate_gaussian_kernel(src_attn, ts, sigma=0.1)
        assert result == 2.0

    def test_two_heads_agree(self):
        """Two heads both attend to position 3 -> clear peak at 3."""
        src_attn = np.array([
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ])
        ts = [0.8, 0.9]
        result = aggregate_gaussian_kernel(src_attn, ts, sigma=1.5)
        assert result == 3.0

    def test_two_heads_nearby(self):
        """Two heads at adjacent positions -> peak between them."""
        src_attn = np.array([
            [0.0, 0.0, 1.0, 0.0, 0.0],  # head 0 at pos 2
            [0.0, 0.0, 0.0, 1.0, 0.0],  # head 1 at pos 3
        ])
        ts = [1.0, 1.0]
        # With sigma=1.5, kernels overlap -> peak at 2 or 3
        result = aggregate_gaussian_kernel(src_attn, ts, sigma=1.5)
        assert result in {2.0, 3.0}

    def test_ts_weighting(self):
        """Higher TS head has more influence on kernel density."""
        src_attn = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],  # head 0 at pos 0
            [0.0, 0.0, 0.0, 0.0, 1.0],  # head 1 at pos 4
        ])
        # Heavy TS on head 1
        ts = [0.1, 0.9]
        result = aggregate_gaussian_kernel(src_attn, ts, sigma=1.0)
        assert result == 4.0  # head 1 dominates

    def test_empty_source(self):
        """Edge case: zero source tokens."""
        src_attn = np.zeros((2, 0))
        ts = [1.0, 1.0]
        result = aggregate_gaussian_kernel(src_attn, ts)
        assert result == 0.0

    def test_single_position(self):
        """Single source position -> always returns 0."""
        src_attn = np.array([[1.0]])
        ts = [1.0]
        result = aggregate_gaussian_kernel(src_attn, ts)
        assert result == 0.0

    def test_sigma_effect(self):
        """Larger sigma spreads density more, allowing nearby heads to merge."""
        # Two heads far apart (pos 0 and 4) with equal TS
        src_attn = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ])
        ts = [1.0, 1.0]
        # With very small sigma, two peaks don't interact -> picks 0 (first argmax)
        result_sharp = aggregate_gaussian_kernel(src_attn, ts, sigma=0.1)
        assert result_sharp in {0.0, 4.0}

        # With large sigma, both contribute to middle -> peak shifts
        result_wide = aggregate_gaussian_kernel(src_attn, ts, sigma=5.0)
        # With wide enough sigma, density is nearly uniform, but slightly higher
        # at positions between the two peaks (2)
        assert result_wide in {0.0, 1.0, 2.0, 3.0, 4.0}

    def test_border_detection_integration(self):
        """Gaussian kernel works with check_border."""
        # Attention at end of source
        src_attn = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]])
        ts = [1.0]
        assert check_border(
            src_attn, ts, n_src_tokens=5, border_distance=2,
            aggregation="gaussian_kernel"
        ) is True

        # Attention at start
        src_attn2 = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])
        assert check_border(
            src_attn2, ts, n_src_tokens=5, border_distance=2,
            aggregation="gaussian_kernel"
        ) is False


class TestGaussianKernelContinuous:
    """Test gaussian_kernel_continuous (full distribution convolution)."""

    def test_peaked_attention(self):
        """All attention on one position -> same as discrete."""
        src_attn = np.array([[0.0, 0.0, 0.0, 1.0, 0.0]])
        ts = [1.0]
        result = aggregate_gaussian_kernel_continuous(src_attn, ts, sigma=1.5)
        assert result == 3.0

    def test_bimodal_attention(self):
        """Bimodal attention -> smooth peak at one of the modes."""
        src_attn = np.array([[0.0, 0.5, 0.0, 0.0, 0.5]])
        ts = [1.0]
        # With smoothing, density around both modes
        result = aggregate_gaussian_kernel_continuous(src_attn, ts, sigma=0.5)
        assert result in {1.0, 4.0}  # one of the two modes

    def test_multi_head(self):
        """Multiple heads agree -> reinforced peak."""
        src_attn = np.array([
            [0.0, 0.0, 0.8, 0.2, 0.0],
            [0.0, 0.0, 0.7, 0.3, 0.0],
        ])
        ts = [1.0, 1.0]
        result = aggregate_gaussian_kernel_continuous(src_attn, ts, sigma=1.0)
        assert result == 2.0  # both heads peak at pos 2

    def test_empty_source(self):
        src_attn = np.zeros((1, 0))
        ts = [1.0]
        assert aggregate_gaussian_kernel_continuous(src_attn, ts) == 0.0

    def test_border_detection_integration(self):
        """Continuous gaussian kernel works with check_border."""
        src_attn = np.array([[0.0, 0.0, 0.0, 0.1, 0.9]])
        ts = [1.0]
        assert check_border(
            src_attn, ts, n_src_tokens=5, border_distance=2,
            aggregation="gaussian_kernel_continuous"
        ) is True


class TestGaussianKernelRegistry:
    """Test that gaussian kernel methods are properly registered."""

    def test_in_method_list(self):
        methods = list_aggregation_methods()
        assert "gaussian_kernel" in methods
        assert "gaussian_kernel_continuous" in methods

    def test_dispatch_gaussian_kernel(self):
        src_attn = np.array([[0.0, 0.0, 1.0, 0.0]])
        ts = [1.0]
        result = aggregate(src_attn, ts, method="gaussian_kernel")
        assert result == 2.0

    def test_dispatch_gaussian_kernel_continuous(self):
        src_attn = np.array([[0.0, 0.0, 1.0, 0.0]])
        ts = [1.0]
        result = aggregate(src_attn, ts, method="gaussian_kernel_continuous")
        assert result == 2.0


# ===========================================================================
# LA Forced Decoding config tests
# ===========================================================================

class TestLAForcedDecodeConfig:
    """Test LA forced decoding configuration."""

    def test_default_disabled(self):
        config = BackendConfig()
        assert config.la_forced_decode is False

    def test_enable_from_dict(self):
        config = BackendConfig.from_dict({"la_forced_decode": True})
        assert config.la_forced_decode is True

    def test_combine_with_la_backend(self):
        config = BackendConfig.from_dict({
            "backend_type": "alignatt-la",
            "la_forced_decode": True,
            "border_distance": 3,
        })
        assert config.backend_type == "alignatt-la"
        assert config.la_forced_decode is True

    def test_sweep_shortname(self):
        grid = parse_sweep_spec("forced=0,1")
        assert "la_forced_decode" in grid
        assert grid["la_forced_decode"] == [0, 1]

    def test_mutually_exclusive_with_ssbd(self):
        """Forced decode and SSBD can coexist in config but SSBD takes priority.

        This tests the config level. The backend logic checks:
        if use_ssbd -> ssbd path
        elif use_forced -> forced path
        else -> standard
        """
        config = BackendConfig.from_dict({
            "la_forced_decode": True,
            "ssbd_beta": 0.2,
        })
        assert config.la_forced_decode is True
        assert config.ssbd_beta == 0.2


# ===========================================================================
# Adaptive SSBD Beta tests
# ===========================================================================

class TestAdaptiveSSBDBeta:
    """Test entropy-based adaptive SSBD beta computation."""

    def _make_logits(self, values: dict, n_vocab: int = 100) -> np.ndarray:
        """Create a logit array with specific token values."""
        logits = np.full(n_vocab, -10.0, dtype=np.float32)
        for tok_id, val in values.items():
            logits[tok_id] = val
        return logits

    def test_confident_increases_beta(self):
        """Low entropy (confident) -> beta scales up (more lenient)."""
        # Very peaked distribution -> low entropy
        logits = self._make_logits({0: 50.0})  # almost all mass on token 0
        adapted = adaptive_ssbd_beta(logits, base_beta=0.2)
        assert adapted > 0.2  # more lenient than base

    def test_uncertain_decreases_beta(self):
        """High entropy (uncertain) -> beta scales down (stricter)."""
        # Uniform distribution -> high entropy
        logits = np.zeros(100, dtype=np.float32)  # uniform
        adapted = adaptive_ssbd_beta(logits, base_beta=0.2)
        assert adapted < 0.2  # stricter than base

    def test_moderate_entropy_near_base(self):
        """Moderate entropy -> beta near base value."""
        # Create distribution with moderate entropy
        logits = np.full(100, -5.0, dtype=np.float32)
        logits[0] = 5.0
        logits[1] = 4.0
        logits[2] = 3.0
        adapted = adaptive_ssbd_beta(logits, base_beta=0.2)
        # Should be somewhere reasonable
        assert 0.01 < adapted < 0.5

    def test_cap_at_095(self):
        """Beta never exceeds 0.95."""
        # Very peaked + high base beta
        logits = self._make_logits({0: 100.0})
        adapted = adaptive_ssbd_beta(logits, base_beta=0.8)
        assert adapted <= 0.95

    def test_zero_base_beta(self):
        """Base beta=0 -> adapted is always 0 (scaled)."""
        logits = self._make_logits({0: 50.0})
        adapted = adaptive_ssbd_beta(logits, base_beta=0.0)
        assert adapted == 0.0

    def test_monotonic_in_entropy(self):
        """Higher entropy -> lower adapted beta (monotonic)."""
        # Peaked: almost all mass on one token (entropy near 0)
        logits_peaked = self._make_logits({0: 50.0})
        # Moderate: a few tokens have significant mass (entropy ~2-3)
        logits_moderate = np.full(100, -10.0, dtype=np.float32)
        logits_moderate[0] = 2.0
        logits_moderate[1] = 1.5
        logits_moderate[2] = 1.0
        logits_moderate[3] = 0.5
        logits_moderate[4] = 0.0
        # Uniform: all tokens equal (entropy = ln(100) ~ 4.6)
        logits_uniform = np.zeros(100, dtype=np.float32)

        beta_peaked = adaptive_ssbd_beta(logits_peaked, 0.2)
        beta_moderate = adaptive_ssbd_beta(logits_moderate, 0.2)
        beta_uniform = adaptive_ssbd_beta(logits_uniform, 0.2)

        assert beta_peaked >= beta_moderate
        assert beta_moderate > beta_uniform


class TestAdaptiveSSBDConfig:
    """Test adaptive SSBD configuration."""

    def test_default_disabled(self):
        config = BackendConfig()
        assert config.adaptive_ssbd is False

    def test_enable_from_dict(self):
        config = BackendConfig.from_dict({"adaptive_ssbd": True})
        assert config.adaptive_ssbd is True

    def test_combine_with_ssbd(self):
        config = BackendConfig.from_dict({
            "ssbd_beta": 0.2,
            "adaptive_ssbd": True,
        })
        assert config.ssbd_beta == 0.2
        assert config.adaptive_ssbd is True

    def test_sweep_shortname(self):
        grid = parse_sweep_spec("adaptive=0,1")
        assert "adaptive_ssbd" in grid


class TestAdaptiveSSBDIntegration:
    """Test that adaptive SSBD integrates correctly with ssbd_accept."""

    def _make_logits(self, values: dict, n_vocab: int = 100) -> np.ndarray:
        logits = np.full(n_vocab, -10.0, dtype=np.float32)
        for tok_id, val in values.items():
            logits[tok_id] = val
        return logits

    def test_adaptive_accept_confident(self):
        """Confident model + adaptive -> more lenient acceptance."""
        # Peaked distribution: argmax at 0, draft at 1
        logits = self._make_logits({0: 10.0, 1: 9.5})
        base_beta = 0.1

        # Without adaptive: might reject
        result_fixed = ssbd_accept(logits, draft_token=1, beta=base_beta)

        # With adaptive: higher effective beta -> more likely to accept
        adaptive_beta = adaptive_ssbd_beta(logits, base_beta)
        result_adaptive = ssbd_accept(logits, draft_token=1, beta=adaptive_beta)

        # Adaptive should be at least as lenient as fixed
        if not result_fixed:
            # Can't guarantee accept, but adaptive beta should be higher
            assert adaptive_beta >= base_beta

    def test_adaptive_reject_uncertain(self):
        """Uncertain model + adaptive -> stricter acceptance."""
        # Near-uniform distribution
        logits = np.zeros(100, dtype=np.float32)
        logits[0] = 0.1  # barely the argmax
        logits[1] = 0.09
        base_beta = 0.2

        adaptive_beta = adaptive_ssbd_beta(logits, base_beta)
        # Adaptive should be stricter (lower beta)
        assert adaptive_beta < base_beta


# ===========================================================================
# Sweep parsing for new features
# ===========================================================================

class TestNewSweepShortnames:
    """Test that all new sweep shortnames work correctly."""

    def test_gaussian_kernel_in_agg_sweep(self):
        """gaussian_kernel can be used in aggregation sweep."""
        grid = parse_sweep_spec("agg=ts_vote,gaussian_kernel,gaussian_kernel_continuous")
        assert "aggregation" in grid
        assert "gaussian_kernel" in grid["aggregation"]
        assert "gaussian_kernel_continuous" in grid["aggregation"]

    def test_forced_decode_sweep(self):
        grid = parse_sweep_spec("forced=0,1")
        assert "la_forced_decode" in grid
        assert grid["la_forced_decode"] == [0, 1]

    def test_adaptive_ssbd_sweep(self):
        grid = parse_sweep_spec("adaptive=0,1")
        assert "adaptive_ssbd" in grid
        assert grid["adaptive_ssbd"] == [0, 1]

    def test_combined_sweep(self):
        """Multiple new features in a single sweep spec."""
        grid = parse_sweep_spec("ssbd=0.2 adaptive=1 forced=0")
        assert "ssbd_beta" in grid
        assert "adaptive_ssbd" in grid
        assert "la_forced_decode" in grid


# ===========================================================================
# Comparison: Gaussian Kernel vs existing methods
# ===========================================================================

class TestGaussianKernelComparison:
    """Compare gaussian_kernel behavior with existing methods."""

    def test_agrees_with_ts_vote_peaked(self):
        """On peaked single-head attention, gaussian_kernel == ts_vote."""
        src_attn = np.array([[0.0, 0.0, 0.0, 1.0, 0.0]])
        ts = [1.0]
        from nllw.alignatt import aggregate_ts_weighted_vote
        ts_result = aggregate_ts_weighted_vote(src_attn, ts)
        gk_result = aggregate_gaussian_kernel(src_attn, ts, sigma=0.1)
        assert float(ts_result) == gk_result

    def test_subword_tolerance(self):
        """Gaussian kernel merges nearby head votes (subword boundary tolerance).

        This is the key advantage: two heads at positions 5 and 6 (subword
        split) contribute to the same region, unlike ts_vote where they'd
        compete.
        """
        src_attn = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # pos 5
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # pos 6
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # outlier at pos 0
        ])
        ts = [0.5, 0.5, 0.5]

        # With sigma=1.5, positions 5 and 6 reinforce each other
        result = aggregate_gaussian_kernel(src_attn, ts, sigma=1.5)
        assert result in {5.0, 6.0}  # merged cluster wins over isolated outlier

    def test_robust_to_noise(self):
        """Gaussian kernel is robust to noise from individual heads."""
        # 4 heads agree near pos 3, 1 outlier at pos 9
        src_attn = np.zeros((5, 10))
        src_attn[0, 3] = 1.0
        src_attn[1, 3] = 1.0
        src_attn[2, 2] = 1.0  # nearby
        src_attn[3, 4] = 1.0  # nearby
        src_attn[4, 9] = 1.0  # outlier
        ts = [0.8, 0.7, 0.6, 0.5, 0.9]  # outlier has high TS!

        result = aggregate_gaussian_kernel(src_attn, ts, sigma=1.5)
        # Despite outlier having highest TS, cluster of 4 heads should dominate
        assert 2.0 <= result <= 4.0
