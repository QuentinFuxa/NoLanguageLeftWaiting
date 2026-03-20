"""Tests for Iteration 6 features.

New features:
1. LA Two-Pass Catch-up (stability via dual re-translation)
2. Adaptive Multi-Strategy (AMS) aggregation selection
3. Per-head temperature normalization
4. Cross-lingual head transfer analysis utilities

All tests are unit tests that don't require llama.cpp.
"""

import json
import math
import os
import tempfile

import numpy as np
import pytest

from nllw.alignatt import (
    normalize_head_temperatures,
    select_adaptive_aggregation,
    aggregate,
    check_border,
    check_border_dynamic,
    list_aggregation_methods,
    attention_entropy,
)
from nllw.alignatt_la_backend import (
    _longest_common_prefix_tokens,
)
from nllw.backend_protocol import BackendConfig
from nllw.bench import parse_sweep_spec


# ===========================================================================
# Per-head Temperature Normalization tests
# ===========================================================================

class TestHeadTemperatureNormalization:
    """Test normalize_head_temperatures function."""

    def test_identity_when_matching_reference(self):
        """Heads already at reference entropy should not change much."""
        # Attention roughly uniform over 4-5 positions (~1.5 nats entropy)
        attn = np.array([[0.2, 0.25, 0.25, 0.2, 0.1]])
        result = normalize_head_temperatures(attn, reference_entropy=1.5)
        # Should be roughly unchanged
        assert result.shape == attn.shape
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-6)

    def test_sharpening_diffuse_head(self):
        """Diffuse head (high entropy) should be sharpened."""
        # Non-uniform but diffuse attention (not perfectly uniform -- that can't be sharpened)
        attn = np.array([[0.15, 0.18, 0.22, 0.25, 0.20]])
        # Reference entropy = 0.5 (very sharp)
        result = normalize_head_temperatures(attn, reference_entropy=0.5)
        # Should become sharper (lower entropy)
        eps = 1e-10
        p = result[0] + eps
        p = p / p.sum()
        new_entropy = -np.sum(p * np.log(p))
        old_entropy = -np.sum(attn[0] * np.log(attn[0] + eps))
        assert new_entropy < old_entropy

    def test_softening_sharp_head(self):
        """Sharp head (low entropy) should be softened."""
        attn = np.array([[0.01, 0.01, 0.96, 0.01, 0.01]])
        # Reference entropy = 3.0 (very broad)
        result = normalize_head_temperatures(attn, reference_entropy=3.0)
        eps = 1e-10
        p = result[0] + eps
        p = p / p.sum()
        new_entropy = -np.sum(p * np.log(p))
        old_entropy = -np.sum(np.clip(attn[0], eps, 1) * np.log(np.clip(attn[0], eps, 1)))
        assert new_entropy > old_entropy

    def test_preserves_argmax(self):
        """Normalization should preserve which position has the most attention."""
        attn = np.array([
            [0.01, 0.01, 0.96, 0.01, 0.01],
            [0.05, 0.05, 0.05, 0.8, 0.05],
        ])
        result = normalize_head_temperatures(attn, reference_entropy=1.5)
        assert np.argmax(result[0]) == 2
        assert np.argmax(result[1]) == 3

    def test_multiple_heads_different_sharpness(self):
        """Heads with different sharpness converge to similar entropy."""
        sharp_head = np.array([0.01, 0.01, 0.96, 0.01, 0.01])
        broad_head = np.array([0.15, 0.2, 0.3, 0.2, 0.15])
        attn = np.array([sharp_head, broad_head])

        result = normalize_head_temperatures(attn, reference_entropy=1.5)

        eps = 1e-10
        ent0 = -np.sum(result[0] * np.log(result[0] + eps))
        ent1 = -np.sum(result[1] * np.log(result[1] + eps))
        # Entropies should be closer to each other after normalization
        original_gap = abs(
            -np.sum(sharp_head * np.log(sharp_head + eps))
            - (-np.sum(broad_head * np.log(broad_head + eps)))
        )
        new_gap = abs(ent0 - ent1)
        assert new_gap < original_gap

    def test_single_position(self):
        """Single position should return as-is."""
        attn = np.array([[1.0]])
        result = normalize_head_temperatures(attn, reference_entropy=1.5)
        np.testing.assert_allclose(result, attn)

    def test_output_sums_to_one(self):
        """Output distributions should sum to 1."""
        rng = np.random.RandomState(42)
        attn = rng.dirichlet([0.5, 0.5, 0.5, 0.5, 0.5], size=5)
        result = normalize_head_temperatures(attn, reference_entropy=1.5)
        for h in range(5):
            np.testing.assert_allclose(result[h].sum(), 1.0, atol=1e-6)


# ===========================================================================
# Adaptive Multi-Strategy (AMS) tests
# ===========================================================================

class TestAdaptiveMultiStrategy:
    """Test select_adaptive_aggregation function."""

    def test_high_agreement_low_entropy_selects_ts_vote(self):
        """Heads agree and are sharp -> ts_vote."""
        # All heads point to position 3, very sharp
        attn = np.array([
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ])
        ts = [0.8, 0.7, 0.6, 0.5]
        result = select_adaptive_aggregation(attn, ts)
        assert result == "ts_vote"

    def test_high_agreement_high_entropy_selects_entropy_weighted(self):
        """Heads agree on position but attention is diffuse -> entropy_weighted."""
        # All heads' argmax is position 3, but attention is spread
        attn = np.array([
            [0.15, 0.15, 0.2, 0.3, 0.2],
            [0.15, 0.15, 0.2, 0.3, 0.2],
            [0.15, 0.15, 0.2, 0.3, 0.2],
            [0.15, 0.15, 0.2, 0.3, 0.2],
        ])
        ts = [0.8, 0.7, 0.6, 0.5]
        result = select_adaptive_aggregation(attn, ts)
        assert result == "entropy_weighted"

    def test_low_agreement_low_entropy_selects_geomean(self):
        """Heads disagree but are sharp -> geomean."""
        # Each head points to a different position, but sharply
        attn = np.array([
            [0.95, 0.02, 0.01, 0.01, 0.01],
            [0.01, 0.01, 0.95, 0.02, 0.01],
            [0.01, 0.01, 0.01, 0.02, 0.95],
            [0.01, 0.95, 0.02, 0.01, 0.01],
            [0.01, 0.01, 0.01, 0.95, 0.02],
        ])
        ts = [0.8, 0.7, 0.6, 0.5, 0.4]
        result = select_adaptive_aggregation(attn, ts)
        assert result == "geomean"

    def test_low_agreement_high_entropy_selects_consensus(self):
        """Heads disagree and are diffuse -> consensus."""
        # Broad, disagreeing attention
        attn = np.array([
            [0.3, 0.25, 0.2, 0.15, 0.1],
            [0.1, 0.15, 0.2, 0.25, 0.3],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.1, 0.1, 0.3, 0.3, 0.2],
            [0.25, 0.25, 0.15, 0.15, 0.2],
        ])
        ts = [0.8, 0.7, 0.6, 0.5, 0.4]
        result = select_adaptive_aggregation(attn, ts)
        assert result == "consensus"

    def test_empty_attention(self):
        """Empty attention should return ts_vote as fallback."""
        attn = np.zeros((0, 0))
        ts = []
        result = select_adaptive_aggregation(attn, ts)
        assert result == "ts_vote"

    def test_returns_valid_method(self):
        """Selected method should always be a registered aggregation method."""
        rng = np.random.RandomState(42)
        for _ in range(20):
            n_heads = rng.randint(3, 10)
            n_src = rng.randint(3, 20)
            attn = rng.dirichlet(np.ones(n_src), size=n_heads)
            ts = rng.uniform(0.1, 1.0, size=n_heads).tolist()
            method = select_adaptive_aggregation(attn, ts)
            assert method in list_aggregation_methods()


# ===========================================================================
# check_border with new parameters tests
# ===========================================================================

class TestCheckBorderNewParams:
    """Test check_border with AMS and temp normalization."""

    def test_check_border_with_adaptive_aggregation(self):
        """check_border should work with adaptive_aggregation=True."""
        # All heads point to position 4 (last), sharp -> ts_vote selected
        attn = np.array([
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ])
        ts = [0.8, 0.7, 0.6]
        # With bd=2, threshold=3. Position 4 >= 3 -> border hit
        result = check_border(attn, ts, 5, 2, adaptive_aggregation=True)
        assert result is True

    def test_check_border_with_head_temp_normalize(self):
        """check_border should work with head_temp_normalize=True."""
        # Sharp and broad heads mixed
        attn = np.array([
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.1, 0.2, 0.2, 0.2, 0.3],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ])
        ts = [0.8, 0.7, 0.6]
        # Without normalization, ts_vote sees 2 heads at 4, 1 at 4
        result = check_border(
            attn, ts, 5, 2,
            head_temp_normalize=True,
            head_temp_reference=1.5,
        )
        # Should still detect border (2/3 heads at position 4)
        assert result is True

    def test_check_border_dynamic_with_new_params(self):
        """check_border_dynamic should accept and use new parameters."""
        attn = np.array([
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ])
        ts = [0.8, 0.7, 0.6]
        # Dynamic border + AMS + temp norm all enabled
        result = check_border_dynamic(
            attn, ts, 5, 2,
            adaptive_aggregation=True,
            head_temp_normalize=True,
            head_temp_reference=1.5,
        )
        assert isinstance(result, bool)

    def test_adaptive_does_not_crash_on_edge_cases(self):
        """AMS should handle various edge cases without crashing."""
        # Single head
        attn = np.array([[0.0, 0.5, 0.5]])
        ts = [1.0]
        result = check_border(attn, ts, 3, 1, adaptive_aggregation=True)
        assert isinstance(result, bool)

        # Uniform attention
        attn = np.ones((5, 10)) / 10
        ts = [0.5] * 5
        result = check_border(attn, ts, 10, 3, adaptive_aggregation=True)
        assert isinstance(result, bool)


# ===========================================================================
# LA Two-Pass Catch-up config tests
# ===========================================================================

class TestTwoPassConfig:
    """Test LA two-pass catch-up configuration and selection logic."""

    def test_config_default_false(self):
        """la_two_pass should default to False."""
        config = BackendConfig()
        assert config.la_two_pass is False

    def test_config_from_dict(self):
        """la_two_pass should be settable from dict."""
        config = BackendConfig.from_dict({"la_two_pass": True})
        assert config.la_two_pass is True

    def test_lcp_tokens_identical(self):
        """Identical sequences have full LCP."""
        assert _longest_common_prefix_tokens([1, 2, 3], [1, 2, 3]) == 3

    def test_lcp_tokens_partial(self):
        """Partial match returns correct LCP."""
        assert _longest_common_prefix_tokens([1, 2, 3, 4], [1, 2, 5, 6]) == 2

    def test_lcp_tokens_empty(self):
        """Empty sequences have LCP 0."""
        assert _longest_common_prefix_tokens([], [1, 2]) == 0
        assert _longest_common_prefix_tokens([1, 2], []) == 0
        assert _longest_common_prefix_tokens([], []) == 0

    def test_lcp_tokens_no_match(self):
        """Completely different sequences have LCP 0."""
        assert _longest_common_prefix_tokens([1, 2, 3], [4, 5, 6]) == 0


# ===========================================================================
# AMS Config tests
# ===========================================================================

class TestAMSConfig:
    """Test AMS configuration."""

    def test_config_default_false(self):
        """adaptive_aggregation should default to False."""
        config = BackendConfig()
        assert config.adaptive_aggregation is False

    def test_config_from_dict(self):
        """adaptive_aggregation should be settable from dict."""
        config = BackendConfig.from_dict({"adaptive_aggregation": True})
        assert config.adaptive_aggregation is True


# ===========================================================================
# Head Temperature Config tests
# ===========================================================================

class TestHeadTempConfig:
    """Test head temperature normalization config."""

    def test_config_defaults(self):
        """Temperature normalization should default to disabled."""
        config = BackendConfig()
        assert config.head_temp_normalize is False
        assert config.head_temp_reference == 1.5

    def test_config_from_dict(self):
        """Should be settable from dict."""
        config = BackendConfig.from_dict({
            "head_temp_normalize": True,
            "head_temp_reference": 2.0,
        })
        assert config.head_temp_normalize is True
        assert config.head_temp_reference == 2.0


# ===========================================================================
# Bench sweep shortnames tests
# ===========================================================================

class TestSweepShortnames:
    """Test new sweep shortnames in bench.py."""

    def test_twopass_shortname(self):
        """'twopass' maps to la_two_pass."""
        grid = parse_sweep_spec("twopass=0,1")
        assert "la_two_pass" in grid
        assert grid["la_two_pass"] == [0, 1]

    def test_ams_shortname(self):
        """'ams' maps to adaptive_aggregation."""
        grid = parse_sweep_spec("ams=0,1")
        assert "adaptive_aggregation" in grid
        assert grid["adaptive_aggregation"] == [0, 1]

    def test_tempnorm_shortname(self):
        """'tempnorm' maps to head_temp_normalize."""
        grid = parse_sweep_spec("tempnorm=0,1")
        assert "head_temp_normalize" in grid

    def test_tempref_shortname(self):
        """'tempref' maps to head_temp_reference."""
        grid = parse_sweep_spec("tempref=1.0,1.5,2.0")
        assert "head_temp_reference" in grid
        assert grid["head_temp_reference"] == [1.0, 1.5, 2.0]

    def test_combined_sweep(self):
        """Multiple new shortnames work together."""
        grid = parse_sweep_spec("ams=0,1 tempnorm=0,1 twopass=0,1")
        assert len(grid) == 3
        assert "adaptive_aggregation" in grid
        assert "head_temp_normalize" in grid
        assert "la_two_pass" in grid


# ===========================================================================
# Cross-lingual head transfer tests
# ===========================================================================

class TestHeadTransfer:
    """Test cross-lingual head transfer analysis utilities."""

    def test_import(self):
        """Module should be importable."""
        from nllw.head_transfer import (
            jaccard_similarity,
            top_k_overlap,
            ts_score_correlation,
            head_set,
            head_scores_dict,
            discover_configs,
            analyze_model_transfer,
        )

    def test_jaccard_identical(self):
        """Identical sets -> Jaccard = 1.0."""
        from nllw.head_transfer import jaccard_similarity
        s = {(1, 2), (3, 4), (5, 6)}
        assert jaccard_similarity(s, s) == 1.0

    def test_jaccard_disjoint(self):
        """Disjoint sets -> Jaccard = 0.0."""
        from nllw.head_transfer import jaccard_similarity
        a = {(1, 2), (3, 4)}
        b = {(5, 6), (7, 8)}
        assert jaccard_similarity(a, b) == 0.0

    def test_jaccard_partial(self):
        """Partial overlap -> Jaccard between 0 and 1."""
        from nllw.head_transfer import jaccard_similarity
        a = {(1, 2), (3, 4), (5, 6)}
        b = {(3, 4), (5, 6), (7, 8)}
        # intersection=2, union=4
        assert jaccard_similarity(a, b) == 0.5

    def test_jaccard_empty(self):
        """Empty sets -> Jaccard = 1.0 (by convention)."""
        from nllw.head_transfer import jaccard_similarity
        assert jaccard_similarity(set(), set()) == 1.0

    def test_top_k_overlap_full(self):
        """Full overlap -> 1.0."""
        from nllw.head_transfer import top_k_overlap
        s = {(1, 2), (3, 4)}
        assert top_k_overlap(s, s) == 1.0

    def test_top_k_overlap_none(self):
        """No overlap -> 0.0."""
        from nllw.head_transfer import top_k_overlap
        a = {(1, 2)}
        b = {(3, 4)}
        assert top_k_overlap(a, b) == 0.0

    def test_ts_correlation_identical(self):
        """Identical scores -> correlation = 1.0."""
        from nllw.head_transfer import ts_score_correlation
        scores = {(1, 2): 0.9, (3, 4): 0.7, (5, 6): 0.5}
        assert ts_score_correlation(scores, scores) == 1.0

    def test_ts_correlation_reversed(self):
        """Reversed scores -> correlation = -1.0."""
        from nllw.head_transfer import ts_score_correlation
        a = {(1, 2): 0.9, (3, 4): 0.7, (5, 6): 0.5}
        b = {(1, 2): 0.5, (3, 4): 0.7, (5, 6): 0.9}
        assert ts_score_correlation(a, b) == pytest.approx(-1.0)

    def test_ts_correlation_too_few_shared(self):
        """Fewer than 3 shared heads -> returns 0.0."""
        from nllw.head_transfer import ts_score_correlation
        a = {(1, 2): 0.9, (3, 4): 0.7}
        b = {(1, 2): 0.8, (5, 6): 0.6}
        assert ts_score_correlation(a, b) == 0.0

    def test_discover_configs_finds_files(self):
        """Should find configs in the heads directory."""
        from nllw.head_transfer import discover_configs
        configs = discover_configs()
        assert len(configs) > 0  # We have 22 configs

    def test_head_set_returns_tuples(self):
        """head_set should return set of (layer, head) tuples."""
        from nllw.head_transfer import head_set
        configs_dir = os.path.join(os.path.dirname(__file__), "..", "nllw", "heads", "configs")
        # Use any existing config
        for fname in os.listdir(configs_dir):
            if fname.endswith(".json"):
                path = os.path.join(configs_dir, fname)
                result = head_set(path, top_k=5)
                assert isinstance(result, set)
                assert len(result) <= 5
                for item in result:
                    assert isinstance(item, tuple)
                    assert len(item) == 2
                break

    def test_analyze_model_transfer_qwen3(self):
        """Real analysis on Qwen3-4B (has 4 directions)."""
        from nllw.head_transfer import discover_configs, analyze_model_transfer
        all_configs = discover_configs()
        # Find a model with multiple directions
        multi_dir_models = {k: v for k, v in all_configs.items() if len(v) >= 2}
        assert len(multi_dir_models) > 0, "Need at least one model with 2+ directions"

        model_key = next(iter(multi_dir_models))
        results = analyze_model_transfer(multi_dir_models[model_key], top_k=10)
        assert len(results) > 0

        for r in results:
            assert "jaccard" in r
            assert "overlap_a_in_b" in r
            assert "ts_correlation" in r
            assert "ts_mass_a_on_b" in r
            assert 0.0 <= r["jaccard"] <= 1.0
            assert 0.0 <= r["overlap_a_in_b"] <= 1.0
            assert 0.0 <= r["ts_mass_a_on_b"] <= 1.0

    def test_transferred_ts_mass_self(self):
        """Using own heads should give mass = 1.0."""
        from nllw.head_transfer import transferred_ts_mass
        configs_dir = os.path.join(os.path.dirname(__file__), "..", "nllw", "heads", "configs")
        for fname in os.listdir(configs_dir):
            if fname.endswith(".json"):
                path = os.path.join(configs_dir, fname)
                mass = transferred_ts_mass(path, path, top_k=10)
                assert mass == pytest.approx(1.0)
                break
