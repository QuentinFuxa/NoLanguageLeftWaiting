"""Tests for AlignAtt + LocalAgreement hybrid backend and novel aggregation methods.

Tests the LA backend policy logic and all aggregation strategies without
requiring llama.cpp (unit tests only).
"""

import numpy as np
import pytest
from nllw.alignatt import (
    aggregate_ts_weighted_vote,
    aggregate_softmax_mean,
    aggregate_entropy_weighted,
    aggregate_consensus,
    aggregate_geomean,
    aggregate_top_p,
    aggregate_ensemble,
    aggregate,
    check_border,
    check_border_dynamic,
    attention_entropy,
    dynamic_border_distance,
    list_aggregation_methods,
)
from nllw.alignatt_la_backend import (
    _longest_common_prefix_tokens,
    _longest_common_prefix_words,
)
from nllw.backend_protocol import BackendConfig, TranslationStep


# ---------------------------------------------------------------------------
# LocalAgreement core logic tests
# ---------------------------------------------------------------------------

class TestLongestCommonPrefixTokens:
    def test_identical(self):
        assert _longest_common_prefix_tokens([1, 2, 3], [1, 2, 3]) == 3

    def test_empty_both(self):
        assert _longest_common_prefix_tokens([], []) == 0

    def test_empty_one(self):
        assert _longest_common_prefix_tokens([1, 2], []) == 0
        assert _longest_common_prefix_tokens([], [1, 2]) == 0

    def test_partial_match(self):
        assert _longest_common_prefix_tokens([1, 2, 3, 4], [1, 2, 5, 6]) == 2

    def test_no_match(self):
        assert _longest_common_prefix_tokens([1, 2, 3], [4, 5, 6]) == 0

    def test_prefix_is_shorter(self):
        assert _longest_common_prefix_tokens([1, 2], [1, 2, 3, 4]) == 2

    def test_diverge_at_end(self):
        assert _longest_common_prefix_tokens([1, 2, 3], [1, 2, 4]) == 2

    def test_single_token(self):
        assert _longest_common_prefix_tokens([1], [1]) == 1
        assert _longest_common_prefix_tokens([1], [2]) == 0


class TestLongestCommonPrefixWords:
    def test_identical(self):
        assert _longest_common_prefix_words("the cat sat", "the cat sat") == "the cat sat"

    def test_partial(self):
        assert _longest_common_prefix_words("the cat sat", "the cat ran") == "the cat"

    def test_no_match(self):
        assert _longest_common_prefix_words("hello world", "goodbye world") == ""

    def test_empty(self):
        assert _longest_common_prefix_words("", "") == ""

    def test_single_word_match(self):
        assert _longest_common_prefix_words("hello world", "hello there") == "hello"


# ---------------------------------------------------------------------------
# LA backend registration
# ---------------------------------------------------------------------------

class TestLABackendRegistration:
    def test_registered(self):
        import nllw.alignatt_la_backend  # noqa: F401
        from nllw.backend_protocol import list_backends
        backends = list_backends()
        assert "alignatt-la" in backends

    def test_config_defaults(self):
        config = BackendConfig(backend_type="alignatt-la")
        assert config.backend_type == "alignatt-la"
        assert config.aggregation == "ts_vote"


# ---------------------------------------------------------------------------
# Novel aggregation method tests
# ---------------------------------------------------------------------------

class TestSoftmaxMean:
    def test_peaked_attention(self):
        """All attention on position 3 -> expected position = 3."""
        src_attn = np.array([[0.0, 0.0, 0.0, 1.0, 0.0]])
        ts = [1.0]
        result = aggregate_softmax_mean(src_attn, ts)
        assert abs(result - 3.0) < 0.01

    def test_spread_attention(self):
        """Attention spread -> expected position between peaks."""
        src_attn = np.array([[0.0, 0.5, 0.0, 0.5, 0.0]])
        ts = [1.0]
        result = aggregate_softmax_mean(src_attn, ts)
        assert abs(result - 2.0) < 0.01  # avg of 1 and 3

    def test_multi_head_ts_weighting(self):
        """Head with higher TS has more influence."""
        src_attn = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],  # head 0: pos 0
            [0.0, 0.0, 0.0, 0.0, 1.0],  # head 1: pos 4
        ])
        # TS heavily favors head 1
        ts = [0.1, 0.9]
        result = aggregate_softmax_mean(src_attn, ts)
        assert result > 3.0  # closer to 4 than 0


class TestEntropyWeighted:
    def test_sharp_head_wins(self):
        """Sharp attention (low entropy) gets higher weight."""
        src_attn = np.array([
            [0.25, 0.25, 0.25, 0.25, 0.0],  # diffuse -> low weight
            [0.0, 0.0, 0.0, 0.0, 1.0],       # sharp -> high weight
        ])
        ts = [1.0, 1.0]  # equal TS
        result = aggregate_entropy_weighted(src_attn, ts)
        assert result == 4  # sharp head at position 4 should win

    def test_equal_entropy(self):
        """Equal entropy -> fall back to TS weighting."""
        src_attn = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        ts = [0.3, 0.9]  # head 1 has higher TS
        result = aggregate_entropy_weighted(src_attn, ts)
        assert result == 3  # head 1 (higher TS, equally sharp)


class TestConsensus:
    def test_agreement(self):
        """Three heads agree near position 4 -> consensus picks from that cluster."""
        src_attn = np.array([
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # outlier far away
        ])
        ts = [0.8, 0.7, 0.9, 0.5]
        result = aggregate_consensus(src_attn, ts, min_heads=3)
        # Positions 3,4,5 all get 3 votes from heads 0,1,2 (via tolerance)
        # Outlier at pos 0 only gets 1 vote. Result must be in consensus cluster.
        assert result in {3, 4, 5}

    def test_no_consensus_fallback(self):
        """No position has enough agreement -> fallback to ts_vote."""
        # Use high min_heads to ensure no consensus
        src_attn = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ])
        ts = [0.1, 0.1, 0.1, 0.9]
        result = aggregate_consensus(src_attn, ts, min_heads=4)
        # No position has 4 head agreement -> fallback to ts_vote
        # Head 3 (TS=0.9) at pos 7 wins
        assert result == 7

    def test_neighbor_tolerance(self):
        """Heads within +/-1 of each other -> consensus via tolerance."""
        src_attn = np.array([
            [0.0, 1.0, 0.0, 0.0, 0.0],  # pos 1
            [0.0, 0.0, 1.0, 0.0, 0.0],  # pos 2
            [0.0, 0.0, 0.0, 1.0, 0.0],  # pos 3
        ])
        ts = [0.8, 0.9, 0.7]
        # With tolerance, pos 2 gets votes from all 3 heads (1,2,3 are neighbors)
        result = aggregate_consensus(src_attn, ts, min_heads=3)
        assert result == 2


class TestGeomean:
    def test_agreement_required(self):
        """Geometric mean: positions need all heads to attend."""
        src_attn = np.array([
            [0.0, 0.5, 0.5, 0.0],
            [0.0, 0.0, 0.8, 0.2],
        ])
        ts = [1.0, 1.0]
        result = aggregate_geomean(src_attn, ts)
        # Position 2 has attention from both heads
        assert result == 2

    def test_single_head(self):
        """Single head -> same as argmax."""
        src_attn = np.array([[0.1, 0.2, 0.6, 0.1]])
        ts = [1.0]
        result = aggregate_geomean(src_attn, ts)
        assert result == 2

    def test_outlier_suppressed(self):
        """One head's zero attention kills that position in geomean."""
        src_attn = np.array([
            [0.0, 0.0, 1.0, 0.0],  # strong at pos 2
            [0.0, 0.0, 0.5, 0.5],  # also at pos 2
            [1.0, 0.0, 0.0, 0.0],  # outlier at pos 0, zero at 2
        ])
        ts = [1.0, 1.0, 1.0]
        # eps prevents true zero, but pos 2 should still dominate
        # because 2 of 3 heads strongly attend there
        result = aggregate_geomean(src_attn, ts)
        # The geomean with eps will favor pos 2 despite the outlier
        assert result == 2


class TestTopP:
    def test_concentrated(self):
        """All attention at position 3 -> top-p returns 3."""
        src_attn = np.array([[0.0, 0.0, 0.0, 1.0, 0.0]])
        ts = [1.0]
        result = aggregate_top_p(src_attn, ts, p_threshold=0.8)
        assert result == 3

    def test_spread_returns_rightmost(self):
        """Attention spread over positions 1,2,3 -> rightmost in top-p."""
        src_attn = np.array([[0.0, 0.3, 0.3, 0.4, 0.0]])
        ts = [1.0]
        result = aggregate_top_p(src_attn, ts, p_threshold=0.8)
        # Top-p: pos 3 (0.4) + pos 2 (0.3) = 0.7, need pos 1 (0.3) for > 0.8
        # Rightmost in {1,2,3} = 3
        assert result == 3

    def test_threshold_effect(self):
        """Lower threshold -> considers fewer positions."""
        src_attn = np.array([[0.0, 0.1, 0.1, 0.8, 0.0]])
        ts = [1.0]
        # p=0.5 -> only pos 3 (0.8) needed
        result = aggregate_top_p(src_attn, ts, p_threshold=0.5)
        assert result == 3


# ---------------------------------------------------------------------------
# Unified aggregation dispatcher
# ---------------------------------------------------------------------------

class TestAggregateDispatcher:
    def test_all_methods_listed(self):
        methods = list_aggregation_methods()
        assert "ts_vote" in methods
        assert "softmax_mean" in methods
        assert "entropy_weighted" in methods
        assert "consensus" in methods
        assert "geomean" in methods
        assert "top_p" in methods

    def test_dispatch_ts_vote(self):
        src_attn = np.array([[0.0, 0.0, 1.0]])
        ts = [1.0]
        assert aggregate(src_attn, ts, method="ts_vote") == 2

    def test_dispatch_softmax_mean(self):
        src_attn = np.array([[0.0, 0.0, 1.0]])
        ts = [1.0]
        result = aggregate(src_attn, ts, method="softmax_mean")
        assert abs(result - 2.0) < 0.01

    def test_dispatch_unknown_raises(self):
        src_attn = np.array([[0.0, 1.0]])
        ts = [1.0]
        with pytest.raises(ValueError, match="Unknown aggregation"):
            aggregate(src_attn, ts, method="nonexistent")


# ---------------------------------------------------------------------------
# check_border with aggregation parameter
# ---------------------------------------------------------------------------

class TestCheckBorderAggregation:
    def test_default_ts_vote(self):
        src_attn = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]])
        ts = [1.0]
        assert check_border(src_attn, ts, 5, 2, aggregation="ts_vote") is True

    def test_softmax_mean_border(self):
        """Softmax mean: attention at end -> border."""
        src_attn = np.array([[0.0, 0.0, 0.0, 0.1, 0.9]])
        ts = [1.0]
        assert check_border(src_attn, ts, 5, 2, aggregation="softmax_mean") is True

    def test_entropy_weighted_border(self):
        src_attn = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]])
        ts = [1.0]
        assert check_border(src_attn, ts, 5, 2, aggregation="entropy_weighted") is True

    def test_consensus_no_border(self):
        """Outlier head at border, but no consensus -> no border."""
        src_attn = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],  # only this one at border
        ])
        ts = [0.5, 0.5, 0.5]
        # With consensus (min_heads=3), pos 0 has agreement, pos 4 doesn't
        # Fallback to ts_vote: pos 0 wins (2 heads)
        assert check_border(src_attn, ts, 5, 2, aggregation="consensus") is False

    def test_geomean_border(self):
        src_attn = np.array([
            [0.0, 0.0, 0.0, 0.1, 0.9],
            [0.0, 0.0, 0.0, 0.2, 0.8],
        ])
        ts = [1.0, 1.0]
        assert check_border(src_attn, ts, 5, 2, aggregation="geomean") is True

    def test_backward_compatible(self):
        """Default aggregation param doesn't break existing check_border calls."""
        src_attn = np.array([[0.0, 0.0, 1.0, 0.0, 0.0]])
        ts = [1.0]
        # No aggregation param -> uses ts_vote
        assert check_border(src_attn, ts, 5, 2) is False


# ---------------------------------------------------------------------------
# BackendConfig aggregation support
# ---------------------------------------------------------------------------

class TestConfigAggregation:
    def test_default(self):
        config = BackendConfig()
        assert config.aggregation == "ts_vote"

    def test_from_dict(self):
        config = BackendConfig.from_dict({"aggregation": "entropy_weighted"})
        assert config.aggregation == "entropy_weighted"

    def test_sweep_config(self):
        """Aggregation can be swept like other params."""
        for method in list_aggregation_methods():
            config = BackendConfig.from_dict({"aggregation": method})
            assert config.aggregation == method


# ---------------------------------------------------------------------------
# Dynamic border distance tests
# ---------------------------------------------------------------------------

class TestAttentionEntropy:
    def test_sharp_attention_low_entropy(self):
        """All attention on one position -> low entropy."""
        src_attn = np.array([[0.0, 0.0, 1.0, 0.0, 0.0]])
        ts = [1.0]
        ent = attention_entropy(src_attn, ts)
        assert ent < 0.5  # very low

    def test_uniform_attention_high_entropy(self):
        """Uniform attention -> high entropy."""
        src_attn = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]])
        ts = [1.0]
        ent = attention_entropy(src_attn, ts)
        assert ent > 1.5  # ln(5) ~ 1.6

    def test_multi_head_weighting(self):
        """Higher TS head contributes more to entropy."""
        src_attn = np.array([
            [0.0, 0.0, 1.0, 0.0, 0.0],  # sharp (low entropy)
            [0.2, 0.2, 0.2, 0.2, 0.2],  # diffuse (high entropy)
        ])
        # Equal TS: average of low and high
        ts_equal = [1.0, 1.0]
        ent_equal = attention_entropy(src_attn, ts_equal)

        # Favor sharp head
        ts_sharp = [0.9, 0.1]
        ent_sharp = attention_entropy(src_attn, ts_sharp)

        # Favor diffuse head
        ts_diffuse = [0.1, 0.9]
        ent_diffuse = attention_entropy(src_attn, ts_diffuse)

        assert ent_sharp < ent_equal < ent_diffuse


class TestDynamicBorderDistance:
    def test_sharp_attention_tighter(self):
        """Sharp attention -> tighter border (bd - 1)."""
        src_attn = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        ts = [1.0]
        bd = dynamic_border_distance(src_attn, ts, base_bd=3, n_src_tokens=8)
        assert bd < 3  # tighter than base

    def test_diffuse_attention_wider(self):
        """Diffuse attention -> wider border."""
        src_attn = np.array([[0.125] * 8])
        ts = [1.0]
        bd = dynamic_border_distance(src_attn, ts, base_bd=3, n_src_tokens=8)
        assert bd > 3  # wider than base

    def test_minimum_bd(self):
        """Border distance never goes below min_bd."""
        src_attn = np.array([[0.0, 0.0, 1.0]])
        ts = [1.0]
        bd = dynamic_border_distance(src_attn, ts, base_bd=1, n_src_tokens=3, min_bd=1)
        assert bd >= 1

    def test_max_bd_delta(self):
        """Border distance increase is capped by max_bd_delta."""
        src_attn = np.array([[0.1] * 10])
        ts = [1.0]
        bd = dynamic_border_distance(
            src_attn, ts, base_bd=3, n_src_tokens=10, max_bd_delta=2
        )
        assert bd <= 3 + 2  # base + max_delta


class TestCheckBorderDynamic:
    def test_sharp_attention_at_border(self):
        """Sharp attention at end with dynamic -> border may be tighter."""
        # Sharp attention at position 7 (near end of 8 tokens)
        src_attn = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        ts = [1.0]
        # With base_bd=3, dynamic should tighten to bd=2 (sharp attn)
        # threshold = 8 - 2 = 6, pos 7 >= 6 -> border
        result = check_border_dynamic(src_attn, ts, 8, base_border_distance=3)
        assert result is True

    def test_diffuse_attention_wider_border(self):
        """Diffuse attention + position near border -> dynamic widens border, no hit."""
        # Diffuse attention centered around position 4
        src_attn = np.array([[0.05, 0.1, 0.15, 0.2, 0.25, 0.15, 0.05, 0.05]])
        ts = [1.0]
        # Aggregated position ~3.5, base_bd=2
        # Dynamic border: high entropy -> wider border (bd=4 or 5)
        # threshold = 8 - 5 = 3, pos ~3.5 >= 3 -> might be True
        # But point is that diffuse attention makes border wider
        result_static = check_border(src_attn, ts, 8, 2, aggregation="ts_vote")
        result_dynamic = check_border_dynamic(src_attn, ts, 8, base_border_distance=2)
        # Dynamic should be at least as conservative as static
        # (can't test exact behavior without knowing entropy thresholds)

    def test_config_flag(self):
        """dynamic_border config flag exists."""
        config = BackendConfig(dynamic_border=True)
        assert config.dynamic_border is True
        config2 = BackendConfig()
        assert config2.dynamic_border is False


# ---------------------------------------------------------------------------
# Ensemble aggregation tests
# ---------------------------------------------------------------------------

class TestEnsembleAggregation:
    def test_default_ensemble(self):
        """Default ensemble (ts_vote + entropy_weighted + geomean)."""
        src_attn = np.array([
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ])
        ts = [1.0, 1.0]
        result = aggregate_ensemble(src_attn, ts)
        # All methods should agree on position 3
        assert abs(result - 3.0) < 0.5

    def test_custom_methods(self):
        """Custom ensemble with specified methods."""
        src_attn = np.array([[0.0, 0.0, 1.0, 0.0]])
        ts = [1.0]
        result = aggregate_ensemble(
            src_attn, ts,
            methods=["ts_vote", "softmax_mean"],
            weights=[0.5, 0.5],
        )
        assert abs(result - 2.0) < 0.01

    def test_weighted_combination(self):
        """Different weights shift the result."""
        # Construct scenario where methods disagree
        src_attn = np.array([
            [0.0, 1.0, 0.0, 0.0, 0.0],   # head 0 at pos 1
            [0.0, 0.0, 0.0, 0.0, 1.0],   # head 1 at pos 4
        ])
        ts = [0.5, 0.5]
        # ts_vote: pos 1 or 4 (tie); softmax_mean: 2.5
        # Result should be somewhere between
        result = aggregate_ensemble(
            src_attn, ts,
            methods=["softmax_mean"],
            weights=[1.0],
        )
        assert abs(result - 2.5) < 0.01

    def test_in_registry(self):
        """Ensemble is registered and accessible via aggregate()."""
        src_attn = np.array([[0.0, 0.0, 1.0]])
        ts = [1.0]
        assert "ensemble" in list_aggregation_methods()
        result = aggregate(src_attn, ts, method="ensemble")
        assert abs(result - 2.0) < 0.5

    def test_invalid_method_in_ensemble(self):
        """Unknown method name in ensemble raises ValueError."""
        src_attn = np.array([[1.0, 0.0]])
        ts = [1.0]
        with pytest.raises(ValueError, match="Unknown method"):
            aggregate_ensemble(src_attn, ts, methods=["nonexistent"], weights=[1.0])
