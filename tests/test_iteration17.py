"""Tests for iteration 17: top_p_threshold parameter, updated configs, XCOMET-XL fix."""

import numpy as np
import pytest

from nllw.backend_protocol import BackendConfig
from nllw.alignatt import (
    aggregate_top_p,
    aggregate,
    check_border,
    check_border_dynamic,
    check_border_combined,
)
from nllw.simulstream import SimulStreamConfig, DIRECTION_DEFAULTS


# ===========================================================================
# top_p_threshold parameter tests
# ===========================================================================

class TestTopPThreshold:
    """Tests for the tunable top_p_threshold parameter."""

    def test_default_threshold(self):
        """Default top_p_threshold should be 0.8."""
        cfg = BackendConfig()
        assert cfg.top_p_threshold == 0.8

    def test_threshold_in_config_dict(self):
        """top_p_threshold should be loadable from dict."""
        cfg = BackendConfig.from_dict({"top_p_threshold": 0.6})
        assert cfg.top_p_threshold == 0.6

    def test_lower_threshold_closer_to_argmax(self):
        """Lower p_threshold should return position closer to argmax."""
        # One strong head + some noise
        src_attn = np.array([
            [0.1, 0.6, 0.2, 0.1],  # Head 0: attends to pos 1
            [0.05, 0.1, 0.7, 0.15],  # Head 1: attends to pos 2
        ])
        ts_scores = [0.6, 0.4]

        # Low threshold: only top few positions needed
        pos_low = aggregate_top_p(src_attn, ts_scores, p_threshold=0.5)
        # High threshold: more positions needed, rightmost goes further
        pos_high = aggregate_top_p(src_attn, ts_scores, p_threshold=0.95)

        # Higher threshold includes more positions -> rightmost goes further
        assert pos_high >= pos_low

    def test_threshold_1_includes_all(self):
        """p_threshold=1.0 should include all positions."""
        src_attn = np.array([
            [0.1, 0.6, 0.2, 0.1],
        ])
        ts_scores = [1.0]

        pos = aggregate_top_p(src_attn, ts_scores, p_threshold=1.0)
        # Should be the rightmost position (3)
        assert pos == 3

    def test_threshold_forwarded_via_aggregate(self):
        """aggregate() should forward p_threshold to top_p method."""
        src_attn = np.array([
            [0.1, 0.1, 0.6, 0.2],
        ])
        ts_scores = [1.0]

        pos_default = aggregate(src_attn, ts_scores, method="top_p")
        pos_low = aggregate(src_attn, ts_scores, method="top_p", p_threshold=0.5)

        # Low threshold = smaller set = closer to argmax
        assert pos_low <= pos_default

    def test_threshold_ignored_for_non_topp(self):
        """p_threshold kwarg should be silently ignored for non-top_p methods."""
        src_attn = np.array([
            [0.1, 0.6, 0.2, 0.1],
        ])
        ts_scores = [1.0]

        # Should not raise
        pos = aggregate(src_attn, ts_scores, method="ts_vote", p_threshold=0.5)
        assert isinstance(pos, (int, float))

    def test_check_border_with_threshold(self):
        """check_border should accept and use top_p_threshold."""
        src_attn = np.array([
            [0.1, 0.1, 0.1, 0.7],  # Attends to last position
        ])
        ts_scores = [1.0]

        # With top_p, rightmost in top-p set
        hit = check_border(
            src_attn, ts_scores, n_src_tokens=4, border_distance=1,
            aggregation="top_p", top_p_threshold=0.8,
        )
        assert hit  # Position 3 >= 4-1 = 3

    def test_check_border_dynamic_with_threshold(self):
        """check_border_dynamic should accept top_p_threshold."""
        src_attn = np.array([
            [0.1, 0.1, 0.1, 0.7],
        ])
        ts_scores = [1.0]

        # Should not raise
        result = check_border_dynamic(
            src_attn, ts_scores, n_src_tokens=4,
            base_border_distance=1, aggregation="top_p",
            top_p_threshold=0.6,
        )
        assert isinstance(result, bool)

    def test_check_border_combined_with_threshold(self):
        """check_border_combined should accept top_p_threshold."""
        src_attn = np.array([
            [0.1, 0.1, 0.1, 0.7],
        ])
        ts_scores = [1.0]

        hit, _, _ = check_border_combined(
            src_attn, ts_scores, n_src_tokens=4,
            border_distance=1, aggregation="top_p",
            top_p_threshold=0.8,
        )
        assert isinstance(hit, bool)


# ===========================================================================
# SimulStreamConfig aggregation field tests
# ===========================================================================

class TestSimulStreamAggregation:
    """Tests for the aggregation field in SimulStreamConfig."""

    def test_default_aggregation(self):
        """Default aggregation should be top_p."""
        cfg = SimulStreamConfig()
        assert cfg.aggregation == "top_p"

    def test_default_word_batch(self):
        """Default word_batch should be 4 (updated from 2)."""
        cfg = SimulStreamConfig()
        assert cfg.word_batch == 4

    def test_default_repetition_halt(self):
        """Default repetition_max_repeats should be None (disabled, hurts EN-ZH)."""
        cfg = SimulStreamConfig()
        assert cfg.repetition_max_repeats is None

    def test_to_backend_config_includes_aggregation(self):
        """to_backend_config should include aggregation field."""
        cfg = SimulStreamConfig(aggregation="top_p")
        bc = cfg.to_backend_config()
        assert bc.aggregation == "top_p"

    def test_to_backend_config_includes_repetition(self):
        """to_backend_config should include repetition_max_repeats."""
        cfg = SimulStreamConfig(repetition_max_repeats=2)
        bc = cfg.to_backend_config()
        assert bc.repetition_max_repeats == 2

    def test_to_backend_config_no_repetition_when_none(self):
        """When repetition_max_repeats is None, it should stay None in BackendConfig."""
        cfg = SimulStreamConfig(repetition_max_repeats=None)
        bc = cfg.to_backend_config()
        assert bc.repetition_max_repeats is None

    def test_direction_defaults_include_aggregation(self):
        """All DIRECTION_DEFAULTS should include aggregation."""
        for direction, cfg in DIRECTION_DEFAULTS.items():
            assert "aggregation" in cfg, f"Missing aggregation for {direction}"
            assert cfg["aggregation"] == "top_p"


# ===========================================================================
# Updated DIRECTION_DEFAULTS validation
# ===========================================================================

class TestUpdatedDirectionDefaults:
    """Validate iteration 16 optimal configs in DIRECTION_DEFAULTS."""

    def test_en_zh_optimal(self):
        """EN-ZH: bd=3, wb=4, top_p -> COMET=0.895."""
        cfg = DIRECTION_DEFAULTS["en-zh"]
        assert cfg["border_distance"] == 3
        assert cfg["word_batch"] == 4
        assert cfg["aggregation"] == "top_p"

    def test_en_de_optimal(self):
        """EN-DE: bd=2, wb=3, top_p -> COMET=0.881."""
        cfg = DIRECTION_DEFAULTS["en-de"]
        assert cfg["border_distance"] == 2
        assert cfg["word_batch"] == 3
        assert cfg["aggregation"] == "top_p"

    def test_en_it_optimal(self):
        """EN-IT: bd=2, wb=3, top_p -> COMET=0.884."""
        cfg = DIRECTION_DEFAULTS["en-it"]
        assert cfg["border_distance"] == 2
        assert cfg["word_batch"] == 3
        assert cfg["aggregation"] == "top_p"

    def test_cs_en_optimal(self):
        """CS-EN: bd=3, wb=3, top_p -> COMET=0.876."""
        cfg = DIRECTION_DEFAULTS["cs-en"]
        assert cfg["border_distance"] == 3
        assert cfg["word_batch"] == 3
        assert cfg["aggregation"] == "top_p"


# ===========================================================================
# top_p sensitivity to threshold (research validation)
# ===========================================================================

class TestTopPBehavior:
    """Validate that top_p aggregation behaves as expected theoretically."""

    def test_uniform_attention(self):
        """With uniform attention, top_p should return rightmost position in set."""
        # Uniform attention = all positions equal
        src_attn = np.array([[0.25, 0.25, 0.25, 0.25]])
        ts_scores = [1.0]

        # At threshold 0.75, need 3 positions -> rightmost of any 3
        pos = aggregate_top_p(src_attn, ts_scores, p_threshold=0.75)
        # All weights equal, so sorted order may vary, but rightmost of top-3 is 3
        assert pos >= 2  # At least 3 positions needed

    def test_peaked_attention(self):
        """With peaked attention on one position, top_p should return that position."""
        src_attn = np.array([[0.01, 0.01, 0.97, 0.01]])
        ts_scores = [1.0]

        # At threshold 0.8, only need the peaked position
        pos = aggregate_top_p(src_attn, ts_scores, p_threshold=0.8)
        assert pos == 2  # The peaked position

    def test_split_attention_frontier(self):
        """With split attention, top_p correctly finds the frontier."""
        # Attention split between pos 1 (0.4) and pos 5 (0.3)
        src_attn = np.array([
            [0.05, 0.4, 0.05, 0.1, 0.1, 0.3],
        ])
        ts_scores = [1.0]

        # At p=0.8, need pos 1 (0.4) + pos 5 (0.3) + something
        pos = aggregate_top_p(src_attn, ts_scores, p_threshold=0.8)
        # Rightmost in the set should be 5 (or close)
        assert pos >= 4

    def test_multiple_heads_merge(self):
        """Multiple heads should be TS-weighted before top_p."""
        src_attn = np.array([
            [0.8, 0.1, 0.1],  # Head 0: attends to pos 0
            [0.1, 0.1, 0.8],  # Head 1: attends to pos 2
        ])
        ts_scores = [0.9, 0.1]  # Head 0 dominates

        pos = aggregate_top_p(src_attn, ts_scores, p_threshold=0.8)
        # Merged dist is 0.9*[0.8,0.1,0.1] + 0.1*[0.1,0.1,0.8] = [0.73,0.1,0.17]
        # Top-p at 0.8: include pos 0 (0.73) -> need more -> include pos 2 (0.17) -> 0.90
        # Rightmost = 2
        assert pos == 2
