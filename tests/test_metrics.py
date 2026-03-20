"""Tests for SimulMT latency metrics."""

import pytest
from nllw.metrics import LatencyMetrics, compute_latency_metrics


class TestComputeLatencyMetrics:
    def test_empty_delays(self):
        m = compute_latency_metrics([], 0, 0)
        assert m.al == 0.0
        assert m.laal == 0.0
        assert m.ap == 0.0
        assert m.max_cw == 0

    def test_perfect_simultaneous(self):
        """Each target token emitted exactly when corresponding source word arrives."""
        # 5 source words, 5 target tokens, delay[t] = t
        delays = [0, 1, 2, 3, 4]
        m = compute_latency_metrics(delays, num_src_words=5, num_tgt_tokens=5)
        # AL should be 1.0 (each token is 1 word behind "perfect")
        assert m.al == pytest.approx(1.0, abs=0.01)
        assert m.max_cw == 1

    def test_wait_then_flush(self):
        """Read all source first, then generate all target."""
        delays = [4, 4, 4, 4, 4]
        m = compute_latency_metrics(delays, num_src_words=5, num_tgt_tokens=5)
        # AL should be high (all tokens wait for all source)
        assert m.al > 2.0
        assert m.max_cw == 5

    def test_incremental(self):
        """Incremental translation: read 2, write 1, read 1, write 1, etc."""
        delays = [1, 2, 3]
        m = compute_latency_metrics(delays, num_src_words=4, num_tgt_tokens=3)
        assert m.al > 0
        assert m.laal > 0
        assert m.ap > 0
        assert m.max_cw >= 1

    def test_more_target_than_source(self):
        """Target is longer than source (common for EN->ZH)."""
        delays = [0, 0, 1, 1, 2, 2, 3, 3]
        m = compute_latency_metrics(delays, num_src_words=4, num_tgt_tokens=8)
        assert m.laal <= m.al  # tau < 1 when S/T < 1

    def test_to_dict(self):
        m = LatencyMetrics(al=1.234, laal=0.567, ap=0.89, max_cw=3)
        d = m.to_dict()
        assert d["al"] == 1.234
        assert d["laal"] == 0.567
        assert d["ap"] == 0.89
        assert d["max_cw"] == 3


class TestEdgeCases:
    def test_single_token(self):
        m = compute_latency_metrics([0], num_src_words=1, num_tgt_tokens=1)
        assert m.al == pytest.approx(1.0, abs=0.01)
        assert m.max_cw == 1

    def test_zero_src_words(self):
        m = compute_latency_metrics([0], num_src_words=0, num_tgt_tokens=1)
        assert m.al == 0.0

    def test_zero_tgt_tokens(self):
        m = compute_latency_metrics([], num_src_words=5, num_tgt_tokens=0)
        assert m.al == 0.0
