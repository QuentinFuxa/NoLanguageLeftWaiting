"""Tests for compute-aware latency metrics."""

import pytest
from nllw.research import (
    compute_ca_delays,
    compute_ca_metrics,
    CAMetrics,
    BenchmarkResult,
)


class TestCADelays:
    def test_basic_conversion(self):
        """CU word delays + inference time -> CA time delays."""
        delays_words = [1, 2, 3]
        step_times_ms = [100, 200, 300]
        wps = 2.5  # 1 word = 0.4s

        ca_delays = compute_ca_delays(delays_words, step_times_ms, wps)
        assert len(ca_delays) == 3
        # First: 1/2.5 + 0.1 = 0.5
        assert ca_delays[0] == pytest.approx(0.5, abs=0.01)
        # Second: 2/2.5 + 0.2 = 1.0
        assert ca_delays[1] == pytest.approx(1.0, abs=0.01)

    def test_empty(self):
        assert compute_ca_delays([], [], 2.5) == []

    def test_zero_inference(self):
        """Zero inference time -> CA = CU in time domain."""
        delays = [1, 2, 3]
        times = [0, 0, 0]
        ca = compute_ca_delays(delays, times, 2.5)
        assert ca[0] == pytest.approx(0.4, abs=0.01)


class TestCAMetrics:
    def test_basic(self):
        delays = [1, 2, 3, 4]
        times = [50, 100, 150, 200]
        m = compute_ca_metrics(delays, times, source_length=4, target_length=4)
        assert m.al_cu > 0
        assert m.yaal_cu > 0
        assert m.al_ca > 0
        assert m.yaal_ca > 0
        assert m.first_token_ms == 50
        assert m.n_tokens_generated == 4
        assert m.tokens_per_second > 0

    def test_ca_reflects_inference_time(self):
        """CA metrics should incorporate inference overhead."""
        # With zero inference time, CA = CU (in time domain)
        delays = [2, 3, 4, 5]
        m_zero = compute_ca_metrics(delays, [0, 0, 0, 0], source_length=5, target_length=4)
        # With nonzero inference time, CA delays are larger
        m_slow = compute_ca_metrics(delays, [500, 1000, 1500, 2000], source_length=5, target_length=4)
        # Slower inference -> higher CA latency
        assert m_slow.yaal_ca > m_zero.yaal_ca


class TestBenchmarkResult:
    def test_summary(self):
        r = BenchmarkResult(
            name="test-config",
            direction="en-zh",
            bleu=42.5,
            comet=0.85,
            avg_yaal_cu=3.2,
            avg_yaal_ca=3.8,
            avg_tokens_per_second=25.0,
            avg_first_token_ms=120,
        )
        s = r.summary()
        assert "test-config" in s
        assert "42.5" in s
        assert "3.2" in s or "3.20" in s
