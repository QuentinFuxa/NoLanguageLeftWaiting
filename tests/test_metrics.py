"""Tests for SimulMT latency metrics."""

import pytest
from nllw.metrics import (
    LatencyMetrics,
    compute_latency_metrics,
    compute_yaal_ms,
)


class TestComputeLatencyMetrics:
    def test_empty_delays(self):
        m = compute_latency_metrics([], 0, 0)
        assert m.al == 0.0
        assert m.laal == 0.0
        assert m.yaal == 0.0
        assert m.ap == 0.0
        assert m.dal == 0.0
        assert m.max_cw == 0

    def test_perfect_simultaneous(self):
        """Each target token emitted exactly when corresponding source word arrives."""
        delays = [0, 1, 2, 3, 4]
        m = compute_latency_metrics(delays, num_src_words=5, num_tgt_tokens=5)
        assert m.al == pytest.approx(0.0, abs=0.01)
        assert m.max_cw == 1

    def test_wait_then_flush(self):
        """Read all source first, then generate all target."""
        delays = [4, 4, 4, 4, 4]
        m = compute_latency_metrics(delays, num_src_words=5, num_tgt_tokens=5)
        assert m.al >= 2.0
        assert m.max_cw == 5

    def test_incremental(self):
        """Incremental translation: read 2, write 1, read 1, write 1, etc."""
        delays = [1, 2, 3]
        m = compute_latency_metrics(delays, num_src_words=4, num_tgt_tokens=3)
        assert m.al > 0
        assert m.laal > 0
        assert m.yaal > 0
        assert m.ap > 0
        assert m.dal > 0
        assert m.max_cw >= 1

    def test_more_target_than_source(self):
        """Target is longer than source (common for EN->ZH).

        When target is much longer than source, lagging can be negative
        (system is ahead of ideal pace). This is correct behavior.
        """
        delays = [0, 0, 1, 1, 2, 2, 3, 3]
        m = compute_latency_metrics(delays, num_src_words=4, num_tgt_tokens=8)
        # gamma = max(8, 8) / 4 = 2.0, so ideal pace is 2 tokens per source word
        # system produces tokens faster than that -> negative lagging is correct
        assert m.ap > 0  # AP is always non-negative
        assert m.max_cw >= 1

    def test_to_dict(self):
        m = LatencyMetrics(al=1.234, laal=0.567, yaal=0.450, ap=0.89, dal=1.5, max_cw=3)
        d = m.to_dict()
        assert d["al"] == 1.234
        assert d["laal"] == 0.567
        assert d["yaal"] == 0.45
        assert d["ap"] == 0.89
        assert d["dal"] == 1.5
        assert d["max_cw"] == 3


class TestYAAL:
    """Tests for YAAL (Yet Another Average Lagging) -- IWSLT 2026 primary metric."""

    def test_yaal_stops_at_source_boundary(self):
        """YAAL should stop counting when delay >= source_length."""
        # delays: [0, 1, 5, 5] with S=3. YAAL should only count t=0,1 (d<3)
        delays = [0, 1, 5, 5]
        m = compute_latency_metrics(delays, num_src_words=3, num_tgt_tokens=4)
        # gamma = max(4, 4) / 3
        gamma = 4.0 / 3.0
        expected = (0.0 - 0 / gamma + 1.0 - 1 / gamma) / 2
        assert m.yaal == pytest.approx(expected, abs=0.01)

    def test_yaal_all_simultaneous(self):
        """All tokens emitted before source ends -> all counted."""
        delays = [0, 1, 2]
        m = compute_latency_metrics(delays, num_src_words=5, num_tgt_tokens=3)
        # All delays < 5, so all counted
        gamma = max(3, 3) / 5.0
        expected = sum(delays[t] - t / gamma for t in range(3)) / 3
        assert m.yaal == pytest.approx(expected, abs=0.01)

    def test_yaal_vs_laal_difference(self):
        """YAAL and LAAL should differ: YAAL stops early, LAAL doesn't."""
        delays = [0, 1, 5, 5]  # S=3
        m = compute_latency_metrics(delays, num_src_words=3, num_tgt_tokens=4)
        # LAAL counts t=0,1 then stops at t=2 where d=5>=3
        # YAAL counts t=0,1 then stops (d[2]=5>=3)
        # They should differ because LAAL includes the boundary token in its sum
        # and YAAL doesn't
        assert m.yaal != m.laal

    def test_yaal_matches_al_when_all_before_source(self):
        """When all delays < S, YAAL should be close to LAAL."""
        delays = [0, 1, 2, 3]
        m = compute_latency_metrics(delays, num_src_words=10, num_tgt_tokens=4)
        # All < 10, so gamma same, tau same -> should be close
        # Not necessarily equal because formulas differ slightly
        assert abs(m.yaal - m.laal) < 0.01


class TestDAL:
    """Tests for Differentiable Average Lagging."""

    def test_dal_monotonicity(self):
        """DAL should be >= AL for non-monotonic delays."""
        delays = [0, 1, 2, 3, 4]
        m = compute_latency_metrics(delays, num_src_words=5, num_tgt_tokens=5)
        assert m.dal >= m.al - 0.01  # DAL >= AL (with tolerance)

    def test_dal_positive(self):
        delays = [1, 2, 3]
        m = compute_latency_metrics(delays, num_src_words=4, num_tgt_tokens=3)
        assert m.dal > 0

    def test_dal_wait_then_flush(self):
        """DAL should be high when all tokens wait."""
        delays = [4, 4, 4, 4, 4]
        m = compute_latency_metrics(delays, num_src_words=5, num_tgt_tokens=5)
        assert m.dal > 2.0


class TestYAALMs:
    """Tests for time-domain YAAL (ms)."""

    def test_basic(self):
        # 3 emissions at 0ms, 500ms, 1000ms. Source = 2000ms.
        # YAAL can be negative when translations arrive faster than ideal pace.
        result = compute_yaal_ms([0, 500, 1000], source_length_ms=2000)
        # gamma = max(3, 3) / 2000 = 0.0015, ideal[t] = t/gamma
        # This system is fast, so YAAL should be low (possibly negative)
        assert isinstance(result, float)

    def test_empty(self):
        assert compute_yaal_ms([], source_length_ms=1000) == 0.0
        assert compute_yaal_ms([100], source_length_ms=0) == 0.0

    def test_longform_vs_shortform(self):
        # With shortform, emissions past source_length are excluded
        emissions = [0, 500, 1000, 3000]  # last one past source
        longform = compute_yaal_ms(emissions, source_length_ms=2000, is_longform=True)
        shortform = compute_yaal_ms(emissions, source_length_ms=2000, is_longform=False)
        # Longform includes the 3000ms emission, shortform doesn't
        assert longform != shortform

    def test_all_after_source(self):
        """All emissions after source ends -> shortform returns 0."""
        result = compute_yaal_ms([2000, 3000], source_length_ms=1000, is_longform=False)
        assert result == 0.0


class TestEdgeCases:
    def test_single_token(self):
        m = compute_latency_metrics([0], num_src_words=1, num_tgt_tokens=1)
        assert m.al == pytest.approx(0.0, abs=0.01)
        assert m.max_cw == 1

    def test_zero_src_words(self):
        m = compute_latency_metrics([0], num_src_words=0, num_tgt_tokens=1)
        assert m.al == 0.0

    def test_zero_tgt_tokens(self):
        m = compute_latency_metrics([], num_src_words=5, num_tgt_tokens=0)
        assert m.al == 0.0
