"""Tests for SimulMT latency metrics."""

import pytest
from nllw.metrics import (
    compute_al,
    compute_laal,
    compute_yaal,
    compute_ap,
    compute_dal,
    compute_max_consecutive_wait,
    compute_all_metrics,
)


class TestAverageLagging:
    """Test Average Lagging (AL)."""

    def test_simultaneous_one_lag(self):
        """Read-one-write-one: delay[t] = t+1 -> AL = 1 (minimum non-trivial lag)."""
        # 4 source words, 4 target words, gamma = 1
        # AL = (1/4) * [(1-0) + (2-1) + (3-2) + (4-3)] = 1.0
        delays = [1, 2, 3, 4]
        al = compute_al(delays, source_length=4, target_length=4)
        assert al == pytest.approx(1.0, abs=0.01)

    def test_wait_all(self):
        """Wait-all policy: read everything then write."""
        delays = [5, 5, 5, 5, 5]
        al = compute_al(delays, source_length=5, target_length=5)
        # AL should be positive (lagging)
        assert al > 0

    def test_empty(self):
        assert compute_al([], 5, 5) == 0.0
        assert compute_al([1, 2], 0, 5) == 0.0

    def test_known_value(self):
        """Known example: 5 source, 4 target, delays = [2, 3, 4, 5]."""
        delays = [2, 3, 4, 5]
        al = compute_al(delays, source_length=5, target_length=4)
        # gamma = 5/4 = 1.25
        # AL = (1/4) * [(2-0) + (3-1.25) + (4-2.5) + (5-3.75)]
        #    = (1/4) * [2 + 1.75 + 1.5 + 1.25] = 6.5/4 = 1.625
        assert al == pytest.approx(1.625, abs=0.01)


class TestYAAL:
    """Test YAAL (OmniSTEval formula)."""

    def test_basic(self):
        """Basic YAAL computation."""
        delays = [1, 2, 3, 4]
        yaal = compute_yaal(delays, source_length=4, target_length=4)
        # gamma = max(4, 4) / 4 = 1
        # yaal = sum(d - t/1 for t, d in enumerate(delays)) / 4
        #      = [(1-0) + (2-1) + (3-2) + (4-3)] / 4 = 4/4 = 1.0
        assert yaal == pytest.approx(1.0, abs=0.01)

    def test_shortform_stops_at_source(self):
        """Shortform YAAL stops counting at source boundary."""
        delays = [1, 2, 10, 10]  # last two are beyond source
        yaal = compute_yaal(delays, source_length=3, target_length=4, is_longform=False)
        # Only counts d < 3: delays[0]=1, delays[1]=2
        # gamma = max(4, 4) / 3 = 4/3
        # yaal = [(1 - 0/1.333) + (2 - 1/1.333)] / 2
        assert yaal > 0

    def test_longform_counts_all(self):
        """Longform YAAL counts all delays."""
        delays = [1, 2, 10, 10]
        yaal = compute_yaal(delays, source_length=3, target_length=4, is_longform=True)
        # Counts all 4 entries
        assert yaal > 0

    def test_empty(self):
        assert compute_yaal([], 5, 5) == 0.0


class TestAverageProportion:
    """Test AP metric."""

    def test_simultaneous(self):
        delays = [1, 2, 3, 4]
        ap = compute_ap(delays, source_length=4, target_length=4)
        # AP = (1+2+3+4) / (4*4) = 10/16 = 0.625
        assert ap == pytest.approx(0.625, abs=0.01)

    def test_wait_all(self):
        delays = [5, 5, 5]
        ap = compute_ap(delays, source_length=5, target_length=3)
        # AP = (5+5+5) / (5*3) = 15/15 = 1.0
        assert ap == pytest.approx(1.0, abs=0.01)


class TestDAL:
    """Test Differentiable AL."""

    def test_monotonic_delays(self):
        """DAL with already-monotonic delays: monotonization adds gamma increments."""
        delays = [2, 3, 4, 5]
        dal = compute_dal(delays, source_length=5, target_length=4)
        # gamma = 5/4 = 1.25
        # mono[0] = 2, mono[1] = max(3, 2+1.25) = 3.25, mono[2] = max(4, 3.25+1.25) = 4.5, ...
        # DAL >= AL due to monotonization
        al = compute_al(delays, source_length=5, target_length=4)
        assert dal >= al - 0.01

    def test_non_monotonic(self):
        """DAL monotonizes non-monotonic delays."""
        delays = [3, 2, 4, 5]  # delay[1]=2 < delay[0]=3, not monotonic
        dal = compute_dal(delays, source_length=5, target_length=4)
        al = compute_al(delays, source_length=5, target_length=4)
        # DAL >= AL for non-monotonic delays
        assert dal >= al - 0.01


class TestMaxConsecutiveWait:
    def test_uniform(self):
        delays = [1, 2, 3, 4]
        assert compute_max_consecutive_wait(delays) == 1

    def test_bursty(self):
        delays = [1, 5, 6, 7]  # big jump from 1 to 5
        assert compute_max_consecutive_wait(delays) == 4

    def test_single(self):
        assert compute_max_consecutive_wait([1]) == 0


class TestComputeAllMetrics:
    def test_returns_all_fields(self):
        delays = [2, 3, 4, 5]
        m = compute_all_metrics(delays, source_length=5, target_length=4)
        assert m.al > 0
        assert m.laal > 0
        assert m.yaal > 0
        assert m.ap > 0
        assert m.dal > 0
        assert m.n_source == 5
        assert m.n_target == 4
