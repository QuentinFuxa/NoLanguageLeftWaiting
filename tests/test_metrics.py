"""Tests for SimulMT latency metrics."""

import pytest
from nllw.metrics import (
    compute_al,
    compute_laal,
    compute_yaal,
    compute_longyaal,
    compute_longyaal_ms,
    compute_stream_laal,
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


class TestLongYAAL:
    """Test LongYAAL -- IWSLT 2026 primary latency metric."""

    def test_longyaal_equals_yaal_longform(self):
        """LongYAAL is identical to YAAL with is_longform=True."""
        delays = [1, 2, 3, 4]
        longyaal = compute_longyaal(delays, source_length=4, target_length=4)
        yaal = compute_yaal(delays, source_length=4, target_length=4, is_longform=True)
        assert longyaal == pytest.approx(yaal)

    def test_longyaal_counts_beyond_source(self):
        """LongYAAL counts all delays, even past source boundary."""
        delays = [1, 2, 10, 10]
        longyaal = compute_longyaal(delays, source_length=3, target_length=4)
        yaal_short = compute_yaal(delays, source_length=3, target_length=4, is_longform=False)
        # LongYAAL includes the large delays, so it should be larger
        assert longyaal > yaal_short

    def test_longyaal_empty(self):
        assert compute_longyaal([], 5, 5) == 0.0
        assert compute_longyaal([1], 0, 5) == 0.0

    def test_longyaal_basic_value(self):
        """Known value: read-one-write-one policy."""
        # gamma = max(4, 4) / 4 = 1
        # longyaal = [(1-0) + (2-1) + (3-2) + (4-3)] / 4 = 1.0
        delays = [1, 2, 3, 4]
        assert compute_longyaal(delays, 4, 4) == pytest.approx(1.0, abs=0.01)


class TestLongYAALMs:
    """Test time-domain LongYAAL (milliseconds)."""

    def test_ms_basic(self):
        """LongYAAL in ms with known values."""
        # 4 words emitted at 500ms, 1000ms, 1500ms, 2000ms
        # Source is 2000ms total, target is 4 words
        # gamma = max(4, 4) / 2000 = 0.002
        # longyaal_ms = [(500-0) + (1000-500) + (1500-1000) + (2000-1500)] / 4
        delays_ms = [500.0, 1000.0, 1500.0, 2000.0]
        longyaal = compute_longyaal_ms(delays_ms, source_length_ms=2000.0, target_length=4)
        assert longyaal > 0
        # Should be about 500ms (each word arrives 500ms after ideal uniform)
        assert longyaal == pytest.approx(500.0, abs=50.0)

    def test_ms_empty(self):
        assert compute_longyaal_ms([], 2000.0, 4) == 0.0
        assert compute_longyaal_ms([500.0], 0.0, 4) == 0.0

    def test_ms_simultaneous(self):
        """Near-zero latency when words arrive immediately."""
        # If words arrive at 0, 250, 500, 750 with source 2000ms
        delays_ms = [0.0, 250.0, 500.0, 750.0]
        longyaal = compute_longyaal_ms(delays_ms, source_length_ms=2000.0, target_length=4)
        # Should be very low (ahead of uniform schedule)
        assert longyaal < 500.0


class TestStreamLAAL:
    """Test StreamLAAL -- IWSLT 2026 secondary latency metric."""

    def test_basic(self):
        """StreamLAAL with monotonic delays."""
        delays = [1, 2, 3, 4]
        stream_laal = compute_stream_laal(delays, source_length=5, target_length=4)
        assert stream_laal > 0

    def test_empty(self):
        assert compute_stream_laal([], 5, 5) == 0.0
        assert compute_stream_laal([1], 0, 5) == 0.0

    def test_monotonizes_delays(self):
        """StreamLAAL should monotonize delays (non-monotonic -> monotonic)."""
        # Non-monotonic: word 2 has lower delay than word 1
        delays_non_mono = [3, 1, 4, 5]
        delays_mono = [1, 2, 3, 4]
        # StreamLAAL of non-monotonic should be >= monotonic version
        sl1 = compute_stream_laal(delays_non_mono, 5, 4)
        sl2 = compute_stream_laal(delays_mono, 5, 4)
        assert sl1 >= sl2 - 0.01

    def test_stops_at_source(self):
        """StreamLAAL stops counting at source boundary."""
        delays = [1, 2, 10, 10]
        stream_laal = compute_stream_laal(delays, source_length=3, target_length=4)
        # Only counts delays < 3 (after monotonization)
        longyaal = compute_longyaal(delays, source_length=3, target_length=4)
        # StreamLAAL should be less than LongYAAL since it stops at boundary
        assert stream_laal < longyaal


class TestComputeAllMetricsNewFields:
    """Test that compute_all_metrics includes new fields."""

    def test_includes_longyaal(self):
        delays = [2, 3, 4, 5]
        m = compute_all_metrics(delays, source_length=5, target_length=4)
        assert m.longyaal > 0
        assert m.stream_laal > 0
        assert m.longyaal_ms == 0.0  # No ms delays provided

    def test_includes_longyaal_ms(self):
        delays = [2, 3, 4, 5]
        delays_ms = [800.0, 1200.0, 1600.0, 2000.0]
        m = compute_all_metrics(
            delays, source_length=5, target_length=4,
            delays_ms=delays_ms, source_length_ms=2500.0,
        )
        assert m.longyaal_ms > 0


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


class TestBLEUTokenization:
    """Test language-aware BLEU tokenization."""

    def test_bleu_tokenize_chinese(self):
        from nllw.metrics import _bleu_tokenize
        assert _bleu_tokenize("zh") == "zh"
        assert _bleu_tokenize("ja") == "zh"

    def test_bleu_tokenize_default(self):
        from nllw.metrics import _bleu_tokenize
        assert _bleu_tokenize("en") == "13a"
        assert _bleu_tokenize("de") == "13a"
        assert _bleu_tokenize(None) == "13a"

    def test_chinese_bleu_nonzero(self):
        from nllw.metrics import compute_bleu
        # Chinese text that should have overlap
        hyp = "我们 现在 已经 拥有 了 小鼠"
        ref = "我们现在有小鼠"
        score = compute_bleu(hyp, ref, target_lang="zh")
        assert score > 0, "Chinese BLEU should be > 0 for similar texts"

    def test_chinese_corpus_bleu_nonzero(self):
        from nllw.metrics import compute_bleu_corpus
        hyps = ["我们现在已经拥有了小鼠"]
        refs = ["我们现在有小鼠"]
        score = compute_bleu_corpus(hyps, refs, target_lang="zh")
        assert score > 0, "Chinese corpus BLEU should be > 0"
