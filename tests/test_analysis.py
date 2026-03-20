"""Tests for analysis module."""

import pytest
from nllw.analysis import (
    ExperimentResult,
    ParetoAnalyzer,
    generate_comparison_table,
    find_edge_cases,
)


def _make_result(name, quality=0.8, yaal=3.0, bleu=40.0, direction="en-zh"):
    return ExperimentResult(
        name=name,
        config={"border_distance": 3, "word_batch": 2},
        bleu=bleu,
        comet=quality,
        avg_yaal=yaal,
        avg_al=yaal * 0.9,
        avg_ap=0.6,
        direction=direction,
        backend_type="alignatt",
    )


class TestExperimentResult:
    def test_from_eval_dict(self):
        d = {
            "backend_type": "alignatt",
            "bleu": 42.5,
            "comet": 0.85,
            "avg_yaal": 3.2,
            "avg_al": 2.8,
            "direction": "en-zh",
            "config": {"border_distance": 3},
        }
        r = ExperimentResult.from_eval_dict(d, name="test")
        assert r.name == "test"
        assert r.bleu == 42.5
        assert r.comet == 0.85
        assert r.direction == "en-zh"

    def test_quality_priority(self):
        """Quality prefers xcomet > comet > bleu."""
        r1 = ExperimentResult(name="a", config={}, xcomet=0.9, comet=0.8, bleu=40)
        assert r1.quality == 0.9

        r2 = ExperimentResult(name="b", config={}, comet=0.8, bleu=40)
        assert r2.quality == 0.8

        r3 = ExperimentResult(name="c", config={}, bleu=40)
        assert r3.quality == pytest.approx(0.4, abs=0.01)

    def test_config_str(self):
        r = ExperimentResult(
            name="test",
            config={"border_distance": 3, "word_batch": 2},
        )
        s = r.config_str()
        assert "bd=3" in s
        assert "wb=2" in s


class TestParetoAnalyzer:
    def test_pareto_frontier(self):
        """Identify Pareto-optimal points."""
        results = [
            _make_result("A", quality=0.9, yaal=5.0),  # high quality, high latency
            _make_result("B", quality=0.8, yaal=3.0),  # Pareto optimal
            _make_result("C", quality=0.7, yaal=2.0),  # Pareto optimal (lowest latency)
            _make_result("D", quality=0.6, yaal=4.0),  # Dominated by B
        ]
        analyzer = ParetoAnalyzer(results)
        frontier = analyzer.pareto_frontier()

        names = [r.name for r in frontier]
        assert "A" in names  # highest quality
        assert "B" in names  # good balance
        assert "C" in names  # lowest latency
        assert "D" not in names  # dominated

    def test_single_result(self):
        results = [_make_result("A")]
        analyzer = ParetoAnalyzer(results)
        frontier = analyzer.pareto_frontier()
        assert len(frontier) == 1

    def test_empty(self):
        analyzer = ParetoAnalyzer([])
        frontier = analyzer.pareto_frontier()
        assert len(frontier) == 0

    def test_best_by(self):
        results = [
            _make_result("A", quality=0.9, yaal=5.0),
            _make_result("B", quality=0.7, yaal=2.0),
        ]
        analyzer = ParetoAnalyzer(results)
        best_q = analyzer.best_by("quality", maximize=True)
        assert best_q.name == "A"
        best_l = analyzer.best_by("avg_yaal", maximize=False)
        assert best_l.name == "B"

    def test_filter_direction(self):
        results = [
            _make_result("A", direction="en-zh"),
            _make_result("B", direction="en-de"),
        ]
        analyzer = ParetoAnalyzer(results)
        zh = analyzer.filter_direction("en-zh")
        assert len(zh.results) == 1
        assert zh.results[0].name == "A"


class TestEdgeCases:
    def test_find_low_quality(self):
        results = [
            _make_result("good", quality=0.9),
            _make_result("bad", quality=0.3),
        ]
        cases = find_edge_cases(results, quality_threshold=0.5)
        assert len(cases["low_quality"]) == 1
        assert cases["low_quality"][0].name == "bad"

    def test_find_high_latency(self):
        results = [
            _make_result("fast", yaal=2.0),
            _make_result("slow", yaal=15.0),
        ]
        cases = find_edge_cases(results, latency_threshold=10.0)
        assert len(cases["high_latency"]) == 1
        assert cases["high_latency"][0].name == "slow"


class TestComparisonTable:
    def test_generates_markdown(self):
        results = [_make_result("A"), _make_result("B")]
        table = generate_comparison_table(results)
        assert "BLEU" in table
        assert "COMET" in table
        assert "|" in table

    def test_empty_results(self):
        table = generate_comparison_table([])
        assert "No results" in table
