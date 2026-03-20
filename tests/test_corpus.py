"""Tests for test corpus module."""

import pytest
from nllw.corpus import (
    get_corpus,
    get_corpus_as_pairs,
    list_directions,
    list_categories,
    corpus_stats,
)


class TestCorpus:
    def test_en_fr_exists(self):
        sentences = get_corpus("en-fr")
        assert len(sentences) > 10

    def test_en_zh_exists(self):
        sentences = get_corpus("en-zh")
        assert len(sentences) > 5

    def test_filter_by_category(self):
        simple = get_corpus("en-fr", categories=["simple"])
        assert all(s.category == "simple" for s in simple)
        assert len(simple) >= 3

    def test_limit(self):
        sentences = get_corpus("en-fr", n=3)
        assert len(sentences) == 3

    def test_as_pairs(self):
        pairs = get_corpus_as_pairs("en-fr", n=5)
        assert len(pairs) == 5
        assert "source" in pairs[0]
        assert "reference" in pairs[0]

    def test_list_directions(self):
        dirs = list_directions()
        assert "en-fr" in dirs
        assert "en-zh" in dirs

    def test_list_categories(self):
        cats = list_categories("en-fr")
        assert "simple" in cats
        assert "complex" in cats
        assert "technical" in cats

    def test_corpus_stats(self):
        stats = corpus_stats("en-fr")
        assert stats["total"] > 10
        assert "simple" in stats

    def test_unknown_direction(self):
        sentences = get_corpus("xx-yy")
        assert len(sentences) == 0


class TestFloresLoading:
    """Test FLORES+ dataset loading (requires internet + datasets library)."""

    @pytest.mark.skipif(
        not pytest.importorskip("datasets", reason="datasets not installed"),
        reason="datasets library required",
    )
    def test_load_flores_en_fr(self):
        from nllw.eval import load_flores
        corpus = load_flores("en", "fr", n=5)
        assert len(corpus) == 5
        assert "source" in corpus[0]
        assert "reference" in corpus[0]
        assert len(corpus[0]["source"]) > 0
        assert len(corpus[0]["reference"]) > 0
