"""Tests for OmniSTEval JSONL export module."""

import json
import io
import pytest
from nllw.omnisteval import (
    OmniSTEvalEntry,
    SimulEvalEntry,
    trace_to_omnisteval,
    eval_result_to_omnisteval,
    eval_result_to_simuleval,
    write_jsonl,
    write_simuleval_jsonl,
)


# ---------------------------------------------------------------------------
# SimulEval format tests (PRIMARY format for OmniSTEval)
# ---------------------------------------------------------------------------

class TestSimulEvalEntry:
    def test_to_dict(self):
        entry = SimulEvalEntry(
            source="segment_0.wav",
            prediction="bonjour le monde",
            delays=[100.0, 200.0, 400.0],
            elapsed=[110.0, 220.0, 430.0],
            source_length=5000.0,
            index=0,
        )
        d = entry.to_dict()
        assert d["source"] == "segment_0.wav"
        assert d["prediction"] == "bonjour le monde"
        assert len(d["delays"]) == 3
        assert d["source_length"] == 5000.0

    def test_to_dict_no_optional(self):
        entry = SimulEvalEntry(
            source="test.wav",
            prediction="hello",
            delays=[100.0],
            elapsed=[110.0],
            source_length=2000.0,
        )
        d = entry.to_dict()
        assert "index" not in d
        assert "reference" not in d or d["reference"] == ""

    def test_to_dict_with_reference(self):
        entry = SimulEvalEntry(
            prediction="test", reference="ref",
            delays=[100.0], elapsed=[100.0], source_length=1000.0,
        )
        d = entry.to_dict()
        # reference not included if empty, but included if set
        # (it's always included in to_dict if non-empty)

    def test_validate_correct(self):
        entry = SimulEvalEntry(
            prediction="hello world",
            delays=[100.0, 200.0],
            elapsed=[110.0, 220.0],
            source_length=3000.0,
        )
        assert entry.validate() == []

    def test_validate_delays_length_mismatch(self):
        entry = SimulEvalEntry(
            prediction="hello world",
            delays=[100.0],  # Should be 2 delays
            elapsed=[110.0, 220.0],
            source_length=3000.0,
        )
        errors = entry.validate()
        assert any("delays length" in e for e in errors)

    def test_validate_elapsed_length_mismatch(self):
        entry = SimulEvalEntry(
            prediction="hello world",
            delays=[100.0, 200.0],
            elapsed=[110.0],  # Should be 2
            source_length=3000.0,
        )
        errors = entry.validate()
        assert any("elapsed length" in e for e in errors)

    def test_validate_zero_source_length(self):
        entry = SimulEvalEntry(
            prediction="test",
            delays=[100.0],
            elapsed=[100.0],
            source_length=0.0,
        )
        errors = entry.validate()
        assert any("source_length" in e for e in errors)


class TestEvalResultToSimulEval:
    def test_basic_conversion(self):
        eval_result = {
            "per_sentence": [
                {
                    "source": "hello world test",
                    "hypothesis": "bonjour le monde",
                    "delays": [1, 1, 3],
                },
            ]
        }
        entries = eval_result_to_simuleval(eval_result)
        assert len(entries) == 1
        entry = entries[0]
        assert entry.prediction == "bonjour le monde"
        assert len(entry.delays) == 3  # One per word
        assert entry.source_length > 0

    def test_delays_in_milliseconds(self):
        eval_result = {
            "per_sentence": [
                {
                    "source": "one two three four five",
                    "hypothesis": "un deux",
                    "delays": [1, 3],
                },
            ]
        }
        entries = eval_result_to_simuleval(eval_result, words_per_second=2.5)
        entry = entries[0]
        # delay[0] = 1 word / 2.5 wps * 1000 = 400ms
        assert entry.delays[0] == pytest.approx(400.0, abs=1)
        # delay[1] = 3 words / 2.5 wps * 1000 = 1200ms
        assert entry.delays[1] == pytest.approx(1200.0, abs=1)

    def test_source_length_in_ms(self):
        eval_result = {
            "per_sentence": [
                {
                    "source": "one two three four five",  # 5 words
                    "hypothesis": "un",
                    "delays": [3],
                },
            ]
        }
        entries = eval_result_to_simuleval(eval_result, words_per_second=2.5)
        # source_length = 5 / 2.5 * 1000 = 2000ms
        assert entries[0].source_length == pytest.approx(2000.0, abs=1)

    def test_override_source_length(self):
        eval_result = {
            "per_sentence": [
                {
                    "source": "test",
                    "hypothesis": "test",
                    "delays": [1],
                },
            ]
        }
        entries = eval_result_to_simuleval(
            eval_result, source_length_ms=5000.0
        )
        assert entries[0].source_length == 5000.0

    def test_multi_sentence(self):
        eval_result = {
            "per_sentence": [
                {
                    "source": "hello world",
                    "hypothesis": "bonjour monde",
                    "delays": [1, 2],
                },
                {
                    "source": "good morning",
                    "hypothesis": "bonjour",
                    "delays": [2],
                },
            ]
        }
        entries = eval_result_to_simuleval(eval_result)
        assert len(entries) == 2
        assert entries[0].prediction == "bonjour monde"
        assert entries[1].prediction == "bonjour"

    def test_delays_match_words(self):
        """Critical: len(delays) must equal number of words in prediction."""
        eval_result = {
            "per_sentence": [
                {
                    "source": "a b c d e",
                    "hypothesis": "x y z",
                    "delays": [1, 3, 5],
                },
            ]
        }
        entries = eval_result_to_simuleval(eval_result)
        entry = entries[0]
        n_words = len(entry.prediction.split())
        assert len(entry.delays) == n_words
        assert len(entry.elapsed) == n_words
        assert entry.validate() == []

    def test_include_reference(self):
        eval_result = {
            "per_sentence": [
                {
                    "source": "hello",
                    "hypothesis": "bonjour",
                    "reference": "salut",
                    "delays": [1],
                },
            ]
        }
        entries = eval_result_to_simuleval(eval_result, include_reference=True)
        assert entries[0].reference == "salut"

    def test_empty_hypothesis(self):
        eval_result = {
            "per_sentence": [
                {
                    "source": "hello world",
                    "hypothesis": "",
                    "delays": [],
                },
            ]
        }
        entries = eval_result_to_simuleval(eval_result)
        assert len(entries) == 1
        assert entries[0].prediction == ""
        assert entries[0].delays == []

    def test_index_assigned(self):
        eval_result = {
            "per_sentence": [
                {"source": "a", "hypothesis": "b", "delays": [1]},
                {"source": "c", "hypothesis": "d", "delays": [1]},
            ]
        }
        entries = eval_result_to_simuleval(eval_result)
        assert entries[0].index == 0
        assert entries[1].index == 1


class TestWriteSimulEvalJsonl:
    def test_write_to_buffer(self):
        entries = [
            SimulEvalEntry(
                source="test.wav",
                prediction="bonjour monde",
                delays=[100.0, 300.0],
                elapsed=[110.0, 320.0],
                source_length=3000.0,
            ),
        ]
        buf = io.StringIO()
        write_simuleval_jsonl(entries, buf)
        buf.seek(0)
        d = json.loads(buf.readline())
        assert d["prediction"] == "bonjour monde"
        assert len(d["delays"]) == 2
        assert d["source_length"] == 3000.0

    def test_unicode_prediction(self):
        entries = [
            SimulEvalEntry(
                prediction="\u4f60\u597d\u4e16\u754c",
                delays=[100.0, 200.0],
                elapsed=[100.0, 200.0],
                source_length=2000.0,
            ),
        ]
        buf = io.StringIO()
        write_simuleval_jsonl(entries, buf)
        buf.seek(0)
        d = json.loads(buf.readline())
        assert "\u4f60\u597d" in d["prediction"]

    def test_multi_entry_output(self):
        entries = [
            SimulEvalEntry(
                prediction="hello", delays=[100.0], elapsed=[100.0],
                source_length=1000.0, index=i,
            )
            for i in range(3)
        ]
        buf = io.StringIO()
        write_simuleval_jsonl(entries, buf)
        buf.seek(0)
        lines = buf.readlines()
        assert len(lines) == 3


# ---------------------------------------------------------------------------
# Legacy format tests
# ---------------------------------------------------------------------------

class TestOmniSTEvalEntry:
    def test_to_dict(self):
        entry = OmniSTEvalEntry(
            talk_id="talk_0",
            offset=1.5,
            duration=0.4,
            emission_cu=1.5,
            emission_ca=1.52,
            text="bonjour",
            is_eos=False,
        )
        d = entry.to_dict()
        assert d["talk_id"] == "talk_0"
        assert d["offset"] == 1.5
        assert d["text"] == "bonjour"
        assert d["is_eos"] is False

    def test_eos_flag(self):
        entry = OmniSTEvalEntry(
            talk_id="t", offset=0, duration=0,
            emission_cu=0, emission_ca=0, text=".", is_eos=True,
        )
        assert entry.is_eos is True


class TestTraceToOmniSTEval:
    def test_basic_conversion(self):
        trace = {
            "actions": [
                {"action": "READ", "word_idx": 0, "emission_time": 0.0},
                {"action": "READ", "word_idx": 1, "emission_time": 1.0},
                {"action": "WRITE", "word_idx": 1, "text": "bonjour le"},
                {"action": "READ", "word_idx": 2, "emission_time": 2.0},
                {"action": "WRITE", "word_idx": 2, "text": "monde"},
            ],
            "n_source": 3,
        }
        entries = trace_to_omnisteval(trace, talk_id="test_0")
        assert len(entries) == 2
        assert entries[0].text == "bonjour le"
        assert entries[1].text == "monde"
        assert entries[-1].is_eos is True
        assert entries[0].is_eos is False

    def test_empty_trace(self):
        trace = {"actions": [], "n_source": 0}
        entries = trace_to_omnisteval(trace)
        assert len(entries) == 0

    def test_time_domain(self):
        trace = {
            "actions": [
                {"action": "READ", "word_idx": 0, "emission_time": 0.0},
                {"action": "WRITE", "word_idx": 0, "text": "test"},
            ],
            "n_source": 5,
        }
        entries = trace_to_omnisteval(trace, source_length_s=10.0)
        assert len(entries) == 1
        assert entries[0].offset == pytest.approx(2.0, abs=0.01)

    def test_empty_text_skipped(self):
        trace = {
            "actions": [
                {"action": "WRITE", "word_idx": 0, "text": ""},
                {"action": "WRITE", "word_idx": 0, "text": "  "},
                {"action": "WRITE", "word_idx": 1, "text": "actual text"},
            ],
            "n_source": 2,
        }
        entries = trace_to_omnisteval(trace)
        assert len(entries) == 1
        assert entries[0].text == "actual text"


class TestWriteJsonl:
    def test_write_to_buffer(self):
        entries = [
            OmniSTEvalEntry("t0", 0.0, 0.4, 0.0, 0.1, "hello", False),
            OmniSTEvalEntry("t0", 0.4, 0.4, 0.4, 0.5, "world", True),
        ]
        buf = io.StringIO()
        write_jsonl(entries, buf)
        buf.seek(0)
        lines = buf.readlines()
        assert len(lines) == 2
        for line in lines:
            d = json.loads(line)
            assert "talk_id" in d
            assert "text" in d

    def test_unicode_preserved(self):
        entries = [
            OmniSTEvalEntry("t0", 0.0, 0.4, 0.0, 0.0, "\u4f60\u597d\u4e16\u754c", True),
        ]
        buf = io.StringIO()
        write_jsonl(entries, buf)
        buf.seek(0)
        d = json.loads(buf.readline())
        assert "\u4f60\u597d" in d["text"]


class TestEvalResultToOmniSTEval:
    def test_per_sentence_conversion(self):
        eval_result = {
            "per_sentence": [
                {
                    "source": "hello world test",
                    "hypothesis": "bonjour le monde",
                    "delays": [1, 1, 3],
                },
                {
                    "source": "good morning",
                    "hypothesis": "bonjour",
                    "delays": [2],
                },
            ]
        }
        entries = eval_result_to_omnisteval(eval_result, talk_id_prefix="talk")
        assert len(entries) >= 2
        talk_ids = set(e.talk_id for e in entries)
        assert len(talk_ids) == 2, f"Expected 2 talk_ids, got {talk_ids}"
