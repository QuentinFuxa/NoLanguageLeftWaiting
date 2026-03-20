"""Tests for OmniSTEval JSONL export module."""

import json
import io
import pytest
from nllw.omnisteval import (
    OmniSTEvalEntry,
    trace_to_omnisteval,
    eval_result_to_omnisteval,
    write_jsonl,
)


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
        """Convert a simple trace with READ/WRITE actions."""
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
        # Last entry should be EOS
        assert entries[-1].is_eos is True
        assert entries[0].is_eos is False

    def test_empty_trace(self):
        trace = {"actions": [], "n_source": 0}
        entries = trace_to_omnisteval(trace)
        assert len(entries) == 0

    def test_write_only_actions(self):
        """Trace with WRITE but no matching READ."""
        trace = {
            "actions": [
                {"action": "WRITE", "word_idx": 0, "text": "hello"},
            ],
            "n_source": 1,
        }
        entries = trace_to_omnisteval(trace)
        assert len(entries) == 1
        assert entries[0].text == "hello"

    def test_time_domain(self):
        """Conversion with explicit source_length_s."""
        trace = {
            "actions": [
                {"action": "READ", "word_idx": 0, "emission_time": 0.0},
                {"action": "WRITE", "word_idx": 0, "text": "test"},
            ],
            "n_source": 5,
        }
        entries = trace_to_omnisteval(trace, source_length_s=10.0)
        assert len(entries) == 1
        # 10.0 / 5 = 2.0 per word, offset = (0+1)*2.0 = 2.0
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
        # Each line is valid JSON
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
        """Convert eval result with per_sentence data."""
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
        # Should produce entries for the last sentence at minimum
        assert len(entries) >= 1
