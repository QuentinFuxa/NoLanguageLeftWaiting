"""Tests for evaluation harness functions (no GPU required)."""

import json
import os
import tempfile

import numpy as np
import pytest

from nllw.eval import (
    LANG_CONFIGS,
    save_results,
    print_summary,
    score_with_comet,
)


def _make_output_data(n=5):
    """Create mock evaluation output data."""
    sentences = []
    for i in range(n):
        sentences.append({
            "id": i,
            "source": f"This is test sentence number {i}.",
            "reference": f"Ceci est la phrase test numéro {i}.",
            "translation": f"Ceci est la phrase de test numéro {i}.",
            "num_source_words": 6,
            "num_target_tokens": 8,
            "al": 2.5 + i * 0.1,
            "laal": 2.0 + i * 0.1,
            "ap": 0.6 + i * 0.01,
            "max_cw": 3,
            "time_ms": 100 + i * 10,
            "border_stops": 2,
        })
    summary = {
        "num_sentences": n,
        "border_distance": 3,
        "word_batch": 1,
        "num_alignment_heads": 10,
        "latency": {
            "avg_al": 2.7,
            "median_al": 2.7,
            "p90_al": 3.0,
            "avg_laal": 2.2,
            "avg_yaal": 2.0,
            "avg_ap": 0.63,
            "avg_dal": 2.9,
            "avg_max_cw": 3.0,
            "max_max_cw": 3,
        },
        "avg_time_ms": 120,
        "total_time_s": 0.6,
    }
    return {"summary": summary, "sentences": sentences}


class TestSaveResults:
    def test_creates_json_and_text_files(self, tmp_path):
        output_data = _make_output_data()
        out = str(tmp_path / "test_results.json")
        save_results(output_data, out, lang_pair="en-fr", prompt_format="hymt")

        assert os.path.exists(out)
        assert os.path.exists(out.replace(".json", "_hyp.txt"))
        assert os.path.exists(out.replace(".json", "_ref.txt"))
        assert os.path.exists(out.replace(".json", "_src.txt"))

        with open(out) as f:
            data = json.load(f)
        assert data["summary"]["language_pair"] == "en-fr"
        assert len(data["sentences"]) == 5

    def test_text_files_have_correct_lines(self, tmp_path):
        output_data = _make_output_data(3)
        out = str(tmp_path / "results.json")
        save_results(output_data, out)

        with open(out.replace(".json", "_hyp.txt")) as f:
            hyp_lines = f.readlines()
        with open(out.replace(".json", "_ref.txt")) as f:
            ref_lines = f.readlines()
        with open(out.replace(".json", "_src.txt")) as f:
            src_lines = f.readlines()

        assert len(hyp_lines) == 3
        assert len(ref_lines) == 3
        assert len(src_lines) == 3


class TestPrintSummary:
    def test_basic(self, capsys):
        summary = _make_output_data()["summary"]
        print_summary(summary, "en-fr")
        captured = capsys.readouterr()
        assert "en-fr" in captured.out
        assert "2.700" in captured.out  # avg_al

    def test_with_comet(self, capsys):
        summary = _make_output_data()["summary"]
        summary["comet"] = {
            "model": "Unbabel/XCOMET-XL",
            "system_score": 0.8421,
            "avg_score": 0.8400,
            "min_score": 0.7500,
            "num_scored": 5,
            "num_empty": 0,
        }
        print_summary(summary, "en-zh")
        captured = capsys.readouterr()
        assert "XCOMET-XL" in captured.out
        assert "0.8421" in captured.out


class TestLangConfigs:
    def test_all_configs_have_required_keys(self):
        for pair, cfg in LANG_CONFIGS.items():
            assert "src_lang" in cfg, f"{pair} missing src_lang"
            assert "tgt_lang" in cfg, f"{pair} missing tgt_lang"
            assert "src_script" in cfg, f"{pair} missing src_script"
            assert "tgt_script" in cfg, f"{pair} missing tgt_script"

    def test_expected_pairs(self):
        assert "en-zh" in LANG_CONFIGS
        assert "en-de" in LANG_CONFIGS
        assert "en-it" in LANG_CONFIGS
        assert "cs-en" in LANG_CONFIGS


class TestScoreWithComet:
    def test_handles_all_empty_translations(self):
        """score_with_comet should handle all-empty translations gracefully."""
        output_data = _make_output_data(3)
        for s in output_data["sentences"]:
            s["translation"] = ""

        # Should not crash, just skip scoring
        result = score_with_comet(output_data, model_name="Unbabel/XCOMET-XL")
        # No comet key added since all empty
        assert "comet" not in result["summary"]
