"""Tests for experiment config and registry (no GPU required)."""

import json
import os
import tempfile
import pytest
from nllw.experiment import (
    ExperimentConfig,
    ExperimentRecord,
    ExperimentRegistry,
)


class TestExperimentConfig:
    def test_defaults(self):
        config = ExperimentConfig(name="test")
        assert config.backend_type == "alignatt"
        assert config.border_distance == 3
        assert config.direction == "en-zh"

    def test_from_dict_flat(self):
        d = {
            "name": "test-exp",
            "backend_type": "alignatt",
            "border_distance": 4,
            "word_batch": 2,
            "direction": "en-de",
        }
        config = ExperimentConfig.from_dict(d)
        assert config.name == "test-exp"
        assert config.border_distance == 4
        assert config.direction == "en-de"

    def test_from_dict_nested(self):
        d = {
            "name": "nested-test",
            "backend": {
                "type": "alignatt",
                "model": "/path/to/model.gguf",
                "border_distance": 3,
            },
            "eval": {
                "direction": "en-zh",
                "n_sentences": 100,
                "comet": True,
            },
        }
        config = ExperimentConfig.from_dict(d)
        assert config.name == "nested-test"
        assert config.model_path == "/path/to/model.gguf"
        assert config.direction == "en-zh"
        assert config.n_sentences == 100
        assert config.compute_comet is True

    def test_from_dict_ignores_unknown(self):
        d = {"name": "test", "unknown_field": 42}
        config = ExperimentConfig.from_dict(d)
        assert config.name == "test"

    def test_to_backend_config(self):
        config = ExperimentConfig(
            name="test",
            model_path="/model.gguf",
            direction="en-fr",
            border_distance=4,
        )
        bc = config.to_backend_config()
        assert bc["model_path"] == "/model.gguf"
        assert bc["direction"] == "en-fr"
        assert bc["border_distance"] == 4
        assert bc["target_lang"] == "fr"


class TestExperimentRecord:
    def test_to_dict(self):
        config = ExperimentConfig(name="test-record")
        record = ExperimentRecord(
            config=config,
            bleu=42.5,
            comet=0.85,
            avg_yaal=3.2,
        )
        d = record.to_dict()
        assert d["bleu"] == 42.5
        assert d["comet"] == 0.85
        assert d["config"]["name"] == "test-record"

    def test_summary_line(self):
        config = ExperimentConfig(name="my-experiment", direction="en-zh")
        record = ExperimentRecord(
            config=config,
            bleu=42.5,
            comet=0.85,
            avg_yaal=3.2,
            avg_al=2.8,
        )
        line = record.summary_line()
        assert "my-experiment" in line
        assert "42.5" in line
        assert "0.850" in line


class TestExperimentRegistry:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ExperimentRegistry(tmpdir)
            config = ExperimentConfig(name="test-save")
            record = ExperimentRecord(config=config, bleu=40.0, comet=0.8)

            path = registry.save(record)
            assert os.path.exists(path)

            # List
            files = registry.list_all()
            assert len(files) == 1

            # Load
            loaded = registry.load(files[0])
            assert loaded.bleu == 40.0
            assert loaded.comet == 0.8

    def test_load_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ExperimentRegistry(tmpdir)
            for i in range(3):
                config = ExperimentConfig(name=f"exp-{i}")
                record = ExperimentRecord(config=config, bleu=float(i * 10))
                registry.save(record)

            records = registry.load_all()
            assert len(records) == 3

    def test_empty_registry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ExperimentRegistry(tmpdir)
            assert registry.list_all() == []
            assert registry.load_all() == []
