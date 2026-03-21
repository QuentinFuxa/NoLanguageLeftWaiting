"""Tests for SimulStream wrapper -- IWSLT 2026 competition readiness."""

import os
import pytest
from unittest.mock import patch, MagicMock

from nllw.simulstream import (
    NLLWSpeechProcessor,
    SimulStreamConfig,
    IncrementalOutput,
    DIRECTION_DEFAULTS,
)


class TestIncrementalOutput:
    """Test the IncrementalOutput dataclass."""

    def test_empty_output(self):
        output = IncrementalOutput()
        assert output.is_empty
        assert output.new_string == ""
        assert output.new_tokens == []

    def test_non_empty(self):
        output = IncrementalOutput(new_string="hello", new_tokens=["hello"])
        assert not output.is_empty

    def test_deleted_tokens(self):
        output = IncrementalOutput(deleted_string="old", deleted_tokens=["old"])
        assert not output.is_empty


class TestSimulStreamConfig:
    """Test SimulStreamConfig creation and conversion."""

    def test_default_config(self):
        config = SimulStreamConfig()
        assert config.direction == "en-zh"
        assert config.backend_type == "alignatt"
        assert config.aggregation == "top_p"
        assert config.top_p_threshold == 0.85

    def test_to_backend_config(self):
        config = SimulStreamConfig(
            model_path="/tmp/test.gguf",
            direction="en-de",
            border_distance=2,
            word_batch=3,
            aggregation="top_p",
            top_p_threshold=0.75,
        )
        bc = config.to_backend_config()
        assert bc.direction == "en-de"
        assert bc.border_distance == 2
        assert bc.word_batch == 3
        assert bc.aggregation == "top_p"
        assert bc.top_p_threshold == 0.75
        assert bc.target_lang == "de"

    def test_target_lang_extraction(self):
        for direction, expected_lang in [
            ("en-zh", "zh"), ("en-de", "de"), ("en-it", "it"), ("cs-en", "en"),
        ]:
            config = SimulStreamConfig(direction=direction)
            bc = config.to_backend_config()
            assert bc.target_lang == expected_lang

    def test_extra_backend_params(self):
        config = SimulStreamConfig(
            extra_backend_params={"n_gpu_layers": 99, "top_k_heads": 10}
        )
        bc = config.to_backend_config()
        assert bc.n_gpu_layers == 99
        assert bc.top_k_heads == 10

    def test_repetition_forwarded(self):
        config = SimulStreamConfig(repetition_max_repeats=2)
        bc = config.to_backend_config()
        assert bc.repetition_max_repeats == 2


class TestDirectionDefaults:
    """Test per-direction optimal configs."""

    def test_all_4_directions_present(self):
        assert "en-zh" in DIRECTION_DEFAULTS
        assert "en-de" in DIRECTION_DEFAULTS
        assert "en-it" in DIRECTION_DEFAULTS
        assert "cs-en" in DIRECTION_DEFAULTS

    def test_en_zh_defaults(self):
        cfg = DIRECTION_DEFAULTS["en-zh"]
        assert cfg["border_distance"] == 3
        assert cfg["word_batch"] == 4
        assert cfg["aggregation"] == "top_p"
        assert cfg["top_p_threshold"] == 0.85

    def test_en_de_defaults(self):
        cfg = DIRECTION_DEFAULTS["en-de"]
        assert cfg["border_distance"] == 2
        assert cfg["word_batch"] == 3
        assert cfg["top_p_threshold"] == 0.75

    def test_en_it_defaults(self):
        cfg = DIRECTION_DEFAULTS["en-it"]
        assert cfg["top_p_threshold"] == 0.9

    def test_cs_en_defaults(self):
        cfg = DIRECTION_DEFAULTS["cs-en"]
        assert cfg["top_p_threshold"] == 0.9


class TestProcessorInit:
    """Test NLLWSpeechProcessor initialization."""

    def test_init_from_config(self):
        config = SimulStreamConfig(model_path="/nonexistent.gguf")
        proc = NLLWSpeechProcessor(config)
        assert not proc._is_initialized
        assert proc._source_lang == "en"
        assert proc._target_lang == "zh"

    def test_init_cs_en(self):
        config = SimulStreamConfig(direction="cs-en")
        proc = NLLWSpeechProcessor(config)
        assert proc._source_lang == "cs"
        assert proc._target_lang == "en"


class TestConfigFromEnv:
    """Test environment variable config loading."""

    def test_default_env_config(self):
        with patch.dict(os.environ, {}, clear=False):
            # Remove NLLW vars if present
            env = {k: v for k, v in os.environ.items() if not k.startswith("NLLW_")}
            with patch.dict(os.environ, env, clear=True):
                config = NLLWSpeechProcessor._config_from_env()
                assert config.direction == "en-zh"
                assert "n_gpu_layers" in config.extra_backend_params

    def test_custom_env_config(self):
        custom_env = {
            "NLLW_MODEL_PATH": "/custom/model.gguf",
            "NLLW_DEFAULT_DIRECTION": "en-de",
            "NLLW_N_GPU_LAYERS": "50",
        }
        with patch.dict(os.environ, custom_env, clear=False):
            config = NLLWSpeechProcessor._config_from_env()
            assert config.model_path == "/custom/model.gguf"
            assert config.direction == "en-de"
            assert config.extra_backend_params["n_gpu_layers"] == 50


class TestConfigFromDict:
    """Test dict config loading."""

    def test_basic_dict(self):
        d = {
            "model_path": "/tmp/test.gguf",
            "direction": "en-it",
            "border_distance": 2,
        }
        config = NLLWSpeechProcessor._config_from_dict(d)
        assert config.model_path == "/tmp/test.gguf"
        assert config.direction == "en-it"
        assert config.border_distance == 2

    def test_extra_params(self):
        d = {
            "model_path": "/tmp/test.gguf",
            "n_gpu_layers": 99,
            "custom_param": True,
        }
        config = NLLWSpeechProcessor._config_from_dict(d)
        assert config.extra_backend_params["n_gpu_layers"] == 99
        assert config.extra_backend_params["custom_param"] is True


class TestLoadModel:
    """Test load_model factory method."""

    def test_load_model_with_none_creates_env_config(self):
        """load_model(None) should use env vars (Docker pattern)."""
        with patch.object(NLLWSpeechProcessor, '_initialize') as mock_init:
            with patch.dict(os.environ, {"NLLW_MODEL_PATH": "/test/model.gguf"}):
                proc = NLLWSpeechProcessor.load_model(None)
                assert proc.config.model_path == "/test/model.gguf"
                mock_init.assert_called_once()

    def test_load_model_with_dict(self):
        """load_model(dict) should create config from dict."""
        with patch.object(NLLWSpeechProcessor, '_initialize') as mock_init:
            proc = NLLWSpeechProcessor.load_model({
                "model_path": "/dict/model.gguf",
                "direction": "cs-en",
            })
            assert proc.config.model_path == "/dict/model.gguf"
            assert proc.config.direction == "cs-en"
            mock_init.assert_called_once()

    def test_load_model_with_config(self):
        """load_model(SimulStreamConfig) should use it directly."""
        with patch.object(NLLWSpeechProcessor, '_initialize') as mock_init:
            config = SimulStreamConfig(model_path="/config/model.gguf")
            proc = NLLWSpeechProcessor.load_model(config)
            assert proc.config.model_path == "/config/model.gguf"
            mock_init.assert_called_once()


class TestDirectionSwitching:
    """Test dynamic direction switching (SimulStream protocol)."""

    def test_set_target_language(self):
        config = SimulStreamConfig()
        proc = NLLWSpeechProcessor(config)
        proc.set_target_language("de")
        assert proc.config.direction == "en-de"
        assert proc.config.border_distance == 2  # en-de default
        assert proc.config.top_p_threshold == 0.75

    def test_set_source_language(self):
        config = SimulStreamConfig(direction="en-zh")
        proc = NLLWSpeechProcessor(config)
        proc.set_source_language("cs")
        assert proc.config.direction == "cs-zh"

    def test_direction_switch_reinitializes(self):
        config = SimulStreamConfig()
        proc = NLLWSpeechProcessor(config)
        proc._is_initialized = True
        proc._backend = MagicMock()

        proc.set_target_language("de")
        assert not proc._is_initialized
        assert proc._backend is None

    def test_same_direction_no_reinit(self):
        config = SimulStreamConfig(direction="en-zh")
        proc = NLLWSpeechProcessor(config)
        proc._is_initialized = True
        mock_backend = MagicMock()
        proc._backend = mock_backend

        proc.set_target_language("zh")  # Same direction
        assert proc._is_initialized
        assert proc._backend is mock_backend

    def test_clear_resets_state(self):
        config = SimulStreamConfig()
        proc = NLLWSpeechProcessor(config)
        proc._committed_text = "some text"
        proc._n_chunks_processed = 10
        proc._word_emission_times = [1.0, 2.0]

        proc.clear()
        assert proc._committed_text == ""
        assert proc._n_chunks_processed == 0
        assert proc._word_emission_times == []


class TestStats:
    """Test stats property."""

    def test_stats_includes_config(self):
        config = SimulStreamConfig(direction="en-de", border_distance=2, word_batch=3)
        proc = NLLWSpeechProcessor(config)
        stats = proc.stats
        assert stats["direction"] == "en-de"
        assert stats["border_distance"] == 2
        assert stats["word_batch"] == 3
        assert stats["aggregation"] == "top_p"


class TestSpeechChunkSize:
    """Test speech_chunk_size property."""

    def test_default_chunk_size(self):
        config = SimulStreamConfig()
        proc = NLLWSpeechProcessor(config)
        # 960 / 16000 = 0.06 seconds = 60ms
        assert proc.speech_chunk_size == pytest.approx(0.06, abs=0.001)

    def test_custom_chunk_size(self):
        config = SimulStreamConfig(speech_chunk_size=1600, sample_rate=16000)
        proc = NLLWSpeechProcessor(config)
        assert proc.speech_chunk_size == pytest.approx(0.1, abs=0.001)


class TestTokensToString:
    """Test tokens_to_string (SimulStream protocol)."""

    def test_join_tokens(self):
        config = SimulStreamConfig()
        proc = NLLWSpeechProcessor(config)
        assert proc.tokens_to_string(["hello", " ", "world"]) == "hello world"

    def test_empty_tokens(self):
        config = SimulStreamConfig()
        proc = NLLWSpeechProcessor(config)
        assert proc.tokens_to_string([]) == ""
