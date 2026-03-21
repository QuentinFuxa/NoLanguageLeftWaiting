"""Tests for SimulStream wrapper -- IWSLT 2026 competition readiness."""

import os
import json
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from nllw.simulstream import (
    NLLWSpeechProcessor,
    SimulStreamConfig,
    IncrementalOutput,
    EmissionEvent,
    DIRECTION_DEFAULTS,
    process_gold_transcript_longform,
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


# ---------------------------------------------------------------------------
# Longform mode tests (CRITICAL for IWSLT 2026 competition)
# ---------------------------------------------------------------------------

class TestEmissionEvent:
    """Test EmissionEvent dataclass."""

    def test_basic_event(self):
        event = EmissionEvent(
            emission_time=1500.0, wall_clock=1600.0,
            text="Hello", is_final=False
        )
        assert event.emission_time == 1500.0
        assert event.text == "Hello"
        assert event.status == "COMPLETE"

    def test_final_event(self):
        event = EmissionEvent(
            emission_time=3000.0, wall_clock=3200.0,
            text="world.", is_final=True
        )
        assert event.is_final


class TestLongformConfig:
    """Test longform configuration."""

    def test_longform_default_true(self):
        config = SimulStreamConfig()
        assert config.longform is True

    def test_auto_sentence_boundary_default_true(self):
        config = SimulStreamConfig()
        assert config.auto_sentence_boundary is True

    def test_longform_disabled(self):
        config = SimulStreamConfig(longform=False)
        assert config.longform is False


class TestLongformState:
    """Test longform state tracking in processor."""

    def test_init_longform_state(self):
        config = SimulStreamConfig()
        proc = NLLWSpeechProcessor(config)
        assert proc._emission_log == []
        assert proc._recording_text == ""
        assert proc._n_sentences_in_recording == 0

    def test_clear_resets_longform_state(self):
        config = SimulStreamConfig()
        proc = NLLWSpeechProcessor(config)
        proc._emission_log = [EmissionEvent(1.0, 2.0, "test")]
        proc._recording_text = "accumulated text"
        proc._n_sentences_in_recording = 3
        proc._recording_start_time = 100.0

        proc.clear()
        assert proc._emission_log == []
        assert proc._recording_text == ""
        assert proc._n_sentences_in_recording == 0
        assert proc._recording_start_time == 0.0


class TestSentenceBoundaryDetection:
    """Test auto sentence boundary detection."""

    def test_detect_period_english(self):
        config = SimulStreamConfig(direction="en-de")
        proc = NLLWSpeechProcessor(config)
        assert proc._detect_sentence_boundary("Hello world.")
        assert proc._detect_sentence_boundary("What is this?")
        assert proc._detect_sentence_boundary("Amazing!")

    def test_no_boundary_mid_sentence(self):
        config = SimulStreamConfig(direction="en-de")
        proc = NLLWSpeechProcessor(config)
        assert not proc._detect_sentence_boundary("Hello world")
        assert not proc._detect_sentence_boundary("The president")

    def test_detect_chinese_punctuation(self):
        config = SimulStreamConfig(direction="en-zh")
        proc = NLLWSpeechProcessor(config)
        assert proc._detect_sentence_boundary("\u7f8e\u56fd\u603b\u7edf\u3002")  # ends with 。
        assert proc._detect_sentence_boundary("\u662f\u4ec0\u4e48\uff1f")  # ends with ？
        assert proc._detect_sentence_boundary("\u592a\u597d\u4e86\uff01")  # ends with ！

    def test_no_boundary_empty(self):
        config = SimulStreamConfig(direction="en-de")
        proc = NLLWSpeechProcessor(config)
        assert not proc._detect_sentence_boundary("")
        assert not proc._detect_sentence_boundary("   ")

    def test_trailing_whitespace_handled(self):
        config = SimulStreamConfig(direction="en-de")
        proc = NLLWSpeechProcessor(config)
        assert proc._detect_sentence_boundary("Hello world.  ")


class TestLongformProcessing:
    """Test longform word processing with mock backend."""

    def _make_mock_processor(self, longform=True, direction="en-zh"):
        config = SimulStreamConfig(
            longform=longform, direction=direction,
            auto_sentence_boundary=False,  # Disable auto for predictable tests
        )
        proc = NLLWSpeechProcessor(config)
        proc._is_initialized = True
        proc._backend = MagicMock()
        return proc

    def test_longform_accumulates_across_sentences(self):
        """In longform mode, recording_text accumulates across sentence boundaries."""
        proc = self._make_mock_processor(longform=True)

        # Mock backend translate to return text
        from nllw.backend_protocol import TranslationStep
        proc._backend.translate.return_value = TranslationStep(
            text="Translated. ", is_final=False
        )

        # Sentence 1
        proc.process_words(["Hello"], is_final=False)
        proc._backend.translate.return_value = TranslationStep(
            text="End.", is_final=True
        )
        proc.process_words(["world"], is_final=True)

        assert proc._n_sentences_in_recording == 1
        assert "End." in proc._recording_text

        # Sentence 2
        proc._backend.translate.return_value = TranslationStep(
            text="More text.", is_final=True
        )
        proc.process_words(["more"], is_final=True)

        assert proc._n_sentences_in_recording == 2
        # Recording text should have both sentences
        assert "End." in proc._recording_text
        assert "More text." in proc._recording_text

    def test_longform_no_double_reset(self):
        """In longform mode, reset() is NOT called after translate(is_final=True)."""
        proc = self._make_mock_processor(longform=True)

        from nllw.backend_protocol import TranslationStep
        proc._backend.translate.return_value = TranslationStep(
            text="Output.", is_final=True
        )

        proc.process_words(["word"], is_final=True)

        # Backend.translate was called with is_final=True (triggers internal segment end)
        proc._backend.translate.assert_called_with(
            "word", is_final=True, emission_time=0.0
        )
        # In longform mode, reset() should NOT be called (backend handles it internally)
        proc._backend.reset.assert_not_called()

    def test_sentence_mode_calls_reset(self):
        """In sentence mode (longform=False), reset() IS called after is_final."""
        proc = self._make_mock_processor(longform=False)

        from nllw.backend_protocol import TranslationStep
        proc._backend.translate.return_value = TranslationStep(
            text="Output.", is_final=True
        )

        proc.process_words(["word"], is_final=True)

        # In sentence mode, reset() should be called
        proc._backend.reset.assert_called_once()

    def test_emission_log_tracks_events(self):
        """Emission events are logged for OmniSTEval output."""
        proc = self._make_mock_processor(longform=True)

        from nllw.backend_protocol import TranslationStep
        proc._backend.translate.return_value = TranslationStep(
            text="Hello ", is_final=False
        )

        proc.process_words(["source"], emission_time=1.5)

        assert len(proc._emission_log) == 1
        assert proc._emission_log[0].text == "Hello "
        assert proc._emission_log[0].emission_time == 1500.0  # Converted to ms

    def test_emission_log_property(self):
        """emission_log property returns a copy."""
        proc = self._make_mock_processor(longform=True)
        proc._emission_log.append(EmissionEvent(1.0, 2.0, "test"))

        log = proc.emission_log
        assert len(log) == 1
        # Modifying the copy doesn't affect the original
        log.append(EmissionEvent(3.0, 4.0, "extra"))
        assert len(proc._emission_log) == 1


class TestOmniSTEvalOutput:
    """Test OmniSTEval longform output generation."""

    def _make_proc_with_emissions(self):
        config = SimulStreamConfig(direction="en-de", longform=True)
        proc = NLLWSpeechProcessor(config)
        proc._recording_start_time = 1000.0  # Some fixed start
        proc._recording_text = "Hallo Welt. Guten Tag."
        proc._emission_log = [
            EmissionEvent(emission_time=500.0, wall_clock=520.0,
                          text="Hallo ", is_final=False),
            EmissionEvent(emission_time=800.0, wall_clock=850.0,
                          text="Welt. ", is_final=True),
            EmissionEvent(emission_time=1200.0, wall_clock=1250.0,
                          text="Guten ", is_final=False),
            EmissionEvent(emission_time=1500.0, wall_clock=1580.0,
                          text="Tag.", is_final=True),
        ]
        return proc

    def test_omnisteval_entry_structure(self):
        proc = self._make_proc_with_emissions()
        entry = proc.to_omnisteval_entry(
            source_name="test.wav",
            source_length_ms=5000.0,
        )

        assert entry["source"] == "test.wav"
        assert entry["prediction"] == "Hallo Welt. Guten Tag."
        assert entry["source_length"] == 5000.0
        assert len(entry["delays"]) == 4  # 4 words
        assert len(entry["elapsed"]) == 4

    def test_omnisteval_delays_per_word(self):
        """Per-word delays map to the last char of each word."""
        proc = self._make_proc_with_emissions()
        entry = proc.to_omnisteval_entry(
            source_name="test.wav",
            source_length_ms=5000.0,
        )

        delays = entry["delays"]
        # "Hallo" -> emission_time 500.0
        # "Welt." -> emission_time 800.0
        # "Guten" -> emission_time 1200.0
        # "Tag." -> emission_time 1500.0
        assert delays[0] == 500.0
        assert delays[1] == 800.0
        assert delays[2] == 1200.0
        assert delays[3] == 1500.0

    def test_omnisteval_char_level(self):
        """Char-level delays produce one delay per character."""
        proc = self._make_proc_with_emissions()
        entry = proc.to_omnisteval_entry(
            source_name="test.wav",
            source_length_ms=5000.0,
            char_level=True,
        )

        prediction = "Hallo Welt. Guten Tag."
        assert len(entry["delays"]) == len(prediction)

    def test_omnisteval_empty_prediction(self):
        config = SimulStreamConfig(longform=True)
        proc = NLLWSpeechProcessor(config)
        proc._recording_text = ""

        entry = proc.to_omnisteval_entry(source_length_ms=1000.0)
        assert entry["prediction"] == ""
        assert entry["delays"] == []
        assert entry["elapsed"] == []

    def test_stats_includes_longform_info(self):
        config = SimulStreamConfig(longform=True)
        proc = NLLWSpeechProcessor(config)
        proc._recording_text = "some text"
        proc._n_sentences_in_recording = 2
        proc._emission_log = [EmissionEvent(1.0, 2.0, "a")] * 3

        stats = proc.stats
        assert stats["longform"] is True
        assert stats["recording_text"] == "some text"
        assert stats["n_sentences_in_recording"] == 2
        assert stats["n_emission_events"] == 3


class TestLongformGoldTranscript:
    """Test process_gold_transcript_longform() -- competition format."""

    def test_longform_processes_multi_sentence(self):
        """Multi-sentence gold transcript produces one OmniSTEval entry."""
        # Create a temporary gold transcript JSONL
        words = [
            {"text": "The", "emission_time": 0.5, "is_final": False},
            {"text": "president", "emission_time": 0.8, "is_final": False},
            {"text": "spoke", "emission_time": 1.1, "is_final": True},
            {"text": "New", "emission_time": 2.0, "is_final": False},
            {"text": "reforms", "emission_time": 2.5, "is_final": True},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for w in words:
                f.write(json.dumps(w) + "\n")
            input_path = f.name

        try:
            # Create processor with mock backend
            config = SimulStreamConfig(longform=True, auto_sentence_boundary=False)
            proc = NLLWSpeechProcessor(config)
            proc._is_initialized = True

            from nllw.backend_protocol import TranslationStep
            mock_backend = MagicMock()
            # Return translations for each word
            mock_backend.translate.side_effect = [
                TranslationStep(text="", is_final=False),
                TranslationStep(text="Le president ", is_final=False),
                TranslationStep(text="a parle.", is_final=True),
                TranslationStep(text="", is_final=False),
                TranslationStep(text="Nouvelles reformes.", is_final=True),
            ]
            mock_backend.get_full_translation.return_value = ""
            mock_backend.reset.return_value = None
            proc._backend = mock_backend

            entry = process_gold_transcript_longform(
                proc, input_path,
                source_name="talk_1.wav",
                source_length_s=3.0,
            )

            assert entry["source"] == "talk_1.wav"
            assert entry["source_length"] == 3000.0  # 3s -> 3000ms
            assert "Le president" in entry["prediction"]
            assert "reformes" in entry["prediction"]
            # Should have delays for each word
            n_words = len(entry["prediction"].split())
            assert len(entry["delays"]) == n_words
            assert len(entry["elapsed"]) == n_words

        finally:
            os.unlink(input_path)


class TestBatchFirstEmissionTime:
    """Test batch_first_emission_time for correct LongYAAL."""

    def test_translation_step_has_batch_first_emission_time(self):
        """TranslationStep should have batch_first_emission_time field."""
        from nllw.backend_protocol import TranslationStep
        step = TranslationStep(text="hello", batch_first_emission_time=1.5)
        assert step.batch_first_emission_time == 1.5

    def test_translation_step_batch_first_defaults_none(self):
        """batch_first_emission_time defaults to None."""
        from nllw.backend_protocol import TranslationStep
        step = TranslationStep(text="hello")
        assert step.batch_first_emission_time is None

    def test_translation_step_has_avg_logprob(self):
        """TranslationStep should have avg_logprob field."""
        from nllw.backend_protocol import TranslationStep
        step = TranslationStep(text="hello", avg_logprob=-2.5)
        assert step.avg_logprob == -2.5

    def test_translation_step_avg_logprob_defaults_none(self):
        """avg_logprob defaults to None."""
        from nllw.backend_protocol import TranslationStep
        step = TranslationStep(text="hello")
        assert step.avg_logprob is None

    def test_emission_uses_batch_first_emission_time(self):
        """SimulStream should use batch_first_emission_time for emission events."""
        config = SimulStreamConfig(
            longform=True, direction="en-de",
            auto_sentence_boundary=False,
        )
        proc = NLLWSpeechProcessor(config)
        proc._is_initialized = True
        proc._backend = MagicMock()

        from nllw.backend_protocol import TranslationStep

        # Simulate word_batch=3: first 2 words return empty, 3rd returns text
        # The 3rd word's translate() returns with batch_first_emission_time=1.0
        # (the time of the 1st word), even though emission_time=3.0 for the 3rd word.
        proc._backend.translate.side_effect = [
            TranslationStep(text="", is_final=False),   # word 1 (batched)
            TranslationStep(text="", is_final=False),   # word 2 (batched)
            TranslationStep(
                text="Translated output",
                is_final=False,
                batch_first_emission_time=1.0,  # Time of first word in batch
            ),
        ]

        # Process words one at a time with different emission times
        proc.process_words(["The"], emission_time=1.0)
        proc.process_words(["president"], emission_time=2.0)
        proc.process_words(["announced"], emission_time=3.0)

        # The emission event should use batch_first_emission_time=1.0, not 3.0
        assert len(proc._emission_log) == 1
        event = proc._emission_log[0]
        # batch_first_emission_time=1.0, converted to ms: 1000.0
        assert event.emission_time == 1000.0  # 1.0s * 1000

    def test_emission_falls_back_to_emission_time_when_none(self):
        """If batch_first_emission_time is None, use the call's emission_time."""
        config = SimulStreamConfig(
            longform=True, direction="en-de",
            auto_sentence_boundary=False,
        )
        proc = NLLWSpeechProcessor(config)
        proc._is_initialized = True
        proc._backend = MagicMock()

        from nllw.backend_protocol import TranslationStep
        proc._backend.translate.return_value = TranslationStep(
            text="Output",
            is_final=False,
            batch_first_emission_time=None,
        )

        proc.process_words(["hello"], emission_time=2.5)

        assert len(proc._emission_log) == 1
        event = proc._emission_log[0]
        assert event.emission_time == 2500.0  # 2.5s * 1000

    def test_batch_first_in_longform_omnisteval(self):
        """batch_first_emission_time should affect OmniSTEval delays."""
        config = SimulStreamConfig(
            longform=True, direction="en-de",
            auto_sentence_boundary=False,
        )
        proc = NLLWSpeechProcessor(config)
        proc._is_initialized = True
        proc._backend = MagicMock()

        from nllw.backend_protocol import TranslationStep

        # Simulate: word at t=1.0 batched, word at t=2.0 triggers translation
        proc._backend.translate.side_effect = [
            TranslationStep(text="", is_final=False),
            TranslationStep(
                text="Uebersetzung fertig.",
                is_final=True,
                batch_first_emission_time=1.0,
            ),
        ]
        proc._backend.get_full_translation.return_value = ""

        proc.process_words(["first"], emission_time=1.0)
        proc.process_words(["second"], emission_time=2.0, is_final=True)

        entry = proc.to_omnisteval_entry(
            source_name="test.wav",
            source_length_ms=5000.0,
            char_level=False,
        )

        # Delays should use batch_first time (1000ms), not last word time (2000ms)
        assert len(entry["delays"]) == 2  # "Uebersetzung" "fertig."
        for d in entry["delays"]:
            assert d == 1000.0  # All words from batch starting at t=1.0

    def test_backend_config_batch_first_tracking(self):
        """Backend tracks batch_first_emission_time across batched words."""
        from nllw.backend_protocol import BackendConfig

        # Create config to simulate batch behavior
        config = BackendConfig(
            model_path="/tmp/test.gguf",
            heads_path="/tmp/heads.json",
            word_batch=3,
        )
        # Can't test full backend without GPU, but verify config is valid
        assert config.word_batch == 3
