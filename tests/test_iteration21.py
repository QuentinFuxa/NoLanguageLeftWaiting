"""Tests for iteration 21: OmniSTEval format fixes, source-aware batching."""

import json
import os
import sys
import unicodedata
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSourceAwareBatching(unittest.TestCase):
    """Tests for should_defer_batch() function."""

    def test_function_word_detected_en(self):
        from nllw.alignatt import should_defer_batch
        # English function words should trigger deferral
        self.assertTrue(should_defer_batch("the", "en"))
        self.assertTrue(should_defer_batch("of", "en"))
        self.assertTrue(should_defer_batch("in", "en"))
        self.assertTrue(should_defer_batch("a", "en"))
        self.assertTrue(should_defer_batch("and", "en"))
        self.assertTrue(should_defer_batch("to", "en"))
        self.assertTrue(should_defer_batch("is", "en"))

    def test_content_word_not_deferred_en(self):
        from nllw.alignatt import should_defer_batch
        # Content words should NOT trigger deferral
        self.assertFalse(should_defer_batch("president", "en"))
        self.assertFalse(should_defer_batch("announced", "en"))
        self.assertFalse(should_defer_batch("France", "en"))
        self.assertFalse(should_defer_batch("policies", "en"))
        self.assertFalse(should_defer_batch("reform", "en"))

    def test_case_insensitive(self):
        from nllw.alignatt import should_defer_batch
        self.assertTrue(should_defer_batch("The", "en"))
        self.assertTrue(should_defer_batch("THE", "en"))
        self.assertTrue(should_defer_batch("Of", "en"))

    def test_czech_function_words(self):
        from nllw.alignatt import should_defer_batch
        self.assertTrue(should_defer_batch("a", "cs"))
        self.assertTrue(should_defer_batch("v", "cs"))
        self.assertTrue(should_defer_batch("na", "cs"))
        self.assertTrue(should_defer_batch("se", "cs"))

    def test_czech_content_words(self):
        from nllw.alignatt import should_defer_batch
        self.assertFalse(should_defer_batch("prezident", "cs"))
        self.assertFalse(should_defer_batch("oznámil", "cs"))

    def test_max_defer_limit(self):
        from nllw.alignatt import should_defer_batch
        # Should defer when under limit
        self.assertTrue(should_defer_batch("the", "en", max_defer=2, deferred_count=0))
        self.assertTrue(should_defer_batch("the", "en", max_defer=2, deferred_count=1))
        # Should NOT defer when at limit
        self.assertFalse(should_defer_batch("the", "en", max_defer=2, deferred_count=2))
        self.assertFalse(should_defer_batch("the", "en", max_defer=2, deferred_count=3))

    def test_unknown_language_uses_english(self):
        from nllw.alignatt import should_defer_batch
        # Unknown language defaults to English function words
        self.assertTrue(should_defer_batch("the", "fr"))
        self.assertFalse(should_defer_batch("president", "fr"))

    def test_empty_word(self):
        from nllw.alignatt import should_defer_batch
        self.assertFalse(should_defer_batch("", "en"))
        self.assertFalse(should_defer_batch("  ", "en"))

    def test_source_aware_config(self):
        """Test that source_aware_batching is in BackendConfig."""
        from nllw.backend_protocol import BackendConfig
        config = BackendConfig()
        self.assertFalse(config.source_aware_batching)
        config2 = BackendConfig(source_aware_batching=True)
        self.assertTrue(config2.source_aware_batching)


class TestOmniSTEvalFormatFixes(unittest.TestCase):
    """Tests for OmniSTEval format improvements."""

    def test_nfkc_normalization_in_char_level(self):
        """NFKC normalization should be applied for char-level delays."""
        from nllw.simulstream import NLLWSpeechProcessor, SimulStreamConfig

        config = SimulStreamConfig(
            model_path="",
            direction="en-zh",
            longform=True,
        )
        processor = NLLWSpeechProcessor(config)
        processor._is_initialized = True  # Skip backend init

        # Simulate recording with full-width characters
        from nllw.simulstream import EmissionEvent
        # Full-width "Ａ" (U+FF21) normalizes to "A" (U+0041) under NFKC
        processor._recording_text = "测试"
        processor._recording_start_time = 1.0
        processor._emission_log = [
            EmissionEvent(emission_time=100.0, wall_clock=100.0, text="测试"),
        ]

        entry = processor.to_omnisteval_entry(
            source_name="test.wav",
            source_length_ms=1000.0,
            char_level=True,
        )

        # Should have NFKC normalized prediction
        self.assertEqual(entry["prediction"], unicodedata.normalize("NFKC", "测试"))
        self.assertEqual(len(entry["delays"]), len(entry["prediction"]))

    def test_end_of_turn_stripped(self):
        """<end_of_turn> tokens should be stripped from prediction."""
        from nllw.simulstream import NLLWSpeechProcessor, SimulStreamConfig, EmissionEvent

        config = SimulStreamConfig(
            model_path="",
            direction="en-de",
            longform=True,
        )
        processor = NLLWSpeechProcessor(config)
        processor._is_initialized = True

        processor._recording_text = "Hallo Welt<end_of_turn>"
        processor._recording_start_time = 1.0
        processor._emission_log = [
            EmissionEvent(emission_time=100.0, wall_clock=100.0, text="Hallo "),
            EmissionEvent(emission_time=200.0, wall_clock=200.0, text="Welt<end_of_turn>"),
        ]

        entry = processor.to_omnisteval_entry(
            source_name="test.wav",
            source_length_ms=1000.0,
            char_level=False,
        )

        self.assertNotIn("<end_of_turn>", entry["prediction"])
        self.assertEqual(entry["prediction"], "Hallo Welt")
        self.assertEqual(len(entry["delays"]), 2)  # 2 words

    def test_endoftext_stripped(self):
        """<|endoftext|> tokens should be stripped from prediction."""
        from nllw.simulstream import NLLWSpeechProcessor, SimulStreamConfig, EmissionEvent

        config = SimulStreamConfig(
            model_path="",
            direction="en-de",
            longform=True,
        )
        processor = NLLWSpeechProcessor(config)
        processor._is_initialized = True

        processor._recording_text = "Hello World<|endoftext|>"
        processor._recording_start_time = 1.0
        processor._emission_log = [
            EmissionEvent(emission_time=100.0, wall_clock=100.0, text="Hello World<|endoftext|>"),
        ]

        entry = processor.to_omnisteval_entry(
            source_name="test.wav",
            source_length_ms=1000.0,
        )

        self.assertNotIn("<|endoftext|>", entry["prediction"])
        self.assertEqual(entry["prediction"], "Hello World")

    def test_delay_monotonicity_enforcement(self):
        """Delays must be monotonically non-decreasing."""
        from nllw.simulstream import NLLWSpeechProcessor, SimulStreamConfig, EmissionEvent

        config = SimulStreamConfig(
            model_path="",
            direction="en-de",
            longform=True,
        )
        processor = NLLWSpeechProcessor(config)
        processor._is_initialized = True

        # Simulate non-monotonic emission times (can happen with async events)
        processor._recording_text = "A B C D"
        processor._recording_start_time = 1.0
        processor._emission_log = [
            EmissionEvent(emission_time=100.0, wall_clock=100.0, text="A "),
            EmissionEvent(emission_time=300.0, wall_clock=300.0, text="B "),
            EmissionEvent(emission_time=200.0, wall_clock=200.0, text="C "),  # Goes backward!
            EmissionEvent(emission_time=400.0, wall_clock=400.0, text="D"),
        ]

        entry = processor.to_omnisteval_entry(
            source_name="test.wav",
            source_length_ms=1000.0,
        )

        delays = entry["delays"]
        self.assertEqual(len(delays), 4)
        # Verify monotonicity
        for i in range(1, len(delays)):
            self.assertGreaterEqual(delays[i], delays[i - 1],
                                    f"delays[{i}]={delays[i]} < delays[{i-1}]={delays[i-1]}")

    def test_empty_recording(self):
        """Empty recording should produce valid OmniSTEval entry."""
        from nllw.simulstream import NLLWSpeechProcessor, SimulStreamConfig

        config = SimulStreamConfig(model_path="", direction="en-zh", longform=True)
        processor = NLLWSpeechProcessor(config)
        processor._is_initialized = True

        processor._recording_text = ""
        entry = processor.to_omnisteval_entry(
            source_name="empty.wav",
            source_length_ms=5000.0,
        )

        self.assertEqual(entry["prediction"], "")
        self.assertEqual(entry["delays"], [])
        self.assertEqual(entry["elapsed"], [])
        self.assertEqual(entry["source_length"], 5000.0)

    def test_char_level_chinese(self):
        """Char-level delays for Chinese should work correctly."""
        from nllw.simulstream import NLLWSpeechProcessor, SimulStreamConfig, EmissionEvent

        config = SimulStreamConfig(model_path="", direction="en-zh", longform=True)
        processor = NLLWSpeechProcessor(config)
        processor._is_initialized = True

        processor._recording_text = "美国总统"
        processor._recording_start_time = 1.0
        processor._emission_log = [
            EmissionEvent(emission_time=100.0, wall_clock=100.0, text="美国"),
            EmissionEvent(emission_time=200.0, wall_clock=200.0, text="总统"),
        ]

        entry = processor.to_omnisteval_entry(
            source_name="test.wav",
            source_length_ms=1000.0,
            char_level=True,
        )

        self.assertEqual(len(entry["delays"]), 4)  # 4 characters
        self.assertEqual(entry["delays"][0], 100.0)  # 美
        self.assertEqual(entry["delays"][1], 100.0)  # 国
        self.assertEqual(entry["delays"][2], 200.0)  # 总
        self.assertEqual(entry["delays"][3], 200.0)  # 统

    def test_whitespace_normalization(self):
        """Multiple spaces should be collapsed to single space."""
        from nllw.simulstream import NLLWSpeechProcessor, SimulStreamConfig, EmissionEvent

        config = SimulStreamConfig(model_path="", direction="en-de", longform=True)
        processor = NLLWSpeechProcessor(config)
        processor._is_initialized = True

        # Simulate output with double spaces (common in longform concatenation)
        processor._recording_text = "Hello  World   Test"
        processor._recording_start_time = 1.0
        processor._emission_log = [
            EmissionEvent(emission_time=100.0, wall_clock=100.0, text="Hello  "),
            EmissionEvent(emission_time=200.0, wall_clock=200.0, text="World   "),
            EmissionEvent(emission_time=300.0, wall_clock=300.0, text="Test"),
        ]

        entry = processor.to_omnisteval_entry(
            source_name="test.wav",
            source_length_ms=1000.0,
        )

        # Prediction should have normalized whitespace
        self.assertEqual(entry["prediction"], "Hello World Test")
        self.assertEqual(len(entry["delays"]), 3)  # 3 words

    def test_word_level_european(self):
        """Word-level delays for European languages."""
        from nllw.simulstream import NLLWSpeechProcessor, SimulStreamConfig, EmissionEvent

        config = SimulStreamConfig(model_path="", direction="en-de", longform=True)
        processor = NLLWSpeechProcessor(config)
        processor._is_initialized = True

        processor._recording_text = "Der Präsident hat"
        processor._recording_start_time = 1.0
        processor._emission_log = [
            EmissionEvent(emission_time=100.0, wall_clock=100.0, text="Der "),
            EmissionEvent(emission_time=200.0, wall_clock=200.0, text="Präsident "),
            EmissionEvent(emission_time=300.0, wall_clock=300.0, text="hat"),
        ]

        entry = processor.to_omnisteval_entry(
            source_name="test.wav",
            source_length_ms=1000.0,
        )

        self.assertEqual(len(entry["delays"]), 3)  # 3 words
        self.assertEqual(entry["delays"][0], 100.0)  # Der
        self.assertEqual(entry["delays"][1], 200.0)  # Präsident
        self.assertEqual(entry["delays"][2], 300.0)  # hat

    def test_char_level_auto_detect_zh(self):
        """char_level should auto-detect to True for zh target."""
        from nllw.simulstream import NLLWSpeechProcessor, SimulStreamConfig, EmissionEvent

        config = SimulStreamConfig(model_path="", direction="en-zh", longform=True)
        processor = NLLWSpeechProcessor(config)
        processor._is_initialized = True

        processor._recording_text = "美国"
        processor._recording_start_time = 1.0
        processor._emission_log = [
            EmissionEvent(emission_time=100.0, wall_clock=100.0, text="美国"),
        ]

        # char_level=None (default) should auto-detect to True for zh
        entry = processor.to_omnisteval_entry(
            source_name="test.wav",
            source_length_ms=1000.0,
        )

        # Should have per-character delays (2 chars), not per-word (1 word)
        self.assertEqual(len(entry["delays"]), 2)

    def test_char_level_auto_detect_de(self):
        """char_level should auto-detect to False for de target."""
        from nllw.simulstream import NLLWSpeechProcessor, SimulStreamConfig, EmissionEvent

        config = SimulStreamConfig(model_path="", direction="en-de", longform=True)
        processor = NLLWSpeechProcessor(config)
        processor._is_initialized = True

        processor._recording_text = "Hallo Welt"
        processor._recording_start_time = 1.0
        processor._emission_log = [
            EmissionEvent(emission_time=100.0, wall_clock=100.0, text="Hallo "),
            EmissionEvent(emission_time=200.0, wall_clock=200.0, text="Welt"),
        ]

        # char_level=None (default) should auto-detect to False for de
        entry = processor.to_omnisteval_entry(
            source_name="test.wav",
            source_length_ms=1000.0,
        )

        # Should have per-word delays (2 words), not per-char
        self.assertEqual(len(entry["delays"]), 2)

    def test_char_level_explicit_override(self):
        """Explicit char_level=False should override auto-detection for zh."""
        from nllw.simulstream import NLLWSpeechProcessor, SimulStreamConfig, EmissionEvent

        config = SimulStreamConfig(model_path="", direction="en-zh", longform=True)
        processor = NLLWSpeechProcessor(config)
        processor._is_initialized = True

        processor._recording_text = "美国 总统"  # With space = 2 words
        processor._recording_start_time = 1.0
        processor._emission_log = [
            EmissionEvent(emission_time=100.0, wall_clock=100.0, text="美国 "),
            EmissionEvent(emission_time=200.0, wall_clock=200.0, text="总统"),
        ]

        entry = processor.to_omnisteval_entry(
            source_name="test.wav",
            source_length_ms=1000.0,
            char_level=False,  # Force word-level
        )

        self.assertEqual(len(entry["delays"]), 2)  # 2 words

    def test_n_ctx_safety_valve_attribute(self):
        """Verify n_ctx config is accessible for safety valve."""
        from nllw.simulstream import SimulStreamConfig
        config = SimulStreamConfig(n_ctx=2048)
        self.assertEqual(config.n_ctx, 2048)
        self.assertEqual(int(config.n_ctx * 0.7), 1433)


class TestExperimentScript(unittest.TestCase):
    """Verify the GPU experiment script syntax is valid."""

    def test_script_imports(self):
        """Experiment script should be importable."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_experiments",
            os.path.join(os.path.dirname(__file__), "..", "scripts", "run_iteration21_experiments.py"),
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Verify key functions exist
        self.assertTrue(hasattr(module, "phase1_smoke_test"))
        self.assertTrue(hasattr(module, "phase2_perplexity_adaptive_border"))
        self.assertTrue(hasattr(module, "phase3_longform_e2e"))
        self.assertTrue(hasattr(module, "phase4_multi_direction_longform"))
        self.assertTrue(hasattr(module, "phase5_competition_format"))
        self.assertTrue(hasattr(module, "phase6_adaptive_top_p_decision"))

    def test_create_synthetic_gold(self):
        """Synthetic gold JSONL creation should work."""
        import importlib.util
        import tempfile
        spec = importlib.util.spec_from_file_location(
            "run_experiments",
            os.path.join(os.path.dirname(__file__), "..", "scripts", "run_iteration21_experiments.py"),
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            module.create_synthetic_gold(path, direction="en-zh", n_sentences=3)
            with open(path) as f:
                lines = [json.loads(l) for l in f if l.strip()]
            self.assertGreater(len(lines), 0)
            # Verify format
            for line in lines:
                self.assertIn("text", line)
                self.assertIn("emission_time", line)
                self.assertIn("is_final", line)
        finally:
            os.unlink(path)


class TestFunctionWordSets(unittest.TestCase):
    """Verify function word sets are comprehensive."""

    def test_common_en_prepositions(self):
        from nllw.alignatt import _EN_FUNCTION_WORDS
        for word in ["of", "in", "to", "for", "with", "on", "at", "from", "by"]:
            self.assertIn(word, _EN_FUNCTION_WORDS, f"Missing preposition: {word}")

    def test_common_en_determiners(self):
        from nllw.alignatt import _EN_FUNCTION_WORDS
        for word in ["the", "a", "an", "this", "that"]:
            self.assertIn(word, _EN_FUNCTION_WORDS, f"Missing determiner: {word}")

    def test_common_en_conjunctions(self):
        from nllw.alignatt import _EN_FUNCTION_WORDS
        for word in ["and", "or", "but"]:
            self.assertIn(word, _EN_FUNCTION_WORDS, f"Missing conjunction: {word}")

    def test_content_words_not_included(self):
        from nllw.alignatt import _EN_FUNCTION_WORDS
        for word in ["president", "announced", "France", "policy", "reform",
                      "economic", "government", "change", "new"]:
            self.assertNotIn(word, _EN_FUNCTION_WORDS, f"Content word included: {word}")


if __name__ == "__main__":
    unittest.main()
