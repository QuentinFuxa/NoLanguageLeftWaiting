"""Tests for iteration 23: Confidence-adaptive word batching, language-pair gen_cap."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestConfidenceAdaptiveWordBatch(unittest.TestCase):
    """Tests for confidence_adaptive_word_batch() function."""

    def test_none_logprob_no_change(self):
        """No previous logprob -> no adjustment."""
        from nllw.alignatt import confidence_adaptive_word_batch
        self.assertEqual(confidence_adaptive_word_batch(3, None), 3)
        self.assertEqual(confidence_adaptive_word_batch(4, None), 4)
        self.assertEqual(confidence_adaptive_word_batch(1, None), 1)

    def test_confident_reduces_wb(self):
        """High logprob (confident) -> reduce wb by 1."""
        from nllw.alignatt import confidence_adaptive_word_batch
        # -0.3 is above default high threshold of -0.5
        self.assertEqual(confidence_adaptive_word_batch(3, -0.3), 2)
        self.assertEqual(confidence_adaptive_word_batch(4, -0.1), 3)
        self.assertEqual(confidence_adaptive_word_batch(5, -0.4), 4)

    def test_uncertain_increases_wb(self):
        """Low logprob (uncertain) -> increase wb by 1."""
        from nllw.alignatt import confidence_adaptive_word_batch
        # -3.0 is below default low threshold of -2.0
        self.assertEqual(confidence_adaptive_word_batch(3, -3.0), 4)
        self.assertEqual(confidence_adaptive_word_batch(2, -5.0), 3)
        self.assertEqual(confidence_adaptive_word_batch(4, -2.5), 5)

    def test_moderate_no_change(self):
        """Moderate logprob -> no adjustment."""
        from nllw.alignatt import confidence_adaptive_word_batch
        # -1.0 is between -0.5 and -2.0
        self.assertEqual(confidence_adaptive_word_batch(3, -1.0), 3)
        self.assertEqual(confidence_adaptive_word_batch(4, -1.5), 4)
        self.assertEqual(confidence_adaptive_word_batch(2, -0.6), 2)

    def test_minimum_wb_clamp(self):
        """wb never goes below 1."""
        from nllw.alignatt import confidence_adaptive_word_batch
        self.assertEqual(confidence_adaptive_word_batch(1, -0.1), 1)
        # wb=1, confident -> max(1, 1-1) = max(1, 0) = 1
        self.assertEqual(confidence_adaptive_word_batch(1, -0.3), 1)

    def test_custom_thresholds(self):
        """Custom thresholds work correctly."""
        from nllw.alignatt import confidence_adaptive_word_batch
        # Tight thresholds: only very confident/uncertain triggers
        result = confidence_adaptive_word_batch(
            3, -0.1, high_threshold=-0.2, low_threshold=-4.0
        )
        self.assertEqual(result, 2)  # -0.1 > -0.2, confident

        result = confidence_adaptive_word_batch(
            3, -0.3, high_threshold=-0.2, low_threshold=-4.0
        )
        self.assertEqual(result, 3)  # -0.3 < -0.2 but > -4.0, moderate

        result = confidence_adaptive_word_batch(
            3, -5.0, high_threshold=-0.2, low_threshold=-4.0
        )
        self.assertEqual(result, 4)  # -5.0 < -4.0, uncertain

    def test_boundary_values(self):
        """Exact threshold values are treated as moderate (no change)."""
        from nllw.alignatt import confidence_adaptive_word_batch
        # At exactly the threshold: NOT above/below, so no change
        self.assertEqual(confidence_adaptive_word_batch(3, -0.5), 3)
        self.assertEqual(confidence_adaptive_word_batch(3, -2.0), 3)


class TestConfidenceAdaptiveConfig(unittest.TestCase):
    """Tests for confidence_adaptive_wb config field."""

    def test_config_field_exists(self):
        """BackendConfig has confidence_adaptive_wb field."""
        from nllw.backend_protocol import BackendConfig
        config = BackendConfig()
        self.assertFalse(config.confidence_adaptive_wb)
        self.assertEqual(config.confidence_wb_high, -0.5)
        self.assertEqual(config.confidence_wb_low, -2.0)

    def test_config_enable(self):
        """Can enable confidence_adaptive_wb."""
        from nllw.backend_protocol import BackendConfig
        config = BackendConfig(confidence_adaptive_wb=True)
        self.assertTrue(config.confidence_adaptive_wb)

    def test_config_custom_thresholds(self):
        """Can set custom thresholds."""
        from nllw.backend_protocol import BackendConfig
        config = BackendConfig(
            confidence_adaptive_wb=True,
            confidence_wb_high=-0.3,
            confidence_wb_low=-3.0,
        )
        self.assertEqual(config.confidence_wb_high, -0.3)
        self.assertEqual(config.confidence_wb_low, -3.0)

    def test_config_from_dict(self):
        """BackendConfig.from_dict handles confidence fields."""
        from nllw.backend_protocol import BackendConfig
        config = BackendConfig.from_dict({
            "confidence_adaptive_wb": True,
            "confidence_wb_high": -0.4,
            "confidence_wb_low": -2.5,
        })
        self.assertTrue(config.confidence_adaptive_wb)
        self.assertEqual(config.confidence_wb_high, -0.4)
        self.assertEqual(config.confidence_wb_low, -2.5)


class TestConfidenceAdaptiveSweep(unittest.TestCase):
    """Tests for sweep shortnames."""

    def test_sweep_shortname_confwb(self):
        """Sweep parser recognizes confwb shortname."""
        from nllw.bench import parse_sweep_spec
        grid = parse_sweep_spec("confwb=0,1")
        self.assertIn("confidence_adaptive_wb", grid)
        self.assertEqual(grid["confidence_adaptive_wb"], [0, 1])

    def test_sweep_shortname_thresholds(self):
        """Sweep parser recognizes confhi and conflo shortnames."""
        from nllw.bench import parse_sweep_spec
        grid = parse_sweep_spec("confhi=-0.3,-0.5,-0.7")
        self.assertIn("confidence_wb_high", grid)
        self.assertEqual(grid["confidence_wb_high"], [-0.3, -0.5, -0.7])


class TestConfidenceAdaptiveIntegration(unittest.TestCase):
    """Integration tests for confidence-adaptive wb in backends."""

    def test_alignatt_backend_has_state(self):
        """AlignAttBackend tracks _prev_avg_logprob."""
        # Can't instantiate without model, but verify import works
        from nllw.alignatt import confidence_adaptive_word_batch
        # Verify the function is importable from alignatt_backend
        from nllw.alignatt_backend import AlignAttBackend
        # Check the class has the right imports
        import inspect
        source = inspect.getsource(AlignAttBackend.__init__)
        self.assertIn("_prev_avg_logprob", source)

    def test_la_backend_has_state(self):
        """AlignAttLABackend tracks _prev_avg_logprob."""
        from nllw.alignatt_la_backend import AlignAttLABackend
        import inspect
        source = inspect.getsource(AlignAttLABackend.__init__)
        self.assertIn("_prev_avg_logprob", source)

    def test_alignatt_backend_resets_on_segment_end(self):
        """_prev_avg_logprob is reset in _handle_segment_end."""
        from nllw.alignatt_backend import AlignAttBackend
        import inspect
        source = inspect.getsource(AlignAttBackend._handle_segment_end)
        self.assertIn("_prev_avg_logprob", source)

    def test_la_backend_resets_on_segment_end(self):
        """_prev_avg_logprob is reset in LA _handle_segment_end."""
        from nllw.alignatt_la_backend import AlignAttLABackend
        import inspect
        source = inspect.getsource(AlignAttLABackend._handle_segment_end)
        self.assertIn("_prev_avg_logprob", source)


class TestLanguagePairGenCap(unittest.TestCase):
    """Tests for language-pair-aware generation cap ratios."""

    def test_gen_cap_ratios_exist(self):
        """Generation cap ratios for known language pairs exist."""
        from nllw.alignatt import language_pair_gen_cap
        # EN-ZH: Chinese is typically shorter
        cap_zh = language_pair_gen_cap(10, "en", "zh")
        # EN-DE: German is typically longer
        cap_de = language_pair_gen_cap(10, "en", "de")
        # ZH should get fewer tokens than DE
        self.assertLessEqual(cap_zh, cap_de)

    def test_gen_cap_minimum(self):
        """Generation cap never goes below minimum."""
        from nllw.alignatt import language_pair_gen_cap
        cap = language_pair_gen_cap(1, "en", "zh")
        self.assertGreaterEqual(cap, 3)

    def test_gen_cap_unknown_pair(self):
        """Unknown language pairs get default 1.0 ratio (with safety margin)."""
        from nllw.alignatt import language_pair_gen_cap
        cap = language_pair_gen_cap(10, "en", "xx")
        # Ratio 1.0 * 1.3 safety = 13
        self.assertEqual(cap, 13)

    def test_gen_cap_scales_with_source(self):
        """Generation cap scales with number of source tokens."""
        from nllw.alignatt import language_pair_gen_cap
        cap_5 = language_pair_gen_cap(5, "en", "de")
        cap_20 = language_pair_gen_cap(20, "en", "de")
        self.assertGreater(cap_20, cap_5)

    def test_gen_cap_en_it(self):
        """EN-IT ratio is close to 1:1."""
        from nllw.alignatt import language_pair_gen_cap
        cap = language_pair_gen_cap(10, "en", "it")
        # Italian is similar length to English, with 1.3x safety margin
        self.assertGreaterEqual(cap, 9)
        self.assertLessEqual(cap, 15)

    def test_gen_cap_cs_en(self):
        """CS-EN ratio works correctly."""
        from nllw.alignatt import language_pair_gen_cap
        cap = language_pair_gen_cap(10, "cs", "en")
        self.assertGreaterEqual(cap, 9)
        self.assertLessEqual(cap, 14)


class TestCompetitionValidation(unittest.TestCase):
    """Competition-readiness checks for new features."""

    def test_confidence_wb_disabled_by_default(self):
        """Confidence-adaptive wb is disabled by default (conservative)."""
        from nllw.backend_protocol import BackendConfig
        config = BackendConfig()
        self.assertFalse(config.confidence_adaptive_wb)

    def test_iwslt_configs_unchanged(self):
        """IWSLT competition configs don't enable confidence_adaptive_wb yet."""
        import yaml
        configs_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs"
        )
        for fname in os.listdir(configs_dir):
            if fname.startswith("iwslt2026-") and fname.endswith(".yaml"):
                with open(os.path.join(configs_dir, fname)) as f:
                    data = yaml.safe_load(f)
                # New features not in production configs until GPU-validated
                self.assertNotIn("confidence_adaptive_wb", data,
                                 f"{fname} should not have confidence_adaptive_wb yet")


if __name__ == "__main__":
    unittest.main()
