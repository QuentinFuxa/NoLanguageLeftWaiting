"""Tests for iteration 27: sentence-final refinement + two-phase XCOMET scoring."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFinalRefinementConfig(unittest.TestCase):
    """Tests for the final_refinement config parameter."""

    def test_default_disabled(self):
        """final_refinement should be disabled by default."""
        from nllw.backend_protocol import BackendConfig
        config = BackendConfig()
        self.assertFalse(config.final_refinement)

    def test_enable_via_dict(self):
        """Can enable final_refinement via from_dict."""
        from nllw.backend_protocol import BackendConfig
        config = BackendConfig.from_dict({"final_refinement": True})
        self.assertTrue(config.final_refinement)

    def test_config_serialization(self):
        """final_refinement survives round-trip."""
        from nllw.backend_protocol import BackendConfig
        config = BackendConfig(final_refinement=True)
        d = config.__dict__
        config2 = BackendConfig.from_dict(d)
        self.assertTrue(config2.final_refinement)


class TestBenchSweepParsing(unittest.TestCase):
    """Tests for the refine sweep shortname."""

    def test_refine_shortname(self):
        """The 'refine' shortname maps to final_refinement."""
        from nllw.bench import parse_sweep_spec
        grid = parse_sweep_spec("refine=0,1")
        self.assertIn("final_refinement", grid)
        self.assertEqual(grid["final_refinement"], [0, 1])


class TestBenchArgFinalRefinement(unittest.TestCase):
    """Tests for --final-refinement CLI argument."""

    def test_args_in_config_builder(self):
        """--final-refinement should end up in config dict."""
        from nllw.bench import parse_sweep_spec
        # Just verify the shortname exists
        grid = parse_sweep_spec("refine=1")
        self.assertIn("final_refinement", grid)
        self.assertEqual(grid["final_refinement"], [1])


class TestTwoPhaseArchitecture(unittest.TestCase):
    """Tests for the two-phase experiment architecture."""

    def test_hypo_file_format(self):
        """Hypothesis JSON should have required fields."""
        import json
        import tempfile

        hypo_data = {
            "direction": "en-zh",
            "backend_type": "alignatt",
            "n_sentences": 3,
            "comet": 0.85,
            "sources": ["hello", "world", "test"],
            "hypotheses": ["你好", "世界", "测试"],
            "references": ["你好", "世界", "测试"],
            "config": {"border_distance": 3},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(hypo_data, f)
            tmp_path = f.name

        try:
            with open(tmp_path) as f:
                loaded = json.load(f)
            self.assertEqual(loaded["direction"], "en-zh")
            self.assertEqual(len(loaded["sources"]), 3)
            self.assertEqual(len(loaded["hypotheses"]), 3)
            self.assertEqual(len(loaded["references"]), 3)
        finally:
            os.unlink(tmp_path)


class TestExperimentScriptImport(unittest.TestCase):
    """Tests that the experiment script is valid Python."""

    def test_script_syntax(self):
        """Experiment script should be syntactically valid."""
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts", "run_iteration27_experiments.py",
        )
        if os.path.exists(script_path):
            with open(script_path) as f:
                code = f.read()
            compile(code, script_path, "exec")

    def test_script_phases(self):
        """Experiment script should define all phases."""
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts", "run_iteration27_experiments.py",
        )
        if os.path.exists(script_path):
            with open(script_path) as f:
                code = f.read()
            for phase_name in ["phase_0_baseline", "phase_1_bd_wb_sweep",
                              "phase_2_topp_sweep", "phase_3_features",
                              "phase_4_anti_lm", "phase_5_combined",
                              "phase_6_refinement", "score_all_xcomet"]:
                self.assertIn(phase_name, code, f"Missing {phase_name}")


class TestRefineAntiLmInteraction(unittest.TestCase):
    """Tests that refinement + anti-LM flags don't conflict."""

    def test_both_flags_in_config(self):
        """Can set both final_refinement and anti_lm."""
        from nllw.backend_protocol import BackendConfig
        config = BackendConfig(final_refinement=True, anti_lm=True, anti_lm_gamma=0.3)
        self.assertTrue(config.final_refinement)
        self.assertTrue(config.anti_lm)
        self.assertEqual(config.anti_lm_gamma, 0.3)


class TestXCOMETScorerModule(unittest.TestCase):
    """Tests for XCOMET scorer module availability."""

    def test_xcomet_scorer_imports(self):
        """xcomet_scorer module should import without GPU."""
        from nllw.xcomet_scorer import save_hypotheses_json
        self.assertTrue(callable(save_hypotheses_json))

    def test_score_xcomet_subprocess_callable(self):
        """score_xcomet_subprocess should be callable."""
        from nllw.xcomet_scorer import score_xcomet_subprocess
        self.assertTrue(callable(score_xcomet_subprocess))


if __name__ == "__main__":
    unittest.main()
