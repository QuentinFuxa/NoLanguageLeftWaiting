"""Tests for Iteration 11 features: n-gram repetition detection + SimulStream wrapper.

1. N-gram repetition detection (novel):
   Detect degenerate repetitive loops during generation and force halt.
   When LLMs hallucinate, they often enter repetitive patterns producing
   the same n-gram repeatedly. This is a within-step, output-space signal --
   orthogonal to all attention-based signals. It measures OUTPUT quality
   directly rather than inferring quality from attention patterns.

   Novel application: no published work on n-gram repetition detection
   as a border/halt signal in simultaneous MT with decoder-only LLMs.

2. SimulStream wrapper:
   Wraps NLLW SimulMT backend as a SimulStream SpeechProcessor for
   IWSLT 2026 submission. Tests the wrapper's text-mode path, config
   handling, and output format without requiring llama.cpp.

All tests are unit tests that don't require llama.cpp.
"""

import json
import os
import tempfile

import numpy as np
import pytest

from nllw.alignatt import (
    compute_attention_shift,
    attention_shift_supports_write,
    check_border_combined,
    detect_ngram_repetition,
    compute_repetition_score,
)
from nllw.backend_protocol import BackendConfig
from nllw.bench import parse_sweep_spec
from nllw.simulstream import (
    IncrementalOutput,
    SimulStreamConfig,
    NLLWSpeechProcessor,
    DIRECTION_DEFAULTS,
)


# ===========================================================================
# detect_ngram_repetition tests
# ===========================================================================

class TestDetectNgramRepetition:
    """Tests for the core repetition detection function."""

    def test_no_repetition(self):
        """Unique tokens should not trigger repetition."""
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert detect_ngram_repetition(tokens) is False

    def test_bigram_repetition(self):
        """Repeating the same bigram should be detected."""
        # [1, 2] repeated 3 times -> above default max_repeats=2
        tokens = [1, 2, 1, 2, 1, 2]
        assert detect_ngram_repetition(tokens) is True

    def test_trigram_repetition(self):
        """Repeating the same trigram should be detected."""
        # [3, 4, 5] repeated 3 times
        tokens = [3, 4, 5, 3, 4, 5, 3, 4, 5]
        assert detect_ngram_repetition(tokens) is True

    def test_fourgram_repetition(self):
        """Repeating the same 4-gram should be detected."""
        # [10, 20, 30, 40] repeated 3 times
        tokens = [10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30, 40]
        assert detect_ngram_repetition(tokens) is True

    def test_exactly_at_threshold(self):
        """Exactly max_repeats occurrences should NOT trigger (needs >)."""
        # [1, 2] appears exactly 2 times with default max_repeats=2
        tokens = [1, 2, 1, 2]
        assert detect_ngram_repetition(tokens) is False

    def test_one_above_threshold(self):
        """One above max_repeats should trigger."""
        # [1, 2] appears 3 times with default max_repeats=2
        tokens = [1, 2, 1, 2, 1, 2]
        assert detect_ngram_repetition(tokens) is True

    def test_custom_max_repeats(self):
        """Custom max_repeats threshold."""
        tokens = [1, 2, 1, 2, 1, 2, 1, 2]  # 4 occurrences of [1,2]
        assert detect_ngram_repetition(tokens, max_repeats=3) is True
        assert detect_ngram_repetition(tokens, max_repeats=4) is False

    def test_short_sequence_no_false_positive(self):
        """Very short sequences should never trigger."""
        assert detect_ngram_repetition([1]) is False
        assert detect_ngram_repetition([1, 2]) is False
        assert detect_ngram_repetition([1, 2, 3]) is False
        assert detect_ngram_repetition([]) is False

    def test_single_token_repetition(self):
        """Single token repeated (n=1 is below default min_n=2)."""
        tokens = [5, 5, 5, 5, 5]
        # With default min_n=2, single token repeat not checked
        assert detect_ngram_repetition(tokens) is False
        # But with min_n=1, should detect
        # Actually min_n=1 means unigrams -- let's check:
        # The token [5] appears 5 times, well above max_repeats=2
        # However our function checks min_n=2 by default

    def test_custom_min_n(self):
        """Custom min_n to detect single token repetition."""
        tokens = [5, 5, 5, 5, 5]
        # With min_n=1, should detect the repeated unigram
        assert detect_ngram_repetition(tokens, min_n=1) is True

    def test_custom_max_n(self):
        """Custom max_n to only check larger patterns."""
        # [1, 2] repeats 3 times but we only check n >= 3
        tokens = [1, 2, 1, 2, 1, 2]
        assert detect_ngram_repetition(tokens, min_n=3, max_n=4) is False

    def test_realistic_hallucination_pattern(self):
        """Simulate a realistic hallucination: token 42 = comma, cycling."""
        # Model outputs: "the, the, the," pattern (token IDs)
        # the=100, comma=42
        tokens = [100, 42, 100, 42, 100, 42, 100, 42]
        assert detect_ngram_repetition(tokens) is True

    def test_normal_translation_no_trigger(self):
        """Normal diverse translation output should not trigger."""
        # Simulated: "Le president de la France a annonce des reformes"
        tokens = [101, 234, 56, 78, 345, 12, 567, 89, 123]
        assert detect_ngram_repetition(tokens) is False

    def test_partial_overlap_not_repetition(self):
        """Partially overlapping patterns should not trigger."""
        # [1, 2, 3, 2, 3, 4] -- [2, 3] appears twice but not 3 times
        tokens = [1, 2, 3, 2, 3, 4]
        assert detect_ngram_repetition(tokens) is False

    def test_late_onset_repetition(self):
        """Repetition starting after normal generation should be caught."""
        # Normal start, then degenerate
        tokens = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                  5, 6, 5, 6, 5, 6]
        assert detect_ngram_repetition(tokens) is True

    def test_max_n_bounded_by_sequence_length(self):
        """max_n should be bounded by half the sequence length."""
        # With only 6 tokens, max_n=4 is effectively limited to 3
        tokens = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        assert detect_ngram_repetition(tokens) is True

    def test_all_same_tokens(self):
        """All identical tokens -- degenerate case."""
        tokens = [42, 42, 42, 42, 42, 42, 42, 42]
        # Bigram [42, 42] appears 7 times -> detected
        assert detect_ngram_repetition(tokens) is True

    def test_empty_sequence(self):
        """Empty sequence should return False."""
        assert detect_ngram_repetition([]) is False


# ===========================================================================
# compute_repetition_score tests
# ===========================================================================

class TestComputeRepetitionScore:
    """Tests for the continuous repetition score."""

    def test_no_repetition_score_zero(self):
        """Fully unique bigrams should give score near 0."""
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        score = compute_repetition_score(tokens)
        assert score < 0.2  # Low repetition

    def test_full_repetition_score_high(self):
        """Fully repetitive sequence should give high score."""
        tokens = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
        score = compute_repetition_score(tokens)
        assert score > 0.7  # High repetition

    def test_all_same_token_max_score(self):
        """All same tokens should give maximum repetition score."""
        tokens = [42] * 20
        score = compute_repetition_score(tokens)
        assert score > 0.9  # Near-maximum

    def test_short_sequence_returns_zero(self):
        """Very short sequences should return 0."""
        assert compute_repetition_score([]) == 0.0
        assert compute_repetition_score([1]) == 0.0
        assert compute_repetition_score([1, 2]) == 0.0

    def test_score_range(self):
        """Score should always be in [0, 1]."""
        for tokens in [
            [1, 2, 3, 4, 5],
            [1, 2, 1, 2, 1, 2],
            [42] * 30,
            list(range(50)),
        ]:
            score = compute_repetition_score(tokens)
            assert 0.0 <= score <= 1.0

    def test_window_parameter(self):
        """Window parameter should limit lookback."""
        # Long unique prefix, short repetitive suffix
        tokens = list(range(50)) + [1, 2, 1, 2, 1, 2, 1, 2]
        # With small window, should see the repetition
        score_small = compute_repetition_score(tokens, window=8)
        # With large window, repetition diluted by unique prefix
        score_large = compute_repetition_score(tokens, window=50)
        assert score_small > score_large

    def test_medium_repetition(self):
        """Moderate repetition should give medium score."""
        # Some repetition but not degenerate
        tokens = [1, 2, 3, 4, 1, 2, 5, 6, 3, 4, 7, 8]
        score = compute_repetition_score(tokens)
        assert 0.1 < score < 0.7

    def test_three_tokens_minimum(self):
        """Exactly 3 tokens should work (minimum for bigrams)."""
        score = compute_repetition_score([1, 2, 3])
        assert score == 0.0  # All unique bigrams: (1,2), (2,3)


# ===========================================================================
# BackendConfig integration tests
# ===========================================================================

class TestRepetitionBackendConfig:
    """Tests that repetition detection is properly integrated into BackendConfig."""

    def test_default_disabled(self):
        """Repetition detection should be disabled by default."""
        config = BackendConfig()
        assert config.repetition_max_repeats is None

    def test_from_dict(self):
        """Should accept repetition_max_repeats from dict."""
        config = BackendConfig.from_dict({"repetition_max_repeats": 2})
        assert config.repetition_max_repeats == 2

    def test_from_dict_ignores_unknown(self):
        """Unknown keys should be ignored (existing behavior)."""
        config = BackendConfig.from_dict({
            "repetition_max_repeats": 3,
            "unknown_key": "value",
        })
        assert config.repetition_max_repeats == 3


# ===========================================================================
# Sweep parser integration tests
# ===========================================================================

class TestRepetitionSweepShortname:
    """Tests that sweep shortname 'rep' works correctly."""

    def test_rep_shortname_maps_correctly(self):
        """'rep' should map to 'repetition_max_repeats'."""
        grid = parse_sweep_spec("rep=2,3,4")
        assert "repetition_max_repeats" in grid
        assert grid["repetition_max_repeats"] == [2, 3, 4]

    def test_rep_in_combo_sweep(self):
        """'rep' should work in combination with other params."""
        grid = parse_sweep_spec("rep=2 bd=3,4")
        assert "repetition_max_repeats" in grid
        assert "border_distance" in grid

    def test_rep_with_none_value(self):
        """Test sweep with 0 (disabled) and active values."""
        grid = parse_sweep_spec("rep=0,2,3")
        assert grid["repetition_max_repeats"] == [0, 2, 3]


# ===========================================================================
# Edge case tests
# ===========================================================================

class TestRepetitionEdgeCases:
    """Edge case tests for robustness."""

    def test_very_long_sequence(self):
        """Should handle long sequences efficiently."""
        # 1000 unique tokens -- should not trigger
        tokens = list(range(1000))
        assert detect_ngram_repetition(tokens) is False

    def test_long_repetitive_sequence(self):
        """Should detect repetition in long sequences."""
        # 500 repetitions of [1, 2]
        tokens = [1, 2] * 500
        assert detect_ngram_repetition(tokens) is True

    def test_repetition_at_exact_boundary(self):
        """Test behavior at the exact min length boundary."""
        # min_n=2, max_repeats=2 -> need at least 2*(2+1)=6 tokens
        tokens_short = [1, 2, 1, 2, 1]  # 5 tokens
        tokens_exact = [1, 2, 1, 2, 1, 2]  # 6 tokens, 3 occurrences
        assert detect_ngram_repetition(tokens_short) is False
        assert detect_ngram_repetition(tokens_exact) is True

    def test_negative_tokens(self):
        """Should handle negative token IDs correctly."""
        tokens = [-1, -2, -1, -2, -1, -2]
        assert detect_ngram_repetition(tokens) is True

    def test_large_token_ids(self):
        """Should handle large token IDs (vocab size ~150k)."""
        tokens = [151643, 151644, 151643, 151644, 151643, 151644]
        assert detect_ngram_repetition(tokens) is True

    def test_mixed_repetition_patterns(self):
        """Multiple different patterns, but none exceeding threshold."""
        # [1,2] appears twice, [3,4] appears twice -- neither exceeds max_repeats=2
        tokens = [1, 2, 3, 4, 1, 2, 3, 4]
        # Actually [1,2] appears 2 times, [3,4] appears 2 times
        # default max_repeats=2 means > 2 needed -> should NOT trigger
        assert detect_ngram_repetition(tokens) is False

    def test_overlapping_bigrams(self):
        """Test that overlapping bigram occurrences are counted correctly."""
        # [1, 1] appears 5 times in [1, 1, 1, 1, 1, 1]
        tokens = [1, 1, 1, 1, 1, 1]
        assert detect_ngram_repetition(tokens, min_n=2) is True


# ===========================================================================
# Integration: repetition with other signals
# ===========================================================================

class TestRepetitionSignalTaxonomy:
    """Verify that repetition detection fits into the signal taxonomy."""

    def test_orthogonal_to_attention_signals(self):
        """Repetition detection should be independent of attention data."""
        # It only uses token IDs, not attention weights
        tokens_repetitive = [1, 2, 1, 2, 1, 2, 1, 2]
        tokens_normal = [1, 2, 3, 4, 5, 6, 7, 8]
        assert detect_ngram_repetition(tokens_repetitive) is True
        assert detect_ngram_repetition(tokens_normal) is False
        # No attention weights needed -- purely output-space

    def test_score_monotonicity(self):
        """Adding more repetitions should increase the score."""
        base = [1, 2, 3, 4, 5]
        scores = []
        for extra_reps in range(5):
            tokens = base + [10, 20] * (extra_reps + 1)
            scores.append(compute_repetition_score(tokens))
        # Score should generally increase with more repetition
        # (not strictly monotonic due to windowing, but trend should hold)
        assert scores[-1] > scores[0]


# ===========================================================================
# SimulStream IncrementalOutput tests
# ===========================================================================

class TestIncrementalOutput:
    """Tests for the IncrementalOutput dataclass."""

    def test_empty_output(self):
        """Default output should be empty."""
        output = IncrementalOutput()
        assert output.is_empty
        assert output.new_string == ""
        assert output.new_tokens == []

    def test_non_empty_output(self):
        """Output with text should not be empty."""
        output = IncrementalOutput(new_string="hello", new_tokens=["hello"])
        assert not output.is_empty

    def test_deleted_tokens(self):
        """Output with only deletions should not be empty."""
        output = IncrementalOutput(deleted_string="old", deleted_tokens=["old"])
        assert not output.is_empty

    def test_fields_are_independent(self):
        """All fields can be set independently."""
        output = IncrementalOutput(
            new_tokens=["a", "b"],
            new_string="a b",
            deleted_tokens=["c"],
            deleted_string="c",
        )
        assert len(output.new_tokens) == 2
        assert output.new_string == "a b"
        assert len(output.deleted_tokens) == 1


# ===========================================================================
# SimulStreamConfig tests
# ===========================================================================

class TestSimulStreamConfig:
    """Tests for SimulStream configuration."""

    def test_default_config(self):
        """Default config should have sensible defaults."""
        config = SimulStreamConfig()
        assert config.direction == "en-zh"
        assert config.backend_type == "alignatt"
        assert config.sample_rate == 16000
        assert config.min_start_seconds == 0.5

    def test_to_backend_config(self):
        """Config should convert to BackendConfig correctly."""
        config = SimulStreamConfig(
            model_path="/path/to/model.gguf",
            direction="en-de",
            border_distance=4,
            word_batch=2,
        )
        bc = config.to_backend_config()
        assert bc.model_path == "/path/to/model.gguf"
        assert bc.direction == "en-de"
        assert bc.border_distance == 4
        assert bc.word_batch == 2
        assert bc.target_lang == "de"

    def test_to_backend_config_with_extras(self):
        """Extra backend params should be forwarded."""
        config = SimulStreamConfig(
            model_path="/model.gguf",
            extra_backend_params={
                "shift_k_threshold": 0.4,
                "lsg_kl_threshold": 7.0,
            },
        )
        bc = config.to_backend_config()
        assert bc.shift_k_threshold == 0.4
        assert bc.lsg_kl_threshold == 7.0

    def test_target_lang_extraction(self):
        """Target language should be extracted from direction."""
        for direction, expected in [
            ("en-zh", "zh"),
            ("en-de", "de"),
            ("en-it", "it"),
            ("cs-en", "en"),
        ]:
            config = SimulStreamConfig(direction=direction)
            bc = config.to_backend_config()
            assert bc.target_lang == expected


# ===========================================================================
# Direction defaults tests
# ===========================================================================

class TestDirectionDefaults:
    """Tests for per-direction default configurations."""

    def test_all_directions_present(self):
        """All 4 IWSLT 2026 directions should have defaults."""
        for d in ["en-zh", "en-de", "en-it", "cs-en"]:
            assert d in DIRECTION_DEFAULTS

    def test_en_zh_config(self):
        """EN-ZH should use bd=3, wb=4, top_p (iteration 16 best: COMET=0.895)."""
        cfg = DIRECTION_DEFAULTS["en-zh"]
        assert cfg["border_distance"] == 3
        assert cfg["word_batch"] == 4
        assert cfg["aggregation"] == "top_p"

    def test_cs_en_config(self):
        """CS-EN should use bd=3, wb=3, top_p (iteration 16 best: COMET=0.876)."""
        cfg = DIRECTION_DEFAULTS["cs-en"]
        assert cfg["border_distance"] == 3
        assert cfg["word_batch"] == 3
        assert cfg["aggregation"] == "top_p"

    def test_en_de_config(self):
        """EN-DE should use bd=2, wb=3, top_p (iteration 16 best: COMET=0.881)."""
        cfg = DIRECTION_DEFAULTS["en-de"]
        assert cfg["border_distance"] == 2
        assert cfg["word_batch"] == 3
        assert cfg["aggregation"] == "top_p"

    def test_en_it_config(self):
        """EN-IT should use bd=2, wb=3, top_p (iteration 16 best: COMET=0.884)."""
        cfg = DIRECTION_DEFAULTS["en-it"]
        assert cfg["border_distance"] == 2
        assert cfg["word_batch"] == 3
        assert cfg["aggregation"] == "top_p"

    def test_all_use_hymt(self):
        """All directions should use HY-MT prompt format."""
        for d, cfg in DIRECTION_DEFAULTS.items():
            assert cfg["prompt_format"] == "hymt"


# ===========================================================================
# NLLWSpeechProcessor text-mode tests (no GPU needed)
# ===========================================================================

class TestNLLWSpeechProcessorConfig:
    """Tests for processor configuration and initialization.

    These tests don't require a GPU -- they only test config handling.
    """

    def test_config_creation(self):
        """Processor config should be creatable without GPU."""
        config = SimulStreamConfig(
            model_path="/nonexistent/model.gguf",
            direction="en-zh",
        )
        # Just test config creation, not initialization
        assert config.model_path == "/nonexistent/model.gguf"

    def test_stats_empty(self):
        """Stats should be zero before any processing."""
        config = SimulStreamConfig()
        proc = NLLWSpeechProcessor(config)
        stats = proc.stats
        assert stats["chunks_processed"] == 0
        assert stats["total_audio_seconds"] == 0.0
        assert stats["committed_text"] == ""

    def test_clear_resets_state(self):
        """Clear should reset all internal state."""
        config = SimulStreamConfig()
        proc = NLLWSpeechProcessor(config)
        proc._n_chunks_processed = 10
        proc._total_audio_seconds = 5.0
        proc._committed_text = "test"
        proc.clear()
        assert proc._n_chunks_processed == 0
        assert proc._total_audio_seconds == 0.0
        assert proc._committed_text == ""

    def test_close_without_init(self):
        """Close should be safe to call without initialization."""
        config = SimulStreamConfig()
        proc = NLLWSpeechProcessor(config)
        proc.close()  # Should not raise

    def test_end_of_stream_without_init(self):
        """end_of_stream should return empty without initialization."""
        config = SimulStreamConfig()
        proc = NLLWSpeechProcessor(config)
        output = proc.end_of_stream()
        assert output.is_empty

    def test_set_source_language(self):
        """set_source_language should update direction."""
        config = SimulStreamConfig(direction="en-zh")
        proc = NLLWSpeechProcessor(config)
        proc.set_source_language("cs")
        assert proc._source_lang == "cs"
        assert proc.config.direction == "cs-zh"

    def test_set_target_language(self):
        """set_target_language should update direction and apply defaults."""
        config = SimulStreamConfig(direction="en-zh")
        proc = NLLWSpeechProcessor(config)
        proc.set_target_language("de")
        assert proc._target_lang == "de"
        assert proc.config.direction == "en-de"
        # Should apply direction defaults (iteration 16 best: bd=2, wb=3, top_p)
        assert proc.config.border_distance == 2
        assert proc.config.word_batch == 3
        assert proc.config.aggregation == "top_p"

    def test_set_target_language_it(self):
        """EN-IT should use bd=2, wb=3, top_p from direction defaults."""
        config = SimulStreamConfig(direction="en-zh")
        proc = NLLWSpeechProcessor(config)
        proc.set_target_language("it")
        assert proc.config.border_distance == 2
        assert proc.config.word_batch == 3
        assert proc.config.aggregation == "top_p"

    def test_tokens_to_string(self):
        """tokens_to_string should concatenate tokens."""
        config = SimulStreamConfig()
        proc = NLLWSpeechProcessor(config)
        assert proc.tokens_to_string(["hello", " ", "world"]) == "hello world"
        assert proc.tokens_to_string([]) == ""

    def test_speech_chunk_size(self):
        """speech_chunk_size should return seconds."""
        config = SimulStreamConfig(speech_chunk_size=960, sample_rate=16000)
        proc = NLLWSpeechProcessor(config)
        assert proc.speech_chunk_size == pytest.approx(0.06, abs=0.001)

    def test_audio_buffer_accumulation(self):
        """Audio chunks should accumulate in the buffer."""
        config = SimulStreamConfig(min_start_seconds=1.0, sample_rate=16000)
        proc = NLLWSpeechProcessor(config)
        # Manually simulate buffering without triggering backend init
        chunk = np.zeros(960, dtype=np.float32)
        proc._audio_buffer = np.concatenate([proc._audio_buffer, chunk])
        proc._total_audio_seconds = len(proc._audio_buffer) / config.sample_rate
        assert proc._total_audio_seconds == pytest.approx(0.06, abs=0.01)
        # Below min_start_seconds, should still be buffering
        assert proc._total_audio_seconds < config.min_start_seconds


# ===========================================================================
# YAML config tests
# ===========================================================================

class TestYAMLConfig:
    """Tests for YAML configuration loading."""

    def test_from_yaml(self):
        """Config should load from YAML file."""
        yaml_content = """
model_path: /path/to/model.gguf
direction: en-de
border_distance: 4
word_batch: 3
shift_k_threshold: 0.4
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                config = SimulStreamConfig.from_yaml(f.name)
                assert config.model_path == "/path/to/model.gguf"
                assert config.direction == "en-de"
                assert config.border_distance == 4
                assert config.word_batch == 3
                # Extra params should be captured
                assert config.extra_backend_params["shift_k_threshold"] == 0.4
            finally:
                os.unlink(f.name)


# ===========================================================================
# Attention shift tracking tests
# ===========================================================================

class TestComputeAttentionShift:
    """Tests for cross-step attention shift computation."""

    def test_no_previous_returns_none(self):
        """First call should return None (no previous attention)."""
        attn = np.array([[0.1, 0.2, 0.3, 0.4]])
        result = compute_attention_shift(attn, None, [1.0])
        assert result is None

    def test_forward_shift_detected(self):
        """Attention moving forward should give positive shift."""
        # Previous: attending to position 1
        prev = np.zeros((2, 5))
        prev[0, 1] = 1.0
        prev[1, 1] = 1.0
        # Current: attending to position 3
        curr = np.zeros((2, 5))
        curr[0, 3] = 1.0
        curr[1, 3] = 1.0
        ts = [1.0, 1.0]
        shift = compute_attention_shift(curr, prev, ts)
        assert shift is not None
        assert shift > 0  # Forward shift

    def test_no_shift_detected(self):
        """Same attention should give zero shift."""
        attn = np.zeros((2, 5))
        attn[0, 2] = 1.0
        attn[1, 2] = 1.0
        ts = [1.0, 1.0]
        shift = compute_attention_shift(attn, attn, ts)
        assert shift is not None
        assert abs(shift) < 0.01  # Near zero

    def test_backward_shift(self):
        """Attention moving backward should give negative shift."""
        prev = np.zeros((1, 5))
        prev[0, 4] = 1.0
        curr = np.zeros((1, 5))
        curr[0, 1] = 1.0
        ts = [1.0]
        shift = compute_attention_shift(curr, prev, ts)
        assert shift is not None
        assert shift < 0  # Backward

    def test_ts_weighting(self):
        """Shift should be weighted by TS scores."""
        prev = np.zeros((2, 5))
        prev[0, 1] = 1.0  # Head 0 at position 1
        prev[1, 3] = 1.0  # Head 1 at position 3
        curr = np.zeros((2, 5))
        curr[0, 3] = 1.0  # Head 0 moved to 3
        curr[1, 3] = 1.0  # Head 1 stayed at 3
        # High TS for head 0 -> shift dominated by head 0 (moved by 2)
        ts_high_h0 = [10.0, 1.0]
        shift_h0 = compute_attention_shift(curr, prev, ts_high_h0)
        # High TS for head 1 -> shift dominated by head 1 (no move)
        ts_high_h1 = [1.0, 10.0]
        shift_h1 = compute_attention_shift(curr, prev, ts_high_h1)
        assert shift_h0 > shift_h1  # Head 0 moved more

    def test_different_source_sizes(self):
        """Should handle attention with different source sizes."""
        prev = np.array([[0.1, 0.9]])  # 2 source tokens
        curr = np.array([[0.05, 0.15, 0.8]])  # 3 source tokens (new word added)
        ts = [1.0]
        shift = compute_attention_shift(curr, prev, ts)
        assert shift is not None
        # Should show forward movement (attending to later position)

    def test_zero_ts_scores(self):
        """Zero TS scores should return None (avoid division by zero)."""
        attn = np.array([[0.5, 0.5]])
        ts = [0.0]
        shift = compute_attention_shift(attn, attn, ts)
        assert shift is None


class TestAttentionShiftSupportsWrite:
    """Tests for attention shift interpretation."""

    def test_large_shift_supports_write(self):
        """Large forward shift should support WRITE."""
        assert attention_shift_supports_write(1.5) is True

    def test_small_shift_supports_read(self):
        """Small shift should support READ."""
        assert attention_shift_supports_write(0.1) is False

    def test_at_threshold(self):
        """Shift at threshold should support WRITE."""
        assert attention_shift_supports_write(0.5) is True

    def test_below_threshold(self):
        """Shift below threshold should support READ."""
        assert attention_shift_supports_write(0.49) is False

    def test_none_returns_none(self):
        """None shift should return None."""
        assert attention_shift_supports_write(None) is None

    def test_negative_shift(self):
        """Backward shift should support READ."""
        assert attention_shift_supports_write(-1.0) is False

    def test_custom_threshold(self):
        """Custom threshold should be respected."""
        assert attention_shift_supports_write(0.3, min_shift=0.2) is True
        assert attention_shift_supports_write(0.1, min_shift=0.2) is False


class TestAttentionShiftConfig:
    """Tests for attention shift in BackendConfig."""

    def test_default_disabled(self):
        """Attention shift should be disabled by default."""
        config = BackendConfig()
        assert config.attention_shift is False

    def test_from_dict(self):
        """Should accept attention_shift from dict."""
        config = BackendConfig.from_dict({"attention_shift": True})
        assert config.attention_shift is True

    def test_sweep_shortname(self):
        """'attshift' should map to 'attention_shift'."""
        grid = parse_sweep_spec("attshift=0,1")
        assert "attention_shift" in grid
        assert grid["attention_shift"] == [0, 1]


class TestAttentionShiftInCombinedBorder:
    """Tests for attention shift integration in check_border_combined."""

    def test_shift_read_inhibits_border(self):
        """When attention didn't shift, border should be inhibited."""
        attn = np.zeros((2, 5))
        attn[0, 4] = 1.0  # Attending to last position (would trigger border)
        attn[1, 4] = 1.0
        ts = [1.0, 1.0]

        # Without shift: should hit border
        hit_no_shift, _, _ = check_border_combined(
            attn, ts, 5, 2,
        )
        assert hit_no_shift is True

        # With shift=False (model didn't shift forward): should inhibit border
        hit_with_shift, _, _ = check_border_combined(
            attn, ts, 5, 2,
            attn_shift_write=False,
        )
        assert hit_with_shift is False

    def test_shift_write_allows_border(self):
        """When attention shifted forward, border should proceed normally."""
        attn = np.zeros((2, 5))
        attn[0, 4] = 1.0
        attn[1, 4] = 1.0
        ts = [1.0, 1.0]

        hit, _, _ = check_border_combined(
            attn, ts, 5, 2,
            attn_shift_write=True,
        )
        assert hit is True

    def test_shift_none_no_effect(self):
        """None attention shift should not affect border decision."""
        attn = np.zeros((2, 5))
        attn[0, 4] = 1.0
        attn[1, 4] = 1.0
        ts = [1.0, 1.0]

        hit, _, _ = check_border_combined(
            attn, ts, 5, 2,
            attn_shift_write=None,
        )
        assert hit is True  # Normal border behavior
