"""Tests for Iteration 15 fixes and improvements.

Bug fixes:
    1. simulate_backend empty translation fallback (segment_end clears committed_ids)
    2. KV cache GPU offload (n_gpu_layers wired through create_context)
    3. Chinese BLEU tokenization (sacrebleu tokenize="zh" for CJK)
    4. Cross-lingual head transfer fallback in _find_heads_for_model

All tests are unit tests that don't require llama.cpp.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestSimulateBackendFallback:
    """Test that simulate_backend collects translation from step texts
    when get_full_translation() returns empty (e.g. after segment_end)."""

    def test_fallback_to_collected_texts(self):
        """When get_full_translation() returns empty, use collected step texts."""
        from nllw.simulate import simulate_backend

        backend = MagicMock()
        backend.get_full_translation.return_value = ""
        backend.reset.return_value = None

        # Simulate 3 words, backend emits text on word 2 and word 3
        from nllw.backend_protocol import TranslationStep
        backend.translate.side_effect = [
            TranslationStep(text="", source_words_seen=1),
            TranslationStep(text="hello ", committed_tokens=1,
                          stopped_at_border=True, source_words_seen=2),
            TranslationStep(text="world", committed_tokens=1,
                          is_final=True, source_words_seen=3),
        ]

        trace = simulate_backend(backend, "a b c", is_final_on_last=True)
        assert trace.translation == "hello world"
        assert len(trace.delays) == 2  # 2 target words

    def test_prefers_get_full_translation(self):
        """When get_full_translation() returns non-empty, prefer it."""
        from nllw.simulate import simulate_backend
        from nllw.backend_protocol import TranslationStep

        backend = MagicMock()
        backend.get_full_translation.return_value = "proper translation"
        backend.reset.return_value = None

        backend.translate.side_effect = [
            TranslationStep(text="partial ", committed_tokens=1,
                          source_words_seen=1),
            TranslationStep(text="text", committed_tokens=1,
                          is_final=True, source_words_seen=2),
        ]

        trace = simulate_backend(backend, "a b", is_final_on_last=True)
        assert trace.translation == "proper translation"

    def test_no_text_emitted(self):
        """When no text emitted at all, translation is empty."""
        from nllw.simulate import simulate_backend
        from nllw.backend_protocol import TranslationStep

        backend = MagicMock()
        backend.get_full_translation.return_value = ""
        backend.reset.return_value = None

        backend.translate.side_effect = [
            TranslationStep(text="", source_words_seen=1),
            TranslationStep(text="", is_final=True, source_words_seen=2),
        ]

        trace = simulate_backend(backend, "a b", is_final_on_last=True)
        assert trace.translation == ""
        assert len(trace.delays) == 0


class TestBLEUChineseTokenization:
    """Test Chinese BLEU uses character-level tokenization."""

    def test_bleu_tokenize_returns_zh(self):
        from nllw.metrics import _bleu_tokenize
        assert _bleu_tokenize("zh") == "zh"
        assert _bleu_tokenize("ja") == "zh"

    def test_bleu_tokenize_returns_default(self):
        from nllw.metrics import _bleu_tokenize
        assert _bleu_tokenize("en") == "13a"
        assert _bleu_tokenize("fr") == "13a"
        assert _bleu_tokenize("de") == "13a"
        assert _bleu_tokenize(None) == "13a"

    def test_chinese_bleu_positive(self):
        """Chinese text with overlapping characters should have positive BLEU."""
        from nllw.metrics import compute_bleu
        hyp = "我们现在已经拥有了小鼠"
        ref = "我们现在有小鼠"
        score = compute_bleu(hyp, ref, target_lang="zh")
        assert score > 0

    def test_corpus_bleu_chinese(self):
        from nllw.metrics import compute_bleu_corpus
        hyps = ["我们现在已经拥有了小鼠", "糖尿病可以治愈"]
        refs = ["我们现在有小鼠", "糖尿病能否治愈"]
        score = compute_bleu_corpus(hyps, refs, target_lang="zh")
        assert score > 0

    def test_english_bleu_unchanged(self):
        """English BLEU should still work the same."""
        from nllw.metrics import compute_bleu
        hyp = "the cat sat on the mat"
        ref = "the cat is on the mat"
        score_with_lang = compute_bleu(hyp, ref, target_lang="en")
        score_without = compute_bleu(hyp, ref)
        assert score_with_lang == score_without


class TestKVCacheOffload:
    """Test that create_context passes n_gpu_layers for KV cache GPU offload."""

    def test_create_context_signature(self):
        """create_context should accept n_gpu_layers parameter."""
        import inspect
        from nllw.llama_backend import create_context
        sig = inspect.signature(create_context)
        assert "n_gpu_layers" in sig.parameters
        assert sig.parameters["n_gpu_layers"].default == 0


class TestCrossLingualHeadFallback:
    """Test that head auto-discovery falls back to cross-lingual transfer."""

    def test_hymt_en_de_finds_config(self):
        """HY-MT EN-DE should find a head config (dedicated or cross-lingual fallback)."""
        from nllw.alignatt_backend import _find_heads_for_model
        import os

        result = _find_heads_for_model(
            "/path/to/HY-MT1.5-7B.Q8_0.gguf", "en-de"
        )
        configs_dir = os.path.join(
            os.path.dirname(__file__), "..", "nllw", "heads", "configs"
        )
        if os.path.isdir(configs_dir):
            # Should find either dedicated en_de or cross-lingual fallback
            hymt_files = [f for f in os.listdir(configs_dir)
                          if ("hymt" in f or "hy_mt" in f) and "1.8b" not in f]
            if hymt_files:
                assert result is not None, "Should find hymt head config"
                assert "hymt" in result.lower() or "hy_mt" in result.lower()

    def test_exact_match_preferred(self):
        """Exact direction match should be preferred over fallback."""
        from nllw.alignatt_backend import _find_heads_for_model
        import os

        result = _find_heads_for_model(
            "/path/to/HY-MT1.5-7B.Q8_0.gguf", "en-zh"
        )
        configs_dir = os.path.join(
            os.path.dirname(__file__), "..", "nllw", "heads", "configs"
        )
        if os.path.isdir(configs_dir):
            assert result is not None
            assert "en_zh" in result.lower()
