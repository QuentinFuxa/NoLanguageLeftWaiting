"""Tests for prompt format registry."""

import pytest
from nllw.prompts import (
    get_prompt_format,
    detect_model_family,
    list_formats,
    PromptFormat,
    PROMPT_REGISTRY,
)


class TestPromptRegistry:
    def test_hymt_en_zh_exists(self):
        fmt = get_prompt_format("hymt", "en-zh")
        assert "翻译" in fmt.prefix or "translate" in fmt.prefix.lower()
        assert "<|extra_0|>" in fmt.suffix

    def test_qwen3_en_zh_exists(self):
        fmt = get_prompt_format("qwen3", "en-zh")
        assert "<|im_start|>" in fmt.prefix
        assert "<think>" in fmt.suffix

    def test_qwen35_en_de_exists(self):
        fmt = get_prompt_format("qwen3.5", "en-de")
        assert "German" in fmt.prefix

    def test_eurollm_en_zh_exists(self):
        fmt = get_prompt_format("eurollm", "en-zh")
        assert "system" in fmt.prefix
        assert "assistant" in fmt.suffix

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            get_prompt_format("nonexistent-model", "en-zh")

    def test_fallback_to_default_direction(self):
        """If exact direction not found, fall back to en-zh."""
        fmt = get_prompt_format("hymt", "en-unknown-lang")
        assert fmt.name == "hymt-en-zh"

    def test_list_formats(self):
        formats = list_formats()
        assert len(formats) > 10
        assert any("hymt" in f for f in formats)
        assert any("qwen3" in f for f in formats)


class TestBuildPrompt:
    def test_no_context(self):
        fmt = get_prompt_format("hymt", "en-zh")
        prompt = fmt.build_prompt("Hello world")
        assert "Hello world" in prompt
        assert fmt.prefix in prompt
        assert fmt.suffix in prompt

    def test_with_context(self):
        fmt = get_prompt_format("hymt", "en-de")
        context = [{"source": "Hello", "translation": "Hallo"}]
        prompt = fmt.build_prompt("How are you", prev_context=context)
        assert "How are you" in prompt
        assert "Hello" in prompt
        assert "Hallo" in prompt

    def test_no_context_template(self):
        """HY-MT en-zh has no context template by default."""
        fmt = get_prompt_format("hymt", "en-zh")
        context = [{"source": "Hello", "translation": "你好"}]
        prompt = fmt.build_prompt("World", prev_context=context)
        # Context should be ignored since no context_tpl
        assert "Hello" not in prompt


class TestDetectModelFamily:
    def test_hymt(self):
        assert detect_model_family("HY-MT1.5-7B-Q5_K_M.gguf") == "hymt"
        assert detect_model_family("/models/hy-mt-7b.gguf") == "hymt"

    def test_qwen3(self):
        assert detect_model_family("Qwen3-8B-Q4_K_M.gguf") == "qwen3"

    def test_qwen35(self):
        assert detect_model_family("Qwen3.5-4B-Q5_K_M.gguf") == "qwen3.5"
        assert detect_model_family("qwen3_5_9b.gguf") == "qwen3.5"

    def test_eurollm(self):
        assert detect_model_family("EuroLLM-9B-Q5.gguf") == "eurollm"

    def test_tower(self):
        assert detect_model_family("TowerInstruct-7B.gguf") == "tower"

    def test_gemma(self):
        assert detect_model_family("TranslateGemma-4B.gguf") == "gemma"

    def test_default(self):
        assert detect_model_family("unknown-model.gguf") == "hymt"
