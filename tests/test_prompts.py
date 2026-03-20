"""Tests for prompt format registry."""

import pytest
from nllw.prompts import (
    PromptFormat,
    build_prompt,
    get_format,
    list_formats,
    register_format,
    PROMPT_FORMATS,
)


class TestPromptFormat:
    def test_build_prompt_basic(self):
        fmt = PromptFormat(prefix="Translate: ", suffix=" [END]")
        result = build_prompt("Hello world", fmt)
        assert result == "Translate: Hello world [END]"

    def test_build_prompt_with_context(self):
        fmt = PromptFormat(
            prefix="Translate: ",
            suffix=" [END]",
            context_tpl="[Context: {context}]\n",
            context_entry='"{source}" -> "{translation}"\n',
        )
        ctx = [{"source": "Hi", "translation": "Bonjour"}]
        result = build_prompt("Hello", fmt, prev_context=ctx)
        assert '"Hi" -> "Bonjour"' in result
        assert "Hello [END]" in result

    def test_build_prompt_no_context_template(self):
        fmt = PromptFormat(prefix="T: ", suffix=" E")
        result = build_prompt("test", fmt, prev_context=[{"source": "x", "translation": "y"}])
        assert result == "T: test E"  # Context ignored when no template

    def test_build_prompt_multiple_contexts(self):
        fmt = PromptFormat(
            prefix="",
            suffix="",
            context_tpl="{context}",
            context_entry="{translation} ",
        )
        ctx = [
            {"source": "a", "translation": "A"},
            {"source": "b", "translation": "B"},
        ]
        result = build_prompt("c", fmt, prev_context=ctx)
        assert "A B " in result


class TestFormatRegistry:
    def test_hymt_registered(self):
        fmt = get_format("hymt")
        assert "翻译" in fmt.prefix
        assert "<|extra_0|>" in fmt.suffix

    def test_qwen35_registered(self):
        fmt = get_format("qwen3.5")
        assert "<|im_start|>" in fmt.prefix
        assert "<think>" in fmt.suffix

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown prompt format"):
            get_format("nonexistent_format_xyz")

    def test_list_formats(self):
        formats = list_formats()
        assert "hymt" in formats
        assert "qwen3.5" in formats
        assert "eurollm" in formats
        assert len(formats) > 15  # We have many formats

    def test_all_formats_have_prefix_suffix(self):
        for name in list_formats():
            fmt = get_format(name)
            assert fmt.prefix is not None
            assert fmt.suffix is not None

    def test_register_custom_format(self):
        register_format("test_custom", PromptFormat(
            prefix="Custom: ",
            suffix=" [done]",
        ))
        fmt = get_format("test_custom")
        assert fmt.prefix == "Custom: "
        # Clean up
        del PROMPT_FORMATS["test_custom"]


class TestLanguageDirections:
    """Verify we have formats for all key language directions."""

    def test_en_zh_formats(self):
        assert "hymt" in PROMPT_FORMATS
        assert "qwen3" in PROMPT_FORMATS
        assert "qwen3.5" in PROMPT_FORMATS

    def test_en_de_formats(self):
        assert "hymt-de" in PROMPT_FORMATS
        assert "qwen3.5-de" in PROMPT_FORMATS

    def test_en_it_formats(self):
        assert "hymt-it" in PROMPT_FORMATS
        assert "qwen3.5-it" in PROMPT_FORMATS

    def test_cs_en_format(self):
        assert "hymt-cs-en" in PROMPT_FORMATS
