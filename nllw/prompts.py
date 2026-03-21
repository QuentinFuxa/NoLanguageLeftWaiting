"""Prompt format registry for SimulMT with decoder-only LLMs.

Each format defines how to build the prompt for a given model family and
language direction. Formats are identified by a (model_family, direction) key,
with a fallback to just model_family for generic formats.

Prompt structure:
    prefix + [context_block] + source_text + suffix + [committed_output]

The context block is optional and built from previous segment translations
to help coherence across sentence boundaries.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass(frozen=True)
class PromptFormat:
    """A prompt template for a specific model + language direction."""
    name: str
    prefix: str
    suffix: str
    context_tpl: Optional[str] = None
    context_entry: Optional[str] = None
    # Stop strings (tokenized at runtime to build stop_ids)
    extra_stop_strings: tuple = ()

    def build_prompt(self, source_text: str, prev_context: Optional[List[Dict[str, str]]] = None) -> str:
        """Build full prompt string from source text and optional context."""
        context_str = ""
        if prev_context and self.context_tpl and self.context_entry:
            entries = "".join(self.context_entry.format(**ctx) for ctx in prev_context)
            context_str = self.context_tpl.format(context=entries)
        return self.prefix + context_str + source_text + self.suffix


# ---------------------------------------------------------------------------
# Context templates (reusable)
# ---------------------------------------------------------------------------
_CTX_PAIR = {
    "context_tpl": "[Previous translations]\n{context}\n[New sentence to translate]\n",
    "context_entry": '"{source}" -> "{translation}"\n',
}

_CTX_ARROW = {
    "context_tpl": "[Previous translations]\n{context}\n[New sentence to translate]\n",
    "context_entry": '"{source}" \u2192 "{translation}"\n',
}

_CTX_ZH = {
    "context_tpl": "\u4e4b\u524d\u7684\u7ffb\u8bd1\uff1a\n{context}\n\u5f53\u524d\u9700\u8981\u7ffb\u8bd1\uff1a\n",
    "context_entry": "{source} \u2192 {translation}\n",
}


# ---------------------------------------------------------------------------
# HY-MT formats
# ---------------------------------------------------------------------------
_HYMT_BASE = dict(
    suffix="<|extra_0|>",
)

_HYMT_FORMATS = {
    ("hymt", "en-zh"): PromptFormat(
        name="hymt-en-zh",
        prefix="\u5c06\u4ee5\u4e0b\u6587\u672c\u7ffb\u8bd1\u4e3a\u4e2d\u6587\uff0c\u6ce8\u610f\u53ea\u9700\u8981\u8f93\u51fa\u7ffb\u8bd1\u540e\u7684\u7ed3\u679c\uff0c\u4e0d\u8981\u989d\u5916\u89e3\u91ca\uff1a\n\n",
        **_HYMT_BASE,
    ),
    ("hymt", "en-de"): PromptFormat(
        name="hymt-en-de",
        prefix=(
            "Translate the following text into German, please only output "
            "the translated result without additional explanation:\n\n"
        ),
        **_HYMT_BASE,
        **_CTX_PAIR,
    ),
    ("hymt", "en-it"): PromptFormat(
        name="hymt-en-it",
        prefix=(
            "Translate the following text into Italian, please only output "
            "the translated result without additional explanation:\n\n"
        ),
        **_HYMT_BASE,
        **_CTX_PAIR,
    ),
    ("hymt", "en-fr"): PromptFormat(
        name="hymt-en-fr",
        prefix=(
            "Translate the following text into French, please only output "
            "the translated result without additional explanation:\n\n"
        ),
        **_HYMT_BASE,
        **_CTX_PAIR,
    ),
    ("hymt", "cs-en"): PromptFormat(
        name="hymt-cs-en",
        prefix=(
            "Translate the following text into English, please only output "
            "the translated result without additional explanation:\n\n"
        ),
        **_HYMT_BASE,
        **_CTX_PAIR,
    ),
}

# HY-MT official prompt format (matches training data exactly)
# Ref: https://huggingface.co/tencent/HY-MT1.5-7B-GGUF chat template
# Key differences from the above: "segment" not "text", no "please only output",
# period instead of colon, context injection removed (dead end for HY-MT).
# NOTE: context injection KILLS HY-MT quality (-0.084 to -0.125 COMET), so
# the official format variants have NO context injection enabled.
_HYMT_OFFICIAL_FORMATS = {
    ("hymt-official", "en-zh"): PromptFormat(
        name="hymt-official-en-zh",
        prefix="\u5c06\u4ee5\u4e0b\u6587\u672c\u7ffb\u8bd1\u4e3a\u4e2d\u6587\uff0c\u6ce8\u610f\u53ea\u9700\u8981\u8f93\u51fa\u7ffb\u8bd1\u540e\u7684\u7ed3\u679c\uff0c\u4e0d\u8981\u989d\u5916\u89e3\u91ca\u3002\n\n",
        **_HYMT_BASE,
    ),
    ("hymt-official", "en-de"): PromptFormat(
        name="hymt-official-en-de",
        prefix="Translate the following segment into German, without additional explanation.\n\n",
        **_HYMT_BASE,
    ),
    ("hymt-official", "en-it"): PromptFormat(
        name="hymt-official-en-it",
        prefix="Translate the following segment into Italian, without additional explanation.\n\n",
        **_HYMT_BASE,
    ),
    ("hymt-official", "en-fr"): PromptFormat(
        name="hymt-official-en-fr",
        prefix="Translate the following segment into French, without additional explanation.\n\n",
        **_HYMT_BASE,
    ),
    ("hymt-official", "cs-en"): PromptFormat(
        name="hymt-official-cs-en",
        prefix="Translate the following segment into English, without additional explanation.\n\n",
        **_HYMT_BASE,
    ),
}


# HY-MT context variants for EN-ZH
_HYMT_CTX_VARIANTS = {
    ("hymt-ctx-v1", "en-zh"): PromptFormat(
        name="hymt-ctx-v1-en-zh",
        prefix="\u5c06\u4ee5\u4e0b\u6587\u672c\u7ffb\u8bd1\u4e3a\u4e2d\u6587\uff0c\u6ce8\u610f\u53ea\u9700\u8981\u8f93\u51fa\u7ffb\u8bd1\u540e\u7684\u7ed3\u679c\uff0c\u4e0d\u8981\u989d\u5916\u89e3\u91ca\uff1a\n\n",
        suffix="<|extra_0|>",
        **_CTX_ZH,
    ),
    ("hymt-ctx-v2", "en-zh"): PromptFormat(
        name="hymt-ctx-v2-en-zh",
        prefix="\u5c06\u4ee5\u4e0b\u6587\u672c\u7ffb\u8bd1\u4e3a\u4e2d\u6587\uff0c\u6ce8\u610f\u53ea\u9700\u8981\u8f93\u51fa\u7ffb\u8bd1\u540e\u7684\u7ed3\u679c\uff0c\u4e0d\u8981\u989d\u5916\u89e3\u91ca\uff1a\n\n",
        suffix="<|extra_0|>",
        context_tpl="{context}\n",
        context_entry="{source}<|extra_0|>{translation}\n",
    ),
    ("hymt-ctx-v3", "en-zh"): PromptFormat(
        name="hymt-ctx-v3-en-zh",
        prefix="\u5c06\u4ee5\u4e0b\u6587\u672c\u7ffb\u8bd1\u4e3a\u4e2d\u6587\uff0c\u6ce8\u610f\u53ea\u9700\u8981\u8f93\u51fa\u7ffb\u8bd1\u540e\u7684\u7ed3\u679c\uff0c\u4e0d\u8981\u989d\u5916\u89e3\u91ca\uff1a\n\n",
        suffix="<|extra_0|>",
        **_CTX_ARROW,
    ),
}


# ---------------------------------------------------------------------------
# Qwen3 / Qwen3.5 formats
# ---------------------------------------------------------------------------
_QWEN_THINK = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_QWEN_NOTHINK = "<|im_end|>\n<|im_start|>assistant\n/no_think\n"

def _qwen_prefix(target_lang: str, target_name: str, example_src: str, example_tgt: str) -> str:
    return (
        "<|im_start|>user\n"
        f"You are a professional English to {target_name} translator. "
        f"Always output {target_name} ({target_lang}) only.\n"
        f"Example: EN: {example_src} \u2192 {target_lang.upper()}: {example_tgt}\n\n"
        f"Translate the following into {target_name}:\n\n"
    )

_QWEN_FORMATS = {
    # Qwen3
    ("qwen3", "en-zh"): PromptFormat(
        name="qwen3-en-zh",
        prefix=_qwen_prefix("\u4e2d\u6587", "Chinese", "Thank you very much.", "\u975e\u5e38\u611f\u8c22\u3002"),
        suffix=_QWEN_THINK,
        **_CTX_ARROW,
    ),
    ("qwen3", "en-de"): PromptFormat(
        name="qwen3-en-de",
        prefix=_qwen_prefix("Deutsch", "German", "Thank you very much.", "Vielen Dank."),
        suffix=_QWEN_THINK,
        **_CTX_ARROW,
    ),
    ("qwen3", "en-it"): PromptFormat(
        name="qwen3-en-it",
        prefix=_qwen_prefix("Italiano", "Italian", "Thank you very much.", "Grazie mille."),
        suffix=_QWEN_THINK,
        **_CTX_ARROW,
    ),
    ("qwen3", "en-fr"): PromptFormat(
        name="qwen3-en-fr",
        prefix=_qwen_prefix("Fran\u00e7ais", "French", "Thank you very much.", "Merci beaucoup."),
        suffix=_QWEN_THINK,
        **_CTX_ARROW,
    ),
    # Qwen3.5
    ("qwen3.5", "en-zh"): PromptFormat(
        name="qwen3.5-en-zh",
        prefix=_qwen_prefix("\u4e2d\u6587", "Chinese", "Thank you very much.", "\u975e\u5e38\u611f\u8c22\u3002"),
        suffix=_QWEN_THINK,
        **_CTX_ARROW,
    ),
    ("qwen3.5", "en-de"): PromptFormat(
        name="qwen3.5-en-de",
        prefix=_qwen_prefix("Deutsch", "German", "Thank you very much.", "Vielen Dank."),
        suffix=_QWEN_THINK,
        **_CTX_ARROW,
    ),
    ("qwen3.5", "en-it"): PromptFormat(
        name="qwen3.5-en-it",
        prefix=_qwen_prefix("Italiano", "Italian", "Thank you very much.", "Grazie mille."),
        suffix=_QWEN_THINK,
        **_CTX_ARROW,
    ),
    ("qwen3.5", "en-fr"): PromptFormat(
        name="qwen3.5-en-fr",
        prefix=_qwen_prefix("Fran\u00e7ais", "French", "Thank you very much.", "Merci beaucoup."),
        suffix=_QWEN_THINK,
        **_CTX_ARROW,
    ),
    # Qwen3.5 no-think (for LoRA SFT models)
    ("qwen3.5-nothink", "en-zh"): PromptFormat(
        name="qwen3.5-nothink-en-zh",
        prefix=_qwen_prefix("\u4e2d\u6587", "Chinese", "Thank you very much.", "\u975e\u5e38\u611f\u8c22\u3002"),
        suffix=_QWEN_NOTHINK,
        **_CTX_ARROW,
    ),
}


# ---------------------------------------------------------------------------
# EuroLLM formats
# ---------------------------------------------------------------------------
def _eurollm_prefix(target_name: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are a professional simultaneous interpreter at an academic conference. "
        f"Translate the following English text into {target_name}.<|im_end|>\n"
        "<|im_start|>user\n"
    )

_EUROLLM_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"

_EUROLLM_FORMATS = {
    ("eurollm", "en-zh"): PromptFormat(
        name="eurollm-en-zh",
        prefix=_eurollm_prefix("Chinese"),
        suffix=_EUROLLM_SUFFIX,
        **_CTX_ARROW,
    ),
    ("eurollm", "en-de"): PromptFormat(
        name="eurollm-en-de",
        prefix=_eurollm_prefix("German"),
        suffix=_EUROLLM_SUFFIX,
        **_CTX_ARROW,
    ),
    ("eurollm", "en-it"): PromptFormat(
        name="eurollm-en-it",
        prefix=_eurollm_prefix("Italian"),
        suffix=_EUROLLM_SUFFIX,
        **_CTX_ARROW,
    ),
    ("eurollm", "en-fr"): PromptFormat(
        name="eurollm-en-fr",
        prefix=_eurollm_prefix("French"),
        suffix=_EUROLLM_SUFFIX,
        **_CTX_ARROW,
    ),
    ("eurollm", "cs-en"): PromptFormat(
        name="eurollm-cs-en",
        prefix=(
            "<|im_start|>system\n"
            "You are a professional simultaneous interpreter at an academic conference. "
            "Translate the following Czech text into English.<|im_end|>\n"
            "<|im_start|>user\n"
        ),
        suffix=_EUROLLM_SUFFIX,
        **_CTX_ARROW,
    ),
}


# ---------------------------------------------------------------------------
# Tower format
# ---------------------------------------------------------------------------
_TOWER_FORMATS = {
    ("tower", "en-zh"): PromptFormat(
        name="tower-en-zh",
        prefix=(
            "<|im_start|>user\n"
            "Translate the following text from English into Chinese.\n"
            "English: "
        ),
        suffix="\nChinese:<|im_end|>\n<|im_start|>assistant\n",
    ),
}


# ---------------------------------------------------------------------------
# Gemma (TranslateGemma) format
# ---------------------------------------------------------------------------
_GEMMA_FORMATS = {
    ("gemma", "en-zh"): PromptFormat(
        name="gemma-en-zh",
        prefix=(
            "<bos><start_of_turn>user\n"
            "You are a professional English (en) to Chinese (zh) translator. "
            "Provide only the translated text without any explanation.\n\n"
        ),
        suffix="<end_of_turn>\n<start_of_turn>model\n",
        context_tpl="Previous translation: {context}\nTranslate:\n",
        context_entry="{translation} ",
    ),
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
PROMPT_REGISTRY: Dict[tuple, PromptFormat] = {}
PROMPT_REGISTRY.update(_HYMT_FORMATS)
PROMPT_REGISTRY.update(_HYMT_OFFICIAL_FORMATS)
PROMPT_REGISTRY.update(_HYMT_CTX_VARIANTS)
PROMPT_REGISTRY.update(_QWEN_FORMATS)
PROMPT_REGISTRY.update(_EUROLLM_FORMATS)
PROMPT_REGISTRY.update(_TOWER_FORMATS)
PROMPT_REGISTRY.update(_GEMMA_FORMATS)


def get_prompt_format(model_family: str, direction: str) -> PromptFormat:
    """Look up a prompt format by (model_family, direction).

    Falls back to (model_family, "en-zh") if the exact direction isn't registered.
    Raises KeyError if no match found.
    """
    key = (model_family, direction)
    if key in PROMPT_REGISTRY:
        return PROMPT_REGISTRY[key]
    # Fallback: try default direction
    fallback = (model_family, "en-zh")
    if fallback in PROMPT_REGISTRY:
        return PROMPT_REGISTRY[fallback]
    raise KeyError(
        f"No prompt format for {key}. Available: "
        f"{sorted(set(k[0] for k in PROMPT_REGISTRY))}"
    )


def detect_model_family(model_path: str) -> str:
    """Auto-detect model family from GGUF filename.

    Examples:
        'HY-MT1.5-7B-Q5_K_M.gguf' -> 'hymt'
        'Qwen3-8B-Q4_K_M.gguf' -> 'qwen3'
        'Qwen3.5-4B-Q5_K_M.gguf' -> 'qwen3.5'
        'EuroLLM-9B-Q5_K_M.gguf' -> 'eurollm'
        'TowerInstruct-7B.gguf' -> 'tower'
        'TranslateGemma-4B.gguf' -> 'gemma'
    """
    name = model_path.lower().rsplit("/", 1)[-1]
    if "hy-mt" in name or "hymt" in name:
        return "hymt"
    if "qwen3.5" in name or "qwen3_5" in name:
        return "qwen3.5"
    if "qwen3" in name:
        return "qwen3"
    if "eurollm" in name:
        return "eurollm"
    if "tower" in name:
        return "tower"
    if "gemma" in name or "translategemma" in name:
        return "gemma"
    return "hymt"  # default


def list_formats() -> List[str]:
    """List all registered format names."""
    return sorted(f"{k[0]}:{k[1]}" for k in PROMPT_REGISTRY)
