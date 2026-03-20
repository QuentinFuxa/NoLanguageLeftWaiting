"""
Prompt format registry for LLM-based simultaneous translation.

Each format specifies:
- prefix: Text before the source sentence (instructions, special tokens)
- suffix: Text after the source sentence (response trigger tokens)
- context_tpl: Template for injecting previous translation context (optional)
- context_entry: Template for each context entry (optional)

Supported models: HY-MT1.5, Qwen3/3.5, EuroLLM, TowerInstruct, Gemma.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PromptFormat:
    """A prompt template for a specific model and language direction."""
    prefix: str
    suffix: str
    context_tpl: Optional[str] = None
    context_entry: Optional[str] = None


def build_prompt(
    source_text: str,
    fmt: PromptFormat,
    prev_context: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Build full prompt string from format template + optional context.

    Args:
        source_text: The source text to translate.
        fmt: The prompt format to use.
        prev_context: List of {"source": ..., "translation": ...} dicts.

    Returns:
        Complete prompt string ready for tokenization.
    """
    context_str = ""
    if prev_context and fmt.context_tpl and fmt.context_entry:
        entries = "".join(fmt.context_entry.format(**ctx) for ctx in prev_context)
        context_str = fmt.context_tpl.format(context=entries)
    return fmt.prefix + context_str + source_text + fmt.suffix


def find_source_token_range(
    tokenize_fn,
    source_text: str,
    fmt: PromptFormat,
    prev_context: Optional[List[Dict[str, str]]] = None,
) -> tuple:
    """Find the token range [src_start, src_end) of source text within the full prompt.

    Args:
        tokenize_fn: Function(text, add_bos, special) -> List[int]
        source_text: The source text.
        fmt: The prompt format.
        prev_context: Optional context entries.

    Returns:
        (src_start, src_end) token indices. Returns (0, 0) if range invalid.
    """
    context_str = ""
    if prev_context and fmt.context_tpl and fmt.context_entry:
        entries = "".join(fmt.context_entry.format(**ctx) for ctx in prev_context)
        context_str = fmt.context_tpl.format(context=entries)

    prefix = fmt.prefix + context_str
    full = prefix + source_text + fmt.suffix

    prefix_tokens = tokenize_fn(prefix, add_bos=True, special=True)
    full_tokens = tokenize_fn(full, add_bos=True, special=True)
    suffix_tokens = tokenize_fn(fmt.suffix, add_bos=False, special=True)

    src_start = len(prefix_tokens)
    src_end = len(full_tokens) - len(suffix_tokens)
    return (src_start, src_end) if src_end > src_start else (0, 0)


# ---------------------------------------------------------------------------
# Format registry
# ---------------------------------------------------------------------------

PROMPT_FORMATS: Dict[str, PromptFormat] = {}


def register_format(name: str, fmt: PromptFormat):
    """Register a prompt format by name."""
    PROMPT_FORMATS[name] = fmt


def get_format(name: str) -> PromptFormat:
    """Get a registered prompt format by name."""
    if name not in PROMPT_FORMATS:
        raise ValueError(
            f"Unknown prompt format: {name}. "
            f"Available: {', '.join(sorted(PROMPT_FORMATS))}"
        )
    return PROMPT_FORMATS[name]


# ---------------------------------------------------------------------------
# HY-MT1.5 formats
# ---------------------------------------------------------------------------

register_format("hymt", PromptFormat(
    prefix="将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：\n\n",
    suffix="<|extra_0|>",
))

_HYMT_CTX = dict(
    context_tpl='[Previous translations]\n{context}\n[New sentence to translate]\n',
    context_entry='"{source}" -> "{translation}"\n',
)

register_format("hymt-de", PromptFormat(
    prefix=(
        "Translate the following text into German, please only output "
        "the translated result without additional explanation:\n\n"
    ),
    suffix="<|extra_0|>",
    **_HYMT_CTX,
))

register_format("hymt-it", PromptFormat(
    prefix=(
        "Translate the following text into Italian, please only output "
        "the translated result without additional explanation:\n\n"
    ),
    suffix="<|extra_0|>",
    **_HYMT_CTX,
))

register_format("hymt-cs-en", PromptFormat(
    prefix=(
        "Translate the following text into English, please only output "
        "the translated result without additional explanation:\n\n"
    ),
    suffix="<|extra_0|>",
    **_HYMT_CTX,
))

# HY-MT context variants (EN-ZH)
register_format("hymt-ctx-v1", PromptFormat(
    prefix="将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：\n\n",
    suffix="<|extra_0|>",
    context_tpl="之前的翻译：\n{context}\n当前需要翻译：\n",
    context_entry="{source} → {translation}\n",
))

register_format("hymt-ctx-v2", PromptFormat(
    prefix="将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：\n\n",
    suffix="<|extra_0|>",
    context_tpl="{context}\n",
    context_entry="{source}<|extra_0|>{translation}\n",
))

register_format("hymt-ctx-v3", PromptFormat(
    prefix="将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：\n\n",
    suffix="<|extra_0|>",
    context_tpl='[Previous translations]\n{context}\n[New sentence to translate]\n',
    context_entry='"{source}" → "{translation}"\n',
))

# ---------------------------------------------------------------------------
# Qwen3 / Qwen3.5 formats
# ---------------------------------------------------------------------------

_QWEN_CTX = dict(
    context_tpl='[Previous translations]\n{context}\n[New sentence to translate]\n',
    context_entry='"{source}" → "{translation}"\n',
)

_QWEN_THINK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_QWEN_NOTHINK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n/no_think\n"

register_format("qwen3", PromptFormat(
    prefix=(
        "<|im_start|>user\n"
        "You are a professional English to Chinese translator. "
        "Always output Chinese (中文) only.\n"
        "Example: EN: Thank you very much. → ZH: 非常感谢。\n\n"
        "Translate the following into Chinese:\n\n"
    ),
    suffix=_QWEN_THINK_SUFFIX,
    **_QWEN_CTX,
))

register_format("qwen3.5", PromptFormat(
    prefix=(
        "<|im_start|>user\n"
        "You are a professional English to Chinese translator. "
        "Always output Chinese (中文) only.\n"
        "Example: EN: Thank you very much. → ZH: 非常感谢。\n\n"
        "Translate the following into Chinese:\n\n"
    ),
    suffix=_QWEN_THINK_SUFFIX,
    **_QWEN_CTX,
))

register_format("qwen3.5-nothink", PromptFormat(
    prefix=(
        "<|im_start|>user\n"
        "You are a professional English to Chinese translator. "
        "Always output Chinese (中文) only.\n"
        "Example: EN: Thank you very much. → ZH: 非常感谢。\n\n"
        "Translate the following into Chinese:\n\n"
    ),
    suffix=_QWEN_NOTHINK_SUFFIX,
    **_QWEN_CTX,
))

register_format("qwen3.5-de", PromptFormat(
    prefix=(
        "<|im_start|>user\n"
        "You are a professional English to German translator. "
        "Always output German (Deutsch) only.\n"
        "Example: EN: Thank you very much. → DE: Vielen Dank.\n\n"
        "Translate the following into German:\n\n"
    ),
    suffix=_QWEN_THINK_SUFFIX,
    **_QWEN_CTX,
))

register_format("qwen3-de", PromptFormat(
    prefix=(
        "<|im_start|>user\n"
        "You are a professional English to German translator. "
        "Always output German (Deutsch) only.\n"
        "Example: EN: Thank you very much. → DE: Vielen Dank.\n\n"
        "Translate the following into German:\n\n"
    ),
    suffix=_QWEN_THINK_SUFFIX,
    **_QWEN_CTX,
))

register_format("qwen3.5-it", PromptFormat(
    prefix=(
        "<|im_start|>user\n"
        "You are a professional English to Italian translator. "
        "Always output Italian (Italiano) only.\n"
        "Example: EN: Thank you very much. → IT: Grazie mille.\n\n"
        "Translate the following into Italian:\n\n"
    ),
    suffix=_QWEN_THINK_SUFFIX,
    **_QWEN_CTX,
))

register_format("qwen3-it", PromptFormat(
    prefix=(
        "<|im_start|>user\n"
        "You are a professional English to Italian translator. "
        "Always output Italian (Italiano) only.\n"
        "Example: EN: Thank you very much. → IT: Grazie mille.\n\n"
        "Translate the following into Italian:\n\n"
    ),
    suffix=_QWEN_THINK_SUFFIX,
    **_QWEN_CTX,
))

# Qwen3.5 context experiment variants
for variant_suffix, ctx_tpl, ctx_entry in [
    ("-ctx-v1", '[Previous translations]\n{context}\n[New sentence to translate]\n',
     '"{source}" → "{translation}"\n'),
    ("-ctx-v2", "Previous translations for reference:\n{context}\nNow translate:\n",
     "EN: {source}\nZH: {translation}\n\n"),
    ("-ctx-v3", "[之前的翻译]\n{context}\n[翻译以下内容]\n",
     "{translation}\n"),
    ("-ctx-v4", "[Context: the speaker previously said]\n{context}\n[Continue translating]\n",
     "{source} ({translation}) "),
]:
    register_format(f"qwen3.5{variant_suffix}", PromptFormat(
        prefix=(
            "<|im_start|>user\n"
            "You are a professional English to Chinese translator. "
            "Always output Chinese (中文) only.\n"
            "Example: EN: Thank you very much. → ZH: 非常感谢。\n\n"
            "Translate the following into Chinese:\n\n"
        ),
        suffix=_QWEN_THINK_SUFFIX,
        context_tpl=ctx_tpl,
        context_entry=ctx_entry,
    ))

# Qwen3.5 extended think
register_format("qwen3.5-ctx-v5", PromptFormat(
    prefix=(
        "<|im_start|>user\n"
        "You are a professional English to Chinese translator. "
        "Always output Chinese (中文) only.\n"
        "Example: EN: Thank you very much. → ZH: 非常感谢。\n\n"
        "Translate the following into Chinese:\n\n"
    ),
    suffix=(
        "<|im_end|>\n<|im_start|>assistant\n<think>\n"
        "I will translate this faithfully into Chinese, maintaining consistency "
        "with previous context.\n</think>\n\n"
    ),
    context_tpl='[Previous translations]\n{context}\n[New sentence to translate]\n',
    context_entry='"{source}" → "{translation}"\n',
))

# Qwen3.5 Extra Context (glossary-aware) -- {glossary} placeholder injected at runtime
for lang_suffix, lang_name in [("", "Chinese"), ("-de", "German"), ("-it", "Italian")]:
    register_format(f"qwen3.5-extra-context{lang_suffix}", PromptFormat(
        prefix=(
            "<|im_start|>user\n"
            f"You are a professional English to {lang_name} translator. "
            f"Always output {lang_name} only.\n"
            "{{glossary}}"
            f"Translate the following into {lang_name}:\n\n"
        ),
        suffix=(
            "<|im_end|>\n<|im_start|>assistant\n<think>\n"
            f"I will translate this faithfully into {lang_name}, maintaining consistency "
            "with previous context.\n</think>\n\n"
        ),
        context_tpl='[Previous translations]\n{context}\n[New sentence to translate]\n',
        context_entry='"{source}" -> "{translation}"\n',
    ))

# ---------------------------------------------------------------------------
# EuroLLM formats
# ---------------------------------------------------------------------------

for lang_suffix, lang_name in [("", "Chinese"), ("-de", "German"), ("-it", "Italian")]:
    register_format(f"eurollm{lang_suffix}", PromptFormat(
        prefix=(
            "<|im_start|>system\n"
            "You are a professional simultaneous interpreter at an academic conference. "
            f"Translate the following English text into {lang_name}.<|im_end|>\n"
            "<|im_start|>user\n"
        ),
        suffix="<|im_end|>\n<|im_start|>assistant\n",
        **_QWEN_CTX,
    ))

# ---------------------------------------------------------------------------
# Tower / Gemma formats
# ---------------------------------------------------------------------------

register_format("tower", PromptFormat(
    prefix=(
        "<|im_start|>user\n"
        "Translate the following text from English into Chinese.\n"
        "English: "
    ),
    suffix="\nChinese:<|im_end|>\n<|im_start|>assistant\n",
))

register_format("gemma", PromptFormat(
    prefix=(
        "<bos><start_of_turn>user\n"
        "You are a professional English (en) to Chinese (zh) translator. "
        "Provide only the translated text without any explanation.\n\n"
    ),
    suffix="<end_of_turn>\n<start_of_turn>model\n",
    context_tpl="Previous translation: {context}\nTranslate:\n",
    context_entry="{translation} ",
))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def list_formats() -> List[str]:
    """Return sorted list of registered format names."""
    return sorted(PROMPT_FORMATS.keys())
