"""AlignAtt translation backend using llama.cpp.

Drop-in replacement for TranslationBackend with the same translate() interface.
Uses attention-based border detection instead of LocalAgreement prefix stability.

Supports multiple prompt formats via the PROMPT_FORMATS registry:
  - "hymt"           HY-MT native format (default)
  - "qwen3"          Qwen3 chat template with <think> block
  - "qwen3.5"        Qwen3.5 chat template with <think> block
  - "qwen3-nothink"  Qwen3 chat template without thinking (for LoRA SFT models)
  - "eurollm"        EuroLLM system + user prompt
  - "custom"         User-supplied template with {source} placeholder
"""

import json
import os
import threading
from typing import Optional

import numpy as np

from nllw import llama_backend as ll
from nllw.timed_text import TimedText


# ---------------------------------------------------------------------------
# Language name mapping for prompt templates
# ---------------------------------------------------------------------------

_LANG_NAMES = {
    "zh": ("Chinese", "\u4e2d\u6587"),
    "en": ("English", "English"),
    "fr": ("French", "Fran\u00e7ais"),
    "de": ("German", "Deutsch"),
    "es": ("Spanish", "Espa\u00f1ol"),
    "pt": ("Portuguese", "Portugu\u00eas"),
    "it": ("Italian", "Italiano"),
    "ja": ("Japanese", "\u65e5\u672c\u8a9e"),
    "ko": ("Korean", "\ud55c\uad6d\uc5b4"),
    "ru": ("Russian", "\u0420\u0443\u0441\u0441\u043a\u0438\u0439"),
    "ar": ("Arabic", "\u0627\u0644\u0639\u0631\u0628\u064a\u0629"),
    "nl": ("Dutch", "Nederlands"),
    "pl": ("Polish", "Polski"),
    "tr": ("Turkish", "T\u00fcrk\u00e7e"),
    "vi": ("Vietnamese", "Ti\u1ebfng Vi\u1ec7t"),
    "th": ("Thai", "\u0e44\u0e17\u0e22"),
    "id": ("Indonesian", "Bahasa Indonesia"),
    "cs": ("Czech", "\u010ce\u0161tina"),
    "uk": ("Ukrainian", "\u0423\u043a\u0440\u0430\u0457\u043d\u0441\u044c\u043a\u0430"),
    "ro": ("Romanian", "Rom\u00e2n\u0103"),
    "hu": ("Hungarian", "Magyar"),
    "sv": ("Swedish", "Svenska"),
    "da": ("Danish", "Dansk"),
    "fi": ("Finnish", "Suomi"),
    "el": ("Greek", "\u0395\u03bb\u03bb\u03b7\u03bd\u03b9\u03ba\u03ac"),
    "bg": ("Bulgarian", "\u0411\u044a\u043b\u0433\u0430\u0440\u0441\u043a\u0438"),
    "hr": ("Croatian", "Hrvatski"),
    "sk": ("Slovak", "Sloven\u010dina"),
    "sl": ("Slovenian", "Sloven\u0161\u010dina"),
    "lt": ("Lithuanian", "Lietuvi\u0173"),
    "lv": ("Latvian", "Latvie\u0161u"),
    "et": ("Estonian", "Eesti"),
    "he": ("Hebrew", "\u05e2\u05d1\u05e8\u05d9\u05ea"),
    "hi": ("Hindi", "\u0939\u093f\u0928\u094d\u0926\u0940"),
    "bn": ("Bengali", "\u09ac\u09be\u0982\u09b2\u09be"),
    "ms": ("Malay", "Bahasa Melayu"),
    "fa": ("Persian", "\u0641\u0627\u0631\u0633\u06cc"),
}

# Example translations for Qwen3/3.5 prompt templates
_LANG_EXAMPLES = {
    "zh": ("EN: Thank you very much.", "ZH: \u975e\u5e38\u611f\u8c22\u3002"),
    "en": ("FR: Merci beaucoup.", "EN: Thank you very much."),
    "fr": ("EN: Thank you very much.", "FR: Merci beaucoup."),
    "de": ("EN: Thank you very much.", "DE: Vielen Dank."),
    "es": ("EN: Thank you very much.", "ES: Muchas gracias."),
    "pt": ("EN: Thank you very much.", "PT: Muito obrigado."),
    "it": ("EN: Thank you very much.", "IT: Grazie mille."),
    "ja": ("EN: Thank you very much.", "JA: \u3069\u3046\u3082\u3042\u308a\u304c\u3068\u3046\u3054\u3056\u3044\u307e\u3059\u3002"),
    "ko": ("EN: Thank you very much.", "KO: \ub300\ub2e8\ud788 \uac10\uc0ac\ud569\ub2c8\ub2e4."),
    "ru": ("EN: Thank you very much.", "RU: \u0411\u043e\u043b\u044c\u0448\u043e\u0435 \u0441\u043f\u0430\u0441\u0438\u0431\u043e."),
    "ar": ("EN: Thank you very much.", "AR: \u0634\u0643\u0631\u0627 \u062c\u0632\u064a\u0644\u0627 \u0644\u0643."),
    "nl": ("EN: Thank you very much.", "NL: Hartelijk dank."),
    "cs": ("EN: Thank you very much.", "CS: Mnohokr\u00e1t d\u011bkuji."),
}


# ---------------------------------------------------------------------------
# Prompt format builders
# ---------------------------------------------------------------------------

def _hymt_prompt(src_iso, tgt_iso):
    """Build HY-MT prompt prefix and suffix for a given target language."""
    prefix = _HYMT_PROMPTS.get(tgt_iso)
    if prefix is None:
        return None, None
    return prefix, "<|extra_0|>"


def _qwen3_prompt(src_iso, tgt_iso):
    """Build Qwen3/Qwen3.5 prompt (with <think> block) for any language pair."""
    src_name, _ = _LANG_NAMES.get(src_iso, (src_iso, src_iso))
    tgt_name, tgt_native = _LANG_NAMES.get(tgt_iso, (tgt_iso, tgt_iso))
    example = _LANG_EXAMPLES.get(tgt_iso)
    example_line = ""
    if example:
        example_line = f"Example: {example[0]} \u2192 {example[1]}\n\n"
    prefix = (
        "<|im_start|>user\n"
        f"You are a professional {src_name} to {tgt_name} translator. "
        f"Always output {tgt_name} ({tgt_native}) only.\n"
        f"{example_line}"
        f"Translate the following into {tgt_name}:\n\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return prefix, suffix


def _qwen3_nothink_prompt(src_iso, tgt_iso):
    """Build Qwen3 prompt without thinking (for LoRA SFT models)."""
    src_name, _ = _LANG_NAMES.get(src_iso, (src_iso, src_iso))
    tgt_name, tgt_native = _LANG_NAMES.get(tgt_iso, (tgt_iso, tgt_iso))
    example = _LANG_EXAMPLES.get(tgt_iso)
    example_line = ""
    if example:
        example_line = f"Example: {example[0]} \u2192 {example[1]}\n\n"
    prefix = (
        "<|im_start|>user\n"
        f"You are a professional {src_name} to {tgt_name} translator. "
        f"Always output {tgt_name} ({tgt_native}) only.\n"
        f"{example_line}"
        f"Translate the following into {tgt_name}:\n\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n/no_think\n"
    return prefix, suffix


def _eurollm_prompt(src_iso, tgt_iso):
    """Build EuroLLM system+user prompt for any language pair."""
    src_name, _ = _LANG_NAMES.get(src_iso, (src_iso, src_iso))
    tgt_name, _ = _LANG_NAMES.get(tgt_iso, (tgt_iso, tgt_iso))
    prefix = (
        "<|im_start|>system\n"
        "You are a professional simultaneous interpreter at an academic conference. "
        f"Translate the following {src_name} text into {tgt_name}.<|im_end|>\n"
        "<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n"
    return prefix, suffix


# ---------------------------------------------------------------------------
# PROMPT_FORMATS registry
# ---------------------------------------------------------------------------

PROMPT_FORMATS = {
    "hymt": {
        "builder": _hymt_prompt,
        "stop_tokens": ["<|extra_0|>", "<|endoftext|>"],
    },
    "qwen3": {
        "builder": _qwen3_prompt,
        "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
    },
    "qwen3.5": {
        "builder": _qwen3_prompt,      # same template structure as qwen3
        "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
    },
    "qwen3-nothink": {
        "builder": _qwen3_nothink_prompt,
        "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
    },
    "eurollm": {
        "builder": _eurollm_prompt,
        "stop_tokens": ["<|im_end|>", "</s>"],
    },
    "custom": {
        "builder": None,    # handled specially -- user provides template
        "stop_tokens": ["<|im_end|>", "<|endoftext|>", "</s>"],
    },
}


# ---------------------------------------------------------------------------
# HY-MT prompt lookup by target language (preserved for backward compat)
# ---------------------------------------------------------------------------

_HYMT_PROMPTS = {
    "zh": "\u5c06\u4ee5\u4e0b\u6587\u672c\u7ffb\u8bd1\u4e3a\u4e2d\u6587\uff0c\u6ce8\u610f\u53ea\u9700\u8981\u8f93\u51fa\u7ffb\u8bd1\u540e\u7684\u7ed3\u679c\uff0c\u4e0d\u8981\u989d\u5916\u89e3\u91ca\uff1a\n\n",
    "en": "Translate the following text into English, please only output the translated result without additional explanation:\n\n",
    "fr": "Translate the following text into French, please only output the translated result without additional explanation:\n\n",
    "de": "Translate the following text into German, please only output the translated result without additional explanation:\n\n",
    "es": "Translate the following text into Spanish, please only output the translated result without additional explanation:\n\n",
    "pt": "Translate the following text into Portuguese, please only output the translated result without additional explanation:\n\n",
    "it": "Translate the following text into Italian, please only output the translated result without additional explanation:\n\n",
    "ja": "Translate the following text into Japanese, please only output the translated result without additional explanation:\n\n",
    "ko": "Translate the following text into Korean, please only output the translated result without additional explanation:\n\n",
    "ru": "Translate the following text into Russian, please only output the translated result without additional explanation:\n\n",
    "ar": "Translate the following text into Arabic, please only output the translated result without additional explanation:\n\n",
    "nl": "Translate the following text into Dutch, please only output the translated result without additional explanation:\n\n",
    "pl": "Translate the following text into Polish, please only output the translated result without additional explanation:\n\n",
    "tr": "Translate the following text into Turkish, please only output the translated result without additional explanation:\n\n",
    "vi": "Translate the following text into Vietnamese, please only output the translated result without additional explanation:\n\n",
    "th": "Translate the following text into Thai, please only output the translated result without additional explanation:\n\n",
    "id": "Translate the following text into Indonesian, please only output the translated result without additional explanation:\n\n",
    "cs": "Translate the following text into Czech, please only output the translated result without additional explanation:\n\n",
    "uk": "Translate the following text into Ukrainian, please only output the translated result without additional explanation:\n\n",
    "ro": "Translate the following text into Romanian, please only output the translated result without additional explanation:\n\n",
    "hu": "Translate the following text into Hungarian, please only output the translated result without additional explanation:\n\n",
    "sv": "Translate the following text into Swedish, please only output the translated result without additional explanation:\n\n",
    "da": "Translate the following text into Danish, please only output the translated result without additional explanation:\n\n",
    "fi": "Translate the following text into Finnish, please only output the translated result without additional explanation:\n\n",
    "el": "Translate the following text into Greek, please only output the translated result without additional explanation:\n\n",
    "bg": "Translate the following text into Bulgarian, please only output the translated result without additional explanation:\n\n",
    "hr": "Translate the following text into Croatian, please only output the translated result without additional explanation:\n\n",
    "sk": "Translate the following text into Slovak, please only output the translated result without additional explanation:\n\n",
    "sl": "Translate the following text into Slovenian, please only output the translated result without additional explanation:\n\n",
    "lt": "Translate the following text into Lithuanian, please only output the translated result without additional explanation:\n\n",
    "lv": "Translate the following text into Latvian, please only output the translated result without additional explanation:\n\n",
    "et": "Translate the following text into Estonian, please only output the translated result without additional explanation:\n\n",
    "he": "Translate the following text into Hebrew, please only output the translated result without additional explanation:\n\n",
    "hi": "Translate the following text into Hindi, please only output the translated result without additional explanation:\n\n",
    "bn": "Translate the following text into Bengali, please only output the translated result without additional explanation:\n\n",
    "ms": "Translate the following text into Malay, please only output the translated result without additional explanation:\n\n",
    "fa": "Translate the following text into Persian, please only output the translated result without additional explanation:\n\n",
}

# Backward-compat alias used by KV-cache and LA backends
_PROMPT_SUFFIX = "<|extra_0|>"

# Map NLLB codes to ISO 639-1 codes
_NLLB_TO_ISO = {
    "eng_Latn": "en", "fra_Latn": "fr", "deu_Latn": "de", "spa_Latn": "es",
    "por_Latn": "pt", "ita_Latn": "it", "nld_Latn": "nl", "pol_Latn": "pl",
    "tur_Latn": "tr", "vie_Latn": "vi", "ind_Latn": "id", "ces_Latn": "cs",
    "ron_Latn": "ro", "hun_Latn": "hu", "swe_Latn": "sv", "dan_Latn": "da",
    "fin_Latn": "fi", "ell_Grek": "el", "bul_Cyrl": "bg", "hrv_Latn": "hr",
    "slk_Latn": "sk", "slv_Latn": "sl", "lit_Latn": "lt", "est_Latn": "et",
    "zho_Hans": "zh", "zho_Hant": "zh", "jpn_Jpan": "ja", "kor_Hang": "ko",
    "rus_Cyrl": "ru", "arb_Arab": "ar", "ukr_Cyrl": "uk", "tha_Thai": "th",
    "heb_Hebr": "he", "hin_Deva": "hi", "ben_Beng": "bn", "zsm_Latn": "ms",
    "pes_Arab": "fa", "lvs_Latn": "lv",
}


def _resolve_lang_code(lang_identifier):
    """Convert any language identifier (NLLB, ISO, name) to ISO 639-1 code."""
    if lang_identifier in _HYMT_PROMPTS:
        return lang_identifier
    if lang_identifier in _NLLB_TO_ISO:
        return _NLLB_TO_ISO[lang_identifier]
    # Also accept bare ISO codes that are in _LANG_NAMES but not _HYMT_PROMPTS
    if lang_identifier in _LANG_NAMES:
        return lang_identifier
    # Try matching by lower-cased prefix
    lower = lang_identifier.lower()
    for nllb, iso in _NLLB_TO_ISO.items():
        if nllb.lower().startswith(lower) or lower.startswith(iso):
            return iso
    return None


def _default_heads_path():
    return os.path.join(os.path.dirname(__file__), "heads", "hymt15_7b_universal.json")


def _auto_detect_prompt_format(model_path):
    """Guess prompt format from the model filename/path.

    Returns a format name from PROMPT_FORMATS, or None if no match.
    """
    if model_path is None:
        return None
    lower = model_path.lower()
    # Order matters: check "qwen3.5" before "qwen3" (substring match)
    if "qwen3.5" in lower:
        return "qwen3.5"
    if "qwen3" in lower:
        return "qwen3"
    if "eurollm" in lower:
        return "eurollm"
    if "hymt" in lower or "hy-mt" in lower:
        return "hymt"
    return None


def _build_prompt_config(prompt_format, src_iso, tgt_iso, custom_template=None):
    """Build (prefix, suffix, stop_token_strings) for the given format.

    Parameters
    ----------
    prompt_format : str
        One of the keys in :data:`PROMPT_FORMATS`.
    src_iso : str
        Source language ISO code.
    tgt_iso : str
        Target language ISO code.
    custom_template : str, optional
        Custom prompt template (must contain ``{source}``).  Only used
        when *prompt_format* is ``"custom"``.

    Returns
    -------
    tuple[str, str, list[str]]
        (prefix, suffix, stop_token_strings)
    """
    fmt = PROMPT_FORMATS.get(prompt_format)
    if fmt is None:
        raise ValueError(
            f"Unknown prompt format: {prompt_format!r}. "
            f"Choose from: {', '.join(PROMPT_FORMATS)}"
        )

    if prompt_format == "custom":
        if custom_template is None:
            raise ValueError(
                "prompt_format='custom' requires custom_template (string with {source})"
            )
        if "{source}" not in custom_template:
            raise ValueError("custom_template must contain {source} placeholder")
        parts = custom_template.split("{source}", 1)
        prefix, suffix = parts[0], parts[1]
        return prefix, suffix, fmt["stop_tokens"]

    builder = fmt["builder"]
    prefix, suffix = builder(src_iso, tgt_iso)
    if prefix is None:
        raise ValueError(
            f"Prompt format {prompt_format!r} does not support "
            f"target language {tgt_iso!r}."
        )
    return prefix, suffix, fmt["stop_tokens"]


def _aggregate_ts_weighted_avg(src_attn, ts_scores):
    """Weighted average of attention distributions across heads, then argmax.

    Instead of argmax-per-head-then-vote, this computes a ts-weighted average
    of the full attention distributions, preserving distribution information
    for a more robust alignment signal.
    """
    ts = np.array(ts_scores, dtype=np.float32)
    # Re-normalize attention per head (may not sum to 1 after source slicing)
    row_sums = src_attn.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    normed = src_attn / row_sums
    # Weighted average across heads
    avg_attn = np.dot(ts, normed) / ts.sum()
    return int(np.argmax(avg_attn))


ALIGNATT_SUPPORTED_LANGUAGES = sorted(_HYMT_PROMPTS.keys())


class AlignAttBackend:
    """Simultaneous translation backend using AlignAtt + llama.cpp.

    Conforms to the SimulMTBackend protocol (backend_protocol.py).
    Implements the same translate(text) -> (stable, buffer) interface as
    TranslationBackend, using attention-based border detection instead of
    LocalAgreement prefix stability.

    Args:
        source_lang: Source language (any format: NLLB code, ISO code, or name).
        target_lang: Target language (any format: NLLB code, ISO code, or name).
        model_path: Path to the GGUF model file.
        heads_path: Path to alignment heads JSON (default: bundled universal heads).
        prompt_format: Prompt template format. One of "hymt" (default), "qwen3",
            "qwen3.5", "qwen3-nothink", "eurollm", "custom".  If "auto" or None,
            the format is auto-detected from the model filename.
        custom_template: Custom prompt template string with ``{source}`` placeholder.
            Only used when ``prompt_format="custom"``.
        border_distance: How close to source end before stopping generation.
        word_batch: Number of source words to accumulate before translating.
        n_ctx: Context window size for llama.cpp.
        top_k: Number of attention heads to use.
        entropy_veto_threshold: Normalized entropy threshold (0-1) above which a
            token is considered uncertain and triggers a border-like stop.
            None disables the veto. Recommended value: 0.75.
        verbose: Print debug info.
    """

    def __init__(
        self,
        source_lang,
        target_lang,
        model_path: str = None,
        heads_path: str = None,
        prompt_format: str = "hymt",
        custom_template: str = None,
        border_distance: int = 3,
        word_batch: int = 3,
        n_ctx: int = 2048,
        top_k: int = 10,
        entropy_veto_threshold: float = None,
        lora_path: str = None,
        lora_scale: float = 1.0,
        verbose: bool = False,
        **kwargs,  # Accept extra kwargs for compat with TranslationBackend init patterns
    ):
        # Resolve language codes
        self.source_lang_iso = _resolve_lang_code(source_lang) or source_lang
        self.target_lang_iso = _resolve_lang_code(target_lang)

        self.source_lang = source_lang
        self.target_lang = target_lang
        self.border_distance = border_distance
        self.word_batch = word_batch
        self.n_ctx = n_ctx
        self.top_k = top_k
        self.entropy_veto_threshold = entropy_veto_threshold
        self.verbose = verbose

        # Model path
        if model_path is None:
            model_path = os.environ.get("HYMT_MODEL_PATH")
        if model_path is None:
            raise ValueError(
                "AlignAtt backend requires a GGUF model path. "
                "Set model_path or HYMT_MODEL_PATH env var.\n"
                "Download: huggingface-cli download tencent/HY-MT1.5-7B-GGUF "
                "HY-MT1.5-7B-Q8_0.gguf --local-dir ."
            )
        self.model_path = model_path

        # --- Resolve prompt format ---
        if prompt_format is None or prompt_format == "auto":
            detected = _auto_detect_prompt_format(model_path)
            prompt_format = detected if detected else "hymt"

        self.prompt_format = prompt_format

        # Validate target language for hymt format (only format that restricts
        # target languages to a fixed set of per-language prompts)
        if prompt_format == "hymt":
            if self.target_lang_iso is None or self.target_lang_iso not in _HYMT_PROMPTS:
                raise ValueError(
                    f"Unsupported target language for AlignAtt (hymt format): {target_lang}. "
                    f"Supported: {', '.join(ALIGNATT_SUPPORTED_LANGUAGES)}"
                )

        # Build prompt prefix/suffix and stop tokens for this format
        prefix, suffix, stop_strs = _build_prompt_config(
            prompt_format,
            self.source_lang_iso,
            self.target_lang_iso,
            custom_template=custom_template,
        )
        self._prompt_prefix = prefix
        self._prompt_suffix = suffix
        self._stop_token_strs = stop_strs
        self._custom_template = custom_template

        # Load alignment heads
        heads_file = heads_path or _default_heads_path()
        with open(heads_file) as f:
            data = json.load(f)
        heads = data["token_alignment_heads"][:top_k]
        self._head_layers = [h["layer"] for h in heads]
        self._head_indices = [h["head"] for h in heads]
        self._ts_scores = [h["ts"] for h in heads]
        self._num_heads = len(heads)

        # Init llama.cpp (suppress verbose logs for clean TUI)
        ll.suppress_stderr()
        ll.init()
        self._model = ll.load_model(model_path)
        ll.restore_stderr()

        # Optionally load a LoRA adapter (e.g. fine-tuned for a specific domain)
        self._lora_adapter = None
        self._lora_scale = lora_scale
        if lora_path is not None:
            if ll.has_lora_support():
                import logging
                logger = logging.getLogger(__name__)
                logger.info("Loading LoRA adapter: %s (scale=%.2f)", lora_path, lora_scale)
                self._lora_adapter = ll.load_lora(self._model, lora_path, lora_scale)
            else:
                import warnings
                warnings.warn(
                    f"LoRA adapter requested ({lora_path}) but the loaded "
                    f"libllama does not support the LoRA C API. "
                    f"The adapter will be ignored. Rebuild llama.cpp with "
                    f"LoRA support to enable this feature.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        self._vocab = ll.get_vocab(self._model)
        self._nv = ll.n_vocab(self._vocab)
        self._eos_id = ll.vocab_eos(self._vocab)

        # Build stop token set from format-specific stop strings
        self._stop_ids = {self._eos_id}
        for tok_str in self._stop_token_strs:
            tids = ll.tokenize(self._vocab, tok_str, add_bos=False, special=True)
            if len(tids) == 1:
                self._stop_ids.add(tids[0])

        # Tokens that signal model uncertainty -- treat as soft stop during
        # incremental generation (don't commit), but allow in finish().
        self._uncertain_ids = set()
        for tok_str in ["\u2026", "..."]:
            tids = ll.tokenize(self._vocab, tok_str, add_bos=False, special=True)
            if len(tids) == 1:
                self._uncertain_ids.add(tids[0])

        # Sentence-ending tokens -- commit them but stop generating after.
        # This prevents the model from producing multiple short sentences
        # ("Okay, fine. Let's get started.") in a single translate() call.
        self._sentence_end_ids = set()
        for tok_str in [".", "!", "?", "\u3002", "\uff01", "\uff1f"]:
            tids = ll.tokenize(self._vocab, tok_str, add_bos=False, special=True)
            if len(tids) == 1:
                self._sentence_end_ids.add(tids[0])

        # Pre-tokenize suffix
        self._suffix_tokens = ll.tokenize(
            self._vocab, self._prompt_suffix, add_bos=False, special=True
        )

        # Translation state
        self._source_words = []
        self._committed_ids = []
        self._batch_counter = 0

        # Context injection: rolling buffer of previous (source, translation) pairs
        self._context_buffer = []  # list of {"source": str, "translation": str}
        self._context_window = kwargs.get("context_window", 0)  # 0 = disabled

        # Provide input_buffer attribute for compatibility with code that checks it
        self.input_buffer = []

        # Lock to prevent concurrent access from Textual worker threads
        self._lock = threading.Lock()

        # Context (created once, reused)
        self._ctx = None
        self._mem = None

    def _ensure_context(self):
        if self._ctx is None:
            ll.suppress_stderr()
            self._ctx = ll.create_context(
                self._model, n_ctx=self.n_ctx, n_batch=self.n_ctx
            )
            self._mem = ll.get_memory(self._ctx)
            ll.set_attn_heads(self._ctx, self._head_layers, self._head_indices)
            # Apply LoRA adapter to the new context
            if self._lora_adapter is not None:
                ll.apply_lora(self._ctx, self._lora_adapter, self._lora_scale)
            ll.restore_stderr()

    def _build_context_str(self):
        """Build context injection string from previous translations."""
        if not self._context_buffer or self._context_window <= 0:
            return ""
        entries = self._context_buffer[-self._context_window:]
        ctx_lines = ""
        for entry in entries:
            ctx_lines += f'"{entry["source"]}" -> "{entry["translation"]}"\n'
        return f"[Previous translations]\n{ctx_lines}[New sentence to translate]\n"

    def _find_source_range(self, tokens):
        """Find the source text token range within the full prompt tokens.

        Works for all prompt formats: source is always between the tokenized
        prefix (with BOS) and the tokenized suffix.
        """
        context_str = self._build_context_str()
        prefix_tokens = ll.tokenize(
            self._vocab, self._prompt_prefix + context_str, add_bos=True, special=True
        )
        src_start = len(prefix_tokens)
        src_end = len(tokens) - len(self._suffix_tokens)
        if src_end <= src_start:
            return (0, 0)
        return (src_start, src_end)

    def translate(self, text: Optional[str | TimedText] = None):
        """Translate incrementally. Returns (stable_translation, buffer).

        Matches TranslationBackend.translate() interface:
        - Accepts str or TimedText
        - Returns (stable_text, buffer_text) where stable is the new committed
          translation and buffer is speculative
        """
        # Extract text from input
        if text is None:
            return "", ""
        if isinstance(text, TimedText):
            self.input_buffer.append(text)
            raw_text = text.text or ""
        elif isinstance(text, str):
            self.input_buffer.append(TimedText(text))
            raw_text = text
        else:
            raw_text = str(text)

        # Accumulate source words
        new_words = raw_text.strip().split()
        if not new_words:
            return "", ""

        self._source_words.extend(new_words)
        self._batch_counter += len(new_words)

        # Word batching: wait until we have enough words
        if self._batch_counter < self.word_batch:
            return "", ""
        self._batch_counter = 0

        with self._lock:
            self._ensure_context()

            # Build full prompt with accumulated source (+ optional context)
            accumulated_source = " ".join(self._source_words)
            context_str = self._build_context_str()
            prompt = self._prompt_prefix + context_str + accumulated_source + self._prompt_suffix
            prompt_tokens = ll.tokenize(self._vocab, prompt, add_bos=True, special=True)

            # Find source token range
            src_start, src_end = self._find_source_range(prompt_tokens)
            n_src = max(0, src_end - src_start)

            # Suppress stderr during decode to avoid ggml Metal logs in TUI
            ll.suppress_stderr()
            try:
                # Full re-decode from scratch
                ll.memory_clear(self._mem)
                ll.decode_batch(self._ctx, prompt_tokens)

                # After decode_batch, logits are at batch index n-1
                logit_idx = len(prompt_tokens) - 1

                # Re-decode previously committed tokens
                pos = len(prompt_tokens)
                for tid in self._committed_ids:
                    ll.decode_single(self._ctx, tid, pos)
                    pos += 1
                    logit_idx = 0  # decode_single: logits at batch index 0

                # Generate new tokens with AlignAtt border detection
                new_tokens = []
                buffer_tokens = []
                gen_pos = pos

                # Cap generation proportional to source length to prevent
                # hallucination from insufficient context.
                # Tight cap for short sources (prevents "I would" -> "Je le ferais
                # avec plaisir." hallucination), relaxed for longer sources.
                if n_src <= 6:
                    max_gen = max(2, n_src)
                else:
                    max_gen = max(6, int(n_src * 1.5))
                # Adaptive border distance: scale with source length so that
                # long sentences don't get a too-tight threshold.
                # Short (n_src<=6): use fixed border_distance (default 3)
                # Long (n_src>6): grow proportionally (~15% of source length)
                effective_bd = max(self.border_distance, int(n_src * 0.15))
                border_threshold = n_src - effective_bd
                # Guarantee at least min_commit tokens per call so translation
                # keeps up with source input during typing
                min_commit = max(1, len(self._source_words) // 4)
                consecutive_border_hits = 0

                for i in range(max_gen):
                    logits = ll.get_logits_array(self._ctx, logit_idx, self._nv)
                    if logits is None:
                        break
                    next_id = int(np.argmax(logits))
                    if next_id < 0 or next_id in self._stop_ids:
                        break

                    # Uncertainty tokens (...) signal the model doesn't have enough
                    # context -- treat as soft stop, don't commit
                    if next_id in self._uncertain_ids:
                        break

                    ll.decode_single(self._ctx, next_id, gen_pos)
                    logit_idx = 0  # decode_single: logits at batch index 0
                    gen_pos += 1

                    # Sentence-ending punctuation: commit and stop -- but only
                    # when enough text is committed. Prevents premature sentence
                    # boundaries from short context ("I would" -> "Je le souhaite.")
                    total_committed = len(self._committed_ids) + len(new_tokens)
                    if next_id in self._sentence_end_ids and total_committed >= 8:
                        new_tokens.append(next_id)
                        break

                    # AlignAtt border check -- skip for the first min_commit
                    # tokens to ensure translation progress each call
                    border_hit = False
                    if len(new_tokens) >= min_commit and n_src > 0 and border_threshold > 0:
                        raw_attn = ll.get_attn_weights(
                            self._ctx, 0, self._num_heads, ll.n_ctx(self._ctx)
                        )
                        if raw_attn is not None and raw_attn.shape[1] >= src_end:
                            src_attn = raw_attn[:, src_start:src_end]
                            if src_attn.shape[1] > 0:
                                voted_pos = _aggregate_ts_weighted_avg(src_attn, self._ts_scores)
                                if voted_pos >= border_threshold:
                                    border_hit = True

                    # Entropy veto: if the model is uncertain about this
                    # token (high normalized entropy), treat it like a
                    # border hit to prevent committing uncertain output.
                    entropy_veto = False
                    H_norm = None
                    if (
                        self.entropy_veto_threshold is not None
                        and len(new_tokens) >= min_commit
                    ):
                        shifted = logits - logits.max()
                        log_Z = np.log(np.exp(shifted).sum())
                        log_p = shifted - log_Z
                        H_norm = -np.dot(np.exp(log_p), log_p) / np.log(len(logits))
                        if H_norm > self.entropy_veto_threshold:
                            entropy_veto = True

                    if self.verbose and H_norm is not None:
                        tok_str = ll.token_to_piece(self._vocab, next_id)
                        print(
                            f"  [entropy] token={tok_str!r} H_norm={H_norm:.4f}"
                            f" threshold={self.entropy_veto_threshold}"
                            f" veto={entropy_veto}"
                        )

                    if border_hit or entropy_veto:
                        consecutive_border_hits += 1
                        if consecutive_border_hits >= 2:
                            buffer_tokens.append(next_id)
                            break
                    else:
                        consecutive_border_hits = 0

                    new_tokens.append(next_id)
            finally:
                ll.restore_stderr()

            # Compute stable delta
            prev_text = ll.tokens_to_text(
                self._vocab, self._committed_ids, errors="ignore"
            ) if self._committed_ids else ""
            self._committed_ids.extend(new_tokens)
            full_committed = ll.tokens_to_text(
                self._vocab, self._committed_ids, errors="ignore"
            ) if self._committed_ids else ""
            stable = full_committed[len(prev_text):]

            # Buffer text
            buffer = ""
            if buffer_tokens:
                buffer = ll.tokens_to_text(self._vocab, buffer_tokens, errors="ignore")

        if self.verbose:
            entropy_tag = ""
            if self.entropy_veto_threshold is not None:
                entropy_tag = f" [evt={self.entropy_veto_threshold}]"
            print(
                f" \033[32m{stable}\033[0m \033[35m{buffer}\033[0m{entropy_tag}"
            )

        return stable, buffer

    def reset(self, hard=False):
        """Reset state for a new sentence.

        If context_window > 0, saves the completed translation as context
        for the next sentence (soft reset). Use hard=True to clear context too.
        """
        # Save context from completed sentence before clearing
        if self._context_window > 0 and self._source_words and self._committed_ids and not hard:
            source_text = " ".join(self._source_words)
            translation = ll.tokens_to_text(
                self._vocab, self._committed_ids, errors="ignore"
            ) if self._committed_ids else ""
            if source_text.strip() and translation.strip():
                self._context_buffer.append({
                    "source": source_text,
                    "translation": translation,
                })
                # Trim to window size
                if len(self._context_buffer) > self._context_window:
                    self._context_buffer = self._context_buffer[-self._context_window:]

        if self._ctx is not None:
            ll.free_context(self._ctx)
            self._ctx = None
            self._mem = None
        self._source_words = []
        self._committed_ids = []
        self._batch_counter = 0
        self.input_buffer = []

        if hard:
            self._context_buffer = []

    def finish(self):
        """Flush remaining translation (generate until EOS).

        Returns the remaining translation text.
        """
        if not self._source_words:
            return ""

        with self._lock:
            self._ensure_context()

            accumulated_source = " ".join(self._source_words)
            context_str = self._build_context_str()
            prompt = self._prompt_prefix + context_str + accumulated_source + self._prompt_suffix
            prompt_tokens = ll.tokenize(self._vocab, prompt, add_bos=True, special=True)

            ll.suppress_stderr()
            try:
                ll.memory_clear(self._mem)
                ll.decode_batch(self._ctx, prompt_tokens)

                # After decode_batch, logits at batch index n-1
                logit_idx = len(prompt_tokens) - 1

                # Re-decode committed tokens
                pos = len(prompt_tokens)
                for tid in self._committed_ids:
                    ll.decode_single(self._ctx, tid, pos)
                    pos += 1
                    logit_idx = 0

                # Generate until EOS
                new_tokens = []
                for _ in range(200):
                    next_id = ll.argmax_logits(self._ctx, logit_idx, self._nv)
                    if next_id < 0 or next_id in self._stop_ids:
                        break
                    new_tokens.append(next_id)
                    ll.decode_single(self._ctx, next_id, pos)
                    logit_idx = 0
                    pos += 1
            finally:
                ll.restore_stderr()

            prev_text = ll.tokens_to_text(
                self._vocab, self._committed_ids, errors="ignore"
            ) if self._committed_ids else ""
            self._committed_ids.extend(new_tokens)
            full_text = ll.tokens_to_text(
                self._vocab, self._committed_ids, errors="ignore"
            ) if self._committed_ids else ""
            remaining = full_text[len(prev_text):]

            return remaining

    def set_target_lang(self, target_lang):
        """Change the target language and reset state."""
        iso = _resolve_lang_code(target_lang)
        if iso is None:
            raise ValueError(f"Unsupported target language: {target_lang}")
        # Rebuild prompt config for the new target language
        prefix, suffix, stop_strs = _build_prompt_config(
            self.prompt_format,
            self.source_lang_iso,
            iso,
            custom_template=self._custom_template,
        )
        self.target_lang = target_lang
        self.target_lang_iso = iso
        self._prompt_prefix = prefix
        self._prompt_suffix = suffix
        self._stop_token_strs = stop_strs
        # Rebuild stop token IDs
        self._stop_ids = {self._eos_id}
        for tok_str in self._stop_token_strs:
            tids = ll.tokenize(self._vocab, tok_str, add_bos=False, special=True)
            if len(tids) == 1:
                self._stop_ids.add(tids[0])
        # Re-tokenize suffix
        self._suffix_tokens = ll.tokenize(
            self._vocab, self._prompt_suffix, add_bos=False, special=True
        )
        self.reset()

    def __del__(self):
        if hasattr(self, "_ctx") and self._ctx is not None:
            try:
                ll.free_context(self._ctx)
            except Exception:
                pass
        if hasattr(self, "_model") and self._model is not None:
            try:
                ll.free_model(self._model)
            except Exception:
                pass
