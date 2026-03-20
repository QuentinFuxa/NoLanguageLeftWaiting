"""Unified backend protocol for simultaneous translation.

Defines the abstract interface that all translation backends must implement.
This allows OnlineTranslation, evaluators, and simulators to work with any
backend (NLLB/TranslationBackend, AlignAtt, future backends) uniformly.

Usage:
    from nllw.backend_protocol import SimulMTBackend, create_backend

    backend = create_backend("alignatt", source_lang="en", target_lang="fr",
                             model_path="/path/to/model.gguf")
    stable, buffer = backend.translate("hello ")
    remaining = backend.finish()
    backend.reset()
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

from nllw.timed_text import TimedText


class SimulMTBackend(ABC):
    """Abstract base class for simultaneous machine translation backends.

    All backends must implement translate(), finish(), reset(), and
    set_target_lang(). The translate() method is called incrementally as
    source words arrive, returning (stable_text, buffer_text) where stable
    is committed and buffer is speculative.
    """

    @abstractmethod
    def translate(self, text: Optional[Union[str, TimedText]] = None) -> Tuple[str, str]:
        """Translate an incremental chunk of source text.

        Args:
            text: New source text (word or phrase with trailing space).

        Returns:
            (stable, buffer) where stable is the new committed translation
            and buffer is speculative text that may change.
        """
        ...

    @abstractmethod
    def finish(self) -> str:
        """Flush remaining translation (generate until EOS).

        Returns:
            The remaining translation text.
        """
        ...

    @abstractmethod
    def reset(self):
        """Reset state for a new sentence."""
        ...

    @abstractmethod
    def set_target_lang(self, target_lang: str):
        """Change the target language (also resets state)."""
        ...


# --- Factory ---

def create_backend(
    backend_type: str,
    source_lang: str,
    target_lang: str,
    *,
    model_path: str = None,
    heads_path: str = None,
    prompt_format: str = "hymt",
    custom_template: str = None,
    model=None,
    tokenizer=None,
    nllb_size: str = "600M",
    border_distance: int = 3,
    word_batch: int = 2,
    lora_path: str = None,
    lora_scale: float = 1.0,
    verbose: bool = False,
    **kwargs,
) -> SimulMTBackend:
    """Create a translation backend by type.

    Args:
        backend_type: One of "alignatt", "alignatt-kv", "alignatt-la", "wait-k",
            "full-sentence", "eager", "transformers", "ctranslate2".
        source_lang: Source language (NLLB code, ISO code, or name).
        target_lang: Target language.
        model_path: Path to GGUF model (for alignatt) or HF model name.
        heads_path: Path to alignment heads JSON (alignatt only).
        prompt_format: Prompt template format for AlignAtt backends. One of
            "hymt" (default), "qwen3", "qwen3.5", "qwen3-nothink", "eurollm",
            "custom", or "auto" (auto-detect from model filename).
        custom_template: Custom prompt template with ``{source}`` placeholder
            (only used when prompt_format="custom").
        model: Pre-loaded model object (NLLB only).
        tokenizer: Pre-loaded tokenizer (NLLB only).
        nllb_size: NLLB model size (NLLB only).
        border_distance: Border distance for AlignAtt.
        word_batch: Word batch size.
        lora_path: Path to a GGUF LoRA adapter file (alignatt backends only).
        lora_scale: LoRA adapter weight, 1.0 = full strength.
        verbose: Enable debug output.

    Returns:
        A SimulMTBackend instance.
    """
    if backend_type == "alignatt":
        from nllw.alignatt_backend import AlignAttBackend
        return AlignAttBackend(
            source_lang=source_lang,
            target_lang=target_lang,
            model_path=model_path,
            heads_path=heads_path,
            prompt_format=prompt_format,
            custom_template=custom_template,
            border_distance=border_distance,
            word_batch=word_batch,
            lora_path=lora_path,
            lora_scale=lora_scale,
            verbose=verbose,
            **kwargs,
        )
    elif backend_type == "alignatt-kv":
        from nllw.alignatt_kvcache_backend import AlignAttKVCacheBackend
        return AlignAttKVCacheBackend(
            source_lang=source_lang,
            target_lang=target_lang,
            model_path=model_path,
            heads_path=heads_path,
            prompt_format=prompt_format,
            custom_template=custom_template,
            border_distance=border_distance,
            word_batch=word_batch,
            lora_path=lora_path,
            lora_scale=lora_scale,
            verbose=verbose,
            **kwargs,
        )
    elif backend_type == "alignatt-la":
        from nllw.alignatt_la_backend import AlignAttLocalAgreementBackend
        return AlignAttLocalAgreementBackend(
            source_lang=source_lang,
            target_lang=target_lang,
            model_path=model_path,
            heads_path=heads_path,
            prompt_format=prompt_format,
            custom_template=custom_template,
            border_distance=border_distance,
            word_batch=word_batch,
            lora_path=lora_path,
            lora_scale=lora_scale,
            verbose=verbose,
            **kwargs,
        )
    elif backend_type == "wait-k":
        from nllw.baselines import WaitKBackend
        return WaitKBackend(
            source_lang=source_lang,
            target_lang=target_lang,
            model_path=model_path,
            heads_path=heads_path,
            verbose=verbose,
            **kwargs,
        )
    elif backend_type == "full-sentence":
        from nllw.baselines import FullSentenceBackend
        return FullSentenceBackend(
            source_lang=source_lang,
            target_lang=target_lang,
            model_path=model_path,
            heads_path=heads_path,
            verbose=verbose,
            **kwargs,
        )
    elif backend_type == "eager":
        from nllw.baselines import EagerBackend
        return EagerBackend(
            source_lang=source_lang,
            target_lang=target_lang,
            model_path=model_path,
            heads_path=heads_path,
            verbose=verbose,
            **kwargs,
        )
    elif backend_type in ("transformers", "ctranslate2"):
        from nllw.core import TranslationBackend
        return TranslationBackend(
            source_lang=source_lang,
            target_lang=target_lang,
            model=model,
            tokenizer=tokenizer,
            backend_type=backend_type,
            verbose=verbose,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown backend type: {backend_type!r}. "
            f"Use 'alignatt', 'alignatt-kv', 'alignatt-la', 'wait-k', "
            f"'full-sentence', 'eager', 'transformers', or 'ctranslate2'."
        )
