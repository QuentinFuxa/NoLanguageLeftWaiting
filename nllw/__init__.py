from .translation import (
    OnlineTranslation,
    MIN_SILENCE_DURATION_DEL_BUFFER,
)

from .core import load_model, TranslationModel, TranslationBackend

from .languages import (
    get_nllb_code,
    get_language_code_code,
    get_language_name_by_language_code,
    get_language_name_by_nllb,
    get_language_info,
    list_all_languages,
    list_all_nllb_codes,
    list_all_language_code_codes,
    LANGUAGES,
)

from .timed_text import TimedText
from .backend_protocol import SimulMTBackend, create_backend

__all__ = [
    "load_model",
    "OnlineTranslation",
    "TranslationModel",
    "TimedText",
    "MIN_SILENCE_DURATION_DEL_BUFFER",
    "TranslationBackend",
    "AlignAttBackend",
    "AlignAttKVCacheBackend",
    "AlignAttLocalAgreementBackend",
    "SimulMTBackend",
    "create_backend",
    "LoRAConfig",
    "list_lora_adapters",
    "get_nllb_code",
    "get_language_code_code",
    "get_language_name_by_language_code",
    "get_language_name_by_nllb",
    "get_language_info",
    "list_all_languages",
    "list_all_nllb_codes",
    "list_all_language_code_codes",
    "LANGUAGES",
]


def __getattr__(name):
    # Lazy imports to avoid loading heavy deps at import time
    if name == "AlignAttBackend":
        from .alignatt_backend import AlignAttBackend
        return AlignAttBackend
    if name == "AlignAttKVCacheBackend":
        from .alignatt_kvcache_backend import AlignAttKVCacheBackend
        return AlignAttKVCacheBackend
    if name == "AlignAttLocalAgreementBackend":
        from .alignatt_la_backend import AlignAttLocalAgreementBackend
        return AlignAttLocalAgreementBackend
    if name == "SimulMTEvaluator":
        from .eval import SimulMTEvaluator
        return SimulMTEvaluator
    if name == "SimulMTSimulator":
        from .simulate import SimulMTSimulator
        return SimulMTSimulator
    if name == "LoRAConfig":
        from .lora import LoRAConfig
        return LoRAConfig
    if name == "list_lora_adapters":
        from .lora import list_lora_adapters
        return list_lora_adapters
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
