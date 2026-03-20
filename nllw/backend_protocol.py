"""SimulMT backend protocol and factory.

All simultaneous translation backends implement the SimulMTBackend ABC.
The create_backend() factory instantiates the right backend from a config dict.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class TranslationStep:
    """Result of a single translate() call."""
    text: str                    # New committed text (may be empty if border hit early)
    is_final: bool = False       # True if this completes a sentence segment
    committed_tokens: int = 0    # Number of new tokens committed
    stopped_at_border: bool = False  # True if AlignAtt border was hit
    source_words_seen: int = 0   # Number of source words processed so far
    generation_time_ms: float = 0.0  # Wall-clock time for this step


@dataclass
class BackendConfig:
    """Configuration for a SimulMT backend."""
    backend_type: str = "alignatt"
    model_path: str = ""
    heads_path: str = ""
    prompt_format: str = "hymt"
    direction: str = "en-zh"
    # AlignAtt parameters
    border_distance: int = 3
    word_batch: int = 3
    top_k_heads: int = 10
    # Context
    context_sentences: int = 0
    # Generation
    max_new_per_step: int = 50
    n_ctx: int = 2048
    # KV cache
    use_kvcache: bool = True
    # Entropy veto
    entropy_veto_threshold: Optional[float] = None
    # Wait-k policy
    wait_k: int = 5
    # Target language (for output validation)
    target_lang: str = "zh"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BackendConfig":
        """Create config from a dict, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


class SimulMTBackend(ABC):
    """Abstract base class for simultaneous MT backends.

    Lifecycle:
        1. __init__(config) -- load model, set up state
        2. translate(word) -- called for each new source word
        3. reset() -- called at sentence boundaries
        4. close() -- free resources
    """

    def __init__(self, config: BackendConfig):
        self.config = config

    @abstractmethod
    def translate(self, source_word: str, is_final: bool = False,
                  emission_time: float = 0.0) -> TranslationStep:
        """Process a new source word and return any new translation.

        Args:
            source_word: The new source word to add
            is_final: True if this word ends the current sentence/segment
            emission_time: ASR emission timestamp (for latency metrics)

        Returns:
            TranslationStep with new committed text
        """
        ...

    @abstractmethod
    def reset(self):
        """Reset state for next sentence segment.

        Called after is_final=True translation completes.
        Should preserve context if context_sentences > 0.
        """
        ...

    def close(self):
        """Free resources (model, context, etc.). Override if needed."""
        pass

    def get_full_translation(self) -> str:
        """Get the full committed translation so far."""
        return ""

    @property
    def name(self) -> str:
        return self.config.backend_type


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_BACKEND_REGISTRY: Dict[str, type] = {}


def register_backend(name: str):
    """Decorator to register a backend class."""
    def decorator(cls):
        _BACKEND_REGISTRY[name] = cls
        return cls
    return decorator


def create_backend(config: BackendConfig) -> SimulMTBackend:
    """Create a backend from config.

    Registered types:
        - "alignatt": AlignAtt border detection (primary)
        - "alignatt-kv": AlignAtt with KV cache (speed)
        - "wait-k": Wait-k baseline
        - "full-sentence": Full sentence baseline (quality upper bound)
        - "eager": Eager baseline (latency lower bound)
    """
    backend_type = config.backend_type
    if backend_type not in _BACKEND_REGISTRY:
        available = ", ".join(sorted(_BACKEND_REGISTRY.keys()))
        raise ValueError(
            f"Unknown backend type '{backend_type}'. Available: {available}"
        )
    return _BACKEND_REGISTRY[backend_type](config)


def list_backends() -> List[str]:
    """List registered backend type names."""
    return sorted(_BACKEND_REGISTRY.keys())
