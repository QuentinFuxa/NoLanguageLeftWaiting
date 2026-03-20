"""Baseline SimulMT policies for comparison with AlignAtt.

Implements standard policies from the SimulMT literature:
    - wait-k (Ma et al., 2019): Read k words, then alternate READ/WRITE
    - fixed-rate: Emit every N source words (simple but poor quality)

These are registered as backends and can be used via the factory:
    create_backend(BackendConfig(backend_type="wait-k", ...))
"""

from typing import List, Optional

from .backend_protocol import (
    SimulMTBackend,
    BackendConfig,
    TranslationStep,
    register_backend,
)


@register_backend("wait-k")
class WaitKBackend(SimulMTBackend):
    """Wait-k simultaneous translation policy.

    Classic policy (Ma et al., 2019):
        1. READ k source words before producing any output
        2. After that, alternate: READ 1 word, WRITE 1 word
        3. On is_final, flush all remaining output

    Uses AlignAtt internally for actual translation, but controls the
    read/write schedule externally.

    Config params:
        wait_k: Number of initial source words to wait (default: 5)
        word_batch: How many words to write per step after initial wait (default: 1)
    """

    def __init__(self, config: BackendConfig):
        super().__init__(config)
        self._source_words: List[str] = []
        self._words_since_write = 0
        self._inner = None

    def translate(self, source_word: str, is_final: bool = False,
                  emission_time: float = 0.0) -> TranslationStep:
        self._source_words.append(source_word)
        self._words_since_write += 1

        n_words = len(self._source_words)

        # Phase 1: Wait for k words
        if n_words < self.config.wait_k and not is_final:
            return TranslationStep(
                text="",
                is_final=False,
                source_words_seen=n_words,
            )

        # Phase 2: Translate after k words, or on final
        if self._inner is None:
            self._init_inner()

        # Feed accumulated words to inner backend
        result = self._inner.translate(source_word, is_final=is_final, emission_time=emission_time)

        # After initial wait, only write every word_batch words
        if not is_final and self._words_since_write < self.config.word_batch:
            return TranslationStep(
                text="",
                is_final=False,
                source_words_seen=n_words,
            )

        self._words_since_write = 0
        return TranslationStep(
            text=result.text,
            is_final=is_final,
            committed_tokens=result.committed_tokens,
            stopped_at_border=False,
            source_words_seen=n_words,
            generation_time_ms=result.generation_time_ms,
        )

    def _init_inner(self):
        """Lazily create the inner AlignAtt backend."""
        from .alignatt_backend import AlignAttBackend
        self._inner = AlignAttBackend(self.config)

    def reset(self):
        self._source_words = []
        self._words_since_write = 0
        if self._inner:
            self._inner.reset()

    def get_full_translation(self) -> str:
        if self._inner:
            return self._inner.get_full_translation()
        return ""

    def close(self):
        if self._inner:
            self._inner.close()


@register_backend("fixed-rate")
class FixedRateBackend(SimulMTBackend):
    """Fixed-rate emission policy.

    Emits translation output at fixed intervals of N source words.
    Simple baseline that doesn't use attention at all.

    Config params:
        word_batch: Emit every N words (default: 3)
    """

    def __init__(self, config: BackendConfig):
        super().__init__(config)
        self._source_words: List[str] = []
        self._word_count = 0
        self._inner = None

    def translate(self, source_word: str, is_final: bool = False,
                  emission_time: float = 0.0) -> TranslationStep:
        self._source_words.append(source_word)
        self._word_count += 1

        n_words = len(self._source_words)

        # Only emit at fixed intervals or on final
        if not is_final and self._word_count < self.config.word_batch:
            return TranslationStep(
                text="",
                is_final=False,
                source_words_seen=n_words,
            )

        self._word_count = 0

        if self._inner is None:
            self._init_inner()

        result = self._inner.translate(source_word, is_final=is_final, emission_time=emission_time)
        return result

    def _init_inner(self):
        from .alignatt_backend import AlignAttBackend
        # Use border_distance=0 to generate freely (no border detection)
        inner_config = BackendConfig(**{
            **self.config.__dict__,
            "border_distance": 0,
        })
        self._inner = AlignAttBackend(inner_config)

    def reset(self):
        self._source_words = []
        self._word_count = 0
        if self._inner:
            self._inner.reset()

    def get_full_translation(self) -> str:
        if self._inner:
            return self._inner.get_full_translation()
        return ""

    def close(self):
        if self._inner:
            self._inner.close()
