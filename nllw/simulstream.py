"""SimulStream SpeechProcessor wrapper for IWSLT 2026 submission.

Wraps the NLLW SimulMT backend as a SimulStream SpeechProcessor, enabling
integration with the official IWSLT 2026 evaluation pipeline.

SimulStream replaces SimulEval for IWSLT 2026. Our SpeechProcessor receives
audio chunks, runs ASR to extract words, then feeds them through our AlignAtt
backend for simultaneous translation.

Architecture:
    audio chunk -> ASR buffer -> word extraction -> AlignAtt translate()
    -> IncrementalOutput (new target text)

The wrapper handles:
    - Audio buffering (accumulate until min_start_seconds)
    - ASR integration (Qwen3-ASR or external)
    - Word-level feeding to SimulMT backend
    - Output formatting (IncrementalOutput with tokens/text/deletions)
    - State management (clear between talks)

Usage:
    # As SimulStream processor
    simulstream.server --speech-processor nllw.simulstream:NLLWSpeechProcessor

    # Standalone test (no SimulStream dependency)
    python -m nllw.simulstream --model /path/to.gguf --lang en-zh --test

Reference:
    - SimulStream: https://github.com/hlt-mt/simulstream
    - IWSLT 2026 SST track: Simultaneous Speech Translation
    - Docker submission: agent_simulstream.py in submission image
"""

import os
import re
import sys
import time
import json
import logging
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

import numpy as np

from .backend_protocol import BackendConfig, TranslationStep, create_backend

# Import backends so they register with the factory
import nllw.alignatt_backend  # noqa: F401
import nllw.alignatt_la_backend  # noqa: F401

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IncrementalOutput: our own dataclass matching SimulStream's interface
# ---------------------------------------------------------------------------

@dataclass
class IncrementalOutput:
    """Output from a single process_chunk() or end_of_stream() call.

    Matches the SimulStream IncrementalOutput protocol:
    - new_tokens: newly emitted target tokens (as strings)
    - new_string: concatenated text of new tokens
    - deleted_tokens: tokens to revise (for re-translation backends)
    - deleted_string: concatenated text of deleted tokens
    """
    new_tokens: List[str] = field(default_factory=list)
    new_string: str = ""
    deleted_tokens: List[str] = field(default_factory=list)
    deleted_string: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.new_string and not self.deleted_string


@dataclass
class EmissionEvent:
    """A single emission event in the longform output stream.

    Used to build OmniSTEval JSONL output for competition evaluation.
    Records per-word emission times for LongYAAL computation.
    """
    emission_time: float     # ASR emission time of triggering source word (ms)
    wall_clock: float        # Wall-clock time of translation output (ms)
    text: str                # Emitted target text
    is_final: bool = False   # True if this is end of a sentence segment
    status: str = "COMPLETE" # COMPLETE or PARTIAL


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SimulStreamConfig:
    """Configuration for the SimulStream wrapper.

    Combines MT backend config with ASR + streaming parameters.
    """
    # MT backend
    model_path: str = ""
    heads_path: str = ""
    direction: str = "en-zh"
    backend_type: str = "alignatt"
    border_distance: int = 3
    word_batch: int = 4
    aggregation: str = "top_p"
    top_p_threshold: float = 0.85
    prompt_format: str = "hymt"
    context_sentences: int = 0
    n_ctx: int = 2048
    repetition_max_repeats: Optional[int] = None  # Disabled: hurts EN-ZH by -0.004 COMET

    # All BackendConfig params are forwarded
    extra_backend_params: Dict[str, Any] = field(default_factory=dict)

    # Longform mode (IWSLT 2026 competition format)
    longform: bool = True           # Accumulate output across sentences, only clear() between recordings
    auto_sentence_boundary: bool = True  # Auto-detect sentence boundaries from target output

    # Audio / ASR
    sample_rate: int = 16000
    min_start_seconds: float = 0.5  # Buffer before first ASR attempt
    speech_chunk_size: int = 960    # Samples per chunk (960 = 60ms at 16kHz)
    asr_model_path: str = ""        # Path to ASR model (if using built-in ASR)
    asr_type: str = "external"      # "external" = words come pre-extracted,
                                    # "qwen3" = use Qwen3-ASR

    # Per-direction optimal configs (loaded from YAML)
    configs_dir: str = ""

    def to_backend_config(self) -> BackendConfig:
        """Convert to BackendConfig for the MT backend."""
        parts = self.direction.split("-")
        target_lang = parts[1] if len(parts) >= 2 else "zh"
        d = {
            "backend_type": self.backend_type,
            "model_path": self.model_path,
            "heads_path": self.heads_path,
            "direction": self.direction,
            "border_distance": self.border_distance,
            "word_batch": self.word_batch,
            "aggregation": self.aggregation,
            "top_p_threshold": self.top_p_threshold,
            "prompt_format": self.prompt_format,
            "context_sentences": self.context_sentences,
            "n_ctx": self.n_ctx,
            "target_lang": target_lang,
        }
        if self.repetition_max_repeats is not None:
            d["repetition_max_repeats"] = self.repetition_max_repeats
        d.update(self.extra_backend_params)
        return BackendConfig.from_dict(d)

    @classmethod
    def from_yaml(cls, path: str) -> "SimulStreamConfig":
        """Load configuration from a YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        # Separate backend params from simulstream params
        ss_keys = {f.name for f in cls.__dataclass_fields__.values()}
        ss_params = {k: v for k, v in data.items() if k in ss_keys}
        extra = {k: v for k, v in data.items() if k not in ss_keys}
        ss_params["extra_backend_params"] = extra
        return cls(**ss_params)


# ---------------------------------------------------------------------------
# Direction-specific optimal configs
# ---------------------------------------------------------------------------

DIRECTION_DEFAULTS: Dict[str, Dict[str, Any]] = {
    # Updated iteration 17: top_p + tuned p_threshold per direction
    # See SHARED_TASK_NOTES.md for full benchmark results
    "en-zh": {
        "border_distance": 3,
        "word_batch": 4,
        "aggregation": "top_p",
        "top_p_threshold": 0.85,  # COMET 0.896 (=offline baseline!)
        "prompt_format": "hymt",
    },
    "en-de": {
        "border_distance": 2,
        "word_batch": 3,
        "aggregation": "top_p",
        "top_p_threshold": 0.75,  # COMET 0.881, lower latency
        "prompt_format": "hymt",
    },
    "en-it": {
        "border_distance": 2,
        "word_batch": 3,
        "aggregation": "top_p",
        "top_p_threshold": 0.9,   # COMET 0.891
        "prompt_format": "hymt",
    },
    "cs-en": {
        "border_distance": 3,
        "word_batch": 3,
        "aggregation": "top_p",
        "top_p_threshold": 0.9,   # COMET 0.876
        "prompt_format": "hymt",
    },
}


# ---------------------------------------------------------------------------
# NLLWSpeechProcessor: the main SimulStream wrapper
# ---------------------------------------------------------------------------

class NLLWSpeechProcessor:
    """SimulStream SpeechProcessor wrapping NLLW's AlignAtt backend.

    This class implements the full SimulStream protocol for IWSLT 2026.
    It can be used with or without the simulstream package:
    - With SimulStream: subclass SpeechProcessor and register
    - Without SimulStream: use directly as a text-mode processor

    SimulStream requires these 7 methods:
        load_model(cls, config) -> cls       # Class method, load models
        process_chunk(waveform) -> Output    # Process audio chunk
        end_of_stream() -> Output            # Flush remaining output
        set_source_language(lang)            # Set source language
        set_target_language(lang)            # Set target language
        tokens_to_string(tokens) -> str      # Detokenize
        clear()                              # Reset state between talks

    Plus property: speech_chunk_size -> float (seconds per chunk)

    The wrapper operates in two modes:
    1. Audio mode (process_chunk receives np.float32 waveform)
       -> buffers audio, runs ASR, feeds words to MT
    2. Text mode (process_words receives pre-extracted words)
       -> directly feeds words to MT (for testing and evaluation)

    Environment variables for Docker (override config):
        NLLW_MODEL_PATH: Path to GGUF model file
        NLLW_HEADS_DIR: Directory containing head config JSONs
        NLLW_CONFIGS_DIR: Directory with per-direction YAML configs
        NLLW_N_GPU_LAYERS: Number of GPU layers (default: 99)
        NLLW_DEFAULT_DIRECTION: Default language direction (default: en-zh)
    """

    # Sentence-ending punctuation for auto boundary detection (by target language)
    _SENTENCE_END_PUNCT = {
        "zh": set("。？！"),
        "ja": set("。？！"),
        "ko": set("。？！"),
        "default": set(".?!"),
    }

    def __init__(self, config: SimulStreamConfig):
        self.config = config
        self._backend = None
        self._audio_buffer = np.array([], dtype=np.float32)
        self._words_buffer: List[str] = []
        self._committed_text = ""
        self._prev_text = ""
        self._current_text = ""
        self._n_chunks_processed = 0
        self._total_audio_seconds = 0.0
        self._is_initialized = False
        self._source_lang = config.direction.split("-")[0]
        self._target_lang = config.direction.split("-")[1] if "-" in config.direction else "zh"
        self._word_emission_times: List[float] = []  # Track emission times for LongYAAL

        # Longform mode: emission tracking for OmniSTEval
        self._emission_log: List[EmissionEvent] = []  # Per-event log for the current recording
        self._recording_start_time: float = 0.0       # Wall-clock start of current recording
        self._recording_text: str = ""                 # Accumulated text for the whole recording
        self._n_sentences_in_recording: int = 0        # Sentence count for context tracking

    @classmethod
    def load_model(cls, config=None) -> "NLLWSpeechProcessor":
        """Factory method matching SimulStream's load_model pattern.

        Accepts SimulStreamConfig, dict, or None (uses env vars).
        This flexibility ensures compatibility with different SimulStream
        server implementations.

        Args:
            config: SimulStreamConfig, dict with config keys, or None.
                    If None, reads from environment variables.

        Returns:
            Initialized NLLWSpeechProcessor ready for process_chunk() calls.
        """
        if config is None:
            config = cls._config_from_env()
        elif isinstance(config, dict):
            config = cls._config_from_dict(config)
        elif not isinstance(config, SimulStreamConfig):
            # Try to convert from any config-like object
            try:
                config = cls._config_from_dict(vars(config))
            except Exception:
                logger.warning("Unknown config type %s, using defaults", type(config))
                config = cls._config_from_env()

        processor = cls(config)
        processor._initialize()
        return processor

    @classmethod
    def _config_from_env(cls) -> SimulStreamConfig:
        """Build config from environment variables (Docker-friendly)."""
        direction = os.environ.get("NLLW_DEFAULT_DIRECTION", "en-zh")
        direction_cfg = DIRECTION_DEFAULTS.get(direction, {})

        model_path = os.environ.get("NLLW_MODEL_PATH", "/app/models/hymt1.5-7b-q8_0.gguf")
        heads_dir = os.environ.get("NLLW_HEADS_DIR", "/app/heads")
        n_gpu = int(os.environ.get("NLLW_N_GPU_LAYERS", "99"))

        # Auto-detect heads path from direction
        src, tgt = direction.split("-") if "-" in direction else ("en", "zh")
        heads_path = os.path.join(heads_dir, f"translation_heads_hymt_en_{tgt}.json")
        if not os.path.exists(heads_path):
            # Try alternative naming
            for pattern in [
                f"translation_heads_hy_mt1_5_7b_q8_0_{src}_{tgt}.json",
                f"translation_heads_hymt_{src}_{tgt}.json",
            ]:
                alt = os.path.join(heads_dir, pattern)
                if os.path.exists(alt):
                    heads_path = alt
                    break

        config = SimulStreamConfig(
            model_path=model_path,
            heads_path=heads_path,
            direction=direction,
            **{k: v for k, v in direction_cfg.items()
               if k in SimulStreamConfig.__dataclass_fields__},
        )
        config.extra_backend_params["n_gpu_layers"] = n_gpu

        logger.info(
            "Config from env: model=%s, heads=%s, direction=%s, n_gpu=%d",
            model_path, heads_path, direction, n_gpu,
        )
        return config

    @classmethod
    def _config_from_dict(cls, d: dict) -> SimulStreamConfig:
        """Build SimulStreamConfig from a dictionary."""
        ss_keys = {f.name for f in SimulStreamConfig.__dataclass_fields__.values()}
        ss_params = {k: v for k, v in d.items() if k in ss_keys}
        extra = {k: v for k, v in d.items() if k not in ss_keys}
        if extra:
            ss_params.setdefault("extra_backend_params", {}).update(extra)
        return SimulStreamConfig(**ss_params)

    def _initialize(self):
        """Initialize the MT backend.

        Handles missing model gracefully for testing without GPU.
        """
        if self._is_initialized:
            return

        # Auto-detect heads if not provided
        if not self.config.heads_path and self.config.model_path:
            self._auto_detect_heads()

        backend_config = self.config.to_backend_config()
        try:
            self._backend = create_backend(backend_config)
            self._is_initialized = True
            logger.info(
                "NLLWSpeechProcessor initialized: %s, %s, bd=%d, wb=%d, agg=%s, p=%.2f",
                self.config.backend_type,
                self.config.direction,
                self.config.border_distance,
                self.config.word_batch,
                self.config.aggregation,
                self.config.top_p_threshold,
            )
        except Exception as e:
            logger.error("Failed to initialize backend: %s", e)
            raise

    def _auto_detect_heads(self):
        """Try to find heads config from model path and direction."""
        heads_dir = os.environ.get("NLLW_HEADS_DIR", "")
        if not heads_dir:
            # Look in standard locations
            for candidate in [
                os.path.join(os.path.dirname(self.config.model_path), "..", "heads"),
                os.path.join(os.path.dirname(__file__), "heads", "configs"),
                "/app/heads",
            ]:
                if os.path.isdir(candidate):
                    heads_dir = candidate
                    break

        if not heads_dir:
            return

        parts = self.config.direction.split("-")
        src = parts[0] if len(parts) >= 1 else "en"
        tgt = parts[1] if len(parts) >= 2 else "zh"

        for pattern in [
            f"translation_heads_hymt_en_{tgt}.json",
            f"translation_heads_hymt_{src}_{tgt}.json",
            f"translation_heads_hy_mt1_5_7b_q8_0_{src}_{tgt}.json",
            f"translation_heads_hymt_en_zh.json",  # Fallback: cross-lingual transfer
        ]:
            path = os.path.join(heads_dir, pattern)
            if os.path.exists(path):
                self.config.heads_path = path
                logger.info("Auto-detected heads: %s", path)
                return

    def process_chunk(self, waveform: np.ndarray) -> IncrementalOutput:
        """Process one audio chunk (SimulStream protocol).

        In audio mode, this buffers audio, runs ASR when enough is accumulated,
        and feeds extracted words to the MT backend.

        Args:
            waveform: Audio samples as np.float32 (at config.sample_rate Hz)

        Returns:
            IncrementalOutput with any new translation text
        """
        if not self._is_initialized:
            self._initialize()

        self._audio_buffer = np.concatenate([self._audio_buffer, waveform])
        self._n_chunks_processed += 1
        self._total_audio_seconds = len(self._audio_buffer) / self.config.sample_rate

        # Wait for minimum audio before first ASR attempt
        if self._total_audio_seconds < self.config.min_start_seconds:
            return IncrementalOutput()

        # Run ASR to extract words
        new_words = self._run_asr()
        if not new_words:
            return IncrementalOutput()

        return self.process_words(new_words, is_final=False)

    def process_words(
        self,
        words: List[str],
        is_final: bool = False,
        emission_time: float = 0.0,
    ) -> IncrementalOutput:
        """Process pre-extracted words (text mode).

        This is the core method that feeds words to the MT backend.
        Can be called directly for testing without audio/ASR.

        In longform mode (default for competition):
            - is_final marks sentence boundaries, NOT recording boundaries
            - Backend resets at sentence boundaries (AlignAtt needs this)
            - Output accumulates across sentences for the whole recording
            - Call end_of_stream() at recording end, clear() between recordings

        In sentence mode (longform=False):
            - is_final marks both sentence and recording boundaries
            - Backend resets and output is per-sentence

        Args:
            words: List of new source words to process
            is_final: True if this is the last word of a sentence/segment
            emission_time: ASR emission timestamp (for latency metrics)

        Returns:
            IncrementalOutput with new translation text
        """
        if not self._is_initialized:
            self._initialize()

        if not self._recording_start_time:
            self._recording_start_time = time.time()

        output = IncrementalOutput()

        for i, word in enumerate(words):
            word_is_final = is_final and (i == len(words) - 1)
            step = self._backend.translate(
                word,
                is_final=word_is_final,
                emission_time=emission_time,
            )

            if step.text:
                output.new_string += step.text
                output.new_tokens.append(step.text)

                # Track emission for OmniSTEval (always, for longform output)
                wall_ms = (time.time() - self._recording_start_time) * 1000.0
                self._emission_log.append(EmissionEvent(
                    emission_time=emission_time * 1000.0,  # Convert to ms
                    wall_clock=wall_ms,
                    text=step.text,
                    is_final=word_is_final,
                ))

                # Always accumulate recording text (for longform OmniSTEval)
                self._recording_text += step.text

            if word_is_final:
                # Sentence boundary: reset backend but keep accumulating
                self._committed_text += output.new_string
                self._n_sentences_in_recording += 1

                # Backend already handled segment end in translate(is_final=True)
                # which calls _handle_segment_end(). Don't call reset() again
                # since that would double-reset (clearing already-empty state).

                if not self.config.longform:
                    # Sentence mode: explicit reset (backward compat)
                    self._backend.reset()

        # Auto sentence boundary detection: if longform + auto + no explicit is_final,
        # check if the generated text ends with sentence-ending punctuation
        if (self.config.longform and self.config.auto_sentence_boundary
                and not is_final and output.new_string):
            if self._detect_sentence_boundary(output.new_string):
                logger.debug("Auto-detected sentence boundary in: %s", output.new_string[-30:])
                # Force the backend to treat this as a sentence end
                self._backend.reset()
                self._committed_text += output.new_string
                self._n_sentences_in_recording += 1

        # Safety valve: force reset if source words approach n_ctx limit
        # Without this, a missing sentence boundary causes unbounded KV cache growth
        if (self.config.longform and not is_final
                and hasattr(self._backend, '_source_words')):
            n_src = len(self._backend._source_words)
            ctx_limit = int(self.config.n_ctx * 0.7)  # 70% of context window
            if n_src > ctx_limit:
                logger.warning(
                    "n_ctx safety valve: %d source words approaching limit %d, forcing reset",
                    n_src, ctx_limit,
                )
                self._backend.reset()
                self._committed_text += output.new_string
                self._n_sentences_in_recording += 1

        return output

    def _detect_sentence_boundary(self, text: str) -> bool:
        """Detect if text ends with sentence-ending punctuation.

        Used in longform mode for auto sentence boundary detection when
        the ASR doesn't provide is_final signals.

        Args:
            text: Generated target text to check

        Returns:
            True if text ends with sentence-ending punctuation
        """
        if not text:
            return False

        text = text.rstrip()
        if not text:
            return False

        punct_set = self._SENTENCE_END_PUNCT.get(
            self._target_lang,
            self._SENTENCE_END_PUNCT["default"]
        )
        return text[-1] in punct_set

    def end_of_stream(self) -> IncrementalOutput:
        """Called when audio stream ends. Flush remaining output.

        In longform mode: flushes the final sentence of the recording.
        Call clear() after this to prepare for the next recording.

        Returns:
            IncrementalOutput with final translation text
        """
        if not self._is_initialized:
            return IncrementalOutput()

        # If there are buffered words, flush them
        if self._words_buffer:
            output = self.process_words(
                self._words_buffer, is_final=True
            )
            self._words_buffer = []
            return output

        # If the backend has uncommitted text, force final
        output = IncrementalOutput()
        remaining = self._backend.get_full_translation()
        if remaining and remaining != self._prev_text:
            output.new_string = remaining[len(self._prev_text):]
            if output.new_string:
                output.new_tokens.append(output.new_string)
                # Track final emission
                if self._recording_start_time:
                    wall_ms = (time.time() - self._recording_start_time) * 1000.0
                    self._emission_log.append(EmissionEvent(
                        emission_time=wall_ms,  # Use wall-clock as CU at end
                        wall_clock=wall_ms,
                        text=output.new_string,
                        is_final=True,
                    ))

        if output.new_string:
            self._recording_text += output.new_string
        self._backend.reset()
        return output

    def set_source_language(self, lang: str):
        """Set source language (SimulStream protocol).

        Called by SimulStream before processing starts.

        Args:
            lang: ISO language code (e.g., "en", "cs")
        """
        self._source_lang = lang
        if self._target_lang:
            self._update_direction(f"{lang}-{self._target_lang}")

    def set_target_language(self, lang: str):
        """Set target language (SimulStream protocol).

        Args:
            lang: ISO language code (e.g., "zh", "de", "it", "en")
        """
        self._target_lang = lang
        if self._source_lang:
            self._update_direction(f"{self._source_lang}-{lang}")

    def _update_direction(self, new_direction: str):
        """Update direction and reload backend if changed."""
        if new_direction == self.config.direction:
            return

        logger.info("Direction changed: %s -> %s", self.config.direction, new_direction)
        self.config.direction = new_direction

        # Apply direction-specific defaults
        if new_direction in DIRECTION_DEFAULTS:
            for k, v in DIRECTION_DEFAULTS[new_direction].items():
                setattr(self.config, k, v)
            logger.info(
                "Applied %s defaults: bd=%d, wb=%d, agg=%s, p=%.2f",
                new_direction,
                self.config.border_distance,
                self.config.word_batch,
                self.config.aggregation,
                self.config.top_p_threshold,
            )

        # Reset heads path so auto-detection re-runs for new direction
        self.config.heads_path = ""

        # Force re-initialization on next call
        if self._backend:
            self._backend.close()
            self._backend = None
            self._is_initialized = False

    def tokens_to_string(self, tokens: List[str]) -> str:
        """Convert token list to human-readable string (SimulStream protocol).

        For our system, tokens are already text strings, so this is a simple join.

        Args:
            tokens: List of text token strings

        Returns:
            Concatenated string
        """
        return "".join(tokens)

    @property
    def speech_chunk_size(self) -> float:
        """Audio chunk size in seconds (SimulStream protocol).

        The SimulStream framework uses this to determine how often
        process_chunk() is called.

        Returns:
            Chunk size in seconds (default: 0.06 = 60ms)
        """
        return self.config.speech_chunk_size / self.config.sample_rate

    def clear(self):
        """Reset state between recordings (SimulStream protocol).

        In longform mode, this is the ONLY method that fully resets state.
        Call after end_of_stream() when moving to a new recording.
        """
        if self._backend:
            self._backend.reset()

        self._audio_buffer = np.array([], dtype=np.float32)
        self._words_buffer = []
        self._committed_text = ""
        self._prev_text = ""
        self._current_text = ""
        self._n_chunks_processed = 0
        self._total_audio_seconds = 0.0
        self._word_emission_times = []

        # Reset longform recording state
        self._emission_log = []
        self._recording_start_time = 0.0
        self._recording_text = ""
        self._n_sentences_in_recording = 0

    def close(self):
        """Free all resources."""
        if self._backend:
            self._backend.close()
            self._backend = None
        self._is_initialized = False

    def _run_asr(self) -> List[str]:
        """Run ASR on buffered audio to extract new words.

        This is a placeholder for actual ASR integration.
        In production, this would use Qwen3-ASR or another ASR model.
        For evaluation with gold transcripts, use process_words() directly.

        Returns:
            List of new source words (may be empty)
        """
        # TODO: Integrate actual ASR model (Qwen3-ASR)
        # For now, this is a stub -- production uses process_words() directly
        # with pre-extracted words from an external ASR system
        return []

    @property
    def stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "chunks_processed": self._n_chunks_processed,
            "total_audio_seconds": self._total_audio_seconds,
            "committed_text": self._committed_text,
            "recording_text": self._recording_text,
            "n_sentences_in_recording": self._n_sentences_in_recording,
            "n_emission_events": len(self._emission_log),
            "direction": self.config.direction,
            "backend_type": self.config.backend_type,
            "border_distance": self.config.border_distance,
            "word_batch": self.config.word_batch,
            "aggregation": self.config.aggregation,
            "top_p_threshold": self.config.top_p_threshold,
            "longform": self.config.longform,
        }

    @property
    def emission_log(self) -> List[EmissionEvent]:
        """Get the emission event log for the current recording."""
        return list(self._emission_log)

    def get_recording_text(self) -> str:
        """Get the full accumulated text for the current recording."""
        return self._recording_text

    # Languages that require char-level delay arrays in OmniSTEval
    _CHAR_LEVEL_LANGS = frozenset({"zh", "ja", "ko"})

    def to_omnisteval_entry(
        self,
        source_name: str = "recording.wav",
        source_length_ms: Optional[float] = None,
        char_level: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Convert the current recording's output to OmniSTEval format.

        Produces ONE JSONL entry per recording, matching OmniSTEval's
        longform evaluation format. This is the PRIMARY competition format.

        The entry format:
            {"source": "recording.wav", "prediction": "...",
             "delays": [...], "elapsed": [...], "source_length": ...}

        delays[i] = CU emission time (ms) for the i-th word/char in prediction
        elapsed[i] = CA emission time (ms) for the i-th word/char in prediction

        Args:
            source_name: Source audio filename
            source_length_ms: Total source audio duration in ms.
                If None, uses wall-clock recording time.
            char_level: If True, produce per-character delays (for zh/ja/ko).
                If False, produce per-word delays.
                If None (default), auto-detect from target language:
                zh/ja/ko -> True, others -> False.

        Returns:
            Dict ready to be serialized as JSONL
        """
        # Auto-detect char_level from target language if not explicitly set
        if char_level is None:
            char_level = self._target_lang in self._CHAR_LEVEL_LANGS

        prediction = self._recording_text.strip()

        # Strip LLM artifacts (matching ss-to-log.py reference converter)
        prediction = prediction.replace("<end_of_turn>", "")
        prediction = prediction.replace("<|endoftext|>", "")
        prediction = prediction.strip()

        # Normalize whitespace: collapse multiple spaces
        prediction = re.sub(r"  +", " ", prediction)

        if not prediction:
            return {
                "source": source_name,
                "prediction": "",
                "delays": [],
                "elapsed": [],
                "source_length": source_length_ms or 0.0,
            }

        # Build per-word (or per-char) delay arrays from emission log
        if char_level:
            # NFKC normalization (matching ss-to-log.py reference converter)
            # Critical for CJK: normalizes full-width/half-width variants
            normalized = unicodedata.normalize("NFKC", prediction)
            units = list(normalized)
            prediction = normalized  # Use normalized form for output
        else:
            units = prediction.split()

        n_units = len(units)

        # Walk through emission events and assign delays to prediction units
        delays_cu: List[float] = []
        delays_ca: List[float] = []

        # Build a mapping: for each character position in prediction,
        # find the emission event that produced it
        char_to_cu: List[float] = []
        char_to_ca: List[float] = []
        char_pos = 0

        for event in self._emission_log:
            # Strip LLM artifacts from event text too
            text = event.text.replace("<end_of_turn>", "").replace("<|endoftext|>", "")
            # NFKC normalize if char_level (matching prediction normalization)
            if char_level:
                text = unicodedata.normalize("NFKC", text)
            for _ in text:
                if char_pos < len(prediction):
                    char_to_cu.append(event.emission_time)
                    char_to_ca.append(event.wall_clock)
                    char_pos += 1

        # Pad if emission log doesn't cover all chars (shouldn't happen)
        last_cu = char_to_cu[-1] if char_to_cu else 0.0
        last_ca = char_to_ca[-1] if char_to_ca else 0.0
        while len(char_to_cu) < len(prediction):
            char_to_cu.append(last_cu)
            char_to_ca.append(last_ca)

        if char_level:
            delays_cu = [round(t, 1) for t in char_to_cu[:n_units]]
            delays_ca = [round(t, 1) for t in char_to_ca[:n_units]]
        else:
            # Per-word: use the delay of the LAST character of each word
            pos = 0
            for word in units:
                # Find where this word ends in the prediction string
                word_start = prediction.find(word, pos)
                if word_start < 0:
                    # Fallback: use current position
                    word_start = pos
                word_end = word_start + len(word)
                # Use delay of last char of this word
                idx = min(word_end - 1, len(char_to_cu) - 1)
                if idx >= 0:
                    delays_cu.append(round(char_to_cu[idx], 1))
                    delays_ca.append(round(char_to_ca[idx], 1))
                else:
                    delays_cu.append(0.0)
                    delays_ca.append(0.0)
                pos = word_end

        # Validate delay count matches unit count (OmniSTEval critical invariant)
        if len(delays_cu) != n_units:
            logger.warning(
                "Delay count mismatch: %d delays for %d units. Padding/trimming.",
                len(delays_cu), n_units,
            )
            last_val_cu = delays_cu[-1] if delays_cu else 0.0
            last_val_ca = delays_ca[-1] if delays_ca else 0.0
            while len(delays_cu) < n_units:
                delays_cu.append(last_val_cu)
                delays_ca.append(last_val_ca)
            delays_cu = delays_cu[:n_units]
            delays_ca = delays_ca[:n_units]

        # Enforce monotonicity: delays must be non-decreasing
        # (OmniSTEval requirement -- emission times can't go backward)
        for arr in (delays_cu, delays_ca):
            for i in range(1, len(arr)):
                if arr[i] < arr[i - 1]:
                    arr[i] = arr[i - 1]

        # Source length: use provided value or wall-clock recording time
        if source_length_ms is None:
            if self._recording_start_time:
                source_length_ms = (time.time() - self._recording_start_time) * 1000.0
            else:
                source_length_ms = 0.0

        return {
            "source": source_name,
            "prediction": prediction,
            "delays": delays_cu,
            "elapsed": delays_ca,
            "source_length": round(source_length_ms, 1),
        }


# ---------------------------------------------------------------------------
# Gold transcript evaluation: process JSONL input file
# ---------------------------------------------------------------------------

def process_gold_transcript(
    processor: NLLWSpeechProcessor,
    input_path: str,
    output_path: Optional[str] = None,
    compute_aware: bool = False,
) -> List[Dict[str, Any]]:
    """Process a gold ASR transcript JSONL through the processor.

    Input JSONL format (one word per line):
        {"text": "the", "emission_time": 0.5, "is_final": false}
        {"text": "president", "emission_time": 0.8, "is_final": false}
        ...
        {"text": "reforms", "emission_time": 3.2, "is_final": true}

    Output JSONL format (SimulStream-compatible):
        {"emission_time": 0.8, "text": "Le president", "is_final": false}
        ...

    Args:
        processor: Initialized NLLWSpeechProcessor
        input_path: Path to gold ASR JSONL file
        output_path: Path to output JSONL (None = stdout)
        compute_aware: If True, use wall-clock time for emission_time

    Returns:
        List of output entries
    """
    with open(input_path) as f:
        asr_words = [json.loads(line) for line in f if line.strip()]

    t0 = time.time()
    entries = []
    outf = open(output_path, "w") if output_path else sys.stdout

    try:
        for word_idx, asr_word in enumerate(asr_words):
            is_final = asr_word.get("is_final", False)
            is_last = (word_idx == len(asr_words) - 1)
            word_is_final = is_final or is_last
            emission_time = asr_word.get("emission_time", 0.0)

            output = processor.process_words(
                [asr_word["text"]],
                is_final=word_is_final,
                emission_time=emission_time,
            )

            # Build output entry
            if compute_aware:
                etime = max(emission_time, time.time() - t0)
            else:
                etime = emission_time

            entry = {
                "emission_time": etime,
                "text": output.new_string,
                "is_final": word_is_final,
            }
            if compute_aware:
                entry["speech_time"] = emission_time

            entries.append(entry)
            outf.write(json.dumps(entry, ensure_ascii=False) + "\n")
            outf.flush()

        # End of stream
        final_output = processor.end_of_stream()
        if not final_output.is_empty:
            entry = {
                "emission_time": time.time() - t0 if compute_aware else emission_time,
                "text": final_output.new_string,
                "is_final": True,
            }
            entries.append(entry)
            outf.write(json.dumps(entry, ensure_ascii=False) + "\n")
            outf.flush()

    finally:
        if output_path:
            outf.close()

    return entries


def process_gold_transcript_longform(
    processor: NLLWSpeechProcessor,
    input_path: str,
    source_name: str = "recording.wav",
    source_length_s: Optional[float] = None,
    output_path: Optional[str] = None,
    compute_aware: bool = False,
    char_level: bool = False,
) -> Dict[str, Any]:
    """Process a gold ASR transcript in longform mode (one recording).

    This is the COMPETITION FORMAT for IWSLT 2026. Produces one
    OmniSTEval JSONL entry per recording, matching what `omnisteval longform`
    expects.

    Input JSONL: same as process_gold_transcript()
    Output: Single OmniSTEval entry with per-word delays

    Args:
        processor: Initialized NLLWSpeechProcessor (longform mode)
        input_path: Path to gold ASR JSONL (one recording's words)
        source_name: Source audio filename for OmniSTEval
        source_length_s: Total source audio duration in seconds.
            If None, computed from last emission_time.
        output_path: Path for emission event log JSONL (optional)
        compute_aware: If True, use wall-clock for CA delays
        char_level: If True, per-character delays (for zh/ja/ko)

    Returns:
        OmniSTEval entry dict: {"source", "prediction", "delays", "elapsed", "source_length"}
    """
    with open(input_path) as f:
        asr_words = [json.loads(line) for line in f if line.strip()]

    # Process all words through the processor
    processor.clear()  # Start fresh for this recording
    last_emission_time = 0.0
    outf = open(output_path, "w") if output_path else None

    try:
        for word_idx, asr_word in enumerate(asr_words):
            is_final = asr_word.get("is_final", False)
            is_last = (word_idx == len(asr_words) - 1)
            # In longform: pass is_final through for sentence boundaries
            # The last word of the recording also gets is_final
            word_is_final = is_final or is_last
            emission_time = asr_word.get("emission_time", 0.0)
            last_emission_time = max(last_emission_time, emission_time)

            output = processor.process_words(
                [asr_word["text"]],
                is_final=word_is_final,
                emission_time=emission_time,
            )

            # Write emission event log (for debugging / SimulStream compat)
            if outf and output.new_string:
                entry = {
                    "emission_time": emission_time,
                    "text": output.new_string,
                    "status": "COMPLETE",
                    "is_final": word_is_final,
                }
                outf.write(json.dumps(entry, ensure_ascii=False) + "\n")
                outf.flush()

        # Flush remaining output
        final_output = processor.end_of_stream()
        if outf and not final_output.is_empty:
            entry = {
                "emission_time": last_emission_time,
                "text": final_output.new_string,
                "status": "COMPLETE",
                "is_final": True,
            }
            outf.write(json.dumps(entry, ensure_ascii=False) + "\n")
            outf.flush()

    finally:
        if outf:
            outf.close()

    # Compute source length
    source_length_ms = source_length_s * 1000.0 if source_length_s else last_emission_time * 1000.0

    # Generate OmniSTEval entry
    omnisteval_entry = processor.to_omnisteval_entry(
        source_name=source_name,
        source_length_ms=source_length_ms,
        char_level=char_level,
    )

    return omnisteval_entry


# ---------------------------------------------------------------------------
# Text-mode evaluation: process plain text sentences
# ---------------------------------------------------------------------------

def process_text_sentences(
    processor: NLLWSpeechProcessor,
    sentences: List[str],
    word_interval: float = 0.4,
) -> List[Dict[str, Any]]:
    """Process plain text sentences through the processor.

    Simulates word-by-word arrival with fixed timing for evaluation.

    Args:
        processor: Initialized NLLWSpeechProcessor
        sentences: List of source sentences
        word_interval: Simulated time between words (seconds)

    Returns:
        List of {source, translation, n_words, n_steps} dicts
    """
    results = []

    for sentence in sentences:
        words = sentence.strip().split()
        all_output = ""
        n_steps = 0

        for i, word in enumerate(words):
            is_final = (i == len(words) - 1)
            emission_time = (i + 1) * word_interval

            output = processor.process_words(
                [word],
                is_final=is_final,
                emission_time=emission_time,
            )

            if output.new_string:
                all_output += output.new_string
                n_steps += 1

        results.append({
            "source": sentence,
            "translation": all_output.strip(),
            "n_words": len(words),
            "n_steps": n_steps,
        })

        processor.clear()

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """CLI entry point for testing the SimulStream wrapper."""
    import argparse

    parser = argparse.ArgumentParser(
        description="NLLW SimulStream wrapper for IWSLT 2026"
    )
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--heads", default="", help="Path to head config JSON")
    parser.add_argument("--lang", default="en-zh", help="Language direction")
    parser.add_argument("--backend", default="alignatt", help="Backend type")
    parser.add_argument("--config", default=None, help="YAML config file")

    # Processing mode
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", help="Gold ASR JSONL input file")
    mode.add_argument("--longform-input", help="Gold ASR JSONL in longform mode (competition format)")
    mode.add_argument("--test", action="store_true",
                      help="Run self-test with sample sentences")
    mode.add_argument("--text", help="Translate a single sentence")

    parser.add_argument("--output", default=None, help="Output JSONL file")
    parser.add_argument("--omnisteval-output", default=None,
                        help="OmniSTEval JSONL output (longform mode)")
    parser.add_argument("--source-name", default="recording.wav",
                        help="Source audio filename for OmniSTEval")
    parser.add_argument("--source-length", type=float, default=None,
                        help="Source audio length in seconds")
    parser.add_argument("--char-level", action="store_true",
                        help="Char-level delays for zh/ja/ko (default: word-level)")
    parser.add_argument("--compute-aware", action="store_true")

    args = parser.parse_args()

    # Build config
    if args.config:
        config = SimulStreamConfig.from_yaml(args.config)
    else:
        # Apply direction defaults
        direction_cfg = DIRECTION_DEFAULTS.get(args.lang, {})
        config = SimulStreamConfig(
            model_path=args.model,
            heads_path=args.heads,
            direction=args.lang,
            backend_type=args.backend,
            **{k: v for k, v in direction_cfg.items()
               if k in SimulStreamConfig.__dataclass_fields__},
        )

    processor = NLLWSpeechProcessor.load_model(config)

    try:
        if args.input:
            entries = process_gold_transcript(
                processor, args.input, args.output, args.compute_aware
            )
            print(f"Processed {len(entries)} entries", file=sys.stderr)

        elif args.longform_input:
            # Competition format: longform processing
            config.longform = True
            omnisteval_entry = process_gold_transcript_longform(
                processor, args.longform_input,
                source_name=args.source_name,
                source_length_s=args.source_length,
                output_path=args.output,
                compute_aware=args.compute_aware,
                char_level=args.char_level,
            )

            # Write OmniSTEval JSONL
            out = args.omnisteval_output or args.output
            if out:
                with open(out, "w") as f:
                    json.dump(omnisteval_entry, f, ensure_ascii=False)
                    f.write("\n")
                print(f"OmniSTEval entry written: {out}", file=sys.stderr)
            else:
                json.dump(omnisteval_entry, sys.stdout, ensure_ascii=False)
                sys.stdout.write("\n")

            n_words = len(omnisteval_entry["prediction"].split())
            print(
                f"Longform: {n_words} words, "
                f"{processor._n_sentences_in_recording} sentences, "
                f"{len(processor._emission_log)} emissions",
                file=sys.stderr,
            )

        elif args.test:
            # Self-test with sample sentences
            test_sentences = [
                "The president of France announced reforms today.",
                "Neural machine translation has improved significantly.",
                "The weather in Prague is beautiful in spring.",
            ]
            results = process_text_sentences(processor, test_sentences)
            for r in results:
                print(f"  SRC: {r['source']}")
                print(f"  TGT: {r['translation']}")
                print(f"  Steps: {r['n_steps']}/{r['n_words']}")
                print()

        elif args.text:
            results = process_text_sentences(processor, [args.text])
            for r in results:
                print(r["translation"])

    finally:
        processor.close()


if __name__ == "__main__":
    main()
