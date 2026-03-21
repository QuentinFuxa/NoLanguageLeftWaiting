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
import sys
import time
import json
import logging
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
    """

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

    @classmethod
    def load_model(cls, config: SimulStreamConfig) -> "NLLWSpeechProcessor":
        """Factory method matching SimulStream's load_model pattern.

        Args:
            config: SimulStreamConfig with model paths and parameters.

        Returns:
            Initialized NLLWSpeechProcessor ready for process_chunk() calls.
        """
        processor = cls(config)
        processor._initialize()
        return processor

    def _initialize(self):
        """Initialize the MT backend."""
        if self._is_initialized:
            return

        backend_config = self.config.to_backend_config()
        self._backend = create_backend(backend_config)
        self._is_initialized = True
        logger.info(
            "NLLWSpeechProcessor initialized: %s, %s, bd=%d, wb=%d",
            self.config.backend_type,
            self.config.direction,
            self.config.border_distance,
            self.config.word_batch,
        )

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

        Args:
            words: List of new source words to process
            is_final: True if this is the last word of a sentence/segment
            emission_time: ASR emission timestamp (for latency metrics)

        Returns:
            IncrementalOutput with new translation text
        """
        if not self._is_initialized:
            self._initialize()

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

            if word_is_final:
                self._committed_text += output.new_string
                self._backend.reset()

        return output

    def end_of_stream(self) -> IncrementalOutput:
        """Called when audio stream ends. Flush remaining output.

        Sends is_final=True for any buffered words, then resets.

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

        self._backend.reset()
        return output

    def set_source_language(self, lang: str):
        """Set source language (SimulStream protocol).

        Called by SimulStream before processing starts.

        Args:
            lang: ISO language code (e.g., "en", "cs")
        """
        self._source_lang = lang
        # Update direction if target is also set
        if self._target_lang:
            new_direction = f"{lang}-{self._target_lang}"
            if new_direction != self.config.direction:
                self.config.direction = new_direction
                # Apply direction-specific defaults
                if new_direction in DIRECTION_DEFAULTS:
                    for k, v in DIRECTION_DEFAULTS[new_direction].items():
                        setattr(self.config, k, v)
                # Force re-initialization on next call
                if self._backend:
                    self._backend.close()
                    self._backend = None
                    self._is_initialized = False

    def set_target_language(self, lang: str):
        """Set target language (SimulStream protocol).

        Args:
            lang: ISO language code (e.g., "zh", "de", "it", "en")
        """
        self._target_lang = lang
        if self._source_lang:
            new_direction = f"{self._source_lang}-{lang}"
            if new_direction != self.config.direction:
                self.config.direction = new_direction
                if new_direction in DIRECTION_DEFAULTS:
                    for k, v in DIRECTION_DEFAULTS[new_direction].items():
                        setattr(self.config, k, v)
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
        """Reset state between talks (SimulStream protocol)."""
        if self._backend:
            self._backend.reset()

        self._audio_buffer = np.array([], dtype=np.float32)
        self._words_buffer = []
        self._committed_text = ""
        self._prev_text = ""
        self._current_text = ""
        self._n_chunks_processed = 0
        self._total_audio_seconds = 0.0

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
    mode.add_argument("--test", action="store_true",
                      help="Run self-test with sample sentences")
    mode.add_argument("--text", help="Translate a single sentence")

    parser.add_argument("--output", default=None, help="Output JSONL file")
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
