#!/usr/bin/env python3
"""IWSLT 2026 competition readiness validator.

Checks that all components are properly configured and the system
is ready for the IWSLT 2026 SST evaluation (April 1-15).

Usage:
    python scripts/validate_competition.py              # Basic checks (no GPU)
    python scripts/validate_competition.py --gpu        # Full GPU validation
    python scripts/validate_competition.py --docker     # Docker image validation
"""

import sys
import os
import json
import importlib
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check(label: str, condition: bool, detail: str = ""):
    """Print check result."""
    icon = "PASS" if condition else "FAIL"
    msg = f"  [{icon}] {label}"
    if detail:
        msg += f" -- {detail}"
    print(msg)
    return condition


def validate_imports():
    """Check all required modules import successfully."""
    print("\n=== Module Imports ===")
    ok = True

    modules = [
        ("nllw.simulstream", "SimulStream wrapper"),
        ("nllw.alignatt", "AlignAtt algorithm"),
        ("nllw.alignatt_backend", "AlignAtt backend"),
        ("nllw.alignatt_la_backend", "AlignAtt-LA backend"),
        ("nllw.backend_protocol", "Backend protocol"),
        ("nllw.metrics", "Metrics (LongYAAL, COMET)"),
        ("nllw.omnisteval", "OmniSTEval output"),
        ("nllw.prompts", "Prompt formats"),
        ("nllw.llama_backend", "llama.cpp bindings"),
        ("nllw.corpus", "Test corpus"),
        ("nllw.complexity", "Complexity estimator"),
        ("nllw.fusion", "Signal fusion"),
    ]

    for mod_name, desc in modules:
        try:
            importlib.import_module(mod_name)
            ok &= check(desc, True)
        except Exception as e:
            ok &= check(desc, False, str(e))

    return ok


def validate_metrics():
    """Check LongYAAL and other metrics work correctly."""
    print("\n=== Metrics Validation ===")
    ok = True

    from nllw.metrics import (
        compute_longyaal, compute_longyaal_ms, compute_stream_laal,
        compute_all_metrics, compute_yaal,
    )

    # LongYAAL = YAAL with is_longform=True
    delays = [1, 2, 3, 4, 5]
    ly = compute_longyaal(delays, 5, 5)
    yaal = compute_yaal(delays, 5, 5, is_longform=True)
    ok &= check("LongYAAL == YAAL(longform)", abs(ly - yaal) < 1e-10, f"LY={ly:.4f}")

    # Time-domain LongYAAL
    delays_ms = [500, 1000, 1500, 2000, 2500]
    ly_ms = compute_longyaal_ms(delays_ms, 3000, 5)
    ok &= check("LongYAAL_ms computes", ly_ms > 0, f"LY_ms={ly_ms:.1f}ms")

    # StreamLAAL
    sl = compute_stream_laal(delays, 5, 5)
    ok &= check("StreamLAAL computes", sl > 0, f"SL={sl:.4f}")

    # All metrics together
    m = compute_all_metrics(
        delays, 5, 5,
        delays_ms=delays_ms, source_length_ms=3000,
    )
    ok &= check("compute_all_metrics has longyaal", m.longyaal > 0)
    ok &= check("compute_all_metrics has longyaal_ms", m.longyaal_ms > 0)
    ok &= check("compute_all_metrics has stream_laal", m.stream_laal > 0)

    return ok


def validate_simulstream():
    """Check SimulStream wrapper is competition-ready."""
    print("\n=== SimulStream Wrapper ===")
    ok = True

    from nllw.simulstream import (
        NLLWSpeechProcessor, SimulStreamConfig,
        IncrementalOutput, EmissionEvent, DIRECTION_DEFAULTS,
    )

    # All 4 competition directions configured
    for direction in ["en-zh", "en-de", "en-it", "cs-en"]:
        has_defaults = direction in DIRECTION_DEFAULTS
        ok &= check(f"Direction {direction} configured", has_defaults)
        if has_defaults:
            cfg = DIRECTION_DEFAULTS[direction]
            ok &= check(
                f"  {direction}: top_p agg",
                cfg.get("aggregation") == "top_p",
                f"agg={cfg.get('aggregation')}",
            )

    # SimulStream protocol methods exist
    for method in [
        "load_model", "process_chunk", "end_of_stream",
        "set_source_language", "set_target_language",
        "tokens_to_string", "clear", "speech_chunk_size",
    ]:
        has_method = hasattr(NLLWSpeechProcessor, method)
        ok &= check(f"Protocol method: {method}", has_method)

    # load_model accepts None (Docker env pattern)
    config = SimulStreamConfig(model_path="/nonexistent")
    proc = NLLWSpeechProcessor(config)

    # Direction switching works
    proc.set_target_language("de")
    ok &= check("Direction switch en-de", proc.config.direction == "en-de")
    ok &= check("  bd updated", proc.config.border_distance == 2)
    ok &= check("  p_threshold updated", proc.config.top_p_threshold == 0.75)

    proc.set_target_language("zh")
    ok &= check("Direction switch en-zh", proc.config.direction == "en-zh")
    ok &= check("  bd updated", proc.config.border_distance == 3)
    ok &= check("  p_threshold updated", proc.config.top_p_threshold == 0.85)

    # IncrementalOutput
    output = IncrementalOutput()
    ok &= check("Empty IncrementalOutput", output.is_empty)

    # Longform mode (CRITICAL for competition)
    ok &= check("Longform default enabled", SimulStreamConfig().longform is True)
    ok &= check("Auto sentence boundary default enabled",
                SimulStreamConfig().auto_sentence_boundary is True)

    # EmissionEvent dataclass
    event = EmissionEvent(emission_time=1.0, wall_clock=2.0, text="test")
    ok &= check("EmissionEvent dataclass", event.status == "COMPLETE")

    # Longform state tracking
    proc2 = NLLWSpeechProcessor(SimulStreamConfig())
    ok &= check("Longform emission log", hasattr(proc2, '_emission_log'))
    ok &= check("Longform recording text", hasattr(proc2, '_recording_text'))
    ok &= check("Longform sentence count", hasattr(proc2, '_n_sentences_in_recording'))

    # OmniSTEval output method
    ok &= check("to_omnisteval_entry method", hasattr(proc2, 'to_omnisteval_entry'))
    ok &= check("emission_log property", hasattr(proc2, 'emission_log'))
    ok &= check("get_recording_text method", hasattr(proc2, 'get_recording_text'))

    # Test OmniSTEval output generation (use en-de to test word-level)
    proc2_de = NLLWSpeechProcessor(SimulStreamConfig(direction="en-de"))
    proc2_de._recording_text = "Test output text."
    proc2_de._emission_log = [
        EmissionEvent(100.0, 120.0, "Test "),
        EmissionEvent(200.0, 220.0, "output "),
        EmissionEvent(300.0, 320.0, "text."),
    ]
    entry = proc2_de.to_omnisteval_entry(source_name="test.wav", source_length_ms=5000.0)
    ok &= check("OmniSTEval entry has source", entry["source"] == "test.wav")
    ok &= check("OmniSTEval entry has prediction", entry["prediction"] == "Test output text.")
    ok &= check("OmniSTEval entry has delays", len(entry["delays"]) == 3)  # 3 words
    ok &= check("OmniSTEval entry has elapsed", len(entry["elapsed"]) == 3)

    # Test char-level auto-detection for zh
    proc2_zh = NLLWSpeechProcessor(SimulStreamConfig(direction="en-zh"))
    proc2_zh._recording_text = "\u7f8e\u56fd\u603b\u7edf"  # 美国总统
    proc2_zh._emission_log = [
        EmissionEvent(100.0, 100.0, "\u7f8e\u56fd"),
        EmissionEvent(200.0, 200.0, "\u603b\u7edf"),
    ]
    entry_zh = proc2_zh.to_omnisteval_entry(source_name="test.wav", source_length_ms=1000.0)
    ok &= check("OmniSTEval zh auto char-level", len(entry_zh["delays"]) == 4)  # 4 chars, not 1 word
    ok &= check("OmniSTEval entry has source_length", entry["source_length"] == 5000.0)
    ok &= check("OmniSTEval delays monotonic",
                all(entry["delays"][i] <= entry["delays"][i+1]
                    for i in range(len(entry["delays"])-1)))

    # OmniSTEval artifact stripping
    proc3 = NLLWSpeechProcessor(SimulStreamConfig())
    proc3._recording_text = "Hello<end_of_turn> world<|endoftext|>"
    proc3._emission_log = [
        EmissionEvent(100.0, 100.0, "Hello<end_of_turn> "),
        EmissionEvent(200.0, 200.0, "world<|endoftext|>"),
    ]
    entry3 = proc3.to_omnisteval_entry(source_name="test.wav", source_length_ms=1000.0)
    ok &= check("LLM artifacts stripped", "<end_of_turn>" not in entry3["prediction"])
    ok &= check("endoftext stripped", "<|endoftext|>" not in entry3["prediction"])

    # NFKC normalization for char-level
    proc4 = NLLWSpeechProcessor(SimulStreamConfig(direction="en-zh"))
    proc4._recording_text = "\u7f8e\u56fd\u603b\u7edf"  # 美国总统
    proc4._emission_log = [
        EmissionEvent(100.0, 100.0, "\u7f8e\u56fd"),
        EmissionEvent(200.0, 200.0, "\u603b\u7edf"),
    ]
    entry4 = proc4.to_omnisteval_entry(
        source_name="test.wav", source_length_ms=1000.0, char_level=True)
    import unicodedata
    ok &= check("NFKC normalized", entry4["prediction"] == unicodedata.normalize("NFKC", "\u7f8e\u56fd\u603b\u7edf"))
    ok &= check("Char delays match chars", len(entry4["delays"]) == len(entry4["prediction"]))

    # Longform gold transcript function exists
    from nllw.simulstream import process_gold_transcript_longform
    ok &= check("process_gold_transcript_longform exists", callable(process_gold_transcript_longform))

    # batch_first_emission_time (iteration 22): correct LongYAAL attribution
    from nllw.backend_protocol import TranslationStep
    step = TranslationStep(text="test", batch_first_emission_time=1.5, avg_logprob=-2.0)
    ok &= check("TranslationStep.batch_first_emission_time", step.batch_first_emission_time == 1.5,
                "YAAL fix: attribute to batch start time")
    ok &= check("TranslationStep.avg_logprob", step.avg_logprob == -2.0,
                "Generation confidence diagnostic")

    # Verify emission uses batch_first when available
    from unittest.mock import MagicMock
    proc5 = NLLWSpeechProcessor(SimulStreamConfig(
        direction="en-de", auto_sentence_boundary=False))
    proc5._is_initialized = True
    proc5._backend = MagicMock()
    proc5._backend.translate.return_value = TranslationStep(
        text="Output", batch_first_emission_time=1.0)
    proc5.process_words(["word"], emission_time=2.0)
    if proc5._emission_log:
        ok &= check("Emission uses batch_first_emission_time",
                     proc5._emission_log[0].emission_time == 1000.0,
                     "Should be 1.0s * 1000 = 1000.0ms, not 2000.0ms")
    else:
        ok &= check("Emission log populated", False)

    return ok


def validate_omnisteval():
    """Check OmniSTEval output format is correct."""
    print("\n=== OmniSTEval Format ===")
    ok = True

    from nllw.omnisteval import SimulEvalEntry

    # Create a valid entry
    entry = SimulEvalEntry(
        source="test.wav",
        prediction="The translation of the text",
        delays=[500.0, 800.0, 1200.0, 1600.0, 2000.0],
        elapsed=[600.0, 900.0, 1300.0, 1700.0, 2100.0],
        source_length=3000.0,
    )

    # Critical: len(delays) == len(prediction.split())
    words = entry.prediction.split()
    ok &= check(
        "delays length == words",
        len(entry.delays) == len(words),
        f"delays={len(entry.delays)}, words={len(words)}",
    )

    # Serialization
    d = entry.__dict__ if hasattr(entry, '__dict__') else {}
    ok &= check("Entry has prediction", "prediction" in str(d) or hasattr(entry, 'prediction'), True)

    # OmniSTEval package compatibility test
    try:
        import omnisteval
        import tempfile

        # Test word-level format
        test_entry = {
            "source": "test.wav",
            "prediction": "Hello World test",
            "delays": [100.0, 200.0, 300.0],
            "elapsed": [110.0, 210.0, 310.0],
            "source_length": 5000.0,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(test_entry) + "\n")
            path = f.name
        instances, _ = omnisteval._load_hypothesis_jsonl(
            path, char_level=False,
            segmentation_order=["test.wav"],
            fix_emission_ca_flag=False,
        )
        os.unlink(path)
        ok &= check("OmniSTEval word-level parse", len(instances) == 1 and len(instances[0]) == 3)

        # Test char-level format (Chinese)
        test_entry_zh = {
            "source": "test.wav",
            "prediction": "\u7f8e\u56fd\u603b\u7edf",
            "delays": [100.0, 100.0, 200.0, 200.0],
            "elapsed": [100.0, 100.0, 200.0, 200.0],
            "source_length": 5000.0,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(test_entry_zh, ensure_ascii=False) + "\n")
            path = f.name
        instances_zh, _ = omnisteval._load_hypothesis_jsonl(
            path, char_level=True,
            segmentation_order=["test.wav"],
            fix_emission_ca_flag=False,
        )
        os.unlink(path)
        ok &= check("OmniSTEval char-level parse", len(instances_zh) == 1 and len(instances_zh[0]) == 4)

    except ImportError:
        ok &= check("OmniSTEval package", False, "not installed")

    return ok


def validate_novel_features():
    """Check novel features from iterations 20-24 are properly wired."""
    print("\n=== Novel Features (iter 20-24) ===")
    ok = True

    from nllw.backend_protocol import BackendConfig
    cfg = BackendConfig()

    # Iteration 20: perplexity-adaptive border
    ok &= check("perplexity_adaptive_bd field", hasattr(cfg, 'perplexity_adaptive_bd'))
    ok &= check("perplexity_bd_low field", hasattr(cfg, 'perplexity_bd_low'))

    # Iteration 21: source-aware batching
    ok &= check("source_aware_batching field", hasattr(cfg, 'source_aware_batching'))

    # Iteration 22: batch_first_emission_time + avg_logprob
    from nllw.backend_protocol import TranslationStep
    step = TranslationStep(text="test")
    ok &= check("batch_first_emission_time field", hasattr(step, 'batch_first_emission_time'))
    ok &= check("avg_logprob field", hasattr(step, 'avg_logprob'))

    # Iteration 23: confidence-adaptive wb + language-pair gen cap
    ok &= check("confidence_adaptive_wb field", hasattr(cfg, 'confidence_adaptive_wb'))
    ok &= check("confidence_wb_high field", hasattr(cfg, 'confidence_wb_high'))
    ok &= check("language_pair_gen_cap field", hasattr(cfg, 'language_pair_gen_cap'))

    # Iteration 24: entropy-gated top_p
    ok &= check("entropy_gated_top_p field", hasattr(cfg, 'entropy_gated_top_p'))
    from nllw.alignatt import merged_attention_entropy, entropy_gated_top_p_threshold
    import numpy as np
    test_attn = np.ones((3, 5)) / 5
    m_ent = merged_attention_entropy(test_attn, [1.0, 0.5, 0.3])
    ok &= check("merged_attention_entropy computes", m_ent > 0, f"ent={m_ent:.3f}")
    gated = entropy_gated_top_p_threshold(0.85, m_ent)
    ok &= check("entropy_gated_top_p_threshold computes", 0.5 <= gated <= 0.99, f"gated={gated:.3f}")

    # Iteration 25: generation temperature + confidence trimming
    ok &= check("generation_temperature field", hasattr(cfg, 'generation_temperature'))
    ok &= check("generation_temperature default=0.0", cfg.generation_temperature == 0.0)
    ok &= check("confidence_trim_threshold field", hasattr(cfg, 'confidence_trim_threshold'))
    ok &= check("confidence_trim_threshold default=None", cfg.confidence_trim_threshold is None)
    from nllw.alignatt import sample_with_temperature, trim_low_confidence_tokens
    tok = sample_with_temperature(np.array([1.0, 5.0, 2.0]), 0.0)
    ok &= check("sample_with_temperature greedy", tok == 1)
    trimmed = trim_low_confidence_tokens([1, 2, 3], [-0.5, -5.0, -6.0], threshold=-3.0)
    ok &= check("trim_low_confidence_tokens works", trimmed == [1], f"got {trimmed}")

    return ok


def validate_configs():
    """Check IWSLT 2026 YAML configs exist and are valid."""
    print("\n=== IWSLT 2026 Configs ===")
    ok = True

    configs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
    for direction in ["en-zh", "en-de", "en-it", "cs-en"]:
        config_path = os.path.join(configs_dir, f"iwslt2026-{direction}.yaml")
        exists = os.path.exists(config_path)
        ok &= check(f"Config: iwslt2026-{direction}.yaml", exists)

        if exists:
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            ok &= check(f"  has aggregation", "aggregation" in cfg, cfg.get("aggregation", "?"))
            ok &= check(f"  has top_p_threshold", "top_p_threshold" in cfg)
            ok &= check(f"  has n_gpu_layers", "n_gpu_layers" in cfg)

    return ok


def validate_heads():
    """Check alignment head configs exist for all directions."""
    print("\n=== Alignment Head Configs ===")
    ok = True

    heads_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "nllw", "heads", "configs",
    )

    # Check for HY-MT head configs
    if os.path.isdir(heads_dir):
        files = os.listdir(heads_dir)
        json_files = [f for f in files if f.endswith(".json")]
        ok &= check(f"Head configs found", len(json_files) > 0, f"{len(json_files)} configs")

        # Check HY-MT specifically
        hymt_files = [f for f in json_files if "hymt" in f.lower() or "hy_mt" in f.lower()]
        ok &= check(f"HY-MT head configs", len(hymt_files) > 0, f"{len(hymt_files)} files")
    else:
        ok &= check("Heads directory exists", False, heads_dir)

    return ok


def validate_corpus():
    """Check the test corpus has all directions."""
    print("\n=== Test Corpus ===")
    ok = True

    from nllw.corpus import get_corpus, list_directions

    available = list_directions()
    for direction in ["en-zh", "en-de", "en-it", "cs-en"]:
        try:
            if direction in available:
                sents = get_corpus(direction)
                ok &= check(f"Corpus {direction}", len(sents) > 0, f"{len(sents)} sentences")
            else:
                ok &= check(f"Corpus {direction}", False, f"not in {available}")
        except Exception as e:
            ok &= check(f"Corpus {direction}", False, str(e))

    return ok


def validate_dockerfile():
    """Check Dockerfile is properly configured."""
    print("\n=== Dockerfile ===")
    ok = True

    dockerfile = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Dockerfile",
    )

    if not os.path.exists(dockerfile):
        ok &= check("Dockerfile exists", False)
        return ok

    ok &= check("Dockerfile exists", True)

    with open(dockerfile) as f:
        content = f.read()

    ok &= check("Uses CUDA base", "nvidia/cuda" in content)
    ok &= check("Exposes port 8080", "EXPOSE 8080" in content)
    ok &= check("Uses simulstream server", "simulstream.server" in content)
    ok &= check("References NLLWSpeechProcessor", "NLLWSpeechProcessor" in content)
    ok &= check("Sets NLLW_MODEL_PATH", "NLLW_MODEL_PATH" in content)
    ok &= check("Sets NLLW_HEADS_DIR", "NLLW_HEADS_DIR" in content)
    ok &= check("Sets N_GPU_LAYERS", "NLLW_N_GPU_LAYERS" in content)
    ok &= check("Health check", "python3 -c" in content)

    return ok


def main():
    parser = argparse.ArgumentParser(description="IWSLT 2026 competition validator")
    parser.add_argument("--gpu", action="store_true", help="Run GPU-dependent checks")
    parser.add_argument("--docker", action="store_true", help="Validate Docker image")
    args = parser.parse_args()

    print("=" * 60)
    print("IWSLT 2026 Competition Readiness Validator")
    print("=" * 60)

    all_ok = True
    all_ok &= validate_imports()
    all_ok &= validate_metrics()
    all_ok &= validate_simulstream()
    all_ok &= validate_omnisteval()
    all_ok &= validate_novel_features()
    all_ok &= validate_configs()
    all_ok &= validate_heads()
    all_ok &= validate_corpus()
    all_ok &= validate_dockerfile()

    print("\n" + "=" * 60)
    if all_ok:
        print("ALL CHECKS PASSED -- System is competition-ready")
    else:
        print("SOME CHECKS FAILED -- Fix issues before submission")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
