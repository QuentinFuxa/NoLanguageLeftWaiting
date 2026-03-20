# NoLanguageLeftWaiting (NLLW) -- Simultaneous Machine Translation Library

## Project Overview

Production-quality Python library for Simultaneous Machine Translation (SimulMT). Supports both:
- **NLLB backend** (encoder-decoder, HuggingFace/CTranslate2) -- existing, for streaming ASR integration
- **LLM backend** (llama.cpp, GGUF models) -- new, for AlignAtt-based SimulMT with any decoder-only LLM

### Architecture

```
nllw/
  core.py              -- NLLB TranslationBackend (existing, stable)
  translation.py       -- WhisperLiveKit OnlineTranslation interface (existing)
  languages.py         -- Language code mapping (existing)
  timed_text.py        -- TimedText dataclass (existing)
  llama_backend.py     -- llama.cpp ctypes bindings (ported from iwslt26-sst)
  alignatt.py          -- AlignAtt border detection engine (core SimulMT algorithm)
  prompts.py           -- Prompt format registry (40+ templates for HY-MT, Qwen, EuroLLM, etc.)
  policies.py          -- Read/write policies (AlignAtt, wait-k, confidence, fixed-rate)
  metrics.py           -- SimulMT latency metrics (AL, LAAL, AP, MaxCW)
  eval.py              -- Evaluation harness (FLORES, COMET, streaming eval)
  heads/               -- Alignment head detection & configs
    detect.py          -- Head detection algorithm (TS scoring via FLORES)
    configs/           -- Pre-computed head configs (JSON) for popular models
```

### Key Algorithm: AlignAtt Border Detection

1. Source words arrive incrementally (from ASR or text)
2. Build prompt: prefix + context + source + suffix + committed_output
3. Generate tokens with LLM (greedy argmax)
4. After each generated token, extract attention from top-K alignment heads
5. TS-weighted vote: if attended source position >= source_len - border_distance -> STOP (border hit)
6. At sentence boundary: commit translation, reset context, start next segment
7. KV cache reuse: only re-decode changed tokens (3-5x speedup)

### Best Known Results (from iwslt26-sst, March 2026)

| Direction | Model | XCOMET-XL | Config |
|-----------|-------|:---------:|--------|
| EN-ZH | HY-MT1.5-7B | 0.842 | bd=3, wb=2, no context |
| EN-DE | HY-MT1.5-7B | 0.786 | bd=3, wb=2 |
| EN-IT | HY-MT1.5-7B | 0.752 | bd=4, wb=3 |
| CS-EN | HY-MT1.5-7B | 0.908 | bd=2, wb=1 |

### Dead Ends (do NOT revisit)

- EAST learned policy (BLEU 27.3 vs baseline 42)
- LoRA no-think block (-0.178 COMET)
- GDN warm-start (33 hallucinations)
- Extra glossary (hurts small models)
- HY-MT context injection (hurts ALL 4 directions)
- Confidence-based stopping alone (COMET 0.468)
- Fixed-rate tokens (COMET 0.293-0.334)
- TAF source lookahead (worse on EN-ZH)
- Seed-X-PPO-7B, Qwen3-4B-2507, HY-MT1.5-1.8B, TranslateGemma-4B (all inferior)

---

## Machines

| Machine | Used for |
|---------|----------|
| **A40** (always up) | llama.cpp MT experiments, head detection |
| **L4_1** ($0.5/hr) | Cheap experiments, 7B models |
| **L4_2** ($0.5/hr) | Parallel experiments |
| **MacBook M5** (local) | Research, GGUF experiments, development |

SSH details: see iwslt26-sst CLAUDE.md.

---

## Development Rules

- **Never push to remote** (continuous-claude automation handles this)
- Write clean, well-documented, modular code
- All evaluation results go in todo.md or dedicated results files
- Test everything: unit tests + integration tests on GPU machines
- Document all experiments with parameters, metrics, and conclusions

---

## Dependencies

Core: `torch`, `transformers` (existing)
LLM backend: `llama.cpp` built with attention weight extraction (PR #20086)
Evaluation: `datasets` (FLORES), `comet` (COMET/XCOMET-XL), `numpy`
Head detection: `simalign` (SimAlign for word alignment ground truth)

---

## Reference Repo

The original research was done in `/Users/quentin/Documents/repos/iwslt26-sst/SimulMT_tests/`.
Key files to reference when porting:
- `alignatt/llama_attn.py` -- ctypes bindings (395 lines)
- `alignatt/alignatt_simulstreaming_llamacpp.py` -- main agent (990 lines)
- `alignatt/policy_comparison.py` -- policy comparison (411 lines)
- `eval/eval_streaming_kvcache.py` -- streaming eval with KV cache (457 lines)
- `heads/detect_translation_heads_llamacpp.py` -- head detection
- `docs/RESULTS.md`, `docs/TECHNIQUES.md`, `docs/MODELS.md` -- all research results
