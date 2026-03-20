# NLLW Development Todo

## Phase 1: Core Infrastructure (porting from iwslt26-sst) -- DONE

- [x] CLAUDE.md project documentation
- [x] todo.md task tracking
- [x] `nllw/llama_backend.py` -- llama.cpp ctypes bindings (LlamaModel, LlamaContext, LlamaBackend)
  - [x] Model/context lifecycle (load, create, free)
  - [x] Tokenization (tokenize, token_to_piece, tokens_to_text with byte-fallback handling)
  - [x] Batch/single decode with position control (decode_batch, decode_single)
  - [x] Attention weight extraction (set_attn_heads, get_attn_weights)
  - [x] KV cache management (memory_seq_rm, memory_clear, memory_seq_rm_attn_only for hybrid models)
  - [x] Logits extraction (argmax_logits, get_logits_array)
  - [x] C shim for context creation with attn_weights=True
- [x] `nllw/prompts.py` -- Prompt format registry (30+ templates)
  - [x] PromptFormat dataclass (prefix, suffix, context_tpl, context_entry)
  - [x] HY-MT formats (hymt, hymt-de, hymt-it, hymt-cs-en, hymt-ctx-v1/v2/v3)
  - [x] Qwen3/3.5 formats (qwen3, qwen3.5, qwen3.5-de, qwen3.5-it, qwen3.5-nothink, ctx variants)
  - [x] EuroLLM formats (eurollm, eurollm-de, eurollm-it)
  - [x] Tower, Gemma formats
  - [x] build_prompt() and find_source_token_range() helpers
- [x] `nllw/alignatt.py` -- AlignAtt border detection engine
  - [x] AlignAttEngine class (stateful, KV cache reuse, segment reset)
  - [x] aggregate_ts_weighted_vote() (TS-weighted head voting)
  - [x] stream_translate() convenience function for sentence-level eval
  - [x] Word batching, context injection, is_target_language() filter
  - [x] load_heads() for JSON head config loading
- [x] `nllw/metrics.py` -- SimulMT latency metrics
  - [x] Average Lagging (AL)
  - [x] Length-Adaptive Average Lagging (LAAL)
  - [x] YAAL (Yet Another Average Lagging) -- IWSLT 2026 primary metric
  - [x] Average Proportion (AP)
  - [x] Differentiable Average Lagging (DAL)
  - [x] Maximum Consecutive Wait (MaxCW)
  - [x] compute_yaal_ms() for time-domain metrics (ms)
  - [x] compute_average_lagging_ms() legacy time-domain metrics
- [x] `nllw/policies.py` -- Read/write policy framework
  - [x] SimulMTPolicy ABC
  - [x] AlignAttPolicy (border detection)
  - [x] WaitKPolicy
  - [x] ConfidencePolicy (entropy-based)
  - [x] FixedRatePolicy
  - [x] NoBorderPolicy (generate freely)
  - [x] create_policy() factory function

## Phase 2: Evaluation & Head Detection -- DONE

- [x] `nllw/eval.py` -- Evaluation harness (CLI: `python -m nllw.eval`)
  - [x] FLORES+ dataset loading (8 language pairs)
  - [x] Streaming evaluation loop with KV cache
  - [x] Results export (JSON + text files for comet-score CLI)
  - [x] Pretty-printed summary statistics
  - [x] Parameter sweep support (run_sweep() + CLI --sweep)
  - [x] XCOMET-XL direct integration (score_with_comet() + CLI --comet)
- [x] `nllw/heads/detect.py` -- Alignment head detection (CLI: `python -m nllw.heads.detect`)
  - [x] Token Similarity (TS) scoring via SimAlign word alignments
  - [x] FLORES-based head ranking (all 6 language pairs)
  - [x] JSON export with TS matrix and ranked heads
  - [x] Batch processing support for large models
  - [x] Hybrid model support (skip_gdn_layers for Qwen3.5)
- [ ] `nllw/heads/configs/` -- Pre-computed head configs (need GPU to generate)
  - [ ] HY-MT1.5-7B (all directions)
  - [ ] Qwen3.5-4B
  - [ ] EuroLLM-9B

## Phase 3: Testing -- IN PROGRESS

- [x] Unit tests for metrics.py (20 tests -- AL, LAAL, YAAL, DAL, AP, MaxCW, compute_yaal_ms)
- [x] Unit tests for prompts.py (10 tests, all passing)
- [x] Unit tests for alignatt.py (15 tests, all passing)
- [x] Unit tests for eval.py (7 tests -- save, print, COMET, configs)
- [x] Total: 56 tests, all passing
- [x] Update pyproject.toml with optional dependencies (datasets, simalign, comet)
- [ ] Integration test: AlignAtt engine end-to-end on GPU (small model, A40/L4)
- [ ] Integration test: eval.py on FLORES (need GPU machine)
- [ ] Integration test: heads/detect.py (need GPU machine)
- [ ] Benchmark: compare NLLW output vs iwslt26-sst on same model/data/config

## Phase 4: Advanced Features

- [ ] Async cascade pipeline (ASR + MT) -- port from iwslt26-sst/SimulMT_tests/cascade/
- [ ] Interactive translator CLI -- port from iwslt26-sst/SimulMT_tests/interactive_translator.py
- [ ] WhisperLiveKit integration for LLM backend (extend nllw/translation.py)
- [ ] Compute-aware latency mode (wall-clock emission times)
- [ ] Adaptive border distance (ASR confidence-based)
- [ ] TAF source lookahead (Translation by Anticipating Future)
- [ ] Policy comparison CLI (compare multiple policies on same data)

## Phase 5: Research & Optimization

### Completed
- [x] **YAAL metric**: Implemented in nllw/metrics.py -- IWSLT 2026 primary latency metric.
      Both word-count (compute_latency_metrics) and time-domain (compute_yaal_ms) versions.
      Formula matches OmniSTEval YAALScorer. Paper: https://arxiv.org/abs/2509.17349
- [x] **XCOMET-XL direct integration**: score_with_comet() in nllw/eval.py + CLI --comet flag.
      Supports any COMET model (XCOMET-XL, wmt22-comet-da). Handles empty translations.

### High Priority (new techniques, March 2026)
- [ ] **ExPosST** (March 2026): Fixed positional slots for incoming source, solves positional
      mismatch in decoder-only LLMs. SOTA on Llama-3.1-8B. Paper: https://arxiv.org/abs/2603.14903
      *Could replace or complement AlignAtt's KV cache strategy.*
- [ ] **Hikari** (March 2026): Policy-free SimulMT with probabilistic WAIT tokens + decoder time
      dilation. SOTA BLEU en-ja/de/ru. Paper: https://arxiv.org/abs/2603.11578
- [ ] **InfiniSST sliding window**: Pre-RoPE KV cache storage for unbounded streaming.
      Code: https://github.com/LeiLiLab/InfiniSST Paper: https://arxiv.org/abs/2503.02969

### Medium Priority
- [ ] **EAST-style append-only interleaving** (ICLR 2025): Explicit READ/WRITE tokens for fully
      reusable KV cache. Paper: https://openreview.net/forum?id=UqR2dFmfRB
- [ ] **StreamUni**: Streaming CoT for large speech-language models. SOTA en-de + en-zh.
      Code: https://github.com/ictnlp/StreamUni Paper: https://arxiv.org/abs/2507.07803
- [ ] **SimulMEGA** (NeurIPS 2025): MoE router as implicit R/W policy, zero overhead.
      Paper: https://arxiv.org/abs/2509.01200
- [ ] Test NLLW with newer models (Qwen4, Llama4, etc.)
- [ ] Speculative decoding for faster generation (extend existing speculative_decoding_v0.py)

### Lower Priority
- [ ] Explore SimulMask (attention masking during fine-tuning): https://arxiv.org/abs/2405.10443
- [ ] LoRA fine-tuning integration for domain adaptation
- [ ] TAF source lookahead (anticipate future, majority vote): https://arxiv.org/abs/2410.22499

## Experiment Log

### 2026-03-20 (iteration 3): Evaluation infrastructure complete

**Added**:
- YAAL metric (IWSLT 2026 primary) + DAL metric in `nllw/metrics.py`
  - Word-count domain: `compute_latency_metrics()` now returns AL, LAAL, YAAL, AP, DAL, MaxCW
  - Time domain: `compute_yaal_ms()` compatible with OmniSTEval YAALScorer
- XCOMET-XL direct scoring in `nllw/eval.py`: `score_with_comet()` + CLI `--comet`
- Parameter sweep in `nllw/eval.py`: `run_sweep()` + CLI `--sweep --sweep-bd --sweep-wb`
- 7 new eval tests, 11 new metric tests (total: 56 tests, all passing)

**YAAL formula** (from OmniSTEval, Polák et al. 2025):
```
gamma = max(|delays|, T) / S
YAAL = (1/tau) * sum_{t=0}^{tau-1} (d[t] - t/gamma)
```
where tau = number of delays where d < source_length (shortform) or all (longform).

**Usage examples**:
```bash
# Single eval with XCOMET-XL scoring
python -m nllw.eval --model model.gguf --heads heads.json --prompt-format hymt \
    --lang en-zh -n 100 --comet Unbabel/XCOMET-XL

# Parameter sweep
python -m nllw.eval --model model.gguf --heads heads.json --prompt-format hymt \
    --lang en-zh --sweep --sweep-bd 1,2,3,4,5 --sweep-wb 1,2,3 --comet Unbabel/XCOMET-XL
```

### 2026-03-20 (iteration 2): Initial port from iwslt26-sst

**Ported**: All core modules (llama_backend, alignatt, prompts, metrics, policies, eval, heads).
**Tests**: 38 unit tests passing. Integration tests pending (need GPU).
**Architecture**: Clean OOP design with LlamaModel -> LlamaContext -> AlignAttEngine hierarchy.
**Key improvement over iwslt26-sst**: Modular design, proper Python packaging, type hints,
docstrings, unit tests. The iwslt26-sst code was a research prototype; this is production-quality.

### Reference: Best known results from iwslt26-sst (March 2026)

| Direction | Model | XCOMET-XL | Config |
|-----------|-------|:---------:|--------|
| EN-ZH | HY-MT1.5-7B | 0.842 | bd=3, wb=2, no context |
| EN-DE | HY-MT1.5-7B | 0.786 | bd=3, wb=2 |
| EN-IT | HY-MT1.5-7B | 0.752 | bd=4, wb=3 |
| CS-EN | HY-MT1.5-7B | 0.908 | bd=2, wb=1 |
