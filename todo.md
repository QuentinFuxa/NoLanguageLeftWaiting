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
  - [x] Length-Adaptive Average Lagging (LAAL/LongYAAL)
  - [x] Average Proportion (AP)
  - [x] Maximum Consecutive Wait (MaxCW)
  - [x] compute_average_lagging_ms() for time-domain metrics
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
  - [ ] Parameter sweep support (TODO: sweep over bd, wb combinations)
  - [ ] COMET/XCOMET-XL direct integration (currently: use comet-score CLI)
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

- [x] Unit tests for metrics.py (9 tests, all passing)
- [x] Unit tests for prompts.py (10 tests, all passing)
- [x] Unit tests for alignatt.py (15 tests, all passing)
- [x] Total: 38 tests, all passing
- [ ] Integration test: AlignAtt engine end-to-end on GPU (small model, A40/L4)
- [ ] Integration test: eval.py on FLORES (need GPU machine)
- [ ] Integration test: heads/detect.py (need GPU machine)
- [ ] Benchmark: compare NLLW output vs iwslt26-sst on same model/data/config
- [ ] Update pyproject.toml with optional dependencies (datasets, simalign, comet)

## Phase 4: Advanced Features

- [ ] Async cascade pipeline (ASR + MT) -- port from iwslt26-sst/SimulMT_tests/cascade/
- [ ] Interactive translator CLI -- port from iwslt26-sst/SimulMT_tests/interactive_translator.py
- [ ] WhisperLiveKit integration for LLM backend (extend nllw/translation.py)
- [ ] Compute-aware latency mode (wall-clock emission times)
- [ ] Adaptive border distance (ASR confidence-based)
- [ ] TAF source lookahead (Translation by Anticipating Future)
- [ ] Policy comparison CLI (compare multiple policies on same data)

## Phase 5: Research & Optimization

- [ ] **EAST-style append-only interleaving** (ICLR 2025): Train LLM with explicit READ/WRITE tokens
      for fully reusable KV cache. Paper: https://openreview.net/forum?id=UqR2dFmfRB
- [ ] **SimulMEGA** (NeurIPS 2025): Use MoE router gating as implicit R/W policy. Zero inference
      overhead. Paper: https://arxiv.org/abs/2509.01200
- [ ] **LongYAAL metric**: Implement in nllw/metrics.py -- primary IWSLT 2026 ranking metric
      (uses SoftSegmenter for alignment). Paper: https://arxiv.org/abs/2509.17349
- [ ] **XCOMET-XL direct integration**: Add to nllw/eval.py (pip install unbabel-comet>=2.2.0)
- [ ] Explore SimulMask (attention masking during fine-tuning): https://arxiv.org/abs/2405.10443
- [ ] InfiniSST sliding window for unbounded streaming: https://arxiv.org/abs/2503.02969
- [ ] Test NLLW with newer models (Qwen4, Llama4, etc.)
- [ ] LoRA fine-tuning integration for domain adaptation
- [ ] Speculative decoding for faster generation (extend existing speculative_decoding_v0.py)

## Experiment Log

### 2026-03-20: Initial port from iwslt26-sst

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
