# Shared Task Notes -- NLLW SimulMT

## What exists now (after Iteration 4, 2026-03-20)

**21 SimulMT modules (~7800 lines), 199 tests passing:**

### Core (Iteration 1):
- `nllw/prompts.py` -- 30+ prompt formats (HY-MT, Qwen3, Qwen3.5, EuroLLM, Tower, Gemma)
- `nllw/llama_backend.py` -- ctypes wrapper for llama.cpp (attention extraction + KV cache)
- `nllw/backend_protocol.py` -- SimulMTBackend ABC + factory + `@register_backend`
- `nllw/alignatt.py` -- Core algorithm (TS-weighted vote, border detection, entropy, lookahead) + **7 aggregation methods**
- `nllw/alignatt_backend.py` -- Full backend with KV cache reuse + baselines (full-sentence, eager)
- `nllw/metrics.py` -- All latency metrics (AL, LAAL, YAAL, AP, DAL, MaxCW) + BLEU/COMET wrappers + **NE metric**
- `nllw/simulate.py` -- Policy simulation and trace replay
- `nllw/eval.py` -- Evaluation harness (FLORES+, parameter sweep, XCOMET-XL)
- `nllw/bench.py` -- Unified CLI (`python -m nllw.bench --lang en-fr --comet`)
- `nllw/heads/configs/` -- 22 pre-computed alignment head configs

### Iteration 2:
- `nllw/detect_heads.py` -- Auto alignment head detection for any GGUF model (559 lines)
- `nllw/omnisteval.py` -- OmniSTEval JSONL export for IWSLT 2026 (258 lines)
- `nllw/baselines.py` -- wait-k + fixed-rate baselines (175 lines)
- `nllw/analysis.py` -- Pareto frontier, edge cases, report generation (309 lines)
- `nllw/experiment.py` -- Experiment config/result registry with YAML support (359 lines)
- `nllw/corpus.py` -- Expanded: 120 sentences across 5 directions (622 lines)
- `nllw/research.py` -- Compute-aware latency (CA-AL, CA-YAAL), benchmark suite (191 lines)

### Iteration 3:
- `nllw/alignatt_la_backend.py` -- LocalAgreement + AlignAtt hybrid (~550 lines with SSBD)
- 7 aggregation methods in `alignatt.py`
- Dynamic border distance
- 8 experiment configs in `configs/`

### New (Iteration 4):
- **SSBD (Self-Speculative Biased Decoding)** in alignatt-la backend:
  - `ssbd_accept()`: Biased acceptance criterion from Zeng et al. (2025)
  - `_retranslate_ssbd()`: 3-phase verification:
    1. Batch verify draft tokens (previous translation) in ONE forward pass with `output_last_only=False`
    2. Find first divergence using biased acceptance (`P'(draft) = (1-beta)*P(draft) + beta`)
    3. Resume autoregressive generation from divergence with border detection
  - Expected speedup: 1.3-1.7x (paper results), zero quality loss at beta=0.2
  - CLI: `--ssbd-beta 0.2`, sweep: `ssbd=0.0,0.1,0.2`
- **NE (Normalized Erasure) metric** in metrics.py:
  - Token-level: `compute_normalized_erasure(revision_history)`
  - Word-level: `compute_normalized_erasure_text(revision_history)`
  - NE < 0.2 = low revision (standard threshold)
- **Revision history tracking** in LA backend:
  - `get_revision_history()` returns all intermediate translations for NE computation
  - `get_ssbd_stats()` returns draft/accepted token counts and acceptance rate
- **Web debug server** (`web_debug/server.py`, 446 lines):
  - FastAPI on port 8777 with embedded HTML/JS frontend
  - Load any backend with full config (SSBD, dynamic border, aggregation)
  - Word-by-word translation visualization with border hit markers
  - SSBD acceptance stats and NE metric display
  - Compare endpoint for side-by-side config comparison
- 27 new tests (199 total, all passing)
- 3 new experiment configs: ssbd-sweep, la-optimizations, ssbd-multidirection

## What to do next

### Priority 1: Run experiments on A40 (all code is ready)

```bash
# 1a. First E2E validation
python -m nllw.bench --model /path/to/HY-MT1.5-7B.gguf --lang en-zh -n 20

# 1b. Novel aggregation sweep (7 methods, no published baselines)
python -m nllw.bench --sweep "agg=ts_vote,softmax_mean,entropy_weighted,consensus,geomean,top_p,ensemble" --lang en-zh --comet --save

# 1c. AlignAtt vs AlignAtt-LA comparison
python -m nllw.bench --compare alignatt alignatt-la --lang en-zh --comet --save

# 1d. SSBD speedup test (NEW)
python -m nllw.bench --backend alignatt-la --sweep "ssbd=0.0,0.1,0.2,0.3" --lang en-zh --comet --save

# 1e. Dynamic border distance test
python -m nllw.bench --lang en-zh --dynamic-border --comet --save

# 1f. Pareto sweep
python -m nllw.bench --sweep "bd=2,3,4,5 wb=1,2,3" --lang en-zh,en-de --comet --save
```

### Priority 2: Further optimizations
- **LA forced decoding**: Force-decode committed prefix (CUNI approach) -- reduces computation
- **LA two-pass catch-up**: Extra re-translation per update for stability
- ~~Web debug server~~ **DONE**: `web_debug/server.py` on port 8777

### Priority 3: Research ideas (see todo.md)
- Adaptive Multi-Strategy aggregation
- Cross-lingual head transfer
- ExPosST position slot reservation

## Key architecture decisions

- `PromptFormat` is a frozen dataclass with `build_prompt()` method
- `BackendConfig.from_dict()` ignores unknown keys (safe for sweep configs)
- `@register_backend("name")` decorator auto-registers with factory
- `simulate_backend()` feeds words one at a time, records delays
- FLORES+ uses `openlanguagedata/flores_plus` with per-language configs (not pairs)
- `detect_heads` uses SimAlign (mBERT) for ground-truth word alignments
- `ExperimentConfig` supports both flat dicts and nested YAML format
- `ParetoAnalyzer` computes quality-latency frontiers across experiments
- `aggregate()` dispatcher routes to any of 7 aggregation strategies
- LA backend clears KV cache from prefix onwards for each re-translation (prefix reused)
- `_longest_common_prefix_tokens()` is the core LA stability check
- **NEW**: SSBD uses `output_last_only=False` in `decode_batch_at` for per-position logits
- **NEW**: SSBD acceptance uses log-ratio trick to avoid full softmax computation
- **NEW**: Revision history is tracked per-segment, reset on segment end

## Important context

- llama.cpp must be built with PR #20086. The build is on A40.
- The logit_idx bug (batch index 0, not KV pos) is fixed in `alignatt_backend.py`.
- YAAL formula: `gamma = max(|delays|, T) / S; yaal = sum(d - t/gamma) / tau`
- FLORES loading is tested and working locally.
- 5 directions available: en-fr, en-zh, en-de, en-it, cs-en
- **7 backends registered**: alignatt, alignatt-la, full-sentence, eager, wait-k, fixed-rate
- **7 aggregation methods**: ts_vote, softmax_mean, entropy_weighted, consensus, geomean, top_p, ensemble
- **SSBD**: `ssbd_beta=0.2` recommended, 0.0 for pure speculative (no bias)
- **NE metric**: `compute_normalized_erasure()` for token-level, `compute_normalized_erasure_text()` for word-level
- SOTA research documented in `docs/research/sota-simulmt-2026.md`

## Key research findings (from SOTA survey)

- **AlignAtt validated**: CUNI won IWSLT 2025 using AlignAtt. ICLR 2026 "Translation Heads" paper confirms alignment heads are sparse, universal, consistent.
- **SSBD** (arxiv 2509.21740) implemented -- can accelerate re-translation (alignatt-la) via speculative draft reuse
- **ExPosST** (arxiv 2603.14903) pre-allocates position slots for zero-recomputation KV cache
- **Open gap**: No published work on alternative attention aggregation -- our 7 methods are novel
- **Hikari** (arxiv 2603.11578) is the main competitor: policy-free WAIT tokens, SOTA on EN-JA/DE/RU
- **IWSLT 2026 metrics**: LongYAAL (primary latency), XCOMET-XL (primary quality) -- our pipeline is aligned
