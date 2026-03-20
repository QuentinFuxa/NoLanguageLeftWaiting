# Shared Task Notes -- NLLW SimulMT

## What exists now (after Iteration 6, 2026-03-20)

**22 SimulMT modules (~9200 lines), 290 tests passing:**

### Core (Iterations 1-5):
- `nllw/prompts.py` -- 30+ prompt formats (HY-MT, Qwen3, Qwen3.5, EuroLLM, Tower, Gemma)
- `nllw/llama_backend.py` -- ctypes wrapper for llama.cpp (attention extraction + KV cache)
- `nllw/backend_protocol.py` -- SimulMTBackend ABC + factory + `@register_backend`
- `nllw/alignatt.py` -- Core algorithm + **9 aggregation methods** + AMS + temp normalization
- `nllw/alignatt_backend.py` -- Full backend with KV cache reuse + baselines (full-sentence, eager)
- `nllw/alignatt_la_backend.py` -- LA + AlignAtt + SSBD + forced decode + adaptive SSBD + two-pass
- `nllw/metrics.py` -- All latency metrics + BLEU/COMET + NE metric
- `nllw/bench.py` -- Unified CLI with 12+ sweep shortnames
- `nllw/baselines.py` -- wait-k + fixed-rate baselines
- `nllw/heads/configs/` -- 22 pre-computed alignment head configs
- Plus: `eval.py`, `simulate.py`, `corpus.py`, `experiment.py`, `analysis.py`, `research.py`, `detect_heads.py`, `omnisteval.py`
- `web_debug/server.py` -- FastAPI debug server on port 8777

### New in Iteration 6:
- **LA Two-Pass Catch-up** (`_retranslate_two_pass()` in LA backend):
  - Run two re-translations per source update, keep the more stable one
  - Pass 2 always uses standard re-translation for diversity
  - `--two-pass` CLI flag, `twopass=0,1` sweep shortname
  - Trades 2x compute for lower NE (output flicker)
- **Adaptive Multi-Strategy (AMS)** (`select_adaptive_aggregation()` in alignatt.py):
  - Per-token aggregation method selection based on attention patterns
  - head agreement >= 0.7 + entropy <= 1.0 -> ts_vote
  - head agreement >= 0.7 + entropy > 1.0 -> entropy_weighted
  - head agreement < 0.7 + entropy <= 1.5 -> geomean
  - head agreement < 0.7 + entropy > 1.5 -> consensus
  - `--adaptive-agg` CLI flag, `ams=0,1` sweep shortname
- **Per-head Temperature Normalization** (`normalize_head_temperatures()` in alignatt.py):
  - Binary search temperature scaling to match reference entropy
  - Ensures sharp heads don't dominate; fair comparison across heads
  - `--head-temp-norm` CLI flag, `tempnorm=0,1 tempref=1.0,1.5,2.0` sweep
- **Cross-lingual Head Transfer** (`nllw/head_transfer.py`):
  - Full analysis utility: Jaccard, overlap, TS correlation, transferred TS mass
  - **RESULTS** (run on real configs):
    - EuroLLM: 98.9% mean TS mass, 97.3% min -- EXCELLENT
    - HY-MT1.8B: 98.4% mean -- EXCELLENT
    - Qwen3.5-4B: 97.8% mean, 92.7% min -- EXCELLENT
    - Qwen3.5-9B: 99.7% mean -- EXCELLENT
    - Qwen3-4B: 79.8% mean, 43.5% worst (en-it->en-zh) -- GOOD but pair-specific
  - Confirms ICLR 2026 finding: most models have universal alignment heads
  - `python -m nllw.head_transfer --all --top-k 10`
- 46 new tests (290 total, all passing)

## What to do next

### Priority 1: Run experiments on A40 (all code is ready)

```bash
# 1a. First E2E validation
python -m nllw.bench --model /path/to/HY-MT1.5-7B.gguf --lang en-zh -n 20

# 1b. Full 9-method aggregation sweep (novel research)
python -m nllw.bench --sweep "agg=ts_vote,softmax_mean,entropy_weighted,consensus,geomean,top_p,gaussian_kernel,gaussian_kernel_continuous,ensemble" --lang en-zh --comet --save

# 1c. AlignAtt vs AlignAtt-LA comparison
python -m nllw.bench --compare alignatt alignatt-la --lang en-zh --comet --save

# 1d. SSBD speedup: fixed vs adaptive beta
python -m nllw.bench --backend alignatt-la --sweep "ssbd=0.0,0.1,0.2,0.3" --lang en-zh --comet --save

# 1e. Forced decoding test
python -m nllw.bench --backend alignatt-la --forced-decode --lang en-zh --comet --save

# 1f. Iteration 6 features sweep
python -m nllw.bench --sweep "ams=0,1 tempnorm=0,1" --lang en-zh --comet --save
python -m nllw.bench --backend alignatt-la --two-pass --lang en-zh --comet --save
python -m nllw.bench --adaptive-agg --head-temp-norm --lang en-zh --comet --save

# 1g. Pareto sweep
python -m nllw.bench --sweep "bd=2,3,4,5 wb=1,2,3" --lang en-zh,en-de --comet --save
```

### Priority 2: Further optimizations
- **Dynamic word_batch**: Adjust wb based on source complexity
- **ExPosST position slots**: Pre-allocate KV positions for zero-recomputation
- **SSD parallel speculation**: Multiple draft continuations for 2x speedup

### Priority 3: Research ideas (see todo.md)
- GRPO fine-tuning (RL-optimize read/write)
- Syntax-aware chunking (SASST)
- Group Position Encoding as alternative KV cache

## Key architecture decisions

- `PromptFormat` is a frozen dataclass with `build_prompt()` method
- `BackendConfig.from_dict()` ignores unknown keys (safe for sweep configs)
- `@register_backend("name")` decorator auto-registers with factory
- FLORES+ uses `openlanguagedata/flores_plus` with per-language configs
- `detect_heads` uses SimAlign (mBERT) for ground-truth word alignments
- `aggregate()` dispatcher routes to any of 9 aggregation strategies
- LA backend retranslation priority: SSBD > forced_decode > standard
- Two-pass uses pass 1 (SSBD/forced/standard) + pass 2 (always standard) for diversity
- `_check_border()` helper centralizes all border check params (AMS, temp norm, dynamic)
- `normalize_head_temperatures()` uses binary search over temperature parameter
- `select_adaptive_aggregation()` uses head agreement ratio + attention entropy
- Cross-lingual head transfer: most models have >90% TS mass transfer across pairs

## Important context

- llama.cpp must be built with PR #20086. The build is on A40.
- The logit_idx bug (batch index 0, not KV pos) is fixed in `alignatt_backend.py`.
- YAAL formula: `gamma = max(|delays|, T) / S; yaal = sum(d - t/gamma) / tau`
- FLORES loading is tested and working locally.
- 5 directions available: en-fr, en-zh, en-de, en-it, cs-en
- **7 backends registered**: alignatt, alignatt-la, full-sentence, eager, wait-k, fixed-rate
- **9 aggregation methods**: ts_vote, softmax_mean, entropy_weighted, consensus, geomean, top_p, gaussian_kernel, gaussian_kernel_continuous, ensemble
- **New params**: `la_two_pass`, `adaptive_aggregation`, `head_temp_normalize`, `head_temp_reference`
- **New sweeps**: `twopass=0,1`, `ams=0,1`, `tempnorm=0,1`, `tempref=1.0,1.5,2.0`
- **Head transfer**: EuroLLM/HY-MT/Qwen3.5 heads are cross-lingually universal. Qwen3-4B less so.
- SOTA research documented in `docs/research/sota-simulmt-2026.md`

## Key research findings

- **Cross-lingual head transfer confirmed**: 4/5 models tested show >90% TS mass transfer
  - Implication: Can detect heads on one language pair, reuse for all others
  - Exception: Qwen3-4B has weaker transfer (79.8% mean), may need per-pair heads
- **AlignAtt validated**: CUNI won IWSLT 2025 using AlignAtt
- **SSBD** implemented -- speculative re-translation speedup
- **ExPosST** not yet implemented -- position slot reservation for KV cache
- **Open gap**: No published work on attention aggregation selection (our AMS is novel)
- **Hikari** is main competitor: policy-free WAIT tokens
- **IWSLT 2026 metrics**: LongYAAL (primary latency), XCOMET-XL (primary quality)
