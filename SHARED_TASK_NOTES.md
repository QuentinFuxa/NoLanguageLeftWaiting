# Shared Task Notes -- NLLW SimulMT

## What exists now (after Iteration 7, 2026-03-20)

**23 SimulMT modules (~9500 lines), 352 tests passing:**

### Core (Iterations 1-6):
- `nllw/prompts.py` -- 30+ prompt formats (HY-MT, Qwen3, Qwen3.5, EuroLLM, Tower, Gemma)
- `nllw/llama_backend.py` -- ctypes wrapper for llama.cpp (attention extraction + KV cache)
- `nllw/backend_protocol.py` -- SimulMTBackend ABC + factory + `@register_backend`
- `nllw/alignatt.py` -- Core algorithm + 9 aggregation + AMS + temp norm + shift-k + info gain
- `nllw/alignatt_backend.py` -- Full backend with KV cache reuse + baselines (full-sentence, eager)
- `nllw/alignatt_la_backend.py` -- LA + AlignAtt + SSBD + forced decode + adaptive SSBD + two-pass
- `nllw/metrics.py` -- All latency metrics + BLEU/COMET + NE metric
- `nllw/bench.py` -- Unified CLI with 15+ sweep shortnames
- `nllw/baselines.py` -- wait-k + fixed-rate baselines
- `nllw/heads/configs/` -- 22 pre-computed alignment head configs
- Plus: `eval.py`, `simulate.py`, `corpus.py`, `experiment.py`, `analysis.py`, `research.py`, `detect_heads.py`, `omnisteval.py`, `head_transfer.py`
- `web_debug/server.py` -- FastAPI debug server on port 8777

### New in Iteration 7:
- **Dynamic word_batch** (`compute_dynamic_word_batch()` in alignatt.py):
  - Source-length-adaptive batching: short sentences -> wb-1, long -> wb+1
  - `--dynamic-wb` CLI flag, `dynwb=0,1` sweep shortname
  - Wired into both AlignAtt and AlignAtt-LA backends
- **Attention Information Gain** (`compute_attention_info_gain()` in alignatt.py):
  - KL divergence between consecutive attention snapshots
  - Large divergence = new source info being processed -> inhibit border stop
  - Small divergence = source exhausted -> reinforce stop
  - Inspired by LSG (arxiv 2501.00868) KL(P_partial || P_full) approach
  - `--info-gain` CLI flag, `infogain=0.2,0.3,0.5` sweep shortname
- **Shift-k Adaptive Border** (`check_border_shift_k()` in alignatt.py):
  - Measure total attention MASS in border region (not just argmax position)
  - Inspired by DrFrattn (EMNLP 2025) shift-k mechanism
  - Catches split-attention cases that argmax misses
  - `--shift-k` CLI flag, `shiftk=0.3,0.4,0.5` sweep shortname
- **Combined Border Check** (`check_border_combined()` in alignatt.py):
  - Multi-signal fusion: standard AlignAtt + shift-k + info gain + dynamic border
  - Decision: info gain inhibits/reinforces, shift-k fires on mass, standard fallback
  - Returns diagnostic values for analysis
- **Border Confirmation** (`border_confirm` config, `--border-confirm` CLI):
  - Require N consecutive border hits before stopping
  - Prevents false positive stops from transient attention patterns
  - `confirm=1,2,3` sweep shortname
- **Source Complexity Estimator** (`nllw/complexity.py`):
  - Novel per-sentence complexity scoring from text features
  - Maps to adaptive parameters (bd, wb, gen_cap)
  - `estimate_complexity()`, `adaptive_params_from_complexity()`
- **Cumulative Attention Aggregation** (10th method, from DrFrattn EMNLP 2025):
  - Compute cumulative mass instead of argmax: captures distribution shape
  - Lambda parameter for continuous latency control
  - `--aggregation cumulative` CLI
- 70 new tests (360 total, all passing)

## What to do next

### Priority 1: Run experiments on A40 (all code is ready)

```bash
# 1a. First E2E validation
python -m nllw.bench --model /path/to/HY-MT1.5-7B.gguf --lang en-zh -n 20

# 1b. Full aggregation sweep (9 methods)
python -m nllw.bench --sweep "agg=ts_vote,softmax_mean,entropy_weighted,consensus,geomean,top_p,gaussian_kernel,gaussian_kernel_continuous,ensemble" --lang en-zh --comet --save

# 1c. Iteration 7 features
python -m nllw.bench --dynamic-wb --lang en-zh --comet --save
python -m nllw.bench --sweep "shiftk=0.3,0.4,0.5,0.6" --lang en-zh --comet --save
python -m nllw.bench --sweep "infogain=0.2,0.3,0.5" --lang en-zh --comet --save
python -m nllw.bench --shift-k 0.4 --info-gain 0.3 --lang en-zh --comet --save
python -m nllw.bench --sweep "confirm=1,2,3" --lang en-zh --comet --save
python -m nllw.bench --shift-k 0.4 --border-confirm 2 --lang en-zh --comet --save

# 1d. AlignAtt vs AlignAtt-LA comparison
python -m nllw.bench --compare alignatt alignatt-la --lang en-zh --comet --save

# 1e. SSBD sweep
python -m nllw.bench --backend alignatt-la --sweep "ssbd=0.0,0.1,0.2,0.3" --lang en-zh --comet --save

# 1f. Iteration 6 features
python -m nllw.bench --sweep "ams=0,1 tempnorm=0,1" --lang en-zh --comet --save

# 1g. Pareto sweep
python -m nllw.bench --sweep "bd=2,3,4,5 wb=1,2,3" --lang en-zh,en-de --comet --save
```

### Priority 2: Further optimizations
- **ExPosST position slots**: Pre-allocate KV positions for zero-recomputation
- **SSD parallel speculation**: Multiple draft continuations for 2x speedup
- **Group Position Encoding**: Separate position IDs for source/target (ACL 2025)

### Priority 3: Research ideas (see todo.md)
- GRPO fine-tuning (RL-optimize read/write)
- Syntax-aware chunking (SASST)
- REINA information gain policy (AAAI 2026)
- AliBaStr-MT learned border (Apple)

## Key architecture decisions

- `PromptFormat` is a frozen dataclass with `build_prompt()` method
- `BackendConfig.from_dict()` ignores unknown keys (safe for sweep configs)
- `@register_backend("name")` decorator auto-registers with factory
- FLORES+ uses `openlanguagedata/flores_plus` with per-language configs
- `detect_heads` uses SimAlign (mBERT) for ground-truth word alignments
- `aggregate()` dispatcher routes to any of 9 aggregation strategies
- LA backend retranslation priority: SSBD > forced_decode > standard
- Two-pass uses pass 1 (SSBD/forced/standard) + pass 2 (always standard) for diversity
- `_check_border()` helper centralizes all border check params (AMS, temp norm, dynamic, shift-k, info gain)
- `check_border_combined()` fuses multiple signals: standard + shift-k mass + info gain
- Cross-lingual head transfer: most models have >90% TS mass transfer across pairs

## Important context

- llama.cpp must be built with PR #20086. The build is on A40.
- The logit_idx bug (batch index 0, not KV pos) is fixed in `alignatt_backend.py`.
- YAAL formula: `gamma = max(|delays|, T) / S; yaal = sum(d - t/gamma) / tau`
- FLORES loading is tested and working locally.
- 5 directions available: en-fr, en-zh, en-de, en-it, cs-en
- **7 backends registered**: alignatt, alignatt-la, full-sentence, eager, wait-k, fixed-rate
- **10 aggregation methods**: ts_vote, softmax_mean, entropy_weighted, consensus, geomean, top_p, gaussian_kernel, gaussian_kernel_continuous, cumulative, ensemble
- **New iter7 params**: `dynamic_word_batch`, `info_gain_threshold`, `shift_k_threshold`, `border_confirm`
- **New iter7 sweeps**: `dynwb=0,1`, `infogain=0.2,0.3,0.5`, `shiftk=0.3,0.4,0.5`, `confirm=1,2,3`
- **New module**: `nllw/complexity.py` -- source complexity estimation for adaptive params
- **Head transfer**: EuroLLM/HY-MT/Qwen3.5 heads are cross-lingually universal. Qwen3-4B less so.
- SOTA research documented in `docs/research/sota-simulmt-2026.md`

## Key research findings

- **Cross-lingual head transfer confirmed**: 4/5 models tested show >90% TS mass transfer
- **AlignAtt validated**: CUNI won IWSLT 2025 using AlignAtt
- **SSBD** implemented -- speculative re-translation speedup
- **Open gap**: No published work on attention aggregation selection (our AMS is novel)
- **Open gap**: No published work on attention mass border detection (our shift-k is novel)
- **Open gap**: No published work on info gain modulation for SimulMT border detection
- **Hikari** is main competitor: policy-free WAIT tokens
- **IWSLT 2026 metrics**: LongYAAL (primary latency), XCOMET-XL (primary quality)
