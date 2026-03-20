# Shared Task Notes -- NLLW SimulMT

## What exists now (after Iteration 3, 2026-03-20)

**19 SimulMT modules (~7200 lines), 172 tests passing:**

### Core (Iteration 1):
- `nllw/prompts.py` -- 30+ prompt formats (HY-MT, Qwen3, Qwen3.5, EuroLLM, Tower, Gemma)
- `nllw/llama_backend.py` -- ctypes wrapper for llama.cpp (attention extraction + KV cache)
- `nllw/backend_protocol.py` -- SimulMTBackend ABC + factory + `@register_backend`
- `nllw/alignatt.py` -- Core algorithm (TS-weighted vote, border detection, entropy, lookahead) + **6 novel aggregation methods**
- `nllw/alignatt_backend.py` -- Full backend with KV cache reuse + baselines (full-sentence, eager)
- `nllw/metrics.py` -- AL, LAAL, YAAL, AP, DAL, MaxCW + BLEU/COMET wrappers
- `nllw/simulate.py` -- Policy simulation and trace replay
- `nllw/eval.py` -- Evaluation harness (FLORES+, parameter sweep, XCOMET-XL)
- `nllw/bench.py` -- Unified CLI (`python -m nllw.bench --lang en-fr --comet`)
- `nllw/heads/configs/` -- 22 pre-computed alignment head configs

### Iteration 2:
- `nllw/detect_heads.py` -- Auto alignment head detection for any GGUF model (559 lines)
- `nllw/omnisteval.py` -- OmniSTEval JSONL export for IWSLT 2026 (258 lines)
- `nllw/baselines.py` -- wait-k + fixed-rate baselines (175 lines)
- `nllw/analysis.py` -- Pareto frontier, edge cases, report generation (309 lines)
- `nllw/experiment.py` -- Experiment YAML config + file-based result registry (359 lines)
- `nllw/corpus.py` -- Expanded: 120 sentences across 5 directions (622 lines)
- `nllw/research.py` -- Compute-aware latency (CA-AL, CA-YAAL), benchmark suite (191 lines)

### New (Iteration 3):
- `nllw/alignatt_la_backend.py` -- **LocalAgreement + AlignAtt hybrid** (~280 lines)
  - Re-translates full source each time, diffs with previous, commits stable prefix
  - KV cache reuse for prefix (only re-decode source+suffix)
  - Token-level agreement check for precision
- **7 aggregation methods** in `alignatt.py` (open research gap -- no published work):
  - `ts_vote` (original), `softmax_mean`, `entropy_weighted`, `consensus`, `geomean`, `top_p`, `ensemble`
  - Ensemble: weighted average of multiple methods (default: ts_vote 0.4 + entropy_weighted 0.3 + geomean 0.3)
  - Unified `aggregate()` dispatcher + `check_border()` with `aggregation` parameter
- **Dynamic border distance**: `check_border_dynamic()` adjusts bd per-token based on attention entropy
  - Sharp attention -> tighter border (aggressive); diffuse attention -> wider border (conservative)
  - `--dynamic-border` CLI flag, `dynbd` sweep shortname
- `aggregation` and `dynamic_border` fields in `BackendConfig`, wired into both backends
- `--aggregation` and `--dynamic-border` CLI flags in bench.py
- 8 experiment configs in `configs/` (4 new: aggregation sweep, LA comparison, dynamic border)

## What to do next

### Priority 1: Run experiments on A40 (all code is ready)

```bash
# 1a. First E2E validation
python -m nllw.bench --model /path/to/HY-MT1.5-7B.gguf --lang en-zh -n 20

# 1b. Novel aggregation sweep (7 methods, no published baselines)
python -m nllw.bench --sweep "agg=ts_vote,softmax_mean,entropy_weighted,consensus,geomean,top_p,ensemble" --lang en-zh --comet --save

# 1c. AlignAtt vs AlignAtt-LA comparison
python -m nllw.bench --compare alignatt alignatt-la --lang en-zh --comet --save

# 1d. Dynamic border distance test
python -m nllw.bench --lang en-zh --dynamic-border --comet --save

# 1e. Pareto sweep
python -m nllw.bench --sweep "bd=2,3,4,5 wb=1,2,3" --lang en-zh,en-de --comet --save
```

### Priority 2: SSBD implementation (highest-value optimization)
- Self-Speculative Biased Decoding (arxiv 2509.21740) for alignatt-la: 1.3-1.7x speedup
- Feed previous translation as draft, verify in one parallel pass, resume from first divergence
- Beta=0.2 recommended, zero quality loss
- See `docs/research/local-agreement-research.md` for full algorithm

### Priority 3: Further LA optimizations
- Forced decoding of committed prefix (CUNI approach)
- Two-pass catch-up (extra LA comparison per update)
- NE (Normalized Erasure) metric implementation for measuring output stability

## Key architecture decisions

- `PromptFormat` is a frozen dataclass with `build_prompt()` method
- `BackendConfig.from_dict()` ignores unknown keys (safe for sweep configs)
- `@register_backend("name")` decorator auto-registers with factory
- `simulate_backend()` feeds words one at a time, records delays
- FLORES+ uses `openlanguagedata/flores_plus` with per-language configs (not pairs)
- `detect_heads` uses SimAlign (mBERT) for ground-truth word alignments
- `ExperimentConfig` supports both flat dicts and nested YAML format
- `ParetoAnalyzer` computes quality-latency frontiers across experiments
- **NEW**: `aggregate()` dispatcher routes to any of 6 aggregation strategies
- **NEW**: LA backend clears KV cache from prefix onwards for each re-translation (prefix reused)
- **NEW**: `_longest_common_prefix_tokens()` is the core LA stability check

## Important context

- llama.cpp must be built with PR #20086. The build is on A40.
- The logit_idx bug (batch index 0, not KV pos) is fixed in `alignatt_backend.py`.
- YAAL formula: `gamma = max(|delays|, T) / S; yaal = sum(d - t/gamma) / tau`
- FLORES loading is tested and working locally.
- 5 directions available: en-fr, en-zh, en-de, en-it, cs-en
- **7 backends registered**: alignatt, alignatt-la, full-sentence, eager, wait-k, fixed-rate
- **7 aggregation methods**: ts_vote, softmax_mean, entropy_weighted, consensus, geomean, top_p, ensemble
- SOTA research documented in `docs/research/sota-simulmt-2026.md`

## Key research findings (from SOTA survey)

- **AlignAtt validated**: CUNI won IWSLT 2025 using AlignAtt. ICLR 2026 "Translation Heads" paper confirms alignment heads are sparse, universal, consistent.
- **SSBD** (arxiv 2509.21740) can accelerate re-translation (alignatt-la) via speculative draft reuse
- **ExPosST** (arxiv 2603.14903) pre-allocates position slots for zero-recomputation KV cache
- **Open gap**: No published work on alternative attention aggregation -- our 6 methods are novel
- **Hikari** (arxiv 2603.11578) is the main competitor: policy-free WAIT tokens, SOTA on EN-JA/DE/RU
- **IWSLT 2026 metrics**: LongYAAL (primary latency), XCOMET-XL (primary quality) -- our pipeline is aligned
