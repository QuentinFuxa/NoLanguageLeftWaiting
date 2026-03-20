# Shared Task Notes -- NLLW SimulMT

## What exists now (after Iteration 5, 2026-03-20)

**21 SimulMT modules (~8300 lines), 244 tests passing:**

### Core (Iterations 1-4):
- `nllw/prompts.py` -- 30+ prompt formats (HY-MT, Qwen3, Qwen3.5, EuroLLM, Tower, Gemma)
- `nllw/llama_backend.py` -- ctypes wrapper for llama.cpp (attention extraction + KV cache)
- `nllw/backend_protocol.py` -- SimulMTBackend ABC + factory + `@register_backend`
- `nllw/alignatt.py` -- Core algorithm (TS-weighted vote, border detection, entropy, lookahead) + **9 aggregation methods**
- `nllw/alignatt_backend.py` -- Full backend with KV cache reuse + baselines (full-sentence, eager)
- `nllw/alignatt_la_backend.py` -- LocalAgreement + AlignAtt + SSBD + forced decoding + adaptive SSBD
- `nllw/metrics.py` -- All latency metrics (AL, LAAL, YAAL, AP, DAL, MaxCW) + BLEU/COMET + NE metric
- `nllw/bench.py` -- Unified CLI with sweep, compare, forced-decode, adaptive-ssbd flags
- `nllw/baselines.py` -- wait-k + fixed-rate baselines
- `nllw/heads/configs/` -- 22 pre-computed alignment head configs
- Plus: `eval.py`, `simulate.py`, `corpus.py`, `experiment.py`, `analysis.py`, `research.py`, `detect_heads.py`, `omnisteval.py`
- `web_debug/server.py` -- FastAPI debug server on port 8777

### New in Iteration 5:
- **Gaussian Kernel Consensus** (2 variants in `alignatt.py`):
  - `gaussian_kernel`: TS-weighted Gaussian density over head argmaxes. Sigma param.
  - `gaussian_kernel_continuous`: Full attention distribution convolution.
  - Key advantage: subword boundary tolerance. Nearby heads reinforce instead of competing.
- **LA Forced Decoding** (`_retranslate_forced()` in LA backend):
  - Force-decode committed prefix before generating continuation
  - Conditions model on stable output (consistency) + fewer tokens to generate (speed)
  - `--forced-decode` CLI flag, `forced=0,1` sweep shortname
  - Priority: SSBD > forced_decode > standard
- **Adaptive SSBD Beta** (`adaptive_ssbd_beta()` in LA backend):
  - Per-token entropy-based bias: confident=beta*1.5, uncertain=beta*0.2
  - `--adaptive-ssbd` CLI flag, `adaptive=0,1` sweep shortname
  - Capped at 0.95 to avoid degenerate acceptance
- 41 new tests (244 total, all passing)

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
python -m nllw.bench --backend alignatt-la --ssbd-beta 0.2 --adaptive-ssbd --lang en-zh --comet --save

# 1e. Forced decoding test
python -m nllw.bench --backend alignatt-la --forced-decode --lang en-zh --comet --save

# 1f. Dynamic border distance test
python -m nllw.bench --lang en-zh --dynamic-border --comet --save

# 1g. Pareto sweep
python -m nllw.bench --sweep "bd=2,3,4,5 wb=1,2,3" --lang en-zh,en-de --comet --save
```

### Priority 2: Further optimizations
- **LA two-pass catch-up**: Extra re-translation per update for stability
- **Adaptive Multi-Strategy (AMS)**: Auto-select aggregation based on attention patterns
- **Per-head temperature normalization**: Learned during head detection

### Priority 3: Research ideas (see todo.md)
- ExPosST position slot reservation
- Cross-lingual head transfer
- Dynamic word_batch based on source complexity

## Key architecture decisions

- `PromptFormat` is a frozen dataclass with `build_prompt()` method
- `BackendConfig.from_dict()` ignores unknown keys (safe for sweep configs)
- `@register_backend("name")` decorator auto-registers with factory
- FLORES+ uses `openlanguagedata/flores_plus` with per-language configs
- `detect_heads` uses SimAlign (mBERT) for ground-truth word alignments
- `aggregate()` dispatcher routes to any of 9 aggregation strategies
- LA backend retranslation priority: SSBD > forced_decode > standard
- `_longest_common_prefix_tokens()` is the core LA stability check
- SSBD uses `output_last_only=False` in `decode_batch_at` for per-position logits
- SSBD acceptance uses log-ratio trick to avoid full softmax computation
- Adaptive SSBD computes per-token entropy and scales beta proportionally
- Forced decoding includes committed_ids in the batch decode, then generates continuation
- Gaussian kernel uses TS-weighted Gaussian density; sigma controls argmax-to-mean interpolation

## Important context

- llama.cpp must be built with PR #20086. The build is on A40.
- The logit_idx bug (batch index 0, not KV pos) is fixed in `alignatt_backend.py`.
- YAAL formula: `gamma = max(|delays|, T) / S; yaal = sum(d - t/gamma) / tau`
- FLORES loading is tested and working locally.
- 5 directions available: en-fr, en-zh, en-de, en-it, cs-en
- **7 backends registered**: alignatt, alignatt-la, full-sentence, eager, wait-k, fixed-rate
- **9 aggregation methods**: ts_vote, softmax_mean, entropy_weighted, consensus, geomean, top_p, gaussian_kernel, gaussian_kernel_continuous, ensemble
- **SSBD**: `ssbd_beta=0.2` recommended, `adaptive_ssbd=True` for entropy-based modulation
- **Forced decoding**: `la_forced_decode=True` for committed prefix conditioning
- **NE metric**: `compute_normalized_erasure()` for token-level, `compute_normalized_erasure_text()` for word-level
- SOTA research documented in `docs/research/sota-simulmt-2026.md`

## Key research findings (from SOTA survey)

- **AlignAtt validated**: CUNI won IWSLT 2025 using AlignAtt. ICLR 2026 "Translation Heads" paper confirms alignment heads are sparse, universal, consistent.
- **SSBD** (arxiv 2509.21740) implemented -- can accelerate re-translation (alignatt-la) via speculative draft reuse
- **ExPosST** (arxiv 2603.14903) pre-allocates position slots for zero-recomputation KV cache
- **Open gap**: No published work on alternative attention aggregation -- our 9 methods are novel
- **Hikari** (arxiv 2603.11578) is the main competitor: policy-free WAIT tokens, SOTA on EN-JA/DE/RU
- **IWSLT 2026 metrics**: LongYAAL (primary latency), XCOMET-XL (primary quality) -- our pipeline is aligned
