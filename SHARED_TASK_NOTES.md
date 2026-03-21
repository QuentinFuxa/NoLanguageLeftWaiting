# Shared Task Notes -- NLLW SimulMT

## Current State (after Iteration 19, 2026-03-21)

**30+ SimulMT modules (~14,500+ lines), 859 tests**
**100-sentence CONFIRMED: COMET=0.894 EN-ZH, 0.881 EN-DE, 0.891 EN-IT, 0.879 CS-EN with top_p**

### What happened in Iteration 19
- **LongYAAL metric implemented** (IWSLT 2026 PRIMARY latency metric):
  - `compute_longyaal()`: word-count domain (= YAAL longform)
  - `compute_longyaal_ms()`: time-domain in milliseconds (for OmniSTEval output)
  - `compute_stream_laal()`: IWSLT 2026 secondary latency metric
  - All three added to `LatencyMetrics` dataclass, `EvalResult`, bench/eval output
  - 13 new metric tests, all passing
- **SimulStream wrapper hardened for competition**:
  - `load_model(None)` reads from environment variables (Docker-friendly)
  - `load_model(dict)` handles dict config from SimulStream server
  - Auto-detect heads config from direction + model path
  - `_update_direction()` centralizes direction switching logic
  - Env vars: `NLLW_MODEL_PATH`, `NLLW_HEADS_DIR`, `NLLW_N_GPU_LAYERS`, `NLLW_DEFAULT_DIRECTION`
  - 32 new SimulStream tests, all passing
- **Dockerfile updated**:
  - Fixed missing `requirements.txt` (now installs from pyproject.toml)
  - Added NLLW env vars for auto-configuration
  - Added health check (`python3 -c "from nllw.simulstream import ..."`)
  - Documented multi-direction support via env vars
- **Competition validator** (`scripts/validate_competition.py`):
  - 50+ checks: imports, metrics, SimulStream protocol, OmniSTEval, configs, heads, corpus, Dockerfile
  - ALL CHECKS PASSING
- **859 tests** (45 new, all passing)

### 100-Sentence Verified Results (iteration 18, with tuned p_threshold + CI)

| Direction | bd | wb | p | BLEU | COMET | 95% CI | YAAL | % offline |
|-----------|---:|---:|:-:|-----:|------:|--------|-----:|:---------:|
| **EN-ZH** | 3 | 4 | 0.85 | 40.0 | **0.894** | [0.887, 0.901] | 6.09 | 99.8% |
| **EN-DE** | 2 | 3 | 0.75 | 27.9 | **0.881** | [0.873, 0.890] | 5.45 | 99.7% |
| **EN-IT** | 2 | 3 | 0.9 | 24.3 | **0.891** | [0.882, 0.899] | 6.76 | **100.2%** |
| **CS-EN** | 3 | 3 | 0.9 | 28.4 | **0.879** | [0.871, 0.886] | 5.81 | 99.8% |

## What to do next

### Priority 1: Competition Prep (IWSLT 2026, eval April 1-15, ~10 days)
- **CRITICAL: Longform mode**: OmniSTEval produces ONE output per recording, NOT per-sentence. SoftSegmenter re-segments. Our SimulStream wrapper must NOT reset between sentences within a recording. Only `clear()` between recordings.
- **Docker build + test**: Build image, run self-test with `--test` flag. Must support `linux/arm64`.
- **SimulStream E2E**: Install simulstream package, test HTTP server integration
- **OmniSTEval validation**: Run our output through OmniSTEval locally, verify LongYAAL + COMET scores match
- **Multi-direction test**: Verify direction switching (set_source_language + set_target_language) works E2E
- **Decision**: Enable `adaptive_top_p` for competition? Phase 1 shows 6-12% latency reduction for <0.002 COMET cost

### Priority 2: Run Remaining A40 Experiments
- Collect iteration 18 A40 results (adaptive top_p validation, XCOMET-XL subprocess, variance)
- If adaptive_top_p confirmed: update IWSLT configs to enable it
- Run competition-format test (SimulStream + OmniSTEval) on A40

### Priority 3: Research Ideas (if time permits)
- **Syntax-aware chunking (SASST)**: Dependency-based word batching for better segmentation
- **ExPosST positional pre-allocation**: Pre-allocate source positions for faster KV cache reuse
- **Perplexity gain signal**: Use LLM perplexity change as border signal

### Dead Ends Confirmed (20+)
See CLAUDE.md for full list. Key ones: context injection, entropy veto, softmax_mean, signal fusion cascade, repetition halt, top_p_weighted, Qwen3.5-9B.

### Sync Workflow
IMPORTANT: When syncing code to A40, do NOT use `rsync --delete` which destroys GPU-generated configs.
Use: `rsync -avz` (without --delete) to preserve files.

## Key Context

- **IWSLT 2026**: Eval April 1-15, ~10 days away
- **Metrics**: LongYAAL (primary latency), **COMET wmt22-comet-da** (primary quality)
- **Best known**: EN-ZH COMET=0.894, EN-DE 0.881, EN-IT 0.891 (>offline!), CS-EN 0.879
- **All directions at 99.7-100.2% of offline quality**
- **Model path on A40**: `/home/fuxa/HY-MT1.5-7B.Q8_0.gguf`
- **Competition validator**: `python scripts/validate_competition.py` (all 50+ checks pass)
- **New in iter 19**: LongYAAL metrics, hardened SimulStream, Docker env vars, 859 tests
