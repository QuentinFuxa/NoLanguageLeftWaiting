# Shared Task Notes -- NLLW SimulMT

## Current State (after Iteration 18, 2026-03-21)

**29+ SimulMT modules (~14,000+ lines), 810+ tests**
**100-sentence CONFIRMED: COMET=0.892 EN-ZH, 0.881 EN-DE with top_p**

### What happened in Iteration 18
- **XCOMET-XL separate-process scorer** (`nllw/xcomet_scorer.py`):
  - Solves persistent OOM: runs XCOMET-XL in fresh subprocess (no translation model loaded)
  - Integrated into eval.py: `--xcomet` flag now uses subprocess
  - Phase 2 experiments validating on A40 (running)
- **Adaptive top_p threshold** (novel):
  - Per-sentence threshold from source complexity (simple -> lower, complex -> higher)
  - **EN-ZH: COMET=0.895 (-0.001), YAAL=5.43 (-0.57 latency!)** -- trades 0.1% quality for 9.5% latency reduction
  - Other directions running on A40
- **Bootstrap confidence intervals** for COMET scores:
  - `bootstrap_confidence_interval()` + `paired_bootstrap_test()` in metrics.py
  - Integrated into eval output: shows [lo, hi] 95% CI alongside point estimates
- **Comprehensive research update**:
  - Qwen3.5-9B: NOT suitable for AlignAtt (hybrid DeltaNet, 25% softmax layers)
  - ExPosST (2603.14903): positional pre-allocation for latency, medium effort
  - Translation Heads (ICLR 2026): validates our TS-scoring approach
  - IWSLT 2026 uses COMET wmt22-comet-da for ranking -- we're correctly aligned

### Phase 1 Complete: Adaptive top_p vs Fixed (A40, 50 sentences)

| Direction | Config | COMET | 95% CI | YAAL | Delta COMET | Delta YAAL |
|-----------|--------|------:|--------|-----:|:-----------:|:----------:|
| **EN-ZH** | Fixed p=0.85 | **0.896** | - | 6.00 | - | - |
| EN-ZH | Adaptive | 0.895 | - | **5.43** | -0.001 | -0.57 |
| **EN-DE** | Fixed p=0.75 | **0.881** | - | 5.55 | - | - |
| EN-DE | Adaptive | 0.879 | [0.868, 0.891] | **5.18** | -0.002 | -0.37 |
| **EN-IT** | Fixed p=0.9 | **0.891** | [0.880, 0.901] | 6.75 | - | - |
| EN-IT | Adaptive | 0.890 | [0.879, 0.901] | **6.27** | -0.001 | -0.48 |
| **CS-EN** | Fixed p=0.9 | **0.876** | [0.866, 0.887] | 6.03 | - | - |
| CS-EN | Adaptive | 0.874 | [0.863, 0.885] | **5.33** | -0.002 | -0.70 |

**Key finding**: Adaptive top_p reduces YAAL by 0.37-0.70 words (6-12% latency) for 0.001-0.002 COMET (overlapping CIs = not significant). Great for competition.

### 100-Sentence Verified Results (iteration 18, with tuned p_threshold + CI)

| Direction | bd | wb | p | BLEU | COMET | 95% CI | YAAL | % offline |
|-----------|---:|---:|:-:|-----:|------:|--------|-----:|:---------:|
| **EN-ZH** | 3 | 4 | 0.85 | 40.0 | **0.894** | [0.887, 0.901] | 6.09 | 99.8% |
| **EN-DE** | 2 | 3 | 0.75 | 27.9 | **0.881** | [0.873, 0.890] | 5.45 | 99.7% |
| **EN-IT** | 2 | 3 | 0.9 | 24.3 | **0.891** | [0.882, 0.899] | 6.76 | **100.2%** |
| **CS-EN** | 3 | 3 | 0.9 | 28.4 | **0.879** | [0.871, 0.886] | 5.81 | 99.8% |

## What to do next

### Priority 1: Collect A40 Experiment Results (Running)
Iteration 18 experiments on A40 (`iter18_results.log`):
- Phase 1: Adaptive top_p vs fixed (4 directions) -- EN-ZH done, others running
- Phase 2: XCOMET-XL subprocess validation
- Phase 3: 100-sentence tuned p_threshold confirmation
- Phase 4: Variance estimation

### Priority 2: Competition Prep (IWSLT 2026, eval April 1-15, ~10 days)
- Docker finalization (OmniSTEval format, SimulStream wrapper)
- SimulStream E2E test with audio
- OmniSTEval longform output verification
- Verify LongYAAL compatibility with SimulStream toolkit
- **Decision**: adaptive_top_p? If Phase 1 shows consistent latency gains with <0.002 COMET loss, enable it for competition.

### Priority 3: Research-Informed Improvements
- **ExPosST positional pre-allocation** (arXiv 2603.14903): Pre-allocate source positions to avoid KV invalidation. Medium effort, could reduce latency.
- **Perplexity gain as fusion signal**: Use LLM's own perplexity change as border signal. Small effort, uncertain benefit.
- **100-sentence verification at tuned p_threshold**: Need 100-sentence confirmation (Phase 3 on A40)

### Priority 4: XCOMET-XL Scoring
- Phase 2 on A40 validates the subprocess scorer
- If it works, score all 4 directions at 100 sentences
- XCOMET-XL results are informative but IWSLT 2026 uses COMET (wmt22-comet-da)

### Dead Ends Confirmed
- Context injection: KILLS quality for HY-MT (-0.084 to -0.125 COMET)
- Entropy veto: All thresholds hurt
- softmax_mean aggregation: COMET 0.812 (terrible)
- Signal fusion/cascade: marginal +0.002
- wb=5/6 with top_p: Saturated
- Repetition halt: HURTS EN-ZH by -0.004 COMET
- top_p_weighted aggregation: COMET 0.885 EN-ZH (much worse than top_p)
- Qwen3.5-9B: Hybrid DeltaNet architecture, only 25% extractable attention layers

### Sync Workflow
IMPORTANT: When syncing code to A40, do NOT use `rsync --delete` which destroys GPU-generated configs.
Use: `rsync -avz` (without --delete) to preserve files.

## Key Context

- **IWSLT 2026**: Eval April 1-15, ~10 days away
- **Metrics**: LongYAAL (primary latency), **COMET wmt22-comet-da** (primary quality, confirmed for ranking)
- **Best known (100-sent confirmed)**: EN-ZH COMET=0.892, EN-DE 0.881, EN-IT **0.890** (>offline!), CS-EN 0.877
- **All directions at 99.5-100.1% of offline quality** with just AlignAtt + top_p
- **A40 running**: iteration 18 experiments (adaptive top_p, XCOMET-XL, variance)
- **New features in iter 18**: xcomet_scorer.py, adaptive_top_p, bootstrap CI
- **Model path on A40**: `/home/fuxa/HY-MT1.5-7B.Q8_0.gguf` (NOT in models/ dir)
