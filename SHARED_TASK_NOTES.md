# Shared Task Notes -- NLLW SimulMT

## Current State (after Iteration 17, 2026-03-21)

**28+ SimulMT modules (~13,500+ lines), 785+ tests**
**100-sentence CONFIRMED: COMET=0.892 EN-ZH, 0.881 EN-DE with top_p**

### What happened in Iteration 17
- **100-sentence verification of top_p configs** (running on A40):
  - EN-ZH bd=3/wb=4/top_p: **COMET=0.892** BLEU=39.9 YAAL=5.84 (confirmed)
  - EN-ZH bd=3/wb=5/top_p: COMET=0.891 (wb=5 doesn't help with top_p)
  - EN-ZH bd=2/wb=3/top_p: COMET=0.889 YAAL=4.29 (balanced config)
  - EN-DE bd=2/wb=3/top_p: **COMET=0.881** BLEU=27.8 YAAL=5.78 (confirmed)
  - EN-DE bd=2/wb=4/top_p: COMET=0.880 (wb=4 doesn't help EN-DE)
  - EN-IT, CS-EN, wb=6 experiments still running
- **Updated ALL competition configs**: IWSLT 2026 YAMLs + SimulStream DIRECTION_DEFAULTS now use top_p + optimal bd/wb
- **Added `aggregation` field to SimulStreamConfig**: was silently dropped before -- now properly wired into `to_backend_config()`
- **Added `top_p_threshold` parameter**: Tunable p_threshold for top_p aggregation (0.5-0.95). Default 0.8 (never tuned -- sweep script prepared)
- **Added `top_p_weighted` aggregation**: Weighted mean of top-p positions instead of rightmost. Continuous output, potentially more robust
- **Added `repetition_max_repeats=2` to all competition configs**: Zero overhead safety net
- **Fixed XCOMET-XL OOM**: Added gc.collect() + torch.cuda.empty_cache() + 2s sleep after backend.close()
- **Previous XCOMET-XL run still OOM'd (rc=-9)**: All 4 directions failed. Fix needs re-verification

### 100-Sentence Verified Results (FLORES, A40, HY-MT1.5-7B, top_p)

| Direction | bd | wb | agg | BLEU | COMET | YAAL | % offline | Notes |
|-----------|---:|---:|-----|-----:|------:|-----:|:---------:|-------|
| **EN-ZH** | 3 | 4 | top_p | 39.9 | **0.892** | 5.84 | 99.6% | Best quality |
| EN-ZH | 3 | 5 | top_p | 39.5 | 0.891 | 6.17 | 99.4% | wb=5 saturates |
| EN-ZH | 2 | 3 | top_p | 38.4 | 0.889 | 4.29 | 99.2% | Balanced |
| **EN-DE** | 2 | 3 | top_p | 27.8 | **0.881** | 5.78 | 99.7% | Best quality |
| EN-DE | 2 | 4 | top_p | 28.4 | 0.880 | 6.22 | 99.5% | wb=4 no help |
| **EN-IT** | 2 | 3 | top_p | 24.5 | **0.890** | 5.76 | **100.1%** | Exceeds offline! |
| EN-IT | 2 | 4 | top_p | 24.0 | 0.887 | 6.07 | 99.8% | wb=4 no help |
| **CS-EN** | 3 | 3 | top_p | 27.5 | **0.877** | 4.90 | 99.5% | Confirmed |
| CS-EN | 3 | 4 | top_p | 28.5 | 0.878 | 5.25 | 99.7% | Marginal +0.001 |

### Phase 2-5 Results (50 sentences, A40)

**wb=6 exploration:** EN-ZH wb=6 COMET=0.895 (+0.003 but YAAL=6.91 too high). EN-DE wb=5 COMET=0.882 (marginal). EN-IT wb=5 COMET=0.887 (hurts).

**Dedicated heads:** Same COMET as cross-lingual transfer. Confirms universality.

**XCOMET-XL: ALL OOM'd (rc=-9)**. Must use `run_xcomet_separate.py` (separate process).

**Repetition halt hurts EN-ZH (-0.004 COMET)**. Disabled in all competition configs.

### Key Finding: wb increase saturates with top_p
- With top_p: wb=4 is optimal; wb=5/6 give no further improvement
- **Optimal: top_p + wb=4 for EN-ZH, wb=3 for others**

### Dead Ends Confirmed
- Context injection: KILLS quality for HY-MT
- Entropy veto: All thresholds hurt
- softmax_mean aggregation: COMET 0.812 (terrible)
- Signal fusion/cascade: marginal +0.002
- bd=1: Too aggressive
- wb=5/6 with top_p: Saturated
- wb=4 for EN-DE/IT: No improvement over wb=3
- **Repetition halt: HURTS EN-ZH by -0.004 COMET. DO NOT USE.**

## What to do next

### Priority 1: Complete Running Experiments (A40)
Iteration 17 experiments running on A40:
- Phase 1: EN-IT + CS-EN 100-sentence verification (in progress)
- Phase 2: wb=6 exploration
- Phase 3: Dedicated heads comparison
- Phase 4: XCOMET-XL scoring (may still OOM)
- Phase 5: Repetition halt safety check

### Priority 2: top_p Threshold Tuning -- DONE!
**BREAKTHROUGH: p=0.85 gives COMET 0.896 EN-ZH = offline quality!**

Optimal thresholds (50 sentences, applied to configs):
| Direction | p_threshold | COMET | YAAL |
|-----------|:-----------:|------:|-----:|
| EN-ZH | **0.85** | **0.896** | 6.00 |
| EN-DE | 0.75 | 0.881 | 5.55 |
| EN-IT | 0.9 | 0.891 | 5.94 |
| CS-EN | 0.9 | 0.876 | 6.03 |

`top_p_weighted` variant is a dead end (COMET 0.885 EN-ZH, 0.852 EN-DE).

Previous script info:
```bash
ssh -p 3622 fuxa@quest.ms.mff.cuni.cz "cd /home/fuxa/nllw_deploy && export LLAMA_CPP_LIB=/home/fuxa/llama.cpp/build/bin/libllama.so && nohup python3 run_topp_tuning.py > topp_tuning_results.log 2>&1 &"
```
Tests: p=0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95 for EN-ZH + top_p_weighted variant.

### Priority 3: XCOMET-XL (Needs Robust Fix)
Previous OOM fix (backend.close + gc + torch cache) may still not be enough.
Options:
- Run translation and XCOMET-XL as **separate processes** (most reliable)
- Save hypotheses to file, then score in a new process
- Try with lower batch_size (currently 8)

### Priority 4: Competition Prep (IWSLT 2026, eval April 1-15, ~10 days)
- Docker finalization (OmniSTEval format, SimulStream wrapper)
- SimulStream E2E test with audio
- OmniSTEval longform output verification
- Update Dockerfile to include NLLW as package
- Verify LongYAAL compatibility with SimulStream toolkit

### Priority 5: Research-Informed Improvements (from web search)
- **Hibiki perplexity-based adaptive border**: Use offline MT perplexity on partial vs full source to dynamically adjust border_distance per-word. No training needed. arXiv 2502.03382
- **Translation Heads (ICLR 2026)**: Validates our TS-scoring + middle-layer head detection. Could refine head pruning. OpenReview q8fTgw8e5E
- **CUNI (2025 winner)**: AlignAtt with Whisper, LocalAgreement for LLMs. Forced decoding. arXiv 2506.17077
- **SimulSense semantic batching**: Batch by semantic units rather than fixed word_batch
- **Qwen3.5-9B**: 201 languages, might outperform HY-MT on some directions. Test if GGUF available.

### Sync Workflow
IMPORTANT: When syncing code to A40, do NOT use `rsync --delete` which destroys GPU-generated configs.
Use: `rsync -avz` (without --delete) to preserve files.
Always pull new configs from A40 BEFORE pushing code.

## Key Context

- **IWSLT 2026**: Eval April 1-15, ~10 days away
- **Metrics**: LongYAAL (primary latency), COMET (primary quality)
- **Best known (100-sent confirmed)**: EN-ZH COMET=0.892, EN-DE 0.881, EN-IT **0.890** (>offline!), CS-EN 0.877
- **All directions at 99.5-100.1% of offline quality** with just AlignAtt + top_p
- **A40 ready**: experiments running, code synced with top_p_threshold
- **New features in iter 17**: top_p_threshold param, top_p_weighted agg, updated DIRECTION_DEFAULTS
