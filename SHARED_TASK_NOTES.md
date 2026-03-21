# Shared Task Notes -- NLLW SimulMT

## Current State (after Iteration 21, 2026-03-21)

**30+ SimulMT modules (~15,000 lines), 920 tests**
**100-sentence CONFIRMED: COMET=0.894 EN-ZH, 0.881 EN-DE, 0.891 EN-IT, 0.879 CS-EN with top_p**

### What happened in Iteration 21
- **OmniSTEval format hardened** (CRITICAL for competition):
  - NFKC normalization for char-level delays (matching `ss-to-log.py` reference)
  - LLM artifact stripping: `<end_of_turn>`, `<|endoftext|>` removed from prediction
  - Monotonic delay enforcement: delays never decrease (OmniSTEval requirement)
  - Event text normalization in char-to-delay mapping
- **Source-aware word batching** (new technique):
  - `source_aware_batching` config + `--source-aware-batch` CLI
  - Defers translation when batch ends on a function word (the, of, in, etc.)
  - Function word lists for English (60+) and Czech (30+)
  - Max 2 extra words deferred per batch to bound latency
  - Sweep: `srcaware=0,1`
  - **Needs GPU testing** to measure quality improvement
- **Comprehensive GPU experiment script** (`scripts/run_iteration21_experiments.py`):
  - Phase 1: Smoke test (all 4 directions, 20 sentences)
  - Phase 2: Perplexity adaptive border (100 sentences, threshold sweep)
  - Phase 3: Longform E2E (gold transcript -> OmniSTEval JSONL)
  - Phase 4: Multi-direction longform validation
  - Phase 5: Competition format output (100 sentences, OmniSTEval)
  - Phase 6: Adaptive top_p decision (fixed vs adaptive)
  - Phase 7: Source-aware batching (fixed vs source-aware)
- **Longform bug fixes** (from code review):
  - char_level auto-detection for zh/ja/ko targets (prevents invalid OmniSTEval output)
  - n_ctx overflow protection (safety valve at 70% of context window)
  - Delay count validation with auto-fix in to_omnisteval_entry()
- **Research update**: CUNI won IWSLT 2025 with same AlignAtt+LA architecture. Translation Heads (ICLR 2026) validates our head detection. Attribution-Guided Decoding (ICLR 2026) is promising zero-shot quality signal.
- **920 tests** (27 new, all passing), **98 competition checks** all passing

### 100-Sentence Verified Results (iteration 18, with tuned p_threshold + CI)

| Direction | bd | wb | p | BLEU | COMET | 95% CI | YAAL | % offline |
|-----------|---:|---:|:-:|-----:|------:|--------|-----:|:---------:|
| **EN-ZH** | 3 | 4 | 0.85 | 40.0 | **0.894** | [0.887, 0.901] | 6.09 | 99.8% |
| **EN-DE** | 2 | 3 | 0.75 | 27.9 | **0.881** | [0.873, 0.890] | 5.45 | 99.7% |
| **EN-IT** | 2 | 3 | 0.9 | 24.3 | **0.891** | [0.882, 0.899] | 6.76 | **100.2%** |
| **CS-EN** | 3 | 3 | 0.9 | 28.4 | **0.879** | [0.871, 0.886] | 5.81 | 99.8% |

## What to do next

### Priority 1: RUN GPU EXPERIMENTS (URGENT -- competition in ~10 days)
- **Run iteration 21 script on A40**:
  ```bash
  # Sync code to A40 first:
  rsync -avz . fuxa@quest.ms.mff.cuni.cz:nllw_deploy/ -e "ssh -p 3622"
  # Then on A40:
  python scripts/run_iteration21_experiments.py --model /home/fuxa/HY-MT1.5-7B.Q8_0.gguf
  # Quick mode for validation:
  python scripts/run_iteration21_experiments.py --model /home/fuxa/HY-MT1.5-7B.Q8_0.gguf --quick
  # With real gold transcript:
  python scripts/run_iteration21_experiments.py --model /home/fuxa/HY-MT1.5-7B.Q8_0.gguf --phase 3 \
      --gold-jsonl /path/to/iwslt26-sst/inputs/en/acl6060.ts/gold-jsonl/2022.acl-long.117.jsonl
  ```
- Key experiments that NEED results:
  - Perplexity adaptive border: YAAL improvement vs quality cost
  - Source-aware batching: quality improvement from better translation units
  - Adaptive top_p: confirm 6-12% YAAL reduction for competition decision
  - Longform E2E: verify OmniSTEval output matches reference format

### Priority 2: Competition Decisions (based on GPU results)
- **Enable perplexity adaptive bd?** If YAAL improves without quality loss -> yes
- **Enable source-aware batching?** If COMET improves -> yes
- **Enable adaptive top_p?** Phase 1 shows promising, need 100-sent confirmation
- **Update IWSLT configs** with any winning features

### Priority 3: Docker + SimulStream E2E
- Docker build + test on linux/amd64
- SimulStream HTTP server integration test
- Full pipeline: SimulStream -> NLLWSpeechProcessor -> OmniSTEval output

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
- **Competition validator**: `python scripts/validate_competition.py` (65+ checks pass)
- **New in iter 21**: OmniSTEval format fixes, source-aware batching, 7-phase GPU script, 920 tests
- **OmniSTEval format**: ONE JSONL line per recording with per-word delays in ms
  - Reference converter: `iwslt26-sst/evaluation/ss-to-log.py`
  - Run: `omnisteval longform --speech_segmentation ... --ref_sentences_file ... --hypothesis_file out.log`
- **Gold transcripts**: `iwslt26-sst/inputs/en/acl6060.ts/gold-jsonl/` (1863 words per recording)
