# Shared Task Notes -- NLLW SimulMT

## Current State (after Iteration 22, 2026-03-21)

**30+ SimulMT modules (~15,000 lines), 928 tests**
**100-sentence CONFIRMED: COMET=0.894 EN-ZH, 0.881 EN-DE, 0.891 EN-IT, 0.879 CS-EN with top_p**

### What happened in Iteration 22
- **YAAL computation fix** (CRITICAL for competition):
  - CU delays now use FIRST word in batch, not last (matching reference implementation)
  - `batch_first_emission_time` tracked in both AlignAtt and AlignAtt-LA backends
  - SimulStream wrapper uses batch start time for EmissionEvent CU attribution
  - Expected YAAL improvement: ~(wb-1) * word_interval per batch (~3 words for wb=4)
  - **Needs GPU validation** to measure actual YAAL reduction
- **Per-step generation confidence** (`avg_logprob` in TranslationStep):
  - Diagnostic: identifies low-confidence segments
  - Future use: retry logic for bad translations
- **Model warmup** (`warmup()` on AlignAttBackend):
  - Eliminates GPU cold-start latency (JIT compilation, memory allocation)
  - Auto-called during SimulStream initialization
- **SimulEvalEntry CJK validation fix**: validate() now accepts char-level delays for zh/ja/ko
- **Dockerfile fix**: added missing `simulstream` dependency (container wouldn't start without it)
- **101 competition checks** (3 new, all passing)
- **7-phase GPU experiment script** (`scripts/run_iteration22_experiments.py`)
- **930 tests** (10 new, all passing)

### 100-Sentence Verified Results (iteration 18, with tuned p_threshold + CI)

| Direction | bd | wb | p | BLEU | COMET | 95% CI | YAAL | % offline |
|-----------|---:|---:|:-:|-----:|------:|--------|-----:|:---------:|
| **EN-ZH** | 3 | 4 | 0.85 | 40.0 | **0.894** | [0.887, 0.901] | 6.09 | 99.8% |
| **EN-DE** | 2 | 3 | 0.75 | 27.9 | **0.881** | [0.873, 0.890] | 5.45 | 99.7% |
| **EN-IT** | 2 | 3 | 0.9 | 24.3 | **0.891** | [0.882, 0.899] | 6.76 | **100.2%** |
| **CS-EN** | 3 | 3 | 0.9 | 28.4 | **0.879** | [0.871, 0.886] | 5.81 | 99.8% |

## What to do next

### Priority 1: RUN GPU EXPERIMENTS (URGENT -- competition in ~8 days)
- **Run iteration 22 script on A40**:
  ```bash
  # Sync code to A40 first:
  rsync -avz . fuxa@quest.ms.mff.cuni.cz:nllw_deploy/ -e "ssh -p 3622"
  # Then on A40:
  python scripts/run_iteration22_experiments.py --model /home/fuxa/HY-MT1.5-7B.Q8_0.gguf
  # Quick mode for validation:
  python scripts/run_iteration22_experiments.py --model /home/fuxa/HY-MT1.5-7B.Q8_0.gguf --quick
  # Specific phase:
  python scripts/run_iteration22_experiments.py --model /home/fuxa/HY-MT1.5-7B.Q8_0.gguf --phase 1
  ```
- Key experiments that NEED results:
  - **YAAL fix validation**: batch_first_emission_time should lower YAAL without quality change
  - **Source-aware batching**: quality improvement from better translation units
  - **Perplexity adaptive border**: YAAL improvement vs quality cost
  - **Adaptive top_p**: confirm 6-12% YAAL reduction for competition decision
  - **Longform E2E**: verify full pipeline produces valid OmniSTEval output
  - **Combined features**: test all winning features together

### Priority 2: Competition Decisions (based on GPU results)
- **Enable perplexity adaptive bd?** If YAAL improves without quality loss -> yes
- **Enable source-aware batching?** If COMET improves -> yes
- **Enable adaptive top_p?** Phase 1 shows promising, need 100-sent confirmation
- **Update IWSLT configs** with any winning features

### Priority 3: ASR Integration + Docker E2E
- `_run_asr()` in simulstream.py is still a stub
- Need Qwen3-ASR or external ASR integration
- Docker build + test on linux/amd64
- Full pipeline: audio chunks -> ASR -> NLLWSpeechProcessor -> OmniSTEval

### Priority 4: Research Opportunities (from iteration 22 survey)
- **IWSLT 2026 baselines repo**: https://github.com/owaski/iwslt-2026-baselines
  - Cascade: Qwen3-ASR-1.7B + Qwen3-4B-Instruct + LocalAgreement
  - Context injection (paper NEs) consistently improves quality
  - "Extra Context" subtrack is new opportunity
- **Translation Heads (ICLR 2026)**: Compare their head scoring with our TS scoring
  - Only 3-8% of heads matter, universal across LLMs
  - Could find better heads or confirm our approach
- **CUNI won IWSLT 2025**: AlignAtt + EuroLLM-9B + forced decoding + context
  - Validates our AlignAtt approach is SOTA for offline models
- **Field consensus**: decoder-only LLMs + AlignAtt = SOTA. Our 99.8% of offline is near frontier.

### Dead Ends Confirmed (20+)
See CLAUDE.md for full list. Key ones: context injection, entropy veto, softmax_mean, signal fusion cascade, repetition halt (EN-ZH), top_p_weighted, Qwen3.5-9B.

### Sync Workflow
IMPORTANT: When syncing code to A40, do NOT use `rsync --delete` which destroys GPU-generated configs.
Use: `rsync -avz` (without --delete) to preserve files.

## Key Context

- **IWSLT 2026**: Eval April 1-15, ~8 days away
- **Metrics**: LongYAAL (primary latency), **COMET wmt22-comet-da** (primary quality)
- **Best known**: EN-ZH COMET=0.894, EN-DE 0.881, EN-IT 0.891 (>offline!), CS-EN 0.879
- **All directions at 99.7-100.2% of offline quality**
- **Model path on A40**: `/home/fuxa/HY-MT1.5-7B.Q8_0.gguf`
- **Competition validator**: `python scripts/validate_competition.py` (101 checks pass)
- **New in iter 22**: YAAL fix (batch_first_emission_time), warmup, avg_logprob, CJK validation fix, Dockerfile fix, 930 tests
- **OmniSTEval format**: ONE JSONL line per recording with per-word delays in ms
- **Gold transcripts**: `iwslt26-sst/inputs/en/acl6060.ts/gold-jsonl/` (1863 words per recording)
