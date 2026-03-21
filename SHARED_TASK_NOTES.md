# Shared Task Notes -- NLLW SimulMT

## Current State (after Iteration 23, 2026-03-21)

**30+ SimulMT modules (~15,500 lines), 955 tests**
**100-sentence CONFIRMED: COMET=0.894 EN-ZH, 0.881 EN-DE, 0.891 EN-IT, 0.879 CS-EN with top_p**

### What happened in Iteration 23
- **Confidence-adaptive word batching** (NOVEL):
  - Uses avg_logprob from iter 22 to adjust word_batch per translate() call
  - Confident generation (logprob > -0.5) -> wb-1 (faster emission, lower YAAL)
  - Uncertain generation (logprob < -2.0) -> wb+1 (more context, better quality)
  - Wired into both AlignAtt and AlignAtt-LA backends
  - CLI: `--confidence-adaptive-wb`, sweep: `confwb=0,1`
  - **Needs GPU validation**
- **Language-pair-aware gen cap**:
  - Tighter generation limits for compact targets (EN->ZH: 0.85 ratio)
  - Looser for verbose targets (EN->DE: 1.15 ratio)
  - CLI: `--language-pair-gen-cap`, sweep: `lpgcap=0,1`
  - **Needs GPU validation**
- **avg_logprob in LA backend**: now tracked for confidence-adaptive wb
- **8-phase GPU experiment script** (`scripts/run_iteration23_experiments.py`)
- **955 tests** (25 new, all passing)

### 100-Sentence Verified Results (iteration 18, with tuned p_threshold + CI)

| Direction | bd | wb | p | BLEU | COMET | 95% CI | YAAL | % offline |
|-----------|---:|---:|:-:|-----:|------:|--------|-----:|:---------:|
| **EN-ZH** | 3 | 4 | 0.85 | 40.0 | **0.894** | [0.887, 0.901] | 6.09 | 99.8% |
| **EN-DE** | 2 | 3 | 0.75 | 27.9 | **0.881** | [0.873, 0.890] | 5.45 | 99.7% |
| **EN-IT** | 2 | 3 | 0.9 | 24.3 | **0.891** | [0.882, 0.899] | 6.76 | **100.2%** |
| **CS-EN** | 3 | 3 | 0.9 | 28.4 | **0.879** | [0.871, 0.886] | 5.81 | 99.8% |

## What to do next

### Priority 1: RUN GPU EXPERIMENTS (URGENT -- competition in ~7 days)
- **Run iteration 23 script on A40**:
  ```bash
  # Sync code to A40 first:
  rsync -avz . fuxa@quest.ms.mff.cuni.cz:nllw_deploy/ -e "ssh -p 3622"
  # Then on A40:
  python scripts/run_iteration23_experiments.py --model /home/fuxa/HY-MT1.5-7B.Q8_0.gguf
  # Quick mode for validation:
  python scripts/run_iteration23_experiments.py --model /home/fuxa/HY-MT1.5-7B.Q8_0.gguf --quick
  # Specific phase:
  python scripts/run_iteration23_experiments.py --model /home/fuxa/HY-MT1.5-7B.Q8_0.gguf --phase 3
  ```
- Key experiments that NEED results:
  - **YAAL fix validation**: batch_first_emission_time should lower YAAL
  - **Confidence-adaptive wb**: does logprob-based wb adjustment help YAAL?
  - **Language-pair gen cap**: does tighter gen cap improve quality?
  - **Source-aware batching**: quality improvement from better translation units
  - **Perplexity adaptive bd**: YAAL improvement vs quality cost
  - **Combined features**: test all winning features together

### Priority 2: Competition Decisions (based on GPU results)
- **Enable confidence-adaptive wb?** If YAAL improves without quality loss -> yes
- **Enable language-pair gen cap?** If COMET improves -> yes
- **Enable perplexity adaptive bd?** If YAAL improves without quality loss -> yes
- **Enable source-aware batching?** If COMET improves -> yes
- **Update IWSLT configs** with any winning features

### Priority 3: ASR Integration + Docker E2E
- `_run_asr()` in simulstream.py is still a stub
- Need Qwen3-ASR or external ASR integration
- Docker build + test on linux/amd64
- Full pipeline: audio chunks -> ASR -> NLLWSpeechProcessor -> OmniSTEval

### Dead Ends Confirmed (20+)
See CLAUDE.md for full list. Key ones: context injection, entropy veto, softmax_mean, signal fusion cascade, repetition halt (EN-ZH), top_p_weighted, Qwen3.5-9B.

### Sync Workflow
IMPORTANT: When syncing code to A40, do NOT use `rsync --delete` which destroys GPU-generated configs.
Use: `rsync -avz` (without --delete) to preserve files.

## Key Context

- **IWSLT 2026**: Eval April 1-15, ~7 days away
- **Metrics**: LongYAAL (primary latency), **COMET wmt22-comet-da** (primary quality)
- **Best known**: EN-ZH COMET=0.894, EN-DE 0.881, EN-IT 0.891 (>offline!), CS-EN 0.879
- **All directions at 99.7-100.2% of offline quality**
- **Model path on A40**: `/home/fuxa/HY-MT1.5-7B.Q8_0.gguf`
- **Competition validator**: `python scripts/validate_competition.py` (101+ checks pass)
- **New in iter 23**: confidence-adaptive wb, language-pair gen cap, avg_logprob in LA backend, 955 tests
- **OmniSTEval format**: ONE JSONL line per recording with per-word delays in ms
- **Gold transcripts**: `iwslt26-sst/inputs/en/acl6060.ts/gold-jsonl/` (1863 words per recording)
