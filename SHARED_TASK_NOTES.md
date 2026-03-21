# Shared Task Notes -- NLLW SimulMT

## Current State (after Iteration 24, 2026-03-21)

**CRITICAL METRIC UPDATE**: Competition uses **XCOMET-XL** (Unbabel/XCOMET-XL) for quality, **StreamLAAL** for latency, **SacreBLEU** for secondary quality. NOT COMET wmt22-comet-da! (Confirmed from github.com/owaski/iwslt-2026-baselines/eval.sh)

**30+ SimulMT modules (~15,700 lines), 974 tests**
**100-sentence results (COMET wmt22): 0.894 EN-ZH, 0.881 EN-DE, 0.891 EN-IT, 0.879 CS-EN -- MUST RE-EVAL WITH XCOMET-XL**

### What happened in Iteration 24
- **Entropy-gated top_p** (NOVEL):
  - Per-token top_p threshold modulation from merged attention entropy during generation
  - Focused attention (low entropy) -> lower threshold -> emit sooner (lower YAAL)
  - Spread attention (high entropy) -> higher threshold -> wait longer (better quality)
  - Different from adaptive_top_p (per-sentence) and entropy_veto (dead end)
  - `merged_attention_entropy()`: entropy of TS-weighted merged attention distribution
  - `entropy_gated_top_p_threshold()`: maps entropy to scale factor [0.88, 1.08] with linear interpolation
  - Wired into both AlignAtt and AlignAtt-LA backends
  - CLI: `--entropy-gated-top-p`, sweep: `entgtp=0,1`
  - Only active when aggregation is "top_p" or "top_p_weighted"
  - **Needs GPU validation**
- **9-phase GPU experiment script** (`scripts/run_iteration24_experiments.py`)
- **974 tests** (19 new, all passing)

### 100-Sentence Verified Results (iteration 18, with tuned p_threshold + CI)

| Direction | bd | wb | p | BLEU | COMET | 95% CI | YAAL | % offline |
|-----------|---:|---:|:-:|-----:|------:|--------|-----:|:---------:|
| **EN-ZH** | 3 | 4 | 0.85 | 40.0 | **0.894** | [0.887, 0.901] | 6.09 | 99.8% |
| **EN-DE** | 2 | 3 | 0.75 | 27.9 | **0.881** | [0.873, 0.890] | 5.45 | 99.7% |
| **EN-IT** | 2 | 3 | 0.9 | 24.3 | **0.891** | [0.882, 0.899] | 6.76 | **100.2%** |
| **CS-EN** | 3 | 3 | 0.9 | 28.4 | **0.879** | [0.871, 0.886] | 5.81 | 99.8% |

## What to do next

### Priority 1: RUN GPU EXPERIMENTS (URGENT -- competition in ~7 days)
- **Run iteration 24 script on A40**:
  ```bash
  # Sync code to A40 first:
  rsync -avz . fuxa@quest.ms.mff.cuni.cz:nllw_deploy/ -e "ssh -p 3622"
  # Then on A40:
  python scripts/run_iteration24_experiments.py --model /home/fuxa/HY-MT1.5-7B.Q8_0.gguf
  # Quick mode for validation:
  python scripts/run_iteration24_experiments.py --model /home/fuxa/HY-MT1.5-7B.Q8_0.gguf --quick
  # Specific phase:
  python scripts/run_iteration24_experiments.py --model /home/fuxa/HY-MT1.5-7B.Q8_0.gguf --phase 2
  ```
- Key experiments that NEED results:
  - **Entropy-gated top_p**: does per-token threshold modulation help YAAL without hurting COMET?
  - **Confidence-adaptive wb** (iter 23): does logprob-based wb adjustment help YAAL?
  - **Language-pair gen cap** (iter 23): does tighter gen cap improve quality?
  - **Source-aware batching** (iter 21): quality improvement from better translation units?
  - **Perplexity adaptive bd** (iter 20): YAAL improvement vs quality cost?
  - **Combined features**: test all winning features together

### Priority 2: Competition Decisions (based on GPU results)
- **Enable entropy-gated top_p?** If YAAL improves without quality loss -> yes
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
- **COMPETITION METRICS** (confirmed from baselines repo):
  - **Quality**: XCOMET-XL (Unbabel/XCOMET-XL) -- primary
  - **Quality**: SacreBLEU -- secondary
  - **Latency**: StreamLAAL -- primary latency
  - **Eval tool**: `omnisteval longform` command
  - **NOT**: COMET wmt22-comet-da (our previous assumption was WRONG)
- **XCOMET-XL amplifies differences 39x vs wmt22** -- rankings may change significantly!
- **Best known (wmt22 metric, needs XCOMET-XL re-eval)**: EN-ZH 0.894, EN-DE 0.881, EN-IT 0.891, CS-EN 0.879
- **Model path on A40**: `/home/fuxa/HY-MT1.5-7B.Q8_0.gguf`
- **Competition validator**: `python scripts/validate_competition.py` (112 checks pass)
- **New in iter 24**: entropy-gated top_p, metric correction, 974 tests
- **OmniSTEval format**: ONE JSONL line per recording with per-word delays in ms

## URGENT: GPU Tasks for Competition

1. **Re-evaluate ALL directions with XCOMET-XL**:
   ```bash
   python -m nllw.bench --lang en-zh --xcomet --save  # all 4 directions
   ```
2. **Run iteration 24 experiments** (entropy-gated top_p etc)
3. **Generate competition-format OmniSTEval output** and validate with `omnisteval longform`
4. **Check StreamLAAL values** (we track this already but need to verify alignment with official scorer)
