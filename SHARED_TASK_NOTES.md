# Shared Task Notes -- NLLW SimulMT

## Current State (after Iteration 25, 2026-03-21)

**CRITICAL METRIC UPDATE**: Competition uses **XCOMET-XL** (Unbabel/XCOMET-XL) for quality, **StreamLAAL** for latency, **SacreBLEU** for secondary quality. NOT COMET wmt22-comet-da! (Confirmed from github.com/owaski/iwslt-2026-baselines/eval.sh)

**30+ SimulMT modules (~15,500 lines), 1003 tests**
**100-sentence results (COMET wmt22): 0.894 EN-ZH, 0.881 EN-DE, 0.891 EN-IT, 0.879 CS-EN -- MUST RE-EVAL WITH XCOMET-XL**

### What happened in Iteration 25
- **Generation temperature** (NOVEL):
  - Low-temperature sampling (0.05-0.3) to explore alternatives near greedy path
  - Can help escape suboptimal greedy decisions
  - `sample_with_temperature()`: temperature-scaled softmax sampling
  - CLI: `--generation-temperature 0.1`, sweep: `temp=0.0,0.05,0.1,0.2,0.3`
  - **Needs GPU validation**
- **Confidence-gated token trimming** (NOVEL):
  - After generation, trim trailing tokens with logprob below threshold
  - Prevents committing hallucinated trailing tokens that hurt XCOMET-XL
  - `trim_low_confidence_tokens()`: scan from end, keep tokens >= threshold
  - CLI: `--confidence-trim -3.0`, sweep: `conftrim=-2.0,-3.0,-4.0,-5.0`
  - **Needs GPU validation**
- **Per-token logprob tracking**: shared computation for both features
- **Entropy-based dynamic temperature (EDT)** (arxiv 2403.14541):
  - Per-token adaptive temperature: confident tokens -> near-greedy, uncertain -> explore
  - More principled than fixed temperature
  - CLI: `--entropy-dynamic-temperature`, sweep: `edt=0,1`
  - **Needs GPU validation**
- **Research findings**:
  - **Anti-LM contrastive decoding** (ACL 2024): promising for hallucination prevention
  - **CUNI beam search + AlignAtt**: competition-validated +1 BLEU (medium effort to implement)
  - **QE-Fusion**: multi-candidate fusion (too slow for streaming)
- **9-phase competition experiment script** (`scripts/run_iteration25_experiments.py`)
- **1012 tests** (38 new, all passing)
- Competition validator: all checks passing

### 100-Sentence Verified Results (iteration 18, COMET wmt22)

| Direction | bd | wb | p | BLEU | COMET | 95% CI | YAAL | % offline |
|-----------|---:|---:|:-:|-----:|------:|--------|-----:|:---------:|
| **EN-ZH** | 3 | 4 | 0.85 | 40.0 | **0.894** | [0.887, 0.901] | 6.09 | 99.8% |
| **EN-DE** | 2 | 3 | 0.75 | 27.9 | **0.881** | [0.873, 0.890] | 5.45 | 99.7% |
| **EN-IT** | 2 | 3 | 0.9 | 24.3 | **0.891** | [0.882, 0.899] | 6.76 | **100.2%** |
| **CS-EN** | 3 | 3 | 0.9 | 28.4 | **0.879** | [0.871, 0.886] | 5.81 | 99.8% |

## What to do next

### Priority 1: RUN GPU EXPERIMENTS (URGENT -- competition in ~7 days)
- **Run iteration 25 script on A40**:
  ```bash
  # Sync code to A40 first:
  rsync -avz . fuxa@quest.ms.mff.cuni.cz:nllw_deploy/ -e "ssh -p 3622"
  # Then on A40:
  python scripts/run_iteration25_experiments.py --model /home/fuxa/HY-MT1.5-7B.Q8_0.gguf
  # XCOMET-XL baseline only (most urgent):
  python scripts/run_iteration25_experiments.py --model /home/fuxa/HY-MT1.5-7B.Q8_0.gguf --phase 0
  # Quick mode for validation:
  python scripts/run_iteration25_experiments.py --model /home/fuxa/HY-MT1.5-7B.Q8_0.gguf --quick
  ```
- Key experiments that NEED results:
  - **XCOMET-XL baseline** (Phase 0): Know where we actually stand with competition metric
  - **HY-MT prompt A/B test** (Phase 1): Official vs current prompt format
  - **Generation temperature** (Phase 2): Does low-temp sampling help quality?
  - **Confidence trimming** (Phase 3): Does trimming trailing tokens help XCOMET-XL?
  - **Entropy-gated top_p** (Phase 4): Per-token threshold modulation for YAAL
  - **All untested iter 20-24 features** (Phases 4-6)
  - **Combined features** (Phase 7): Test all winning features together

### Priority 2: Competition Decisions (based on GPU results)
- **Which prompt format?** If hymt-official is better -> switch all configs
- **Enable temperature?** If 0.1 improves XCOMET-XL -> yes
- **Enable confidence trim?** If -3.0 improves XCOMET-XL -> yes
- **Enable entropy-gated top_p?** If YAAL improves without quality loss -> yes
- **Enable confidence-adaptive wb?** If YAAL improves without quality loss -> yes
- **Enable language-pair gen cap?** If XCOMET-XL improves -> yes
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
- **Competition validator**: `python scripts/validate_competition.py` (all checks pass)
- **New in iter 25**: generation temperature, confidence trimming, 1003 tests
- **OmniSTEval format**: ONE JSONL line per recording with per-word delays in ms

## Baseline Scores (from IWSLT 2026 baselines repo)

**XCOMET-XL baseline scores** (Qwen3-4B-Instruct, approximate from tradeoff plot):
| Direction | Segment | Context | XCOMET-XL | LongYAAL (CU) |
|-----------|---------|---------|:---------:|:-------------:|
| EN-DE     | 960ms   | yes     | ~81       | ~2250ms       |
| EN-IT     | 960ms   | yes     | ~75       | ~2750ms       |
| EN-ZH     | 960ms   | yes     | ~79       | ~2600ms       |
| CS-EN     | -       | -       | N/A       | N/A           |

**CRITICAL RANKING INSIGHT**: Systems are classified into LOW or HIGH latency regime by LongYAAL (CU), then ranked by XCOMET-XL quality WITHIN that regime. No smooth tradeoff -- maximize quality while staying under latency budget.

**Latency thresholds**: NOT YET ANNOUNCED. Will be per-direction.

**Our target**: Beat baselines significantly. Our HY-MT1.5-7B at 99.8% of offline should score higher.

## CRITICAL Competition Details

- **Docker target**: Single NVIDIA H100 80GB, linux/arm64
- **Dev set**: MCIF (not FLORES!) for en-{zh,de,it}. Test: ACL talks + accent challenge
- **char_level for zh/ja/ko**, word_level for de/it/en
- **No cs-en baseline** -- opportunity to be strong
- **Context helps baselines** +2-4 XCOMET-XL (but context KILLS HY-MT, so skip)
- **OmniSTEval resegmentation**: SoftSegmenter (Needleman-Wunsch DP alignment)
- **LongYAAL (CU) is PRIMARY latency** (not StreamLAAL as we assumed)

## URGENT: GPU Tasks for Competition

1. **Re-evaluate ALL directions with XCOMET-XL** (Phase 0):
   ```bash
   python scripts/run_iteration25_experiments.py --model /home/fuxa/HY-MT1.5-7B.Q8_0.gguf --phase 0
   ```
2. **Run new feature experiments** (Phases 1-6):
   ```bash
   python scripts/run_iteration25_experiments.py --model /home/fuxa/HY-MT1.5-7B.Q8_0.gguf --phase 1,2,3,4,5,6
   ```
3. **Test combined features** (Phase 7)
4. **Generate competition OmniSTEval output** (Phase 8) and validate with `omnisteval longform`
5. **Check StreamLAAL values** align with official scorer
