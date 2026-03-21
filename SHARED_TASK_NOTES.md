# Shared Task Notes -- NLLW SimulMT

## Current State (after Iteration 26, 2026-03-21)

**CRITICAL METRIC UPDATE**: Competition uses **XCOMET-XL** (Unbabel/XCOMET-XL) for quality, **StreamLAAL** for latency, **SacreBLEU** for secondary quality. NOT COMET wmt22-comet-da!

**30+ SimulMT modules (~15,700 lines), 1034 tests**
**100-sentence results (COMET wmt22): 0.894 EN-ZH, 0.881 EN-DE, 0.891 EN-IT, 0.879 CS-EN -- MUST RE-EVAL WITH XCOMET-XL**

### What happened in Iteration 26
- **Anti-LM contrastive decoding** (Sia et al., NAACL 2024, arxiv 2311.08324):
  - Subtracts source-language continuation penalty from translation logits
  - Formula: `logits_adjusted = logits - gamma^step * anti_lm_log_probs`
  - Prevents hallucination and source-language copying
  - O(1) extra forward pass per translate() call (source-only, cached)
  - Uses separate seq_id (99) for anti-LM KV cache, cleaned up after
  - CLI: `--anti-lm --anti-lm-gamma 0.3`
  - Sweep: `antilm=0,1`, `almgamma=0.1,0.3,0.5`
  - Wired into AlignAtt backend generation loop (applied before EDT/temperature)
  - **Needs GPU validation**
- **Research findings**:
  - **Anti-LM** (NAACL 2024): +10 BLEU on weak models, hallucination prevention on strong models
  - **Beam search** (CUNI IWSLT 2025): +1 ChrF with beam_size=5, but HIGH technical risk for decoder-only LLMs (attention extraction with multiple beams untested). Deprioritized for competition.
  - **ContraDecode** (EACL 2024): source-contrastive + language-contrastive, 67-83% hallucination reduction. Our Anti-LM is the decoder-only equivalent.
- **6-phase competition experiment script** (`scripts/run_iteration26_experiments.py`):
  - Phase 0: XCOMET-XL baseline
  - Phase 1: Anti-LM gamma sweep + per-direction validation
  - Phase 2: Anti-LM + confidence trimming combined
  - Phase 3: Anti-LM + EDT combined
  - Phase 4: Best features combined (5 combinations x 4 directions)
  - Phase 5: Competition OmniSTEval output
- **1034 tests** (22 new, all passing)

### 100-Sentence Verified Results (iteration 18, COMET wmt22)

| Direction | bd | wb | p | BLEU | COMET | 95% CI | YAAL | % offline |
|-----------|---:|---:|:-:|-----:|------:|--------|-----:|:---------:|
| **EN-ZH** | 3 | 4 | 0.85 | 40.0 | **0.894** | [0.887, 0.901] | 6.09 | 99.8% |
| **EN-DE** | 2 | 3 | 0.75 | 27.9 | **0.881** | [0.873, 0.890] | 5.45 | 99.7% |
| **EN-IT** | 2 | 3 | 0.9 | 24.3 | **0.891** | [0.882, 0.899] | 6.76 | **100.2%** |
| **CS-EN** | 3 | 3 | 0.9 | 28.4 | **0.879** | [0.871, 0.886] | 5.81 | 99.8% |

## What to do next

### Priority 1: RUN GPU EXPERIMENTS (URGENT -- competition in ~7 days)
- **Run iteration 26 script on A40**:
  ```bash
  # Sync code to A40 first:
  rsync -avz . fuxa@quest.ms.mff.cuni.cz:nllw_deploy/ -e "ssh -p 3622"
  # Then on A40:
  python scripts/run_iteration26_experiments.py --model /home/fuxa/HY-MT1.5-7B.Q8_0.gguf
  # Anti-LM only:
  python scripts/run_iteration26_experiments.py --model /home/fuxa/HY-MT1.5-7B.Q8_0.gguf --phase 1
  # Quick validation:
  python scripts/run_iteration26_experiments.py --model /home/fuxa/HY-MT1.5-7B.Q8_0.gguf --quick
  ```
- **Also run iter 25 experiments if not done**:
  ```bash
  python scripts/run_iteration25_experiments.py --model /home/fuxa/HY-MT1.5-7B.Q8_0.gguf --phase 0
  ```
- Key experiments that NEED results:
  - **XCOMET-XL baseline** (Phase 0): Know where we actually stand
  - **Anti-LM gamma sweep** (Phase 1): Does contrastive decoding help XCOMET-XL?
  - **Anti-LM + confidence trim** (Phase 2): Combined hallucination prevention
  - **Best combined features** (Phase 4): Final competition config

### Priority 2: Competition Decisions (based on GPU results)
- **Enable Anti-LM?** If it improves XCOMET-XL -> yes (gamma=0.3 default)
- **Enable confidence trim?** If -3.0 improves XCOMET-XL -> yes
- **Enable entropy-gated top_p?** If YAAL improves without quality loss -> yes
- **Which prompt format?** hymt vs hymt-official
- **Update IWSLT configs** with winning features

### Priority 3: Competition Readiness
- Docker build + test on linux/amd64
- OmniSTEval format validation with official scorer
- ASR integration (if time permits)

### Dead Ends Confirmed (20+)
See CLAUDE.md for full list. Key ones: context injection, entropy veto, softmax_mean, signal fusion cascade, repetition halt (EN-ZH), top_p_weighted, Qwen3.5-9B, beam search (too risky for decoder-only LLMs).

### Sync Workflow
IMPORTANT: When syncing code to A40, do NOT use `rsync --delete` which destroys GPU-generated configs.
Use: `rsync -avz` (without --delete) to preserve files.

## Key Context

- **IWSLT 2026**: Eval April 1-15, ~7 days away
- **COMPETITION METRICS** (confirmed from baselines repo):
  - **Quality**: XCOMET-XL (Unbabel/XCOMET-XL) -- primary
  - **Quality**: SacreBLEU -- secondary
  - **Latency**: StreamLAAL -- primary latency
  - **NOT**: COMET wmt22-comet-da (our previous assumption was WRONG)
- **XCOMET-XL amplifies differences 39x vs wmt22** -- rankings may change significantly!
- **Model path on A40**: `/home/fuxa/HY-MT1.5-7B.Q8_0.gguf`
- **New in iter 26**: Anti-LM contrastive decoding, 1034 tests

## Baseline Scores (from IWSLT 2026 baselines repo)

| Direction | Segment | Context | XCOMET-XL | LongYAAL (CU) |
|-----------|---------|---------|:---------:|:-------------:|
| EN-DE     | 960ms   | yes     | ~81       | ~2250ms       |
| EN-IT     | 960ms   | yes     | ~75       | ~2750ms       |
| EN-ZH     | 960ms   | yes     | ~79       | ~2600ms       |
| CS-EN     | -       | -       | N/A       | N/A           |

## CRITICAL Competition Details

- **Docker target**: Single NVIDIA H100 80GB, linux/arm64
- **Dev set**: MCIF (not FLORES!) for en-{zh,de,it}. Test: ACL talks + accent challenge
- **char_level for zh/ja/ko**, word_level for de/it/en
- **No cs-en baseline** -- opportunity to be strong
- **Context helps baselines** +2-4 XCOMET-XL (but context KILLS HY-MT, so skip)
- **LongYAAL (CU) is PRIMARY latency** (not StreamLAAL as we assumed)
- **Latency thresholds**: NOT YET ANNOUNCED. Will be per-direction
- **Ranking**: Systems classified into LOW/HIGH latency regime, then ranked by XCOMET-XL WITHIN regime
