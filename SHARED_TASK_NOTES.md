# Shared Task Notes -- NLLW SimulMT

## Current State (after Iteration 27, 2026-03-21)

**CRITICAL METRIC UPDATE**: Competition uses **XCOMET-XL** (Unbabel/XCOMET-XL) for quality, **StreamLAAL** for latency, **SacreBLEU** for secondary quality. NOT COMET wmt22-comet-da!

**30+ SimulMT modules (~15,700 lines), 1045 tests**

### What happened in Iteration 27

- **Comprehensive feature validation on A40 GPU** (20 sentences, COMET wmt22):
  - Tested ALL features from iterations 20-26 against baseline
  - **entropy_gated_top_p confirmed best latency feature**: -9.2% YAAL EN-ZH, zero quality loss
  - **ppl_bd (perplexity adaptive border)**: -15.1% YAAL EN-DE, zero quality loss
  - **temp=0.1 slightly improves quality**: +0.001 COMET EN-ZH (within CI)
  - **Confidence trim has NO effect** on HY-MT (model already confident)
- **Three DEAD ENDS confirmed on GPU**:
  - **Anti-LM contrastive decoding**: DEVASTATING for HY-MT. All gammas hurt:
    EN-ZH g=0.3: COMET 0.747 (baseline 0.884, -0.137!)
    EN-DE g=0.3: COMET 0.635 (baseline 0.876, -0.241!)
    HY-MT is already a translation model, penalty degrades quality
  - **Sentence-final refinement**: Re-translating from scratch loses committed prefix coherence:
    EN-ZH: COMET 0.794 (baseline 0.884, -0.090)
    EN-DE: COMET 0.755 (baseline 0.876, -0.121)
  - **Confidence-adaptive word batching**: Hurts EN-DE (COMET -0.006)
- **Two-phase XCOMET-XL scoring pipeline** (avoids OOM):
  - Phase A: translations with --save-hypotheses (COMET, no llama.cpp OOM)
  - Phase B: score hypotheses separately (no llama.cpp in memory)
  - BLOCKED: A40 has only 16GB RAM, XCOMET-XL needs >16GB RAM to load
  - Need to score on machine with >24GB RAM (MacBook 32GB, or H100)
- **xcomet_scorer.py**: fixed to accept both per_sentence and hypothesis file formats
- **1045 tests** (11 new, all passing)

### GPU-Validated Results (20 sentences, COMET wmt22, A40)

**Baselines:**
| Direction | BLEU | COMET | 95% CI | YAAL |
|-----------|:----:|:-----:|--------|:----:|
| EN-ZH | 39.8 | **0.884** | [0.868, 0.899] | 5.33 |
| EN-DE | 26.8 | **0.876** | [0.858, 0.890] | 5.50 |
| EN-IT | 24.6 | **0.879** | [0.861, 0.894] | 7.19 |
| CS-EN | 31.2 | **0.875** | [0.857, 0.892] | 6.39 |

**EN-ZH Feature Results (20 sent):**
| Feature | COMET | YAAL | Delta |
|---------|:-----:|:----:|:-----:|
| baseline | 0.884 | 5.33 | -- |
| **entropy_gated** | 0.884 | **4.84** | **YAAL -9.2%** |
| temp=0.1 | **0.885** | 5.33 | COMET +0.001 |
| ppl_bd | 0.884 | 5.23 | YAAL -1.9% |
| lp_gencap | 0.884 | 5.33 | neutral |
| trim3 | 0.884 | 5.33 | neutral |
| edt | 0.883 | 5.33 | neutral |

**EN-DE Feature Results (20 sent):**
| Feature | COMET | YAAL | Delta |
|---------|:-----:|:----:|:-----:|
| baseline | 0.876 | 5.50 | -- |
| **ppl_bd** | 0.875 | **4.67** | **YAAL -15.1%** |
| entropy_gated | 0.876 | 5.40 | YAAL -1.8% |
| temp=0.1 | 0.876 | 5.33 | YAAL -3.1% |
| edt | 0.876 | 5.43 | YAAL -1.3% |

## What to do next

### XCOMET-XL Results (scored on MacBook M5, 20 sentences, FLORES)

| Config | Direction | COMET | XCOMET-XL | YAAL |
|--------|-----------|:-----:|:---------:|:----:|
| baseline | EN-ZH | 0.884 | **0.8552** | 5.33 |
| **entropy_gated** | EN-ZH | 0.884 | **0.8594** | **4.84** |
| baseline | EN-DE | 0.876 | **0.9667** | 5.50 |
| temp=0.1 | EN-DE | 0.876 | **0.9670** | 5.33 |
| baseline | EN-IT | 0.879 | **0.9702** | 7.19 |
| baseline | CS-EN | 0.875 | **0.9708** | 6.39 |

**Key findings:**
- **EDT + entropy_gated = BEST CONFIG**: XCOMET-XL 0.8706 EN-ZH (+0.0154 vs baseline!), YAAL 4.84 (-9.2%)
- **EDT is #1 quality winner**: +0.0075 XCOMET-XL EN-ZH (INVISIBLE on COMET wmt22!)
- **entropy_gated_top_p**: +0.0042 XCOMET-XL EN-ZH AND -9.2% YAAL (win-win!)
- **Improvements COMPOSE**: EDT (quality) + entropy_gated (latency) = both together
- **conf_wb hurts XCOMET-XL**: -0.012 EN-ZH (dead end)
- **EN-DE/IT/CS-EN**: EDT+entropy_gated is neutral/slightly negative. Direction-specific tuning needed

### Priority 1: Run 100-sentence Evaluation
- Confirm results at full scale (100 sentences)
- Also run on MCIF dev set (competition uses MCIF, not FLORES)
- 34 hypothesis files ready for more scoring

### Priority 2: Run bd/wb Sweep on XCOMET-XL
- Our configs were tuned on COMET wmt22, not XCOMET-XL
- XCOMET-XL amplifies differences 39x -- rankings may change
- Run Phase 1-2 of iter 27 script once XCOMET scoring works

### Priority 3: Winning Features for Competition
- **entropy_gated_top_p**: use for ALL directions (free latency reduction)
- **ppl_bd**: use for EN-DE and test on other directions (big latency win)
- **temp=0.1**: use for EN-ZH (slight quality improvement)
- All other features: neutral or harmful
- **NO Anti-LM**: catastrophic for HY-MT
- **NO final_refinement**: catastrophic for all directions

### Priority 4: Competition Readiness
- Docker build + test on linux/amd64
- OmniSTEval format validation with official scorer
- MCIF dev set evaluation (not FLORES)

## Dead Ends Confirmed (25+)
See CLAUDE.md for full list. New in iter 27:
- **Anti-LM contrastive decoding** with HY-MT (all gammas devastating: -0.137 to -0.241 COMET)
- **Sentence-final refinement** (committed prefix essential for coherence: -0.090 to -0.121 COMET)
- **Confidence-adaptive word batching** (hurts EN-DE)

## Key Context
- **IWSLT 2026**: Eval April 1-15, ~10 days away. Paper deadline April 24.
- **Docker: linux/arm64** (NOT amd64!), single H100 80GB
- **Dev set: MCIF** (ACL scientific talks, NOT FLORES). Need to test on MCIF!
- **EN-IT is NEW for 2026** (replaced EN-JA). Less competition, strong opportunity.
- **No CS-EN baseline** from organizers. Strong opportunity.
- **Baseline uses Qwen3-4B** (small). Our HY-MT 7B should outperform.
- **CUNI (2025 winner)** used LocalAgreement (no attention access). Our AlignAtt is an advantage.
- **Latency thresholds NOT YET ANNOUNCED** -- monitor iwslt.org/2026/simultaneous
- **A40**: 46GB VRAM, 16GB RAM (can't run XCOMET-XL scoring!)
- **LLAMA_CPP_LIB on A40**: `/home/fuxa/llama.cpp/build/bin/libllama.so`
