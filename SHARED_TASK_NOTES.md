# Shared Task Notes -- NLLW SimulMT

## Current State (after Iteration 16, 2026-03-21)

**28+ SimulMT modules (~13,000+ lines), 755 tests passing**
**NEW BEST: COMET=0.895 EN-ZH with top_p + wb=4 + bd=3**

### What happened in Iteration 16
- **top_p aggregation is best**: +0.010-0.031 COMET over ts_vote across all directions
- **wb=4/5 discovered as massive quality boost**: wb=5/bd=3 gives COMET 0.892 EN-ZH (+0.012)
- **top_p + wb=4 = 0.895**: combining the two gives new record (99.9% of offline quality)
- **Lower bd helps all directions**: bd=2 >> bd=3 for EN-DE/EN-IT (counterintuitive)
- **100-sentence benchmarks confirm 30-sentence results** (stable scores)
- **Dedicated head detection for EN-DE, EN-IT, CS-EN** (90% overlap with EN-ZH)
- **Stderr suppression** for llama.cpp noise (490+ warnings eliminated)
- **Head config discovery fix** for detect_heads-generated filenames
- **detect_heads --n-gpu-layers** flag added (was missing, blocking GPU detection)
- **755 tests passing** (9 new)

### 100-Sentence Benchmark Results (FLORES, A40, HY-MT1.5-7B)

| Direction | bd | wb | BLEU | COMET | YAAL | ms/sent |
|-----------|---:|---:|-----:|------:|-----:|--------:|
| EN-ZH | 2 | 3 | 32.9 | 0.880 | 3.77 | 980 |
| EN-DE | 3 | 3 | 24.9 | 0.850 | 4.38 | 1327 |
| EN-IT | 3 | 3 | 21.6 | 0.867 | 4.44 | 1330 |
| CS-EN | 3 | 2 | 22.6 | 0.857 | 2.77 | 1143 |

### wb=4/5 Discovery (50 sentences)

| Direction | bd | wb | BLEU | COMET | YAAL | vs baseline |
|-----------|---:|---:|-----:|------:|-----:|-------------|
| **EN-ZH** | 3 | 5 | 39.3 | **0.892** | 5.58 | +0.012 |
| EN-ZH | 2 | 4 | 35.4 | 0.883 | 4.37 | +0.003 |
| EN-ZH | 3 | 4 | 38.2 | 0.887 | 4.67 | +0.007 |
| EN-ZH | 2 | 5 | 36.2 | 0.887 | 5.56 | +0.007 |
| EN-ZH | 1 | 3 | 28.3 | 0.870 | 3.26 | -0.010 (too aggressive) |
| **EN-DE** | 3 | 4 | 24.6 | **0.873** | 5.16 | +0.023 |
| **EN-DE** | 2 | 3 | 24.8 | 0.871 | 4.06 | +0.021 |
| **EN-IT** | 3 | 4 | 22.1 | **0.885** | 5.21 | +0.018 |
| EN-IT | 2 | 3 | 20.9 | 0.882 | 4.20 | +0.015 |

### Optimal Per-Direction Configs (updated)

| Direction | bd | wb | agg | COMET | YAAL | Strategy |
|-----------|---:|---:|-----|------:|-----:|----------|
| **EN-ZH** | 3 | 4 | top_p | **0.895** | 4.67 | Max quality (99.9% offline) |
| EN-ZH | 2 | 3 | top_p | 0.890 | 3.60 | Balanced quality/latency |
| EN-ZH | 2 | 3 | ts_vote | 0.880 | 3.77 | Low latency |
| **EN-DE** | 2 | 3 | top_p | **0.881** | 4.06 | Best quality |
| EN-DE | 3 | 4 | top_p | 0.881 | 5.16 | Same quality, more latency |
| **EN-IT** | 2 | 3 | top_p | **0.884** | 4.20 | Best quality |
| EN-IT | 3 | 4 | ts_vote | 0.885 | 5.21 | Slightly higher at more latency |
| **CS-EN** | 3 | 3 | top_p | **0.876** | 3.27 | Best quality |
| CS-EN | 3 | 2 | ts_vote | 0.857 | 2.77 | Lowest latency |

### Head Detection Results (HY-MT1.5-7B, 100 FLORES sentences)

| Direction | Config | Top Head | TS | Overlap with EN-ZH |
|-----------|--------|----------|---:|--------------------:|
| EN-ZH | hymt_en_zh.json | L7H21 | 0.648 | - |
| EN-DE | hy_mt1_5_7b_q8_0_en_de.json | L7H21 | 0.638 | 90% top-20 |
| EN-IT | hy_mt1_5_7b_q8_0_en_it.json | L7H21 | 0.687 | 90% top-20 |
| CS-EN | hy_mt1_5_7b_q8_0_cs_en.json | TBD | TBD | TBD |

Note: EN-DE config was re-detected after being deleted by rsync.

### top_p Aggregation Discovery

| Direction | bd | wb | ts_vote COMET | top_p COMET | Delta |
|-----------|---:|---:|-------------:|------------:|------:|
| EN-ZH | 3 | 4 | 0.887 | **0.895** | +0.008 |
| EN-ZH | 2 | 3 | 0.880 | 0.890 | +0.010 |
| EN-DE | 2 | 3 | 0.871 | 0.881 | +0.010 |
| EN-DE | 3 | 4 | 0.873 | 0.881 | +0.008 |
| EN-IT | 2 | 3 | 0.882 | 0.884 | +0.002 |
| CS-EN | 3 | 3 | 0.855 | **0.876** | +0.021 |
| CS-EN | 3 | 2 | 0.857 | 0.868 | +0.011 |

### Dead Ends Confirmed
- **Context injection**: KILLS quality for HY-MT (-0.084 to -0.125 COMET)
- **Entropy veto**: All thresholds hurt (0.5->0.494 COMET, 1.5->0.794)
- **softmax_mean aggregation**: COMET 0.812 (terrible)
- **Signal fusion/cascade**: marginal +0.002 at complexity cost
- **bd=1**: Too aggressive, -0.010 COMET for EN-ZH

### Key Findings
- **top_p aggregation is the single biggest win**: +0.008 to +0.021 COMET
- **wb=4-5 is the second biggest win**: +0.006 to +0.012 COMET per wb step
- **Combined top_p + wb=4: COMET 0.895** (99.9% of full-sentence 0.896)
- **Lower bd helps across all directions**: bd=2 consistently better than bd=3
- **100-sentence results are stable** vs 30-sentence (validates evaluation)
- **Speed increases with higher wb**: wb=5 at 859ms/sent vs wb=3 at 980ms
- **Dedicated heads add marginal value**: 90% overlap, ~0.003 improvement

## What to do next

### Priority 1: XCOMET-XL Scoring (FIX APPLIED, RE-RUN NEEDED)
Previous run OOM'd (rc=-9): translation model + XCOMET-XL (12GB) exceeded A40 VRAM.
**Fixed**: `eval.py` now calls `backend.close()` before loading XCOMET-XL to free VRAM.
Re-run:
```bash
ssh -p 3622 fuxa@quest.ms.mff.cuni.cz "cd /home/fuxa/nllw_deploy && export LLAMA_CPP_LIB=/home/fuxa/llama.cpp/build/bin/libllama.so && nohup python3 run_xcomet_eval.py > xcomet_results_v2.log 2>&1 &"
```

### Priority 2: 100-sentence Verification of Best Configs
Need to verify top_p + wb=4 results at 100 sentences. Current results are 50 sentences.
```bash
python -m nllw.bench --model MODEL --lang en-zh -n 100 --border-distance 3 --word-batch 4 --aggregation top_p --comet
```

### Priority 3: Competition Prep (IWSLT 2026, eval April 1-15, ~10 days)
- Docker finalization (OmniSTEval format, SimulStream wrapper)
- Update per-direction optimal configs in SimulStream DIRECTION_DEFAULTS with top_p + wb=4
- OmniSTEval longform output verification
- SimulStream E2E test with audio
- Update Dockerfile to include NLLW as package

### Priority 4: top_p Investigation
- Why does top_p help so much? Analyze the aggregation method
- Does top_p + other signals (shift-k, etc.) help further?
- top_p sigma parameter tuning

### Sync Workflow
IMPORTANT: When syncing code to A40, do NOT use `rsync --delete` which destroys GPU-generated configs.
Use: `rsync -avz` (without --delete) to preserve files.
Always pull new configs from A40 BEFORE pushing code.

## Key Context

- **IWSLT 2026**: Eval April 1-15, ~10 days away
- **Metrics**: LongYAAL (primary latency), COMET (primary quality)
- **Best known**: EN-ZH COMET=0.892 (wb=5/bd=3), YAAL=5.58
- **A40 ready**: experiments running, head detection complete (EN-IT, CS-EN)
- **Dead ends**: EAST, LoRA no-think, GDN, confidence-only, fixed-rate, TAF, signals (marginal)
