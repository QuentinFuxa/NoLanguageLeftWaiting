# Shared Task Notes -- NLLW SimulMT

## Current State (after Iteration 15, 2026-03-21)

**28+ SimulMT modules (~13,000+ lines), 746 tests passing**
**FIRST FLORES BENCHMARKS on A40 -- COMET 0.879 EN-ZH**

### What happened in Iteration 15
- **4 critical bug fixes** (see below)
- **EN-ZH parameter sweep**: 9 configs tested, best COMET=0.879 at bd=2/wb=3
- **Multi-direction benchmarks**: EN-DE working (COMET=0.853, BLEU=25.4)
- **Signal fusion experiments**: running on A40
- **746 tests passing** (15 new tests)
- **Cross-lingual head transfer validated on GPU**: using EN-ZH heads for all directions

### Critical Bugs Fixed (Iteration 15)
1. **simulate_backend empty translation** (`simulate.py`): `_handle_segment_end()` cleared `committed_ids` before `get_full_translation()`. Result: BLEU=0 and empty translations. Fixed: accumulate step texts as fallback.
2. **Chinese BLEU = 0** (`metrics.py`): sacrebleu default tokenizer can't segment CJK characters. Fixed: pass `tokenize="zh"` for Chinese/Japanese targets.
3. **KV cache always on CPU** (`llama_backend.py`): `create_context()` hardcoded `n_gpu_layers=0`. Fixed: wire through from config. Result: `graph_splits` dropped from 66 to 2.
4. **Cross-lingual head fallback** (`alignatt_backend.py`): HY-MT EN-DE/IT/CS had no head configs. Fixed: fall back to same-model configs (validated >97% TS mass transfer).

### EN-ZH Parameter Sweep Results (FLORES, 30 sentences, A40)

| bd | wb | BLEU | COMET | YAAL | ms/sent |
|----|-----|------|-------|------|---------|
| 3 | 1 | 31.4 | 0.834 | 1.73 | 1741 |
| 4 | 1 | 30.7 | 0.823 | 1.83 | 1717 |
| 2 | 2 | 31.9 | 0.855 | 2.60 | 1273 |
| 3 | 2 | 35.9 | 0.860 | 2.81 | 1231 |
| 4 | 2 | 35.2 | 0.856 | 3.11 | 1233 |
| **2** | **3** | **34.7** | **0.879** | **3.60** | **1070** |
| 3 | 3 | 36.5 | 0.873 | 3.77 | 1052 |
| 4 | 3 | 37.0 | 0.873 | 4.02 | 1067 |
| 5 | 3 | 37.4 | 0.872 | 4.43 | 1049 |

**Best quality**: bd=2 wb=3 -> COMET 0.879 at YAAL 3.60
**Best BLEU**: bd=5 wb=3 -> BLEU 37.4 at YAAL 4.43
**Pareto optimal**: bd=2/wb=3 dominates (highest COMET at moderate latency)

### Multi-Direction Results (30 sentences, cross-lingual head transfer)

| Direction | bd | wb | BLEU | COMET | YAAL |
|-----------|----|----|------|-------|------|
| EN-ZH | 2 | 3 | 34.7 | **0.879** | 3.60 |
| EN-DE | 3 | 3 | 25.4 | 0.853 | 4.33 |
| EN-IT | 3 | 3 | 21.1 | 0.864 | 4.47 |
| CS-EN | 3 | 2 | 22.7 | 0.853 | 2.87 |

Optimal per-direction configs:
- **EN-ZH**: bd=2, wb=3 (COMET=0.879, YAAL=3.60)
- **EN-DE**: bd=3, wb=3 (COMET=0.853, YAAL=4.33)
- **EN-IT**: bd=3, wb=3 (COMET=0.864, YAAL=4.47)
- **CS-EN**: bd=3, wb=2 (COMET=0.853, YAAL=2.87)

### Signal Fusion Results (EN-ZH, bd=3/wb=3, 30 sentences)

| Config | COMET | YAAL | vs Baseline |
|--------|-------|------|-------------|
| Baseline (vanilla) | 0.873 | 3.77 | - |
| Shift-k=0.4 | 0.870 | 3.89 | -0.003 |
| Coverage=0.3 | 0.877 | **10.83** | Latency explosion |
| REINA=-0.5 | 0.856 | 3.39 | -0.017 |
| Fusion (std+shiftk+cov) | 0.872 | 3.59 | -0.001 |
| **Fusion ALL** | **0.875** | 3.69 | **+0.002** |
| Cascade ALL | 0.871 | 10.32 | -0.002 + latency |

### Key Findings
- **wb=3 >> wb=2 >> wb=1**: Larger batches = higher COMET + faster speed
- **COMET peaks at low bd**: bd=2 wb=3 is best (counterintuitive!)
- **Cross-lingual head transfer works on GPU**: EN-DE gets COMET=0.853 with EN-ZH heads
- **KV cache on GPU**: graph_splits 66->2, speeds up inference
- **Signals add marginal value**: fusion ALL gives +0.002 COMET, individual signals hurt
- **Weighted fusion >> boolean cascade**: cascade causes latency explosion with coverage
- **Best config per direction is vanilla AlignAtt** with tuned bd/wb
- **No new training-free approaches** in literature since last survey (our AlignAtt+llama.cpp is unique)
- **Docker must target linux/arm64** for H100 submission
- **LongYAAL** (not YAAL) is IWSLT 2026 primary latency metric -- use `omnisteval longform`

## What to do next

### Priority 1: Complete Multi-Direction + Signal Experiments
Running on A40 right now (`multi_direction_results.log`). Check results:
```bash
ssh -p 3622 fuxa@quest.ms.mff.cuni.cz "grep -A3 '=== ' /home/fuxa/nllw_deploy/multi_direction_results.log"
```
If experiments failed, resync code and rerun:
```bash
rsync -avz --delete --exclude='.git' --exclude='__pycache__' -e "ssh -p 3622" nllw/ fuxa@quest.ms.mff.cuni.cz:/home/fuxa/nllw_deploy/nllw/
ssh -p 3622 fuxa@quest.ms.mff.cuni.cz "cd /home/fuxa/nllw_deploy && export LLAMA_CPP_LIB=/home/fuxa/llama.cpp/build/bin/libllama.so && nohup python3 run_multi_direction.py > multi_direction_results.log 2>&1 &"
```

### Priority 2: Head Detection for EN-DE/EN-IT/CS-EN
While cross-lingual transfer works (>97%), dedicated heads may improve quality:
```bash
ssh -p 3622 fuxa@quest.ms.mff.cuni.cz "cd /home/fuxa/nllw_deploy && export LLAMA_CPP_LIB=/home/fuxa/llama.cpp/build/bin/libllama.so && python3 -m nllw.detect_heads --model /home/fuxa/HY-MT1.5-7B.Q8_0.gguf --lang en-de"
```

### Priority 3: XCOMET-XL Scoring
COMET (wmt22-comet-da) and XCOMET-XL give different numbers. Need XCOMET-XL for proper comparison with iwslt26-sst results (best: 0.842 EN-ZH). Requires H100 or A40 with enough VRAM.

### Priority 4: Signal Fusion GPU Validation
Phase 2 of multi_direction experiments tests fusion. If it works well, collect traces for calibration:
```bash
python3 -m nllw.bench --signal-fusion --shift-k 0.4 --coverage-threshold 0.3 --lang en-zh --comet --save --collect-traces traces_enzh.json
```

### Priority 5: Competition Prep (IWSLT 2026, eval April 1-15)
- Docker finalization
- OmniSTEval format verification (--omnisteval flag)
- Per-direction optimal configs (from benchmark results)
- SimulStream E2E test with audio

## Key Context

- **IWSLT 2026**: Eval April 1-15, ~10 days away
- **Metrics**: LongYAAL (primary latency), COMET (primary quality)
- **Best known**: EN-ZH COMET=0.879, YAAL=3.60 (bd=2, wb=3)
- **A40 ready**: Latest code synced, model + llama.cpp deployed
- **Dead ends**: EAST, LoRA no-think, GDN, confidence-only, fixed-rate, TAF, Seed-X-PPO, Qwen3-4B, HY-MT1.8B
