# FLORES Benchmark Results -- HY-MT1.5-7B on A40

## Setup
- **Model**: HY-MT1.5-7B.Q8_0.gguf (~8.1 GB)
- **GPU**: NVIDIA A40 (46GB), all 33 layers offloaded
- **Backend**: AlignAtt with KV cache reuse
- **Corpus**: FLORES+ (30-50 sentences)
- **Metrics**: BLEU (sacrebleu, zh tokenizer), COMET (wmt22-comet-da)
- **Date**: 2026-03-21 (Iteration 15)

## EN-ZH Parameter Sweep (30 sentences)

| bd | wb | BLEU | COMET | AL | YAAL | AP | DAL | MaxCW | ms/sent |
|----|-----|------|-------|-----|------|-----|-----|-------|---------|
| 2 | 2 | 31.9 | 0.855 | 2.60 | 2.60 | 0.562 | 3.33 | 4.4 | 1273 |
| 3 | 2 | 35.9 | 0.860 | 2.81 | 2.81 | 0.575 | 3.49 | 4.4 | 1231 |
| 4 | 2 | 35.2 | 0.856 | 3.11 | 3.11 | 0.578 | 4.04 | 5.7 | 1233 |
| 2 | 3 | 34.7 | **0.879** | 3.61 | 3.60 | 0.604 | 3.92 | 3.7 | 1070 |
| 3 | 3 | 36.5 | 0.873 | 3.79 | 3.77 | 0.604 | 4.15 | 4.9 | 1052 |
| 4 | 3 | 37.0 | 0.873 | 4.04 | 4.02 | 0.616 | 4.73 | 6.1 | 1067 |
| 5 | 3 | 37.4 | 0.872 | 4.45 | 4.43 | 0.626 | 5.28 | 6.7 | 1049 |
| 3 | 1 | 31.4 | 0.834 | 1.74 | 1.73 | 0.539 | 2.78 | 3.9 | 1741 |
| 4 | 1 | 30.7 | 0.823 | 1.83 | 1.83 | 0.540 | 3.02 | 4.7 | 1717 |

## EN-ZH 50-sentence Baseline (bd=3, wb=3)
- **BLEU**: 0.0 (before zh tokenizer fix)
- **COMET**: 0.875
- **AL**: 4.10, **YAAL**: 4.09
- **AP**: 0.622, **DAL**: 4.68
- **MaxCW**: 5.3
- **Speed**: 1054 ms/sent

## Key Findings

### Quality (COMET)
1. **wb=3 >> wb=2**: Consistently higher COMET (+0.013 to +0.024)
2. **Diminishing returns after bd=2-3**: COMET plateaus around 0.87
3. **Best COMET**: bd=2, wb=3 -> 0.879 (lower latency, higher quality!)
4. **BLEU peaks at higher bd**: 37.4 at bd=5 wb=3 (BLEU and COMET disagree on best config)

### Latency (YAAL -- IWSLT primary metric)
1. **Fastest low-latency**: bd=2 wb=2 -> YAAL=2.60 (but COMET=0.855)
2. **Best quality-latency**: bd=2 wb=3 -> YAAL=3.60 with COMET=0.879
3. **Latency scales linearly with bd**: ~0.4 YAAL per bd increase

### Speed
- wb=3 configs are ~15% faster (1050-1070ms) vs wb=2 (1230ms) vs wb=1 (1720ms)
- Larger batches = fewer translate() calls = less overhead (model load amortized)
- wb=1 is 65% slower: many small batches = more KV cache operations

### Comparison with iwslt26-sst (Best known: XCOMET-XL 0.842)
- Direct comparison impossible (COMET != XCOMET-XL)
- But COMET=0.879 is very promising
- Need XCOMET-XL scoring to compare properly (requires H100 for 3.5B model)

## Multi-Direction Results (30 sentences, cross-lingual head transfer)

Using EN-ZH heads for all directions (validated >97% TS mass transfer).

| Direction | bd | wb | BLEU | COMET | YAAL | DAL | ms/sent |
|-----------|----|----|------|-------|------|-----|---------|
| EN-DE | 3 | 3 | 25.4 | 0.853 | 4.33 | 6.29 | 1437 |
| EN-DE | 4 | 3 | 25.2 | 0.853 | 4.96 | 7.11 | 1426 |
| EN-IT | 4 | 3 | 20.5 | 0.863 | 5.12 | 7.34 | 1431 |
| EN-IT | 3 | 3 | 21.1 | **0.864** | 4.47 | 6.50 | 1439 |
| CS-EN | 2 | 1 | 16.4 | 0.809 | 2.06 | 3.64 | 1739 |
| CS-EN | 3 | 2 | 22.7 | **0.853** | 2.87 | 4.73 | 1275 |

### Multi-Direction Key Findings
- **EN-DE**: bd=3 is Pareto-optimal (same COMET as bd=4, lower latency)
- **EN-IT**: bd=3 is better than bd=4 on both quality and latency
- **CS-EN**: bd=3 wb=2 >> bd=2 wb=1 (+0.044 COMET, +6.3 BLEU for +0.81 YAAL)
- All directions work with cross-lingual head transfer from EN-ZH heads

## Signal Fusion Experiments (30 sentences, EN-ZH, bd=3/wb=3)

| Config | BLEU | COMET | YAAL | vs Baseline |
|--------|------|-------|------|-------------|
| **Baseline** (no signals) | 36.5 | **0.873** | 3.77 | - |
| Shift-k=0.4 | 36.1 | 0.870 | 3.89 | -0.003 COMET |
| Coverage=0.3 | 39.3 | 0.877 | **10.83** | +0.004 COMET but latency explosion |
| REINA=-0.5 | 30.4 | 0.856 | 3.39 | -0.017 COMET |
| Fusion (std+shiftk+cov) | 35.3 | 0.872 | 3.59 | -0.001 COMET |
| **Fusion ALL signals** | **37.0** | **0.875** | **3.69** | **Best: +0.002 COMET** |
| Cascade ALL signals | 38.2 | 0.871 | 10.32 | Latency explosion from coverage |

### Signal Findings
1. **Individual signals don't improve COMET** over vanilla AlignAtt at bd=3/wb=3
2. **Coverage=0.3 causes latency explosion** (YAAL 10.83): too conservative, prevents most border stops
3. **REINA=-0.5 hurts quality** (-0.017 COMET): entropy change inhibits too many border stops
4. **Shift-k=0.4 is nearly neutral** (-0.003 COMET): border mass doesn't add much over argmax
5. **Fusion (std+shiftk+cov)** is nearly identical to baseline (-0.001 COMET)
6. **Fusion ALL is slightly better** (+0.002 COMET): weak signals combine when all 8 are enabled
7. **Cascade ALL explodes in latency** (YAAL 10.32): coverage guard in boolean cascade is too aggressive
8. **Weighted fusion >> boolean cascade**: fusion handles weak signals gracefully, cascade doesn't

### Summary
- **Best overall**: bd=2 wb=3 with vanilla AlignAtt -> COMET=0.879, YAAL=3.60
- **Best fusion**: bd=3 wb=3 with all 8 signals -> COMET=0.875, YAAL=3.69
- **Recommended config**: Use vanilla AlignAtt with tuned bd/wb per direction
- **Signals add marginal value** at current tuning; may help more on edge cases / different directions

## Bugs Fixed in This Iteration
1. **simulate_backend empty translation**: `_handle_segment_end()` cleared committed_ids before `get_full_translation()` was called. Fixed: accumulate text from step results.
2. **Chinese BLEU = 0**: sacrebleu default tokenizer can't handle CJK. Fixed: pass `tokenize="zh"` for Chinese/Japanese.
3. **KV cache always on CPU**: `create_context()` hardcoded n_gpu_layers=0. Fixed: wire through from config.
4. **Cross-lingual head fallback**: HY-MT EN-DE/IT/CS had no head configs. Fixed: fall back to any matching model heads (validated >97% TS mass transfer).
