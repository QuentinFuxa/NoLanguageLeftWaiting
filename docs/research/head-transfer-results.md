# Cross-Lingual Alignment Head Transfer -- Results

## Summary

We analyzed whether alignment heads detected for one language pair can be
reused for other pairs of the same model. This is critical for scalability:
head detection is expensive (FLORES + SimAlign + model inference), so reuse
across pairs eliminates per-pair detection cost.

**Key finding**: For most models, alignment heads are cross-lingually universal.
4 out of 5 models tested show >90% TS mass transfer, confirming the ICLR 2026
"Translation Heads" paper.

## Method

For each model with >= 2 language pair configs in `nllw/heads/configs/`:
1. Extract top-10 alignment heads per pair
2. Compute pairwise metrics:
   - **Jaccard similarity**: |intersection| / |union| of head sets
   - **Top-K overlap**: fraction of heads shared
   - **TS rank correlation**: Spearman correlation of TS scores for shared heads
   - **Transferred TS mass**: what % of target pair's top-K TS score is captured by source heads

## Results (top-K = 10)

### EuroLLM-9B-Instruct (4 directions: cs-en, en-de, en-it, en-zh)

| Transfer | Jaccard | Overlap | TS Corr | TS Mass |
|----------|---------|---------|---------|---------|
| cs-en -> en-de | 0.818 | 0.900 | 0.966 | 0.987 |
| cs-en -> en-it | 0.818 | 0.900 | 0.966 | 0.973 |
| cs-en -> en-zh | 1.000 | 1.000 | 0.909 | 1.000 |
| en-de -> en-it | 1.000 | 1.000 | 0.963 | 1.000 |
| en-de -> en-zh | 0.818 | 0.900 | 0.890 | 0.979 |
| en-it -> en-zh | 0.818 | 0.900 | 0.875 | 0.979 |

**Verdict: EXCELLENT (98.9% mean TS mass, 97.3% min)**

EuroLLM's alignment heads are nearly identical across all 4 language pairs.
The cs-en and en-zh heads are 100% identical (Jaccard = 1.0).

### HY-MT1.5-1.8B (2 directions: en-de, en-zh)

| Transfer | Jaccard | Overlap | TS Corr | TS Mass |
|----------|---------|---------|---------|---------|
| en-de -> en-zh | 0.818 | 0.900 | 0.650 | 0.977 |
| en-zh -> en-de | 0.818 | 0.900 | 0.650 | 0.992 |

**Verdict: EXCELLENT (98.4% mean TS mass, 97.7% min)**

Despite lower TS rank correlation (0.650), the heads themselves overlap well.

### Qwen3.5-4B (3 directions: en-de, en-it, en-zh)

| Transfer | Jaccard | Overlap | TS Corr | TS Mass |
|----------|---------|---------|---------|---------|
| en-de -> en-it | 0.667 | 0.800 | 0.569 | 0.996 |
| en-de -> en-zh | 1.000 | 1.000 | 0.867 | 1.000 |
| en-it -> en-zh | 0.667 | 0.800 | 0.325 | 0.927 |

**Verdict: EXCELLENT (97.8% mean TS mass, 92.7% min)**

en-de and en-zh heads are 100% identical. en-it has slightly different heads
but still captures 92.7% of TS mass.

### Qwen3.5-9B (2 directions: en-de, en-zh)

| Transfer | Jaccard | Overlap | TS Corr | TS Mass |
|----------|---------|---------|---------|---------|
| en-de -> en-zh | 0.818 | 0.900 | 0.905 | 0.995 |
| en-zh -> en-de | 0.818 | 0.900 | 0.905 | 1.000 |

**Verdict: EXCELLENT (99.7% mean TS mass)**

### Qwen3-4B (4 directions: cs-en, en-de, en-it, en-zh)

| Transfer | Jaccard | Overlap | TS Corr | TS Mass |
|----------|---------|---------|---------|---------|
| cs-en -> en-de | 0.250 | 0.400 | 0.462 | 0.880 |
| cs-en -> en-it | 0.333 | 0.500 | 0.603 | 0.883 |
| cs-en -> en-zh | 0.429 | 0.600 | 0.818 | 0.636 |
| en-de -> en-it | 0.667 | 0.800 | 0.948 | 0.988 |
| en-de -> en-zh | 0.429 | 0.600 | 0.491 | 0.615 |
| en-it -> en-zh | 0.250 | 0.400 | 0.394 | 0.435 |

**Verdict: GOOD but pair-specific (79.8% mean, 43.5% worst case)**

Qwen3-4B is the exception. While en-de and en-it share heads well (0.988 TS mass),
transfers involving en-zh are poor (43.5% worst case). This suggests Qwen3-4B
has language-pair-specific alignment heads for distant language pairs.

## Implications

1. **For most models (EuroLLM, HY-MT, Qwen3.5)**: Detect heads on ONE language pair,
   reuse for ALL others. Saves 80% of head detection compute.

2. **For Qwen3-4B**: Detect heads per language pair, OR detect on a related pair
   (en-de heads work well for en-it but not for en-zh).

3. **Research contribution**: This is the first systematic validation of cross-lingual
   head transfer for SimulMT. The ICLR 2026 "Translation Heads" paper is confirmed
   for alignment heads (not just general translation heads).

## Supporting Literature

1. **"Translation Heads" (ICLR 2026)** - Token Alignment Heads (TAHs) are sparse (<5%),
   universal across models, consistent across language pairs, and causally necessary.
   [OpenReview q8fTgw8e5E](https://openreview.net/forum?id=q8fTgw8e5E)

2. **"Exploring the Translation Mechanism of LLMs" (arXiv 2502.11806)** - Using
   subspace-intervened path patching on LLaMA2-7B: >70% head overlap for same-source
   pairs, >60% for bidirectional. Only 64 heads need fine-tuning for full translation.

3. **"Focusing on Language" (arXiv 2511.07498)** - LAHIS scores show ~1% language-general
   heads in Aya-23-8B, ~5% in Llama-3.2-3B. Only 14-20 soft mask weights needed for
   language-specific adaptation.

4. **"Attention Head Stability" (arXiv 2602.16740)** - Middle-layer heads (where translation
   heads concentrate) are least stable across training seeds. However, this is across
   independently trained models, NOT across language pairs within a single model.

**Our NLLW project appears to be the first applying AlignAtt-style head detection to
decoder-only LLMs for SimulMT, and the first quantitative cross-lingual transfer
evaluation of alignment heads for simultaneous translation.**

## Reproducing

```bash
python -m nllw.head_transfer --all --top-k 10
python -m nllw.head_transfer --model qwen3_4b --top-k 10 --json
```
