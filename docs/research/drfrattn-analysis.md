# DrFrattn Analysis (EMNLP 2025)

**Paper:** "DrFrattn: Directly Learn Adaptive Policy from Attention for Simultaneous Machine Translation"
**Authors:** Libo Zhao, Jing Li, Ziqian Zeng (Hong Kong Polytechnic University)
**Venue:** EMNLP 2025 (pages 34893-34906)
**Code:** No public repo available

## Key Mechanism

DrFrattn derives READ/WRITE policies from cross-attention matrices of encoder-decoder Transformers.

### Cumulative Attention Matrix
For cross-attention `alpha_ij` (target i attending to source j):
```
c_ij = 1 - sum_{k=1}^{j} alpha_{ik}
```
`c_ij` = how much attention mass target token i has NOT placed on source positions 1..j.
When `c_ij < lambda`, switch from READ to WRITE.

### Shift-k Mechanism (Training Only)
```
g_shifted(t) = min{g(t) + k, |X|}    where k ~ Uniform({0, 1, ..., |X|})
```
Randomly shifts READ/WRITE path rightward by k positions during training.
Creates diverse latency coverage. NOT applicable at inference.

### Temperature Sharpening
`tau = 0.6` found optimal for attention sharpening before policy extraction.
Confirms our head_temp_normalize approach (reference entropy 1.5 nats).

## What We Adapted

### Cumulative Attention Aggregation (`aggregate_cumulative_attention`)
Implemented as 10th aggregation method in `nllw/alignatt.py`.

Instead of argmax voting, we compute cumulative mass per head:
- `c_j = 1 - cumsum(attn[0:j+1])`
- Frontier = rightmost position where `c_j >= lambda`
- TS-weighted vote on frontier positions

**Advantages over argmax:**
- Captures distribution shape (not just peak)
- Split attention (40% at pos 5, 40% at pos 7) -> frontier correctly at 7
- Lambda parameter gives continuous latency control

### What We Didn't Take
- Shift-k: training-time trick, not applicable (we're training-free)
- Policy network: requires training additional decoder layer
- Encoder-decoder architecture: we use decoder-only LLMs

## Key Differences from Our Approach

| Aspect | DrFrattn | Our AlignAtt |
|--------|----------|-------------|
| Model | Encoder-decoder | Decoder-only LLM |
| Attention | Cross-attention | Self-attention (TS-scored heads) |
| Training | Required (policy net + shift-k) | Training-free |
| Decision | Threshold on cumulative matrix | Argmax/vote vs border distance |
| Heads | Average all in one layer | Individual heads across all layers |

## Results (from paper)

DrFrattn Shift-k outperforms multi-path wait-k by +2 BLEU on Zh-En at AL~3.
Policy prediction accuracy: 0.907-0.921 (vs 0.803-0.843 for wait-k).

## Sweep Commands

```bash
# Test cumulative aggregation
python -m nllw.bench --aggregation cumulative --lang en-zh --comet --save

# Compare all 10 aggregation methods
python -m nllw.bench --sweep "agg=ts_vote,softmax_mean,entropy_weighted,consensus,geomean,top_p,gaussian_kernel,gaussian_kernel_continuous,cumulative,ensemble" --lang en-zh --comet --save
```
