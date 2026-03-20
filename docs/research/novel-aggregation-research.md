# Novel Attention Aggregation for SimulMT Border Detection

## Literature Gap

**No published work** explores attention aggregation strategies beyond:
- Single head argmax (EDAtt, Papi 2023)
- Average across heads then argmax (AlignAtt, Papi 2023)
- Sum-threshold (EDAtt)

Our TS-weighted vote (from iwslt26-sst) is itself novel. All 7 methods in NLLW are unpublished.

## Implemented Methods (Iteration 3)

| Method | Key Property | Expected Behavior |
|--------|-------------|-------------------|
| `ts_vote` | Original, argmax per head | Fast, established baseline |
| `softmax_mean` | Weighted avg position | Smoother, captures bimodal attention |
| `entropy_weighted` | Dynamic head reliability | Adapts per-token, suppresses confused heads |
| `consensus` | Multi-head agreement | Conservative, filters outlier heads |
| `geomean` | Product of Experts | Veto power -- requires cross-head agreement |
| `top_p` | Nucleus attention | Ignores noise in attention tails |
| `ensemble` | Combines methods | Best of multiple strategies |

## Priority for GPU Testing

1. `entropy_weighted` -- lowest-hanging fruit, should help on uncertain sentences
2. `geomean` -- elegant veto property, should reduce hallucination-causing early stops
3. `ensemble` -- combines strengths
4. `consensus` -- requires agreement, more conservative
5. `softmax_mean` -- smoother decisions

## Future Hybrid Ideas (not yet implemented)

### Entropy-Gated Geometric Mean (EGGM)
- Hard gate: only include heads with entropy < threshold in geomean
- Combines dynamic reliability (entropy) with consensus requirement (geomean)

### Adaptive Multi-Strategy (AMS)
```
avg_entropy = mean(entropy per head)
agreement_ratio = max_agreement_count / K

if avg_entropy < 0.3 and agreement_ratio > 0.6:
    method = "ts_vote"       # confident + agreed: fast simple method
elif avg_entropy > 1.0:
    method = "geomean"       # uncertain: use conservative veto
else:
    method = "entropy_weighted"  # mixed: adapt per head
```

### Gaussian Kernel Consensus
```
K_agree[j] = sum_h TS_h * exp(-(j - pos_h)^2 / (2 * sigma^2))
```
Generalizes ts_vote (sigma->0) and softmax_mean (sigma->inf). Single hyperparameter.

### Temperature-Normalized Attention
Per-head temperature learned during head detection:
```
tau_h = mean_entropy_h / H_target
A_h_norm = softmax(log(A_h) / tau_h)
```
Normalizes heads to common sharpness before aggregation.

## Key References

- Papi et al. 2023 (INTERSPEECH): AlignAtt uses average across 8 heads of layer 4
- Papi et al. 2023 (ACL): EDAtt uses sum-threshold, found average > individual heads
- Yang et al. 2026: CompilerKV uses attention entropy for KV cache decisions
- Goindani & Shrivastava 2021: DHICM learned head importance (encoder-decoder only)
