# LocalAgreement Research Notes

## Core Algorithm

LocalAgreement (Polak et al., IWSLT 2022/2023) is a simultaneous translation policy:
1. Re-translate full accumulated source from scratch on each update
2. Compare new translation with previous translation
3. Commit only the longest common prefix (stable part)
4. Hold unconfirmed suffix for next comparison
5. On is_final: commit everything

## Key Papers

| Paper | ID | Contribution |
|-------|------|-------------|
| Arivazhagan et al. 2020 | arxiv 2004.03643 | Re-translation vs streaming; biased beam search; NE metric |
| Polak et al. 2023 | arxiv 2309.11379 | LA + hold-n policies; incremental blockwise beam search |
| Machacek & Polak 2025 | arxiv 2506.17077 | CUNI IWSLT 2025 winner; AlignAtt (ASR) + LA (MT) |
| Zeng et al. 2025 | arxiv 2509.21740 | SSBD; 1.3-1.7x speedup for re-translation |
| Papi et al. 2023 | INTERSPEECH 2023 | AlignAtt original paper |

## CUNI IWSLT 2025 Architecture

- AlignAtt for Whisper (PyTorch, needs attention access) -- controls ASR streaming
- LocalAgreement for EuroLLM text-to-text (CTranslate2, no attention) -- MT stability
- Key: LA doesn't need attention weights, works with any model/runtime

## SSBD (Self-Speculative Biased Decoding)

Directly applicable to our alignatt-la backend for 1.3-1.7x speedup:

```
Biased verification:
  P'(y_i) = (1 - beta) * P_model(y_i) + beta * delta(y_i = y_i_prev)
  beta = 0.2 recommended (good speed/quality balance)
```

Algorithm:
1. Feed previous translation as draft tokens
2. Verify in ONE parallel forward pass (not token-by-token)
3. Find first divergence point
4. Accept all tokens before divergence (skip autoregressive generation)
5. Resume autoregressive only from divergence point

Results (from paper):
- EN-DE: 1.69x speedup, same COMET
- EN-ZH: 1.43x speedup, same COMET
- EN-JA: 1.36x speedup, same COMET

## Normalized Erasure (NE) Metric

```
NE = (1/J) * sum_{i=2}^{J} [ |o_{i-1}| - |LCP(o_i, o_{i-1})| ]
```
NE < 0.2 = low revision (< 1 token revised per 5 final tokens)

## Implementation Notes

Our alignatt_la_backend.py improvements vs iwslt26-sst:
- Token-level LCP (more precise than word-level)
- KV cache prefix reuse across re-translations
- Hybrid: AlignAtt border detection within each re-translation
- Dynamic border distance support
- 7 aggregation methods for border detection

Potential improvements (from CUNI):
- Forced decoding of committed prefix before generating
- Two-pass catch-up: extra LA comparison per update
- ~~SSBD integration for speculative draft reuse~~ **DONE (Iteration 4)**

## SSBD Implementation (Iteration 4)

Implemented in `alignatt_la_backend.py`:
- `ssbd_accept()`: Biased acceptance via log-ratio trick (avoids full softmax)
- `_retranslate_ssbd()`: 3-phase algorithm:
  1. Batch verify draft (one forward pass with `output_last_only=False`)
  2. Biased acceptance: `P'(draft) = (1-beta)*P(draft) + beta`
  3. Autoregressive continuation from divergence point
- Config: `ssbd_beta=0.2` recommended, sweep via `ssbd=0.0,0.1,0.2,0.3`
- Stats: `get_ssbd_stats()` returns acceptance rate
- **Needs GPU testing on A40**
