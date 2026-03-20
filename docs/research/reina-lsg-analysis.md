# REINA & LSG Research Analysis

## REINA (arXiv 2508.04946, AAAI 2026 Oral)

**Title:** "Regularized Entropy Information-Based Loss for Efficient Simultaneous Speech Translation"
**Status:** Requires training (6M-param policy net). SimulST only (speech, not text).

### Core formula
```
F(a, S, n, t) := H(s_{n+1} | a_t, S_n) - H(s_{n+1} | a_T, S_n)
```
Policy: READ when F > alpha (model benefits from more source), WRITE otherwise.
Needs full source for training -> NOT training-free.

### Relevance: LOW
- Cannot compute at inference without full source access
- Our `info_gain_threshold` (attention KL) is a simpler proxy for the same concept
- Speech-only (Whisper Medium encoder)

---

## LSG (arXiv 2501.00868, AAAI 2025) -- HIGH PRIORITY

**Title:** "Large Language Models Are Read/Write Policy-Makers for Simultaneous Generation"
**Authors:** Shoutao Guo et al. (ICTNLP)
**Code:** https://github.com/ictnlp/LSG

### Core mechanism (TRAINING-FREE)

WRITE when:
```
KL[p(y_i | x_<=j, y_<i) || p(y_i | x_<=i, y_<i)] > delta
```
OR:
```
max p(y_i | x_<=j, y_<i) > alpha
```

Where:
- `p(y_i | x_<=j)` = logit distribution with current source (j tokens)
- `p(y_i | x_<=i)` = logit distribution with minimal source (wait-1)
- delta = 7.0-9.0 (language-pair dependent)
- alpha = 0.5-0.6 (confidence threshold)

### Why this matters for us

| Signal | What it measures | Type |
|--------|-----------------|------|
| Our attention KL (info_gain) | "Is the model still absorbing source?" | Indirect |
| LSG logit KL | "Did more source CHANGE the output?" | Direct |
| AlignAtt border | "Is the model looking at the end?" | Attention position |

These are complementary. Attention borders catch when the model LOOKS at the boundary.
Logit KL catches when the model's OUTPUT actually changes.

### Results
- Llama2-7B: 33.22 BLEU @ 7.37 AL on WMT15 De-En
- Critical: removing range constraint drops BLEU from 31.60 to 21.95

### Implementation plan

1. After generating each token, fork KV cache (`llama_kv_cache_seq_cp`)
2. Evaluate with reduced source (remove last word_batch)
3. Compare output logit distributions via KL divergence
4. High KL = source mattered -> keep generating (inhibit border)
5. Low KL = source didn't matter -> reinforce border stop

Config: `logit_kl_threshold` (None=disabled, 7.0=recommended), `logit_kl_confidence` (0.6)
Cost: ~0.5ms extra per token (single forward pass with cached KV)

### Key requirement
Needs `llama_kv_cache_seq_cp` in our llama_backend.py. Must test on GPU (A40).

### Sweep commands (once implemented)
```bash
python -m nllw.bench --sweep "logit_kl=5.0,7.0,9.0" --lang en-zh --comet --save
python -m nllw.bench --shift-k 0.4 --logit-kl 7.0 --lang en-zh --comet --save
```
