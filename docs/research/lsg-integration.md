# LSG Logit KL Divergence -- Integration Analysis

## Paper: Look, See, and Generate (arxiv 2501.00868, AAAI 2025)

Authors: Shoutao Guo et al. (ICT/CAS)
Core idea: Training-free simultaneous generation using KL divergence between
output logit distributions with full vs reduced source.

## Key Insight

Standard AlignAtt border detection uses attention patterns (WHERE the model
looks) to decide when to stop generating. LSG provides an orthogonal signal:
whether the output logit DISTRIBUTION changes when source tokens are removed
(WHETHER looking there changed the OUTPUT).

These are fundamentally different signals:
- **Attention border**: "The model is looking at the end of available source"
- **LSG logit KL**: "Removing recent source tokens changes what the model predicts"

Combining both gives a stronger read/write decision than either alone.

## Implementation in NLLW

### Signal Flow

```
AlignAtt border check (attention) -> border_hit = True
  |
  v
LSG probe (logit KL):
  1. Get logits_full (already computed from main decode)
  2. Fork KV cache: memory_seq_cp(0, 1, 0, pos)
  3. Remove last K source positions: memory_seq_rm(1, src_end-K, src_end)
  4. Remove last decoded position: memory_seq_rm(1, pos-1, pos)
  5. Re-decode same token on fork: decode_single_at(tok, pos-1, seq=1)
  6. Get logits_reduced from fork
  7. KL = compute_logit_kl(logits_full, logits_reduced)
  8. Cleanup: memory_seq_rm(1, 0, -1)
  |
  v
Decision:
  KL < threshold -> CONFIRM stop (source exhausted, safe to WRITE)
  KL > threshold -> OVERRIDE stop (source still matters, keep generating)
```

### Cost Analysis

Per LSG probe:
- `memory_seq_cp`: O(1), adds sequence tags
- `memory_seq_rm`: O(1), removes tags
- `decode_single_at`: ONE forward pass for ONE token
  - On A40 (7B model): ~1-3ms
  - On L4 (7B model): ~3-8ms
- `get_logits_array` + KL computation: ~0.01ms
- Total: ~1-3ms on A40 per probe

The probe only runs when attention border hits (confirmed by border_confirm),
so amortized cost is low. With border_confirm=2 and typical generation of
10-20 tokens per translate() call, this adds ~5-15% overhead.

### Why KV Cache Fork Works

In llama.cpp, `memory_seq_cp(src, dst, p0, p1)` adds sequence tag `dst` to
existing KV entries in range [p0, p1). It doesn't copy data -- entries are
shared between sequences.

When we `memory_seq_rm(1, ...)`, we only remove the seq 1 tag. Seq 0 entries
are untouched. When we decode on seq 1, the attention mask uses seq 1's
entries. The removed source positions simply have no seq 1 tag, so the model
doesn't attend to them.

After cleanup (`memory_seq_rm(1, 0, -1)`), all seq 1 tags are removed.
The main seq 0 is completely unaffected.

### Threshold Selection

From the LSG paper (Table 2, Llama-2-7B on WMT15 De-En):
- delta = 7.0: BLEU 33.22, AL 7.37 (recommended balance)
- delta = 9.0: BLEU 33.88, AL 8.27 (higher quality, more latency)
- delta = 5.0: BLEU 31.15, AL 5.82 (lower quality, less latency)

Our recommended default: 7.0 for 7B models. May need tuning per model family.

### Known Limitation

The reduced-source logits are approximate: we only remove KV entries, but the
suffix and committed token KV values were originally computed WITH those source
tokens. A full re-computation would require re-decoding everything from the
fork point, which is too expensive.

In practice, this approximation should be sufficient because:
1. The attention mechanism is the primary pathway for source influence
2. Removing KV entries correctly prevents attending to removed source
3. The residual "ghost" influence through position-encoded values is minimal

## Future Work

1. **Standalone LSG policy**: Use logit KL as the PRIMARY read/write signal,
   replacing attention-based border detection entirely. Would require forking
   at EVERY generation step (higher cost but no head detection needed).

2. **Confidence-gated probing**: Only run LSG when attention border is
   "ambiguous" (e.g., attended_pos is close to border_threshold). Skip when
   the attention signal is strong in either direction.

3. **Cached partial-source KV**: Maintain a persistent seq 1 with reduced
   source (LSG paper approach). Avoids fork overhead but requires more
   careful KV cache management.

4. **Multi-K probing**: Compare logits at multiple K values (remove 1, 3, 5
   source tokens) for a richer dependency signal.

## Related Approaches

- **Info gain** (our iteration 7): KL divergence between ATTENTION snapshots.
  Different signal: measures attention pattern change, not output change.
- **REINA** (AAAI 2026): Entropy-based read/write with 6M-param network.
  Requires training, but principled information-theoretic formulation.
- **DrFrattn** (EMNLP 2025): Attention mass-based policy. Our shift-k
  feature implements a variant of this.
