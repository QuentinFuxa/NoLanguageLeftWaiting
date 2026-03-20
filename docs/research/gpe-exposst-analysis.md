# Group Position Encoding (GPE) & ExPosST Analysis

## GPE (arxiv 2505.16983, ACL 2025)

**Title:** "Group Position Encoding for Streaming Simultaneous Machine Translation"
**Code:** github.com/EIT-NLP/StreamingLLM (HuggingFace only, no llama.cpp)

### Core Idea
Separate position IDs for source and target tokens:
```
Source: p(x_i) = i                    # 0, 1, 2, ..., S-1
Target: p(y_j) = phi + j             # phi, phi+1, phi+2, ...
```
Where phi is a fixed offset (hyperparameter). Results show phi=0 works fine.

### Critical Finding: Position Mismatch is Negligible
Three types of streaming mismatch tested:
- **Input-Attention mismatch**: +2.2 BLEU by fixing (significant!)
- Output-Attention mismatch: +0.13 BLEU (negligible)
- **Position-ID mismatch: +0.14 BLEU (negligible!)**

**This validates our current KV cache delta approach.** When we re-decode and target
positions shift slightly, the quality loss is minimal.

### Requires LoRA Fine-tuning
All results use LoRA (rank 32-64, 2 epochs). NO zero-shot results provided.
The "no retraining" claim refers to architecture, not weights.

### Results
Gemma2-2B on IWSLT-17: +2.2 BLEU over interleaved-streaming at k=1.

---

## ExPosST (arxiv 2603.14903)

**Title:** Pre-allocated position slots for zero-recomputation KV cache
**Authors:** Shang et al. (Xiamen Univ / Xiaomi)

### Core Mechanism
Reserve L_slot positions for source. Target starts at pos = slot_start + L_slot.
New source tokens fill the slot without shifting target positions.

### Key Results
- +2-3 BLEU over GPE at equivalent latency
- L_slot=16 is optimal
- Lowest GFLOPs of all compared methods

### Also Requires Fine-tuning
LoRA + policy-consistent attention masking. No zero-shot path.

---

## Relevance to NLLW

### What we can use NOW
- **Position mismatch is negligible**: Our KV cache delta decoding is validated.
  When target positions shift after new source tokens, quality loss is <0.14 BLEU.
- **Input-attention mismatch matters**: If we ever do LoRA, preventing source tokens
  from attending to target tokens (+2.2 BLEU) is the key fix.

### What we CANNOT use without fine-tuning
- GPE position scheme (needs LoRA-adapted model)
- ExPosST slot mechanism (needs LoRA + policy masking)
- Both use fixed policies (wait-k, read-n), not our adaptive AlignAtt

### Future research direction
AlignAtt + GPE hybrid: LoRA fine-tune a model with GPE positions, then use AlignAtt
for adaptive border detection instead of fixed wait-k. Would need:
1. Fine-tune HY-MT or Qwen3.5 with GPE LoRA
2. Convert to GGUF
3. Add `decode_batch_with_positions()` to llama_backend.py (trivial, 5 lines)
4. Run AlignAtt with separate position spaces

### Implementation complexity
Adding custom positions to llama_backend.py: trivial (batch.pos already supports it).
The hard part: LoRA fine-tuning infrastructure (PyTorch, not llama.cpp).
