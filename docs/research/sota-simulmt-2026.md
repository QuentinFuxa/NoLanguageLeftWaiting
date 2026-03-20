# SOTA SimulMT Research (2025-2026)

Research compiled 2026-03-20 for IWSLT 2026 submission preparation.

---

## IWSLT 2026 Simultaneous Track

- **Primary latency metric**: LongYAAL (via OmniSTEval)
- **Quality metric**: XCOMET-XL primary, SacreBLEU/chrF secondary
- **Hardware constraint**: Single H100 80GB
- **Data condition**: Any open-weight model with permissive license
- **Official baselines repo**: https://github.com/owaski/iwslt-2026-baselines

### IWSLT 2025 Winner: CUNI/SimulStreaming

- Architecture: Whisper (ASR) + EuroLLM (cascade MT)
- Policies: AlignAtt for Whisper, LocalAgreement for EuroLLM
- Won highest COMET in CS-EN (2s/4s), EN-DE/ZH/JA (4-5s)
- +2 BLEU on CS-EN, +13-22 BLEU on EN-DE/ZH/JA vs baselines
- Code: https://github.com/ufal/SimulStreaming
- Paper: https://arxiv.org/abs/2506.17077

### IWSLT 2025 Results

| System | EN-ZH BLEU | EN-DE BLEU | Latency | Architecture |
|--------|:----------:|:----------:|:-------:|--------------|
| CUNI | best | best | 2-5s | Whisper + EuroLLM |
| CMU (InfiniSST) | 44.3 | 25.1 | 2.2-2.7s | Wav2Vec2.0 + Qwen2.5-7B |
| BeaverTalk | -- | 24.6-27.8 | 1.8-3.3s | Whisper-L + Gemma-3-12B |
| MLLP-VRAIN | -- | 31.96 | 2.94s | Whisper-L-V3-T + NLLB-3.3B |

---

## Key Papers for NLLW

### Directly Relevant

1. **Translation Heads (ICLR 2026)** -- Confirms "token alignment heads" are sparse, universal, and consistent across LLMs. Validates NLLW's TS scoring approach.
   https://openreview.net/forum?id=q8fTgw8e5E

2. **ExPosST (March 2026)** -- Pre-allocates position slots for incoming source tokens. Zero-recomputation with full KV cache reuse. Works with RoPE/ALiBi.
   https://arxiv.org/abs/2603.14903

3. **SSBD -- Self-Speculative Biased Decoding (Sept 2025)** -- Reuses previous translation as speculative draft. Verifies in single forward pass. **Applicable to alignatt-la re-translation backend.**
   https://arxiv.org/abs/2509.21740

4. **SimulSense (Sept 2025)** -- Detects "sense units" for chunking. 9.6x faster policy decisions.
   https://arxiv.org/abs/2509.21932

### Competitive Threats

5. **Hikari (March 2026)** -- Policy-free WAIT tokens. SOTA on EN-JA/DE/RU.
   https://arxiv.org/abs/2603.11578

6. **EAST (ICLR 2025)** -- Interleaved READ/WRITE tokens via SFT. SOTA across SiMT benchmarks.
   https://arxiv.org/abs/2504.09570
   *Note: We tested EAST and got BLEU 27.3 vs baseline 42 -- dead end for us.*

7. **SeqPO-SiMT (ACL 2025)** -- RL optimization. +1.13 COMET, -6.17 AL.
   https://arxiv.org/abs/2505.20622

8. **Group Position Encoding (ACL 2025)** -- Separate src/tgt positional encodings.
   https://arxiv.org/abs/2505.16983, code: https://github.com/eit-nlp/streamingllm

### Other Notable

9. **InfiniSST (ACL 2025)** -- Multi-turn dialogue for unbounded speech. Sliding window KV cache.
   https://arxiv.org/abs/2503.02969, code: https://github.com/LeiLiLab/InfiniSST

10. **LSG (AAAI 2025)** -- LLM devises its own R/W policy from minimum-latency baseline.
    https://arxiv.org/abs/2501.00868

11. **Human-Like Strategies (Jan 2026)** -- SENTENCE_CUT, DROP, PRONOMINALIZATION actions.
    https://arxiv.org/abs/2601.11002

---

## Detailed Paper Findings (Iteration 4, 2026-03-20)

### SSBD Results (arxiv 2509.21740v2)

Beta tradeoff (EN-ZH, Tower+ 2B):
| beta | COMET | NE   | A/D   | Speedup |
|------|-------|------|-------|---------|
| 0.0  | 0.882 | 1.53 | 53.8% | 1.28x   |
| 0.1  | 0.881 | 1.19 | 66.2% | 1.40x   |
| 0.2  | 0.880 | 1.01 | 71.7% | 1.48x   |
| 0.3  | 0.870 | 0.83 | 77.3% | 1.58x   |
| 0.4  | 0.842 | 0.57 | 85.0% | 1.70x   |
| 0.5  | 0.742 | 0    | 100%  | 2.10x   |

Key technique: **Display-only mask-k** hides last k tokens from user but keeps
them as draft. NE drops from 1.01 to 0.53 with display-only mask-3.

### Hikari Architecture (arxiv 2603.11578)

- Based on Whisper medium (~769M params) with WAIT token (token 93)
- Decoder Time Dilation (DTD): D=4, each decoder token = 80ms audio
- Latency control: bias WAIT token logits at inference (like our border_distance)
- **Not applicable** to text-based pipeline -- requires end-to-end training (96 H100s)
- Results: 42.17 BLEU EN-JA (best), beats CUNI by +8.73 BLEU

### ExPosST Details (arxiv 2603.14903)

- Position formula: pos(t_start) = pos(s_start) + L_slot (L_slot=16 optimal)
- Requires LoRA fine-tuning (rank=32, alpha=16, 2 epochs)
- **Requires modified positional encoding** at inference -- not drop-in for llama.cpp
- Results: +2-3 BLEU over GPE baseline at equivalent latency

### LSG KL-Divergence Policy (arxiv 2501.00868)

Training-free: READ if KL(P_partial || P_full) > delta, WRITE if max(P) > alpha.
Thresholds: delta=7-9, alpha=0.5-0.6. Requires two forward passes per decision.
Could complement AlignAtt but doubles compute per step.

### IWSLT 2026 Details

- Conference: July 3-4, 2026, San Diego (co-located with ACL 2026)
- Language pairs: EN-DE, EN-ZH, EN-IT, CS-EN
- **Primary latency**: LongYAAL (was StreamLAAL in 2025)
- **Primary quality**: COMET (+ chrF, BLEURT)
- Evaluation period: April 1-15, 2026
- Hardware: Single H100 80GB
- New sub-track: Extra Context (ACL paper PDFs as context)

---

## Research Opportunities for NLLW

### Open gaps (no published work):
1. **Alternative attention aggregation** for AlignAtt -- our 7 methods are novel (DONE in Iteration 3)
2. **AlignAtt + speculative generation** combination (DONE: SSBD in Iteration 4)
3. **AlignAtt + human-like strategies** (SENTENCE_CUT, DROP)
4. **Adaptive SSBD beta**: Use per-token entropy to adjust bias (novel combination)
5. **Display-only mask-k**: Keep unstable suffix as draft, hide from user

### Implemented:
1. ~~SSBD for accelerating alignatt-la re-translation~~ DONE (Iteration 4)
2. ~~Attention entropy as dynamic border distance~~ DONE (Iteration 3)
3. ~~NE (Normalized Erasure) metric~~ DONE (Iteration 4)

### To investigate:
1. **Cross-lingual head transfer** -- do EN-ZH heads work for EN-DE?
2. **LSG KL-divergence** as complement to attention-based borders
3. **Confidence-modulated speculative decoding** (arxiv 2508.15371) for adaptive draft length

### Our advantages:
- AlignAtt validated by IWSLT 2025 winner (CUNI)
- Translation Heads paper (ICLR 2026) validates our TS scoring
- KV cache reuse already implemented (3-5x speedup)
- SSBD implemented (expected 1.3-1.7x additional speedup)
- 7 novel aggregation methods (no published baselines)
- HY-MT1.5-7B: 0.842 XCOMET-XL EN-ZH (strong baseline)
