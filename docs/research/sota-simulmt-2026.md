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

## Research Opportunities for NLLW

### Open gaps (no published work):
1. **Alternative attention aggregation** for AlignAtt (softmax pooling, entropy-weighted voting, temperature scaling across heads) -- novel direction
2. **AlignAtt + speculative generation** combination
3. **AlignAtt + human-like strategies** (SENTENCE_CUT, DROP)

### Low-hanging fruit:
1. **SSBD** for accelerating alignatt-la re-translation
2. **Attention entropy** as dynamic border distance signal
3. **Cross-lingual head transfer** -- do EN-ZH heads work for EN-DE?

### Our advantages:
- AlignAtt validated by IWSLT 2025 winner (CUNI)
- Translation Heads paper (ICLR 2026) validates our TS scoring
- KV cache reuse already implemented (3-5x speedup)
- HY-MT1.5-7B: 0.842 XCOMET-XL EN-ZH (strong baseline)
