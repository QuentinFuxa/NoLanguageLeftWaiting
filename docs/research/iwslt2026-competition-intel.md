# IWSLT 2026 Competition Intelligence

## Critical Dates
- **April 1-15, 2026**: Evaluation window (~10 days away)
- **April 15, 2026**: System submission deadline
- **April 24, 2026**: System description paper submission
- All deadlines 11:59PM UTC-12:00

## Evaluation Setup
- **Hardware**: Single NVIDIA H100 80GB for Docker submissions
- **Language pairs**: EN-DE, EN-ZH, EN-IT, CS-EN
- **Quality metric (primary)**: COMET (not BLEU!)
- **Latency metric (primary)**: LongYAAL via OmniSTEval
- **Latency metric (secondary)**: StreamLAAL
- **Two latency regimes**: Low and High (thresholds per language pair TBA)
- **Ranking**: Quality within latency constraints, non-computation-aware LongYAAL
- **Sub-tracks**: (1) Speech-to-Text, (2) Speech-to-Text with Extra Context (paper PDFs)

## IWSLT 2025 Top Systems (Our Competitors)

| Team | Approach | Result |
|------|----------|--------|
| **CUNI (Winner)** | Whisper + AlignAtt + EuroLLM cascade | Best COMET in CS-EN, EN-DE/ZH/JA (high lat) |
| CMU | InfiniSST: Wav2Vec2.0 + Qwen2.5-7B | Multi-turn LLM, KV management |
| NAIST | Whisper + DeCo + Qwen LLM | Local agreement, SHAS segmenter |
| BeaverTalk | Whisper + Gemma 3 12B cascade | LoRA, conversational prompting |
| MLLP-VRAIN | Whisper + NLLB-3.3B | Document-level, wait-k + RALCP |

## IWSLT 2026 Official Baseline
- Cascade: Qwen3-ASR-1.7B + Qwen3-4B-Instruct-2507
- NER extraction from paper PDFs (Qwen3-30B-A3B) for context sub-track
- GitHub: https://github.com/owaski/iwslt-2026-baselines

## Key Insights for Our System

### Our Unique Advantages
1. **Decoder-only LLM AlignAtt is novel**: CUNI only used AlignAtt with encoder-decoder Whisper
2. **Multi-signal fusion is publishable**: No one else has weighted border signals
3. **top_p aggregation discovery**: Not reported in any paper
4. **99.9% of offline quality**: top_p + wb=4 gives COMET 0.895 vs full-sentence 0.896

### Actionable Ideas (Not Dead Ends)
1. **Syntax-aware word batching (SASST)**: Dependency-based chunking instead of fixed wb
2. **Extra Context sub-track**: NER extraction from paper PDFs for terminology injection
3. **SENTENCE_CUT policy**: Split long source into translatable chunks (SimulInterpret paper)
4. **Group Position Encoding**: Preserve relative positions within source/target for RoPE models

### Papers to Cite
- CUNI SimulStreaming: https://arxiv.org/abs/2506.17077
- toLLMatch (LLMs as zero-shot SimulMT): https://arxiv.org/abs/2406.13476
- SSBD: https://arxiv.org/abs/2509.21740
- REINA (entropy policy): https://arxiv.org/abs/2508.04946
- OmniSTEval/LongYAAL: https://arxiv.org/abs/2509.17349
- ExPosST (positional slots): https://arxiv.org/abs/2603.14903

## Repos to Watch
- OmniSTEval: https://github.com/pe-trik/OmniSTEval
- SimulStreaming: https://github.com/ufal/SimulStreaming
- IWSLT 2026 baselines: https://github.com/owaski/iwslt-2026-baselines
- toLLMatch: https://github.com/RomanKoshkin/toLLMatch
- BeaverTalk: https://github.com/OSU-STARLAB/BeaverTalk
