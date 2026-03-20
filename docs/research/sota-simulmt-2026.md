# SOTA SimulMT Research (2025-2026)

Research compiled 2026-03-20, updated with March 2026 findings (iteration 10 update).

---

## IWSLT 2026 Simultaneous Track

- **Primary latency metric**: Non-computation-aware LongYAAL (via OmniSTEval)
- **Secondary latency**: StreamLAAL for year-over-year consistency
- **Quality metric**: COMET (primary), plus chrF, BLEURT. XCOMET-XL for internal eval.
- **Hardware constraint**: Single H100 80GB, Docker submission preferred
- **Data condition**: "Constrained with LLMs" -- any open-weight model with permissive license
- **Two tracks**: Standard S2T and S2T with Extra Context (ACL paper PDFs)
- **Language pairs**: En-De, En-Zh, En-It, Cs-En
- **Official baselines repo**: https://github.com/owaski/iwslt-2026-baselines
  - Cascade: Qwen3-ASR-1.7B -> Qwen3-4B-Instruct-2507
  - NER via Qwen3-30B-A3B-Instruct-2507-FP8 (vLLM)
  - Alignment: Qwen3-ForcedAligner-0.6B
  - Context consistently improves quality across all language pairs
  - **Note**: Baseline uses Qwen3-4B which we know is inferior to HY-MT

### IWSLT 2025 Winner: CUNI/SimulStreaming (direct competitor)

- Architecture: Whisper large-v3 (AlignAtt) + EuroLLM-9B-Instruct (LocalAgreement)
- **Forced decoding** of stable hypothesis prefix into KV cache (we implement this)
- Context buffer: max 300-500 tokens, trimmed from beginning
- **Scores**: En-Zh 46.44 BLEU, En-De 38.46 BLEU, Cs-En 18.83 BLEU
- StreamLAAL: 2630-3934ms
- Domain-specific prompting, beam search (5 beams for Cs-En)
- Buffer trimming: sentence-level for German, segment-level for Zh/Ja

### CMU IWSLT 2025 System

- Wav2Vec 2.0 + Qwen2.5-7B-Instruct
- RoPE in speech encoder, sliding window KV cache (1K tokens)
- En-Zh: 44.3 BLEU / 2.189s, En-De: 25.1 BLEU / 1.689s
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

## New Papers (March 2026 Update)

### Hibiki-Zero -- GRPO RL for SimulMT (arxiv 2602.11072, Feb 2026)
- Kyutai Labs, 3B model using **Group Relative Policy Optimization** RL
- BLEU-based reward to learn efficient translation policy
- SOTA on 5 X-to-English tasks; surpasses Seamless in speaker similarity by 30+ pts
- **Relevance**: GRPO could fine-tune our LLM for better READ/WRITE decisions, complementing AlignAtt

### SimulU -- Training-Free Long-Form Policy (arxiv 2603.16924, Mar 2026)
- Uses cross-attention in pre-trained E2E models to regulate input/output
- Better tradeoff than cascaded models on MuST-C across 8 languages
- **Relevance**: Low for text-based, but history management approach could improve context

### EASiST -- Lightweight Policy Head (arxiv 2504.11809, Apr 2025)
- Lightweight head predicts READ/WRITE based on semantics (simpler than full attention)
- Superior on MuST-C EN-DE, EN-ES
- **Relevance**: A small learned policy head could be faster than AlignAtt attention analysis

### SimulSA -- 1% Data Activates SimulMT (arxiv 2509.15692, Sep 2025)
- Only 1% simultaneous data added to SFT activates SimulMT capabilities
- **Relevance**: If we fine-tune HY-MT, very few SimulMT examples needed

### SASST -- Syntax-Aware Chunking (arxiv 2508.07781, Aug 2025)
- Dependency-relation-based chunking (noun phrases, verb-object structures)
- Qwen3-8B outperforms LLaMA3-8B by 1.2-3.2 BLEU
- **Relevance**: Could replace fixed word_batch with syntax-aware boundaries

### Speculative Decoding Advances (ICLR 2026)
- **SSD** (arxiv 2603.03251): Draft predicts verification outcomes in parallel. Up to 2x over standard spec dec.
- **Mirror** (arxiv 2510.13161): Bidirectional speculation. 2.8-5.8x speedup on 14-66B models.
- **ConFu** (arxiv 2603.08899): "Contemplate tokens" expose reasoning signals for drafting. 8-11% over EAGLE-3.

### Seed LiveInterpret 2.0 (ByteDance, early 2026)
- Production ZH-EN/EN-ZH system: >70% accuracy, 2-3s latency
- Zero-shot voice cloning, full-duplex
- Sets commercial viability benchmark

### NEW (Iteration 6 Survey) -- High Priority

**AliBaStr-MT** (arxiv 2503.22051, Apple, Mar 2025)
- Trains 6M-param binary classifier on cross-attention alignment pseudo-labels
- Tunable delta threshold at inference (like our border_distance but learned)
- EN-ES 30.44 BLEU, narrowing gap to offline (32.4)
- **Action**: Could distill our TS-scored alignment data into training labels for a learned border

**REINA** (arxiv 2508.04946, AAAI 2026 Oral)
- Information-theoretic policy: "wait only if reading more input decreases entropy"
- 21% improvement in latency/quality tradeoff
- Extends our `entropy_veto_threshold` with principled information gain
- **Action**: Compare H(token|partial) vs H(token|full) -- if delta is small, WRITE now

**DrFrattn** (EMNLP 2025, ACL Anthology 2025.emnlp-main.1767)
- Learns adaptive READ/WRITE policy directly from attention with "Shift-k" mechanism
- Closest published work to our AlignAtt approach but with learned thresholds
- **Action**: Must-read. Shift-k could improve our dynamic_border implementation

**StreamingThinker** (arxiv 2510.17238, ICLR 2026)
- Parallel KV caches decouple source encoding from generation
- 80% reduction in pre-reasoning latency, 60% total latency reduction
- **Action**: Parallel KV streams could enable true incremental source processing

**Group Position Encoding** (arxiv 2505.16983, ACL 2025, code available)
- Separate position IDs for source and target groups -- no retraining needed
- Eliminates positional interference when source grows incrementally
- Code: github.com/EIT-NLP/StreamingLLM
- **Action**: Implementable in llama.cpp with position ID manipulation. HIGH priority.

**Learning to Draft** (arxiv 2603.01639, ICLR 2026)
- RL-based speculative decoding: 2.24-4.32x speedup, +36.4% vs EAGLE-3
- **Action**: Could RL-optimize SSBD beta per-context instead of fixed 0.2

### NEW (Iteration 6 Survey) -- Medium Priority

**SimulMask** (arxiv 2405.10443, EMNLP 2024)
- Modify attention mask during SFT to enforce SimulMT policy (simpler than EAST)
- Works with LoRA fine-tuning. **Action**: If fine-tuning HY-MT for SimulMT

**DPO Segmentation** (arxiv 2510.12195, Oct 2025)
- DPO-tune LLM to predict optimal chunk boundaries for translation quality
- **Action**: Could replace fixed word_batch with learned segmentation

**RASST** (arxiv 2601.22777, Jan 2026)
- Retrieval-augmented SimulMT: +3 BLEU, +16% terminology accuracy
- **Action**: For IWSLT 2026 "Extra Context" subtrack

**PEARL** (arxiv 2408.11850, ICLR 2025, code: github.com/smart-lty/ParallelSpeculativeDecoding)
- Pre-verify + post-verify for speculative decoding. Adaptive draft length
- **Action**: Could extend SSBD with pre-verification during drafting

**Nightjar** (arxiv 2512.22420, Dec 2025)
- MAB planner for dynamic speculative length selection per batch
- **Action**: Decide whether to use SSBD at all per sentence

### NEW (Iteration 10 Survey) -- Key Validation

**Research gap confirmed (March 2026)**: No new training-free attention-based border
detection methods found beyond what NLLW already implements. AlignAtt remains SOTA
for training-free border detection with decoder-only LLMs. The field is moving toward
end-to-end trained models (Hikari, Hibiki-Zero) or RL-optimized policies rather than
training-free attention-based approaches.

**Publication opportunity**: No one has published attention head detection/selection
for decoder-only LLMs applied to SimulMT border detection (our detect_heads.py).
This is novel and could be a paper contribution.

**Our implementations are validated by top venues:**
- LSG logit KL (2501.00868) -> AAAI 2026
- SSBD (2509.21740) -> updated Jan 2026
- REINA entropy change (2508.04946) -> AAAI 2026 Oral

### IWSLT 2026 Update
- **Baselines repo**: github.com/owaski/iwslt-2026-baselines
  - Qwen3-ASR-1.7B + Qwen3-4B-Instruct-2507
  - New "Extra Context" subtrack (paper context improves quality)
- **Two latency regimes**: Low and High (explicit thresholds TBD)
- **Evaluation period**: April 1-15, 2026 (approaching!)
- **EN->IT replaces EN->JP** for 2026 (we have corpus + head configs ready)
- **CUNI won IWSLT 2025** using AlignAtt + EuroLLM with LocalAgreement
  - Beat organizer baseline by 2-22 BLEU across directions
  - Code: github.com/ufal/SimulStreaming
  - CUNI could NOT use AlignAtt with EuroLLM (no attention extraction in CTranslate2)
  - They only used AlignAtt for CS->EN (direct Whisper), LA for cascade EN->X
  - **Our key advantage**: llama.cpp attention extraction lets us use AlignAtt with ANY model

### CUNI IWSLT 2025 Detailed Results
| Direction | BLEU | SLAAL | vs Baseline |
|-----------|------|-------|-------------|
| CS->EN (low) | 18.49 | 2000ms | +3.3 |
| CS->EN (high) | 18.83 | 4000ms | +2.2 |
| EN->DE (cascade) | 38.46-39.84 | 2472-3934ms | +13 |
| EN->ZH (cascade) | 46.44-49.91 | 3698-5449ms | +22 |
| EN->JP (cascade) | 34.69 | 4654ms | +18 |

### IWSLT 2026 Baseline Configuration
- Qwen3-ASR-1.7B + Qwen3-4B-Instruct-2507 (general-purpose, NOT translation-specific)
- Local Agreement policy, speech_chunk_size 640-1280ms
- repetition_penalty 1.05, temperature 0.0 (greedy)
- NER context injection from paper PDFs (for Extra Context subtrack)
- SimulStream for inference, OmniSTEval for evaluation

### Competition Integration Requirements
- **SimulStream**: WebSocket server, subclass `SpeechProcessor`, implement `process_chunk()`
- **OmniSTEval**: `omnisteval longform --comet --comet_model Unbabel/XCOMET-XL`
- **Docker on H100 80GB**: Q8_0 7B GGUF = ~8GB, plenty of room for KV cache

### Strategic Position
1. **HY-MT + AlignAtt is unexplored in competition** -- CUNI used EuroLLM + LA
2. **Our baseline is much stronger**: HY-MT 0.842 XCOMET-XL vs Qwen3-4B (general-purpose)
3. **AlignAtt with llama.cpp attention extraction** is unique capability -- no one else has this
4. **Need SimulStream integration** for Docker submission (TODO)

---

## Research Opportunities for NLLW

### Implemented (Iterations 1-6):
1. ~~SSBD for accelerating alignatt-la re-translation~~ DONE (Iteration 4)
2. ~~Attention entropy as dynamic border distance~~ DONE (Iteration 3)
3. ~~NE (Normalized Erasure) metric~~ DONE (Iteration 4)
4. ~~Adaptive SSBD beta (entropy-based per-token bias)~~ DONE (Iteration 5)
5. ~~Gaussian kernel consensus aggregation~~ DONE (Iteration 5)
6. ~~LA forced decoding (CUNI approach)~~ DONE (Iteration 5)
7. ~~Display-only mask-k~~ DONE (Iteration 4)
8. ~~LA two-pass catch-up~~ DONE (Iteration 6)
9. ~~Adaptive Multi-Strategy (AMS) aggregation~~ DONE (Iteration 6)
10. ~~Per-head temperature normalization~~ DONE (Iteration 6)
11. ~~Cross-lingual head transfer analysis~~ DONE (Iteration 6) -- **confirmed: EuroLLM/HY-MT/Qwen3.5 >97% TS mass transfer**

### High priority (to investigate):
1. **ExPosST position slot reservation** -- eliminates KV recomputation entirely
2. **Group Position Encoding** (ACL 2025) -- simpler alternative, code available
3. **GRPO fine-tuning** (SeqPO-SiMT/Hibiki-Zero) -- RL-optimize READ/WRITE decisions
4. **Syntax-aware chunking** (SASST) -- replace fixed word_batch with dependency-aware chunking
5. **SSD parallel speculation** -- extend SSBD to pre-compute multiple draft continuations
6. **Human-like strategies** (Sentence_Cut, Drop) -- aggressive latency reduction

### Additional references from cross-lingual research:
- **arXiv 2502.11806** -- "Exploring the Translation Mechanism of LLMs": >70% head overlap for same-source pairs via subspace-intervened path patching. 64 heads sufficient for full translation quality.
- **arXiv 2511.07498** -- "Focusing on Language" (LAHIS): ~1% language-general heads, only 14-20 params for per-language adaptation.
- **arXiv 2602.16740** -- "Attention Head Stability": middle-layer heads least stable across seeds, but stable within a model.

### Our advantages:
- AlignAtt validated by IWSLT 2025 winner (CUNI)
- Translation Heads paper (ICLR 2026) validates our TS scoring
- KV cache reuse already implemented (3-5x speedup)
- SSBD + adaptive beta implemented (expected 1.3-1.7x additional speedup)
- 9 novel aggregation methods + AMS auto-selection (no published baselines)
- LA forced decoding + two-pass for consistency (CUNI approach)
- Per-head temperature normalization for fair aggregation
- Cross-lingual head transfer validated: most models need only 1 detection run
- HY-MT1.5-7B: 0.842 XCOMET-XL EN-ZH (strong baseline)
- **NLLW is the first system applying AlignAtt to decoder-only LLMs for SimulMT**
