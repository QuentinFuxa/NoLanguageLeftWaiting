# NoLanguageLeftWaiting (NLLW) -- Simultaneous Machine Translation Library

## Project Overview

Production-quality Python library for Simultaneous Machine Translation (SimulMT). Supports both:
- **NLLB backend** (encoder-decoder, HuggingFace/CTranslate2) -- existing, for streaming ASR integration
- **LLM backend** (llama.cpp, GGUF models) -- new, for AlignAtt-based SimulMT with any decoder-only LLM

### Architecture

```
nllw/
  core.py              -- NLLB TranslationBackend (existing, stable)
  translation.py       -- WhisperLiveKit OnlineTranslation interface (existing)
  languages.py         -- Language code mapping (existing)
  timed_text.py        -- TimedText dataclass (existing)
  llama_backend.py     -- llama.cpp ctypes bindings (ported from iwslt26-sst)
  alignatt.py          -- AlignAtt border detection engine (core SimulMT algorithm)
  prompts.py           -- Prompt format registry (40+ templates for HY-MT, Qwen, EuroLLM, etc.)
  policies.py          -- Read/write policies (AlignAtt, wait-k, confidence, fixed-rate)
  metrics.py           -- SimulMT latency metrics (AL, LAAL, YAAL, AP, DAL, MaxCW)
  eval.py              -- Evaluation harness (FLORES, XCOMET-XL, parameter sweep)
  calibrate.py         -- Fusion weight calibration pipeline (trace collection, label generation, grid search)
  heads/               -- Alignment head detection & configs
    detect.py          -- Head detection algorithm (TS scoring via FLORES)
    configs/           -- Pre-computed head configs (JSON) for popular models
```

### Key Algorithm: AlignAtt Border Detection

1. Source words arrive incrementally (from ASR or text)
2. Build prompt: prefix + context + source + suffix + committed_output
3. Generate tokens with LLM (greedy argmax)
4. After each generated token, extract attention from top-K alignment heads
5. TS-weighted vote: if attended source position >= source_len - border_distance -> STOP (border hit)
6. At sentence boundary: commit translation, reset context, start next segment
7. KV cache reuse: only re-decode changed tokens (3-5x speedup)

### Best Known Results

**NLLW results (COMET wmt22-comet-da, FLORES, 100 sentences, A40, iteration 18):**

| Direction | Model | COMET | 95% CI | YAAL | Config | % offline |
|-----------|-------|:-----:|--------|:----:|--------|:---------:|
| EN-ZH | HY-MT1.5-7B | **0.894** | [0.887, 0.901] | 6.09 | bd=3, wb=4, top_p, p=0.85 | 99.8% |
| EN-DE | HY-MT1.5-7B | **0.881** | [0.873, 0.890] | 5.45 | bd=2, wb=3, top_p, p=0.75 | 99.7% |
| EN-IT | HY-MT1.5-7B | **0.891** | [0.882, 0.899] | 6.76 | bd=2, wb=3, top_p, p=0.9 | **100.2%** |
| CS-EN | HY-MT1.5-7B | **0.879** | [0.871, 0.886] | 5.81 | bd=3, wb=3, top_p, p=0.9 | 99.8% |

**Previous iwslt26-sst results (XCOMET-XL, different metric):**

| Direction | Model | XCOMET-XL | Config |
|-----------|-------|:---------:|--------|
| EN-ZH | HY-MT1.5-7B | 0.842 | bd=3, wb=2, no context |
| EN-DE | HY-MT1.5-7B | 0.786 | bd=3, wb=2 |
| EN-IT | HY-MT1.5-7B | 0.752 | bd=4, wb=3 |
| CS-EN | HY-MT1.5-7B | 0.908 | bd=2, wb=1 |

### Dead Ends (do NOT revisit)

- EAST learned policy (BLEU 27.3 vs baseline 42)
- LoRA no-think block (-0.178 COMET)
- GDN warm-start (33 hallucinations)
- Extra glossary (hurts small models)
- HY-MT context injection (hurts ALL 4 directions: EN-DE -0.084, EN-IT -0.125, CS-EN -0.107)
- Confidence-based stopping alone (COMET 0.468)
- Fixed-rate tokens (COMET 0.293-0.334)
- TAF source lookahead (worse on EN-ZH)
- Seed-X-PPO-7B, Qwen3-4B-2507, HY-MT1.5-1.8B, TranslateGemma-4B (all inferior)
- Entropy veto threshold (all thresholds hurt: 0.5->0.494, 0.75->0.589, 1.0->0.683)
- softmax_mean aggregation (COMET 0.812 vs ts_vote 0.879)
- Signal fusion / cascade (marginal +0.002 COMET, not worth complexity)
- Boolean cascade signals (coverage causes latency explosion)
- Repetition halt (rep=2 hurts EN-ZH by -0.004 COMET, neutral for EN-DE)
- top_p_weighted aggregation (COMET 0.885 EN-ZH, 0.852 EN-DE -- much worse than top_p)
- Qwen3.5-9B (hybrid DeltaNet architecture: only 25% of layers have softmax attention, AlignAtt unusable)

---

## Machines

| Machine | Used for |
|---------|----------|
| **A40** (always up) | llama.cpp MT experiments, head detection |
| **L4_1** ($0.5/hr) | Cheap experiments, 7B models |
| **L4_2** ($0.5/hr) | Parallel experiments |
| **MacBook M5** (local) | Research, GGUF experiments, development |

SSH details: see iwslt26-sst CLAUDE.md.

---

## Development Rules

- **Never push to remote** (continuous-claude automation handles this)
- Write clean, well-documented, modular code
- All evaluation results go in todo.md or dedicated results files
- Test everything: unit tests + integration tests on GPU machines
- Document all experiments with parameters, metrics, and conclusions

---

## Dependencies

Core: `torch`, `transformers` (existing)
LLM backend: `llama.cpp` built with attention weight extraction (PR #20086)
Evaluation: `datasets` (FLORES), `comet` (COMET/XCOMET-XL), `numpy`
Head detection: `simalign` (SimAlign for word alignment ground truth)

---

## Reference Repo

The original research was done in `/Users/quentin/Documents/repos/iwslt26-sst/SimulMT_tests/`.
Key files to reference when porting:
- `alignatt/llama_attn.py` -- ctypes bindings (395 lines)
- `alignatt/alignatt_simulstreaming_llamacpp.py` -- main agent (990 lines)
- `alignatt/policy_comparison.py` -- policy comparison (411 lines)
- `eval/eval_streaming_kvcache.py` -- streaming eval with KV cache (457 lines)
- `heads/detect_translation_heads_llamacpp.py` -- head detection
- `docs/RESULTS.md`, `docs/TECHNIQUES.md`, `docs/MODELS.md` -- all research results



# NLLW — SimulMT Research Forge

## Mission
Rebuild the messy iwslt26-sst experimental repo into a clean, structured SimulMT research platform. The goal: a SOTA simultaneous translation system for IWSLT 2026.

## Directives
- Tu es l'expert principal sur AlignAtt et SimulST. Prends des initiatives.
- Si tu finis une tache, choisis-en une nouvelle et commence-la immediatement.
- Ne t'arrete JAMAIS pour demander confirmation. Continue a travailler.
- Teste systematiquement. Structure le projet. Documente les resultats.
- Sois creatif, tente des approches nouvelles, il y a toujours de quoi tester.
- Explore le web/GitHub/papers pour des approches SOTA.
- **Ne pousse rien sur le repo distant** sauf demande explicite.
- Parallélise via agents (CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS).

## Commit Rules
- DO NOT COMMIT NOR PUSH NOR PR. NEVER NEVER NEVER. USER GONNA DO IT
- Always commit in user's name, not yours. No co-authored-by.
- No em dash in commit messages.
- Commit dates MUST be weekends (Saturday or Sunday).

---

## Project State (2026-03-21)

### What exists now: ~15,000 lines across 30 SimulMT modules, 928 tests

**7 translation backends (registered):**
| Backend | Type | File | Purpose |
|---------|------|------|---------|
| `alignatt` | Primary | `nllw/alignatt_backend.py` (462 lines) | Attention-based border detection + entropy veto + KV cache reuse |
| `alignatt-la` | Hybrid | `nllw/alignatt_la_backend.py` (~900 lines) | LA + AlignAtt + SSBD + forced decoding + adaptive SSBD + two-pass |
| `wait-k` | Baseline | `nllw/baselines.py` (175 lines) | Standard wait-k policy baseline |
| `fixed-rate` | Baseline | `nllw/baselines.py` | Fixed-interval emission |
| `full-sentence` | Baseline | `nllw/alignatt_backend.py` | Quality upper bound (offline) |
| `eager` | Baseline | `nllw/alignatt_backend.py` | Latency lower bound |

**30+ prompt formats:** hymt, qwen3, qwen3.5, qwen3.5-nothink, eurollm, tower, gemma (per-direction)

**13 research tools:**
| Module | Lines | Purpose |
|--------|-------|---------|
| `eval.py` | 410 | BLEU/COMET/xCOMET-XL evaluation, parameter sweep |
| `simulate.py` | 135 | Policy replay, Average Lagging computation |
| `corpus.py` | 622 | 120-sentence categorized test corpus (5 directions) |
| `experiment.py` | 359 | Experiment YAML config/result registry |
| `analysis.py` | 309 | Pareto frontier, edge cases, report generation |
| `detect_heads.py` | 559 | Auto alignment head detection for any GGUF model |
| `metrics.py` | 560 | BLEU, COMET, xCOMET-XL + LongYAAL (IWSLT primary), StreamLAAL, all latency metrics + NE |
| `bench.py` | 365 | Unified one-command benchmarking CLI with sweep, compare, 20+ shortnames |
| `omnisteval.py` | 258 | OmniSTEval JSONL output format for IWSLT submission |
| `research.py` | 191 | Compute-aware latency (CA-AL, CA-YAAL), benchmark suite |
| `prompts.py` | 354 | Prompt format registry (frozen dataclasses) |
| `alignatt.py` | 1750+ | Core border detection + 10 aggregation + AMS + temp norm + shift-k + info gain + cumulative + combined check + LSG logit KL + source coverage + monotonicity + n-gram repetition |
| `head_transfer.py` | 310 | Cross-lingual alignment head transfer analysis + validation |
| `complexity.py` | 175 | Source complexity estimation for adaptive parameter tuning |
| `simulstream.py` | 950+ | SimulStream SpeechProcessor wrapper, longform mode, OmniSTEval output, auto sentence boundary, env-var config |
| `fusion.py` | 600 | Weighted signal fusion: 8 signals -> continuous scores -> weighted sum -> border decision |
| `calibrate.py` | 650 | Fusion weight calibration: trace collection, alignment-based labeling, grid search optimization |
| `xcomet_scorer.py` | 220 | Standalone XCOMET-XL scorer (separate process, avoids OOM) |

**Infrastructure:**
- `backend_protocol.py` (145 lines) -- SimulMTBackend ABC + `create_backend()` factory + ssbd_beta config
- `llama_backend.py` (541 lines) -- ctypes wrapper for custom llama.cpp with attention extraction API
- `alignatt_la_backend.py` (~550 lines) -- LocalAgreement + AlignAtt + SSBD hybrid backend
- 22 alignment head configs in `nllw/heads/` (HY-MT, Qwen3, Qwen3.5, EuroLLM, Tower, TranslateGemma)
- Context injection (rolling buffer of previous translations)
- 11 aggregation methods: ts_vote, softmax_mean, entropy_weighted, consensus, geomean, top_p, top_p_weighted, gaussian_kernel, gaussian_kernel_continuous, cumulative, ensemble
- SSBD (Self-Speculative Biased Decoding): previous translation as draft, batch verify, 1.3-1.7x speedup
- LA Forced Decoding: force-decode committed prefix for consistency + speed (CUNI approach)
- LA Two-Pass Catch-up: dual re-translation for output stability (lower NE)
- Adaptive Multi-Strategy (AMS): per-token aggregation selection based on attention patterns
- Per-head temperature normalization: binary search temperature scaling for fair head weighting
- Cross-lingual head transfer: validated on 5 models, EuroLLM/HY-MT/Qwen3.5 >97% TS mass transfer
- Adaptive SSBD Beta: per-token entropy-based bias modulation (confident=lenient, uncertain=strict)
- Dynamic word_batch: source-length-adaptive batching (short->wb-1, long->wb+1)
- Attention information gain: KL-divergence border modulation (inhibit/reinforce stops)
- Shift-k border: attention mass threshold in border region (DrFrattn-inspired)
- Combined border check: multi-signal fusion (standard + shift-k + info gain + entropy change + prediction stability + coverage + monotonicity)
- LSG logit KL divergence (arxiv 2501.00868): KV cache fork + probe, output logit comparison for border confirmation
- Entropy change tracking (REINA-inspired, AAAI 2026): cross-step entropy delta as border modulation
- Prediction stability index (novel): cross-step top-K prediction overlap as border modulation
- Source coverage guard (novel): attention coverage tracking for hallucination prevention
- Attention monotonicity (novel): dynamic border adjustment from attention movement patterns
- **Weighted signal fusion** (novel, iteration 12): replaces boolean cascade with continuous scoring + weighted sum
  - 8 signal scorers: standard, shift_k, info_gain, coverage, monotonicity, entropy_change, pred_stability, attn_shift
  - FusionWeights per direction, FusionDiagnostic for observability
  - calibrate_threshold() and grid_search_weights() for auto-tuning
  - Key advantage: weak signals combine (two marginal signals trigger stops neither would alone)

**Not yet built (planned):**
- `lora.py` -- LoRA adapter loading
- MCP server for editor integration

### Key parameters and their optimal values
| Parameter | Default | Notes |
|-----------|---------|-------|
| `border_distance` | 3 | Per-direction: EN-ZH=2-3, EN-DE=2-3, EN-IT=2-3, CS-EN=3. Lower bd improves quality. |
| `word_batch` | 3 | **wb=5 is best for quality** (+0.012 COMET). wb=4 is good balanced. wb=3 for low latency. |
| `context_window` | 0 | **HY-MT: context KILLS quality** (-0.084 to -0.125 COMET for EN-DE/IT/CS). Never use with HY-MT. |
| `entropy_veto_threshold` | None | **Dead end**: all thresholds hurt (0.5->0.494, 1.0->0.683). Do not use. |
| `aggregation` | "ts_vote" | **top_p is best** (COMET 0.892 vs ts_vote 0.880 at 100 sent). geomean second. 11 methods total. |
| `top_p_threshold` | 0.8 | Cumulative mass threshold for top_p. **0.85 is best for EN-ZH** (COMET 0.896=offline!). Per-direction: EN-ZH=0.85, EN-DE=0.75, EN-IT=0.9, CS-EN=0.9 |
| `adaptive_top_p` | False | Per-sentence threshold from source complexity. Simple->lower, complex->higher. All 4 dirs: -0.001/-0.002 COMET, -0.37 to -0.70 YAAL (6-12% latency reduction) |
| `dynamic_border` | False | When True, adjusts bd per-token based on attention entropy |
| `prompt_format` | "hymt" | Auto-detected from model filename |
| `ssbd_beta` | None | SSBD bias for LA backend. None=disabled, 0.0=pure speculative, 0.2=recommended |
| `la_forced_decode` | False | Force-decode committed prefix in LA backend (CUNI approach) |
| `adaptive_ssbd` | False | Entropy-based per-token SSBD beta modulation |
| `la_two_pass` | False | Run two re-translations, keep more stable output. 2x compute, lower NE. |
| `adaptive_aggregation` | False | AMS: per-token aggregation selection based on attention patterns |
| `head_temp_normalize` | False | Normalize attention sharpness per head before aggregation |
| `head_temp_reference` | 1.5 | Reference entropy (nats) for temperature normalization |
| `dynamic_word_batch` | False | Adjust wb by source length: short->wb-1, long->wb+1 |
| `info_gain_threshold` | None | KL-divergence threshold for border modulation. None=disabled, 0.3=recommended |
| `shift_k_threshold` | None | Attention mass threshold for border stop. None=disabled, 0.4=recommended |
| `border_confirm` | 1 | Require N consecutive border hits. 1=disabled, 2=recommended |
| `lsg_kl_threshold` | None | LSG logit KL (arxiv 2501.00868). None=disabled, 7.0=recommended for 7B |
| `lsg_k` | 3 | Number of source tokens to remove for LSG probe (1-5) |
| `complexity_adaptive` | False | Per-sentence adaptive bd/wb/gen from text complexity features |
| `entropy_change_threshold` | None | REINA entropy change (AAAI 2026). None=disabled, -0.5=recommended |
| `prediction_stability` | False | Cross-step prediction stability modulation. Novel signal |
| `coverage_threshold` | None | Source coverage guard. None=disabled, 0.3=recommended. Hallucination prevention |
| `attention_monotonicity` | False | Attention monotonicity-based border adjustment. Novel signal |
| `repetition_max_repeats` | None | N-gram repetition halt. None=disabled, 2=recommended. Hallucination prevention |
| `attention_shift` | False | Cross-step attention position shift tracking. Novel signal |
| `perplexity_adaptive_bd` | False | Hibiki-inspired: adjust bd from generation confidence. Low ppl=tighten, high ppl=widen |
| `perplexity_bd_low` | 2.0 | Below this ppl -> bd-1 (confident, lower latency) |
| `perplexity_bd_high` | 5.0 | Above this ppl -> bd+1 (uncertain, safer) |
| `source_aware_batching` | False | Defer translate() when batch ends on function word (the, of, in...). Novel |
| `signal_fusion` | False | Weighted signal fusion mode (replaces boolean cascade). Novel |
| `fusion_threshold` | 0.0 | Fusion decision threshold. 0.0=balanced, positive=conservative |
| `gen_cap` | adaptive | `n_src` (short) or `n_src*1.5` (long) |
| `min_commit` | `n_words//4` | Guarantees progress per translate() call |

### Critical bugs fixed (don't revert these!)
1. **logit_idx**: After `decode_single`, `llama_get_logits_ith` expects batch index 0, not absolute KV position. Track `logit_idx` separately.
2. **Border threshold guard**: `n_src - border_distance` can go negative. Guard with `border_threshold > 0`.
3. **Thread safety**: `threading.Lock` around all llama_context operations.
4. **Stderr suppression**: Metal JIT logs flood TUI. Wrap decode calls with `suppress/restore_stderr`.

### Quality metrics (2026-03-21, FLORES, A40, COMET wmt22-comet-da)

**100-sentence confirmed results (iteration 18, tuned p_threshold with CI):**
| Direction | bd | wb | p | BLEU | COMET | 95% CI | YAAL | % of offline |
|-----------|---:|---:|:-:|-----:|------:|--------|-----:|:------------:|
| **EN-ZH** | 3 | 4 | 0.85 | 40.0 | **0.894** | [0.887, 0.901] | 6.09 | 99.8% |
| **EN-DE** | 2 | 3 | 0.75 | 27.9 | **0.881** | [0.873, 0.890] | 5.45 | 99.7% |
| **EN-IT** | 2 | 3 | 0.9 | 24.3 | **0.891** | [0.882, 0.899] | 6.76 | **100.2%** |
| **CS-EN** | 3 | 3 | 0.9 | 28.4 | **0.879** | [0.871, 0.886] | 5.81 | 99.8% |

EN-IT **exceeds offline baseline** (0.891 > 0.889). All directions within 0.3% of offline.
Full-sentence baselines: EN-ZH=0.896, EN-DE=0.884, EN-IT=0.889, CS-EN=0.881.

### Key findings
- **top_p aggregation is the biggest win**: +0.008-0.021 COMET over ts_vote
- **wb=4-5 improves quality**: COMET 0.892 EN-ZH at wb=4 (+0.012 over wb=3)
- **Lower bd improves quality**: bd=2 better than bd=3 for EN-DE/EN-IT
- **HY-MT1.5-7B is best model**: beats all other tested models
- **AlignAtt is critical**: Without it, COMET collapses 0.87 -> 0.29-0.47
- **KV cache reuse**: 3-5x speedup, zero quality loss
- **Context KILLS HY-MT quality**: -0.084 to -0.125 COMET (dead end)
- **Entropy veto is dead**: all thresholds hurt quality significantly
- **Adaptive top_p trades latency for free**: 6-12% YAAL reduction, <0.2% COMET cost (CIs overlap)
- **Bootstrap CI enables rigorous comparison**: 95% CIs show most differences are not significant
- **20+ failed experiments documented**: EAST, LoRA, GDN, confidence, signals, etc.
- **XCOMET-XL amplifies differences 39x vs wmt22** -- use the right metric!
- **IWSLT 2026 uses COMET wmt22-comet-da** for ranking (confirmed from shared task page)

---

## Machines

| Machine | SSH | GPU | Used for |
|---------|-----|-----|----------|
| **A40** (always up) | `ssh -p 3622 fuxa@quest.ms.mff.cuni.cz` (key auth) | A40 46GB | llama.cpp MT, head detection |
| **H100_1** ($3/hr) | JarvisLab `jl instance resume <id>` | H100 80GB | Heavy eval, xCOMET-XL |
| **L4_1** ($0.5/hr) | JarvisLab | L4 | Cheap experiments, LoRA |
| **L4_2** ($0.5/hr) | JarvisLab | L4 | Parallel experiments |
| **MacBook M5** | local | Metal | Research, GGUF experiments |

JarvisLab: `jl instance list`, SSH key at `/Users/quentin/Documents/repos/jarvis/id_ed25519_jarvis`

---

## Repos
- **NLLW** (this repo): Clean SimulMT research platform
- **iwslt26-sst**: `/Users/quentin/Documents/repos/iwslt26-sst/` — messy experiment repo (knowledge extracted into docs/research/)
- **WhisperLiveKit**: `/Users/quentin/Documents/repos/WhisperLiveKit` — ASR streaming

---

## Running the tools

```bash
# Start web server
HYMT_MODEL_PATH=/path/to/model.gguf python web_debug/server.py

# --- Benchmarking (preferred entry point) ---
# Basic benchmark (requires web server on :8777)
python -m nllw.bench --lang en-fr

# Full corpus with COMET, save to registry
python -m nllw.bench --suite corpus --lang en-fr --comet --save

# Compare backends head-to-head
python -m nllw.bench --compare alignatt alignatt-la --lang en-fr --comet --save

# Parameter sweep
python -m nllw.bench --sweep "bd=2,3,4 wb=2,3" --lang en-fr --comet --save

# Multi-direction sweep
python -m nllw.bench --sweep "bd=2,3,4 wb=1,2,3" --lang en-zh,en-de,en-it --comet --save

# Export to OmniSTEval format (IWSLT submission)
python -m nllw.bench --suite corpus --lang en-zh --omnisteval output.jsonl --save

# SSBD speedup test for LA backend
python -m nllw.bench --backend alignatt-la --ssbd-beta 0.2 --lang en-zh --comet --save

# SSBD sweep
python -m nllw.bench --backend alignatt-la --sweep "ssbd=0.0,0.1,0.2,0.3" --lang en-zh --comet --save

# --- Other tools ---
# Run evaluation (lower-level)
python -m nllw.eval --backend web --lang en-fr --comet

# Run benchmark suite (lower-level)
python -m nllw.research --suite flores_mini --configs "bd=2,3,4 wb=2,3"

# Run experiment from config
python -m nllw.experiment run config.yaml

# Detect heads for a new model
python -m nllw.detect_heads --model /path/to/model.gguf --lang en-fr

# Compare backends (single sentence)
python -m nllw.simulate "the president of france announced reforms" --configs '{"backend_type":"alignatt"}' '{"backend_type":"alignatt-la"}'

# Convert traces to OmniSTEval JSONL
python -m nllw.omnisteval traces.json --talk-id demo --source-length 120.5 -o output.jsonl

# Signal fusion mode (replaces boolean cascade with weighted scoring)
python -m nllw.bench --signal-fusion --lang en-zh --comet --save
python -m nllw.bench --signal-fusion --shift-k 0.4 --coverage-threshold 0.3 --lang en-zh --comet --save
python -m nllw.bench --signal-fusion --sweep "fthr=-0.2,0.0,0.2,0.4" --lang en-zh --comet --save

# Automated experiment runner (GPU machines)
./scripts/run_experiments.sh 1 --lang en-zh --model /path/to/model.gguf --comet
```
