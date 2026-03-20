---
date: 2026-03-20
topic: alignatt-simulmt-architecture
focus: Rebuild NLLW as the clean production version of iwslt26-sst experiments
---

# Ideation: AlignAtt + SimulMT Architecture Rebuild

## Codebase Context

**NLLW** is a Python library for real-time simultaneous translation. Rebuilt today from a broken AlignAtt backend into a full SimulMT research platform.

**Current state (end of session):**
- 3 AlignAtt backends: original (entropy veto), LocalAgreement hybrid, KV cache delta
- 3 baseline backends being added: wait-k, full-sentence, eager
- Research tools: eval.py, simulate.py, research.py, detect_heads.py, metrics.py (COMET/xCOMET)
- 23 alignment head configs (HY-MT, Qwen3, EuroLLM, Tower, TranslateGemma)
- Ollama-style web UI with backend selector, param sliders, /compare, /evaluate endpoints
- Being implemented: Qwen3/3.5 prompt support, LoRA adapters, COMET-XL metrics, baselines
- EN→FR BLEU ~11 (wb=3), FR→EN BLEU ~71

## Round 1 Ideas (ALL IMPLEMENTED)

### 1. Evaluation Pipeline ✅ Implemented
### 2. KV Cache Prefix Reuse ✅ Implemented
### 3. Policy Simulator ✅ Implemented
### 4. Unified Backend Protocol ✅ Implemented
### 5. Automatic Head Detection ✅ Implemented
### 6. Adaptive Border Distance ✅ Implemented
### 7. Entropy-Based Border Gating ✅ Implemented

## Round 2 Ideas (NEW — next priorities)

### 1. `nllw` CLI Tool — serve/translate/bench/heads
**Description:** Single unified CLI entry point mirroring ollama. `nllw serve --model X --port 8777` starts server. `nllw translate < text.txt` streams translation. `nllw bench --suite flores_mini --configs "bd=2,3,4"` runs benchmarks. `nllw heads detect --model X --lang en-fr` detects heads. Researchers use NLLW without writing Python.
**Rationale:** Transforms NLLW from a library into a standalone tool. 10x broader adoption. Matches ollama/whisper.cpp/ctranslate2 UX patterns.
**Downsides:** Packaging effort, need to maintain CLI compatibility alongside API.
**Confidence:** 95%
**Complexity:** Low-Medium
**Status:** Unexplored

### 2. Cross-Sentence Context Injection
**Description:** Rolling buffer of last N committed translations injected into the next sentence's prompt. Format: `[Previous translations]\n{ctx}\n[New sentence]\n{source}`. Configurable window size (0-5 sentences) and token budget.
**Rationale:** Each sentence currently translated cold — pronouns, entities, terminology break across utterances. iwslt26-sst showed +0.037 COMET with context for Qwen3.5.
**Downsides:** Increases prompt length, may slow KV cache. Some models (HY-MT) showed -0.028 with context.
**Confidence:** 90%
**Complexity:** Medium
**Status:** Unexplored

### 3. IWSLT Corpus Loader + OmniSTEval Output
**Description:** Load FLORES+ devtest via HuggingFace datasets, IWSLT talk segments from JSONL. Run full eval, export to OmniSTEval submission format. One-click standardized benchmarking.
**Rationale:** Removes friction of manual dataset wrangling. Enables direct comparison against published SimulMT systems.
**Downsides:** datasets dependency. OmniSTEval format may change across IWSLT editions.
**Confidence:** 90%
**Complexity:** Medium
**Status:** Unexplored

### 4. Experiment Registry + Reproducible Run Format
**Description:** Structured YAML/JSON experiment schema capturing all variables. `nllw run config.yaml` executes full eval + saves dated results artifact. Growing results database enables auto-comparison.
**Rationale:** iwslt26-sst's 40+ experiment dirs were ad-hoc. This makes every experiment a first-class citizen. Papers become one command away.
**Downsides:** Schema design needs care. Storage grows over time.
**Confidence:** 85%
**Complexity:** Medium
**Status:** Unexplored

### 5. WebSocket Streaming + WhisperLiveKit Cascade
**Description:** `/ws/translate` WebSocket endpoint for real-time word streaming. WhisperLiveKit produces ASR words over WS, NLLW consumes and translates incrementally. End-to-end cascade.
**Rationale:** Current HTTP polling adds latency. WebSocket enables true real-time. Prerequisite for live conference deployment.
**Downsides:** Concurrency complexity. Need to handle backpressure when model is slower than speech.
**Confidence:** 85%
**Complexity:** Medium-High
**Status:** Unexplored

### 6. Side-by-Side A/B Backend Comparison in UI
**Description:** Split-pane view rendering /compare results. Each column shows word-by-word replay with stable/buffer coloring. Diff highlighting marks divergences. Summary row with BLEU/COMET delta.
**Rationale:** /compare endpoint exists but returns raw JSON. Researchers need visual comparison without parsing.
**Downsides:** UI complexity. May need WebSocket for live comparison.
**Confidence:** 85%
**Complexity:** Medium
**Status:** Unexplored

### 7. AL/DAL Standard Latency Metrics + CI Quality Gate
**Description:** Average Lagging and Differentiable Average Lagging in eval pipeline. GitHub Actions workflow that runs FLORES_MINI on PRs touching backend code, fails if BLEU/AL regresses.
**Rationale:** AL is the standard SimulMT latency metric used in all IWSLT papers. Without it, results aren't comparable to prior art. CI gate prevents silent quality regressions.
**Downsides:** CI requires GPU or CPU fallback. AL computation needs careful timestamp handling.
**Confidence:** 90%
**Complexity:** Low-Medium
**Status:** Unexplored

## Round 2 Rejection Summary

| # | Idea | Reason Rejected |
|---|------|-----------------|
| 1 | Committed-ratio auto-tuner | Premature — need more eval data first |
| 2 | Model loading SSE | Nice UX but low research impact |
| 3 | Attention debug panel | Partially available via verbose; low priority |
| 4 | BLEU regression guard | Subsumed by CI quality gate |
| 5 | Config snapshots | Subsumed by experiment registry |
| 6 | Live plot dashboard | High effort, premature without WebSocket |
| 7 | Clause-gated emission | Adds NLP dependency; research-only |
| 8 | Draft-then-verify ensemble | Too complex for current stage |
| 9 | Learned RL emission policy | Research paper, not near-term |
| 10 | Probabilistic emission | UI experiment, not core architecture |
| 11 | Human-in-the-loop correction | Very high effort, premature |
| 12 | Head registry + quality index | Subsumed by experiment registry |
| 13 | Parallel batch eval | Engineering, not architectural |
| 14 | Multi-model comparison view | Duplicates A/B view |
| 15 | Persistent SQLite store | Subsumed by experiment registry |
| 16 | Auto figure generation | Premature without experiment registry |
| 17 | Glossary/terminology injection | Useful but separate concern |
| 18 | LoRA hot-swap pipeline | Being implemented separately |

## Session Log
- 2026-03-20 AM: Initial ideation — 48 candidates generated, 22 unique, 7 survived
- 2026-03-20 AM-PM: All 7 Round 1 ideas implemented (eval, KV cache, simulator, protocol, heads, adaptive border, entropy)
- 2026-03-20 PM: Additional implementations: LocalAgreement hybrid, research.py, detect_heads.py, Ollama UI, COMET metrics, Qwen3 support, LoRA, baselines
- 2026-03-20 PM: Round 2 ideation — 32 candidates generated (5 agents), 7 survived
