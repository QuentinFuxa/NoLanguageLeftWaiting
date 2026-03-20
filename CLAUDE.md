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
- DO NOT COMMIT NOR PUSH EXCEPT IF EXPLICITLY ASKED.
- Always commit in user's name, not yours. No co-authored-by.
- No em dash in commit messages.
- Commit dates MUST be weekends (Saturday or Sunday).

---

## Project State (2026-03-20)

### What exists now: ~15,300 lines across 21 modules

**6 translation backends:**
| Backend | Type | File | Purpose |
|---------|------|------|---------|
| `alignatt` | Primary | `nllw/alignatt_backend.py` (856 lines) | Attention-based border detection + entropy veto + context injection |
| `alignatt-la` | Quality | `nllw/alignatt_la_backend.py` (468 lines) | LocalAgreement hybrid (re-translate + diff for stability) |
| `alignatt-kv` | Speed | `nllw/alignatt_kvcache_backend.py` (697 lines) | KV cache delta decoding (5-10x faster) |
| `wait-k` | Baseline | `nllw/baselines.py` (654 lines) | Standard wait-k policy baseline |
| `full-sentence` | Baseline | `nllw/baselines.py` | Quality upper bound (offline) |
| `eager` | Baseline | `nllw/baselines.py` | Latency lower bound |

**6 prompt formats:** hymt, qwen3, qwen3.5, qwen3-nothink, eurollm, custom

**11 research tools:**
| Module | Lines | Purpose |
|--------|-------|---------|
| `eval.py` | 1006 | BLEU/COMET/xCOMET-XL evaluation, parameter sweep |
| `research.py` | 1026 | Compute-aware latency (CA-AL), benchmark suite |
| `simulate.py` | 539 | Policy replay, Average Lagging computation |
| `corpus.py` | 1614 | 130-sentence categorized test corpus |
| `experiment.py` | 1227 | Experiment config/result registry, Pareto analysis |
| `analysis.py` | 1405 | Pareto frontier, edge cases, report generation |
| `detect_heads.py` | 1030 | Auto alignment head detection for any GGUF model |
| `metrics.py` | 320 | BLEU, COMET, xCOMET-XL wrappers |
| `lora.py` | 164 | LoRA adapter loading + discovery |
| `bench.py` | ~440 | Unified one-command benchmarking CLI with sweep, compare, OmniSTEval |
| `omnisteval.py` | ~300 | OmniSTEval JSONL output format for IWSLT submission |

**Infrastructure:**
- `backend_protocol.py` — SimulMTBackend ABC + `create_backend()` factory
- `llama_backend.py` — ctypes wrapper for custom llama.cpp with attention extraction API
- 23 alignment head configs in `nllw/heads/` (HY-MT, Qwen3, Qwen3.5, EuroLLM, Tower, TranslateGemma)
- 9 experiment configs in `configs/` (alignatt-vs-la, per-direction, entropy, context, kv-cache, sweep)
- Web debug: FastAPI server + Ollama-style UI + MCP server
- LoRA C API bindings (load/apply/clear adapters)
- Context injection (rolling buffer of previous translations)

### Key parameters and their optimal values
| Parameter | Default | Notes |
|-----------|---------|-------|
| `border_distance` | 3 | Per-direction: EN-ZH=3, EN-IT=4, CS-EN=2 |
| `word_batch` | 3 | wb=2 hallucinates on some inputs; wb=3 safer |
| `context_window` | 0 | HY-MT: context hurts (-0.028). Qwen3.5: helps (+0.037) |
| `entropy_veto_threshold` | None | Optional, 0.75 recommended. Catches uncertain tokens. |
| `prompt_format` | "hymt" | Auto-detected from model filename |
| `gen_cap` | adaptive | `n_src` (short) or `n_src*1.5` (long) |
| `min_commit` | `n_words//4` | Guarantees progress per translate() call |

### Critical bugs fixed (don't revert these!)
1. **logit_idx**: After `decode_single`, `llama_get_logits_ith` expects batch index 0, not absolute KV position. Track `logit_idx` separately.
2. **Border threshold guard**: `n_src - border_distance` can go negative. Guard with `border_threshold > 0`.
3. **Thread safety**: `threading.Lock` around all llama_context operations.
4. **Stderr suppression**: Metal JIT logs flood TUI. Wrap decode calls with `suppress/restore_stderr`.

### Quality metrics (2026-03-20 baseline)
| Direction | BLEU | COMET (est) | Committed% |
|-----------|------|-------------|------------|
| EN→FR | ~11 | ~0.75 | 63% |
| FR→EN | ~71 | ~0.85 | 77% |

### Key findings from iwslt26-sst (see docs/research/iwslt26-sst-findings.md)
- **HY-MT1.5-7B champion**: 0.842 XCOMET-XL EN-ZH, beats Qwen3.5 by +0.039
- **AlignAtt is critical**: Without it, COMET collapses 0.87 → 0.29-0.47
- **KV cache reuse**: 3-5x speedup, zero quality loss
- **15+ failed experiments documented**: EAST, LoRA no-think, GDN warm-start, confidence stopping, etc.
- **Context helps Qwen3.5 (+0.037) but hurts HY-MT (-0.028)**
- **XCOMET-XL amplifies differences 39x vs wmt22** — use the right metric!

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
```
