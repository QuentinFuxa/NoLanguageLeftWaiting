# Shared Task Notes -- NLLW SimulMT

## What exists now (after Iteration 2, 2026-03-20)

**17 SimulMT modules (~5400 lines), 115 tests passing:**

### Core (Iteration 1):
- `nllw/prompts.py` -- 30+ prompt formats (HY-MT, Qwen3, Qwen3.5, EuroLLM, Tower, Gemma)
- `nllw/llama_backend.py` -- ctypes wrapper for llama.cpp (attention extraction + KV cache)
- `nllw/backend_protocol.py` -- SimulMTBackend ABC + factory + `@register_backend`
- `nllw/alignatt.py` -- Core algorithm (TS-weighted vote, border detection, entropy, lookahead)
- `nllw/alignatt_backend.py` -- Full backend with KV cache reuse + baselines (full-sentence, eager)
- `nllw/metrics.py` -- AL, LAAL, YAAL, AP, DAL, MaxCW + BLEU/COMET wrappers
- `nllw/simulate.py` -- Policy simulation and trace replay
- `nllw/eval.py` -- Evaluation harness (FLORES+, parameter sweep, XCOMET-XL)
- `nllw/bench.py` -- Unified CLI (`python -m nllw.bench --lang en-fr --comet`)
- `nllw/heads/configs/` -- 22 pre-computed alignment head configs

### New (Iteration 2):
- `nllw/detect_heads.py` -- Auto alignment head detection for any GGUF model (559 lines)
- `nllw/omnisteval.py` -- OmniSTEval JSONL export for IWSLT 2026 (258 lines)
- `nllw/baselines.py` -- wait-k + fixed-rate baselines (175 lines)
- `nllw/analysis.py` -- Pareto frontier, edge cases, report generation (309 lines)
- `nllw/experiment.py` -- Experiment YAML config + file-based result registry (359 lines)
- `nllw/corpus.py` -- Expanded: 120 sentences across 5 directions (622 lines)
- `nllw/research.py` -- Compute-aware latency (CA-AL, CA-YAAL), benchmark suite (191 lines)
- 4 experiment configs in `configs/` (baseline, sweep, compare, entropy)

## What to do next

1. **Run first end-to-end experiment on A40** -- Validate the entire pipeline:
   ```bash
   python -m nllw.bench --model /path/to/HY-MT1.5-7B.gguf --lang en-zh -n 20
   ```

2. **Build `nllw/alignatt_la_backend.py`** -- LocalAgreement hybrid for output stability. Key idea: re-translate from scratch each time, diff against previous, only commit stable tokens.

3. **Build `nllw/research.py`** -- Compute-aware latency (CA-AL): include inference time in delay measurements. Needed for realistic IWSLT evaluation.

4. **Run Pareto sweep** -- Once E2E works, run: `python -m nllw.bench --sweep "bd=2,3,4,5 wb=1,2,3" --lang en-zh,en-de --comet --save`

5. **OmniSTEval pipeline** -- Connect bench output directly to OmniSTEval JSONL for IWSLT submission validation.

## Key architecture decisions

- `PromptFormat` is a frozen dataclass with `build_prompt()` method
- `BackendConfig.from_dict()` ignores unknown keys (safe for sweep configs)
- `@register_backend("name")` decorator auto-registers with factory
- `simulate_backend()` feeds words one at a time, records delays
- FLORES+ uses `openlanguagedata/flores_plus` with per-language configs (not pairs)
- `detect_heads` uses SimAlign (mBERT) for ground-truth word alignments
- `ExperimentConfig` supports both flat dicts and nested YAML format
- `ParetoAnalyzer` computes quality-latency frontiers across experiments

## Important context

- llama.cpp must be built with PR #20086. The build is on A40.
- The logit_idx bug (batch index 0, not KV pos) is fixed in `alignatt_backend.py`.
- YAAL formula: `gamma = max(|delays|, T) / S; yaal = sum(d - t/gamma) / tau`
- FLORES loading is tested and working locally.
- 5 directions available: en-fr, en-zh, en-de, en-it, cs-en
- 6 backends registered: alignatt, full-sentence, eager, wait-k, fixed-rate (+ alignatt-la planned)
- SOTA research documented in `docs/research/sota-simulmt-2026.md`

## Key research findings (from SOTA survey)

- **AlignAtt validated**: CUNI won IWSLT 2025 using AlignAtt. ICLR 2026 "Translation Heads" paper confirms alignment heads are sparse, universal, consistent.
- **SSBD** (arxiv 2509.21740) can accelerate re-translation (alignatt-la) via speculative draft reuse
- **ExPosST** (arxiv 2603.14903) pre-allocates position slots for zero-recomputation KV cache
- **Open gap**: No published work on alternative attention aggregation (softmax, entropy-weighted voting) -- novel contribution opportunity
- **Hikari** (arxiv 2603.11578) is the main competitor: policy-free WAIT tokens, SOTA on EN-JA/DE/RU
- **IWSLT 2026 metrics**: LongYAAL (primary latency), XCOMET-XL (primary quality) -- our pipeline is aligned
