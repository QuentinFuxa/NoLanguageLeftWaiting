# Shared Task Notes -- NLLW SimulMT

## What exists now (after Iteration 1, 2026-03-20)

**10 new modules (~3000 lines), 62 tests passing:**
- `nllw/prompts.py` -- 30+ prompt formats (HY-MT, Qwen3, Qwen3.5, EuroLLM, Tower, Gemma)
- `nllw/llama_backend.py` -- ctypes wrapper for llama.cpp (attention extraction + KV cache)
- `nllw/backend_protocol.py` -- SimulMTBackend ABC + factory + `@register_backend`
- `nllw/alignatt.py` -- Core algorithm (TS-weighted vote, border detection, entropy, lookahead)
- `nllw/alignatt_backend.py` -- Full backend with KV cache reuse + baselines (full-sentence, eager)
- `nllw/metrics.py` -- AL, LAAL, YAAL, AP, DAL, MaxCW + BLEU/COMET wrappers
- `nllw/simulate.py` -- Policy simulation and trace replay
- `nllw/eval.py` -- Evaluation harness (FLORES+, parameter sweep, XCOMET-XL)
- `nllw/bench.py` -- Unified CLI (`python -m nllw.bench --lang en-fr --comet`)
- `nllw/corpus.py` -- Categorized test corpus (38 sentences, 7 categories)
- `nllw/heads/configs/` -- 22 pre-computed alignment head configs

## What to do next

1. **Run first end-to-end experiment on A40** -- Test the AlignAtt backend with HY-MT1.5-7B on FLORES EN-ZH. This validates the whole pipeline. Command: `python -m nllw.bench --model /path/to/HY-MT1.5-7B.gguf --lang en-zh -n 20`

2. **Build `nllw/detect_heads.py`** -- Port head detection from `iwslt26-sst/heads/detect_translation_heads_llamacpp.py`. Needed for testing new models.

3. **Build `nllw/omnisteval.py`** -- OmniSTEval JSONL export for IWSLT 2026 submission.

4. **Expand corpus.py** -- Add EN-DE, EN-IT, CS-EN sentences. Target 130+ sentences.

5. **Research: better aggregation** -- Current TS-weighted argmax is effective but simple. Worth testing: softmax over source positions, attention entropy weighting, multi-head consensus.

## Key architecture decisions

- `PromptFormat` is a frozen dataclass with `build_prompt()` method
- `BackendConfig.from_dict()` ignores unknown keys (safe for sweep configs)
- `@register_backend("name")` decorator auto-registers with factory
- `simulate_backend()` feeds words one at a time, records delays
- FLORES+ uses `openlanguagedata/flores_plus` with per-language configs (not pairs)

## Important context

- llama.cpp must be built with PR #20086. The build is on A40.
- The logit_idx bug (batch index 0, not KV pos) is fixed in `alignatt_backend.py`.
- YAAL formula: `gamma = max(|delays|, T) / S; yaal = sum(d - t/gamma) / tau`
- FLORES loading is tested and working locally.
