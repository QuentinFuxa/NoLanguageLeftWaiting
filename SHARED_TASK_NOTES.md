# Shared Task Notes -- Next Iteration

## What was done this iteration

Ported the entire AlignAtt SimulMT infrastructure from iwslt26-sst to NLLW as a clean, modular Python package:

- `nllw/llama_backend.py` -- llama.cpp ctypes bindings (LlamaModel, LlamaContext, LlamaBackend)
- `nllw/alignatt.py` -- AlignAtt border detection engine with KV cache reuse
- `nllw/prompts.py` -- 30+ prompt format templates for HY-MT, Qwen, EuroLLM, etc.
- `nllw/metrics.py` -- SimulMT latency metrics (AL, LAAL, AP, MaxCW)
- `nllw/policies.py` -- 5 read/write policies (AlignAtt, WaitK, Confidence, FixedRate, NoBorder)
- `nllw/eval.py` -- FLORES evaluation harness (`python -m nllw.eval`)
- `nllw/heads/detect.py` -- Alignment head detection (`python -m nllw.heads.detect`)
- 38 unit tests, all passing
- CLAUDE.md + todo.md with full project documentation

## What to do next (priority order)

1. **GPU integration tests** -- Run on A40 or L4 machine:
   ```bash
   # First, build llama.cpp with attn weight extraction (PR #20086)
   # Then test with HY-MT1.5-7B:
   python -m nllw.heads.detect --model /path/to/HY-MT1.5-7B-Q8_0.gguf --prompt-format hymt --lang en-zh -n 50
   python -m nllw.eval --model /path/to/HY-MT1.5-7B-Q8_0.gguf --heads translation_heads.json --prompt-format hymt --lang en-zh -n 50 --border-distance 3
   ```

2. **Benchmark vs iwslt26-sst** -- Run same model/config on both codebases, verify identical output.

3. **Copy pre-computed head configs** from iwslt26-sst to `nllw/heads/configs/`:
   ```bash
   cp /path/to/iwslt26-sst/SimulMT_tests/heads/translation_heads_*.json nllw/heads/configs/
   ```

4. **Parameter sweep** -- Add `--sweep` mode to eval.py that tests combinations of bd/wb.

5. **Interactive translator CLI** -- Port from iwslt26-sst for live keyboard-driven translation.

## Key architectural decisions

- **OOP design**: LlamaModel -> LlamaContext (vs flat functions in iwslt26-sst)
- **No sys.path hacks**: Proper Python package imports
- **Lazy lib loading**: llama.cpp ctypes only loaded when LLM features used (NLLB users unaffected)
- **C shim compilation**: Context creation uses compiled-on-the-fly C shim to avoid struct layout issues

## Research findings (from SOTA agent)

- **SimulStreaming (UFAL) won IWSLT 2025** using AlignAtt + EuroLLM cascade -- validates our approach
- **EAST (ICLR 2025)**: Append-only interleaving with READ/WRITE tokens = fully reusable KV cache
- **SimulMEGA (NeurIPS 2025)**: MoE router as implicit R/W policy, zero overhead
- **LongYAAL** is the IWSLT 2026 primary latency metric (not StreamLAAL) -- needs implementation
- **IWSLT 2026 baselines**: Qwen3-ASR + Qwen3-4B cascade (https://github.com/owaski/iwslt-2026-baselines)
- Consider exploring EAST-style policy as alternative/complement to AlignAtt

## Known issues

- C shim compilation requires a C compiler + llama.cpp source headers
- `memory_seq_rm_attn_only` only available in patched llama.cpp builds
- Head detection requires `simalign` which needs `torch` (already a dependency)
