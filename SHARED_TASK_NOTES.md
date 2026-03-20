# Shared Task Notes -- Iteration 3

## What was done this iteration

- Cherry-picked iteration-2 AlignAtt infrastructure onto this branch
- **YAAL metric** (IWSLT 2026 primary): Implemented in `nllw/metrics.py` with word-count
  and time-domain (ms) versions. Formula verified against OmniSTEval `YAALScorer`.
  Also added DAL (Differentiable Average Lagging).
- **XCOMET-XL scoring**: `score_with_comet()` in `nllw/eval.py` + CLI `--comet` flag
- **Parameter sweep**: `run_sweep()` in `nllw/eval.py` + CLI `--sweep --sweep-bd --sweep-wb`
- 56 tests, all passing (was 38)

## What to do next (priority order)

1. **GPU integration tests** on A40/L4:
   ```bash
   python -m nllw.eval --model /path/to/HY-MT1.5-7B-Q8_0.gguf \
       --heads /path/to/heads.json --prompt-format hymt \
       --lang en-zh -n 50 --comet Unbabel/XCOMET-XL
   ```

2. **Benchmark vs iwslt26-sst** -- Same model/config on both codebases -> verify identical output

3. **Pre-computed head configs** -- Copy from iwslt26-sst or generate fresh:
   ```bash
   python -m nllw.heads.detect --model HY-MT1.5-7B-Q8_0.gguf --lang en-zh -n 50
   ```

4. **Interactive translator CLI** -- Port from `iwslt26-sst/SimulMT_tests/interactive_translator.py`

5. **Cascade pipeline** -- Port ASR+MT from `iwslt26-sst/SimulMT_tests/cascade/`

## Key files changed this iteration

- `nllw/metrics.py` -- YAAL, DAL, compute_yaal_ms()
- `nllw/eval.py` -- score_with_comet(), run_sweep(), updated summary/CLI
- `tests/test_metrics.py` -- 20 tests (was 9)
- `tests/test_eval.py` -- 7 new tests
- `pyproject.toml` -- COMET version bump
- `todo.md`, `CLAUDE.md` -- Updated

## Known issues

- C shim compilation requires a C compiler + llama.cpp source headers
- `memory_seq_rm_attn_only` only available in patched llama.cpp builds
- Head detection requires `simalign` which needs `torch`
