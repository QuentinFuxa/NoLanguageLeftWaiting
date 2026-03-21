# Shared Task Notes -- NLLW SimulMT

## Current State (after Iteration 13, 2026-03-21)

**28 SimulMT modules (~13,000 lines), 731 tests passing**

### Module Overview
- **Core**: prompts.py, llama_backend.py, backend_protocol.py, alignatt.py (~1750 lines), alignatt_backend.py, alignatt_la_backend.py
- **Baselines**: baselines.py (wait-k, fixed-rate), full-sentence, eager in alignatt_backend.py
- **Research tools**: eval.py, bench.py, simulate.py, corpus.py, experiment.py, analysis.py, research.py, detect_heads.py, omnisteval.py, head_transfer.py, complexity.py, metrics.py
- **Novel frameworks**: fusion.py (8-signal weighted fusion), calibrate.py (data-driven weight optimization)
- **IWSLT 2026**: simulstream.py (SpeechProcessor wrapper), Dockerfile, per-direction YAML configs
- **Infrastructure**: web_debug/server.py (FastAPI), scripts/run_experiments.sh (GPU automation)
- **Head configs**: 22 pre-computed configs in nllw/heads/configs/

### 8 Border Detection Signals (Complete Taxonomy)
| Signal | Type | Module |
|--------|------|--------|
| Standard AlignAtt | Within-step, position | alignatt.py |
| Shift-k mass | Within-step, position | alignatt.py |
| Info gain (KL) | Within-step, position | alignatt.py |
| Source coverage | Within-step, coverage | alignatt.py |
| Attention monotonicity | Within-step, temporal | alignatt.py |
| N-gram repetition | Within-step, output | alignatt.py |
| Entropy change (REINA) | Cross-step, output | alignatt.py |
| Prediction stability | Cross-step, output | alignatt.py |
| Attention shift | Cross-step, input | alignatt.py |
| LSG logit KL | Cross-step, logit | alignatt_backend.py |

### New in Iteration 13
- **Fusion weight calibration** (`nllw/calibrate.py`, 1040 lines):
  - TraceCollector for recording signal scores during translation
  - Alignment-based + quality-based border labeling
  - Grid search weight optimization per direction
  - Signal importance analysis (discriminative power + correlation)
  - CLI: `python -m nllw.calibrate --demo` or `--traces FILE --analyze`
  - `--collect-traces FILE` in bench.py for GPU trace collection
  - `--calibrate-traces FILE` in bench.py for running calibration
- **OmniSTEval bug fix**: `eval_result_to_omnisteval()` now returns all sentences (was only returning last)
- **OmniSTEval format rewrite** (CRITICAL): Old format was per-emission-event (wrong!). New `SimulEvalEntry` + `eval_result_to_simuleval()` produces correct per-segment JSONL with `prediction`, `delays[]` (ms), `elapsed[]` (ms), `source_length` (ms). Validated against OmniSTEval v0.1.6 schema.
- **Trace collection in backend**: `set_trace_collector()` on SimulMTBackend, wired into fusion border check

## What to do next

### Priority 1: GPU Testing (NOTHING has been tested on real GPU yet!)

All code (iterations 1-13) is unit-tested but never run with an actual model. This is the #1 blocker.

```bash
# Step 1: Basic E2E validation
python -m nllw.bench --model /path/to/HY-MT1.5-7B.gguf --lang en-zh -n 20

# Step 2: Collect traces for calibration
python -m nllw.bench --signal-fusion --shift-k 0.4 --coverage-threshold 0.3 \
  --lang en-zh --comet --save --collect-traces traces_enzh.json

# Step 3: Calibrate fusion weights from real traces
python -m nllw.bench --calibrate-traces traces_enzh.json --lang en-zh \
  --calibrate-output weights_enzh.json

# Step 4: Benchmark with calibrated weights vs defaults
# (use the exported weights in experiment configs)

# OR use the full experiment runner:
./scripts/run_experiments.sh 1 --lang en-zh --model /path/to/model.gguf --comet
```

### Priority 2: IWSLT 2026 Competition (Eval April 1-15)
- **Docker packaging**: H100 80GB, Q8_0 7B GGUF = ~8GB VRAM
- **SimulStream E2E test**: Wire ASR into NLLWSpeechProcessor
- **OmniSTEval verification**: Test output with official `omnisteval longform` + XCOMET-XL
- **Extra Context subtrack**: Terminology extraction from ACL PDFs (RASST-style, +3 BLEU)
- **Per-direction tuning**: Use calibrated fusion weights per direction

### Priority 3: Research Directions
- **RASST retrieval-augmented** (arxiv 2601.22777): +3 BLEU, +16% terminology for Extra Context
- **AliBaStr-MT learned border** (arxiv 2503.22051): Train classifier on our TS alignment data
- **ExPosST position slots** (arxiv 2603.14903): Zero-recomputation KV cache
- **GRPO fine-tuning** (SeqPO-SiMT, arxiv 2505.20622): RL-optimize R/W decisions

## Key Context

- **IWSLT 2026**: Eval April 1-15, language pairs: EN-ZH, EN-DE, EN-IT, CS-EN
- **Metrics**: LongYAAL (primary latency), COMET (primary quality), XCOMET-XL (internal)
- **CUNI won 2025**: AlignAtt + LA + forced decode. We extend with 8+ additional signals + fusion
- **Hikari is main competitor**: Policy-free WAIT tokens, SOTA on EN-JA/DE/RU
- **Our advantage**: AlignAtt with ANY model via llama.cpp (CUNI couldn't use AlignAtt with EuroLLM)
- **HY-MT1.5-7B is champion**: 0.842 XCOMET-XL EN-ZH
- llama.cpp built with PR #20086 on A40
- **Dead ends**: EAST, LoRA no-think, GDN, confidence-only, fixed-rate, TAF, Seed-X-PPO, Qwen3-4B, HY-MT1.8B
