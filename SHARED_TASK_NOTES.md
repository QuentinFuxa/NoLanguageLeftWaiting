# Shared Task Notes -- NLLW SimulMT

## Current State (after Iteration 20, 2026-03-21)

**30+ SimulMT modules (~14,800 lines), 893 tests**
**100-sentence CONFIRMED: COMET=0.894 EN-ZH, 0.881 EN-DE, 0.891 EN-IT, 0.879 CS-EN with top_p**

### What happened in Iteration 20
- **Longform mode implemented** (CRITICAL for competition):
  - `longform=True` in SimulStreamConfig (default): accumulates output across sentence boundaries
  - Backend resets at sentence boundaries (AlignAtt needs this) but output is one continuous stream
  - `_recording_text` tracks full recording, `_emission_log` tracks per-event timestamps
  - Fixed double-reset bug (translate + reset calling _handle_segment_end twice)
  - `clear()` is only full reset (between recordings)
- **OmniSTEval longform output** (`to_omnisteval_entry()`):
  - ONE JSONL entry per recording: `{"source", "prediction", "delays", "elapsed", "source_length"}`
  - Per-word or per-character delays matching `ss-to-log.py` reference format
  - `process_gold_transcript_longform()` for competition-format evaluation
- **Auto sentence boundary detection**: target-side punctuation (。？！.?!) triggers segment reset
- **IWSLT configs updated**: all 4 directions now have `longform: true`
- **Perplexity-based adaptive border** (Hibiki-inspired):
  - Adjusts bd per translate() based on generation confidence (logit perplexity)
  - Low ppl (confident) -> bd-1 (faster). High ppl (uncertain) -> bd+1 (safer)
  - Targets YAAL latency. Unlike entropy veto (dead end), adjusts R/W policy not generation
  - CLI: `--perplexity-adaptive-bd`, sweep: `pplbd=0,1 ppllow=1.5,2.0 pplhigh=4.0,5.0`
  - **Needs GPU testing** to measure YAAL reduction
- **SOTA research completed**: Hibiki, ExPosST, Translation Heads ICLR 2026, DuoAttention
- **Competition validator**: 65+ checks, ALL PASSING
- **893 tests** (34 new, all passing)

### 100-Sentence Verified Results (iteration 18, with tuned p_threshold + CI)

| Direction | bd | wb | p | BLEU | COMET | 95% CI | YAAL | % offline |
|-----------|---:|---:|:-:|-----:|------:|--------|-----:|:---------:|
| **EN-ZH** | 3 | 4 | 0.85 | 40.0 | **0.894** | [0.887, 0.901] | 6.09 | 99.8% |
| **EN-DE** | 2 | 3 | 0.75 | 27.9 | **0.881** | [0.873, 0.890] | 5.45 | 99.7% |
| **EN-IT** | 2 | 3 | 0.9 | 24.3 | **0.891** | [0.882, 0.899] | 6.76 | **100.2%** |
| **CS-EN** | 3 | 3 | 0.9 | 28.4 | **0.879** | [0.871, 0.886] | 5.81 | 99.8% |

## What to do next

### Priority 1: Competition E2E Testing (IWSLT 2026, eval April 1-15, ~10 days)
- **Docker build + test**: Build image, run self-test. Must support linux/amd64 (H100).
- **Longform E2E on A40**: Run `process_gold_transcript_longform()` on a real recording from iwslt26-sst
  - Use gold JSONL from `iwslt26-sst/inputs/en/acl6060.ts/gold-jsonl/`
  - Verify output matches OmniSTEval format via `omnisteval longform`
  - Compare LongYAAL + COMET with reference system scores
- **SimulStream HTTP server E2E**: Install simulstream package, test HTTP server integration
  - `simulstream.server --speech-processor nllw.simulstream:NLLWSpeechProcessor`
  - Verify direction switching, clear() between recordings, longform accumulation
- **Multi-direction longform test**: Verify all 4 directions work in longform mode
- **Decision**: Enable `adaptive_top_p` for competition? Phase 1 shows 6-12% latency reduction for <0.002 COMET cost

### Priority 2: Quality Improvements
- **Context injection for longform**: In longform mode, context_sentences could help since segments are consecutive. Test context=1 in longform (even though it hurts HY-MT in isolation, longform continuity might change the equation)
- Run competition-format test (SimulStream + OmniSTEval) on A40
- If adaptive_top_p confirmed: update IWSLT configs to enable it

### Priority 3: Research Ideas (if time permits)
- **Syntax-aware chunking (SASST)**: Dependency-based word batching for better segmentation
- **ExPosST positional pre-allocation**: Pre-allocate source positions for faster KV cache reuse
- **Perplexity gain signal**: Use LLM perplexity change as border signal

### Dead Ends Confirmed (20+)
See CLAUDE.md for full list. Key ones: context injection, entropy veto, softmax_mean, signal fusion cascade, repetition halt, top_p_weighted, Qwen3.5-9B.

### Sync Workflow
IMPORTANT: When syncing code to A40, do NOT use `rsync --delete` which destroys GPU-generated configs.
Use: `rsync -avz` (without --delete) to preserve files.

## Key Context

- **IWSLT 2026**: Eval April 1-15, ~10 days away
- **Metrics**: LongYAAL (primary latency), **COMET wmt22-comet-da** (primary quality)
- **Best known**: EN-ZH COMET=0.894, EN-DE 0.881, EN-IT 0.891 (>offline!), CS-EN 0.879
- **All directions at 99.7-100.2% of offline quality**
- **Model path on A40**: `/home/fuxa/HY-MT1.5-7B.Q8_0.gguf`
- **Competition validator**: `python scripts/validate_competition.py` (65+ checks pass)
- **New in iter 20**: Longform mode, OmniSTEval output, perplexity adaptive border, 893 tests
- **OmniSTEval format**: ONE JSONL line per recording with per-word delays in ms
  - Reference converter: `iwslt26-sst/evaluation/ss-to-log.py`
  - Run: `omnisteval longform --speech_segmentation ... --ref_sentences_file ... --hypothesis_file out.log`
