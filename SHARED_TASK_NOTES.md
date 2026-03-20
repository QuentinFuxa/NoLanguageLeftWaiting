# Iteration Notes - 2026-03-20

## What was done this iteration

### New modules built
- **`nllw/bench.py`** -- unified one-command benchmarking CLI
  - `--compare` for head-to-head backend comparison
  - `--sweep` for parameter grid search (e.g. `"bd=2,3,4 wb=2,3"`)
  - `--omnisteval` for direct IWSLT submission format export
  - `--save` to persist results to experiment registry
  - Supports `--suite corpus` for the 130-sentence corpus, `flores_mini` (20), `default` (15)

- **`nllw/omnisteval.py`** -- OmniSTEval JSONL format converter
  - Required for IWSLT 2026 SST submission
  - Handles NFKC normalization, char-level and word-level delays
  - Integrated into bench.py via `--omnisteval` flag

- **`configs/`** -- 9 experiment YAML configs ready to run:
  - `alignatt-vs-la.yaml` -- P0: AlignAtt vs LocalAgreement head-to-head
  - `per-direction-{en-zh,en-de,en-it,cs-en}.yaml` -- P0: direction-specific optimization
  - `entropy-veto-tuning.yaml`, `context-injection-ab.yaml`, `kv-cache-validation.yaml`
  - `full-sweep-en-fr.yaml` -- comprehensive bd x wb sweep

### Other changes
- Added "corpus" (130-sentence) to experiment.py corpus resolver
- Updated CLAUDE.md with new modules, bench.py usage docs
- Updated todo.md with completed items and IWSLT submission requirements

## What to do next

### Highest priority: run the P0 benchmarks on A40
All commands are ready, just need the web server running with HY-MT model:

```bash
# On A40: start the server
HYMT_MODEL_PATH=/path/to/HY-MT1.5-7B-Q8_0.gguf python web_debug/server.py

# 1. AlignAtt vs LocalAgreement (the most critical experiment)
python -m nllw.bench --compare alignatt alignatt-la --suite corpus --lang en-fr --comet --save

# 2. Per-direction sweeps
python -m nllw.bench --sweep "bd=2,3,4 wb=1,2,3" --suite corpus --lang en-zh --comet --save
python -m nllw.bench --sweep "bd=2,3,4 wb=1,2,3" --suite corpus --lang en-de --comet --save

# 3. Entropy veto tuning
python -m nllw.bench --sweep "ev=0.70,0.75,0.80" --suite corpus --lang en-fr --comet --save

# 4. KV cache validation
python -m nllw.bench --compare alignatt alignatt-kv --suite corpus --lang en-fr --comet --save
```

### If the A40 experiments are blocked
- Work on FLORES+ HuggingFace integration (standardized benchmarks)
- Implement LongYAAL metric computation locally (currently OmniSTEval CLI only)
- Build WhisperLiveKit cascade integration (ASR->MT pipeline)
- Test Qwen3.5 think=on vs think=off

### Research directions worth exploring
- Learned emission policy (MLP replacing border heuristic)
- Clause-level batching using syntactic parsing

## IWSLT 2025 Competition Intelligence
- **CUNI won IWSLT 2025 SST with Whisper + AlignAtt** -- our exact approach is validated as SOTA
- CUNI additions: beam search decoding, initial prompts, context from preceding audio buffers, 5x faster
- **NAIST** used LocalAgreement + Qwen LLM -- this is the main competitor to beat
- **CMU** used chunkwise Wav2Vec2 + Qwen2.5-7B -- 44.3 BLEU EN-ZH
- Papers: CUNI (arxiv.org/html/2506.17077), NAIST (aclanthology.org/2025.iwslt-1.39.pdf)
- IWSLT 2026 baselines repo: https://github.com/owaski/iwslt-2026-baselines
- SimulStream Docker framework: https://github.com/hlt-mt/simulstream

## Research Techniques Worth Exploring (from literature survey)
| Technique | Effort | Expected Impact |
|-----------|--------|----------------|
| **SSBD for alignatt-la** (speculative re-translate) | Low | 2-3x speedup for LA backend |
| **Beam search decoding** (CUNI's addition) | Medium | Quality boost, validated by IWSLT 2025 winner |
| **EAST interleaved decoding** | High | Adaptive R/W policy without attention heads |
| **ExPosST position slots** | Medium | Cleaner KV cache reuse than delta approach |
| **Conversational SimulMT framing** | Medium | Natural KV cache reuse |

## Key IWSLT 2026 submission facts
- **Deadline**: April 15, 2026
- **Primary metric**: LongYAAL (non-computation-aware) via OmniSTEval
- **Quality metric**: XCOMET-XL (NOT wmt22-comet-da)
- **Hardware**: Single NVIDIA H100 80GB
- **Toolkit**: OmniSTEval (https://github.com/pe-trik/OmniSTEval)
- **Format**: Docker image OR system log + OmniSTEval JSONL
