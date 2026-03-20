# NLLW TODO — SimulMT Research Forge

Last updated: 2026-03-20

## Done (this session)

### Bugs fixed
- [x] `logit_idx` tracking (batch index 0 vs n-1 after decode_single vs decode_batch)
- [x] Border threshold guard for short sources (n_src <= border_distance)
- [x] Thread safety (threading.Lock on shared llama_context)
- [x] Stderr suppression during Metal JIT compilation
- [x] Generation cap (n_src * 2 → prevents hallucination from 2-word context)
- [x] Ellipsis/uncertainty token filtering (soft stop on "...")
- [x] Sentence-end commit-and-stop (prevents multiple short sentences per call)
- [x] Adaptive border distance (scales with source length for long sentences)
- [x] min_commit guarantee (translation keeps up with typing)
- [x] Consecutive border hits (require 2 to trigger stop)
- [x] word_batch=3 default (fixes "I would" and "il fait beau" hallucinations)
- [x] Weighted attention averaging (ts-weighted avg of full distributions, not argmax-per-head)

### Backends built
- [x] AlignAttBackend — original with all fixes + entropy veto + context injection
- [x] AlignAttLocalAgreementBackend — re-translate + diff for stability
- [x] AlignAttKVCacheBackend — KV cache delta decoding (5-10x speedup)
- [x] WaitKBackend — standard wait-k baseline
- [x] FullSentenceBackend — quality upper bound (offline translation)
- [x] EagerBackend — latency lower bound

### Multi-model support
- [x] 6 prompt formats: hymt, qwen3, qwen3.5, qwen3-nothink, eurollm, custom
- [x] Auto-detect prompt format from model filename
- [x] Per-format stop tokens (HY-MT: extra_0, Qwen3: im_end, EuroLLM: im_end)
- [x] LoRA adapter loading (full C API bindings: load/apply/clear)
- [x] 23 alignment head configs ported from iwslt26-sst

### Research tools
- [x] eval.py — BLEU/COMET/xCOMET-XL, parameter sweep
- [x] research.py — compute-aware latency (CA-AL), benchmark suite, FLORES_MINI
- [x] simulate.py — policy replay, Average Lagging
- [x] corpus.py — 130-sentence categorized test corpus with discourse pairs
- [x] experiment.py — config/result registry, Pareto analysis, CLI
- [x] analysis.py — Pareto frontier, edge case analysis, report generation, LaTeX tables
- [x] detect_heads.py — auto head detection for any GGUF model
- [x] metrics.py — BLEU, COMET, xCOMET-XL wrappers (graceful fallbacks)
- [x] lora.py — LoRA adapter discovery and config
- [x] **bench.py** — unified one-command benchmarking CLI with sweep, compare, OmniSTEval export
- [x] **omnisteval.py** — OmniSTEval JSONL output format for IWSLT submission

### Infrastructure
- [x] backend_protocol.py — SimulMTBackend ABC + create_backend() factory
- [x] llama_backend.py — extended with get_logits_array, decode_batch_at, decode_single_at, n_layer, n_head, LoRA C API
- [x] Web debug: FastAPI server with /compare, /evaluate, /backends, /heads, /prompt_formats
- [x] Ollama-style web UI with sidebar controls, chat interface, backend selector
- [x] MCP server for Claude Code direct access to translation testing
- [x] Context injection (rolling buffer of previous translations)
- [x] **9 experiment YAML configs** in `configs/` for all standard benchmarks

### Documentation
- [x] CLAUDE.md — project state, parameters, findings
- [x] docs/research/iwslt26-sst-findings.md — knowledge extraction from experiment repo
- [x] docs/ideation/ — 14 implemented + 7 Round 2 ideas
- [x] docs/brainstorms/ — research forge requirements
- [x] docs/plans/ — 4-phase research forge plan

---

## TODO — Next priorities

### P0: Critical for IWSLT competition
- [ ] **AlignAtt vs LocalAgreement head-to-head comparison** — NEVER tested. The 2025 IWSLT winner and 2026 baseline both use LocalAgreement. We have alignatt-la but haven't benchmarked it properly against alignatt. **Ready to run**: `python -m nllw.bench --compare alignatt alignatt-la --suite corpus --lang en-fr --comet --save`
- [ ] **Run full COMET/xCOMET eval on the 130-sentence corpus** — Currently only have BLEU. Need COMET scores to establish a real baseline. **Ready to run**: `python -m nllw.bench --suite corpus --lang en-fr,fr-en --comet --save`
- [ ] **Per-direction parameter optimization** — iwslt26-sst found EN-ZH=bd3/wb2, EN-IT=bd4/wb3, CS-EN=bd2/wb1. Verify these with our eval pipeline. **Ready to run**: `python -m nllw.bench --suite corpus --lang en-zh --sweep "bd=2,3,4 wb=1,2,3" --comet --save`
- [x] **OmniSTEval output format** — `nllw/omnisteval.py` module + `--omnisteval` flag in bench.py. Format: per-sentence JSONL with NFKC-normalized prediction, char/word-level delays, source_length. Eval with `omnisteval longform` CLI.
- [ ] **HY-MT cascade test on A40** — Run full ASR->MT pipeline with Qwen3-ASR + HY-MT.
- [ ] **LongYAAL metric implementation** — Primary IWSLT 2026 ranking metric. Currently compute CA-AL but need to also compute LongYAAL via OmniSTEval pipeline.

### P1: High-leverage improvements
- [ ] **Improve EN->FR committed ratio** — Currently 63% with wb=3. Investigate: does alignatt-la do better? Does adjusting min_commit or gen_cap help?
- [ ] **Context injection A/B on discourse pairs** — Test the 7 discourse-dependent pairs with context_window=0 vs 3 vs 5. Config ready: `configs/context-injection-ab.yaml`
- [x] **`nllw bench` CLI** — `python -m nllw.bench --suite corpus --lang en-fr --comet --sweep "bd=2,3,4 wb=2,3" --save --omnisteval output.jsonl`
- [x] **Experiment config YAML files** — 9 configs in `configs/`: alignatt-vs-la, per-direction (en-zh/en-de/en-it/cs-en), entropy-veto-tuning, context-injection-ab, kv-cache-validation, full-sweep-en-fr
- [ ] **FLORES+ dataset integration** — Load from HuggingFace for standardized benchmarks

### P2: Quality improvements to explore
- [ ] **Entropy veto threshold tuning** — Test 0.70, 0.75, 0.80 on the full corpus with COMET. Config ready: `configs/entropy-veto-tuning.yaml`
- [ ] **KV cache backend validation** — Verify alignatt-kv produces identical output to alignatt. Config ready: `configs/kv-cache-validation.yaml`
- [ ] **Think block for Qwen3.5** — Test think=on vs think=off on EN->FR with COMET
- [ ] **Border distance sensitivity curves** — bd=1,2,3,4,5 on each language pair, plot quality vs latency. Config ready: `configs/full-sweep-en-fr.yaml`
- [ ] **Prompt format comparison** — Same model (HY-MT), different prompt styles
- [ ] **LoRA adapter testing** — If domain adapters are available, test quality impact

### P3: Infrastructure improvements
- [ ] **WebSocket streaming endpoint** — Replace HTTP polling with real-time /ws/translate
- [ ] **WhisperLiveKit cascade integration** — ASR->MT pipeline
- [ ] **Side-by-side A/B comparison UI** — Render /compare results in split pane
- [ ] **Model loading progress indicator** — SSE stream during /load
- [ ] **Experiment config save/load in UI** — Named configs with one-click replay
- [ ] **CI quality gate** — GitHub Actions running BLEU on PRs touching backend code

### P4: Research directions (from literature survey, March 2026)
- [ ] **SSBD for alignatt-la** — Self-Speculative Biased Decoding: reuse previous translation as speculative draft in LocalAgreement re-translate, verify in single forward pass. 2-3x speedup, no quality loss. **Low effort, high impact.** (arxiv.org/abs/2509.21740)
- [ ] **EAST interleaved decoding** — Interleave source and target tokens with separators, let LLM learn adaptive READ/WRITE from sequence structure. SOTA on SiMT benchmarks with limited SFT data. Could replace border heuristic entirely. (ACL 2025 Findings)
- [ ] **ExPosST position slots** — Pre-allocate positional slots for streaming source tokens so target positions stay invariant. Zero-recomputation inference. Cleaner solution than our KV cache delta approach. (arxiv.org/abs/2603.14903)
- [ ] **Conversational SimulMT framing** — Reformulate as multi-turn dialogue (user=source chunks, assistant=target chunks). Natural KV cache reuse without re-computation. Tested with Llama2-7b. (IWSLT 2025)
- [ ] **SeqPO-SiMT RL policy** — RL fine-tuning for read/write decisions. +1.13 COMET, -6.17 AL vs SFT on Qwen-2.5-7B. Rivals offline translation quality. **High effort.** (arxiv.org/abs/2505.20622)
- [ ] **Human-like interpreter actions** — Extend READ/WRITE with SENTENCE_CUT, DROP, PARTIAL_SUMMARIZATION, PRONOMINALIZATION. Improves semantic metrics and lowers delay. (arxiv.org/abs/2601.11002)
- [ ] **Clause-level batching** — Use syntactic parsing instead of fixed word_batch
- [ ] **Adaptive border from speech rate** — TimedText timestamps -> dynamic border_distance
- [ ] **Document-level context via KV cache** — Persistent cross-sentence KV state

---

## Known issues
- EN->FR BLEU is low (~11) partly because reference translations use different phrasing (valid translations score low on BLEU)
- `argmax_logits(ctx, -1)` doesn't work in current llama.cpp build -- must use logit_idx tracking
- HY-MT context injection hurts quality (-0.028 COMET) -- disabled by default
- Sentence-end commit-and-stop can produce unwanted "." in partial translations
- Full re-decode from scratch on every translate() call is slow (alignatt-kv fixes this but needs validation)

---

## IWSLT 2026 Submission Requirements (from iwslt26-sst repo)

### Format
- **Docker image** (linux/arm64) OR system log
- **Hardware**: Single NVIDIA H100 80GB
- **Primary metric**: LongYAAL (non-computation-aware) via OmniSTEval
- **Quality metric**: XCOMET-XL (NOT wmt22-comet-da)
- **Toolkit**: OmniSTEval (https://github.com/pe-trik/OmniSTEval)

### OmniSTEval JSONL format
```json
{"source": "recording.wav", "prediction": "translated text", "delays": [1510.0, ...], "elapsed": [1510.0, ...], "source_length": 695000}
```
- Character-level delays (ms) for CJK languages
- Word-level delays (ms) for European languages
- NFKC Unicode normalization required

### Running OmniSTEval
```bash
omnisteval longform \
  --speech_segmentation MASTER.yml \
  --ref_sentences_file REF.txt --source_sentences_file SRC.txt \
  --hypothesis_file HYP.jsonl --hypothesis_format jsonl \
  --output_folder eval/ --lang zh --char_level \
  --comet_model Unbabel/XCOMET-XL
```

---

## IWSLT 2025 Competition Results (intelligence for 2026)

### Top systems at IWSLT 2025 SST
| Team | Approach | Key Numbers | Notes |
|------|----------|-------------|-------|
| **CUNI** (1st) | Whisper-large-v3 + AlignAtt + EuroLLM cascade | Best COMET in CS-EN (2s/4s), EN-DE/ZH/JA (4-5s) | **Our approach!** +2 BLEU CS-EN, +13-22 BLEU EN-DE/ZH/JA over baselines |
| NAIST | Whisper encoder + DeCo projector + Qwen LLM + LocalAgreement | -- | End-to-end with LLM decoder |
| CMU | Chunkwise Wav2Vec2 + Qwen2.5-7B | 44.3 BLEU EN-ZH, 25.1 BLEU EN-DE | -- |
| BeaverTalk | Oregon State | 24.6-27.8 BLEU EN-DE, 34.1-37.2 BLEU EN-ZH | -- |

### CUNI's winning additions over base AlignAtt
- Beam search decoding (we don't have this yet)
- Initial prompts for context
- Context from preceding audio buffers
- 5x faster than prior WhisperStreaming

### Key takeaways for 2026
- AlignAtt won 2025 -- our approach is validated as SOTA
- LocalAgreement (NAIST) is the main competitor -- need head-to-head comparison
- CUNI's beam search could be a low-effort quality boost for us
- SimulStream (https://github.com/hlt-mt/simulstream) is the Docker framework
- IWSLT 2026 baselines: https://github.com/owaski/iwslt-2026-baselines
- IWSLT 2026 co-located with ACL 2026, San Diego, July 3-4, 2026
- New "extra context" sub-track (can leverage paper content for ACL talks)

---

## Reference numbers (from iwslt26-sst)

### Best known system: HY-MT1.5-7B
| Direction | XCOMET-XL | wmt22 | LongLAAL | Config |
|-----------|:---------:|:-----:|:--------:|--------|
| EN->ZH | **0.842** | 0.858 | ~1270ms | bd=3, wb=2, no context |
| EN->DE | **0.786** | 0.721 | 1551ms | bd=3, wb=2 |
| EN->IT | **0.752** | 0.763 | 1464ms | bd=4, wb=3 |
| CS->EN | **0.908** | 0.841 | -- | bd=2, wb=1 |

### Recommended final system
```
Audio -> [Qwen3-ASR-1.7B] -> words -> [HY-MT1.5-7B AlignAtt] -> translation
        WER 9.2%, RTF 0.074          XCOMET-XL 0.842 (EN-ZH)
        ~4 GB VRAM                    ~7.5 GB VRAM
                    Total: RTF ~0.15, ~12 GB / 80 GB H100
```
