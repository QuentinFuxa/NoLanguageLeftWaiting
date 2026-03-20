# NLLW -- SimulMT Research Platform TODO

## Status: Active Development (2026-03-20)

---

## DONE -- Core Infrastructure (Iteration 1)

- [x] `nllw/prompts.py` -- Prompt format registry (30+ formats: HY-MT, Qwen3, Qwen3.5, EuroLLM, Tower, Gemma)
- [x] `nllw/llama_backend.py` -- ctypes wrapper for llama.cpp with attention extraction
- [x] `nllw/backend_protocol.py` -- SimulMTBackend ABC + factory with `@register_backend`
- [x] `nllw/alignatt.py` -- Core AlignAtt algorithm (border detection, TS-weighted vote, entropy)
- [x] `nllw/alignatt_backend.py` -- Full AlignAtt backend with KV cache + baselines (full-sentence, eager)
- [x] `nllw/metrics.py` -- All latency metrics (AL, LAAL, YAAL, AP, DAL, MaxCW) + BLEU/COMET wrappers
- [x] `nllw/simulate.py` -- Policy simulation and trace replay
- [x] `nllw/heads/` -- 22 pre-computed alignment head configs from iwslt26-sst
- [x] 62 unit tests (all passing)
- [x] `nllw/eval.py` -- Evaluation harness (FLORES+ corpus, parameter sweep, XCOMET-XL)
- [x] `nllw/bench.py` -- Unified CLI: `python -m nllw.bench --lang en-fr --comet --save`

## DONE -- Evaluation & Research Tools (Iteration 2)

- [x] `nllw/detect_heads.py` -- Auto alignment head detection for any GGUF model (559 lines)
- [x] `nllw/omnisteval.py` -- OmniSTEval JSONL output for IWSLT 2026 submission (258 lines)
- [x] `nllw/baselines.py` -- wait-k + fixed-rate baselines (175 lines)
- [x] `nllw/analysis.py` -- Pareto frontier, edge cases, report generation (309 lines)
- [x] `nllw/experiment.py` -- Experiment config/result registry with YAML support (359 lines)
- [x] `nllw/corpus.py` -- Expanded to 120 sentences across 5 directions (EN-FR, EN-ZH, EN-DE, EN-IT, CS-EN)
- [x] 109 unit tests (all passing) -- 47 new tests for iteration 2 modules

---

## DONE -- Research Tools (Iteration 2, continued)

- [x] `nllw/research.py` -- Compute-aware latency (CA-AL, CA-YAAL), benchmark suite (191 lines)
- [x] 4 experiment configs in `configs/` (baseline, sweep, compare, entropy)
- [x] 115 unit tests (all passing)

## DONE -- Iteration 3: LA Backend + Novel Aggregation

- [x] `nllw/alignatt_la_backend.py` -- LocalAgreement + AlignAtt hybrid (re-translate, diff, commit stable prefix)
- [x] 6 novel aggregation methods in `alignatt.py`: ts_vote, softmax_mean, entropy_weighted, consensus, geomean, top_p
- [x] `aggregation` parameter in BackendConfig + wired into AlignAttBackend and AlignAttLABackend
- [x] `--aggregation` CLI flag in bench.py + `agg` shortname in sweep parser
- [x] `alignatt-la` backend registered in factory, auto-imported by bench.py
- [x] 2 new experiment configs: `sweep-aggregation-en-zh.yaml`, `compare-la-en-zh.yaml`
- [x] Attention entropy-based dynamic border distance (`check_border_dynamic`, `dynamic_border_distance`)
- [x] `--dynamic-border` CLI flag + `dynbd` sweep shortname
- [x] Ensemble aggregation method (weighted average of multiple strategies)
- [x] 7 experiment configs in `configs/` (3 new: aggregation sweep, LA comparison, dynamic border)
- [x] 172 unit tests (57 new, all passing)

## DONE -- Iteration 4: SSBD + NE Metric

- [x] `ssbd_accept()` -- SSBD biased acceptance criterion (Zeng et al., 2025)
- [x] `_retranslate_ssbd()` in alignatt-la -- 3-phase speculative verification:
  1. Batch verify draft tokens (previous translation) in ONE forward pass
  2. Find first divergence (biased acceptance with beta parameter)
  3. Resume autoregressive generation with border detection from divergence
- [x] `ssbd_beta` parameter in BackendConfig (None=disabled, 0.0=pure speculative, 0.2=recommended)
- [x] `--ssbd-beta` CLI flag in bench.py + `ssbd` shortname for sweep
- [x] `compute_normalized_erasure()` -- NE metric for token-level output stability
- [x] `compute_normalized_erasure_text()` -- NE metric for word-level output stability
- [x] Revision history tracking in LA backend (`get_revision_history()`, `get_ssbd_stats()`)
- [x] Display-only mask-k (`display_mask_k` param): hide last k unstable tokens from display while keeping as SSBD draft. SSBD paper shows NE drops from 1.01 to 0.53 with mask-3.
- [x] 203 unit tests (31 new, all passing)

## TODO -- Infrastructure

- [x] Web debug server (FastAPI + embedded UI) -- `web_debug/server.py` (port 8777)
  - Load any backend with full config (SSBD, dynamic border, aggregation, etc.)
  - Word-by-word translation with border detection visualization
  - SSBD stats (acceptance rate) and NE metric display
  - Compare endpoint for side-by-side backend comparison
- [ ] MCP server for editor integration
- [ ] LoRA adapter loading + discovery

## TODO -- Experiments to Run

- [ ] **First E2E validation on A40**: HY-MT1.5-7B on FLORES EN-ZH with AlignAtt
- [ ] Compare AlignAtt vs wait-k vs full-sentence on FLORES mini
- [ ] Test Qwen3.5-4B with context injection (Qwen3.5 benefits +0.037)
- [ ] Multi-direction sweep: en-zh, en-de, en-it, cs-en
- [ ] KV cache speedup measurement (with vs without)
- [ ] Entropy veto threshold tuning (0.5, 0.75, 1.0)
- [ ] Pareto frontier analysis: bd={2,3,4,5} x wb={1,2,3} x all directions
- [ ] **Aggregation method sweep on A40**: `python -m nllw.bench --sweep "agg=ts_vote,softmax_mean,entropy_weighted,consensus,geomean,top_p" --lang en-zh --comet --save` -- Novel research, no published baselines
- [ ] **AlignAtt vs AlignAtt-LA comparison**: `python -m nllw.bench --compare alignatt alignatt-la --lang en-zh --comet --save` -- Measure output stability + latency tradeoff
- [ ] **Cross-aggregation x direction**: Sweep all 6 aggregation methods across en-zh, en-de, en-it, cs-en
- [ ] **Dynamic border distance test**: `python -m nllw.bench --lang en-zh --dynamic-border --comet --save` vs fixed bd=3

## TODO -- Research Ideas (informed by SOTA survey, see docs/research/sota-simulmt-2026.md)

### High Priority (open research gaps, no published work)
- [x] **Novel aggregation** (IMPLEMENTED): 6 methods -- softmax_mean, entropy_weighted, consensus, geomean, top_p. Sweepable via `--aggregation` or `agg=` in sweep spec. **Needs GPU testing.**
- [x] **SSBD for alignatt-la** (IMPLEMENTED): Self-Speculative Biased Decoding (arxiv 2509.21740). 3-phase: batch verify draft -> biased acceptance -> autoregressive from divergence. `--ssbd-beta 0.2` or `ssbd=0.0,0.1,0.2` sweep. **Needs GPU testing.**
- [x] **NE metric** (IMPLEMENTED): Normalized Erasure for output stability. `compute_normalized_erasure()` + revision history tracking in LA backend.
- [ ] **LA forced decoding**: Force-decode committed prefix tokens before generating new ones (CUNI does this -- reduces computation)
- [ ] **LA two-pass catch-up**: Run two re-translations per update (CUNI approach) for extra LA comparison opportunity
- [x] **Attention entropy as dynamic border distance** (IMPLEMENTED): `--dynamic-border` flag, adjusts bd per-token based on attention entropy. **Needs GPU testing.**
- [ ] **Aggregation sweep on GPU**: Run `python -m nllw.bench --sweep "agg=ts_vote,softmax_mean,entropy_weighted,consensus,geomean,top_p" --lang en-zh --comet`
- [ ] **SSBD sweep on GPU**: Run `python -m nllw.bench --backend alignatt-la --sweep "ssbd=0.0,0.1,0.2,0.3" --lang en-zh --comet --save`
- [ ] **SSBD + mask-k sweep**: Run `python -m nllw.bench --backend alignatt-la --sweep "ssbd=0.0,0.2 mask=0,1,2,3" --lang en-zh --comet --save`
- [ ] **Adaptive SSBD beta**: Per-token entropy-based bias (high entropy -> lower beta). Combine arxiv 2509.21740 + 2508.15371.

### Medium Priority (validated by SOTA papers)
- [ ] Cross-lingual transfer of alignment heads (ICLR 2026 "Translation Heads" paper confirms heads are universal)
- [x] **LocalAgreement + AlignAtt hybrid** (IMPLEMENTED): `alignatt-la` backend. **Needs GPU testing.**
- [ ] ExPosST-style position slot reservation (arxiv 2603.14903) for zero-recomputation KV cache
- [ ] Dynamic word_batch based on source complexity (short sentences -> smaller wb)
- [ ] **Adaptive Multi-Strategy (AMS)**: Auto-select aggregation based on input (entropy, agreement ratio)
- [ ] **Gaussian kernel consensus**: Generalization of ts_vote and softmax_mean with single sigma param
- [ ] **Per-head temperature normalization**: Learned during head detection, normalizes sharpness

### Lower Priority (competitive intelligence)
- [ ] Test Group Position Encoding (ACL 2025, github.com/eit-nlp/streamingllm) as alternative to our KV cache approach
- [ ] Evaluate SimulSense-style sense unit detection (arxiv 2509.21932) for chunking
- [ ] OmniSTEval integration: end-to-end IWSLT eval pipeline
- [ ] Human-like strategies: SENTENCE_CUT, DROP, PRONOMINALIZATION (arxiv 2601.11002)
- [ ] **LSG KL-divergence policy** (arxiv 2501.00868): Training-free, uses KL(P_partial || P_full) for read/write decisions. Could complement AlignAtt but needs 2 forward passes.
- [ ] **Confidence-modulated speculative decoding** (arxiv 2508.15371): Dynamically adjust draft length based on entropy/margin uncertainty.

---

## Dead Ends (from iwslt26-sst -- DO NOT REVISIT)

- EAST learned policy (BLEU 27.3 vs baseline 42)
- LoRA no-think block (-0.178 COMET)
- GDN warm-start (33 hallucinations)
- Extra glossary (hurts small models)
- HY-MT context injection (hurts ALL 4 directions)
- Confidence-based stopping alone (COMET 0.468)
- Fixed-rate tokens (COMET 0.293-0.334)
- TAF source lookahead (worse on EN-ZH)
- Seed-X-PPO-7B, Qwen3-4B-2507, HY-MT1.5-1.8B, TranslateGemma-4B (all inferior)
