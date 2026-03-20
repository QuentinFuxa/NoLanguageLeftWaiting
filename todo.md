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

## DONE -- Iteration 5: Gaussian Kernel + Forced Decoding + Adaptive SSBD

- [x] **Gaussian Kernel Consensus** aggregation (2 variants):
  - `gaussian_kernel`: Place Gaussian at each head's argmax, peak of TS-weighted density
  - `gaussian_kernel_continuous`: Convolve full attention distributions with Gaussian kernel
  - Sigma parameter: 0.5=near-argmax, 1.5=moderate, 3.0=wide blur
  - Key advantage: subword boundary tolerance (nearby heads reinforce each other)
  - Registered in `_BASE_AGGREGATION_METHODS`, usable in sweeps via `agg=gaussian_kernel`
- [x] **LA Forced Decoding** (`la_forced_decode` config, `--forced-decode` CLI):
  - Force-decode committed prefix tokens before generating new ones (CUNI approach)
  - Conditions model on committed output (consistency) + fewer tokens to generate (speed)
  - Strategy priority: SSBD > forced decoding > standard re-translation
  - Sweep: `forced=0,1`
- [x] **Adaptive SSBD Beta** (`adaptive_ssbd` config, `--adaptive-ssbd` CLI):
  - Per-token entropy-based bias: confident=higher beta, uncertain=lower beta
  - Combines SSBD (2509.21740) with entropy-modulated confidence (2508.15371)
  - Scale: ent<=1.0 -> beta*1.5, ent>=4.0 -> beta*0.2, capped at 0.95
  - Sweep: `adaptive=0,1`
- [x] 244 unit tests (41 new, all passing)

## DONE -- Iteration 6: Two-Pass, AMS, Temp Norm, Head Transfer

- [x] **LA Two-Pass Catch-up** (`la_two_pass` config, `--two-pass` CLI):
  - Run two independent re-translations per source update
  - Keep the one with the longer common prefix with previous output (more stable)
  - Pass 2 uses standard re-translation even if SSBD/forced was used for pass 1 (diversity)
  - Trades 2x compute for better output stability (lower NE)
  - Sweep: `twopass=0,1`
- [x] **Adaptive Multi-Strategy (AMS)** (`adaptive_aggregation` config, `--adaptive-agg` CLI):
  - Auto-select aggregation method per token based on attention patterns
  - Analyzes head agreement ratio + attention entropy to pick method:
    - High agreement + low entropy -> ts_vote
    - High agreement + high entropy -> entropy_weighted
    - Low agreement + low entropy -> geomean
    - Low agreement + high entropy -> consensus
  - Novel: no published work on per-token aggregation selection
  - Sweep: `ams=0,1`
- [x] **Per-head temperature normalization** (`head_temp_normalize` config, `--head-temp-norm` CLI):
  - Normalize attention sharpness per head before aggregation
  - Binary search for temperature that matches reference entropy
  - Ensures fair weighting: sharp heads don't dominate due to peaky distributions
  - Sweep: `tempnorm=0,1 tempref=1.0,1.5,2.0`
- [x] **Cross-lingual head transfer analysis** (`nllw/head_transfer.py`, `python -m nllw.head_transfer`):
  - Validates whether heads detected for one language pair work for another
  - Metrics: Jaccard similarity, top-K overlap, TS rank correlation, transferred TS mass
  - **Key finding**: EuroLLM heads are EXCELLENT (98.9% TS mass transfer, min 97.3%)
  - **Key finding**: HY-MT1.8B heads are EXCELLENT (98.4% TS mass transfer)
  - **Key finding**: Qwen3.5 heads are EXCELLENT (97.8% TS mass transfer)
  - **Key finding**: Qwen3-4B heads are GOOD but weaker (79.8% mean, 43.5% worst-case en-it->en-zh)
  - Confirms ICLR 2026 "Translation Heads" paper: most models have universal alignment heads
- [x] 290 unit tests (46 new, all passing)

## DONE -- Iteration 7: Dynamic WB, Info Gain, Shift-K Border

- [x] **Dynamic word_batch** (`dynamic_word_batch` config, `--dynamic-wb` CLI):
  - Adjust word_batch based on source sentence length
  - Short sentences (< 8 words) -> wb - 1 (faster latency)
  - Long sentences (> 20 words) -> wb + 1 (safer, less hallucination)
  - Medium sentences keep base wb. Minimum wb = 1.
  - Sweep: `dynwb=0,1`
- [x] **Attention information gain** (`info_gain_threshold` config, `--info-gain` CLI):
  - KL divergence between consecutive attention snapshots as secondary border signal
  - Large KL = model processing new source info -> keep generating (inhibit stop)
  - Small KL = source exhausted -> reinforce border stop
  - Inspired by LSG (arxiv 2501.00868) KL(P_partial || P_full) approach
  - Modulates both standard AlignAtt and shift-k border decisions
  - Sweep: `infogain=0.2,0.3,0.5`
- [x] **Shift-k adaptive border** (`shift_k_threshold` config, `--shift-k` CLI):
  - Measure total attention MASS in border region instead of just argmax position
  - Inspired by DrFrattn (EMNLP 2025) "shift-k" mechanism
  - Softer, more robust signal: catches cases where attention splits between
    border and near-border positions (30% here + 30% there = stop)
  - TS-weighted border mass: if >= threshold -> stop
  - Sweep: `shiftk=0.3,0.4,0.5`
- [x] **Combined border check** (`check_border_combined()` in alignatt.py):
  - Multi-signal fusion: standard AlignAtt + shift-k mass + info gain
  - Decision logic: info gain inhibits/reinforces, shift-k fires on mass, standard fallback
  - Returns diagnostic values (info_gain, border_mass) for analysis
  - Wired into both AlignAtt and AlignAtt-LA backends
- [x] **Border confirmation** (`border_confirm` config, `--border-confirm` CLI):
  - Require N consecutive border hits before stopping generation
  - Prevents false positive stops from transient attention patterns
  - 1 = standard (stop on first hit), 2 = recommended for high quality
  - Sweep: `confirm=1,2,3`
- [x] **Source complexity estimator** (`nllw/complexity.py`):
  - Novel: per-sentence complexity scoring (0-1) from text features
  - Signals: word count, avg word length, subword ratio, numeral density, punctuation
  - Maps to adaptive parameter adjustments (bd, wb, gen_cap)
  - `estimate_complexity()`, `adaptive_params_from_complexity()`, `classify_complexity()`
- [x] **Cumulative attention aggregation** (10th method, from DrFrattn EMNLP 2025):
  - Compute cumulative attention mass instead of argmax position
  - Frontier = rightmost position where remaining mass >= lambda
  - Captures distribution shape: split attention correctly identified
  - Lambda parameter gives continuous latency control
  - `--aggregation cumulative` CLI, usable in `agg=cumulative` sweep
  - Research documented in `docs/research/drfrattn-analysis.md`
- [x] 360 unit tests (70 new, all passing)

## DONE -- Iteration 8: LSG Logit KL Divergence

- [x] **LSG logit KL divergence** (`lsg_kl_threshold` config, `--lsg-kl` CLI):
  - Implements LSG paper (arxiv 2501.00868, AAAI 2025): training-free border confirmation
  - KV cache fork + probe: copy cache, remove last K source tokens, re-decode, compare logits
  - Low KL = source exhausted -> confirm border stop (WRITE)
  - High KL = source still matters -> override border stop (READ more)
  - Orthogonal signal to attention-based border: attention = WHERE model looks, logit KL = WHETHER output CHANGES
  - `compute_logit_kl()` in alignatt.py (stable softmax, numerical guard)
  - Integrated into both AlignAtt and AlignAtt-LA backends (all 3 retranslation methods)
  - `--lsg-kl 7.0 --lsg-k 3` CLI flags
  - `lsg=5.0,7.0,9.0 lsgk=1,3,5` sweep shortnames
  - Overhead: ~1 extra forward pass per border check (~1-3ms on A40)
- [x] **Complexity-adaptive parameters** (`complexity_adaptive` config, `--complexity-adaptive` CLI):
  - Wires `complexity.py` into both AlignAtt and AlignAtt-LA backends
  - Per-sentence estimation: word count, avg word length, numeral density, subword ratio
  - Simple sentences -> reduced bd/wb (lower latency)
  - Complex sentences -> increased bd/wb (higher quality)
  - Applied BEFORE dynamic word_batch (stacks with it)
  - In LA backend: sets `_effective_bd` used by `_check_border()`
  - `--complexity-adaptive` CLI flag, `cmplx=0,1` sweep shortname
- [x] 396 unit tests (36 new, all passing)

## DONE -- Iteration 9: Cross-Step Border Signals (REINA + Prediction Stability)

- [x] **Entropy change tracking** (`entropy_change_threshold` config, `--entropy-change` CLI):
  - REINA-inspired (arxiv 2508.04946, AAAI 2026 Oral): track generation entropy across translate() calls
  - If adding a new source word reduces entropy significantly (delta < threshold), the model is still learning from source -> inhibit border stop (READ more)
  - If entropy change is small or positive, source is exhausted -> allow border stop (WRITE)
  - `compute_entropy_change()` in alignatt.py: returns (delta, current_entropy)
  - `entropy_change_supports_write()`: interpret delta as READ/WRITE signal
  - Integrated as pre-filter in `check_border_combined()` (fires before attention checks)
  - Wired into both AlignAtt and AlignAtt-LA backends (all 3 retranslation methods)
  - `--entropy-change -0.5` CLI flag, `entchg=-0.5,-1.0,-2.0` sweep shortname
  - Zero overhead: reuses existing logits from generation position
- [x] **Prediction stability index** (`prediction_stability` config, `--prediction-stability` CLI):
  - Novel: no published work on cross-step prediction stability for SimulMT border detection
  - Track how model's top predictions change between consecutive translate() calls
  - Two metrics: top-1 rank stability (rank of prev top-1 in current dist) + top-K Jaccard overlap
  - `compute_prediction_stability()`: returns (top1_rank_change, topk_overlap)
  - `prediction_stability_supports_write()`: interpret as READ/WRITE signal
  - Stable predictions (rank <= 3, overlap >= 0.4) = WRITE (enough source context)
  - Volatile predictions = READ (model still adapting to new source info)
  - Integrated as post-filter in `check_border_combined()` (fires after attention checks)
  - `--prediction-stability` CLI flag, `predstab=0,1` sweep shortname
  - Overhead: one logits copy per translate() call
- [x] **Cross-step signal architecture**:
  - Both signals computed BEFORE generation loop (once per translate() call)
  - Stored as instance variables, consumed by `_check_border()` / `check_border_combined()`
  - State reset on segment boundary (_handle_segment_end / _init_segment)
  - Orthogonal to all attention-based signals (entropy change = output-space, stability = prediction-space)
  - `use_combined` condition extended to trigger on any multi-signal feature
- [x] 441 unit tests (45 new, all passing)

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
- [ ] **Aggregation method sweep on A40** (now 10 methods): `python -m nllw.bench --sweep "agg=ts_vote,softmax_mean,entropy_weighted,consensus,geomean,top_p,gaussian_kernel,gaussian_kernel_continuous,cumulative,ensemble" --lang en-zh --comet --save`
- [ ] **AlignAtt vs AlignAtt-LA comparison**: `python -m nllw.bench --compare alignatt alignatt-la --lang en-zh --comet --save`
- [ ] **Cross-aggregation x direction**: Sweep all 9 aggregation methods across en-zh, en-de, en-it, cs-en
- [ ] **Dynamic border distance test**: `python -m nllw.bench --lang en-zh --dynamic-border --comet --save` vs fixed bd=3
- [ ] **Gaussian kernel sigma sweep**: `python -m nllw.bench --sweep "agg=gaussian_kernel" --lang en-zh --comet` (vary sigma via config)
- [ ] **Forced decoding test**: `python -m nllw.bench --backend alignatt-la --forced-decode --lang en-zh --comet --save` -- speed & quality vs standard LA
- [ ] **Adaptive SSBD sweep**: `python -m nllw.bench --backend alignatt-la --ssbd-beta 0.2 --adaptive-ssbd --lang en-zh --comet --save` vs fixed beta
- [ ] **Two-pass stability test**: `python -m nllw.bench --backend alignatt-la --two-pass --lang en-zh --comet --save` -- compare NE with and without
- [ ] **AMS aggregation test**: `python -m nllw.bench --adaptive-agg --lang en-zh --comet --save` vs fixed ts_vote
- [ ] **Head temp normalization test**: `python -m nllw.bench --head-temp-norm --lang en-zh --comet --save` vs unnormalized
- [ ] **Combined AMS + temp norm**: `python -m nllw.bench --adaptive-agg --head-temp-norm --lang en-zh --comet --save`
- [ ] **Full iteration 6 sweep**: `python -m nllw.bench --sweep "ams=0,1 tempnorm=0,1 twopass=0,1" --backend alignatt-la --lang en-zh --comet --save`
- [ ] **Dynamic word_batch test**: `python -m nllw.bench --dynamic-wb --lang en-zh --comet --save` vs fixed wb=3
- [ ] **Shift-k border sweep**: `python -m nllw.bench --sweep "shiftk=0.3,0.4,0.5,0.6" --lang en-zh --comet --save`
- [ ] **Info gain sweep**: `python -m nllw.bench --sweep "infogain=0.2,0.3,0.5" --lang en-zh --comet --save`
- [ ] **Combined shift-k + info gain**: `python -m nllw.bench --shift-k 0.4 --info-gain 0.3 --lang en-zh --comet --save`
- [ ] **Full iteration 7 sweep**: `python -m nllw.bench --sweep "dynwb=0,1 shiftk=0.4 infogain=0.3" --lang en-zh --comet --save`
- [ ] **Border confirmation sweep**: `python -m nllw.bench --sweep "confirm=1,2,3" --lang en-zh --comet --save`
- [ ] **Combined shift-k + confirm**: `python -m nllw.bench --shift-k 0.4 --border-confirm 2 --lang en-zh --comet --save`
- [ ] **LSG KL threshold sweep**: `python -m nllw.bench --sweep "lsg=5.0,7.0,9.0" --lang en-zh --comet --save`
- [ ] **LSG K sweep**: `python -m nllw.bench --sweep "lsg=7.0 lsgk=1,3,5" --lang en-zh --comet --save`
- [ ] **LSG + shift-k combined**: `python -m nllw.bench --lsg-kl 7.0 --shift-k 0.4 --lang en-zh --comet --save`
- [ ] **LSG + border confirm**: `python -m nllw.bench --lsg-kl 7.0 --border-confirm 2 --lang en-zh --comet --save`
- [ ] **Full iteration 8 sweep**: `python -m nllw.bench --sweep "lsg=5.0,7.0,9.0 shiftk=0.4" --lang en-zh --comet --save`
- [ ] **Complexity-adaptive test**: `python -m nllw.bench --complexity-adaptive --lang en-zh --comet --save` vs fixed bd/wb
- [ ] **Complexity + LSG combined**: `python -m nllw.bench --complexity-adaptive --lsg-kl 7.0 --lang en-zh --comet --save`
- [ ] **Complexity + dynamic-wb combined**: `python -m nllw.bench --complexity-adaptive --dynamic-wb --lang en-zh --comet --save`
- [ ] **REINA entropy change sweep**: `python -m nllw.bench --sweep "entchg=-0.3,-0.5,-1.0,-2.0" --lang en-zh --comet --save`
- [ ] **Prediction stability test**: `python -m nllw.bench --prediction-stability --lang en-zh --comet --save` vs baseline
- [ ] **REINA + stability combined**: `python -m nllw.bench --entropy-change -0.5 --prediction-stability --lang en-zh --comet --save`
- [ ] **REINA + shift-k combined**: `python -m nllw.bench --entropy-change -0.5 --shift-k 0.4 --lang en-zh --comet --save`
- [ ] **REINA + LSG combined**: `python -m nllw.bench --entropy-change -0.5 --lsg-kl 7.0 --lang en-zh --comet --save`
- [ ] **Full iteration 9 sweep**: `python -m nllw.bench --sweep "entchg=-0.5,-1.0 predstab=0,1 shiftk=0.4" --lang en-zh --comet --save`
- [ ] **All cross-step signals**: `python -m nllw.bench --entropy-change -0.5 --prediction-stability --lsg-kl 7.0 --shift-k 0.4 --lang en-zh --comet --save`

## TODO -- Research Ideas (informed by SOTA survey, see docs/research/sota-simulmt-2026.md)

### HIGHEST Priority (ready to implement)
- [x] **LSG logit KL divergence** (IMPLEMENTED, iteration 8): KV cache fork + probe for border confirmation. `--lsg-kl 7.0` CLI. **Needs GPU testing.**

### High Priority (open research gaps, no published work)
- [x] **Novel aggregation** (IMPLEMENTED): 9 methods total -- ts_vote, softmax_mean, entropy_weighted, consensus, geomean, top_p, gaussian_kernel, gaussian_kernel_continuous, ensemble. **Needs GPU testing.**
- [x] **SSBD for alignatt-la** (IMPLEMENTED): `--ssbd-beta 0.2` or `ssbd=0.0,0.1,0.2` sweep. **Needs GPU testing.**
- [x] **NE metric** (IMPLEMENTED): `compute_normalized_erasure()` + revision history tracking.
- [x] **LA forced decoding** (IMPLEMENTED): `--forced-decode` flag. Force-decode committed prefix. **Needs GPU testing.**
- [x] **Adaptive SSBD beta** (IMPLEMENTED): `--adaptive-ssbd` flag. Per-token entropy-based bias. **Needs GPU testing.**
- [x] **Attention entropy as dynamic border distance** (IMPLEMENTED): `--dynamic-border` flag. **Needs GPU testing.**
- [x] **Gaussian kernel consensus** (IMPLEMENTED): 2 variants with sigma parameter. **Needs GPU testing.**
- [x] **LA two-pass catch-up** (IMPLEMENTED): `--two-pass` flag. 2x compute for stability. **Needs GPU testing.**
- [ ] **Aggregation sweep on GPU**: Run full 9-method sweep
- [ ] **SSBD sweep on GPU**: `ssbd=0.0,0.1,0.2,0.3` x `adaptive=0,1`
- [ ] **SSBD + mask-k sweep**: `ssbd=0.0,0.2 mask=0,1,2,3`

### Medium Priority (validated by SOTA papers)
- [x] **Cross-lingual head transfer** (IMPLEMENTED): `python -m nllw.head_transfer --all`. **Results: EuroLLM/HY-MT/Qwen3.5 excellent (>97%), Qwen3-4B good (80%).**
- [x] **LocalAgreement + AlignAtt hybrid** (IMPLEMENTED): `alignatt-la` backend. **Needs GPU testing.**
- [ ] ExPosST-style position slot reservation (arxiv 2603.14903) -- **requires LoRA fine-tuning**, see `docs/research/gpe-exposst-analysis.md`
- [ ] Dynamic word_batch based on source complexity (short sentences -> smaller wb)
- [x] **Adaptive Multi-Strategy (AMS)** (IMPLEMENTED): `--adaptive-agg` flag. Per-token aggregation selection. **Needs GPU testing.**
- [x] **Per-head temperature normalization** (IMPLEMENTED): `--head-temp-norm` flag. Binary search temperature. **Needs GPU testing.**

### Medium-High Priority (new from March 2026 SOTA survey)
- [ ] **GRPO fine-tuning** (SeqPO-SiMT, arxiv 2505.20622): RL-optimize READ/WRITE decisions using BLEU/COMET reward. SimulMT results on 7B LLM rival offline translation.
- [ ] **Syntax-aware chunking** (SASST, arxiv 2508.07781): Replace fixed word_batch with dependency-aware boundaries. Qwen3-8B: +1.2-3.2 BLEU.
- [ ] **SSD parallel speculation** (arxiv 2603.03251): Extend SSBD to predict verification outcomes and pre-compute multiple draft continuations. Up to 2x over standard spec dec.

### NEW High Priority (Iteration 6 SOTA survey, see docs/research/sota-simulmt-2026.md)
- [ ] **Group Position Encoding** (arxiv 2505.16983, ACL 2025): Separate position IDs for source/target. **Requires LoRA fine-tuning** (NOT zero-shot despite claims). Key finding: position mismatch is negligible (<0.14 BLEU) -- validates our KV cache approach. See `docs/research/gpe-exposst-analysis.md`.
- [x] **REINA-inspired entropy change** (IMPLEMENTED, iteration 9): Cross-step entropy change tracking. `--entropy-change -0.5`. Extends REINA info gain (arxiv 2508.04946) with output-space entropy delta. **Needs GPU testing.**
- [ ] **AliBaStr-MT learned border** (arxiv 2503.22051, Apple): Train 6M-param classifier on our TS alignment data. Tunable delta threshold at inference. Replaces heuristic border_distance.
- [ ] **DrFrattn attention-based policy** (EMNLP 2025): Closest published work to our AlignAtt. "Shift-k" mechanism for adaptive thresholds -- must read.
- [ ] **StreamingThinker parallel KV** (arxiv 2510.17238, ICLR 2026): Parallel KV caches decouple source encoding from generation. 80% pre-reasoning latency reduction.
- [ ] **RL-optimized SSBD** (arxiv 2603.01639, ICLR 2026): RL-optimize SSBD beta per-context instead of fixed 0.2. 2.24-4.32x speedup.

### NEW High Priority (March 2026 Research Agent Findings)
- [ ] **SimulU cross-attention policy** (arxiv 2603.16924, March 2026): Training-free policy using cross-attention for long-form S2S. No training needed. Could be adapted for our LLM-based approach.
- [ ] **Hikari WAIT token analysis** (arxiv 2603.11578, March 2026): Policy-free, embeds READ/WRITE into vocabulary. Fundamentally different from AlignAtt. Good competitive intelligence. SOTA on En-Ja/De/Ru.
- [ ] **IWSLT 2026 baseline comparison**: Run our system against official baseline scores (Qwen3-4B). We should significantly outperform since Qwen3-4B is known inferior.
- [ ] **Beam search for low-resource pairs**: CUNI uses 5 beams for Cs-En (their weakest pair). Consider adding beam search option.
- [ ] **Sentence-level buffer trimming**: CUNI trims context differently per target language. Our context injection should be language-adaptive.

### Lower Priority (competitive intelligence)
- [ ] Test Group Position Encoding (ACL 2025) -- needs LoRA fine-tuning. Position mismatch validated as negligible. See analysis.
- [ ] Evaluate SimulSense-style sense unit detection (arxiv 2509.21932) for chunking
- [ ] OmniSTEval integration: end-to-end IWSLT eval pipeline
- [ ] Human-like strategies: SENTENCE_CUT, DROP, PRONOMINALIZATION (arxiv 2601.11002)
- [ ] **LSG KL-divergence policy** (arxiv 2501.00868): Training-free, uses KL(P_partial || P_full) for read/write decisions.
- [ ] **Confidence-modulated speculative decoding** (arxiv 2508.15371): Dynamically adjust draft length based on entropy/margin uncertainty.
- [ ] **SimulSA 1% activation** (arxiv 2509.15692): Minimal SimulMT examples needed to activate streaming capabilities when fine-tuning.
- [ ] **SimulMask SFT** (arxiv 2405.10443, EMNLP 2024): Modify attention mask during LoRA fine-tuning to enforce SimulMT policy. Simpler than EAST.
- [ ] **DPO segmentation** (arxiv 2510.12195): DPO-tune LLM to predict optimal chunk boundaries. Replace fixed word_batch.
- [ ] **RASST retrieval** (arxiv 2601.22777): Retrieval-augmented SimulMT for IWSLT 2026 "Extra Context" subtrack. +3 BLEU, +16% terminology.
- [ ] **PEARL pre-verify** (arxiv 2408.11850, ICLR 2025): Pre-verify first draft token during drafting phase. Extend SSBD pipeline.
- [ ] **Nightjar MAB** (arxiv 2512.22420): Multi-armed bandit to decide whether to use SSBD at all per sentence.

### NEW Iteration 9 Research Findings

- [ ] **Hibiki-Zero GRPO** (arxiv 2602.11072, Feb 2026): Training-free SimulST using GRPO to optimize latency without quality loss. Kyutai team. If training resources become available, most promising RL direction.
- [ ] **ExPosST position slot pre-allocation** (arxiv 2603.14903, March 2026): Pre-allocate KV positions for incoming source tokens, zero recomputation. SOTA results with Llama-3.1-8B. Requires fine-tuning but could eliminate our KV cache overhead.
- [ ] **EASiST end-to-end SimulST** (arxiv 2504.11809): End-to-end approach with explicit attention to synchronize speech and text. Different from our cascaded approach.
- [ ] **SimulS2S-LLM** (arxiv 2504.15509): Speech-to-speech simultaneous translation with LLMs.
- [x] **Translation Heads paper** (OpenReview 2025): Validates our head detection approach -- alignment heads are universal, sparse, cross-lingually consistent. Already confirmed by our `head_transfer.py` results.

### Unported from iwslt26-sst (see audit)

- [ ] **Async ASR-MT pipeline** (`cascade/async_cascade.py`, 725 lines): Full end-to-end pipeline with ASR integration. Important for final system.
- [ ] **Compute-aware latency metrics** (CA-AL, CA-YAAL): Formulas in `research.py` but not wired through bench.py or backends.
- [ ] **LoRA training framework** (`lora/train_lora.py`, 248 lines): Required for ExPosST, AliBaStr, and other fine-tuning approaches.
- [ ] **Per-talk latency variance analysis**: HY-MT has lower latency variance than Qwen3.5 (more stable).
- [ ] **Speech segmentation for long talks**: Pre-process ACL6060 talks into evaluation segments.

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
