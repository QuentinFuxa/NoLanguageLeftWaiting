# Shared Task Notes -- NLLW SimulMT

## What exists now (after Iteration 10, 2026-03-20)

**23 SimulMT modules (~10300 lines), 495 tests passing:**

### Core (Iterations 1-7):
- `nllw/prompts.py` -- 30+ prompt formats (HY-MT, Qwen3, Qwen3.5, EuroLLM, Tower, Gemma)
- `nllw/llama_backend.py` -- ctypes wrapper for llama.cpp (attention extraction + KV cache)
- `nllw/backend_protocol.py` -- SimulMTBackend ABC + factory + `@register_backend`
- `nllw/alignatt.py` -- Core algorithm + 10 aggregation + AMS + temp norm + shift-k + info gain + LSG logit KL
- `nllw/alignatt_backend.py` -- Full backend with KV cache reuse + LSG probe + baselines
- `nllw/alignatt_la_backend.py` -- LA + AlignAtt + SSBD + forced decode + adaptive SSBD + two-pass + LSG
- `nllw/metrics.py` -- All latency metrics + BLEU/COMET + NE metric
- `nllw/bench.py` -- Unified CLI with 18+ sweep shortnames
- `nllw/baselines.py` -- wait-k + fixed-rate baselines
- `nllw/heads/configs/` -- 22 pre-computed alignment head configs
- Plus: `eval.py`, `simulate.py`, `corpus.py`, `experiment.py`, `analysis.py`, `research.py`, `detect_heads.py`, `omnisteval.py`, `head_transfer.py`, `complexity.py`
- `web_debug/server.py` -- FastAPI debug server on port 8777

### New in Iteration 8:
- **LSG Logit KL Divergence** (arxiv 2501.00868, AAAI 2025):
  - `compute_logit_kl()` in alignatt.py: stable softmax KL(P_full || P_reduced)
  - KV cache fork + probe in AlignAttBackend._lsg_probe():
    1. Get full-source logits (already available)
    2. Fork KV cache to seq_id=1
    3. Remove last `lsg_k` source token positions from fork
    4. Re-decode last token on fork (without those source tokens)
    5. Compare logit distributions via KL divergence
    6. Clean up fork
  - Integrated as border CONFIRMATION: when attention says "stop", LSG confirms/denies
    - Low KL (< threshold): source exhausted -> confirm stop (WRITE)
    - High KL (> threshold): source still matters -> override stop, keep generating (READ)
  - Wired into both AlignAtt and AlignAtt-LA backends (all 3 retranslation methods)
  - `--lsg-kl 7.0 --lsg-k 3` CLI flags
  - `lsg=5.0,7.0,9.0 lsgk=1,3,5` sweep shortnames
  - Overhead: ~1 extra forward pass per border check
  - 27 new tests for LSG
- **Complexity-Adaptive Parameters** (wires `complexity.py` into backends):
  - Per-sentence text feature analysis: word count, avg length, numeral density, subword ratio
  - Simple sentences -> reduced bd/wb (faster latency)
  - Complex sentences -> increased bd/wb (safer, higher quality)
  - Applied BEFORE dynamic word_batch (stacks with it)
  - In AlignAtt: overrides `effective_bd`, `effective_wb`, `effective_gen_cap`
  - In LA: sets `_effective_bd` used by `_check_border()`
  - `--complexity-adaptive` CLI, `cmplx=0,1` sweep shortname
  - 9 new tests for complexity-adaptive
- 36 new tests total (396 all passing)

### New in Iteration 9:
- **Entropy change tracking** (REINA-inspired, arxiv 2508.04946, AAAI 2026):
  - `compute_entropy_change()` in alignatt.py: track H(first_token) across translate() calls
  - If new source word caused large entropy drop -> model is learning -> inhibit border stop (READ)
  - If entropy change is small -> source exhausted -> allow border stop (WRITE)
  - Pre-filter in `check_border_combined()`: fires before attention checks
  - `--entropy-change -0.5` CLI, `entchg=-0.5,-1.0` sweep shortname
  - Zero overhead: reuses existing logits
- **Prediction stability index** (novel, no published work):
  - `compute_prediction_stability()`: top-1 rank change + top-K Jaccard overlap
  - Stable predictions (rank <= 3, overlap >= 0.4) -> model has enough context -> WRITE
  - Volatile predictions -> model still adapting -> override border stop -> READ
  - Post-filter in `check_border_combined()`: fires after attention checks
  - `--prediction-stability` CLI, `predstab=0,1` sweep shortname
  - Overhead: one logits copy per translate() call
- **Cross-step signal architecture**: both signals computed once per translate() call, stored
  as instance variables, consumed by `_check_border()`, reset on segment boundary
- 45 new tests (441 all passing)

### New in Iteration 10:
- **Source coverage guard** (novel, no published work):
  - `compute_source_coverage()` in alignatt.py: TS-weighted coverage ratio
  - Tracks what fraction of source positions are well-attended during generation
  - Low coverage = hallucination (model generating from priors, not source)
  - Integrated as early guard in `check_border_combined()` (step 0b)
  - `--coverage-threshold 0.3` CLI, `cov=0.2,0.3,0.4` sweep shortname
  - Zero overhead: reuses existing attention weights
- **Attention monotonicity** (novel, no published work):
  - `compute_attention_monotonicity()`: Kendall tau-like score [-1, 1]
  - Tracks how monotonically attention moves forward through source during generation
  - `monotonicity_border_adjustment()`: adapt border distance per generation context
    - Monotonic (>0.7) -> tighter border (bd-1), non-monotonic (<0) -> wider border
  - Position history tracked per generation loop, reset per translate()/retranslate()
  - `--attention-monotonicity` CLI, `mono=0,1` sweep shortname
  - Minimal overhead: one float per generation step
- Both signals wired into AlignAtt and AlignAtt-LA backends
- 54 new tests (495 all passing)

### New in Iteration 11:
- **N-gram repetition detection** (novel, no published work):
  - `detect_ngram_repetition()` in alignatt.py: check 2-4gram repetition in generated tokens
  - `compute_repetition_score()`: continuous 0-1 quality score for generation monitoring
  - Within-step, output-space signal -- catches degenerate repetitive hallucination loops
  - Wired into AlignAtt backend (main loop) + all 3 LA retranslation methods (standard, forced, SSBD)
  - `--repetition-halt 2` CLI, `rep=2,3,4` sweep shortname
  - Zero overhead: simple tuple matching on last few tokens
- **SimulStream wrapper** (`nllw/simulstream.py`):
  - `NLLWSpeechProcessor` wraps any NLLW backend as SimulStream SpeechProcessor
  - Audio mode: `process_chunk(waveform)` -> buffer -> ASR -> MT -> IncrementalOutput
  - Text mode: `process_words(words)` -> MT -> IncrementalOutput (for eval/testing)
  - `end_of_stream()`, `clear()` for stream lifecycle
  - `SimulStreamConfig` with YAML support, `DIRECTION_DEFAULTS` for all 4 IWSLT directions
  - `process_gold_transcript()`: JSONL-in/JSONL-out eval pipeline
  - CLI: `python -m nllw.simulstream --model X --lang en-zh --test`
- **Attention shift tracking** (novel, no published work):
  - `compute_attention_shift()` in alignatt.py: TS-weighted position delta across translate() calls
  - Cross-step, input-space signal -- completes the signal taxonomy
  - Large shift = model consuming source (WRITE). Small = stuck (READ more)
  - Pre-filter in `check_border_combined()` (step 0a2, after REINA, before coverage)
  - Wired into AlignAtt and AlignAtt-LA backends
  - `--attention-shift` CLI, `attshift=0,1` sweep shortname
- **Dockerfile** for IWSLT 2026 submission (multi-stage, CUDA 12.4, H100)
- **Per-direction YAML configs** for all 4 IWSLT 2026 directions
- 86 new tests (581 all passing)

### New in Iteration 12:
- **Weighted signal fusion framework** (`nllw/fusion.py`, ~600 lines):
  - Novel: no published work on weighted multi-signal fusion for SimulMT border detection
  - Replaces boolean cascade (`check_border_combined()`) with principled weighted scoring
  - 8 signal scorers, each producing continuous [-1, +1] WRITE confidence:
    - `score_standard_border()`: AlignAtt argmax position (foundation)
    - `score_shift_k()`: Attention mass in border region
    - `score_info_gain()`: Attention KL divergence (low = source exhausted)
    - `score_coverage()`: Source coverage (low = hallucination = force stop)
    - `score_monotonicity()`: Attention movement regularity
    - `score_entropy_change()`: REINA entropy delta
    - `score_pred_stability()`: Prediction stability
    - `score_attn_shift()`: Cross-step attention position shift
  - `FusionWeights` dataclass with per-direction defaults (en-zh, en-de, en-it, cs-en, en-fr)
  - `fused_border_check()`: normalized weighted sum -> threshold decision
  - `FusionDiagnostic`: per-signal scores, weights, contributions, dominant signal
  - `calibrate_threshold()`: find optimal threshold from labeled examples
  - `grid_search_weights()`: optimize weights via grid search + F1 score
  - Key advantage: weak signals combine (two marginal signals trigger stops neither would alone)
  - Wired into AlignAtt backend (main loop) + LA backend (`_check_border()`)
  - `--signal-fusion` CLI, `fusion=0,1` sweep shortname
  - `--fusion-threshold` CLI, `fthr=-0.2,0.0,0.2` sweep shortname
- **GPU experiment runner** (`scripts/run_experiments.sh`):
  - Phase 1: Basic validation (~5 min)
  - Phase 2: Signal sweeps (~30 min)
  - Phase 3: Full Pareto + multi-direction (~2 hours)
  - Phase 4: Fusion experiments (~30 min)
  - Phase 5: All above
  - Auto-logging, machine-aware, COMET-optional
- 72 new tests (653 all passing)

## What to do next

### Priority 1: Run experiments on A40 (ALL code is ready, NOTHING tested on GPU yet)

```bash
# 1a. First E2E validation
python -m nllw.bench --model /path/to/HY-MT1.5-7B.gguf --lang en-zh -n 20

# 1b. LSG sweep (new in iteration 8)
python -m nllw.bench --sweep "lsg=5.0,7.0,9.0" --lang en-zh --comet --save
python -m nllw.bench --sweep "lsg=7.0 lsgk=1,3,5" --lang en-zh --comet --save
python -m nllw.bench --lsg-kl 7.0 --shift-k 0.4 --lang en-zh --comet --save
python -m nllw.bench --lsg-kl 7.0 --border-confirm 2 --lang en-zh --comet --save

# 1b2. Complexity-adaptive (new in iteration 8)
python -m nllw.bench --complexity-adaptive --lang en-zh --comet --save
python -m nllw.bench --complexity-adaptive --lsg-kl 7.0 --lang en-zh --comet --save
python -m nllw.bench --complexity-adaptive --dynamic-wb --lang en-zh --comet --save

# 1c. Full aggregation sweep (10 methods)
python -m nllw.bench --sweep "agg=ts_vote,softmax_mean,entropy_weighted,consensus,geomean,top_p,gaussian_kernel,gaussian_kernel_continuous,cumulative,ensemble" --lang en-zh --comet --save

# 1d. Iteration 7 features
python -m nllw.bench --dynamic-wb --lang en-zh --comet --save
python -m nllw.bench --sweep "shiftk=0.3,0.4,0.5,0.6" --lang en-zh --comet --save
python -m nllw.bench --sweep "infogain=0.2,0.3,0.5" --lang en-zh --comet --save
python -m nllw.bench --shift-k 0.4 --info-gain 0.3 --lang en-zh --comet --save

# 1e. AlignAtt vs AlignAtt-LA comparison
python -m nllw.bench --compare alignatt alignatt-la --lang en-zh --comet --save

# 1f. SSBD sweep
python -m nllw.bench --backend alignatt-la --sweep "ssbd=0.0,0.1,0.2,0.3" --lang en-zh --comet --save

# 1g. Iteration 6 features
python -m nllw.bench --sweep "ams=0,1 tempnorm=0,1" --lang en-zh --comet --save

# 1h. Pareto sweep
python -m nllw.bench --sweep "bd=2,3,4,5 wb=1,2,3" --lang en-zh,en-de --comet --save

# 1i. REINA entropy change (new in iteration 9)
python -m nllw.bench --sweep "entchg=-0.3,-0.5,-1.0,-2.0" --lang en-zh --comet --save
python -m nllw.bench --entropy-change -0.5 --prediction-stability --lang en-zh --comet --save
python -m nllw.bench --entropy-change -0.5 --shift-k 0.4 --lang en-zh --comet --save
python -m nllw.bench --entropy-change -0.5 --lsg-kl 7.0 --lang en-zh --comet --save

# 1j. Full cross-step signals (new in iteration 9)
python -m nllw.bench --entropy-change -0.5 --prediction-stability --lsg-kl 7.0 --shift-k 0.4 --lang en-zh --comet --save
python -m nllw.bench --sweep "entchg=-0.5,-1.0 predstab=0,1" --lang en-zh --comet --save

# 1k. Source coverage guard (new in iteration 10)
python -m nllw.bench --sweep "cov=0.2,0.3,0.4,0.5" --lang en-zh --comet --save
python -m nllw.bench --coverage-threshold 0.3 --shift-k 0.4 --lang en-zh --comet --save
python -m nllw.bench --coverage-threshold 0.3 --lsg-kl 7.0 --lang en-zh --comet --save

# 1l. Attention monotonicity (new in iteration 10)
python -m nllw.bench --attention-monotonicity --lang en-zh --comet --save
python -m nllw.bench --attention-monotonicity --coverage-threshold 0.3 --lang en-zh --comet --save

# 1m. Full iteration 10 sweep
python -m nllw.bench --sweep "cov=0.3 mono=0,1 shiftk=0.4" --lang en-zh --comet --save
python -m nllw.bench --coverage-threshold 0.3 --attention-monotonicity --entropy-change -0.5 --prediction-stability --shift-k 0.4 --lsg-kl 7.0 --lang en-zh --comet --save

# 1n. N-gram repetition detection (new in iteration 11)
python -m nllw.bench --repetition-halt 2 --lang en-zh --comet --save
python -m nllw.bench --sweep "rep=2,3,4" --lang en-zh --comet --save
python -m nllw.bench --repetition-halt 2 --shift-k 0.4 --coverage-threshold 0.3 --lang en-zh --comet --save

# 1o. All signals combined (iterations 7-11)
python -m nllw.bench --coverage-threshold 0.3 --attention-monotonicity --entropy-change -0.5 --prediction-stability --shift-k 0.4 --lsg-kl 7.0 --repetition-halt 2 --lang en-zh --comet --save

# 1p. Signal fusion (new in iteration 12)
python -m nllw.bench --signal-fusion --lang en-zh --comet --save
python -m nllw.bench --signal-fusion --shift-k 0.4 --coverage-threshold 0.3 --lang en-zh --comet --save
python -m nllw.bench --signal-fusion --sweep "fthr=-0.2,0.0,0.2,0.4" --lang en-zh --comet --save

# 1q. Fusion vs boolean cascade (same signals, different decision logic)
python -m nllw.bench --shift-k 0.4 --coverage-threshold 0.3 --lang en-zh --comet --save
python -m nllw.bench --signal-fusion --shift-k 0.4 --coverage-threshold 0.3 --lang en-zh --comet --save

# 1r. Fusion per direction
python -m nllw.bench --signal-fusion --shift-k 0.4 --coverage-threshold 0.3 --lang en-de --comet --save
python -m nllw.bench --signal-fusion --shift-k 0.4 --coverage-threshold 0.3 --lang en-it --comet --save
python -m nllw.bench --signal-fusion --shift-k 0.4 --coverage-threshold 0.3 --lang cs-en --comet --save

# 1s. Full fusion with all signals
python -m nllw.bench --signal-fusion --coverage-threshold 0.3 --attention-monotonicity --entropy-change -0.5 --prediction-stability --shift-k 0.4 --lsg-kl 7.0 --repetition-halt 2 --attention-shift --lang en-zh --comet --save

# OR use the experiment runner script:
./scripts/run_experiments.sh 4 --lang en-zh --model /path/to/model.gguf --comet
```

### Priority 2: IWSLT 2026 Competition Integration (URGENT -- eval April 1-15)
- **SimulStream wrapper**: DONE (iteration 11). `nllw/simulstream.py` wraps NLLW as SimulStream SpeechProcessor. Needs ASR integration + GPU E2E test.
- **OmniSTEval compatibility**: Verify `omnisteval.py` output with latest `omnisteval longform` + XCOMET-XL
- **Docker on H100 80GB**: Q8_0 7B GGUF = ~8GB, package llama.cpp + model + NLLW
- **EN->IT is new for 2026** (replaces EN->JP). We have corpus + head configs ready.
- **ASR integration**: Wire Qwen3-ASR into `NLLWSpeechProcessor._run_asr()` for full audio pipeline.
- **Key strategic insight**: CUNI won 2025 but could NOT use AlignAtt with EuroLLM (no attention extraction in CTranslate2). We CAN use AlignAtt with ANY model via llama.cpp -- unique advantage.

### Priority 3: Further optimizations
- **ExPosST position slots**: Pre-allocate KV positions for zero-recomputation
- **SSD parallel speculation**: Multiple draft continuations for 2x speedup
- **Group Position Encoding**: Separate position IDs for source/target (ACL 2025)

### Priority 4: Research ideas (see todo.md)
- GRPO fine-tuning (RL-optimize read/write)
- Syntax-aware chunking (SASST)
- REINA information gain policy (AAAI 2026)
- AliBaStr-MT learned border (Apple)

## Key architecture decisions

- `PromptFormat` is a frozen dataclass with `build_prompt()` method
- `BackendConfig.from_dict()` ignores unknown keys (safe for sweep configs)
- `@register_backend("name")` decorator auto-registers with factory
- FLORES+ uses `openlanguagedata/flores_plus` with per-language configs
- `detect_heads` uses SimAlign (mBERT) for ground-truth word alignments
- `aggregate()` dispatcher routes to any of 10 aggregation strategies
- LA backend retranslation priority: SSBD > forced_decode > standard
- Two-pass uses pass 1 (SSBD/forced/standard) + pass 2 (always standard) for diversity
- `_check_border()` helper centralizes all border check params (AMS, temp norm, dynamic, shift-k, info gain, REINA, stability)
- `check_border_combined()` fuses multiple signals via boolean cascade: standard + shift-k mass + info gain + entropy change + prediction stability + source coverage guard + attention monotonicity
- `fused_border_check()` (iter 12) replaces boolean cascade with weighted scoring: each signal -> [-1,+1] score -> weighted sum -> threshold decision. Per-direction weight profiles.
- Cross-lingual head transfer: most models have >90% TS mass transfer across pairs
- **LSG probe uses seq_id=1 fork**: `memory_seq_cp(0, 1)` + `memory_seq_rm(1, ...)` + re-decode + cleanup. Zero impact on main seq 0.
- **Cross-step signals**: entropy change and prediction stability are computed once per translate() call (not per generation step). They modulate the border decision from attention-based checks.

## Important context

- llama.cpp must be built with PR #20086. The build is on A40.
- The logit_idx bug (batch index 0, not KV pos) is fixed in `alignatt_backend.py`.
- YAAL formula: `gamma = max(|delays|, T) / S; yaal = sum(d - t/gamma) / tau`
- FLORES loading is tested and working locally.
- 5 directions available: en-fr, en-zh, en-de, en-it, cs-en
- **7 backends registered**: alignatt, alignatt-la, full-sentence, eager, wait-k, fixed-rate
- **10 aggregation methods**: ts_vote, softmax_mean, entropy_weighted, consensus, geomean, top_p, gaussian_kernel, gaussian_kernel_continuous, cumulative, ensemble
- **New iter8 params**: `lsg_kl_threshold` (None=disabled, 7.0=recommended), `lsg_k` (3=default), `complexity_adaptive` (False=disabled)
- **New iter8 sweeps**: `lsg=5.0,7.0,9.0`, `lsgk=1,3,5`, `cmplx=0,1`
- **New iter9 params**: `entropy_change_threshold` (None=disabled, -0.5=recommended), `prediction_stability` (False=disabled)
- **New iter9 sweeps**: `entchg=-0.3,-0.5,-1.0`, `predstab=0,1`
- **New iter10 params**: `coverage_threshold` (None=disabled, 0.3=recommended), `attention_monotonicity` (False=disabled)
- **New iter10 sweeps**: `cov=0.2,0.3,0.4`, `mono=0,1`
- **New iter11 params**: `repetition_max_repeats` (None=disabled, 2=recommended), `attention_shift` (False=disabled)
- **New iter11 sweeps**: `rep=2,3,4`, `attshift=0,1`
- **New iter11 module**: `nllw/simulstream.py` -- SimulStream SpeechProcessor wrapper for IWSLT 2026
- **New iter12 params**: `signal_fusion` (False=disabled, True=use weighted fusion), `fusion_threshold` (0.0=balanced)
- **New iter12 sweeps**: `fusion=0,1`, `fthr=-0.2,0.0,0.2,0.4`
- **New iter12 module**: `nllw/fusion.py` -- Weighted signal fusion (8 scorers + FusionWeights + FusionDiagnostic + calibration)
- **New iter12 script**: `scripts/run_experiments.sh` -- Automated 4-phase GPU experiment runner
- **Key insight (iter11)**: Attention shift is a CROSS-STEP, INPUT-SPACE signal -- the only quadrant missing from our taxonomy. It measures how much the model's SOURCE focus changes between consecutive translate() calls, orthogonal to within-step signals and output-space signals.
- **Key insight**: Entropy change and prediction stability are OUTPUT-SPACE cross-step signals, orthogonal to attention-based within-step signals. They measure whether the MODEL'S BEHAVIOR changed after adding a new source word.
- **Key insight (iter10)**: Source coverage is an INPUT-SPACE within-step signal -- it measures whether the model is grounded in ALL the source, not just which position it attends to. Orthogonal to argmax-based border detection.
- **Key insight (iter10)**: Attention monotonicity is a TEMPORAL within-step signal -- it measures attention DYNAMICS across generation steps, not static attention at a single step.
- **LSG implementation**: KV cache fork (seq_cp + seq_rm) + single-token re-decode + logit KL. ~1-3ms per probe.
- **Key insight**: LSG is ORTHOGONAL to attention-based border: attention = WHERE model looks, logit KL = WHETHER removing source changes OUTPUT. Combining both gives stronger signal.
- SOTA research documented in `docs/research/sota-simulmt-2026.md`

## Key research findings

- **Cross-lingual head transfer confirmed**: 4/5 models tested show >90% TS mass transfer
- **AlignAtt validated**: CUNI won IWSLT 2025 using AlignAtt
- **SSBD** implemented -- speculative re-translation speedup
- **LSG** implemented -- training-free logit KL for border confirmation (AAAI 2025)
- **Open gap**: No published work on attention aggregation selection (our AMS is novel)
- **Open gap**: No published work on attention mass border detection (our shift-k is novel)
- **Open gap**: No published work on info gain modulation for SimulMT border detection
- **Open gap**: No published work on combining logit KL + attention border (our LSG integration is novel)
- **Open gap**: No published work on cross-step prediction stability for SimulMT (our prediction stability index is novel)
- **Open gap**: No published work on entropy change as cross-step border modulation (our REINA integration is novel extension)
- **Open gap**: No published work on attention coverage as hallucination guard for SimulMT (our coverage guard is novel)
- **Open gap**: No published work on attention monotonicity scoring for decoder-only LLM SimulMT (our monotonicity metric is novel)
- **Open gap (iter11)**: No published work on n-gram repetition detection as halt signal in SimulMT
- **Open gap (iter11)**: No published work on cross-step attention position shift for SimulMT border detection
- **Signal taxonomy (iteration 11 -- COMPLETE)**:
  - Within-step, position-based: standard AlignAtt, shift-k mass, info gain, AMS
  - Within-step, input-coverage: source coverage guard (iter 10)
  - Within-step, temporal: attention monotonicity (iter 10)
  - Within-step, output-space: n-gram repetition detection (iter 11)
  - Cross-step, input-space: attention shift tracking (iter 11) -- COMPLETES TAXONOMY
  - Cross-step, output-space: entropy change (REINA), prediction stability
  - Cross-step, output-logit: LSG logit KL divergence
- **Open gap (iter12)**: No published work on weighted multi-signal fusion for SimulMT border detection. Closest: DrFrattn (single aggregated signal), LSG (independent logit KL check)
- **Key insight (iter12)**: Boolean cascade has fundamental limitation: signal ORDER determines priority, and two "weak" signals can never combine. Weighted fusion fixes both: order-independent, weak signals combine naturally.
- **Hikari** is main competitor: policy-free WAIT tokens
- **IWSLT 2026 metrics**: LongYAAL (primary latency), XCOMET-XL (primary quality)
