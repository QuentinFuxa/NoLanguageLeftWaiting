---
title: "feat: SimulMT Research Forge — Systematic Optimization Pipeline for IWSLT"
type: feat
status: active
date: 2026-03-20
origin: docs/brainstorms/2026-03-20-simulmt-research-forge-requirements.md
---

# SimulMT Research Forge — Systematic Optimization Pipeline

## Overview

Build a multi-agent autonomous research pipeline that systematically optimizes NLLW's simultaneous translation quality for IWSLT competition. The system runs hundreds of experiments across models, parameters, context strategies, and language pairs — testing, measuring, and iterating until we have a diamond-quality SimulMT system.

This plan turns the ad-hoc iwslt26-sst experimental workflow (40+ experiment dirs, 19 figures, 3 benchmark docs built manually over weeks) into a reproducible, automated research machine.

## Problem Statement

NLLW now has 6 backends, evaluation tools, and a web UI — but translations are "good enough," not competition-grade. The gap to IWSLT-competitive quality requires systematic iteration:
- EN→FR BLEU is only ~11 (committed ratio 63% with wb=3)
- Context injection doesn't help HY-MT but helps Qwen3.5 — need model-specific tuning
- No standardized benchmark corpus (only 15-20 test pairs)
- No experiment tracking — results are ephemeral
- No Pareto frontier analysis (quality vs latency tradeoff)

(see origin: docs/brainstorms/2026-03-20-simulmt-research-forge-requirements.md)

## Proposed Solution

A 4-phase implementation with autonomous agent support:

### Phase 1: Research Corpus & Knowledge Base (Foundation)

**Goal:** Curate a 100+ sentence test corpus and extract all findings from iwslt26-sst.

**Unit 1.1: Curated Test Corpus**
- **Files:** `nllw/corpus.py`, `nllw/data/test_corpus.json`
- **Goal:** 100+ sentence pairs across 4 language directions (en-fr, fr-en, en-de, en-zh)
- **Categories:**
  - `short` (2-4 words): "il fait beau", "hello world"
  - `medium` (5-10 words): "I would like to order a coffee"
  - `long` (15+ words): news sentences, conference-style
  - `pronoun_ambiguity`: "il est parti" (he/it depending on context)
  - `idiom`: "il fait beau" (weather idiom), "ça va" (how's it going)
  - `numbers`: dates, currency, statistics
  - `named_entities`: proper names, places, organizations
  - `discourse_dependent`: sentences requiring previous context
  - `reordering`: SOV/SVO order differences
- **Approach:** Combine FLORES_MINI (20 pairs), DEFAULT_TEST_CASES (14 pairs), hand-crafted edge cases (30+), and FLORES+ subset (40+ loaded via HuggingFace datasets if available)
- **Patterns to follow:** `nllw/eval.py:DEFAULT_TEST_CASES` format
- **Verification:** Each pair has source, reference, source_lang, target_lang, category tags

**Unit 1.2: iwslt26-sst Knowledge Extraction**
- **Files:** `docs/research/iwslt26-sst-findings.md`
- **Goal:** Extract ALL experimental findings from the reference repo into a structured document
- **Content:** Model rankings with exact scores, parameter sensitivity data, negative results, best practices, IWSLT submission requirements
- **Approach:** Read BENCHMARK_EN_ZH.md, CASCADE.md, STT_BENCHMARK.md, docs/*.md thoroughly
- **Verification:** Document covers all models tested, all parameter ranges, all negative results

**Unit 1.3: Experiment Schema & Registry**
- **Files:** `nllw/experiment.py`
- **Goal:** Structured YAML/JSON format for experiment configs + results
- **Content:**
  - `ExperimentConfig` dataclass: backend_type, model_path, border_distance, word_batch, context_window, entropy_veto_threshold, prompt_format, lora_path, corpus_name, lang_pair
  - `ExperimentResult` dataclass: config, bleu, comet, xcomet_xl, committed_ratio, ca_al_ms, al_words, per_sentence_results, timestamp, hardware_tag
  - `ExperimentRegistry` class: save/load/query/compare results in `experiments/` directory
  - `nllw run <config.yaml>` CLI entry point
- **Patterns to follow:** iwslt26-sst outputs directory structure
- **Verification:** Can save, load, and query experiment results

### Phase 2: Systematic Parameter Optimization (Core)

**Goal:** Run comprehensive sweeps and identify optimal configs per model/language pair.

**Unit 2.1: Extended Parameter Sweep Infrastructure**
- **Files:** Modify `nllw/eval.py` and `nllw/research.py`
- **Goal:** Sweep ALL parameter dimensions, not just bd × wb
- **Dimensions:**
  - `border_distance`: [2, 3, 4, 5]
  - `word_batch`: [2, 3, 4]
  - `backend_type`: [alignatt, alignatt-la, alignatt-kv]
  - `prompt_format`: [hymt, qwen3, qwen3.5] (model-dependent)
  - `context_window`: [0, 1, 3, 5]
  - `entropy_veto_threshold`: [None, 0.70, 0.75, 0.80]
  - `min_commit` scaling: [n_words//4, n_words//3, 1]
  - `gen_cap` multiplier: [1.0, 1.5, 2.0]
- **Approach:** Use existing `run_parameter_sweep()` but extend param_grid to support all dimensions. Add progress reporting.
- **Verification:** Sweep of 4×3×3 = 36 configs completes and produces ranked results table

**Unit 2.2: Pareto Frontier Analysis**
- **Files:** `nllw/analysis.py`
- **Goal:** Identify quality-latency Pareto optimal configs
- **Content:**
  - `compute_pareto_frontier(results, quality_metric="comet", latency_metric="ca_al_ms")` → list of Pareto-optimal configs
  - ASCII table + optional matplotlib plot
  - Per-language-pair Pareto frontiers
- **Approach:** Simple dominated-solution filtering
- **Verification:** Given sweep results, correctly identifies non-dominated configs

**Unit 2.3: Edge Case Analysis Pipeline**
- **Files:** `nllw/edge_analysis.py`
- **Goal:** For each test sentence, identify WHERE quality breaks down (which word, which step)
- **Content:**
  - For each sentence: replay word-by-word, record per-step (stable, buffer, committed_tokens, entropy)
  - Flag steps where: committed tokens = 0 (no progress), entropy > 0.8 (uncertain), border fires immediately
  - Aggregate: which categories fail most? which language pairs? which backends handle edge cases best?
- **Approach:** Use existing `ResearchBenchmark.benchmark_sentence()` traces
- **Verification:** Produces per-category quality breakdown

### Phase 3: Model-Specific Optimization (Deep)

**Goal:** Fine-tune the pipeline for each target model.

**Unit 3.1: HY-MT 1.5 Optimization**
- **Goal:** Push HY-MT from current quality to competition-grade
- **Focus:** BD sensitivity, gen_cap tuning, sentence-end handling, committed ratio improvement
- **Key finding from iwslt26-sst:** HY-MT with bd=3, wb=2, no context = 0.842 XCOMET-XL (EN→ZH). No context helps. Focus on border detection and commit strategy.
- **Verification:** COMET improvement on FLORES_MINI over baseline

**Unit 3.2: Qwen3.5 Optimization**
- **Goal:** Leverage Qwen3.5's think block and context injection
- **Focus:** Think budget, context window size, prompt format variants (ctx-v1 through v5)
- **Key finding from iwslt26-sst:** Qwen3.5 with ctx=5, think, bd=3, wb=2 = 0.803 XCOMET-XL. Context helps (+0.037).
- **Verification:** COMET improvement on FLORES_MINI with context vs without

**Unit 3.3: Context Injection A/B Testing**
- **Goal:** Quantify context injection impact per model
- **Test matrix:** {HY-MT, Qwen3.5} × {ctx=0, ctx=1, ctx=3, ctx=5} × {en-fr, en-de, en-zh}
- **Special focus:** Pronoun resolution (the train → il = it, not he)
- **Verification:** Clear recommendation per model: use context or not, and which format

### Phase 4: Competition Readiness (Polish)

**Goal:** IWSLT submission-ready output.

**Unit 4.1: OmniSTEval Output Format**
- **Files:** `nllw/omnisteval.py`
- **Goal:** Export results in IWSLT's OmniSTEval format
- **Content:** Convert per-word emission_times + translations to the required JSONL format
- **Approach:** Port `convert_agent_to_simuleval.py` from iwslt26-sst
- **Verification:** Output passes OmniSTEval validation

**Unit 4.2: `nllw bench` CLI**
- **Files:** Entry point in `pyproject.toml`, wraps `nllw/research.py`
- **Goal:** `nllw bench --model X.gguf --suite flores_mini --metrics bleu,comet,al --output report/`
- **Content:** Runs full benchmark, outputs Markdown summary table + JSON results
- **Verification:** One command produces a complete benchmark report

**Unit 4.3: Research Report Generator**
- **Files:** `nllw/report.py`
- **Goal:** Auto-generate a Markdown research report from experiment registry
- **Content:** Model comparison tables, Pareto frontiers, per-language breakdowns, best config recommendations
- **Verification:** Report is publication-ready (LaTeX table export optional)

## Agent Architecture for Autonomous Operation

The forge can run autonomously with 4 cooperating agents:

### Agent 1: Corpus Builder (runs once)
- Generates `test_corpus.json` with 100+ categorized pairs
- Sources: FLORES+, hand-crafted edge cases, iwslt26-sst test data

### Agent 2: Parameter Sweeper (runs repeatedly)
- Takes a config grid + corpus
- Runs `nllw.eval` or `nllw.research` for each config
- Stores results in experiment registry
- Reports: "Config X scored 0.85 COMET, 1200ms CA-AL — new Pareto point"

### Agent 3: Edge Case Investigator (runs in parallel)
- Takes the worst-performing sentences from sweeper results
- Does word-by-word trace analysis
- Reports: "Sentence 'il fait beau' fails at step 3 because entropy=0.92 but border doesn't fire"
- Proposes targeted fixes

### Agent 4: Research Logger (runs continuously)
- Maintains `docs/research/research_log.md`
- For each experiment: Hypothesis → Config → Result → Insight → Next hypothesis
- Cross-references across experiments to find patterns
- Reports: "Context injection helps Qwen3.5 by +0.037 but hurts HY-MT by -0.028. Insight: HY-MT's alignment heads are already strong enough that context noise outweighs the benefit."

## Acceptance Criteria

- [ ] Test corpus with 100+ sentence pairs, categorized, covering all edge case types (R1)
- [ ] Experiment registry: save, load, query, compare experiment results (R5, R8)
- [ ] Parameter sweep across bd × wb × backend × prompt_format × ctx × entropy (R3)
- [ ] Pareto frontier analysis: quality vs latency for each model/lang pair (R7)
- [ ] Edge case analysis pipeline: per-sentence, per-step breakdown (R7)
- [ ] Context injection A/B results for HY-MT and Qwen3.5 (R4)
- [ ] Measurable COMET improvement over current baseline (Success Criteria)
- [ ] Research log with hypothesis → experiment → insight chain (R5)
- [ ] `nllw bench` CLI produces complete report in one command (R8)
- [ ] OmniSTEval output format for IWSLT submission (Competition readiness)

## Dependencies & Risks

- **GPU access:** A40 machine at quest.ms.mff.cuni.cz (SSH key auth set up). Needed for xCOMET-XL scoring and Qwen3.5 inference.
- **COMET dependency:** `unbabel-comet` requires ~2GB (wmt22) or ~14GB (XCOMET-XL) VRAM.
- **FLORES+ dependency:** `datasets` library for HuggingFace dataset loading (optional — fallback to built-in corpus).
- **Risk: Experiment runtime:** Full sweep of 36+ configs × 100 sentences = ~3600 translate calls × ~500ms = ~30 min per sweep. Multiple sweeps needed.
- **Risk: Model reload time:** Switching backend_type requires model reload (~10s). Batch configs that share the same model.

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| EN→FR COMET | ~0.75 (est) | 0.82+ |
| FR→EN COMET | ~0.85 (est) | 0.88+ |
| EN→FR committed ratio | 63% | 75%+ |
| Experiment configs tested | ~10 | 200+ |
| Test corpus size | 20 | 100+ |
| Pareto-optimal configs documented | 0 | 5+ per lang pair |

## Sources & References

### Origin
- **Origin document:** [docs/brainstorms/2026-03-20-simulmt-research-forge-requirements.md](docs/brainstorms/2026-03-20-simulmt-research-forge-requirements.md) — Agent architecture (4 agents), scope (EN→FR, FR→EN, EN→DE, EN→ZH), success criteria
- Key decisions carried forward: systematic parameter sweeps, reproducible experiment registry, autonomous agent operation

### Internal References
- Architecture ideation: `docs/ideation/2026-03-20-alignatt-simulmt-architecture-ideation.md`
- Optimal parameters: `memory/project_alignatt_optimization.md` — bd=3, wb=3, adaptive gen_cap, entropy veto
- Eval infrastructure: `nllw/eval.py`, `nllw/research.py`, `nllw/simulate.py`
- Backend protocol: `nllw/backend_protocol.py`
- Metrics: `nllw/metrics.py` (BLEU, COMET, xCOMET-XL)
- Web API: `web_debug/server.py` (/compare, /evaluate endpoints)

### External References
- iwslt26-sst experimental results: `/Users/quentin/Documents/repos/iwslt26-sst/SimulMT_tests/`
- IWSLT26 SST track: OmniSTEval format
- AlignAtt paper: Polák et al., ICLR 2026
- HY-MT champion config: bd=3, wb=2, no context → 0.842 XCOMET-XL (EN→ZH)
