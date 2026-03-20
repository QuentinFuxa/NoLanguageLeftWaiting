---
date: 2026-03-20
topic: simulmt-research-forge
---

# SimulMT Research Forge — Systematic Optimization for IWSLT

## Problem Frame

We have a working SimulMT system (NLLW) with 6 backends, evaluation tools, and a web UI. But the translations are "good enough" not "diamond quality." The gap between current output and IWSLT-competitive output requires **systematic research iteration** — not one-off fixes but hundreds of experiments across models, parameters, context strategies, and language pairs.

The iwslt26-sst repo proved this approach works (40+ experiments, 19 figures, 3 benchmark docs) but the work was ad-hoc. We need to formalize it into an autonomous research pipeline.

## Requirements

- R1. Build a curated test corpus of 100+ sentence pairs covering edge cases (pronoun ambiguity, idioms, numbers, named entities, long sentences, short phrases, mixed-register)
- R2. Create an orchestrator agent that plans experiments, dispatches test agents, collects results, and identifies the highest-leverage improvements
- R3. Implement systematic parameter sweeps: border_distance × word_batch × backend_type × prompt_format × context_window × entropy_threshold — with COMET/xCOMET scoring
- R4. Test context injection effectiveness per model (HY-MT doesn't benefit, Qwen3.5 does — quantify this properly)
- R5. Build a research log that captures every experiment configuration + results + insights, growing over time
- R6. The system should be able to run autonomously for hours, testing hypotheses and reporting findings
- R7. Focus on the simultaneous translation specific challenges: partial-source quality, border detection timing, latency-quality Pareto frontier
- R8. All results must be reproducible — any experiment can be re-run with one command

## Success Criteria

- Measurable COMET/xCOMET improvement over current baseline on a standard test set
- Clear Pareto frontier visualization: quality vs latency across all tested configs
- Documented set of "best practices" for each model/language pair
- Ready for IWSLT competition submission

## Scope Boundaries

- NOT training new models or LoRA adapters (that's a separate workstream)
- NOT ASR integration (cascade is separate)
- Focus on EN→FR, FR→EN, EN→DE, EN→ZH as primary test directions

## Key Decisions

- Use NLLW's existing eval/research/simulate infrastructure (already built today)
- Test on local Mac (Metal) for iteration, A40 for final benchmarks
- Agent architecture: orchestrator + corpus builder + parameter sweeper + insight analyzer

## Agent Architecture

### Agent 1: Corpus Builder
- Creates test_corpus.json with 100+ carefully designed sentence pairs
- Categories: short (2-4 words), medium (5-10), long (15+), ambiguous pronouns, idioms, numbers, named entities, discourse-dependent
- Each pair has source, reference, category tags, and known failure modes

### Agent 2: Parameter Sweeper
- Runs systematic grid search: bd × wb × backend × prompt_format
- Records COMET + committed_ratio + time_per_word for each config
- Outputs ranked results table

### Agent 3: Edge Case Investigator
- Focuses on the hard cases: "il fait beau", "je suis parti", pronoun resolution
- Tests word-by-word output at each step to identify WHERE quality breaks down
- Proposes targeted fixes (context injection, gen_cap tuning, etc.)

### Agent 4: Research Logger
- Maintains a running research log (markdown) with:
  - Hypothesis → Experiment → Result → Insight → Next hypothesis
- Cross-references findings across experiments
- Identifies patterns and suggests the highest-leverage next experiment

## Outstanding Questions

### Deferred to Planning
- [Affects R3][Needs research] What is the optimal COMET threshold for "good enough" simultaneous translation?
- [Affects R4][Technical] Should context injection use the model's native format or a generic [Previous translations] block?
- [Affects R6][Technical] How to handle model reload time during long autonomous runs?

## Next Steps
→ /ce:plan for structured implementation planning, then launch the agent team
