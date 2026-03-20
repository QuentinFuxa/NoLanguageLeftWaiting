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
- [x] `nllw/corpus.py` -- Categorized test corpus (30 EN-FR + 8 EN-ZH, 7 categories)

---

## IN PROGRESS -- Evaluation & Benchmarking

- [ ] `nllw/omnisteval.py` -- OmniSTEval JSONL output for IWSLT 2026 submission
- [ ] Expand corpus.py to 130+ sentences (add more EN-ZH, EN-DE, EN-IT, CS-EN)

## TODO -- Research Tools

- [ ] `nllw/research.py` -- Compute-aware latency (CA-AL), benchmark suite
- [ ] `nllw/experiment.py` -- Experiment config/result registry, Pareto analysis
- [ ] `nllw/analysis.py` -- Pareto frontier, edge cases, report generation
- [ ] `nllw/detect_heads.py` -- Auto alignment head detection for any GGUF model

## TODO -- Additional Backends

- [ ] `nllw/alignatt_la_backend.py` -- LocalAgreement hybrid (re-translate + diff for stability)
- [ ] `nllw/baselines.py` -- wait-k baseline with proper read/write policy

## TODO -- Infrastructure

- [ ] Web debug server (FastAPI + Ollama-style UI)
- [ ] MCP server for editor integration
- [ ] LoRA adapter loading + discovery
- [ ] Context injection (rolling buffer of previous translations)

## TODO -- Experiments to Run

- [ ] Baseline EN-FR quality on A40 with HY-MT1.5-7B
- [ ] Compare AlignAtt vs wait-k vs full-sentence on FLORES mini
- [ ] Test Qwen3.5-4B with context injection
- [ ] Multi-direction sweep: en-zh, en-de, en-it, cs-en
- [ ] KV cache speedup measurement (with vs without)
- [ ] Entropy veto threshold tuning

## TODO -- Research Ideas

- [ ] Explore better aggregation than TS-weighted argmax (attention entropy? softmax?)
- [ ] Test speculative generation in AlignAtt (generate ahead, validate)
- [ ] Investigate LocalAgreement + AlignAtt hybrid
- [ ] Dynamic word_batch based on source complexity
- [ ] Multi-pass alignment refinement
- [ ] Cross-lingual transfer of alignment heads

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
