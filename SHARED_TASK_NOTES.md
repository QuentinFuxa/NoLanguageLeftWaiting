# Shared Task Notes -- NLLW SimulMT

## Current State (after Iteration 14, 2026-03-21)

**28 SimulMT modules (~13,000 lines), 731+ tests passing**
**FIRST GPU E2E TEST PASSED on A40 with HY-MT1.5-7B**

### What happened in Iteration 14
- **GPU E2E validated**: NLLW deployed to A40, full AlignAtt pipeline works with HY-MT1.5-7B.Q8_0.gguf
- **3 critical bugs fixed** (see below)
- **Research update**: Latest SimulMT papers reviewed (Hikari, ExPosST, RASST, SeqPO-SiMT, SimulU, Translation Heads ICLR 2026)
- All tests pass locally (731+)

### Critical Bugs Fixed (Iteration 14)
1. **Attention stride bug** (`llama_backend.py`): `get_attn_weights()` used caller-provided `ctx_size` (current position) as stride between heads. The actual C layout is `n_heads * n_ctx` (full context window). Fixed: stride is now always `n_ctx(ctx)`. Without this fix, all heads except the first read garbage data.
2. **n_gpu_layers missing** (`backend_protocol.py`, `alignatt_backend.py`, `alignatt_la_backend.py`): `BackendConfig` had no `n_gpu_layers` field, defaulting to CPU-only. Added and wired into both backends.
3. **Backend auto-import** (`backend_protocol.py`): `create_backend()` failed with "Unknown backend type" because modules with `@register_backend` decorators were never imported. Added `_ensure_backends_imported()`.

### GPU Performance (A40, HY-MT1.5-7B.Q8_0.gguf)
| Metric | Value |
|--------|-------|
| Model load | 2.3s |
| Prompt throughput | 136 tok/s |
| Generation throughput | 37-39 tok/s |
| VRAM | ~8.5 GB |
| Latency per word | ~140ms |
| SimulMT quality | Good (needs tuning) |

### Sample Translation (bd=3, wb=2)
```
"The president of the United States announced new economic policies today"
-> 总统宣布了新的经济政策。 (1528ms, 7.2 words/s)
```
Note: Missing "美国" because bd=3/wb=2 emits before "United States" is seen. Higher bd/wb would improve.

## What to do next

### Priority 1: FLORES Benchmark on A40
All infra is deployed. Run benchmarks:
```bash
# On A40 (already deployed to /home/fuxa/nllw_deploy/)
cd /home/fuxa/nllw_deploy
export LLAMA_CPP_LIB=/home/fuxa/llama.cpp/build/bin/libllama.so

# Basic benchmark (need to set up bench.py to work with direct model path)
python3 -m nllw.bench --model /home/fuxa/HY-MT1.5-7B.Q8_0.gguf --lang en-zh -n 20

# With COMET scoring
python3 -m nllw.bench --model /home/fuxa/HY-MT1.5-7B.Q8_0.gguf --lang en-zh --comet --save

# Signal fusion experiment
python3 -m nllw.bench --signal-fusion --shift-k 0.4 --coverage-threshold 0.3 \
  --lang en-zh --comet --save --collect-traces traces_enzh.json
```

**Blocker**: `bench.py` may need a `--model` CLI flag and `--n-gpu-layers` flag to work on the A40. Check and add if missing.

### Priority 2: Parameter Sweep
```bash
python3 -m nllw.bench --sweep "bd=2,3,4 wb=2,3" --lang en-zh --comet --save
```

### Priority 3: Competition Prep (IWSLT 2026, eval April 1-15)
- Docker finalization (llama.cpp already built on A40)
- OmniSTEval format verification
- Per-direction optimal configs
- Extra Context subtrack (terminology injection via prompting, +2-3 BLEU from RASST)

### Priority 4: Research Directions (from iteration 14 research)
- **Translation Heads (ICLR 2026)**: Compare with our TS-scoring. May reveal better heads.
- **RASST terminology injection**: No training, just prompt augmentation. +2-3 BLEU.
- **ExPosST position slots**: Zero KV recomputation. Requires LoRA fine-tuning.
- **SimulU training-free policy**: Training-free cross-attention, 8 MuST-C languages.

## Key Context

- **IWSLT 2026**: Eval April 1-15, ~10 days away
- **Metrics**: LongYAAL (primary latency), COMET (primary quality), XCOMET-XL (internal)
- **CUNI won 2025**: AlignAtt + LA + forced decode. We extend with 8+ additional signals + fusion
- **Hikari is main competitor**: Policy-free WAIT tokens, SOTA on EN-JA/DE/RU
- **Our advantage**: AlignAtt with ANY model via llama.cpp (CUNI couldn't use AlignAtt with EuroLLM)
- **HY-MT1.5-7B is champion**: 0.842 XCOMET-XL EN-ZH
- **A40 ready**: Model + llama.cpp + NLLW deployed at /home/fuxa/nllw_deploy/
- **Dead ends**: EAST, LoRA no-think, GDN, confidence-only, fixed-rate, TAF, Seed-X-PPO, Qwen3-4B, HY-MT1.8B
