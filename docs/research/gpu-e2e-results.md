# GPU E2E Test Results -- Iteration 14

## Date: 2026-03-21

## Setup
- **Machine**: A40 (46GB VRAM, CUDA 12.4)
- **Model**: HY-MT1.5-7B.Q8_0.gguf (~8GB VRAM)
- **llama.cpp**: Custom build with attention extraction API (PR #20086)
- **Backend**: NLLW AlignAtt, bd=3, wb=2, top_k_heads=8

## Test 1: Low-Level E2E (llama_backend.py)

Direct llama.cpp bindings test: model load, tokenize, decode, generate, attention extraction.

- Model loading: 2.3s
- Prompt decode (29 tokens): 0.213s (136 tok/s)
- Generation: 37-39 tok/s
- Attention extraction: Working (8 heads, shape (8, n_kv))

**Translation**: "The president of the United States announced new policies today"
-> "美国总统今天宣布了新的政策。" (correct, high quality)

### Critical Bug Found & Fixed

**Attention stride bug**: `get_attn_weights()` used `ctx_size` (current position) as stride
between heads, but the actual layout is `n_heads * n_ctx` (full context window). This caused
all heads except the first to read garbage data. Fixed: stride is now always `n_ctx(ctx)`.

**n_gpu_layers missing**: `BackendConfig` had no `n_gpu_layers` field, so `load_model()`
defaulted to CPU-only. Added field and wired into both backends.

**Backend auto-import**: `create_backend()` failed because backend modules weren't imported
(no `@register_backend` decorators fired). Added `_ensure_backends_imported()` with lazy imports.

## Test 2: Full AlignAtt Backend (alignatt_backend.py)

Word-by-word simultaneous translation with border detection.

**Source**: "The president of the United States announced new economic policies today" (11 words)

```
[1/11] +'The'       -> (waiting)
[2/11] +'president' -> '总统'           (850ms)
[3/11] +'of'        -> (waiting)
[4/11] +'the'       -> (waiting)
[5/11] +'United'    -> (waiting)
[6/11] +'States'    -> (waiting)
[7/11] +'announced' -> (waiting)
[8/11] +'new'       -> (waiting)
[9/11] +'economic'  -> (waiting)
[10/11] +'policies' -> '宣布了'         (126ms)
[11/11] +'today'    -> '新的经济政策。'  (147ms)
```

**Output**: 总统宣布了新的经济政策。
**Time**: 1528ms total (7.2 words/s)

### Observations

1. Border detection works correctly -- waits for enough context before emitting
2. First emission at word 2 ("president") is immediate -- bd=3, wb=2 means wait 2 words
3. Missing "美国" (United States) from output -- the model emits "总统" too early
   before seeing "United States". This is expected with bd=3, wb=2.
4. With bd=4 or wb=3, quality would improve but latency would increase
5. Multi-sentence test works: reset() between sentences is functional

### Additional Sentences

```
'Hello world'                                    -> '你好，世界'
'The weather is beautiful today in Prague'        -> '天气状况非常好，今天布拉格的天气十分宜人。'
'Machine translation has improved dramatically'  -> '机器翻译技术已经取得了显著的进步。'
```

## Key Metrics

| Metric | Value |
|--------|-------|
| Model load time | 2.3s |
| Prompt throughput | 136 tok/s |
| Generation throughput | 37-39 tok/s |
| VRAM usage (est.) | ~8.5 GB |
| Latency per word (avg) | ~140ms |
| Translation quality | Good (needs per-direction tuning) |

## Test 3: Mini Benchmark (5 sentences, EN-ZH, bd=3, wb=3)

| # | English | Chinese (SimulMT) | Time |
|---|---------|-------------------|------|
| 1 | On Monday, scientists from the Stanford University School of Medicine announced the invention of a new diagnostic tool. | 周一，来自斯坦福大学医学院的科学家们宣布，他们发明了一种新的诊断工具。 | 2.0s |
| 2 | He is now planning to run for president in the next election. | 他现在是这样。他打算在下次选举中参加总统竞选。 | 1.4s |
| 3 | The weather forecast predicts rain for the entire week. | 天气预报显示，这一整周都会下雨。 | 1.2s |
| 4 | International trade has been growing steadily over the past decade. | 国际贸易已经……在过去的十年里稳步发展。 | 1.8s |
| 5 | The European Union announced new climate change regulations yesterday. | 欧洲联盟昨天宣布了新的气候变化法规。 | 1.3s |

**Total**: 58 words in 7.7s = 7.5 words/s

### Quality Notes
- Sentences 1, 3, 5: Excellent quality
- Sentence 2: Spurious prefix "他现在是这样" from early border emission
- Sentence 4: "……" artifact from border interruption mid-phrase
- These artifacts are expected with SimulMT and can be reduced with higher bd/wb
