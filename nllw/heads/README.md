# Alignment Head Configurations

Pre-computed translation alignment head configurations for use with the AlignAtt simultaneous translation backend.

Each JSON file contains ranked attention heads that exhibit strong source-target token alignment behavior, identified via translation score (ts) analysis.

## Available Configurations

### HY-MT1.5-7B

| File | Language Pair | Top Head | ts | Total Heads |
|------|--------------|----------|------|-------------|
| `hymt15_7b_universal.json` | Universal (EN-ZH/DE/IT/FR, CS-EN) | L7/H21 | 0.737 | 10 |
| `translation_heads_hymt_en_zh.json` | en-zh | L7/H21 | 0.731 | 641 |

### HY-MT1.5-1.8B

| File | Language Pair | Top Head | ts | Total Heads |
|------|--------------|----------|------|-------------|
| `translation_heads_hymt1.8b_en_de.json` | en-de | L8/H6 | 0.639 | 377 |
| `translation_heads_hymt1.8b_en_zh.json` | en-zh | L9/H5 | 0.688 | 368 |

### EuroLLM-9B-Instruct

| File | Language Pair | Top Head | ts | Total Heads |
|------|--------------|----------|------|-------------|
| `translation_heads_cs_en_eurollm.json` | cs-en | L13/H7 | 0.700 | 38 |
| `translation_heads_en_de_eurollm.json` | en-de | L13/H7 | 0.637 | 30 |
| `translation_heads_en_it_eurollm.json` | en-it | L13/H7 | 0.706 | 30 |
| `translation_heads_en_zh_eurollm.json` | en-zh | L13/H7 | 0.704 | 28 |

### Qwen3-4B

| File | Language Pair | Top Head | ts | Total Heads |
|------|--------------|----------|------|-------------|
| `translation_heads_qwen3_4b_cs_en.json` | cs-en | L24/H22 | 0.477 | 81 |
| `translation_heads_qwen3_4b_en_de.json` | en-de | L24/H5 | 0.363 | 63 |
| `translation_heads_qwen3_4b_en_it.json` | en-it | L24/H5 | 0.425 | 72 |
| `translation_heads_qwen3_4b_en_zh.json` | en-zh | L24/H22 | 0.155 | 10 |

### Qwen3-8B

| File | Language Pair | Top Head | ts | Total Heads |
|------|--------------|----------|------|-------------|
| `translation_heads_qwen3_8b_en_de.json` | en-de | L11/H10 | 0.613 | 730 |
| `translation_heads_qwen3_8b_en_zh.json` | en-zh | L24/H22 | 0.134 | 8 |

### Qwen3-14B

| File | Language Pair | Top Head | ts | Total Heads |
|------|--------------|----------|------|-------------|
| `translation_heads_qwen3_14b_en_zh.json` | en-zh | L19/H22 | 0.130 | 4 |

### Qwen3.5-4B

| File | Language Pair | Top Head | ts | Total Heads |
|------|--------------|----------|------|-------------|
| `translation_heads_qwen3.5_4b_en_de.json` | en-de | L7/H5 | 0.659 | 128 |
| `translation_heads_qwen3.5_4b_en_it.json` | en-it | L7/H5 | 0.059 | 107 |
| `translation_heads_qwen3.5_4b_en_zh.json` | en-zh | L7/H5 | 0.722 | 121 |

### Qwen3.5-9B

| File | Language Pair | Top Head | ts | Total Heads |
|------|--------------|----------|------|-------------|
| `translation_heads_qwen3.5_9b_en_de.json` | en-de | L7/H5 | 0.665 | 125 |
| `translation_heads_qwen3.5_9b_en_zh.json` | en-zh | L7/H5 | 0.721 | 119 |

### TowerInstruct-7B

| File | Language Pair | Top Head | ts | Total Heads |
|------|--------------|----------|------|-------------|
| `translation_heads_en_zh_tower.json` | en-zh | L8/H25 | 0.542 | 21 |

### TranslateGemma-4B

| File | Language Pair | Top Head | ts | Total Heads |
|------|--------------|----------|------|-------------|
| `translation_heads_en_zh_translategemma.json` | en-zh | L21/H0 | 0.388 | 41 |

### Seed-X-PPO-7B

| File | Language Pair | Top Head | ts | Total Heads |
|------|--------------|----------|------|-------------|
| `translation_heads_seedx_7b_en_zh.json` | en-zh | L7/H18 | 0.705 | 1024 |
