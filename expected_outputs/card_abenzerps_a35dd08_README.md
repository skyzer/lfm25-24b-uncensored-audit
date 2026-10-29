---
language:
- en
- ar
- zh
- fr
- de
- ja
- ko
- es
- pt
- it
- vi
- th
- id
- hi
- ru
- pl
license: other
license_name: lfm1.0
license_link: LICENSE
library_name: gguf
tags:
- text-generation
- gsq
- rco
- llama.cpp
- quantized
- reasoning
- liquid
- lfm2.5
- moe
- mixed-precision
- speculative-decoding
- dspark
base_model: LiquidAI/LFM2.5-24B-A2B
base_model_relation: quantized
pipeline_tag: text-generation
---

[![🤗 Model Requests](https://img.shields.io/badge/%F0%9F%A4%97_Model_Requests-Submit_Here-f59e0b?style=for-the-badge&labelColor=1f2937)](https://abenzerps.com/request-a-model/)

> [!NOTE]
> GSQ-RCO vs. BF16 tests are underway and will be added as benchmarks.

# LFM2.5-24B-A2B Uncensored GSQ-RCO GGUF

Mixed-precision **GSQ-RCO** GGUF quantizations of [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B), engineered for high-fidelity reasoning, agentic tool calling, and high-speed `DSpark` speculative decoding in `llama.cpp`.

## Benchmark

![LFM2.5-24B-A2B Benchmark](assets/benchmark.png)

## GGUF Files

| File | bpw | Size |
| :--- | :---: | :---: |
| [LFM2.5-24B-A2B-GSQ-RCO-IQ2_XXS.gguf](LFM2.5-24B-A2B-GSQ-RCO-IQ2_XXS.gguf) | 2.65 | 7.93 GB |
| [LFM2.5-24B-A2B-GSQ-RCO-IQ2_XS.gguf](LFM2.5-24B-A2B-GSQ-RCO-IQ2_XS.gguf) | 2.85 | 8.52 GB |
| [LFM2.5-24B-A2B-GSQ-RCO-IQ2_M.gguf](LFM2.5-24B-A2B-GSQ-RCO-IQ2_M.gguf) | 3.15 | 9.37 GB |
| **[LFM2.5-24B-A2B-GSQ-RCO-IQ3_XS.gguf](LFM2.5-24B-A2B-GSQ-RCO-IQ3_XS.gguf)** *(Recommended)* | **3.50** | **10.54 GB** |
| [LFM2.5-24B-A2B-GSQ-RCO-IQ3_M.gguf](LFM2.5-24B-A2B-GSQ-RCO-IQ3_M.gguf) | 4 | 12.01 GB |
| [LFM2.5-24B-A2B-GSQ-RCO-IQ4_XS.gguf](LFM2.5-24B-A2B-GSQ-RCO-IQ4_XS.gguf) | 4.60 | 13.84 GB |
| [LFM2.5-24B-A2B-GSQ-RCO-Q5_K_M.gguf](LFM2.5-24B-A2B-GSQ-RCO-Q5_K_M.gguf) | 5.75 | 17.20 GB |

### Speculative Decoding Drafters (`DSpark`)

Pair any GSQ-RCO model above with the **DSpark** speculative draft model for faster generation:

| Drafter File | Precision | Size |
| :--- | :---: | :---: |
| **[LFM2.5-24B-A2B-DSpark-F16.gguf](LFM2.5-24B-A2B-DSpark-F16.gguf)** *(Recommended)* | **F16** | **664 MB** |
| [LFM2.5-24B-A2B-DSpark-Q8_0.gguf](LFM2.5-24B-A2B-DSpark-Q8_0.gguf) | Q8_0 | 356 MB |

## Usage

> [!TIP]
> **Recommended Sampling Parameters:** `LFM2.5-24B-A2B` is a hybrid ShortConv + GQA + 64-Expert MoE reasoning model that always thinks (`<think>...</think>`) before answering. For optimal reasoning and tool-calling accuracy, use `temperature = 0.1`, `top_k = 50`, and `repeat_penalty = 1.05`.

### 1. Speculative Decoding with `DSpark-F16` (Recommended — Fastest)

Pairing the main model with `LFM2.5-24B-A2B-DSpark-F16.gguf` (`--spec-type draft-dspark --spec-draft-n-max 3`) delivers **60%–80%+ draft acceptance** and significant single-stream speedups:

**Interactive CLI (`DSpark-F16`)**
```bash
llama-cli \
  -m LFM2.5-24B-A2B-GSQ-RCO-IQ3_XS.gguf \
  -md LFM2.5-24B-A2B-DSpark-F16.gguf \
  --spec-type draft-dspark \
  --spec-draft-n-max 3 \
  -ngl 99 -ngld 99 \
  -fa on \
  -b 2048 \
  -c 32768 \
  --temp 0.1 --top-k 50 --repeat-penalty 1.05 \
  -cnv
```

**Local OpenAI-Compatible Server for Agents (`DSpark-F16`)**
```bash
llama-server \
  -m LFM2.5-24B-A2B-GSQ-RCO-IQ3_XS.gguf \
  -md LFM2.5-24B-A2B-DSpark-F16.gguf \
  --spec-type draft-dspark \
  --spec-draft-n-max 3 \
  -ngl 99 -ngld 99 \
  -fa on \
  -b 2048 \
  -c 32768 \
  --temp 0.1 --top-k 50 --repeat-penalty 1.05 \
  --jinja \
  --port 8080
```

### 2. Standard Inference (`llama.cpp`)

**Interactive Chat**
```bash
llama-cli \
  -m LFM2.5-24B-A2B-GSQ-RCO-IQ3_XS.gguf \
  -ngl 99 \
  -fa on \
  -b 2048 \
  -c 32768 \
  --temp 0.1 --top-k 50 --repeat-penalty 1.05 \
  -cnv
```

**128K Full-Context Deployment (Low-VRAM Advantage)**
Because 30 of the 40 layers in `LFM2.5-24B-A2B` use constant-state `ShortConv` blocks and only 10 layers use GQA Attention (with KV heads shielded in `F16`), its KV cache footprint is **4× smaller** than standard 40-layer transformers—allowing full **131,072 (128K) context** on consumer 12 GB–16 GB GPUs:
```bash
llama-server \
  -m LFM2.5-24B-A2B-GSQ-RCO-IQ3_XS.gguf \
  -ngl 99 \
  -fa on \
  -b 2048 \
  -c 131072 \
  -ctk q8_0 -ctv q8_0 \
  --temp 0.1 --top-k 50 --repeat-penalty 1.05 \
  --jinja \
  --port 8080
```

### 3. Ollama

```bash
ollama run hf.co/abenzerps/LFM2.5-24B-A2B-Uncensored-GSQ-RCO-GGUF:IQ3_XS
```

## Attribution & License

- Distributed under the [LFM Open License v1.0](LICENSE), matching upstream [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B).
- Base model: [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B).
- Speculative drafter: [LiquidAI/LFM2.5-24B-A2B-DSpark](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-DSpark).
