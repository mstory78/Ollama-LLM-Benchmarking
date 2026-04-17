# 🚀 LLM Bench Pipeline

**🇺🇸 English** | [🇰🇷 한국어](README_KO.md)

> One-command pipeline: auto-detect hardware → recommend LLM models → benchmark with Ollama → generate HTML report

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Required-green?logo=ollama)
![Platform](https://img.shields.io/badge/Platform-Windows%20|%20Linux%20|%20macOS-lightgrey)
![License](https://img.shields.io/badge/License-MIT-yellow)

A pure Python script that auto-detects your hardware, recommends the best LLM models for your specs, benchmarks them through Ollama, and generates a visual report. No additional packages required — runs on Python standard library only.

---

## ✨ Features

| Step | What it does | Details |
|:----:|-------------|---------|
| 1️⃣ | **Hardware Detection** | CPU, RAM, GPU (NVIDIA/AMD/Intel/Apple Silicon), AMD iGPU (ROCm/Vulkan/HSA) |
| 2️⃣ | **Model Recommendation** | 31 built-in models + web catalog update, score-based ranking with latest-gen bonus |
| 3️⃣ | **Benchmark** | Sequential model testing via Ollama, 6 auto-scored tasks |
| 4️⃣ | **Triple Ranking** | By score / by speed / composite (adjustable weights) |
| 5️⃣ | **Report** | Terminal + HTML (Chart.js) + JSON |

---

## 📦 Requirements

- **Python 3.8+** (no pip install needed)
- **[Ollama](https://ollama.com/download)** — install and run `ollama serve`

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/mstory78/llm-bench-pipeline.git
cd llm-bench-pipeline

# 2. Start Ollama server (separate terminal)
ollama serve

# 3. Run the full pipeline
python llm_bench.py --output report.html
```

On Windows, use `py` instead of `python` if needed.

---

## 📋 Command Reference

### Basics

```bash
python llm_bench.py --output report.html                    # Full pipeline + HTML report
python llm_bench.py --output r.html --json-output r.json    # HTML + JSON
```

### Step-by-Step

```bash
python llm_bench.py --detect-only                           # Hardware detection only
python llm_bench.py --recommend-only                        # Detection + recommendation
python llm_bench.py --recommend-only --top-n 15             # Show top 15 recommendations
```

### Model & Task Selection

```bash
python llm_bench.py --models "qwen3:8b,deepseek-r1:7b,gemma3:4b"
python llm_bench.py --skip-pull                             # Test already-installed models only
python llm_bench.py --tasks reasoning,coding,korean         # Specific tasks only
```

> Available tasks: `reasoning`, `coding`, `korean`, `summarization`, `instruction_following`, `math`

### Ranking Weights

```bash
python llm_bench.py --weight-score 0.9 --weight-speed 0.1  # Prioritize accuracy
python llm_bench.py --weight-score 0.5 --weight-speed 0.5  # Balanced
```

### Catalog Management

```bash
python llm_bench.py --update-catalog --recommend-only       # Fetch latest models from ollama.com
python llm_bench.py --catalog my_catalog.json               # Use external catalog
```

### Advanced

```bash
python llm_bench.py --ollama-url http://192.168.1.100:11434 # Remote Ollama server
python llm_bench.py --timeout 300                           # 5-minute timeout
```

### Full Workflow

```bash
python llm_bench.py --detect-only
python llm_bench.py --update-catalog --recommend-only --top-n 10
python llm_bench.py --models "qwen3:8b,deepseek-r1:7b" \
    --tasks reasoning,coding,korean \
    --weight-score 0.9 --weight-speed 0.1 \
    --output report.html --json-output report.json
```

---

## 📊 Sample Output

```
🚀 LLM Hardware Detection & Benchmark Pipeline
==================================================

[1/4] 🔍 Detecting hardware...
============================================================
  🖥️  Hardware Profile
============================================================
  OS        : macOS 15.2
  CPU       : Apple M3 Pro (12C/12T)
  RAM       : 36.0 GB (available 28.4 GB)
  GPU 0     : Apple M3 Pro
    VRAM    : 27648 MB (Unified Memory)
    Backend : METAL
  Tier      : HIGH
  Usable    : ~22.7 GB
============================================================

  🎯 Recommended Models
  1. 🇰🇷 Qwen 3 14B    → reasoning, korean, coding, general
  2.    DeepSeek R1 14B → reasoning, math, coding
  3. 🇰🇷 Qwen 3 8B     → reasoning, korean, coding, general
  ...

📋 Ranking Summary
──────────────────────────────────────────────
  Model                Score    Speed    Overall
  qwen3:14b           #1       #3       #1
  deepseek-r1:14b     #2       #2       #2
  qwen3:8b            #3       #1       #3
```

---

## 🎯 Recommendation Logic

Not just "more RAM = bigger model." Multiple factors are scored and combined:

| Factor | Bonus | Description |
|--------|-------|-------------|
| Base priority | 30–95 | Per-model verified performance level |
| GPU VRAM fit | +15 | Model fits entirely in VRAM |
| iGPU offload | +5 | Partial GPU offload possible |
| Korean support | +10 | Qwen, Gemma series, etc. |
| RAM headroom | +5 | 4GB+ above minimum requirement |
| Latest generation | +8 | Qwen 3, Gemma 3, DeepSeek R1, Phi-4 |
| Current generation | +4 | Qwen 2.5, Llama 3.2, Gemma 2 |
| iGPU large model penalty | -10 | 14B+ models on shared memory |

---

## 🏗️ Built-in Model Catalog (31 models)

| Tier | Size | Models | Min RAM |
|------|------|--------|---------|
| **Tiny** | 0.5–3B | Qwen 3 0.6B/1.7B, Llama 3.2 1B/3B, Gemma 3 1B, Phi-4 Mini | 1–3 GB |
| **Small** | 4–9B | Qwen 3 4B/8B, DeepSeek R1 7B, Gemma 3 4B/12B, Llama 3.1 8B | 3.5–8 GB |
| **Medium** | 14–30B | Qwen 3 14B/30B-A3B, Phi-4 14B, DeepSeek R1 14B, Gemma 3 27B | 10–18 GB |
| **Large** | 32B+ | Qwen 3 32B, DeepSeek R1 32B/70B, Llama 3.3 70B | 22–42 GB |

Use `--update-catalog` to fetch the latest models from ollama.com automatically.

---

## 🍎 Apple Silicon Guide

For Mac M1/M2/M3/M4 users:

**Unified Memory advantage** — Apple Silicon shares memory between CPU and GPU, so your RAM capacity directly determines the maximum model size.

| Mac Model | RAM | Recommended Max Model |
|-----------|-----|-----------------------|
| MacBook Air (M1/M2) | 8–16 GB | 7–8B |
| MacBook Pro (M3 Pro) | 18–36 GB | 14–32B |
| Mac Studio (M2 Ultra) | 64–192 GB | 70B+ |

**Metal auto-acceleration** — Ollama automatically detects and uses the Metal backend. No configuration needed.

```bash
# Just run it
ollama serve
```

**Memory limit tuning** (advanced):
```bash
sudo sysctl iogpu.wired_limit_mb=12288  # Allocate 12GB to GPU
```

---

## 🔧 AMD iGPU Optimization Guide

For AMD APU users (Ryzen 5000G/6000/7000/8000):

**1. BIOS** — Set UMA Frame Buffer Size → 4GB (maximum)

**2. GPU Offload**
```bash
# Windows
set OLLAMA_NUM_GPU=999
set HSA_OVERRIDE_GFX_VERSION=11.0.0   # Phoenix (7000/8000)
ollama serve

# Linux
HSA_OVERRIDE_GFX_VERSION=11.0.0 OLLAMA_NUM_GPU=999 ollama serve
```

| APU Generation | GFX Version |
|----------------|-------------|
| Renoir/Cezanne (5000G) | 9.0.0 |
| Rembrandt (6000) | 10.3.0 |
| Phoenix (7000/8000) | 11.0.0 |

**3. Dual-channel RAM is essential** — nearly 2x token generation speed vs single-channel.

---

## 📁 Project Structure

```
llm-bench-pipeline/
├── llm_bench.py          # Main pipeline (single file)
├── README.md             # English
├── README_KO.md          # 한국어
├── LICENSE
├── .gitignore
└── docs/
    ├── blog_post.md      # Blog post (Korean)
    └── cheatsheet.md     # Command cheatsheet
```

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🤝 Contributing

Issues and PRs are welcome! Especially looking for:

- New model additions (model_catalog.json)
- Additional evaluation tasks
- Benchmark results from different hardware environments
