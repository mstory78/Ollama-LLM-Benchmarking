# 🚀 LLM Bench Pipeline

[🇺🇸 English](README.md) | **🇰🇷 한국어**

> 하드웨어 감지 → 모델 추천 → Ollama 벤치마크 → 리포트 생성, **원커맨드 파이프라인**

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Required-green?logo=ollama)
![Platform](https://img.shields.io/badge/Platform-Windows%20|%20Linux%20|%20macOS-lightgrey)
![License](https://img.shields.io/badge/License-MIT-yellow)

내 PC 사양을 자동으로 읽어서 돌릴 수 있는 LLM 모델을 추천해주고, 실제 성능까지 벤치마크해주는 **순수 파이썬 스크립트**입니다.  
추가 패키지 설치 없이 Python 표준 라이브러리만으로 동작합니다.

---

## ✨ 주요 기능

| 단계 | 기능 | 세부 내용 |
|:----:|------|-----------|
| 1️⃣ | **하드웨어 감지** | CPU, RAM, GPU (NVIDIA/AMD/Intel/Apple Silicon), AMD iGPU (ROCm/Vulkan/HSA) |
| 2️⃣ | **모델 추천** | 31개 내장 카탈로그 + 웹 갱신, 점수 기반 추천 (최신 세대 가산점) |
| 3️⃣ | **벤치마크** | Ollama로 모델 순차 실행, 6가지 태스크 자동 채점 |
| 4️⃣ | **3중 순위** | 점수순 / 속도순 / 종합 (가중치 조절 가능) |
| 5️⃣ | **리포트** | 터미널 + HTML (Chart.js) + JSON |

---

## 📦 준비물

- **Python 3.8+** (추가 패키지 불필요)
- **[Ollama](https://ollama.com/download)** 설치 후 `ollama serve`

---

## 🚀 빠른 시작

```bash
# 1. 클론
git clone https://github.com/YOUR_USERNAME/llm-bench-pipeline.git
cd llm-bench-pipeline

# 2. Ollama 서버 시작 (별도 터미널)
ollama serve

# 3. 전체 파이프라인 실행
python llm_bench.py --output report.html
```

Windows에서 `python` 명령이 안 되면 `py`를 사용하세요.

---

## 📋 명령어 레퍼런스

### 기본

```bash
python llm_bench.py --output report.html                    # 전체 파이프라인 + HTML 리포트
python llm_bench.py --output r.html --json-output r.json    # HTML + JSON 동시 출력
```

### 단계별

```bash
python llm_bench.py --detect-only                           # 하드웨어 감지만
python llm_bench.py --recommend-only                        # 감지 + 추천까지
python llm_bench.py --recommend-only --top-n 15             # 추천 15개로 확대
```

### 모델 & 태스크 선택

```bash
python llm_bench.py --models "qwen3:8b,deepseek-r1:7b,gemma3:4b"
python llm_bench.py --skip-pull                             # 이미 설치된 모델만
python llm_bench.py --tasks reasoning,coding,korean         # 특정 태스크만
```

> 사용 가능한 태스크: `reasoning`, `coding`, `korean`, `summarization`, `instruction_following`, `math`

### 순위 가중치

```bash
python llm_bench.py --weight-score 0.9 --weight-speed 0.1  # 점수 중시
python llm_bench.py --weight-score 0.5 --weight-speed 0.5  # 균등
```

### 카탈로그 관리

```bash
python llm_bench.py --update-catalog --recommend-only       # 웹에서 최신 모델 갱신
python llm_bench.py --catalog my_catalog.json               # 외부 카탈로그 사용
```

### 고급

```bash
python llm_bench.py --ollama-url http://192.168.1.100:11434 # 원격 Ollama 서버
python llm_bench.py --timeout 300                           # 타임아웃 5분
```

### 실전 풀코스

```bash
python llm_bench.py --detect-only
python llm_bench.py --update-catalog --recommend-only --top-n 10
python llm_bench.py --models "qwen3:8b,deepseek-r1:7b" \
    --tasks reasoning,coding,korean \
    --weight-score 0.9 --weight-speed 0.1 \
    --output report.html --json-output report.json
```

---

## 📊 출력 예시

```
🚀 LLM Hardware Detection & Benchmark Pipeline
==================================================

[1/4] 🔍 하드웨어 감지 중...
============================================================
  🖥️  하드웨어 프로필
============================================================
  CPU       : Intel Core i9-14900K (24C/32T)
  RAM       : 64.0 GB (가용 52.3 GB)
  GPU 0     : NVIDIA GeForce RTX 4090
    VRAM    : 24564 MB
    Backend : CUDA
  성능 등급   : HIGH
============================================================

[2/4] 🎯 모델 추천 중...
  1. 🇰🇷 Qwen 3 8B   → reasoning, korean, coding, general
  2.    DeepSeek R1 7B → reasoning, math, coding
  3. 🇰🇷 Qwen 2.5 7B  → reasoning, korean, coding, general
  ...

📋 순위 요약
──────────────────────────────────────────────
  모델                 점수순   속도순   종합
  qwen3:8b            #1      #3      #1
  deepseek-r1:7b      #2      #2      #2
  gemma3:4b           #3      #1      #3

🏆 종합 1위: qwen3:8b (종합 91.2)
🎯 점수 1위: qwen3:8b (점수 92.5/100)
⚡ 속도 1위: gemma3:4b (속도 45.3 tok/s)
```

---

## 🎯 추천 로직

단순 "RAM 크면 큰 모델"이 아닌, **다중 요소 점수 합산 방식**입니다.

| 요소 | 보너스 | 설명 |
|------|--------|------|
| 기본 우선순위 | 30~95 | 모델별 검증 수준 기반 |
| GPU VRAM 적합 | +15 | VRAM에 완전히 들어가는 경우 |
| iGPU 오프로드 | +5 | 부분 오프로드 가능 시 |
| 한국어 지원 | +10 | Qwen, Gemma 시리즈 등 |
| RAM 여유 | +5 | 4GB 이상 여유 시 |
| 최신 세대 | +8 | Qwen 3, Gemma 3, DeepSeek R1, Phi-4 |
| 현세대 | +4 | Qwen 2.5, Llama 3.2, Gemma 2 |
| iGPU 대형 모델 | -10 | 14B+ 모델의 메모리 공유 이슈 |

---

## 🏗️ 내장 모델 카탈로그 (31개)

| 등급 | 크기 | 모델 | 최소 RAM |
|------|------|------|----------|
| **Tiny** | 0.5~3B | Qwen 3 0.6B/1.7B, Llama 3.2 1B/3B, Gemma 3 1B, Phi-4 Mini | 1~3 GB |
| **Small** | 4~9B | Qwen 3 4B/8B, DeepSeek R1 7B, Gemma 3 4B/12B, Llama 3.1 8B | 3.5~8 GB |
| **Medium** | 14~30B | Qwen 3 14B/30B-A3B, Phi-4 14B, DeepSeek R1 14B, Gemma 3 27B | 10~18 GB |
| **Large** | 32B+ | Qwen 3 32B, DeepSeek R1 32B/70B, Llama 3.3 70B | 22~42 GB |

`--update-catalog` 옵션으로 ollama.com에서 최신 모델을 자동 갱신할 수 있습니다.

---

## 🍎 Apple Silicon 최적화 가이드

Mac M1/M2/M3/M4 시리즈 환경에서의 팁:

**Unified Memory 장점** — Apple Silicon은 CPU와 GPU가 같은 메모리를 공유하므로, RAM 용량이 곧 모델 크기의 상한선입니다.

| Mac 모델 | RAM | 추천 최대 모델 |
|----------|-----|---------------|
| MacBook Air (M1/M2) | 8~16 GB | 7~8B |
| MacBook Pro (M3 Pro) | 18~36 GB | 14~32B |
| Mac Studio (M2 Ultra) | 64~192 GB | 70B+ |

**Metal 자동 가속** — Ollama가 Metal 백엔드를 자동 감지합니다. 별도 설정이 필요 없습니다.

```bash
# 그냥 실행하면 됩니다
ollama serve
```

**메모리 한도 조절** (고급) — 메모리 압박 시 GPU에 할당할 wired 메모리를 조절할 수 있습니다.
```bash
sudo sysctl iogpu.wired_limit_mb=12288  # 12GB 할당
```

---

## 🔧 AMD iGPU 최적화 가이드

AMD APU (Ryzen 5000G/6000/7000/8000) 환경에서의 팁:

**1. BIOS** — UMA Frame Buffer Size → 4GB (최대)

**2. GPU 오프로드**
```bash
# Windows
set OLLAMA_NUM_GPU=999
set HSA_OVERRIDE_GFX_VERSION=11.0.0   # Phoenix(7000/8000)
ollama serve

# Linux
HSA_OVERRIDE_GFX_VERSION=11.0.0 OLLAMA_NUM_GPU=999 ollama serve
```

| APU 세대 | GFX 버전 |
|----------|----------|
| Renoir/Cezanne (5000G) | 9.0.0 |
| Rembrandt (6000) | 10.3.0 |
| Phoenix (7000/8000) | 11.0.0 |

**3. 듀얼 채널 RAM 필수** — 싱글 채널 대비 토큰 생성 속도 ~2배 차이

---

## 📁 프로젝트 구조

```
llm-bench-pipeline/
├── llm_bench.py          # 메인 파이프라인 (단일 파일)
├── README.md
├── LICENSE
├── .gitignore
└── docs/
    ├── blog_post.md      # 블로그 포스트 (한국어)
    └── cheatsheet.md     # 명령어 치트시트
```

---

## 📄 License

MIT License — 자유롭게 사용, 수정, 배포하실 수 있습니다.

---

## 🤝 Contributing

이슈, PR 환영합니다. 특히 아래 기여를 기다리고 있습니다:

- 새로운 모델 추가 (model_catalog.json)
- 평가 태스크 추가/개선
- 다양한 하드웨어 환경에서의 벤치마크 결과 공유
