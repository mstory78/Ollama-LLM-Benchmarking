# LLM Bench 명령어 치트시트

## 기본
py llm_bench.py --output report.html              # 전체 파이프라인 + HTML 리포트
py llm_bench.py --output r.html --json-output r.json  # HTML + JSON 동시 출력

## 단계별
py llm_bench.py --detect-only                      # 1단계: 하드웨어 감지만
py llm_bench.py --recommend-only                   # 2단계: 감지 + 추천까지
py llm_bench.py --recommend-only --top-n 15        # 추천 15개로 확대

## 모델 선택
py llm_bench.py --models "qwen3:8b,deepseek-r1:7b,gemma3:4b"
py llm_bench.py --skip-pull                        # 이미 설치된 모델만

## 태스크 선택
py llm_bench.py --tasks reasoning,coding,korean
# 전체: reasoning, coding, korean, summarization, instruction_following, math

## 순위 가중치
py llm_bench.py --weight-score 0.9 --weight-speed 0.1   # 점수 중시
py llm_bench.py --weight-score 0.5 --weight-speed 0.5   # 균등

## 카탈로그
py llm_bench.py --update-catalog --recommend-only  # 웹에서 최신 모델 갱신
py llm_bench.py --catalog my_catalog.json          # 외부 카탈로그 사용

## 고급
py llm_bench.py --ollama-url http://192.168.1.100:11434  # 원격 서버
py llm_bench.py --timeout 300                      # 타임아웃 5분

## 실전 풀코스
py llm_bench.py --detect-only
py llm_bench.py --update-catalog --recommend-only --top-n 10
py llm_bench.py --models "qwen3:8b,deepseek-r1:7b,gemma3:4b" --tasks reasoning,coding,korean --weight-score 0.9 --weight-speed 0.1 --output report.html --json-output report.json
