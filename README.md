# dLLM Quantization x Fast-dLLM 실험

## 개요
- QDLM (DuQuant W4A4) + Fast-dLLM (KV cache) 통합 실험
- 모델: LLaDA-8B-Instruct

## 실험 구조
- QDLM으로 duquant_parameters.pth 생성
- Fast-dLLM에서 base model + fake-quant params 로드
- KV cache / parallel decoding으로 추론

## GSM8K 결과 (100 samples, 5-shot)
| Model            | flexible | strict |
|------------------|----------|--------|
| FP16 no cache    | 0.72     | 0.40   |
| FP16 cache ON    | 0.74     | 0.40   |
| W4A4 no cache    | 0.73     | 0.43   |
| W4A4 cache ON    | 0.74     | 0.38   |

## Pilot 분석 결과
- 가설 3 확인: masked vs unmasked token activation 통계 유의미한 차이
- 가설 4 부분 확인: Layer 16에서 high mask -> masked token MSE +66.5%
- Layer 24 역전 발견: low mask -> unmasked token MSE -90.2%
- 결론: masked/unmasked 섞어서 calibration하면 레이어별로 오차 방향이 다름

## 파일 구조
```
code/
  apply_duquant.py   # DuQuant params -> Fast-dLLM 어댑터
  chat.py            # Fast-dLLM chat 패치본
  eval_llada.py      # Fast-dLLM eval 패치본

results/
  pilot/             # activation 분석 그래프
  100samples/        # GSM8K 100샘플 eval JSON
```

## 다음 단계
- 본실험: 32개 레이어 전체, 100샘플
- mask-aware calibration 구현 및 비교
