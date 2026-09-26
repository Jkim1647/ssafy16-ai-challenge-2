# 미시험 레버 스크리닝 결과

`tools/score_screen.py`. GPU 미사용.

> train2000 은 train 출처라 train 을 쓰는 기법(few-shot)을 과대평가한다. dev408 에서 악화가 없어야 채택 후보로 본다 — 오늘 27B LoRA 가 val667 1위인데 Public 꼴찌였던 것과 같은 함정이다.

## train1000 (n=1000)

| 설정 | 정답 | 정확도 | 95% CI |
|---|---|---|---|
| base_predictions.jsonl | 962 | 0.9620 | 0.9483–0.9722 |

- 1문항 = 0.100%p

### base_predictions.jsonl 대비 (paired bootstrap 10,000회)

| 비교 | 평균차 | 95% CI | 예측 변경 | 판정 |
|---|---|---|---|---|

## dev229 (n=229)

| 설정 | 정답 | 정확도 | 95% CI |
|---|---|---|---|
| base_predictions.jsonl | 204 | 0.8908 | 0.8438–0.9250 |

- 1문항 = 0.437%p

### base_predictions.jsonl 대비 (paired bootstrap 10,000회)

| 비교 | 평균차 | 95% CI | 예측 변경 | 판정 |
|---|---|---|---|---|

