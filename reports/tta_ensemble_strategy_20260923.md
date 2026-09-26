# TTA 완료 후 앙상블 전략 요약 (2026-09-23)

## 결론

- 선택지 순환 TTA의 4개 view를 **독립 모델 4개로 취급하지 않는다.**
- 먼저 35B 내부에서 TTA를 결합해 **하나의 `35B_TTA` 분포**를 만들고, 기존 앙상블의 35B 자리만 교체한다.
- 첫 비교는 `mean4`(정규화 로그확률 평균)와 `prob_mean4`(확률 산술평균) 두 개로 제한한다.
- 그 다음에만 calibration, ETTC-style 문항별 선택/가중, PriDe, ensemble selection을 순차적으로 본다.
- 이미 반복 사용한 검증셋에 맞춘 대규모 weight search는 피하고, 현재 팀의 깨끗한 평가 gate와 NC/NW/NetGain을 우선한다.

## 1. 기본 결합 구조

```text
35B view0 / shift1 / shift2 / shift3
                ↓
        내부 TTA 결합
                ↓
            35B_TTA
                ↓
기존 모델 앙상블의 35B 자리만 교체
```

4개 view를 그대로 모델 4개처럼 넣으면 35B 계열의 총 가중치가 과도하게 커져
'TTA 효과'와 '35B 가중치 증가 효과'를 분리하기 어렵다.

## 2. 가장 먼저 비교할 두 결합법

### A. mean4 — 현재 코드 방식
각 view의 a~d 점수를 원래 보기 순서로 복원하고 log-softmax 정규화한 뒤 평균한다.

```text
mean4 = mean(log p_view0, log p_shift1, log p_shift2, log p_shift3)
```

정규화 확률의 기하평균에 해당한다.

### B. prob_mean4 — 추가 비교 필요
각 view의 정규화 확률을 산술평균한다.

```text
prob_mean4 = mean(p_view0, p_shift1, p_shift2, p_shift3)
```

기존 `mean4`와 같은 것이 아니므로 별도 구현·명명이 필요하다.

### 첫 실험
1. 기존 최종 앙상블
2. 35B → `35B_TTA(mean4)` 교체
3. 35B → `35B_TTA(prob_mean4)` 교체

다른 모델과 모델-level 가중치는 우선 고정한다.

## 3. 다음 후보

### Calibration
모델마다 confidence scale이 다를 수 있으므로 temperature scaling 등으로 분포를 보정한 뒤 결합한다.
최종 분포에 공통 temperature만 적용해도 argmax는 바뀌지 않으므로, calibration은 모델별 신뢰도 정렬이나
게이트/가중치 계산에 쓰는 것이 목적이다.

### ETTC-style uncertainty selection
문항별로 각 모델의 선택지 분포 entropy를 계산해 더 확실한 모델을 선택하거나 가중치를 조절한다.

```text
R3 / 35B_TTA / 397B / 기타 후보
             ↓
       문항별 entropy
             ↓
   hard select 또는 soft weight
```

원 논문을 그대로 재현한다고 보지 말고 **ETTC-style 적용 실험**으로 기록한다.
낮은 entropy로 틀리는 과신 모델이 있으면 실패할 수 있으므로 calibration과 함께 본다.

### PriDe
선택지 순환 결과를 이용해 A/B/C/D 위치·기호 선호 prior를 추정하고 제거하는 방법.
저마진 subset에서 추정한 prior를 전체 test에 그대로 일반화하는 것은 별도 가설이므로 우선 같은 정책의 검증 구간에서만 비교한다.

### Ensemble Selection
후보의 단독 점수보다 **현재 앙상블에 추가했을 때의 NC/NW/NetGain**으로 채택 여부를 결정한다.
이미 많은 조합을 탐색한 상태이므로 조합 수를 다시 크게 늘리지 않는다.

## 4. TTA view disagreement는 진단값으로 분리

같은 평균 분포라도
- 네 view가 모두 비슷하게 애매한 경우
- 보기 순서에 따라 서로 다른 답을 강하게 지지하는 경우

는 다르다.

필요하면 Jensen-Shannon divergence 등으로 view 간 충돌을 측정해,
충돌 높은 구간에서 TTA가 실제로 New Correct를 만드는지 / New Wrong을 만드는지 먼저 분석한다.
충돌 자체를 곧바로 품질 신호로 간주하지 않는다.

## 5. 현재 `tools/eval_shift_tta.py` 확인사항

1. **`mean4`는 확률 평균이 아니라 정규화 로그확률 평균**이다.
2. test export는 **결합 계산 대상과 gold가 있는 채점 대상을 분리**해야 한다.
   현재 구조는 gold가 있는 ID 중심으로 `merged`를 만들기 때문에 test-only 내보내기 전에 확인이 필요하다.
3. `vote`는 승자 0 / 나머지 -20의 pseudo-logprob를 만든다.
   이것을 실제 calibrated logprob처럼 후속 soft ensemble에 넣지 않는다.
4. 완료 후에는 TTA 선택 ID가 base/shift1/shift2/shift3에 모두 존재하는지와,
   내보낸 test 결합값이 네 view로 재계산한 값과 일치하는지 검증한다.

## 6. 권장 실행 순서

| 단계 | 비교 | 목적 |
|---|---|---|
| E0 | 현재 최종 앙상블 | 기준선 |
| E1 | 35B만 mean4 TTA로 교체 | 기존 TTA 결합 순효과 |
| E2 | 35B만 prob_mean4로 교체 | 산술평균 vs 기하평균 |
| E3 | 유망 TTA + 모델별 calibration | confidence scale 보정 |
| E4 | calibration 후 ETTC-style 선택/가중 | 문항별 모델 신뢰 활용 |
| 후속 | PriDe / 제한된 ensemble selection | 위치 편향·추가 기여 검증 |

평가는 Accuracy 하나보다 **New Correct / New Wrong / NetGain, 변경 문항 수, 적용 subset 성능**을 같이 본다.
Public leaderboard는 기록만 하고 임계값·가중치를 다시 맞추는 기준으로 사용하지 않는다.

## 참고 문헌

- Zheng et al., *Large Language Models Are Not Robust Multiple Choice Selectors*, ICLR 2024. https://arxiv.org/abs/2309.03882
- Shanmugam et al., *Better Aggregation in Test-Time Augmentation*, ICCV 2021. https://openaccess.thecvf.com/content/ICCV2021/html/Shanmugam_Better_Aggregation_in_Test-Time_Augmentation_ICCV_2021_paper.html
- Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017. https://proceedings.mlr.press/v70/guo17a.html
- Tong et al., *Diversity Matters: Revisiting Test-Time Compute in Vision-Language Models*, 2026. https://arxiv.org/abs/2605.30713
- Caruana et al., *Ensemble Selection from Libraries of Models*, ICML 2004.
