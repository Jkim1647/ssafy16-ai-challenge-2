# A/AB/ABC 데이터 Ablation 결과 — 2026-09-22

**목적:** dev 유래 추가 학습데이터(dev_pseudo, codex_regen)가 어려운 문항 성능을 실제로 올리는지,
한 조각씩만 바꿔(동일 조건) 검증. 9B QLoRA-free BF16 LoRA, seed 1, r8/alpha16, lr 1e-4, 1 epoch, view full.
데이터셋만 교체(`--train-csv`). Colab Pro+ RTX PRO 6000(G4) 3장 병렬.

## 데이터셋 정의
- **A** = train_orig 6,002 (원본 train.csv − val667 홀드아웃 − 중복 45)
- **AB** = A + dev_pseudo 692 (3모델 합의 pseudo 라벨)
- **ABC** = AB + codex_regen 662 (사람검수 통과 r2/r3/r4)

평가셋(학습 제외, canonical-id assert로 강제): val667(667, 일반) / gold_v3(331, 어려운) / H184(184, 최난, gold_v2).

## 결과

| 실험 | 데이터 | val667 | gold_v3 (331) | H184 (184) | 학습시간 |
|---|---|---|---|---|---|
| A | train_orig 6,002 | 0.9610 | 0.6284 | 0.6576 | 43.7분 (2623s) |
| AB | +dev_pseudo 692 | **0.9655** | 0.6344 | 0.6576 | 49.7분 (2980s) |
| ABC | +codex 662 | 0.9625 | **0.6465** | **0.6630** | 53.8분 (3230s) |

A 대비: AB = val +0.45%p / gold +0.60%p / H184 ±0. **ABC = val +0.15%p / gold +1.81%p / H184 +0.54%p.**

gold_v3 answer별(ABC): a 0.6951 / b 0.6296 / c 0.5366 / d 0.6991 — c(정답) 유형이 약함.

## 해석
- **gold_v3(어려운셋)가 A < AB < ABC로 단조 증가.** 특히 codex(C)를 넣은 ABC에서 최대 점프(AB→ABC +1.2%p ≈ 4문항).
- **ABC가 세 지표 모두 A 이상** → 채택 기준(val + 어려운셋 동반 개선) 통과. dev 유래 데이터(특히 codex)가 어려운 문항에 기여.
- AB는 val667 최고지만 gold는 ABC가 우세. dev_pseudo(B)는 쉬운 문항, codex(C)는 어려운 문항 경향.

## 통계 주의
델타가 작다(gold_v3 n=331 → 1문항 ≈ 0.30%p, 95% CI ≈ ±5%p). gold_v3 A→ABC +1.8%p(≈6문항)는 **약한 양성**이지 확정 아님.
단조 증가 + ABC가 전부 A 이상이라 방향은 분명. paired bootstrap 재확인 권장.

## 다음 결정
1. **codex(C)가 어려운 문항에 실제 기여 확인** → HOLD 중인 codex r1·r5(약 1,054건) 사람검수 재개가 값어치 있다는 근거.
2. **학습 데이터 구성은 ABC(train+dev_pseudo+codex)로 확정** 가능.
3. 다음 축: ABC 위에 판독(read)+타일(tiles2x2) 입력, 또는 35B로 스케일업.

## 재현/자료
- 코드: `tools/run_ablation_abc.py`(러너), `tools/colab_lora_train.py`(`--train-csv`/`--source-weight`/canonical assert), `tools/ablation_metrics.py`(집계).
- 학습셋 빌드/QA: `tools/qa_train_all.py`. 데이터 lineage·누수검사: `runs/dev_validation_v1/FINAL_DATA_REPORT_20260922.md`(git 미추적, 로컬).
- **산출물(팀 Drive 158): `MyDrive/ai2_abc/out/abc_{A,AB,ABC}/`** — adapter, meta.json, metrics.json, 예측 jsonl.
- Drive 재구성 번들: `MyDrive/ai2_abc/ai2_abc_recipe.zip`, `train_{A,AB,ABC}.csv`, `tools/`.

---

## 정정 (2026-09-23) — "ABC 확정 가능"은 순환성을 못 본 판단이다

위 결론은 `gold_v3` 가 **사람 검수 라벨셋**이고 `codex_regen` 이 **사람 검수 학습데이터**라는
점을 고려하지 않았다. 사람 검수 라벨로 학습해서 사람 검수 평가셋이 오른 것은 순환이다.

**공식 라벨 평가셋(val667)만 보면 방향이 반대다.**

| 추가 데이터 | 성격 | val667 | gold_v3 |
|---|---|---|---|
| dev_pseudo 692 | **모델 3개 합의** (dev 라벨 미사용) | **0.9610 → 0.9655 (+3문항)** | +0.60%p |
| codex 662 | **사람 검수** | **0.9655 → 0.9625 (−2문항)** | +1.21%p |

### 09/22 에는 몰랐던 것

1. **채점 대상은 사람들의 합의**다(담당 프로 안내, 09/23). 즉 `train.csv` 의 공식 정답이
   목표이고 우리 검수 라벨은 목표가 아니다.
2. **우리 검수자들의 수정이 합의에서 멀어졌다** — 81% → 43%.
3. 검수 gold 와 dev 다수표의 일치율이 **0.567** 이다(97건 겹침, 09/23 측정).

따라서 **사람 검수 라벨을 학습에 넣지 않는다.** 검수 작업의 쓸모는 결함 문항 제외
(71건, R3 에 반영)와 진단이다.

### 반대로 dev_pseudo 는 살릴 값어치가 있다

pseudo 라벨은 **dev 의 라벨을 쓰지 않는다.** dev 이미지만 가져오고 정답은 모델 합의로
붙이므로, 09/23 에 확인된 "dev 채점자 표의 최다 득표가 3/5 를 넘는 항목이 0건" 문제를
통째로 비껴간다. 과녁도 합의 쪽이라 실제 채점 대상과 맞는다.

지금은 09/22 보다 조건이 낫다 — 그때는 9B 합의였는데 지금은 35B·397B·27B 의
read+tiles 예측이 있다. dev 2,683장에서 세 모델 전원 일치 항목만 뽑으면 되고
**학습 데이터 생성 자체는 비용 0** 이다.

단, 1차 챌린지에서 dev 추가 학습이 0.92116 → 0.91919 로 떨어진 선례가 있으므로
**이중 게이트(train 홀드아웃 + H408)를 반드시 건다.** 또한 위 수치는 9B·view=full
조건이라 현재 27B/35B·read+tiles 구성으로의 전이는 재확인이 필요하다.

### 규칙 확인 필요

Kaggle 기본규칙 4-b 는 validation·test 레코드에 대한 hand labeling 을 제출에 쓰는 것을
금지한다. 검수 대상이 dev 였으므로, **사람 검수 라벨을 제출 모델의 학습에 쓰는 것**이
이 조항에 걸리는지는 확인이 필요하다(감사·평가 용도는 별개로 정리돼 있다).
어차피 위 근거로 쓰지 않기로 하므로 당장 문제가 되지는 않는다.
