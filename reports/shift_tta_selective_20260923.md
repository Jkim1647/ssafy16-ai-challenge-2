# 선택적 선택지 순환(shift) TTA — 실행 준비 (2026-09-23)

`tools/select_low_margin.py` · `tools/eval_shift_tta.py` · `tools/build_shift_tta_data.py`
· `tools/runpod/run_shift_tta_35b.sh` · `tools/nebius/run_shift_tta_397b.sh`. **여기까지 GPU 비용 0.**

## 왜 전 문항을 돌리지 않는가 — 이득이 저마진 밴드에만 있다

우리가 가진 유일한 "같은 모델 전 문항 4순서" 세트(val667, 35B `ours`)를 마진 구간별로 쪼갰다.

| 구간 | n | shift0 | 4순서 평균 | Δ |
|---|---|---|---|---|
| **마진 하위 5%** | 33 | 14 | 19 | **+5** |
| 5~10% | 33 | 24 | 24 | +0 |
| 10~25% | 100 | 94 | 94 | +0 |
| 상위 75% | 501 | 498 | 498 | +0 |

**전 문항 4배 추론의 이득 +5 가 하위 5% 밴드에서 전부 나온다.** 나머지 세 구간은 정확히 0이다.
고마진 문항은 순서를 바꿔도 답이 안 바뀌므로 추론비만 쓴다.

train6047 앙상블에서도 같은 구조다 — 마진 하위 5%(302건)에 오답 **94/178 = 53%** 가 몰린다.
세 지표(p1−p2, p1/p2, 엔트로피)가 거의 같은 302건을 고르므로 지표 선택은 중요하지 않다.

## 왜 이득이 나는가 — 위치 편향

train6047 에서 정답 글자별 오답률:

| 정답 | a | b | c | d |
|---|---|---|---|---|
| 오답률 | **3.69%** | 2.56% | **1.86%** | **3.72%** |

a·d 가 c 의 약 2배다. 모델이 a·d 를 덜 고른다. 선택지를 순환시켜 평균하면 이 편향이 상쇄된다.
서울_9반 다른 참가자(38위)의 선택적 재검사(+1.25%p)도 같은 메커니즘이다.

## 비용 — 4배가 아니라 0.15배

| 대상 | 선택 | 추가 추론 |
|---|---|---|
| train6047 (검증용) | 302 / 6,047 | 302 × 3 = 906 |
| test (적용용) | 336 / 6,714 | 336 × 3 = 1,008 |
| **합계** | **638** | **1,914** |

base(shift0)는 두 셋 모두 이미 있으므로 다시 돌리지 않는다.
train·test 를 **한 파일로 묶어 한 번만 로드**한다 — 비용의 대부분이 모델 로드다.

| 모델 | 실측 근거 | 예상 소요 | 예상 비용 |
|---|---|---|---|
| 35B read+tiles | 0.673초/문항, 로드 83초 (RTX PRO 6000) | 약 25분 | **약 $0.9** ($2.09/h) |
| 397B FP8 | 0.56초/문항, 로드 201초 (8×H100 TP8) | 약 25분 | **약 $13** ($30.81/h) · 4×H200 이면 약 $8 |

**35B 쪽은 사실상 공짜다.** 397B 를 붙일지는 35B 결과를 보고 정하면 된다.

## 결합 규칙은 정해두지 않았다 — 넷을 같은 자로 비교한다

| 규칙 | 내용 |
|---|---|
| `base` | 재추론 없음(기준선) |
| `mean4` | base + s1~3 로그확률 평균 ← val667 에서 +5 를 낸 방식 |
| `mean3` | s1~3 만 평균(base 를 버린다) ← 다른 참가자 "교체" 방식. 그쪽 n=200 에서 교체 97.0% > 평균 96.0% |
| `vote` | 네 순서 argmax 다수결, 동률이면 base |

`mean3` 과 `mean4` 가 갈리는 이유가 있다. base 는 "원래 순서에서 헷갈린 그 답"이라 위치 편향을
그대로 담는다. 편향을 지우려면 빼야 한다는 주장과, 표본 4개 중 하나를 버리면 분산이 커진다는
주장이 둘 다 성립한다. 재는 수밖에 없다.

도구 검증(val667, 비용 0)에서 네 규칙이 이렇게 나왔다:

| 규칙 | 정답 (n=667) | base 대비 |
|---|---|---|
| base | 630 | — |
| mean4 | 635 | +5 |
| mean3 | 635 | +5 |
| vote | 634 | +4 |

n=667 에서는 셋 다 paired bootstrap CI 가 0을 포함한다(잡음). **판정은 train6047 에서 한다** —
1문항이 0.017%p 라 같은 크기가 유의해진다.

## 남은 불확실성

- **정확도 구간이 다르다.** 위 val667 근거는 `ours` 프롬프트 35B(0.945)에서 나왔다. 최종 구성은
  read+tiles(0.9655)다. 높은 정확도에서도 같은 이득이 나오는지는 **미확인**이며, 그걸 재는 것이
  이 실험이다. 안 나오면 접는다.
- **선택 기준을 test 자체의 예측으로 정한다.** gold 를 쓰지 않으므로 누수는 아니지만,
  test 336건은 "우리 모델이 헷갈린 것"이라는 뜻이라 train 에서 잰 이득률이 그대로 옮겨간다는
  보장은 없다.

## 채택 게이트

평소와 같다 — **train6047 에서 유의 개선 + H408 에서 유의 악화 없음**.
train6047 에서 잡음이면 채택하지 않는다.

## 실행 순서

```bash
# 1. 대상 선정 (이미 실행됨, 결과가 runs/shift_tta/ 에 있다)
python tools/select_low_margin.py --pred <35B train6047> --pred <397B train6047> --pct 5 --out runs/shift_tta/train6047
python tools/select_low_margin.py --pred <35B test>      --pred <397B test>      --pct 5 --out runs/shift_tta/test
python tools/build_shift_tta_data.py --ids runs/shift_tta/train6047/ids.json --ids runs/shift_tta/test/ids.json --out runs/shift_tta/data

# 2. 추론 (GPU) — runs/shift_tta/data 를 파드로 올린 뒤
bash tools/runpod/run_shift_tta_35b.sh          # 약 $0.9
bash tools/nebius/run_shift_tta_397b.sh         # 약 $8~13, 35B 결과 보고 결정

# 3. 결합·채점 (GPU 불필요)
python tools/eval_shift_tta.py --ids runs/shift_tta/train6047/ids.json \
  --model "35B=<base>,<s1>,<s2>,<s3>" --model "397B=<base>,<s1>,<s2>,<s3>" \
  --emit-dir runs/shift_tta/merged --out reports/shift_tta_result.md

# 4. 채택되면 제출 (emit 된 jsonl 을 기존 제출 경로에 그대로 넘긴다)
python tools/ensemble_submit.py submissions/ens_shift_tta_test.csv \
  runs/shift_tta/merged/35B_mean4_predictions.jsonl runs/shift_tta/merged/397B_mean4_predictions.jsonl --logprob
```
