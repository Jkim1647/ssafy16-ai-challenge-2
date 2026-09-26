# FINAL DATA REPORT — 2026-09-22

학습 데이터 동결(freeze) QA 및 A/AB/ABC ablation 준비 보고서.
기존 `DATA_CONSOLIDATION_20260922.md`를 파괴하지 않고 새로 작성했다. 숫자는 재검증으로 일치함을 확인했다.

산출물(모두 `runs/dev_validation_v1/`):
- `train_all_20260922.csv` (7,356) — 통합 원장(source 컬럼 포함)
- `train_A.csv` (6,002) / `train_AB.csv` (6,694) / `train_ABC.csv` (7,356) — ablation용, source 유지
- `qa_train_all_20260922.json`, `QA_TRAIN_ALL_20260922.txt` — 무결성 재검증 결과
- `codex_regen_train_ok.csv` (662) / `codex_regen_excluded.csv` (30)
- 코드: `tools/qa_train_all.py`, `tools/run_ablation_abc.py`, `tools/ablation_metrics.py`
- 학습 코드 변경: `tools/colab_lora_train.py` — `--train-csv`, `--source-weight`, canonical 누수 assert 추가

---

## 1. 최종 데이터 lineage

```
대회 원본
├─ train.csv (6,714)
│    ├─ − val667 홀드아웃 (667)          [EVAL, train_XXXX]
│    ├─ − 중복격리 (45, known_duplicates) [train_duplicate_exclusions.csv]
│    └─ = A: train_orig (6,002)          ← 주력 학습셋
│
└─ dev.csv (2,683)  ※ 라벨러 5인 합의 3/5 이하만 모인 편향표본(만장일치 0건)
     ├─ 3모델(Claude+9B+35B) consensus → train_pseudo_candidates_final (1,002)
     │    ├─ − gold_v3 누수 (26)
     │    ├─ − codex_regen과 같은 이미지 중복 (284, codex 우선)
     │    └─ = B: dev_pseudo (692)
     │
     └─ codex 재생성 dev_regen (1,317, reviewer_1~5)
          ├─ r1(0001-0544) 264  ─ 미검수(HOLD)
          ├─ r2(0545-1076) 264  ─ 사람검수 완료
          ├─ r3(1077-1593) 263  ─ 사람검수 완료
          ├─ r4(1594-2123) 263  ─ 사람검수 완료
          ├─ r5(2129-2683) 263  ─ 미검수(HOLD)
          └─ 검수완료 790 중
               − 결함(정답오류/복수정답/공백/누수노트) 30
               − gold_v3 누수 98
               = C: codex_regen clean (662)

최종 학습셋 = A ∪ B ∪ C = 7,356  (id·이미지 중복 0, gold/val 누수 0)
```

## 2. source별 건수 / answer 분포

| source | n | a | b | c | d |
|---|---|---|---|---|---|
| A train_orig | 6,002 | 1,532 | 1,430 | 1,544 | 1,496 |
| B dev_pseudo | 692 | 146 | 149 | 176 | 221 |
| C codex_regen | 662 | 167 | 175 | 158 | 162 |
| **합계** | **7,356** | 1,845 | 1,754 | 1,878 | 1,879 |

- A는 균형이 좋다. B(dev_pseudo)는 d편중(221)이 있으나 소량이라 저비율 혼합 시 영향 제한적.

## 3. 제외 건수와 이유

| 대상 | 제외 | 이유 |
|---|---|---|
| train.csv | 667 | val667 홀드아웃(EVAL) |
| train.csv | 45 | 중복 격리(known_duplicates) |
| dev_pseudo(1,002) | 26 | gold_v3(331)와 같은 dev 이미지 → 평가 누수 |
| dev_pseudo(1,002) | 284 | codex_regen과 같은 dev 이미지(codex=사람검수라 우선, 중복 제거) |
| codex_regen(790) | 30 | 사람검수 결함: 생성정답≠이미지(유사 오답), 복수정답, 공백, 정답부재 |
| codex_regen(790) | 98 | gold_v3와 같은 dev 이미지 → 평가 누수 |
| **codex_regen r1·r5** | **~1,054** | **사람검수 미완(HOLD) — 자동판정만으로 학습 투입 금지** |

## 4. 누수 검사 결과 (`tools/qa_train_all.py`)

문자열 id가 아니라 **canonical image id(파일 stem) + 실제 파일 SHA256 + group(known_duplicates∪duplicate_candidates)** 세 축으로 검사.

| 검사 | 결과 |
|---|---|
| id 중복 | 0 |
| 필드 결측(질문/보기/answer) | 0 |
| answer ∉ {a,b,c,d} | 0 |
| 정답 선택지 텍스트 공백 | 0 |
| 이미지 파일 부재 | 0 / 7,356 |
| 이미지 decode 실패 | 0 |
| train_all 내부 이미지 hash 중복 | 0 |
| **gold_v3(331) 누수** | **id 0 · imagehash 0 · group 0** |
| **val667(667) 누수** | **id 0 · imagehash 0 · group 0** |
| source 상호 중복(같은 이미지) | 0 |

### ⚠️ test exact-pixel 중복 — 보고만, 자동수정 안 함
`duplicate_candidates.csv`의 exact_pixels 4쌍 중 **2건이 현재 학습셋 이미지와 test가 픽셀 동일**:
- (개별 문항 서술 생략)
- (개별 문항 서술 생략)

- (개별 문항 서술 생략)
→ test 라벨은 없고 hand-labeling도 금지이므로 **정답 누수는 아니다.** 다만 모델이 test 원본 2장을 학습 중 보게 된다. 규칙 위반은 아니나 기록으로 남긴다. 지시대로 자동 제거하지 않았다.

## 5. 평가셋 정의 (학습 절대 포함 금지)

| 평가셋 | 크기 | 이미지 | 용도 |
|---|---|---|---|
| **val667** | 667 | train_XXXX (홀드아웃) | 일반 분포 성능 |
| **gold_v3** | 331 | dev_XXXX (사람 확정) | 어려운 문항 성능 |
| ┗ H184 (gold_v2) | 184 | gold_v3의 hard subset | 가장 어려운 부분집합 |

`tools/colab_lora_train.py`에 **canonical assert 추가**(id + 파일 stem 두 축):
```
assert not (set(fit.id) & val_ids)              # 문자열 id
assert not (fit_cid & val_cid)                  # canonical (val667)
assert not (set(fit.id) & hard_ids)             # 문자열 id
assert not (fit_cid & hard_cid)                 # canonical (gold_v3)  ← 기존엔 없어 codex_regen을 못 잡았음
```
격리 검증에서 A/AB/ABC 모두 4개 assert 통과 확인.

## 6. A/AB/ABC 데이터셋 정의

| 실험 | 파일 | fit | 구성 |
|---|---|---|---|
| **A** | `train_A.csv` | 6,002 | train_orig only |
| **AB** | `train_AB.csv` | 6,694 | + dev_pseudo 692 |
| **ABC** | `train_ABC.csv` | 7,356 | + codex_regen 662 |

- 동일 조건 고정(`tools/run_ablation_abc.py`): model `Qwen/Qwen3.5-9B`, seed 1, r8/alpha16, lr 1e-4, 1 epoch, grad-accum 16, view full. **데이터셋(`--train-csv`)만 교체.**
- 각 실험 기록(`tools/ablation_metrics.py`): val667 acc, gold_v3 acc, H184 acc, answer별 a/b/c/d acc, auto_type별 acc(OCR/text-heavy vs visual), source별 건수, best checkpoint, train/val loss, 학습 시간 → `metrics.json` + `ablation_summary.csv`.
- 실행(Colab/RunPod):
  ```
  python tools/run_ablation_abc.py --data /content/data --out <OUT> \
         --val-ids val667_ids.json --hard-gold hard_eval_gold_v3.json
  ```
- **본 환경(로컬 CPU)에서는 GPU 학습을 실행하지 않았다.** 데이터·코드·assert·러너까지 실행 준비 완료 상태로 동결한다.

### 비율 실험(ABC 유효 시)
`--source-weight`로 B+C 오버샘플. 우선순위 A→AB→ABC 확인 후:
```
# 예: dev계열 2배
python tools/run_ablation_abc.py ... --only ABC --source-weight 'train_orig:1,dev_pseudo:2,codex_regen:2'
```
10/20/30% 비율은 base 6,002 대비 B+C(1,354=약 18.4%)를 기준으로 가중치를 조절한다.

## 7. 아직 HOLD인 데이터

- **codex_regen r1(0001–0544) + r5(2129–2683) ≈ 1,054건.** 파일은 있으나 **사람검수 미완.**
- 근거: 사람검수 완료 2배치(r2·r4)에서 Codex 자동 플래그가 실제 결함의 **1/4~1/3만** 포착함이 확인됨. 미검수분을 자동판정만으로 넣으면 "생성정답≠이미지" 오라벨이 섞인다.
- HOLD 유지. ABC 실험에서 C의 gold_v3 기여가 입증된 경우에만 검수 재개 가치 있음.

## 8. 추가 human labeling이 필요한가 — 근거

- **지금은 불필요.** 먼저 A/AB/ABC를 돌려 C(codex_regen 662)가 gold_v3/H184를 실제로 올리는지 본다.
- 판단 규칙:
  - **ABC의 gold_v3·H184가 AB보다 의미있게↑** → C가 유효 → r1·r5(1,054) 사람검수 재개가 gold 성능 향상으로 이어질 기대값이 큼 → 검수 투입.
  - **ABC ≈ AB (또는 val667은 오르나 gold 정체)** → C의 한계효용 낮음 → r1·r5 검수는 비용 대비 가치 낮음 → HOLD 유지, 대신 gold 평가셋 확장(H184→더 큰 hard set)에 라벨링 예산을 쓴다.
- 어느 쪽이든 **평가셋(gold)의 신뢰도·크기**가 병목이지 학습량이 병목이 아니다(dev 만장일치 0건이라 라벨 상한 자체가 낮음).

---

## 숫자 일치 확인 (DATA_CONSOLIDATION_20260922.md 대비)

| 항목 | consolidation | 본 재검증 | 일치 |
|---|---|---|---|
| A train_orig | 6,002 | 6,002 | ✅ |
| B dev_pseudo | 692 | 692 | ✅ |
| C codex_regen | 662 | 662 | ✅ |
| 합계 | 7,356 | 7,356 | ✅ |
| gold/val 누수 | 0 | 0 (id·hash·group) | ✅ |

불일치 없음. 단, **초기 탐색 메시지에서 보고했던 "dev_pseudo ∩ gold = 0, 사용가능 1,002"는 오류였다** — `.jpg` 접미사 불일치로 인한 가짜 0이었고, 이미지 기준 재검사에서 gold 누수 26 + codex 중복 284가 드러나 692로 정정됨. 이번 QA는 canonical id 기준이라 이 오류가 재발하지 않는다.
