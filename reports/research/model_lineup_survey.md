# 전체 모델 라인업 기술 조사 (2B / 9B / 122B-A10B / 397B-A17B)

작성일: 2026-09-17
대상: 팀 조사 분담의 나머지 4개 모델 — 김현호·박건순(9B, 2B) / 김민우(122B-A10B) / 윤석웅(397B-A17B)
성격: 공개 모델 카드·config.json 기반 조사. **실측은 포함하지 않는다.** 각 담당자의 세부 운용 매뉴얼(smoke test, resolution×batch×VRAM 실측표 등)을 대체하지 않고, 공통으로 필요한 기초 자료를 한 번에 정리해 조사 착수 시간을 줄이는 것이 목적이다. Qwen3.6-35B-A3B의 상세 방법론은 「07. Qwen3.6-35B-A3B 운용 매뉴얼」(`reports/research/qwen36_35b_a3b.md`) 참조 — 동일한 계산 방식을 사용했다.

> 표기: `[확인]` 공식 모델 카드·config.json 기준 / `[계산]` 공식 수치로부터 직접 계산 / `[추정]` 근사식 기반, 실측 필요

---

## 1. 아키텍처 비교 `[확인]`

| 항목 | Qwen3.5-2B | Qwen3.5-9B | Qwen3.6-35B-A3B | Qwen3.5-122B-A10B | Qwen3.5-397B-A17B |
|---|---|---|---|---|---|
| 구조 | Dense | Dense | MoE | MoE | MoE |
| 총 / 활성 파라미터 | 2B | 9B | 35B / 3B | 122B / 10B | 397B / 17B |
| 레이어 수 | 24 | 32 | 40 | 48 | 60 |
| 레이어 패턴 | 6×(3×DeltaNet→1×Attention) | 8×(3×DeltaNet→1×Attention) | 10×(3×DeltaNet→1×Attention) | 12×(3×DeltaNet→1×Attention) | 15×(3×DeltaNet→1×Attention) |
| Full attention 레이어 | 6 | 8 | 10 | 12 | 15 |
| Hidden size | 2,048 | 4,096 | 2,048 | 3,072 | 4,096 |
| Gated Attention | Q 8 / KV 2, dim 256 | Q 16 / **KV 4**, dim 256 | Q 16 / KV 2, dim 256 | Q 32 / KV 2, dim 256 | Q 32 / KV 2, dim 256 |
| Expert 수 | — (dense) | — (dense) | 256 (8 routed+1 shared) | 256 (8 routed+1 shared) | 512 (10 routed+1 shared) |
| Expert intermediate dim | — | — | 512 | 1,024 | 1,024 |
| Context (native/YaRN) | 262,144 / 1,010,000 | 262,144 / 1,010,000 | 262,144 / 1,010,000 | 262,144 / 1,010,000 | 262,144 / 1,010,000 |
| 라이선스 | Apache 2.0 | Apache 2.0 | Apache 2.0 | Apache 2.0 | Apache 2.0 |

**주의 — 9B는 KV head가 4개로 다른 모델(2개)과 다르다.** Group Query Attention 비율이 달라 KV cache 계산식이 다른 모델과 다르게 나온다(2절 참조).

**2026-09-17 확인 — vision encoder가 라인업 전체에서 동일하다.** 9B·122B의 config.json을 직접 조회한 결과 vision_config가 35B와 완전히 같았다(hidden 1152 / intermediate 4304 / head 16 / patch 16 / depth 27 / spatial_merge_size 2). 즉 visual token 계산 규칙(patch 16×16, 2×2 spatial merge, 토큰당 1,024px)과 vision encoder 파라미터 추정치(약 420M, 「07」 5절)가 **2B·9B·122B에 그대로 적용된다**(397B는 아직 개별 확인 전이나 같은 패턴일 가능성이 높다). 122B의 레이어 그룹 패턴도 config의 `full_attention_interval: 4` 필드로 직접 확인해 [추정]에서 [확인]으로 격상했다(1절 표 갱신).

---

## 2. KV cache `[계산]`

공식: `토큰당 KV = 2(K,V) × KV head 수 × head dim × 2 bytes(BF16) × full attention 레이어 수`

| 모델 | 계산 | 토큰당 KV | 10,240 토큰 기준 |
|---|---|---|---|
| Qwen3.5-2B | 2×2×256×2×6 | 4 KB | 약 40 MB |
| Qwen3.5-9B | 2×4×256×2×8 | **16 KB** | 약 160 MB |
| Qwen3.6-35B-A3B | 2×2×256×2×10 | 2 KB | 약 20 MB (「07」 4절과 동일) |
| Qwen3.5-122B-A10B | 2×2×256×2×12 | 2.4 KB | 약 24 MB |
| Qwen3.5-397B-A17B | 2×2×256×2×15 | 3 KB | 약 30 MB |

전 모델에서 KV cache는 가중치·activation 대비 무시할 수 있는 수준이다. 9B가 상대적으로 가장 크지만(KV head 4개), 그래도 100MB대라 병목이 아니다.

---

## 3. 벤치마크 `[확인]`

| 모델 | OCRBench | RealWorldQA | MMBench EN-DEV | OmniDocBench1.5 | CC-OCR | CharXiv(RQ) |
|---|---|---|---|---|---|---|
| Qwen3.5-2B (non-thinking) | 85.4 | 71.2 | 81.3 | 80.9 | 75.8 | 52.6 |
| Qwen3.5-9B | 89.2 | 80.3 | 90.1 | 87.7 | 79.3 | 73.0 |
| Qwen3.6-35B-A3B | `[미확인]` | 85.3 | 92.8 | 89.9 | 81.9 | 78.0 |
| Qwen3.5-122B-A10B | 92.1 | 85.1 | 92.8 | 89.8 | 81.8 | 77.2 |
| Qwen3.5-397B-A17B | 93.1 | **83.9** | 93.7 | 90.8 | 82.0 | 80.8 |

**주목할 점 — 397B가 RealWorldQA에서 122B(85.1)·3.6-35B(85.3)보다 낮다(83.9).** 나머지 5개 지표는 모두 397B가 최고점이지만, 이 지표만 역전됐다. 일반화하면 안 되지만("397B가 전 지표 1위"가 아니라는 사실 자체는), 대형 모델이라고 모든 지표에서 단조 증가하지 않는다는 이번 대회 전체의 관찰(3.6-35B→122B 구간 정체, 「07」 2-1절)과 같은 결의 신호다. CC-OCR·CharXiv 기준 122B→397B 구간 이득은 각각 +0.2, +3.6으로 — CharXiv에서는 여전히 의미 있는 이득이 있다.

**CharXiv 성장 곡선이 가장 가파르다.** 2B(52.6) → 9B(73.0, +20.4) → 3.6-35B(78.0, +5.0) → 122B(77.2, −0.8) → 397B(80.8, +3.6). 9B까지는 급격히 오르고 그 이후 완만해지다가 397B에서 다시 오른다 — CharXiv(복잡한 차트 추론)가 scale에 가장 민감한 지표일 가능성이 있다.

---

## 4. 추론 VRAM `[계산]`

가중치 = 파라미터 × dtype bytes. Dense 모델은 vision encoder(~수백M, 「07」 5절 참조)와 런타임 오버헤드를 합쳐 소형 모델은 +1~2GB, 대형 모델은 +3~5GB로 근사했다.

| 모델 | BF16 | FP8/INT8 | NF4(4bit) | RTX 5060 Ti 16GB |
|---|---|---|---|---|
| Qwen3.5-2B | 약 5GB | 약 3GB | 약 2GB | **가능(전 정밀도)** |
| Qwen3.5-9B | 약 19GB | 약 10GB | 약 6GB | FP8·NF4만 가능, BF16 초과 |
| Qwen3.6-35B-A3B | 약 74GB | 약 39GB | 약 23GB | 불가(「07」 5절) |
| Qwen3.5-122B-A10B | 약 247GB | 약 125GB | 약 65GB | 불가 — H100/A100 다중 또는 B200/B300 |
| Qwen3.5-397B-A17B | 약 799GB | 약 400GB | 약 203GB | 불가 — 대형 클러스터 또는 고용량 클라우드 카드 필요 |

---

## 5. FFT vs PEFT 실행 가능성 `[계산]`

mixed-precision AdamW 학습 상태(약 18 bytes/parameter, 「07」 6-1절과 동일 근사)로 계산했다. MoE 모델은 optimizer state를 라우팅되지 않은 expert까지 **total parameter 전체**에 대해 유지해야 하므로 active parameter 기준으로 줄지 않는다(Dense 모델은 이 문제가 없다).

| 모델 | FFT 학습 상태 | B300(288GB) 기준 | B200(180GB) 기준 | 판정 |
|---|---|---|---|---|
| Qwen3.5-2B | 약 36GB | ×1 | ×1 | 실행 가능 |
| Qwen3.5-9B | 약 162GB | ×1 | ×1 | 실행 가능 |
| Qwen3.6-35B-A3B | 약 630GB | ×3~4 | ×4~5 | 조건부(최종 한 방 후보) |
| Qwen3.5-122B-A10B | 약 2.2TB | ×8 | ×13 | 사실상 불가 — PEFT만 |
| Qwen3.5-397B-A17B | 약 7TB+ | ×25 | ×40 | 불가 — PEFT만 |

NF4 QLoRA는 122B까지 카드 1장으로 가능하다(4절 참조). 397B는 NF4로도 약 203GB라 단일 카드로는 어렵고 다중 카드 또는 고용량 카드(B300 288GB 등)가 필요하다.

**Dense 모델(2B, 9B)은 FFT가 확실히 실행 가능한 범위다.** MoE 모델의 함정(optimizer state가 total parameter 전체에 물림)이 없기 때문이다. 9B 담당(현호·박건순)은 FFT를 우선 후보로 적극 검토할 수 있다 — 이는 「07」 문서의 "9B FFT 우위가 확인되면 35B에서 short pilot" 전략의 전제이기도 하다.

---

## 6. 담당자별 참고 사항

### 6-1. Qwen3.5-9B / 2B (김현호·박건순)

- **9B의 KV head가 4개**로 35B/122B/397B(2개)보다 많다는 점을 놓치지 않아야 한다 — KV cache 계산식을 그대로 복사하면 틀린다(2절 참조).
- 9B는 RTX 5060 Ti에서 BF16이 안 들어간다(약 19GB > 16GB). FP8이나 NF4로 시작해야 한다.
- 2B는 5060 Ti에서 전 정밀도가 가능해 로컬 반복 실험에 가장 유리하다 — 9B에서 확정하기 전에 방향성(view·해상도·crop) 가설을 2B에서 더 싸게 먼저 걸러볼 수도 있다.
- 2B의 CharXiv(52.6)가 9B(73.0)보다 20점 이상 낮다 — 복잡한 추론이 필요한 유형에서 2B는 한계가 뚜렷할 수 있으므로, OCR류보다 추론이 섞인 유형에서 2B 결과를 곧이곧대로 못 믿을 수 있다.
- 두 모델 모두 FFT가 실행 가능한 범위이므로, 9B FFT vs PEFT 비교 실험이 "작은 모델 FFT vs 큰 모델 PEFT" 축(HANDOFF 참조)의 핵심 데이터가 된다.

### 6-2. Qwen3.5-122B-A10B (김민우)

- 3.6-35B-A3B → 122B 구간 이득이 비전 지표 전반에서 거의 없다(0~+0.1, 「07」 2-1절). **122B 투입의 유일한 근거는 우리 데이터(다국어 tiny scene text)에서 이 패턴이 깨지는 경우**뿐이다 — 대회 시작 후 동일 subset 비교가 최우선이다.
- FFT는 약 2.2TB로 사실상 불가능(B300 8장 이상). NF4 QLoRA(카드 1장, 약 65GB)가 유일하게 현실적인 학습 경로다.
- MoE expert weight가 fused 3D 텐서로 저장되는 구조와 관련 LoRA 로딩 버그(vLLM #38520, 「07」 8절)는 122B에도 동일하게 적용될 가능성이 높다 — 35B에서 smoke test로 먼저 확인된 결과를 재사용할 수 있다.
- expert intermediate dim이 35B(512)의 2배(1,024)다 — 같은 rank로 LoRA를 걸어도 어댑터 파라미터 수가 달라질 수 있으니 VRAM 추정 시 반영한다.

### 6-3. Qwen3.5-397B-A17B (윤석웅)

- 이 라인업에서 유일하게 **RealWorldQA가 122B·3.6-35B보다 낮다(83.9)** — 나머지 지표는 최고점이므로 특이 케이스다. "제일 크니까 제일 좋다"고 단정하지 말고 우리 데이터에서도 이런 비단조성이 나타나는지 확인 가치가 있다.
- FFT는 논외 수준(약 7TB, B300 25장 이상). NF4 QLoRA도 약 203GB로 카드 1장에 안 들어가 다중 카드 구성이 필요하다 — 예산(100만원) 대비 실행 시간이 가장 빠듯한 모델이다.
- Expert 수가 512개(다른 모델의 2배), 활성 expert도 11개(10 routed + 1 shared)로 라우팅 구조가 다르다 — 35B/122B에서 확인된 fused weight 구조·LoRA 버그가 동일하게 적용되는지 별도 확인이 필요하다(구조는 같은 패밀리이므로 가능성은 높다).
- 레이어 패턴이 15×(3×DeltaNet→1×Attention)로 가장 큰 그룹 반복이다 — KV cache는 여전히 무시할 수준(약 30MB @ 1만 토큰)이라 메모리 병목은 전적으로 가중치·activation이다.

---

## 7. 미확인 목록

1. ~~vision encoder config~~ → **2026-09-17 해소.** 9B·122B의 config.json을 직접 조회한 결과 vision encoder는 hidden 1152 / intermediate 4304 / head 16 / patch 16 / depth 27 / spatial_merge_size 2로 **35B와 완전히 동일**했다. Qwen3.5/3.6 전 라인업이 같은 vision tower를 공유하는 것으로 보인다 — 「07」 5절에서 계산한 약 420M(0.4B) 추정치가 2B·9B·122B·397B에도 그대로 적용된다.
2. ~~122B의 레이어 그룹 패턴~~ → **2026-09-17 해소.** config.json에 `full_attention_interval: 4`가 명시돼 있어 48층 = 12×(3×DeltaNet→1×Attention)이 확인값으로 격상됐다(기존 [추정]이었음).
3. 9B·2B의 expert 관련 필드는 해당 없음(dense 모델).
4. 전 모델의 실제 학습·추론 VRAM 실측, MoE LoRA 버그 재현 여부(122B·397B 기준)는 각 담당자가 대회 시작 후 확인한다.
5. **(신규) 397B·2B의 config.json은 아직 직접 조회하지 않았다.** 9B·122B와 같은 vision tower를 쓸 가능성이 높지만(Qwen 계열은 세대·크기 무관하게 vision encoder를 공유하는 패턴), 397B는 아직 확인 전이다.

## 출처

- [Qwen3.5-9B 모델 카드](https://huggingface.co/Qwen/Qwen3.5-9B)
- [Qwen3.5-2B 모델 카드](https://huggingface.co/Qwen/Qwen3.5-2B)
- [Qwen3.5-122B-A10B 모델 카드](https://huggingface.co/Qwen/Qwen3.5-122B-A10B)
- [Qwen3.5-397B-A17B 모델 카드](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)
- [Qwen3.6-35B-A3B 모델 카드 및 config.json](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (교차 검증 기준)
