# Qwen3.6-35B-A3B 운용 매뉴얼

담당: 김진영 · 작성일 2026-09-16 · 기한 09/18(금)
대상: **Qwen3.6-35B-A3B** (팀 조사 분담의 "32B 근처" 슬롯 — 2026-09-16 3.5-35B-A3B에서 교체 확정)

> **표기.** `[확인]` 공식 모델 카드·문서 기준 / `[계산]` 공식 수치로부터 직접 계산 / `[추정]` 근사식 기반, 실측 필요 / `[미확인]` 아직 근거 없음
> 이 문서의 VRAM·batch 수치는 **실측이 아니다.** 대회 당일 채울 자리를 표로 만들어 두었다.

> **2026-09-16 교체 기록.** 이 문서는 원래 Qwen3.5-35B-A3B 조사로 시작했으나, 벤치마크상 3.6-35B-A3B가 같은 비용에서 우위라 팀 결정으로 채택 모델을 교체했다. 단순 문자열 치환이 아니라 아래 원칙으로 다시 썼다.
> - 제목·모델 ID는 전부 Qwen3.6-35B-A3B 기준.
> - 벤치마크는 3.6 수치로 교체하고, OCRBench·CC-OCR처럼 **3.6에서 아직 공개되지 않은 항목은 `[미확인]`으로 남긴다** (3.5 수치로 대체하지 않는다).
> - 아키텍처·VRAM·batch 수치는 3.5와 아키텍처가 동일하다는 근거로 **재사용하는 초기 planning estimate**이며, 대회 시작 후 첫 smoke test에서 실측 확인한다. "그대로 적용된다"고 단정하지 않는다.
> - visual token 계산 규칙은 그대로 시작하되, **첫 실행에서 processor가 만든 실제 토큰 수를 출력해 확정**하는 것이 여전히 최우선 검증 항목이다.
> - QLoRA/QDoRA는 3.6에서도 몇 step 학습 → save → 프로세스 재시작 → reload → inference 순서로 다시 통과시켜야 한다 (3.5에서 검증됐다고 3.6에서 자동으로 되는 것이 아니다).
> - LR·rank·effective batch는 3.5에서 좋았던 값을 그대로 확정하지 않고 **초기 후보로만** 쓴다. 모델이 바뀌면(크기가 같아도) 수치형 하이퍼파라미터는 재검증 대상이다.
> - Qwen3.5-35B-A3B는 부록 A에 **비교 baseline**으로 남긴다.

---

## 0. 요약 — 먼저 읽을 세 줄

1. **3.6-35B-A3B가 122B-A10B와 비전 지표에서 사실상 동급이다 — 이제 다국어 OCR 지표로도 확인됐다.** RealWorldQA 85.3 vs 85.1(−0.2), MMBench 92.8 vs 92.8(0.0), OmniDocBench 89.9 vs 89.8(−0.1), **CC-OCR 81.9 vs 81.8(+0.1), CharXiv 78.0 vs 77.2(+0.8)**(2026-09-16 확인). 파라미터는 122B가 3.5배인데 이득이 거의 없어, **122B 투입 근거가 3.5-35B 기준일 때보다 더 약해졌다.**
2. **KV cache가 거의 공짜다.** 40층 중 full attention이 10층이고 KV head가 2개라 토큰당 약 20KB다. 1만 visual token을 넣어도 KV는 약 200MB다. **고해상도·다중 crop 전략에 구조적으로 유리하다.** (아키텍처가 3.5와 동일해 그대로 성립)
3. **FFT 메모리는 active 3B가 아니라 total 35B 기준이다.** 약 630GB로 B300 ×3~4가 필요하다. 반면 NF4 QLoRA는 한 장으로 충분하다. (아키텍처 동일 — 3.5 추정치를 초기값으로 재사용, smoke test로 확인 예정)

---

## 1. 모델 기본정보 `[확인, Qwen3.6-35B-A3B 모델 카드 기준 — 3.5와 아키텍처 동일]`

| 항목 | 값 |
|---|---|
| 구조 | MoE + 하이브리드 어텐션 |
| 총 / 활성 파라미터 | 35B / 3B |
| Expert | 총 256개, 토큰당 9개 활성 (routed 8 + shared 1) |
| Expert intermediate dim | 512 |
| Hidden size | 2,048 |
| 레이어 | 40층 = `10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))` |
| Gated DeltaNet | V head 32, QK head 16, head dim 128 |
| Gated Attention | Q head 16, **KV head 2**, head dim 256, RoPE dim 64 |
| Context | 262,144 native (YaRN으로 1,010,000까지) |
| 모달리티 | 텍스트 · 이미지 · 비디오 (early fusion 네이티브 멀티모달) |
| 언어 | 201개 언어·방언 |
| 라이선스 | **Apache 2.0** |
| 권장 dtype | BF16 / F32, safetensors |

**아키텍처 해석.** 40층 중 **Gated Attention은 10층뿐**이고 나머지 30층은 선형 어텐션(Gated DeltaNet)이다. 선형 어텐션은 시퀀스 길이에 무관한 고정 크기 상태를 유지하므로, **긴 입력(= 많은 visual token)에서 메모리·시간이 거의 선형으로만 증가한다.** 이번 대회의 고해상도·crop 전략과 궁합이 좋다.

---

## 2. 벤치마크 — Qwen 라인업 내 위치 `[확인]`

Qwen3.6은 27B(dense)와 35B-A3B(MoE)만 공개됐다. 9B·122B급 3.6 체크포인트는 없으므로, scale-up 흐름을 보려면 **세대를 섞어서 각 scale의 최강 모델을 잇는다** — 9B는 3.5, 35B는 3.6(채택), 122B·397B는 3.5. 세대가 섞인 비교이므로 절대 수치 차이에는 아키텍처·학습 데이터 차이도 섞여 있다는 점을 감안한다.

| 모델 | OCRBench | RealWorldQA | MMBench EN-DEV | OmniDocBench1.5 | CC-OCR | CharXiv(RQ) |
|---|---|---|---|---|---|---|
| Qwen3.5-9B | 89.2 | 80.3 | 90.1 | 87.7 | 79.3 | 73.0 |
| Qwen3.5-35B-A3B (참고, 부록 A) | 91.0 | 84.1 | 91.5 | 89.3 | `[미확인]` | `[미확인]` |
| **Qwen3.6-35B-A3B (채택)** | `[미확인]` | **85.3** | **92.8** | **89.9** | **81.9** | **78.0** |
| Qwen3.5-122B-A10B | 92.1 | 85.1 | 92.8 | 89.8 | 81.8 | 77.2 |
| Qwen3.5-397B-A17B | 석웅 담당 | 석웅 담당 | 석웅 담당 | 석웅 담당 | | |

**2026-09-16 갱신 — CC-OCR·CharXiv 실측값 확보.** Hugging Face 공식 모델 카드(`huggingface.co/Qwen/Qwen3.6-35B-A3B`)에서 CC-OCR 81.9, CharXiv(RQ) 78.0을 확인했다. **다국어 OCR 지표(CC-OCR)에서도 3.6-35B-A3B(81.9)가 3.5-122B-A10B(81.8)를 근소하게 앞선다.** CharXiv도 78.0 vs 77.2로 마찬가지다. 아래 2-2절의 "일반 실사진 QA에서만 확인됐다"는 한계가 이제 OCR 특화 지표에서도 해소됐다 — 단 OCRBench는 여전히 3.6에서 미공개다.

### 2-1. 구간별 이득 `[계산]`

| 구간 | RealWorldQA | MMBench | OmniDocBench | CC-OCR | CharXiv | OCRBench |
|---|---|---|---|---|---|---|
| 9B → **3.6**-35B-A3B | **+5.0** | +2.7 | +2.2 | +2.6 | +5.0 | `[미확인]` |
| **3.6**-35B-A3B → 122B-A10B | **−0.2** | **0.0** | **−0.1** | **−0.1** | **−0.8** | `[미확인]` |
| 참고: 9B → 3.5-35B → 122B (전부 3.5) | +3.8 / +1.0 | +1.4 / +1.3 | +1.6 / +0.5 | `[미확인]` / +2.5 | `[미확인]` | +1.8 / +1.1 |

**핵심 — 3.5일 때보다 122B 투입 근거가 더 약해졌고, 이제 다국어 OCR 지표로도 확인된다.** 3.6-35B-A3B → 122B 구간은 CC-OCR −0.1, CharXiv −0.8로 **OCR 특화 지표에서도 이득이 없거나 마이너스다.** 파라미터는 122B가 3.5배인데 얻는 게 없다면, 예산을 122B FFT/QLoRA가 아니라 crop·OCR·앙상블에 쓰는 쪽이 더 타당하다. 남은 유일한 공백은 OCRBench뿐이다.

### 2-2. 이 표를 그대로 믿으면 안 되는 이유

- RealWorldQA·MMBench는 **일반 실사진 QA**다. 이번 과제는 한국어 tiny scene text다. 다만 2026-09-16 CC-OCR·CharXiv 확인으로 이 한계는 상당 부분 해소됐다 — 다국어 OCR 지표에서도 같은 결론(3.6-35B ≈ 122B)이 재현됐다.
- **OCRBench만 여전히 3.6-35B-A3B에서 미공개다.** `[미확인]` 나머지 5개 지표는 전부 확보됐다.
- 3.5 경로(9B→122B)에서 CC-OCR 이득은 +2.5였다. 반면 3.6 경로(9B→3.6-35B→122B)는 CC-OCR 이득이 +2.6 → −0.1로, **scale-up 이득이 35B 구간에서 이미 소진된 패턴이 CC-OCR에서도 뚜렷하다.**
- 한국어 scene text 성능은 공개 벤치마크 어디에도 없다. **대회 시작 후 우리 데이터 300~500문제에서 직접 재는 것이 유일한 확정 근거다.**

---

## 3. 이미지 해상도와 visual token

### 3-1. 토큰 계산 규칙 `[확인 — 2026-09-16 config.json으로 근거 격상]`

| 항목 | 값 | 근거 |
|---|---|---|
| Patch size | 16 × 16 | 공식 저장소 Q&A + `vision_config.patch_size` |
| Spatial merge size | **2 × 2** | `vision_config.spatial_merge_size` — 이번에 config.json에서 직접 확인 |
| 지원 해상도 범위 | 256×256 ~ 4096×4096, 범위 내에서는 **리사이즈하지 않음** | 공식 저장소 Q&A |
| 권장 `min_pixels` | 313,600 (≈ 560×560) | 실사용 vLLM 설정 |
| 권장 `max_pixels` | 7,840,000 (≈ 2800×2800) | 실사용 vLLM 설정 |
| **토큰당 픽셀** | **약 1,024 px (32×32)** | 아래 근거 |

**32×32로 보는 근거 — 2026-09-16 확정.** 기존에는 공식 Q&A의 모순된 서술(토큰당 256픽셀 vs "28×28 가정이 30% 적게 나옴")로부터 2×2 spatial merge를 역산으로 추정했다. 이번에 `huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json`의 `vision_config`를 직접 조회해 `spatial_merge_size: 2`를 확인했다 — **더 이상 추정이 아니라 확인된 값이다.** 16×16 patch × 2×2 merge = 32×32 = 1,024픽셀/토큰이 공식 구성과 일치한다. `min_pixels`/`max_pixels`를 1,024로 나누면 306 / 7,656으로 깔끔하게 떨어지는 것도 그대로 성립한다.

> **남은 검증.** 계산 규칙 자체는 확인됐지만, 실제 `AutoProcessor`가 만드는 `visual_tokens` 개수가 이 규칙대로 나오는지는 여전히 대회 당일 첫 실행에서 찍어 확인한다 — 라이브러리 구현이 config와 다르게 동작할 가능성은 남아있다.

### 3-2. 해상도별 visual token `[계산]`

| 입력 | 픽셀 | visual token |
|---|---|---|
| 560 × 560 (min) | 313,600 | 306 |
| 768 × 768 | 589,824 | 576 |
| 1024 × 768 | 786,432 | 768 |
| 1024 × 1024 | 1,048,576 | 1,024 |
| 1536 × 1152 (4:3) | 1,769,472 | 1,728 |
| 1536 × 1536 | 2,359,296 | 2,304 |
| 2048 × 1536 (4:3) | 3,145,728 | 3,072 |
| 2048 × 2048 | 4,194,304 | 4,096 |
| 2800 × 2800 (max) | 7,840,000 | 7,656 |
| ~~4032 × 3024 (휴대폰 12MP)~~ `[정정]` | 12,192,768 | 11,907 → max 초과 시 강제 축소 (아래 참조) |

**정정 (2026-09-16, ZIP 중앙 디렉터리 실측).** "원본이 12MP 내외"라는 전제 자체가 근거 없이 가정된 것이었다. `ssafy-16-2-ai.zip` 중앙 디렉터리(암호 없이 읽을 수 있는 메타데이터)를 실측하니 JPEG 평균 크기가 약 111KB(train med 107KB, dev med 112KB, test med 107KB, p25~p75가 83~134KB로 좁게 분포)였다. 무압축 12MP 원본은 보통 3~5MB이므로, 배포된 파일은 그 1/30~1/45 수준이다 — **주최측이 배포 전 이미 리사이즈했을 가능성이 높다.** 단, 파일 크기만으로 정확한 픽셀 해상도를 역산할 수는 없으므로(JPEG 압축률은 장면 복잡도·품질 설정에 따라 다름) 정확한 해상도는 암호 해제 후 재확인한다. 출처: `tools/zip_stats.py` 실행 결과, `tools/zip_stats.txt`.

**용량 계획 시 주의 — 파일 크기 최댓값을 해상도 최댓값으로 오해하지 않는다.** 전체 이미지 중 최대 파일 크기는 test 세트의 441,728 bytes(약 432KB, p90의 약 2.7배)다. VRAM·토큰 예산은 **평균이 아니라 이 꼬리값 기준으로 스트레스 테스트해야 한다**는 점은 맞다. 다만 p25~p75가 좁게 분포한다는 것은 대부분 이미지가 비슷한 해상도로 일괄 전처리됐을 가능성을 시사하며, 이 경우 432KB 이미지는 "해상도가 가장 큰 이미지"가 아니라 **장면이 복잡하거나 텍스트가 많아 JPEG 압축 후 크기가 커진 이미지**일 가능성이 높다(같은 해상도라도 디테일이 많으면 압축 후 파일이 커진다). 즉 이 이미지는 해상도 스트레스 테스트보다 **과제 난이도(dense scene text) 스트레스 테스트**로서의 의미가 더 크다. 실제 픽셀 해상도의 최댓값은 암호 해제 후 별도로 확인해야 한다.

**적용 — 7절 실측표는 단일 지점이 아니라 분포 전 구간에서 채운다.** median 하나만으로 VRAM·시간을 추정하면 꼬리 케이스에서 OOM이 날 수 있다. 최소 min / p50(median) / p90 / max 4개 지점에서 VRAM·처리시간을 실측하고, max 지점(약 432KB, `test` 세트)은 우선순위 1로 먼저 확인한다.

**crop 전략의 근거 재구성.** `max_pixels` 자동 축소 논리 자체(패치 기반 인코더가 총 픽셀 수를 제한한다는 사실)는 여전히 유효하지만, "12MP 원본이 2800×2800으로 잘려 작은 글씨가 뭉개진다"는 시나리오는 전제가 흔들렸다. 대신 crop의 진짜 근거는 **토큰 배분**이다 — 업스케일은 픽셀에 없던 정보를 만들어내지 못하지만, 작은 글씨 영역만 crop해서 확대하면 인코더가 그 영역에 쓰는 visual token 수가 늘어나 표현 용량이 커진다. 예: 1000×750 이미지에서 40×15px 글씨는 32×32 패치 기준 1토큰 남짓이지만, 그 영역만 4배로 crop하면 약 10토큰이 된다.

**전처리 금지 항목 — 전체 이미지 업스케일 및 AI super-resolution.**
- **단순 업스케일(bicubic 등)**: 픽셀 보간일 뿐 정보를 만들지 못하고, 해상도가 커지는 만큼 visual token만 늘어나 계산량이 낭비된다. 이미 CLAUDE.md 「이미지 처리 가설」에 "단순 4K/8K 업스케일 후 큰 모델 입력은 피한다"로 명시돼 있다.
- **AI 기반 super-resolution(GAN/diffusion 계열 생성형 업스케일)은 사용하지 않는다.** 이런 모델은 실제로 없던 고주파 디테일을 그럴듯하게 "만들어낸다(hallucinate)". 일반 사진에서는 허용될 수 있어도, 정답이 텍스트 자체인 scene text 과제에서는 **흐려서 읽을 수 없던 글자가 "선명하지만 틀린 글자"로 둔갑할 위험**이 있다(예: 흐린 "8"이 "3"으로 복원). 원래는 모델도 사람도 "판독 불가"로 처리했을 샘플이 확신에 찬 오답으로 바뀌는 것이 대량 검증도 어렵다. 이미지 전처리 파이프라인에 생성형 업스케일 단계를 넣지 않는다.
- 허용되는 방향은 **crop(원본 픽셀 그대로, 영역만 잘라 리사이즈)**과 **비생성형 대비/명암 보정** 정도이며, 이것도 정보 복원이 아니라 토큰 배분·가독성 보조로만 취급한다.

### 3-3. Crop 조합별 토큰 `[계산]`

각 view가 독립적으로 토큰을 만든다고 가정한다.

| 구성 | 계산 | 총 visual token |
|---|---|---|
| Full 1024 | 1,024 | 1,024 |
| Full 2048 | 4,096 | 4,096 |
| Full 1024 + 4 crops @1024 | 1,024 × 5 | 5,120 |
| Full 1024 + 9 crops @1024 | 1,024 × 10 | 10,240 |
| Full 1536 + 4 crops @1024 | 2,304 + 4,096 | 6,400 |
| Full 2048 + 9 crops @1024 | 4,096 + 9,216 | 13,312 |

**262K context 대비 여유가 크다.** 13,312 토큰은 native context의 5%다. 컨텍스트가 병목이 아니라 **VRAM과 속도가 병목**이다.

---

## 4. KV cache — 이 모델의 결정적 장점 `[계산]`

Gated Attention만 KV를 남기고, 40층 중 10층이다.

```
토큰당 KV = 2(K,V) × 2(KV head) × 256(head dim) × 2 bytes(BF16) = 2,048 bytes = 2 KB / layer
× full attention 10층 = 약 20 KB / token
```

| 총 토큰 | KV cache |
|---|---|
| 1,024 | 약 20 MB |
| 5,120 | 약 100 MB |
| 10,240 | 약 200 MB |
| 100,000 | 약 2 GB |

나머지 30층의 Gated DeltaNet은 **시퀀스 길이와 무관한 고정 크기 상태**만 유지한다.

> **결론.** 고해상도·다중 crop을 넣어도 KV cache는 사실상 무시할 수 있다. 메모리를 지배하는 것은 **가중치와 activation**이다. 이 점은 이번 대회 입력 전략에 직접 유리하게 작용한다.

---

## 5. 추론 VRAM `[추정 — 3.5-35B-A3B 기준 초기 planning estimate를 재사용]`

Qwen3.6-35B-A3B는 3.5-35B-A3B와 아키텍처·파라미터 구성이 동일하므로 아래 수치를 **초기 계획값으로 재사용**한다. 다만 이는 계산 근사치이지 실측이 아니므로, **대회 시작 후 첫 smoke test에서 실제 VRAM을 찍어 확정한다.** 가중치 = 35B × (dtype bytes). 양자화 scale·zero point 오버헤드 약 5~10% 반영.

| 정밀도 | 가중치 | + vision encoder·런타임 | 합계(대략) | 들어가는 카드 |
|---|---|---|---|---|
| BF16 | 70 GB | +3~5 GB | **약 73~75 GB** | 180GB·288GB급 ×1 여유 · H100 80GB 빠듯 |
| FP8 | 35 GB | +3~5 GB | **약 38~40 GB** | H100/A100 80GB 여유 · 48GB 가능 |
| INT8 | 35 GB | +3~5 GB | **약 38~40 GB** | 동일 |
| NF4 (4bit) | 17.5 GB → 실효 약 19 GB | +3~5 GB | **약 22~24 GB** | 24GB 카드 빠듯 · 32GB 이상 여유 |

- **RTX 5060 Ti 16GB에는 어떤 정밀도로도 들어가지 않는다.** 이 모델은 H100급 이상 또는 클라우드 전용이다.
- 위 표에 batch 증가분과 activation은 포함되어 있지 않다.

**2026-09-16 갱신 — vision encoder 파라미터 계산 `[계산]`.** config.json `vision_config`(hidden 1152, layer 27, intermediate 4304, head 16)로 직접 추정하면 vision encoder는 약 4억(0.4B) 파라미터다.

```
per-layer ≈ attention(4 × 1152²) + MLP(2 × 1152 × 4304) ≈ 5.3M + 9.9M ≈ 15.2M
× 27 layers ≈ 411M + patch embed·merger projection(수백만) ≈ 약 420M
```

BF16 기준 약 0.8~0.9GB에 불과하다 — 즉 위 표의 "+3~5GB"는 vision encoder 가중치 자체가 아니라 대부분 **런타임 오버헤드(activation, CUDA 컨텍스트, KV cache 예비분)**다. 이 구분은 VRAM이 빠듯한 구성(NF4, 24GB 카드)에서 실측치가 예상보다 튀는 원인을 진단할 때 유용하다.

---

## 6. 학습 VRAM — PEFT vs FFT `[추정 — 3.5-35B-A3B 기준 초기 planning estimate를 재사용, smoke test로 확정]`

### 6-1. FFT `[계산]`

mixed-precision AdamW 학습 상태를 파라미터당 16~18 bytes로 근사한다.

```
BF16 weight 2 + BF16 grad 2 + FP32 master 4 + Adam m 4 + Adam v 4 = 16 bytes
버퍼 포함 18 bytes로 잡으면
35B × 18 bytes ≈ 630 GB + activation
```

| 구성 (288GB 카드 기준) | 판단 |
|---|---|
| ×3 | 최소선 |
| ×4 | 실질 도전 구성 |

**active 3B로 줄지 않는다.** optimizer state는 라우팅되지 않은 expert를 포함한 **total 35B 전체**에 대해 유지해야 한다. 이것이 MoE FFT의 함정이다.

### 6-2. PEFT `[추정]`

LoRA/DoRA 학습 파라미터를 전체의 0.1~0.5%로 잡으면 35M~175M이다. AdamW 상태 16 bytes/param → **0.6~2.8 GB**로 작다. 실제 메모리는 **base 가중치 + activation**이 지배한다.

| 방식 | base | adapter + optimizer | activation(grad ckpt ON, mb=1, ~4k tok) | 합계(대략) |
|---|---|---|---|---|
| NF4 QLoRA / QDoRA | 19 GB | 1~3 GB | 5~12 GB | **약 25~35 GB** |
| FP8 + adapter | 35 GB | 1~3 GB | 5~12 GB | **약 41~50 GB** |
| BF16 LoRA / DoRA | 70 GB | 1~3 GB | 5~12 GB | **약 76~85 GB** |

> FP8 frozen base + PEFT가 실제 학습 스택에서 안정적으로 지원되는지는 **별도 smoke test가 필요하다** `[미확인]`.

---

## 7. resolution × batch × VRAM 표 (실측용)

**아래는 대회 당일 실측으로 채운다.** 지금은 visual token만 계산으로 채워 두었다. **해상도 열은 이미지 크기 분포의 min / p50(median) / p90 / max 4개 지점을 기준으로 실측한다** — median 하나만 재면 꼬리 케이스(대회 데이터의 max ≈ 432KB, 3-2절 참조)에서 OOM이 날 수 있다. max 지점은 우선순위 1로 먼저 확인한다.

### 7-1. NF4 QLoRA 학습 (microbatch 1, grad checkpointing ON)

| Resolution / View | visual token | Batch | Peak VRAM | sec/step | 비고 |
|---|---|---|---|---|---|
| Full 1024 | 1,024 | 1 | ? | ? | |
| Full 1536 | 2,304 | 1 | ? | ? | |
| Full 2048 | 4,096 | 1 | ? | ? | |
| Full 1024 + 4 crops | 5,120 | 1 | ? | ? | |
| Full 1024 + 9 crops | 10,240 | 1 | ? | ? | |
| Full 2048 + 9 crops | 13,312 | 1 | ? | ? | |

### 7-2. NF4 추론 (batch sweep)

| Resolution | visual token | Batch 1 | 2 | 4 | 8 | sec/image |
|---|---|---|---|---|---|---|
| Full 1024 | 1,024 | ? | ? | ? | ? | ? |
| Full 1536 | 2,304 | ? | ? | ? | ? | ? |
| Full 2048 | 4,096 | ? | ? | ? | ? | ? |
| Full + 4 crops | 5,120 | ? | ? | ? | ? | ? |
| Full + 9 crops | 10,240 | ? | ? | ? | ? | ? |

### 7-3. 반드시 함께 기록할 값

`model_path` · `dtype` · `load_in_4bit` · `bnb_4bit_quant_type` · `입력 해상도` · **`processor가 실제로 만든 visual_tokens`** · `microbatch` · `grad_accum` · `grad_checkpointing` · `model load 직후 VRAM` · `forward peak` · `backward peak` · `sec/step` 또는 `sec/image`

---

## 8. 학습 설정 후보

| 항목 | 후보 | 비고 |
|---|---|---|
| 양자화 | NF4, Double Quantization ON, BF16 compute | |
| PEFT | QLoRA r=8 / QDoRA r=8 / QLoRA r=16 / QDoRA r=16 | 4개 비교. **3.5에서 나온 최적값이 아니라 초기 후보다** — 3.6-35B-A3B에서 다시 탐색한다 |
| Target module | 일반 Linear projection부터 | expert weight는 fused 3D 텐서로 **확인됨**(2026-09-16). vLLM에 known LoRA 로딩 버그(#38520, 미해결) 있어 우리 스택 재현 여부를 smoke test에서 먼저 확인 |
| LR | 1e-5 ~ 5e-5 | **초기 후보.** 모델 크기가 바뀌면 물론이고, 3.5→3.6처럼 크기가 같아도 모델이 바뀌면 재탐색 대상이다 |
| microbatch | 1 | 해상도가 크면 1 고정 |
| grad accumulation | 8 ~ 32 | effective batch로 조정 |
| gradient checkpointing | ON | 고해상도에서 필수 |
| optimizer | AdamW (또는 paged AdamW 8bit) | 메모리 여유 없으면 8bit |
| loss masking | 질문·선택지 토큰 마스킹, **정답 토큰에만 loss** | |
| checkpoint 선택 | validation **Accuracy** 기준 | loss 기준 아님 |

### MoE PEFT smoke test 순서 (3.6-35B-A3B에서 반드시 다시 통과 — 3.5에서 됐다고 자동 성립 아님)

```
1. expert weight가 fused인지 확인 → target module 결정
2. 일반 Linear projection만으로 몇 step 학습
3. adapter 저장
4. 프로세스 재시작
5. 재로드 → 추론
6. merge / adapter load 양쪽 검증
```

3.6-35B-A3B가 3.5와 파라미터 구성이 같더라도 실제 구현·체크포인트 포맷이 같은 방식으로 저장/로드되는지는 별개 문제다. **이 6단계를 3.6에서 처음부터 다시 통과시킨 뒤에야** 122B로 확대한다.

**2026-09-16 갱신 — 1번 항목 답 확정 + 알려진 버그 발견 `[확인]`.** expert weight는 **fused다** — HF 구현에서 라우팅되는 expert 가중치가 `model.layers.{L}.mlp.experts.gate_up_proj [n_experts, 2×moe_inter, hidden]`, `...experts.down_proj [n_experts, hidden, moe_inter]` 형태의 **3D 텐서 하나**로 저장된다(expert별 독립 `nn.Linear`가 아니다). 따라서 target module 선정 시 개별 `gate_proj`/`up_proj`/`down_proj` 대신 이 fused 텐서를 대상으로 해야 한다.

**위험 신호.** 관련 이슈를 조사하다 Qwen3.5-MoE 계열(3.6도 같은 구조 공유)에서 **"LoRA loading fails ... due to expert module name mismatch"** 버그를 발견했다([vLLM #38520](https://github.com/vllm-project/vllm/issues/38520)) — 이 이슈는 **fix 없이 "stale"로 closed(not planned)** 상태다. 원인은 `language_model.model.layers.0.mlp.experts.0.down_proj` 같은 경로에서 expert 인덱스를 포함한 전체 경로를 모듈명으로 잘못 파싱하는 것이다. 별도로 unsloth 진영에서도 유사한 shape mismatch 버그가 보고됐다([unsloth-zoo #601](https://github.com/woct0rdho/transformers-qwen3-moe-fused)). **이게 실제로 우리 학습 스택(transformers/peft 버전)에서도 재현되는지가 smoke test 1번 항목의 핵심 목적이다** — 재현되면 expert 대상 LoRA는 포기하고 attention/MLP의 non-expert Linear만 타겟팅하는 쪽으로 축소해야 한다.

---

### 8-2. 고정할 설정 vs 변동시킬 설정

실험을 서로 비교 가능하게 유지하려면 **무엇을 고정할지 먼저 정해야 한다.** 고정 항목이 흔들리면 어떤 실험 결과도 원인을 특정할 수 없다.

#### 고정 (한 번 정하면 대회 내내 바꾸지 않음)

| 항목 | 값 | 이유 |
|---|---|---|
| seed | 1개 고정 (재현성 실험 때만 변경) | 결과 차이가 설정 때문인지 운 때문인지 분리 |
| validation split | Group Split 고정, 파일로 저장 | 매번 새로 나누면 비교 불가 |
| prompt template | 모든 모델 동일 | 모델 비교의 전제 |
| decoding | `temperature=0`, greedy | 같은 입력에 같은 출력 |
| `max_new_tokens` | 짧게 고정 | 답이 A/B/C/D 한 글자 |
| 종횡비 처리 | 원본 비율 유지, pad/letterbox 정책 고정 | 왜곡이 OCR 성능에 직접 영향 |
| compute dtype | BF16 | |
| 양자화 방식 | NF4 + Double Quantization ON | 주력 구성 |
| gradient checkpointing | ON | 고해상도에서 필수 |
| loss masking | 질문·선택지 마스킹, 정답 토큰에만 loss | |
| checkpoint 선택 기준 | validation **Accuracy** | loss 기준 금지 |
| 라이브러리 버전 | transformers / accelerate / bitsandbytes / peft 고정 | 버전이 바뀌면 processor 동작이 달라질 수 있음 |
| processor `use_fast` | 명시적으로 고정 | fast/slow processor 출력이 다를 수 있음 |
| 로그 스키마 | 7-3절 항목 | 나중에 표로 합칠 수 있어야 함 |

#### 변동 (실험 축 — 한 번에 하나씩만 바꾼다)

| 축 | 후보 | 우선순위 |
|---|---|---|
| **입력 view** | Full / High-res Full / Full+Crop / Full+OCR / Full+Crop+OCR | **1순위** |
| **해상도** | 1024 / 1536 / 2048 / 원본(max_pixels 상한) | **1순위** |
| **crop 개수** | 0 / 4 (2×2) / 9 (3×3) / 질문 기반 선택 crop | **1순위** |
| PEFT 방식 | LoRA / DoRA | 2순위 |
| rank | r=8 / 16 / 32 | 2순위 |
| Learning Rate | 1e-5 / 2e-5 / 5e-5 | 2순위 |
| effective batch | grad accum 8 / 16 / 32 | 3순위 |
| epoch / early stopping | 1 / 2 / 3, accuracy 기준 정지 | 3순위 |
| 양자화 | NF4 / FP8 / BF16 | 3순위 (메모리 여유 따라) |
| target module | 일반 Linear만 / attention+MLP / expert 포함 | 3순위 |
| prompt 형식 | 답만 출력 / 짧은 근거 후 답 | 3순위 |
| 선택지 순서 | 고정 / shuffle | 위치 편향 측정용 별도 실험 |

**원칙.** 1순위 세 축(view·해상도·crop)이 이번 과제의 핵심 가설이다. 여기서 이득이 확인되기 전에 rank나 LR을 돌리는 것은 순서가 틀렸다. 그리고 **한 실험에서 두 축을 동시에 바꾸지 않는다.**

#### 재검증이 필요한 것

모델이 바뀌면 **수치형 하이퍼파라미터는 그대로 옮기면 안 된다.** 크기가 다를 때(9B→35B)는 물론이고, **크기가 같아도 세대가 바뀌면(3.5-35B-A3B→3.6-35B-A3B) 마찬가지다.** LR, rank, effective batch, epoch이 9B에서 좋았다고 35B에서 좋다는 보장이 없듯, 3.5-35B-A3B에서 좋았다고 3.6-35B-A3B에서도 좋다는 보장이 없다. 반면 **방향성 결론**(고해상도가 유리한가, crop이 유리한가, OCR 보조가 유리한가, 짧은 근거 출력이 유리한가)은 전이될 가능성이 높다. 작은 모델은 **최적값을 대신 찾는 용도가 아니라 검증할 가설을 싸게 추리는 용도**로 쓴다.

#### 8-3. 추가 보완 설정 (2026-09-16, 팀 요청 — "고정/변동으로 둘 수 있는 설정 더 찾기")

기존 8-2절 고정 14항목·변동 12축 외에, 실무에서 자주 결과를 흔드는데 놓치기 쉬운 항목들이다.

**추가 고정 후보**

| 항목 | 권장 | 이유 |
|---|---|---|
| attention 구현 | `flash_attention_2` (지원 시) 고정 | `sdpa`/`eager`와 수치가 미세하게 달라질 수 있음. 카드·라이브러리 버전에 따라 미지원일 수 있어 먼저 확인 |
| 이미지 보간(interpolation) 방식 | bicubic 등 하나로 고정 | resize 시 보간법이 다르면 같은 해상도라도 픽셀 값이 달라짐 |
| 정규화 통계(mean/std) | processor 기본값 고정, 임의 변경 금지 | 사전학습 때 쓴 값과 다르면 분포가 어긋남 |
| tokenizer padding side | left/right 중 하나로 고정 | 생성형 모델은 보통 left, 버전마다 기본값이 다를 수 있음 |
| weight decay | 0.0 또는 0.01 중 하나로 고정 | LoRA는 보통 decay 약하게 |
| warmup ratio/step | 고정 (예: 3~5%) | 짧은 학습에서는 warmup 비율이 최종 성능에 크게 작용 |
| dropout | base 모델 기본값 유지 | 소규모 데이터+LoRA에서 임의로 올리면 과소적합 위험 |
| 평가 주기·기준 | step 또는 epoch 하나로 고정, 매 실험 동일 | 비교 가능성을 위해 필수 |
| 데이터 shuffle seed | 학습 seed와 별도로 고정 여부 결정 | 같은 seed로 고정하면 데이터 순서까지 통제 가능 |

**추가 변동 후보 (우선순위는 팀 판단, 1~3순위 축과 겹치지 않게)**

| 축 | 후보 | 비고 |
|---|---|---|
| optimizer 종류 | AdamW / paged AdamW 8bit / Lion | 메모리·속도 트레이드오프, 3순위 |
| gradient clipping | 없음 / 1.0 / 0.5 | 불안정한 loss 대응, 3순위 |
| label smoothing | 0 / 0.05 / 0.1 | 객관식 답 분포가 편향될 때 시도, 3순위 |
| EMA(지수이동평균) 적용 여부 | ON/OFF | 짧은 학습에서는 효과 불확실, 3순위 |
| 선택지 텍스트 증강 | 원문 유지 / 동의어 치환 | OCR 오탈자 강건성 테스트용, 여유 시 |
| crop 경계 처리 | 정확히 자르기 / 여백 포함 | 텍스트가 crop 경계에 걸릴 때 영향, 1순위 crop 축과 함께 고려 |
| few-shot in-context 예시 개수 | 0 / 1 / 2 | zero-shot 대비 추론 시간·정확도 트레이드오프, 여유 시 |

이 항목들은 1순위(view·해상도·crop) 검증이 끝난 뒤, 2~3순위 축과 같은 층위에서 필요할 때만 추가한다. 전부 한꺼번에 벌이면 실험이 조합 폭발로 끝나지 않는다.

---

## 9. 속도와 1 epoch 예상 `[미확인]`

`sec/image`와 `sec/step`은 실측 전이라 비워 둔다. 계산식만 고정한다.

```
1 epoch 예상시간 = (sec/step) × ceil(학습행수 / effective_batch) × 1.15
```

- 학습 이미지 6,714장. **CSV 행 수는 암호 해제 전까지 미확인**이므로 이미지 수를 행 수로 쓰지 않는다.
- 여유 계수 1.15는 검증·저장 시간을 포함한 근사다.
- active 3B라 **토큰당 연산량은 3B 모델 수준**이다. 메모리는 35B, 속도는 3B에 가깝다는 것이 이 모델의 성격이다.

---

## 10. 결론 — 이 모델을 어디에 쓸 것인가

**포지션: 주력 후보 + scale-up 분기점.**

1. **9B에서 찾은 방향이 실제로 전이되는지 확인하는 지점.** 9B → 3.6-35B 이득(RealWorldQA +5.0)이 3.6-35B → 122B 이득(−0.2, 사실상 없음)보다 훨씬 크다.
2. **FFT 최종 한 방 후보.** 9B에서 FFT 우위가 확인되면 여기서 short pilot 후 288GB급 ×3~4로 한 번 시도한다. 122B FFT는 약 2.2TB, 397B FFT는 7TB 이상이라 사실상 불가능하므로 **FFT를 시도해 볼 수 있는 가장 큰 모델**이 35B-A3B다.
3. **122B 투입 여부를 결정하는 기준선.** 3.6-35B가 122B에 이미 근접해 있어(2-1절), 예산을 122B 학습보다 crop·OCR·앙상블에 먼저 쓰는 쪽에 무게가 실린다 — 단 이는 일반 실사진 QA 지표 기준이며, 우리 과제(다국어 tiny scene text)에서 재확인이 필요하다.

### 대회 시작 후 가장 먼저 돌릴 설정 3개

| # | 설정 | 목적 |
|---|---|---|
| 1 | Qwen3.6-35B-A3B, NF4, Full 1024, batch 1, zero-shot, 300~500 대표 subset | 기준점 확보 + visual token 가정 검증(processor 실제 토큰 수 확인) |
| 2 | NF4, Full 2048 / Full+4crops 비교, 동일 subset | 해상도·crop 이득이 실제로 있는지 |
| 3 | NF4 QLoRA r8 smoke test (저장 → 재로드 → 추론 → merge) | MoE PEFT가 **3.6에서** 도는지 확인 (3.5에서 됐어도 재확인 필요) |

1·2번 결과를 9B·122B 담당자의 같은 subset 결과와 맞대면 **scale-up 이득 곡선**이 그려진다. 그 곡선이 122B 투입 여부를 결정한다. 여유가 있으면 같은 subset에서 3.5-35B-A3B도 함께 돌려 부록 A의 벤치마크 차이가 우리 데이터에서도 재현되는지 확인한다.

---

## 11. 미확인 목록

1. ~~visual token 계산 규칙~~ → **2026-09-16 해소.** config.json에서 `spatial_merge_size: 2` 확인, 32×32 가정이 공식 값임을 확인. 단 실제 processor 출력은 여전히 대회 당일 확인한다.
2. ~~3.6-35B-A3B의 CC-OCR·CharXiv 점수~~ → **2026-09-16 해소.** CC-OCR 81.9, CharXiv 78.0 확인(122B의 81.8/77.2와 사실상 동급 또는 근소 우위). **OCRBench만 여전히 미확인.**
3. ~~vision encoder 파라미터 수~~ → **2026-09-16 해소(계산).** config.json 구조로 약 420M(0.4B)로 계산. BF16 기준 <1GB — VRAM 표의 "+3~5GB"는 대부분 런타임 오버헤드임을 확인.
4. ~~expert weight가 fused parameter인지~~ → **2026-09-16 해소.** fused 3D 텐서로 확인. 단 이로 인한 **LoRA 로딩 버그(vLLM #38520, 미해결)** 가 우리 스택에서도 재현되는지가 새로운 미확인 항목이다.
5. **FP8 frozen base + PEFT 지원 여부** — 2026-09-17 부분 조사. bitsandbytes는 NF4(4bit)·INT8만 지원하고 **네이티브 FP8은 지원하지 않는다**(PyTorch 자체도 FP8 텐서 타입 지원이 제한적). `Qwen/Qwen3.6-35B-A3B-FP8` 같은 공식 FP8 체크포인트는 존재하지만 이는 주로 vLLM/SGLang 추론용이며, 이 위에서 PEFT 학습을 하려면 bitsandbytes 표준 경로가 아니라 torchao나 TransformerEngine 같은 별도 라이브러리가 필요하다 — NF4 QLoRA만큼 검증된 경로가 아니다. **결론: FP8 학습 경로는 후순위로 두고 NF4를 주력으로 유지한다.** PEFT 라이브러리의 정확한 지원 버전은 여전히 확인이 필요하다.
6. **한국어 scene text 성능** — 공개 벤치마크 어디에도 없다. 우리 데이터로만 잴 수 있다.
7. **3.6-35B-A3B의 OCRBench 점수** — 2026-09-17 재조회(HF 모델 카드, 공식 블로그, 서드파티 벤치마크 사이트, HF discussion 스레드)에서도 여전히 공개된 곳을 찾지 못했다. 공식 블로그는 SWE-bench·Terminal-Bench 등 코딩·에이전트 벤치마크와 일반 VQA만 강조하고 OCRBench는 언급하지 않는다 — **의도적으로 비공개했을 가능성**도 있다(OCR 특화 지표를 덜 강조하는 코딩 중심 모델 포지셔닝과 일치). 대회 데이터로 직접 측정하는 것 외에는 확보 경로가 없다고 결론.
8. **MoE LoRA target module 버그**(vLLM #38520) — 우리가 쓸 transformers/peft 버전에서 재현되는지 smoke test 1번 항목에서 확인 필요. 웹 조사로는 더 진행할 수 없고 실제 환경에서 확인해야 한다.

---

## 부록 A. Qwen3.5-35B-A3B — 교체 전 baseline 기록

이 문서는 원래 **Qwen3.5-35B-A3B** 조사로 시작했다. 같은 35B-A3B 슬롯에 **Qwen3.6-35B-A3B**가 존재하고, 아키텍처·파라미터·context·라이선스가 3.5와 동일하면서 비전 벤치마크가 대체로 더 높아 2026-09-16 팀 결정으로 **3.6으로 교체**했다(위 본문 전체가 3.6 기준). 이 부록은 교체 근거였던 비교 수치를 기록으로 남긴다 — **운용 수치(VRAM, 토큰, batch)는 아키텍처가 같으므로 본문에서 그대로 재사용했고**, 달라진 것은 성능 수치뿐이었다.

| 항목 | 3.5-35B-A3B | 3.6-35B-A3B | 변화 |
|---|---|---|---|
| RealWorldQA | 84.1 | **85.3** | +1.2 |
| MMBench EN-DEV | 91.5 | **92.8** | +1.3 |
| OmniDocBench1.5 | 89.3 | **89.9** | +0.6 |
| OCRBench | 91.0 | `[미확인]` | — |
| CC-OCR | `[미확인]` | `[미확인]` | — |
| MMLU-Pro | 85.3 | 85.2 | −0.1 |
| SWE-bench Verified | 70.0 | 73.4 | +3.4 |
| Terminal-Bench 2.0 | 40.5 | **51.5** | **+11.0** |

**해석.** 3.6의 주된 개선은 **에이전트·코딩**이다. 비전 쪽 이득은 +0.6 ~ +1.3으로 완만하다. 다만 같은 크기·같은 비용에서 손해 보는 항목이 사실상 없으므로 **교체 자체는 타당하다.**

**한 가지 더.** 3.6-35B의 RealWorldQA 85.3 / MMBench 92.8 / OmniDocBench 89.9는 **Qwen3.5-122B-A10B의 85.1 / 92.8 / 89.8과 사실상 동일하다.** 파라미터는 1/3.5, 토큰당 활성은 3B vs 10B다. 이 수치가 우리 데이터에서도 재현되면 122B 투입의 근거가 크게 약해진다.

단 3.6-35B의 **OCRBench와 CC-OCR이 공개되지 않아** 이 비교는 아직 일반 실사진 QA 지표에 한정된다. 122B의 OCRBench 우위(92.1 vs 3.5-35B의 91.0)가 3.6에서도 남아 있는지는 확인할 수 없다.

**권고.** 교체는 확정됐지만, 대회 시작 후 같은 subset에서 `3.5-35B vs 3.6-35B`를 한 번 실측 비교해 우리 데이터에서도 이 판단이 맞는지 재검증한다. 아키텍처가 같아 실험 코드 변경 없이 모델 경로만 바꾸면 되므로 비용이 거의 들지 않는다.

---


## 부록 B. Qwen3-VL-8B 추가 후보 검토

팀 논의에서 "3.5 VL 8B"가 언급되었다. **한 가지 정정이 필요하다 — Qwen3.5에는 별도의 VL 브랜치가 없다.** Qwen3.5 이후 계열은 본체가 네이티브 멀티모달(early fusion)이라 `Qwen3.5-9B` 자체가 이미 비전 모델이다.

별도 VL 브랜치는 이전 세대인 **Qwen3-VL** 계열(2B / 4B / 8B / 32B)이다. 따라서 "3.5 VL 8B"가 가리키는 것은 실질적으로 둘 중 하나다.

| 후보 | 성격 |
|---|---|
| `Qwen3.5-9B` | 네이티브 멀티모달. 현호·건순 담당 슬롯과 동일 |
| `Qwen3-VL-8B` | 비전 전용으로 따로 학습된 이전 세대 브랜치 |

**Qwen3-VL 계열을 남겨둘 가치는 있다.** 1차 챌린지 Public 0.92116이 `Qwen3-VL-4B-Instruct`에서 나왔으므로, 이 계열은 **검증된 기준점**이다. 기준점이 없으면 "세대를 올려서 얼마나 올랐는지"를 말할 수 없다.

다만 **scale-only 비교로 해석하면 안 된다.** 학습 데이터와 아키텍처 세대가 다르므로 `Qwen3-VL-8B` vs `Qwen3.5-9B`의 차이는 크기 차이가 아니라 세대·설계 차이다. 독립 비교군으로만 둔다.

**권고.** 로컬 5060 Ti 탐색이 끝난 뒤 추가 후보로 넣는다는 팀 방향에 동의한다. 우선순위는 입력 전략(view·해상도·crop) 검증 이후다.

## 출처

- [Qwen3.6-35B-A3B 모델 카드](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (채택 모델)
- [이미지 해상도·토큰 논의](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/discussions/36)
- [Qwen3.5-35B-A3B 모델 카드](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) (부록 A, baseline)
- [Qwen3.5-122B-A10B 모델 카드](https://huggingface.co/Qwen/Qwen3.5-122B-A10B)
- [Qwen3.5-9B 모델 카드](https://huggingface.co/Qwen/Qwen3.5-9B)
- [Qwen3.5 아키텍처 문서 (NVIDIA Megatron-Bridge)](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/qwen/qwen35-vl.html) (3.6과 아키텍처 동일 가정의 근거)
- [Qwen3-VL 저장소](https://github.com/QwenLM/Qwen3-VL) (부록 B)
- [Qwen3.6-35B-A3B config.json](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json) (2026-09-16, spatial_merge_size·vision_config·CC-OCR·CharXiv 확인)
- [vLLM Issue #38520 — Qwen3.5 MoE LoRA 로딩 버그](https://github.com/vllm-project/vllm/issues/38520) (2026-09-16, closed as stale, 미해결)
- [transformers-qwen3-moe-fused](https://github.com/woct0rdho/transformers-qwen3-moe-fused) (2026-09-16, fused MoE 구조 관련 커뮤니티 도구)
