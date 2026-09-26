# dev 재라벨링 가이드 초안

이 문서는 팀원이 함께 수정할 초안이다. 두 작업자는 본 작업 전에 `calibration` 50건을 각각
독립적으로 처리하고, 불일치 원인을 합의해 이 문서를 확정한다.

## 입력 경계

- 작업 패킷의 이미지, 질문, 선택지 a~d만 사용한다.
- 원본 `dev.csv`의 `answer1`~`answer5`는 열람하거나 힌트로 사용하지 않는다.
- 이미지에 보이지 않거나 질문에서 요구하지 않은 정보를 추측하지 않는다.
- 선택지 점수만 낸 로컬 VLM 출력은 초안이다. 최종 `approved`는 지정된 검수자가 원본 이미지를
  직접 확인한 뒤 기록한다.

## 기록 열

| 열 | 허용값/작성법 |
|---|---|
| `human_answer` | `a`, `b`, `c`, `d` 중 하나. 판정 불가면 비운다. |
| `confidence` | `high`, `medium`, `low` |
| `readability` | `clear`, `partial`, `unreadable` |
| `quality_reason` | `none`, `blur`, `low_resolution`, `glare`, `occlusion`, `bad_crop`, `other` |
| `data_usage` | `vqa_train`, `vision_auxiliary_candidate`, `exclude` |
| `ambiguity_reason` | 복수 정답, 질문 오류, 이미지 불명확 등의 짧은 사유 |
| `evidence` | 이미지에서 확인한 글자·위치·대상의 짧은 근거 |
| `reviewer` | 팀 내 식별 가능한 작업자 이름 또는 약칭 |
| `llm_assistance` | `none` 또는 합의한 도구/로컬 모델 식별자 |
| `review_status` | `approved`, `needs_review`, `rejected` |
| `review_note` | 수정·재검토에 필요한 메모 |

## Codex-assisted visual review

2026-09-21 사용자 전달 승인에 따라 mine 배치는 Codex가 원본 이미지를 직접 대조해 검수할 수
있다. 이 경우 아래 provenance를 반드시 남긴다.

```text
reviewer = codex-assisted-visual
llm_assistance = Codex direct image review; local Qwen2.5 draft retained separately
```

`human_answer`는 기존 exporter와의 호환을 위한 레거시 열 이름이며, 위 reviewer가 기록된 행을
사람이 직접 라벨링한 것으로 표현하면 안 된다. 검수 시트에는 모델 후보와 dev의 교육생 응답을
표시하지 않는다. 먼저 이미지·질문·선지만 보고 판정한 다음 모델 간 불일치 여부를 사후 점검한다.
3B/9B 합의만으로 자동 승인하지 않는다.

## 판정 규칙

1. 질문이 요구하는 대상과 범위를 먼저 고정한다.
2. 각 선택지를 이미지에서 하나씩 대조한다.
3. 답이 하나이고 근거가 선명하면 `approved`로 기록한다.
4. 두 선택지가 모두 가능하거나 문맥 해석이 갈리면 `needs_review`로 둔다.
5. 이미지에서 답을 확인할 수 없거나 질문·선지가 잘못됐으면 `rejected`로 둔다.
6. `low` confidence 행을 억지로 승인하지 않는다.
7. LLM 제안과 사람 판단이 다르면 사람 판단을 우선하되 `review_note`에 차이를 남긴다.

## 기존 문항 재생성 규칙

기존 질문이나 선택지가 잘못됐지만 이미지 자체에는 명확한 정보가 있으면 행을 버리는 대신 별도
재생성 데이터로 옮길 수 있다. 원본 행을 직접 덮어쓰지 않는다.

1. 기존 질문의 대략적인 유형(OCR, 상호, 메뉴, 가격, 시간, 객체, 위치)은 참고할 수 있지만,
   새 정답은 이미지에서 직접 확인되는 정보만 사용한다.
2. 흐리거나 가려진 세부 글자를 억지로 묻지 않고, 같은 이미지에서 선명하게 보이는 상호·제목·
   메뉴·큰 숫자·객체·위치 관계로 질문을 낮춘다.
3. 질문은 정답 근거의 위치나 대상을 특정해 한 이미지에서 답이 하나만 되게 한다.
4. 오답 세 개는 정답과 같은 의미 범주로 작성하고, 이미지에서 함께 참이 되는 선택지는 넣지 않는다.
5. 정답 위치는 배치 전체에서 a~d가 균형에 가깝도록 섞는다. 위치 균형을 위해 정답이나 근거를
   바꾸지 않는다.
6. `source_id`, `image_id`, `path`를 보존한다. 같은 `image_id`에서 여러 문항을 만들 경우
   Group Split으로 모두 같은 split에 배치한다.
7. 새 문항도 `approved + clear + high/medium + vqa_train` 게이트를 통과해야 학습에 넣는다.
8. 모델 후보를 보고 질문을 맞춰 만들지 않는다. 먼저 이미지를 보고 문항을 만든 뒤 사후 QA만 한다.

## 이미지 품질 게이트

라벨을 고르기 전에 원본 크기로 이미지를 확대해 **정답 근거가 되는 영역**을 먼저 확인한다.

- `clear`: 정답 근거를 사람이 안정적으로 판독할 수 있다.
- `partial`: 일부는 보이지만 정답 판정이 확대·추측에 크게 의존한다.
- `unreadable`: 흐림, 저해상도, 반사, 가림, 잘못된 crop 때문에 정답을 확인할 수 없다.

VQA 학습 후보는 아래 조건을 모두 만족해야 한다.

```text
review_status = approved
readability = clear
confidence = high 또는 medium
human_answer = a/b/c/d
data_usage = vqa_train
```

`partial` 또는 `unreadable`은 `human_answer`를 비우고 `data_usage`를
`vision_auxiliary_candidate` 또는 `exclude`로 둔다. 보조비전 후보도 별도 목적과 라벨을
확정하기 전에는 어떤 학습에도 자동 투입하지 않는다.

## 공동 교정 절차

1. 두 작업자가 `calibration/labels.csv` 50건을 서로 결과를 보지 않고 완료한다.
2. `human_answer`, `review_status`, `confidence` 불일치를 비교한다.
3. 불일치와 흐림 판정 사례별로 판정 원칙을 이 문서에 추가한다.
4. 합의가 끝난 뒤에만 각자의 본 배치를 시작한다.
5. 본 배치의 `low`와 `needs_review`는 마지막에 함께 재검토한다.

## LLM 요청 형식 권장안

허용된 도구에 이미지·질문·선지 네 개만 제공하고 다음 형식으로 초안을 받는다.

```json
{
  "candidate_answer": "a",
  "confidence": "medium",
  "evidence": "이미지에서 확인한 짧은 근거",
  "ambiguity": "없음 또는 애매한 이유"
}
```

LLM의 설명이 그럴듯하다는 이유만으로 승인하지 말고, `evidence`가 실제 이미지와 일치하는지
작업자가 확인한다.
