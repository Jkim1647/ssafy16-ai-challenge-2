# dev 라벨링 동료 전달 문서

## 작업 범위

이 패킷은 팀 내부 dev 재라벨링 전용이다.

| 단계 | 파일 | 행 수 | 목적 |
|---|---|---:|---|
| 1 | `calibration/labels.csv` | 50 | 두 작업자의 판정 기준 맞추기 |
| 2 | `teammate/labels.csv` | 1,316 | 동료 독립 담당분 |
| 합계 |  | 1,366 | 동료가 확인할 전체 행 수 |

원본 dev 2,683건 중 나머지 1,317건은 다른 작업자가 담당한다. 두 본 작업분의 ID는 겹치지
않는다. test, train, KAIST 데이터는 이 패킷의 작업 범위가 아니다.

패킷에는 `answer1`~`answer5` 값이 없다. 별도로 원본 dev CSV를 열어 해당 응답을 확인하지 않는다.
이미지는 패킷에 포함되지 않는다. CSV의 `path` 열(`dev/dev_XXXX.jpg`)을 사용해 각자 보유한
공식 dev 이미지 폴더에서 파일을 연다.

## 작업 순서

### 1단계: calibration 50건

1. `calibration/labels.csv`를 연다.
2. 상대 작업자의 결과를 보지 않고 50건을 모두 판정한다.
3. 작성한 파일을 서로 비교한다.
4. `human_answer`, `readability`, `review_status` 불일치를 합의한다.
5. 합의된 사례를 `LABELING_GUIDE_DRAFT.md`에 추가한 뒤 본 작업을 시작한다.

### 2단계: 본 작업 1,316건

- 200~250건 단위로 나눠 작업한다.
- 각 묶음 종료 시 `low`, `needs_review`, `rejected` 개수를 기록한다.
- 약 300건을 완료한 시점에 상대 작업자와 애매 사례를 한 번 중간 점검한다.
- 마지막에는 `needs_review`와 이미지 품질 보류 건만 함께 재검토한다.

## 문항별 판정

이미지를 원본 크기로 확인하고 질문과 a~d 선택지만 사용한다.

### VQA 학습 후보로 승인

아래 조건을 모두 만족할 때만 승인한다.

```text
human_answer = a/b/c/d
confidence = high 또는 medium
readability = clear
quality_reason = none
data_usage = vqa_train
review_status = approved
```

`evidence`에는 정답 근거가 된 이미지 속 글자, 위치 또는 대상을 한 문장 이내로 적는다.

### 애매한 경우

다음 중 하나면 `review_status=needs_review`로 둔다.

- 두 선택지가 모두 답이 될 수 있음
- 질문이 가리키는 대상이 여러 개임
- 이미지 문구 일부만 읽힘
- LLM 제안과 사람 판단이 다름
- 외부 상식 없이는 답을 고를 수 없음

이때 억지로 `human_answer`를 확정하지 않는다.

### 흐림·저품질 이미지

- `readability`: `partial` 또는 `unreadable`
- `quality_reason`: `blur`, `low_resolution`, `glare`, `occlusion`, `bad_crop`, `other`
- `data_usage`: `vision_auxiliary_candidate` 또는 `exclude`
- `review_status`: `rejected` 또는 `needs_review`
- `human_answer`: 비워 둠

`vision_auxiliary_candidate`도 별도 목적과 라벨이 확정되기 전에는 학습에 사용하지 않는다.

## LLM 사용 규칙

- 주최 측과 팀이 허용한 도구 범위에서 후보 답을 받을 수 있다.
- LLM에는 패킷의 이미지·질문·a~d만 제공한다.
- LLM이 제시한 답은 초안이며 작업자가 이미지를 직접 대조한다.
- `llm_assistance`에 사용한 도구 또는 로컬 모델 식별자를 기록한다.
- LLM 설명에 이미지에서 보이지 않는 사실이 포함되면 근거로 사용하지 않는다.
- 여러 번 질문해 다수결로 정답을 만들지 않는다. 사람의 단일 최종 판정을 남긴다.

권장 출력 형식:

```json
{
  "candidate_answer": "a",
  "confidence": "medium",
  "evidence": "이미지에서 확인한 짧은 근거",
  "ambiguity": "없음 또는 애매한 이유"
}
```

## 수정하면 안 되는 열

다음 열은 배정·원본 데이터이므로 수정하지 않는다.

```text
id, path, question, a, b, c, d, auto_type, assignment, is_calibration
```

작성 대상은 다음 열뿐이다.

```text
human_answer, confidence, readability, quality_reason, data_usage,
ambiguity_reason, evidence, reviewer, llm_assistance, review_status, review_note
```

행 삭제·정렬·ID 변경·이미지 파일명 변경은 하지 않는다.

## 반환물

작업 완료 후 다음 두 파일을 원래 이름 그대로 반환한다.

```text
calibration/labels.csv
teammate/labels.csv
```

반환 전 확인:

- [ ] calibration 합의가 완료됐다.
- [ ] 1,316개 본 작업 행의 `review_status`가 모두 채워졌다.
- [ ] `approved` 행은 필수 여섯 조건을 모두 만족한다.
- [ ] `partial/unreadable` 행에 억지 정답을 넣지 않았다.
- [ ] `reviewer`, `llm_assistance`, `evidence`를 기록했다.
- [ ] ID·질문·선지·행 수를 변경하지 않았다.
