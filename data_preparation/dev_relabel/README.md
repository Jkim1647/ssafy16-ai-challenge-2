# dev 재라벨링 분할

공식 dev 2,683행에서 교육생 응답 열을 완전히 제거하고, 공동 교정용 50행과 두 사람의 독립
작업분을 결정적으로 생성한다.

```powershell
python data_preparation/dev_relabel/split_dev.py
```

기본 출력:

```text
generated/dev_split_markers_v1/
├─ calibration/
│  └─ labels.csv
├─ mine/
│  └─ labels.csv
├─ teammate/
│  └─ labels.csv
├─ manifest.json
└─ teammate_handoff.zip
```

`teammate_handoff.zip`에는 팀원 작업분, 공동 calibration, 상세 전달 문서, 가이드 초안과
manifest만 들어간다. 이미지는 포함하지 않으며 CSV의 `path`는 동료가 보유한 공식 데이터의
`dev/dev_XXXX.jpg`를 가리킨다. 이미지까지 필요한 경우에만 `--include-images`를 명시한다.
원본 `answer1`~`answer5`는 어떤 산출물에도 포함하지 않는다.

흐림·저해상도·반사·가림 이미지는 라벨링 단계에서 `vision_auxiliary_candidate`로 격리한다.
`export_reviewed.py`는 명확하고 승인된 행만 VQA 학습 후보로 내보내는 하드 게이트다.

```powershell
python data_preparation/dev_relabel/export_reviewed.py `
  --input data_preparation/dev_relabel/generated/dev_split_markers_v1/calibration/labels.csv `
          data_preparation/dev_relabel/generated/dev_split_markers_v1/mine/labels.csv `
          data_preparation/dev_relabel/generated/dev_split_markers_v1/teammate/labels.csv `
  --output data_preparation/dev_relabel/generated/dev_split_markers_v1/reviewed
```

같은 seed와 원본 CSV로 다시 실행하면 같은 ID 배정이 나온다. 기존 출력 디렉터리가 비어 있지
않으면 중단해 작성 중인 라벨을 덮어쓰지 않는다.

## 우리 배치 로컬 VLM 초안

`label_mine_with_local_vlm.py`는 `assignment=mine` 행만 읽어 로컬 오픈웨이트 VLM의 후보를
JSONL에 행 단위로 누적한다. 원본 dev 응답 열은 읽지 않으며 외부 모델 API를 호출하지 않는다.
모델 결과는 `model_*` 열에만 기록하고 `human_answer`는 비운 채 모든 처리 행을
`needs_review`로 유지한다. 중단 후 같은 명령을 다시 실행하면 완료된 ID를 건너뛴다.

```powershell
python data_preparation/dev_relabel/label_mine_with_local_vlm.py `
  --image-root C:\path\to\ssafy-16-2-ai `
  --model downloads\models\Qwen2.5-VL-3B-Instruct
```

초안 생성 후에는 확률이 낮고 선택지 간 margin이 작은 행부터 검수 큐를 만든다.

```powershell
python data_preparation/dev_relabel/prepare_review_queue.py
```

`review_queue_low.csv` → `review_queue_medium.csv` → `review_queue_high.csv` 순서로 원본 이미지를
직접 확인한다. 큐 생성 자체는 어떤 행도 승인하거나 `human_answer`를 채우지 않는다.

mine의 Codex-assisted 검수는 모델 후보를 숨긴 시트로 먼저 판정한다.

```powershell
python data_preparation/dev_relabel/prepare_codex_review_sheets.py `
  --image-root ..\ssafy-16-2-ai `
  --limit 20

python data_preparation/dev_relabel/apply_codex_reviews.py
```

결정 journal은 `codex_review/decisions.jsonl`, 원본을 덮어쓰지 않은 병합본은
`codex_review/mine_codex_reviewed.csv`다. `reviewer=codex-assisted-visual`을 유지하며 로컬 모델
간 합의만으로 승인하지 않는다.

9B의 독립 후보를 붙인 뒤 아직 검수하지 않은 행의 우선순위를 다시 계산할 수 있다.

```powershell
python data_preparation/dev_relabel/prepare_multimodel_review_queue.py `
  --predictions runs/9b-dev-mine-review-20260921-r2/predictions.jsonl
```

출력 `codex_review/multimodel_queue/pending_priority.csv`는 3B/9B 불일치(`P0`)를 먼저 두고,
두 모델이 같을 때는 낮은 공통 신뢰도부터 `P1`~`P3`으로 정렬한다. 이 우선순위는 검수 순서만
정하며 자동 승인을 만들지 않는다. 2026-09-21 파일럿 40건은 독립 시각 판정 후 9B 충돌 사례를
원본으로 재확인했고, 최종 16건 승인·24건 제외 상태다.

기존 질문·선지가 결함이지만 이미지에 선명한 정보가 있으면 `regenerated_questions/`에 원본을
덮어쓰지 않는 새 문항을 만든다. `source_id`와 `image_id`를 보존하고, 질문 대상을 명시하며,
같은 범주의 오답 세 개와 단일 정답을 구성한다. 첫 파일럿은
`regenerated_questions/pilot_20.csv`이며 정답 위치는 a~d 각 5건이다.
재생성 CSV의 `source_id`는 다음 멀티모델 검수 큐에서 자동으로 제외되지만, 원본 CSV 행이나
직접 검수 journal을 변경하지 않는다.

```powershell
python data_preparation/dev_relabel/validate_regenerated_questions.py `
  data_preparation/dev_relabel/generated/dev_split_markers_v1/mine/regenerated_questions/pilot_20.csv `
  --image-root ..\ssafy-16-2-ai `
  --require-balanced-answers
```

## 재생성 문항 사람 최종 검수

전수 이미지 대조 기록은 `regenerated_questions/semantic_review_ledger.csv`, 이상 목록은
`semantic_review_flags.csv`에 있다. 번호순 검수본은 다음 명령으로 한 번 생성한다.

```powershell
python data_preparation/dev_relabel/prepare_human_review.py
```

결과인 `regenerated_questions/human_review_sorted.csv`는 원본
`codex_regenerated_all.csv`를 보존한다. 사진으로 근거가 확인된 이상 11건을 교정하고,
13건의 이상 표시와 `human_review_status`를 추가한다. 흐릿해 반려된 2건은 빈 문항을
유지한다. `dev_0411.jpg`는 판독이 어려운 상단 글자 대신 선명한 세로 문구로 다시
작성했다. 이 파일을 검수하다가
다시 준비 명령을 실행하면 기존 파일을 덮어쓰지 않고 중단한다.

`human_review.html`을 Chrome 또는 Edge에서 열고 `CSV 열기`로 위 검수본을 선택한 뒤,
`사진 폴더 선택`에서 이미지가 들어 있는 폴더를 고른다. 선택한 사진의 개수와 관계없이
파일명의 `dev_숫자`를 우선 사용하고, 그 형식이 없으면 마지막 숫자 묶음으로 CSV의
사진 번호와 연결한다. `시작 번호`·`끝 번호`로
- (개별 문항 서술 생략)
`표시 범위 CSV 내보내기`로 현재 작업분만 동료에게 전달할 수 있다.
질문·선지·정답·사람 검수 상태·메모를 직접 수정한 후 `CSV 저장`을 누른다.
브라우저가 파일 쓰기를 지원하면 연 CSV에 저장하고, 지원하지 않으면 편집본을
다운로드한다. 사진과 CSV는 브라우저 안에서만 처리하며 HTTP 서버는 사용하지 않는다.
사람 검수가 끝나기 전 `human_review_status`가 `pending` 또는 `needs_review`인 행을
학습 확정 데이터로 취급하지 않는다.

5명이 나눠 검수할 때는 번호순으로 정렬된 위 CSV를 다음 명령으로 분할한다.

```powershell
python data_preparation/dev_relabel/split_human_review_five.py
```

`regenerated_questions/human_review_5way/`에 전체본 1개와 담당자별 CSV 5개가 생긴다.
담당자 파일은 각각 264·264·263·263·263건이며 서로 겹치지 않는다. 각자 HTML 검수
화면에서 자기 파일만 열어 작업한다. 이미 편집한 출력 파일은 이 명령이 덮어쓰지 않는다.
