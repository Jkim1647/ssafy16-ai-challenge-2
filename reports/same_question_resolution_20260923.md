# 동일질문_다른정답 대기열 기계 판정 (2026-09-23)

`tools/resolve_same_question.py`. 질문이 같아도 **이미지가 다르면 정상**이므로, 같은 질문 안에서 이미지까지 같은 경우만 모순으로 본다.
문항 텍스트가 든 표는 `reports/review_queue_20260923/train_same_question_resolved.csv`(git 제외)에 있고 여기에는 집계만 적는다.

- 질문이 2건 이상 겹치는 train 행: **851**건
- 이 중 이미지까지 같아 판정 대상이 된 행: **0**건

| 판정 | 행 수 | 뜻 | 조치 |
|---|---|---|---|
| confirmed_conflict | 0 | 같은 질문·같은 이미지인데 정답이 다르다 | 학습 제외(`data_meta/train_exclusions_v3.csv`) |
| exact_duplicate | 0 | 같은 질문·같은 이미지·같은 정답 | 결함 아님. 중복 가중만 주의 |
| near_dup_conflict | 0 | dhash는 같고 픽셀은 다른데 정답이 다르다 | 사람 확인 필요(리사이즈본일 수 있음) |

- 학습(5,975)에 실제 포함된 confirmed_conflict: **0**건 → `--exclude-csv data_meta/train_exclusions_v3.csv`
- 나머지 동일질문 행은 이미지가 달라 **사람 검수가 필요 없다**. 대기열에서 내린다.
