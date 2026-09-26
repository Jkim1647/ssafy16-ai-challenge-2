# H408 오답 원인 분류 (2026-09-24)

입력: `reports/h408_errors_20260924.jsonl`(33문항, dev 정답 라벨 사용). 33장 이미지를 전부 직접 열어 보고, 필요한 경우 해당 영역을 확대 크롭해 확인했다. 행별 근거는 `reports/h408_error_taxonomy_20260924.csv` 참조.

분류 기준: LABEL_DEFECT(정답 오류·복수정답·정답 없음) / AMBIGUOUS(두 답 모두 방어 가능) / UNREADABLE(사람도 확인 불가) / SHOULD_GET(정보가 명확히 보이는데 선택 오류) / NEEDS_BETTER_READ(확대하면 사람은 읽히는데 모델이 오독·누락) / EXTERNAL_KNOWLEDGE.

## 1. primary × all_wrong

| primary | all_wrong=True | all_wrong=False | 합계 |
|---|---:|---:|---:|
| LABEL_DEFECT | 5 | 3 | 8 |
| AMBIGUOUS | 3 | 1 | 4 |
| UNREADABLE | 2 | 0 | 2 |
| SHOULD_GET | 11 | 1 | 12 |
| NEEDS_BETTER_READ | 3 | 4 | 7 |
| EXTERNAL_KNOWLEDGE | 0 | 0 | 0 |
| **합계** | **24** | **9** | **33** |

SHOULD_GET 세부(all_wrong=True만): COUNT_SPATIAL 4, FIELD 4, GROUNDING 2, OTHER(수치비교) 1

## 2. fixable

| fixable | all_wrong=True | all_wrong=False | 합계 |
|---|---:|---:|---:|
| 불가(데이터·모호·판독불가) | 10 | 4 | 14 |
| 해상도·크롭 | 3 | 3 | 6 |
| 새모델시각 | 1 | 1 | 2 |
| 프롬프트·연결 | 10 | 1 | 11 |
| 외부지식 | 0 | 0 | 0 |

## 3. 요약

- (개별 문항 서술 생략)

**(b) '맞혔어야 하는' 문항(SHOULD_GET): all_wrong 11건.** 주된 원인은 FIELD 4건(현재역↔행선 방향, 단지명↔동 번호, 학교명↔기념 문구, 상호↔메뉴)과 COUNT_SPATIAL 4건(층 세기 3건, 차로 1건)이다. 그 밖에 GROUNDING 2건은 가장 크고 눈에 띄는 간판을 질문 대상에 잘못 연결한 경우이고, OTHER 1건은 4.5와 5.0을 정확히 읽고도 비교를 틀린 경우다. 공통 패턴은 **글자는 제대로 읽었는데 가장 큰 간판 또는 질문과 문자열이 가장 많이 겹치는 선택지를 고른다**는 것이다. OCR(글자 판독) 실패가 아니라 선택 단계의 오류다.

- (개별 문항 서술 생략)

## 4. 관찰과 불확실성

- (개별 문항 서술 생략)
- (개별 문항 서술 생략)
- (개별 문항 서술 생략)
- NEEDS_BETTER_READ 7건 중 4건은 all_wrong=False였다(일부 제출에서는 이미 정답). 크롭·고해상도 파이프라인을 추가해도 all_wrong 기준으로 기대할 수 있는 이득은 약 3건이 상한이다.
- 이 분류는 H408 오답 33건만 대상으로 했으며, 정답 문항 가운데 우연히 맞힌 라벨 결함 문항은 포함하지 않았다.
