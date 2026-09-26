# 모델 담당자 전달: 외국어 장면문자 VQA 300문항 파일럿

## 목적과 현재 상태

대회 `train.csv`의 **한국어 질문 + 외국어가 적힌 실제 이미지 + a~d 선지** 형태를 확장해, 일본어·영어·중국어 각 100문항을 만들었다. 원본 공개 VQA 데이터의 사람 정답을 유지하고 한국어 질문·오답 선지만 변환했다. **현재 전 문항 `review_status=pending`: 모델 성능을 보는 탐색용/사람 검수용이며 학습 확정본이 아니다.**

## 전달 파일

기준 디렉터리: `AI_2_CHALLENGE/data_preparation/multilingual_vqa_pilot/`

| 용도 | 경로 | 내용 |
| --- | --- | --- |
| 전체 문항 | `generated/pilot_300.csv` | 300행, 첫 8열은 공식 train 스키마와 동일한 `id,path,question,a,b,c,d,answer` |
| 언어별 문항 | `generated/pilot_mtvqa_ja_100.csv`, `generated/pilot_textvqa_en_100.csv`, `generated/pilot_estvqa_zh_100.csv` | 각 100행 |
| 이미지 | `generated/images/{source}/{0001..0100}.jpg` | CSV의 `path`를 `generated/` 기준으로 해석 |
| 검토 시트 | `AI_2_CHALLENGE/outputs/multilingual_vqa_pilot_20260922/multilingual_vqa_pilot_review.xlsx` | Japanese / English / Chinese 탭, 원문 QA와 검수 칸 포함 |
| 방법·권리 | `README.md`, `generated/manifest.json` | 생성 방식, 출처, 라이선스 주의, 유형별 수량 |
| 추론 프롬프트 | `INFERENCE_PROMPT.txt` | 이미지 1장+질문 1개+선지 4개를 입력할 때 사용 |

CSV의 `answer`, `source_answer`는 **정답/검수 메타데이터**다. 평가·추론 모델 입력에는 절대 넣지 않는다. 모델 입력은 이미지와 `question,a,b,c,d`뿐이다. `source_question`도 평가 때는 제외한다. 결과는 `id,prediction` 형태로 별도 파일에 기록하고, 원본 CSV를 덮어쓰지 않는다.

## 모델 입력·출력 계약

1. 이미지 경로: `generated / row.path` (예: `generated/images/mtvqa_ja/0001.jpg`).
2. 텍스트 입력: `INFERENCE_PROMPT.txt`의 `{question}`, `{a}`~`{d}`만 치환. 이미지 자체는 로컬 VLM의 이미지 입력으로 전달.
3. 출력: `a`, `b`, `c`, `d` 중 **하나의 소문자**. 파싱 실패·다중 선택·빈 응답은 조용히 정답으로 강제하지 말고 오류로 기록한다.
4. 같은 이미지에 질문이 2개일 수 있다(TextVQA 3쌍). 이미지 해시가 같아도 각 `id`의 질문을 독립적으로 처리한다.
5. 외부 모델 API로 이미지/질문을 전송하거나 API 응답을 정답·teacher 신호로 사용하지 않는다. 허용된 서버에서 open-weight 모델을 직접 실행하고 예측을 파일로 남긴다.

## 비교 실험 요청

- **먼저** 공식 train의 고정 Group Split 검증 세트에서 기존 모델·프롬프트의 기준선을 측정한다. 경쟁 test는 검증 세트가 아니다.
- 이 300문항은 언어별 약점 탐색에만 우선 사용한다. 모델 예측과 `answer`의 일치율을 언어별·유형별로 기록하되, 사람 검수 전 정확도는 *참고치*로만 표기한다.
- 학습 실험은 `review_status=approved`인 행만 별도 파일로 추출한 후 진행한다. 학습/검증 간 동일·근접 이미지가 섞이지 않도록 해시/유사도 검사를 한다. 공식 train 검증 부분은 그대로 두고 외부 데이터는 학습 부분에만 추가한다.
- **원본 train만** vs **원본 train + 검수 승인 외부 데이터**를 같은 모델·seed·split·해상도·학습 설정으로 비교한다. 전체 및 외국어/회전 이미지 유형별 Accuracy를 함께 보고한다.
- 모델 버전, 프롬프트 버전, 사용 데이터 ID, seed, LoRA/양자화 설정, 학습 시간, GPU, 예측 파일을 기록한다.

## 사용 전 필수 검수

- 사진의 정답 문구 가독성, 질문의 지시 대상, 선지의 복수정답 여부, 일본어/중국어 표기와 숫자 단위를 사람 눈으로 확인한다.
- 흐릿하거나 대상이 불명확한 문항은 `rework` 또는 `excluded`; 확인된 것만 `approved`.
- EST-VQA 원본의 이용·재배포 조건을 확인하기 전에는 중국어 100문항을 학습·외부 공유에 넣지 않는다. MTVQA는 CC BY-NC 4.0, TextVQA 공식 카드는 CC BY 4.0으로 표시한다. 자세한 출처 링크는 `README.md` 참조.

## 현재 자동 점검 (2026-09-22)

- 300행(일본어 100/영어 100/중국어 100), 이미지 누락 0, 중복 선지 0, 선지-원본 정답 불일치 0.
- 정답 위치 a/b/c/d 각각 75개. 동일 이미지 3쌍으로 고유 이미지 297장.
- **이미지별 전수 사람 검수와 공식 train 대비 이미지 중복 검사는 아직 완료되지 않음.**
