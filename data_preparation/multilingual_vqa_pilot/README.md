# 다국어 실물 장면문자 VQA 파일럿 (검수용)

한국어 질문 + 외국어 표기 4지선다를 대회 `train.csv` 형식에 맞춰 시험해 보는 초안이다. 일본어·영어·중국어 각 100문항, 총 300문항이다. **아직 학습에 넣지 말 것.** `review_status=pending`을 이미지별로 검수한 뒤 확정 문항만 사용한다. 대회 `test`/`dev` 이미지는 생성·정답 결정에 사용하지 않았다.

아래 `generated/` 파일과 `outputs/` 검토 산출물은 로컬 작업물이며 이 Git 저장소에는 포함하지 않는다. 따라서 이 문서는 결과의 출처·처리 방법을 설명하며, 파일이 필요한 팀원은 승인된 팀 채널에서 별도로 받아야 한다.

## 파일

- `generated/pilot_300.csv`: 전체 300문항. 첫 8열 `id,path,question,a,b,c,d,answer`는 대회 train과 호환되는 순서다. 나머지 열은 출처 추적·검수용이다.
- `generated/pilot_mtvqa_ja_100.csv`, `pilot_textvqa_en_100.csv`, `pilot_estvqa_zh_100.csv`: 언어별 검수 분할.
- `generated/images/{source}/{number}.jpg`: CSV의 `path`는 `generated` 폴더를 기준으로 한다. 다운로드된 원본 이미지를 편집하지 않았다.
- `generated/manifest.json`: 샘플링 시드와 문항 유형별 수량.
- `../../outputs/multilingual_vqa_pilot_20260922/multilingual_vqa_pilot_review.xlsx`: 3개 언어 탭의 검토용 통합 문서.
- `relational/relational_specs.json`, `relational/relational_specs_part2.json`, `generated/relational_pilot_60.csv`: 대상·위치·관계 식별을 요구하는 일본어·영어·중국어 각 20문항 초안. 기존 30문항 결과는 보존했다. 세부 검수 기준은 `relational/README.md`와 `RELATIONAL_QUESTION_GUIDE.md` 참조.
- `../../outputs/multilingual_vqa_relational_20260922/relational_pilot_review_60.xlsx`: 60문항 검토용 통합 문서.

## 데이터 출처·권리

| 소스 | 사용 구간 | 이미지/정답 출처 | 권리 주의 |
| --- | --- | --- | --- |
| 일본어 | MTVQA `train`, `lang=JA` | [ByteDance/MTVQA](https://huggingface.co/datasets/ByteDance/MTVQA)의 사람 작성 QA | CC BY-NC 4.0; 비상업 조건 확인 |
| 영어 | TextVQA `train` | [Meta TextVQA](https://huggingface.co/datasets/facebook/textvqa)의 사람 10명 응답 중 7명 이상 일치한 답. 이미지 포함 뷰어는 [ReplugLens/TextVQA](https://huggingface.co/datasets/ReplugLens/TextVQA) | 공식 카드 CC BY 4.0; 이미지 원 출처 OpenImages 조건도 확인 |
| 중국어 | EST-VQA `train` | [원 연구 저장소](https://github.com/xinke-wang/EST-VQA)의 QA가 포함된 [FineVision `est_vqa`](https://huggingface.co/datasets/HuggingFaceM4/FineVision) 뷰어 | 원 데이터의 재배포·학습 조건을 별도로 확인할 때까지 검수용으로만 유지 |

질문은 원어 QA의 좁은 유형(상호·문구·가격·전화번호·숫자·연도 등)을 규칙 기반 한국어 문장으로 바꿨다. 정답 텍스트는 원본 사람 주석을 유지하고, 오답은 같은 소스·유형의 다른 정답에서 골랐다. 정답 위치는 각 언어에서 a/b/c/d가 25회씩 되도록 배치했다. 외부 추론 API나 교사 모델로 정답을 생성하지 않았다.

## 검수 규칙

1. 이미지를 열어 정답 문구가 실제로 보이는지 확인한다. 흐림·가림·회전 등으로 확인 불가능하면 제외하거나 객체인식형으로 *새로* 작성한다.
2. 질문의 대상이 이미지에서 하나로 특정되는지 확인한다. `표시`·`대상`처럼 모호하면 `가게 간판`, `병 아래쪽` 등으로 고친다.
3. 나머지 선지가 이미지의 다른 문구로도 성립하는지 확인한다. 복수정답이면 선지를 교체한다.
4. 원본 답변의 오탈자·표기 변형, 문자 간체/번체 혼용, 숫자 단위 누락을 확인한다.
5. 확인 완료만 `review_status=approved`로 표시한다. 애매한 문항은 `rework` 또는 `excluded`로 표시하고 `review_note`에 이유를 남긴다.

`TextVQA`는 원본에서 한 이미지에 여러 질문이 있을 수 있어 전체 300문항 중 동일 이미지 3쌍이 있다. 서로 다른 질문이므로 문항 수는 300개지만 고유 이미지는 297개다. 경쟁 데이터와의 이미지 중복 여부 및 사람이 확정한 정답의 타당성은 아직 검증하지 않았다.

재생성: `python tools/build_multilingual_vqa_pilot.py`. 이 명령은 `generated`의 문항 파일을 덮어쓰므로 검수 결과를 입력한 뒤에는 실행하지 않는다.
