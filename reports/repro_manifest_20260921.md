# 재현 manifest — 2026-09-21 밤 Colab 실험

내일(09/22) 강의실이나 RunPod에서 같은 결과를 다시 내기 위한 기록이다. 브랜치는 `feat/9b-setup-dev-relabel`.
대회 데이터·재라벨 CSV·가중치·adapter는 git에 넣지 않는다. 여기에는 **위치와 해시만** 적는다.

## 1. 코드

| 파일 | 역할 | SHA-256 (앞 16자) |
|---|---|---|
| `tools/colab_35b_infer.py` | HF transformers 추론. `--prompt ours/en/read`, `--view full/tiles2x2`, `--rotate`, `--shift`, `--adapter` | `ca9ef507700c092b` |
| `tools/colab_vllm_infer.py` | vLLM in-process 추론(122B GPTQ). HTTP 서버를 쓰지 않는다 | `5ca61a4686501bbd` |
| `tools/colab_lora_train.py` | LoRA/QLoRA 학습 → val667·어려운 셋·test 채점 | `4a65f1ed6ce7cb7d` |
| `tools/ensemble_submit.py` | 확률 평균 앙상블 → 제출 CSV | `d267999b5bdc728e` |

## 2. 모델 (Hugging Face revision, 2026-09-21 조회)

실행 시 revision을 고정하지 않았다. 아래는 조회 시점의 최신 commit이다. 재현할 때는 `--revision`으로 고정하는 것을 권한다(스크립트에 옵션 추가 필요).

| 모델 | revision | 비고 |
|---|---|---|
| `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | bf16 LoRA / NF4 QLoRA |
| `Qwen/Qwen3.6-27B` | `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` | bf16 |
| `Qwen/Qwen3.6-35B-A3B` | `995ad96eacd98c81ed38be0c5b274b04031597b0` | bf16 |
| `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` | `30cd92cba9707a9aba09d1e490ed4b66b78e9606` | vLLM `moe_wna16` |

## 3. 환경

- **Colab G4** (RTX PRO 6000 Blackwell 96GB), Python 3.13.
- HF 경로: `pip install "git+https://github.com/huggingface/transformers.git@ffddd25146e4225b97a6b5e5b66dd9a823bf39d1" accelerate peft bitsandbytes pillow pandas` → transformers `5.18.0.dev0`. **torchao는 제거**(`pip uninstall -y torchao`, peft와 충돌). torch는 Colab 기본값이며 버전을 기록하지 않았다(다음 실행부터 기록할 것).
- vLLM 경로: `pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly` → `pip uninstall -y torchaudio; pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130`. 확인된 조합은 torch `2.13.0+cu130` / torchvision `0.28.0+cu130` / vllm `0.29.1rc1.dev470+g15859bb3a`.

## 4. 고정 설정

- 채점: 답 위치의 다음 토큰 로그확률을 a~d 4개로 제한해 argmax를 고른다. `enable_thinking=False`. 이미지는 원본 해상도 그대로(EXIF 회전 보정만) 넣는다.
- 프롬프트 `ours`: `이미지를 보고 질문에 답하시오.\n질문: {question}\n{a. .. d. ..}\n정답 기호 하나만 출력하시오.`
- `read`: 1단계에서 질문 관련 글자를 원문 그대로 받아 적게 한다(greedy, 최대 80토큰). 2단계에서 그 판독을 대화에 남긴 채 a~d를 채점한다. 문구는 `colab_35b_infer.py`의 `READ_1`/`READ_2`에 있다.
- `tiles2x2`: 전체 사진 1장 + 2×2 타일 4장(겹침 10%, 2배 BICUBIC 확대). 원본 픽셀을 자르기만 하고 생성형 업스케일은 쓰지 않는다.
- LoRA: r=8, alpha=16, dropout=0.05, lr=1e-4, epoch 1, grad_accum 16, seed 1. 대상은 `language_model`의 q/k/v/o/gate/up/down(비전 제외). 손실은 a~d 글자 토큰 CE다(채점 위치와 같음). QLoRA는 NF4 + double quant + bf16 compute.
- 122B vLLM: `--kv-gb 8 --max-num-seqs 8 --enforce-eager --max-model-len 8192`. KV 캐시 자동 추정이 G4에서 OOM이 나서 고정했다. 정확도와는 무관하다.

## 5. 데이터 (git 밖 — 팀 승인 채널로 전달)

| 자료 | 위치 | SHA-256 | 비고 |
|---|---|---|---|
| 공식 데이터 | 팀 Drive `ssafy-16-2-ai.zip`, 해제본 `ai2_data_extracted.tar` | — | 대회 후 삭제 |
| val667 분할 | **git** `data_meta/splits/val667_ids.json` | `d6d3da6548e71d9b…` | train 이미지 그룹 단위 |
| dev 학습 후보 1,002건 | 로컬 `runs/dev_validation_v1/train_pseudo_candidates_final.csv` | `4574fddc798a84e2…` | 3자 일치 + 근거 명확. 기준은 `reports/dev_validation_20260921.md` |
| 어려운 평가셋 정답 184건 | 로컬 `runs/dev_validation_v1/hard_eval_gold_v2.json` | `c500931ba55864ba…` | 사람 검토 반영. 사용 허용 범위는 팀 확인 필요 |
| 어려운 평가셋 원표 | 로컬 `runs/dev_validation_v1/hard_eval_visual_verified.csv` | `1c8b6e21ee4535b2…` | |
| 예측·로그·adapter | 팀 계정 Drive `MyDrive/ai2_35b_out/` (adapter는 `lora/<tag>/adapter`) | — | 파일 목록은 `shared_predictions/20260921/README.md` |

dev 학습 후보와 어려운 평가셋은 이미지 그룹이 겹치지 않는다. `colab_lora_train.py`가 학습 시작 전에 assert로 확인한다.

## 6. 실행 명령 (Colab, `/content`에 데이터·스크립트가 있을 때)

```bash
# 35B 기준 / 판독 / 타일 — 어려운 셋 184
python colab_35b_infer.py --model Qwen/Qwen3.6-35B-A3B --data /content/data --split dev --ids-json h184_ids.json --gold-json h184_gold.json --tag-suffix H184 --out OUT --prompt ours
python colab_35b_infer.py ... --prompt read
python colab_35b_infer.py ... --prompt ours --view tiles2x2
# val667
python colab_35b_infer.py --model Qwen/Qwen3.6-35B-A3B --data /content/data --split val667 --split-json val667_ids.json --out OUT --prompt read
# 122B
python colab_vllm_infer.py --model Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 --quantization moe_wna16 --kv-gb 8 --max-num-seqs 8 --enforce-eager --max-model-len 8192 --data /content/data --split val667 --split-json val667_ids.json --out OUT
# 9B LoRA + dev 1,002 (dev_train_1002.csv = 위 후보 CSV의 id,path,question,a,b,c,d,answer)
python colab_lora_train.py --model Qwen/Qwen3.5-9B --data /content/data --val-ids val667_ids.json --out OUT/lora --hard-gold h189_gold.json --skip-test --extra-csv dev_train_1002.csv --tag 9b_lora_devtrain
```

## 7. 결과 (재현 목표값)

| 실험 | val667 | 어려운 셋 |
|---|---|---|
| 9B zero-shot | 0.9430 | H189 0.609 |
| 9B LoRA (train) | 0.9580 | H189 0.635 |
| 9B LoRA (train + dev 1,002) | **0.9640** | H189 **0.661** |
| 27B zero-shot / 타일 | 0.9550 / 0.9595 | T1(156) 0.609 / 0.692 |
| 35B ours / read | 0.9445 / 진행 중 | H184 0.739 / **0.755** |
| 122B GPTQ-Int4 | 0.9490 | H184 진행 중 |

## 8. 알려진 간극 (RunPod 전에 메울 것)

1. **`colab_lora_train.py`는 이미지 1장 입력이다.** 계획의 "35B LoRA + 타일 + dev 1,002"는 이 스크립트로 바로 돌릴 수 없다. 학습에 `--view tiles2x2`를 추가하고, 2~3 step 학습 → 저장 → 새 프로세스 재로드 → 타일 추론 smoke를 통과시킨 뒤 결과를 git에 올린다.
2. `src/train.py`에는 crop 입력 경로가 있지만, dev 추가 학습(`--extra-csv`)과 같은 경로로 연결되는지는 확인하지 않았다.
3. 모델 revision과 torch·peft 버전을 스크립트가 meta에 기록하지 않는다(revision 고정 옵션 추가 필요).
