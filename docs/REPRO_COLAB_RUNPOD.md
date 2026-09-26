# Colab·RunPod 재현 가이드 (2026-09-21 밤 실험 기준)

교육장 PC나 다른 팀원이 9/21 밤 Colab 실험을 **같은 결과로** 다시 돌리고, RunPod에 옮기기 위한 순서다.
결과 요약은 `HANDOFF.md` 000절, 해시·버전·모델 revision은 `reports/repro_manifest_20260921.md`에 있다.

## 0. 받을 것

```bash
git clone https://github.com/Jkim1647/AI_2_CHALLENGE.git
```

- 저장소는 팀 전용(private)이다. 대회 데이터 원본(zip·이미지)은 git에 없다. 팀 계정 Drive의 `ssafy-16-2-ai.zip`(암호는 팀 공유) 또는 해제본 `ai2_data_extracted.tar`를 쓴다.
- 평가·학습에 쓰는 파생 자료는 git에 있다:
  - `data_meta/splits/val667_ids.json`: train 이미지 그룹 검증셋
  - `runs/dev_validation_v1/hard_eval_gold_v2.json`: 어려운 평가셋 184건 정답
  - `runs/dev_validation_v1/train_pseudo_candidates_final.csv`: 검증된 dev 학습 후보 1,002건
- 지난 실험의 예측·로그는 `shared_predictions/`에 있다. Colab Drive 출력 사본은 `shared_predictions/colab_ai2_35b_out/`이다.

## 1. Colab 준비 (G4 = RTX PRO 6000 96GB)

1. 새 노트북 → 런타임 유형 **G4**.
2. Drive 마운트 후 데이터를 푼다.
   ```python
   from google.colab import drive; drive.mount('/content/drive')
   !tar -xf /content/drive/MyDrive/ai2_data_extracted.tar -C /content && ls /content/data | head
   ```
3. 저장소의 `tools/colab_35b_infer.py`, `tools/colab_lora_train.py`, `tools/colab_vllm_infer.py`를 `/content`에 둔다. `val667_ids.json`, `h184_gold.json`, `h184_ids.json`, `dev_train_1002.csv`도 같이 둔다(아래 4절 참고).
4. HF 경로 설치:
   ```python
   !pip -q uninstall -y torchao
   !pip -q install "git+https://github.com/huggingface/transformers.git@ffddd25146e4225b97a6b5e5b66dd9a823bf39d1" accelerate peft bitsandbytes pillow pandas
   ```
   - torchao는 peft와 충돌해서 지운다.
   - 확인된 버전: torch 2.11.0+cu128(Colab 기본), transformers 5.18.0.dev0, peft 0.20.0.
5. vLLM 경로(122B GPTQ)는 **별도 런타임**에서 설치한다. HF 경로와 torch 버전이 다르다.
   ```python
   !pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
   !pip uninstall -y torchaudio; pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
   ```

## 2. 평가 파일 만들기 (로컬 git → Colab)

```python
import json, pandas as pd
g = json.load(open('hard_eval_gold_v2.json'))                        # runs/dev_validation_v1/
json.dump(g, open('/content/h184_gold.json', 'w')); json.dump(sorted(g), open('/content/h184_ids.json', 'w'))
d = pd.read_csv('train_pseudo_candidates_final.csv', encoding='utf-8-sig')
d[['id', 'path', 'question', 'a', 'b', 'c', 'd', 'answer']].to_csv('/content/dev_train_1002.csv', index=False)
```

`dev_train_1002.csv`의 `path`는 `dev/…jpg` 형식이고 `--data /content/data` 기준이다.

## 3. 재현 명령과 목표값

`OUT=/content/drive/MyDrive/ai2_35b_out` 기준.

| 실험 | 명령 | val667 | H184 |
|---|---|---|---|
| 35B 기본 | `python colab_35b_infer.py --model Qwen/Qwen3.6-35B-A3B --prompt ours --data /content/data --split val667 --split-json val667_ids.json --out $OUT` | 0.9445 | 0.739 |
| 35B 타일 | 위 명령 + `--view tiles2x2` | 0.9610 | 0.815 |
| **35B 판독+타일** | 위 명령 + `--prompt read --view tiles2x2` | **0.9655** | **0.837** |
| 27B LoRA 학습 | `python colab_lora_train.py --model Qwen/Qwen3.6-27B --data /content/data --val-ids val667_ids.json --out $OUT/lora --tag 27b_lora_r8` | 0.9685 | H189 0.709 |
| 27B LoRA + 판독+타일 | `colab_35b_infer.py --model Qwen/Qwen3.6-27B --adapter $OUT/lora/27b_lora_r8/adapter --prompt read --view tiles2x2 …` | HANDOFF 참조 | 0.804 |
| 122B Int4 | `python colab_vllm_infer.py --model Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 --quantization moe_wna16 --kv-gb 8 --max-num-seqs 8 --enforce-eager --max-model-len 8192 …` | 0.9490 | 0.717 |

- 어려운 셋(H184) 채점: `--split dev --ids-json h184_ids.json --gold-json h184_gold.json --tag-suffix H184`.
- test 추론: `--split test`. 끝나면 `<tag>_submission.csv`가 생기며, 6,714행·중복 없음·a~d만 들어 있는지 스크립트가 검사한다. 제출 CSV에는 BOM을 넣지 않는다.
- 속도(G4): 35B 기본 0.29 s/건, 타일 1.4 s/건, 판독+타일 1.8 s/건. 27B LoRA 학습 1.15 s/샘플, 타일 학습은 약 7.6배 느리다(9B 기준 3.36 s/샘플).

## 4. RunPod로 옮길 때

- 코드는 같다. `--data`와 `--out`만 RunPod 경로로 바꾼다.
- 학습 1순위: **35B(또는 27B) LoRA + `--view tiles2x2` + `--extra-csv dev_train_1002.csv`**. 추론은 `--prompt read --view tiles2x2 --adapter …`로 한다.
- 4비트(QLoRA)는 9B에서 어려운 문항이 −7.4%p였다. 큰 모델도 BF16 LoRA를 우선하고, 4비트를 쓰면 H184로 반드시 비교한다.
- 122B·397B는 판독+타일 조건에서 H184로 먼저 비교한다. 397B는 Colab에서 측정한 적이 없다.
- 재현용 고정: `--revision`(`colab_lora_train.py`)에 manifest의 모델 commit을 넣는다.

## 5. Colab에서 git으로 올리기

`tools/colab_drive_to_git.py`를 쓴다. 토큰은 Colab 보안 비밀 `GH_TOKEN`에 넣는다. 세분화 토큰으로 이 저장소만 고르고, 권한은 Contents: Read and write만 준다. 토큰은 출력·`.git/config`에 남지 않는다. 45MB가 넘는 파일(adapter 등)은 `LARGE_FILES.md`에 경로·크기·SHA-256만 남긴다.
