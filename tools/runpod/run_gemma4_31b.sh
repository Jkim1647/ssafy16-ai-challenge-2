#!/usr/bin/env bash
# Gemma 4 31B IT — 6,047건 LoRA 학습. 집 노트북에서 클라우드 파드에 띄운다.
#
# 왜 6,047 인가.
#   지금까지 Gemma 파인튜닝은 전부 1,000건 규모였고 둘 다 실패했다
#   (팀원 QLoRA 763->763, 다른 팀 95.83->95.54). 반면 우리 27B 는 6,047건으로
#   zero-shot 637 -> LoRA 646, +9문항 CI[+0.30,+2.40]%p 로 유의하게 벌었다.
#   즉 실패 원인이 모델이 아니라 학습량일 가능성이 크다. 그 가설을 직접 잰다.
#
# 왜 학습이 필요한가.
#   둘 다 틀린 144건의 86%는 판독이 맞고 합의 라벨과 어긋난 건이다.
#   zero-shot 은 원리적으로 이 규칙을 못 배운다. Gemma base 가 25건 중 4건을
#   맞힌 것은 운에 가깝고, 학습해야 규칙 자체를 배운다.
#
# 평가는 val667 로 고정한다. 27B LoRA 0.9685 와 **같은 자**여야 비교가 된다
# (팀원의 806건 자체 셋으로는 비교 불가).
#
# bf16 로 간다. 4-bit QLoRA 는 팀원 실행에서 이미 이득이 없었고, 141GB 급
# 파드면 31B bf16 이 여유 있게 들어간다.
set -uo pipefail
export HF_HOME=/workspace/hf
OUT=/workspace/out_gemma
MODEL="${MODEL:-google/gemma-4-31B-it}"
VIEW="${VIEW:-tiles2x2}"
mkdir -p "$OUT"
st() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/STATUS"; }

[ "$(ls /workspace/data/train 2>/dev/null | wc -l)" -ge 6714 ] || { st "이미지 부족 - 중단"; exit 1; }
df -h /workspace | tail -1 | awk '{print "  디스크 여유: "$4}'
st "$MODEL 다운로드 시작"
python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL', max_workers=16)" \
  || { st "다운로드 실패 - 중단"; exit 1; }

cd /workspace/tools || exit 1
BASE="--model $MODEL --data /workspace/data --out $OUT --view $VIEW \
  --val-ids /workspace/data/val667_ids.json \
  --exclude-csv /workspace/data/train_exclusions_v2.csv \
  --r 8 --alpha 16 --lr 1e-4 --epochs 1 --grad-accum 16 --seed 1 --skip-test"

# Gemma 는 우리 파이프라인에서 처음 돌린다. 스모크에서 확인할 것:
#   - a/b/c/d 가 각각 토큰 1개인가 (학습기가 assert 로 잡는다)
#   - chat template 이 enable_thinking 없이 렌더링되는가 (TypeError 대비 코드 있음)
#   - 저장 -> 재로드 -> 추론이 되는가
#   - 초기 손실이 ln4 ≈ 1.386 근처인가
st "스모크 시작 (48/40) | VIEW=$VIEW"
timeout 7200 python colab_lora_train.py $BASE --limit-train 48 --limit-val 40 --tag smoke_gemma \
  || { st "스모크 실패 - 중단 (로그에서 토큰 ID·chat template 확인)"; exit 1; }
st "SMOKE_GEMMA_END"

st "본학습 시작 (학습 5976 / 평가 667)"
timeout 86400 python colab_lora_train.py $BASE --tag gemma4_31b_lora
rc=$?
st "본학습 종료 rc=$rc"

# 파드에 볼륨이 없으면 정지 시 전부 사라진다. 회수가 한 번에 끝나도록 묶어 둔다.
tar czf "$OUT/gemma_artifacts.tgz" -C "$OUT" gemma4_31b_lora 2>/dev/null \
  && st "묶음 생성 $(du -h "$OUT/gemma_artifacts.tgz" | cut -f1)" \
  || st "묶음 실패 - gemma4_31b_lora/ 를 직접 회수할 것"
st "GEMMA_DONE"
ls -la "$OUT"
