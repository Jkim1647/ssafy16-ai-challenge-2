#!/usr/bin/env bash
# C안 전용: 35B LoRA + 2x2 타일. A안(타일 없음)과 **변수 하나(입력 형식)**만 다르게 한다.
# A안은 별도 파드에서 동시에 돌고 있으므로 여기서는 C만 실행한다.
#
# 평가는 holdout1500(val667 포함 상위집합, 이미지 group split, 1문항 0.067%p).
set -uo pipefail
export HF_HOME=/workspace/hf
OUT=/workspace/out_lora
mkdir -p "$OUT"
st() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/STATUS"; }

grep -q SETUP_DONE /workspace/setup.log || { st "setup 미완료 - 중단"; exit 1; }
[ "$(ls /workspace/data/train | wc -l)" -ge 6714 ] || { st "이미지 부족 - 중단"; exit 1; }
KERN=$(grep -oE 'KERNEL_FLA_OK|KERNEL_FLA_FAIL|KERNEL_CONV1D_OK|KERNEL_CONV1D_FAIL|TILELANG_DONE' /workspace/setup.log | tr '\n' ' ')
st "준비 완료 | 커널: ${KERN:-없음}"

cd /workspace/tools || exit 1
COMMON="--model Qwen/Qwen3.6-35B-A3B --data /workspace/data \
  --val-ids /workspace/data/holdout1500_valgroups.json \
  --exclude-csv /workspace/data/train_exclusions_v2.csv \
  --out $OUT --r 8 --alpha 16 --lr 1e-4 --epochs 1 --grad-accum 16 --seed 1 --skip-test"

# 저장->재로드 검증을 겸한 소량 스모크. 여기서 죽으면 본학습에 시간을 쓰지 않는다.
st "smokeC 시작 (48/40, 타일)"
timeout 5400 python colab_lora_train.py $COMMON --view tiles2x2 \
  --limit-train 48 --limit-val 40 --tag smokeC || { st "smokeC 실패 - 중단"; exit 1; }
st "smokeC 종료"

st "C 본학습 시작 (학습 5143 / 평가 1500, 타일)"
timeout 57600 python colab_lora_train.py $COMMON --view tiles2x2 --tag C_35b_tiles
st "C 종료 rc=$?"
st "C_DONE"
ls -la "$OUT"
