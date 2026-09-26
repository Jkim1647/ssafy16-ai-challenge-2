#!/usr/bin/env bash
# 35B LoRA: A안(타일 없음) -> C안(타일) 순차 실행. 같은 GPU라 병렬 불가.
#
#   A : 27B LoRA(val667 0.9685)와 **변수 하나(모델 27B->35B)**만 다르게 -> 모델 크기 효과 분리
#   C : A와 **변수 하나(타일)**만 다르게 -> 입력 형식 효과 분리
#
# 평가는 holdout1500(val667 포함 상위집합, 이미지 group split, 1문항 0.067%p).
# 본학습 전에 48/40 스모크를 A·C 각각 돌려 실제 초/문항을 재고, 그 값으로 남은 시간을 추정한다.
#
# 전제: /workspace/{tools,data} 업로드 완료, setup.sh 가 SETUP_DONE 을 남김.
set -uo pipefail
export HF_HOME=/workspace/hf
OUT=/workspace/out_lora
mkdir -p "$OUT"
st() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/STATUS"; }

st "대기 시작 (setup + 이미지 6714)"
for _ in $(seq 1 480); do
  ok=1
  grep -q SETUP_DONE /workspace/setup.log 2>/dev/null || ok=0
  [ "$(ls /workspace/data/train 2>/dev/null | wc -l)" -ge 6714 ] || ok=0
  [ "$ok" -eq 1 ] && break
  sleep 15
done
grep -q SETUP_DONE /workspace/setup.log || { st "setup 미완료 - 중단"; exit 1; }
[ "$(ls /workspace/data/train | wc -l)" -ge 6714 ] || { st "이미지 부족 - 중단"; exit 1; }

KERN=$(grep -oE 'FLA_OK|FLA_FAIL|CONV1D_OK|CONV1D_FAIL' /workspace/setup.log | tr '\n' ' ')
st "준비 완료 | 커널: ${KERN:-없음}"

cd /workspace/tools || exit 1
COMMON="--model Qwen/Qwen3.6-35B-A3B --data /workspace/data \
  --val-ids /workspace/data/holdout1500_valgroups.json \
  --exclude-csv /workspace/data/train_exclusions_v2.csv \
  --out $OUT --r 8 --alpha 16 --lr 1e-4 --epochs 1 --grad-accum 16 --seed 1 --skip-test"

# ---- 속도 실측용 스모크 (저장->재로드 검증도 겸한다) ----
st "smokeA 시작 (48/40, 타일 없음)"
timeout 3600 python colab_lora_train.py $COMMON --view full \
  --limit-train 48 --limit-val 40 --tag smokeA || { st "smokeA 실패 - 중단"; exit 1; }
st "smokeA 종료"

st "smokeC 시작 (48/40, 타일)"
timeout 5400 python colab_lora_train.py $COMMON --view tiles2x2 \
  --limit-train 48 --limit-val 40 --tag smokeC || st "smokeC 실패 - C 본학습은 건너뛸 수 있음"
st "smokeC 종료"

# ---- 본학습 ----
st "A 본학습 시작 (학습 5143 / 평가 1500, 타일 없음)"
timeout 28800 python colab_lora_train.py $COMMON --view full --tag A_35b_full
st "A 종료 rc=$?"
st "A_DONE"

st "C 본학습 시작 (학습 5143 / 평가 1500, 타일)"
timeout 57600 python colab_lora_train.py $COMMON --view tiles2x2 --tag C_35b_tiles
st "C 종료 rc=$?"
st "ALL_DONE"
ls -la "$OUT"
