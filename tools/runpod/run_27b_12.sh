#!/usr/bin/env bash
# 27B dense 데이터 확대 ①②. C안(35B+타일) 결과로 VIEW를 정한 뒤 실행한다.
#
#   R1 기준선 : 5,143건, 증강 없음         -> 지금 저장소에 없는 "깨끗한 27B 기준점"
#   R2 증강   : 5,143건 + 비생성 이미지증강 -> R1과 **변수 하나(증강)**만 다르다
#
# ③(전체 6,047 재학습)은 R1·R2 결과를 보고 증강 채택 여부를 정한 뒤
# run_27b_3.sh 로 따로 돌린다. 검증 안 된 선택을 스크립트에 박아두지 않는다.
#
# 평가는 holdout1500(val667 포함 상위집합, 이미지 group split, 1문항 0.067%p).
set -uo pipefail
export HF_HOME=/workspace/hf
OUT=/workspace/out_27b
MODEL=Qwen/Qwen3.6-27B
VIEW="${VIEW:-full}"          # full | tiles2x2
mkdir -p "$OUT"
st() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/STATUS"; }

[ "$(ls /workspace/data/train 2>/dev/null | wc -l)" -ge 6714 ] || { st "이미지 부족 - 중단"; exit 1; }
st "27B 다운로드 시작"
python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL', max_workers=16)" \
  || { st "다운로드 실패 - 중단"; exit 1; }
st "준비 완료 | VIEW=$VIEW"

cd /workspace/tools || exit 1
BASE="--model $MODEL --data /workspace/data --out $OUT --view $VIEW \
  --val-ids /workspace/data/holdout1500_valgroups.json \
  --exclude-csv /workspace/data/train_exclusions_v2.csv \
  --r 8 --alpha 16 --lr 1e-4 --epochs 1 --grad-accum 16 --seed 1 --skip-test"

# 실제 초/샘플을 재고 저장->재로드까지 확인한다. 증강 경로도 같이 태워 PIL 호출을 검증한다.
st "스모크 시작 (48/40, 증강 ON)"
timeout 5400 python colab_lora_train.py $BASE --aug-image \
  --limit-train 48 --limit-val 40 --tag smoke27 || { st "스모크 실패 - 중단"; exit 1; }
st "SMOKE_END"

st "R1 시작 (5143, 증강 없음)"
timeout 43200 python colab_lora_train.py $BASE --tag R1_27b_base
st "R1_DONE rc=$?"

st "R2 시작 (5143, 이미지 증강)"
timeout 43200 python colab_lora_train.py $BASE --aug-image --tag R2_27b_augimg
st "R2_DONE rc=$?"

st "R12_ALL_DONE"
ls -la "$OUT"
