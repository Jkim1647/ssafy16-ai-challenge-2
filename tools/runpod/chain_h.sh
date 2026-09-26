#!/usr/bin/env bash
# 2026-09-24 밤 파드 H — 10번 Qwen3.5-9B LoRA 설정 비교 **전체판**(사용자 결정: DoRA 포함 전체판, 채점할 수 있는 것 전부).
#
# 네 설정을 R3 와 같은 fit 5,976건으로 1 epoch 학습하고, 같은 프로세스에서 val667·H408·test 6,714 를 모두 채점한다.
#   base  r8/α16 lr 1e-4 (기존 9B LoRA 와 같은 설정)
#   dora  DoRA r8/α16 lr 1e-4
#   r16   r16/α32 lr 1e-4
#   lr2   r8/α16 lr 2e-4
# 순서는 사용자가 강조한 DoRA 를 기준 바로 다음에 둔다. $W/STOP_AFTER 가 있으면 기준·DoRA 만 돈다(B안, 충전 없이 가능).
# 입력은 기존 9B LoRA 와 같은 view=full. 보기 셔플은 넣지 않는다 — 셔플 효과는 9번(Qwen3.5-27B)이 보고,
# 여기서는 설정 차이만 본다.
set -uo pipefail
W=/workspace
L=$W/chainH.log
OUT=$W/out_h
MODEL=Qwen/Qwen3.5-9B
export HF_HOME=$W/hf
mkdir -p $OUT
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$L"; }

log "시작"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 | tee -a "$L"
python -c "import huggingface_hub" 2>/dev/null || pip install --break-system-packages -q huggingface_hub 2>&1 | tail -1
( python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL', max_workers=16)" \
    > $W/dlh.log 2>&1; echo "DL_EXIT $?" >> $W/dlh.log ) &
DL=$!
pip install --break-system-packages -q "transformers==5.17.0" "peft==0.21.0" accelerate pandas pillow bitsandbytes 2>&1 | tail -2 | tee -a "$L"
pip install --break-system-packages -q flash-linear-attention 2>&1 | tail -1 | tee -a "$L"
python -c "import torch, transformers, peft; print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__)" 2>&1 | tee -a "$L"

for _ in $(seq 1 120); do [ -f $W/IMAGES_READY ] && break; sleep 30; done
for x in train dev test; do log "이미지 $x $(ls $W/data/$x 2>/dev/null | wc -l)장"; done
[ "$(ls $W/data/train | wc -l)" -ge 6714 ] && [ "$(ls $W/data/test | wc -l)" -ge 6714 ] && [ "$(ls $W/data/dev | wc -l)" -ge 408 ] \
  || { log "이미지 부족 - 중단"; exit 1; }
wait $DL; grep -q "DL_EXIT 0" $W/dlh.log || { log "다운로드 실패 - 중단"; exit 1; }

cd $W/tools || exit 1
BASE="--model $MODEL --data $W/data --out $OUT --view full --val-ids $W/data/val667_ids.json \
  --exclude-csv $W/data/train_exclusions_v2.csv --epochs 1 --grad-accum 16 --seed 1 \
  --hard-gold $W/data/hard_gold_h408.json"

# 스모크: 학습 48 / 평가 40 — 속도(s/샘플)를 재서 비용을 다시 계산한다. 실패하면 본실행에 돈을 쓰지 않는다.
log "스모크 시작"
timeout 3600 python colab_lora_train.py $BASE --r 8 --alpha 16 --lr 1e-4 --dora --limit-train 48 --limit-val 40 --skip-test \
  --tag smoke9b > $W/smoke_h.log 2>&1 || { log "스모크 실패 - 중단: $(grep -a -m1 -E 'Error' $W/smoke_h.log | cut -c1-200)"; exit 1; }
log "SMOKE_OK (DoRA) $(grep -aoE '[0-9.]+s/샘플' $W/smoke_h.log | tail -1)"

for cfg in "base --r 8 --alpha 16 --lr 1e-4" "dora --r 8 --alpha 16 --lr 1e-4 --dora" "r16 --r 16 --alpha 32 --lr 1e-4" "lr2 --r 8 --alpha 16 --lr 2e-4"; do
  set -- $cfg; name=$1; shift
  # 기준·DoRA 는 항상 돈다(B안). r16·lr2 는 STOP_AFTER 가 없을 때만(A안 — 충전이 확인되면 사람이 STOP_AFTER 를 지운다)
  if [ "$name" != base ] && [ "$name" != dora ] && [ -f $W/STOP_AFTER ]; then log "STOP_AFTER - $name 이후 건너뜀"; break; fi
  log "$name 시작"
  python colab_lora_train.py $BASE "$@" --tag 9b_$name > $W/h_$name.log 2>&1 \
    && log "$name 끝 | $(grep -a '최종 val acc\|어려운' $W/h_$name.log | tail -2 | tr '\n' ' ')" \
    || log "$name 실패: $(grep -a -m1 -E 'Error' $W/h_$name.log | cut -c1-200)"
  # 볼륨이 없는 파드다. 예측·meta 는 설정마다 묶어 둔다(어댑터 가중치는 크기 때문에 뺀다).
  tar czf $W/h_artifacts.tgz -C $OUT --exclude='*.safetensors' . 2>/dev/null
done
log "H_DONE"
