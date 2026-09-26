#!/usr/bin/env bash
# 2026-09-24 저녁 파드 G — 9번 Qwen3.5-27B LoRA + 보기 셔플 (사용자 승인 "1~10번까지 진행해", "제출까지").
#
# 근거: 1위 팀(다른 팀) — Qwen3.5-27B LoRA 단독 Public 0.97169, LoRA 학습 때 보기 셔플 +12/1000.
# 우리 R3 는 Qwen3.6-27B·셔플 없음. 두 단서를 한 번에 넣는다. 나머지는 R3 와 같다
# (tiles2x2, r8/α16, lr 1e-4, 1 epoch, fit 5,976 / val667, 결함 71건 제외) — val667 로 R3(646)와 바로 비교된다.
# 학습이 끝나면 같은 프로세스에서 H408(408) + test 6,714 까지 채점한다(제출 후보).
set -uo pipefail
W=/workspace
L=$W/chainG.log
OUT=$W/out_g
MODEL=Qwen/Qwen3.5-27B
export HF_HOME=$W/hf
mkdir -p $OUT
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$L"; }

log "시작"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 | tee -a "$L"
python -c "import huggingface_hub" 2>/dev/null || pip install --break-system-packages -q huggingface_hub 2>&1 | tail -1
( python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL', max_workers=16)" \
    > $W/dlg.log 2>&1; echo "DL_EXIT $?" >> $W/dlg.log ) &
DL=$!
pip install --break-system-packages -q "transformers==5.17.0" "peft==0.21.0" accelerate pandas pillow bitsandbytes 2>&1 | tail -2 | tee -a "$L"
# 선형 어텐션 층의 빠른 커널. 없으면 torch 기본 구현으로 느리게 돈다(09-23 R3 로그의 fallback 경고).
pip install --break-system-packages -q flash-linear-attention 2>&1 | tail -1 | tee -a "$L"
python -c "import torch, transformers, peft; print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__)" 2>&1 | tee -a "$L"
python -c "import fla; print('fla', fla.__version__)" 2>&1 | tail -1 | tee -a "$L"

for _ in $(seq 1 120); do [ -f $W/IMAGES_READY ] && break; sleep 30; done
for x in train dev test; do log "이미지 $x $(ls $W/data/$x 2>/dev/null | wc -l)장"; done
[ "$(ls $W/data/train | wc -l)" -ge 6714 ] && [ "$(ls $W/data/test | wc -l)" -ge 6714 ] && [ "$(ls $W/data/dev | wc -l)" -ge 408 ] \
  || { log "이미지 부족 - 중단"; exit 1; }
wait $DL; grep -q "DL_EXIT 0" $W/dlg.log || { log "다운로드 실패 - 중단"; exit 1; }

cd $W/tools || exit 1
BASE="--model $MODEL --data $W/data --out $OUT --view tiles2x2 --aug-shift \
  --val-ids $W/data/val667_ids.json --exclude-csv $W/data/train_exclusions_v2.csv \
  --r 8 --alpha 16 --lr 1e-4 --epochs 1 --grad-accum 16 --seed 1"

# 스모크: 학습 48 / 평가 40, 저장까지. 실패하면 fla 를 빼고 한 번 더 — 그래도 실패면 본학습에 돈을 쓰지 않는다.
smoke(){ timeout 3600 python colab_lora_train.py $BASE --limit-train 48 --limit-val 40 --skip-test --tag smoke35_27b > $W/smoke.log 2>&1; }
log "스모크 시작"
if ! smoke; then
  log "스모크 실패(fla 포함) - fla 제거 후 재시도: $(grep -a -m1 -E 'Error|error' $W/smoke.log | cut -c1-200)"
  pip uninstall --break-system-packages -y -q flash-linear-attention fla-core 2>&1 | tail -1
  smoke || { log "스모크 실패 - 중단: $(grep -a -m1 -E 'Error|error' $W/smoke.log | cut -c1-200)"; exit 1; }
fi
log "SMOKE_OK $(grep -ao 'LoRA 모듈 [0-9]*개' $W/smoke.log | head -1) | $(grep -aoE '[0-9.]+s/(샘플|it)' $W/smoke.log | tail -1)"

log "본학습 시작"
timeout 64800 python colab_lora_train.py $BASE --hard-gold $W/data/hard_gold_h408.json --tag Q35_27b_shuf > $W/train_g.log 2>&1
rc=$?
log "본학습 종료 rc=$rc | $(grep -a 'val acc\|어려운' $W/train_g.log | tail -2 | tr '\n' ' ')"
tar czf $W/g_artifacts.tgz -C $OUT Q35_27b_shuf 2>/dev/null && log "묶음 $(du -h $W/g_artifacts.tgz | cut -f1)"
log "G_DONE"
