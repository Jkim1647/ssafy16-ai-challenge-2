#!/usr/bin/env bash
# 2026-09-24 저녁 파드 F — 7번 R3·Gemma 보기 회전 TTA (사용자 승인 "1~10번까지 진행해", "제출까지").
#
# 근거: Discussion 742479 — 보기 순서를 바꿨을 때 답이 흔들린 고확신 문항의 오답률 26.7%, 안 흔들림 2.4%.
# R3·Gemma 는 보기 셔플 없이 학습해 위치 편향이 남아 있을 수 있다. 보기를 1·2·3칸 순환해 다시 채점하고
# 로그확률을 원래 a~d 순서로 되돌려 저장한다(colab_lora_train.py --eval-shifts). 회전 0 은 기존 예측을 쓴다.
#   H408 408 × 3 + test 저마진 600(T6 1·2위 확률 차 하위, lowmargin600_ids.json) × 3.
# 재로드 확인으로 val667 앞 50건을 먼저 채점한다 — 기존 값(R3 0.98, Gemma 0.96)과 같아야 한다.
set -uo pipefail
W=/workspace
L=$W/chainF.log
export HF_HOME=$W/hf
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$L"; }
GEMMA_REV=842da3794eaa0b77d5f08bae87a17459d91ff475     # 학습 때 쓴 커밋(RUN_META.json)

log "시작"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 | tee -a "$L"
python -c "import huggingface_hub" 2>/dev/null || pip install --break-system-packages -q huggingface_hub 2>&1 | tail -1
( python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.6-27B', max_workers=16)" \
    > $W/dl27.log 2>&1; echo "DL_EXIT $?" >> $W/dl27.log ) &
D27=$!
( python -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-4-31B-it', revision='$GEMMA_REV', max_workers=16)" \
    > $W/dlg.log 2>&1; echo "DL_EXIT $?" >> $W/dlg.log ) &
DG=$!
# 파드 B(09-24) R3·Gemma 재채점과 같은 transformers/peft 버전. val50 재로드 값으로 환경을 확인한다
pip install --break-system-packages -q "transformers==5.17.0" "peft==0.21.0" accelerate pandas pillow 2>&1 | tail -2 | tee -a "$L"
python -c "import torch, transformers, peft; print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__)" 2>&1 | tee -a "$L"

for _ in $(seq 1 120); do [ -f $W/IMAGES_READY ] && break; sleep 30; done
for x in train dev test; do log "이미지 $x $(ls $W/data/$x 2>/dev/null | wc -l)장"; done
[ "$(ls $W/data/test | wc -l)" -ge 600 ] && [ "$(ls $W/data/dev | wc -l)" -ge 408 ] || { log "이미지 부족 - 중단"; exit 1; }

mkdir -p $W/adapters
for t in R3_artifacts gemma_artifacts; do tar xzf $W/imgup/$t.tgz -C $W/adapters 2>&1 | tail -1; done
ls $W/adapters/*/adapter/adapter_model.safetensors 2>&1 | tee -a "$L"

cd $W/tools || exit 1
run(){  # $1 모델 $2 adapter $3 tag $4 revision(선택)
  log "$3 시작"
  python colab_lora_train.py --model "$1" ${4:+--revision $4} --data "$W/data" \
    --val-ids "$W/data/val667_ids.json" --out "$W/out_f" --view tiles2x2 \
    --exclude-csv "$W/data/train_exclusions_v2.csv" \
    --eval-adapter "$2" --hard-gold "$W/data/hard_gold_h408.json" \
    --limit-val 50 --skip-test --reload-score-test --eval-shifts 1,2,3 --shifts-only \
    --shift-test-ids "$W/lowmargin600_ids.json" --tag "$3" > $W/$3.log 2>&1
  log "$3 끝 rc=$? $(grep -ao '재로드 val acc [0-9.]*' $W/$3.log | tail -1) | $(grep -ao 'shift[0-9] 어려운 평가셋 acc [0-9.]*' $W/$3.log | tr '\n' ' ')"
  tar czf $W/f_artifacts.tgz -C $W/out_f . 2>/dev/null
}
wait $D27; grep -q "DL_EXIT 0" $W/dl27.log || { log "27B 다운로드 실패 - 중단"; exit 1; }
run Qwen/Qwen3.6-27B $W/adapters/R3_27b_tiles/adapter R3_shift
wait $DG; grep -q "DL_EXIT 0" $W/dlg.log || { log "Gemma 다운로드 실패 - 중단"; exit 1; }
run google/gemma-4-31B-it $W/adapters/gemma4_31b_lora/adapter Gemma_shift $GEMMA_REV
log "F_DONE"
