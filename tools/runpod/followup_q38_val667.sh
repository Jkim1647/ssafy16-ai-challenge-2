#!/usr/bin/env bash
# 2026-09-24 — 트랙 A 후속: chain_q38.sh 가 끝나면(Q_DONE) 같은 조건으로 Qwen3.8-27B val667 을 돌린다.
# 사용자 지시: 제출 조합은 H408 v2 와 val667 을 합산해 고른다. 다른 네 구성원은 val667 예측이 이미 있다.
# val667 이미지(667장)는 /workspace/imgup/v0*.tar 로 올려 둔다.
set -uo pipefail
W=/workspace
L=$W/chainQ.log
export HF_HOME=$W/hf VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$L"; }

for f in $W/imgup/v0*.tar; do tar xf "$f" -C $W/data; done
log "val667 이미지 $(ls $W/data/train | wc -l)장 준비"
until grep -q "Q_DONE" "$L"; do sleep 30; done
cd $W/tools || exit 1
C="--model Qwen/Qwen3.8-27B --tp 1 --prompt read --view tiles2x2 --tile-scale 2 --max-model-len 16384 --max-num-seqs 32 --gpu-mem 0.90"
log "val667 시작"
python colab_vllm_infer.py $C --data $W/data --split val667 --split-json $W/data/val667_ids.json --out $W/out_q38 \
  > $W/q38_val667.log 2>&1 && log "val667 끝" || log "val667 실패"
log "V_DONE"
