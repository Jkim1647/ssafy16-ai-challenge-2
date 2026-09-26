#!/usr/bin/env bash
# 파드 E 보정 실행 — 6번 해상도 mp1536·mp2304 를 올바른 단위로 다시 돈다(2026-09-24).
#
# 버그: --min-pixels/--max-pixels 는 **픽셀 수**인데 chain_e.sh·run_screen_35b.sh 는 1536·2048 처럼 비주얼 토큰 수를
# 넣었다 → 이미지가 약 40x40 으로 줄어 train1000 이 962 -> 600 으로 무너졌다(결과 폐기).
# Qwen3.6 은 patch 16 x merge 2 라 토큰 1개 = 32x32 = 1,024 픽셀. test 이미지 중앙값은 약 69만 픽셀(약 675 토큰)이다.
# chain_e.sh 가 도는 중에는 그 파일을 고칠 수 없어(bash 가 깨진다) 따로 두고, t3 가 끝나면 GPU 를 이어받는다.
set -uo pipefail
W=/workspace
L=$W/chainE.log
export HF_HOME=$W/hf VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_MOE_FP8=0 VLLM_USE_FLASHINFER_MOE_FP4=0
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$L"; }
until grep -qE "scr t3 (끝|실패)" $L 2>/dev/null; do sleep 20; done
cd $W/tools || exit 1
C="--model Qwen/Qwen3.6-35B-A3B --tp 1 --prompt read --view tiles2x2 --tile-scale 2 --max-model-len 16384 --gpu-mem 0.90 --moe-backend triton"
S=$W/screendata
for cfg in "mp1536px 1572864 2097152" "mp2304px 2359296 3145728"; do
  set -- $cfg
  log "scr $1 시작 (min $2 / max $3 픽셀)"
  python colab_vllm_infer.py $C --data $S --split dev --max-num-seqs 64 --min-pixels $2 --max-pixels $3 \
    --out $W/out_e --tag-suffix "SCR_$1" > $W/e_scr_$1.log 2>&1 \
    && log "scr $1 끝 $(cat $W/out_e/*SCR_$1*predictions.jsonl 2>/dev/null | wc -l)건" || log "scr $1 실패"
  tar czf $W/e_artifacts.tgz -C $W/out_e . 2>/dev/null
done
log "E2_DONE"
