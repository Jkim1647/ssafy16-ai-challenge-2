#!/usr/bin/env bash
# 2026-09-24 오후 — test COMPOSITE 2,197건을 fs16 으로 돈다(사용자 결정: H408 이 COMPOSITE 라우팅을 판별하지 못하니
# Public 으로 직접 확인한다). 결과로 "COMPOSITE 는 35B 계열(35B·r2v3·fs16) 평균" 후보를 만든다.
#
# 설치 절차는 chain_parallel_b.sh 와 같다(pandas → 드라이버 CUDA 에 맞춘 vLLM → 35B → 스모크 8건).
set -uo pipefail
W=/workspace
L=$W/chainC.log
export HF_HOME=$W/hf VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_MOE_FP8=0 VLLM_USE_FLASHINFER_MOE_FP4=0
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$L"; }

log "시작"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 | tee -a "$L"
python -c "import pandas, PIL" 2>/dev/null || pip install --break-system-packages -q pandas pillow 2>&1 | tail -2 | tee -a "$L"
python -c "import huggingface_hub" 2>/dev/null || pip install --break-system-packages -q huggingface_hub 2>&1 | tail -1
( python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.6-35B-A3B', max_workers=16))" \
    > $W/dl.log 2>&1; echo "DL_EXIT $?" >> $W/dl.log ) &
DLPID=$!

DRV=$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9.]*' | grep -o '[0-9.]*$')
log "드라이버 지원 CUDA ${DRV:-알수없음}"
NEED_NEW=$(python -c "
try:
    p=('${DRV:-0}'.split('.')+['0'])[:2]; print('1' if (int(p[0]),int(p[1]))>=(12,9) else '0')
except Exception: print('0')")
if [ "$NEED_NEW" = "1" ]; then
  pip install --break-system-packages -q -U vllm 2>&1 | tail -3 | tee -a "$L"
else
  pip install --break-system-packages -q vllm "torch==2.8.0" 2>&1 | tail -3 | tee -a "$L"
fi
python -c "import vllm, torch; print('VLLM', vllm.__version__, '| torch', torch.__version__)" 2>&1 | tee -a "$L" \
  || { log "vllm 사용 불가 - 중단"; exit 1; }
wait $DLPID
grep -q "DL_EXIT 0" $W/dl.log || { log "다운로드 실패 - 중단"; exit 1; }

D=$W/tdata_c
for x in train dev test; do ln -sfn "$W/data/$x" "$D/$x"; done
cp -f "$W/data/train.csv" "$D/train.csv"        # fs16 예시 풀

log "엔진 스모크 8건"
cd $W/tools || exit 1
timeout 1800 python colab_vllm_infer.py --model Qwen/Qwen3.6-35B-A3B --data "$D" --out "$W/out_smoke" \
  --tp 1 --prompt read --view tiles2x2 --tile-scale 2 --max-model-len 16384 --max-num-seqs 64 --gpu-mem 0.90 \
  --moe-backend triton --split dev --tag-suffix SMOKE --limit 8 2>&1 | tail -4 | tee -a "$L"
[ -s "$W/out_smoke"/*SMOKE*predictions.jsonl ] 2>/dev/null || { log "스모크 실패 - 중단(본실행에 돈을 쓰지 않는다)"; exit 1; }
log "SMOKE_OK"

EVAL=$D OUT=$W/out_test_c CONFIGS="fs16" bash $W/run_screen_35b.sh 2>&1 | tail -5 | tee -a "$L"
log "C_DONE"
