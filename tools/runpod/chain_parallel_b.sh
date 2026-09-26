#!/usr/bin/env bash
# 2026-09-24 병렬 파드 B — 설치 → 35B 받기 → 스모크 → 검증(base·r2v3 2,908건, fs16 COMPOSITE 1,908건).
#
# 파드 A 가 스크리닝 꼬리(shift·r1v3·dev179)를 도는 동안 검증을 먼저 끝내 약 1시간을 줄인다(사용자 결정).
# 설치 절차는 chain_today.sh 와 같다 — 오늘 아침 세 번 막혔던 것(pandas 없음, 드라이버에 안 맞는 vLLM,
# sm_120 FlashInfer MoE)을 그대로 피한다. 스모크가 예측을 못 내면 본실행에 돈을 쓰지 않고 멈춘다.
#
# fs16 비교의 base 는 2,908건 base 실행의 COMPOSITE 부분을 쓴다(vdata_c ⊂ vdata, 같은 파드·같은 조건).
set -uo pipefail
W=/workspace
L=$W/chainB.log
export HF_HOME=$W/hf VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_MOE_FP8=0 VLLM_USE_FLASHINFER_MOE_FP4=0
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$L"; }

log "시작"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 | tee -a "$L"
python -c "import pandas, PIL" 2>/dev/null || pip install --break-system-packages -q pandas pillow 2>&1 | tail -2 | tee -a "$L"

# 모델 다운로드를 설치와 동시에 시작한다(서로 독립). huggingface_hub 는 템플릿에 있다.
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
tail -2 $W/dl.log | tee -a "$L"
grep -q "DL_EXIT 0" $W/dl.log || { log "다운로드 실패 - 중단"; exit 1; }

for D in $W/vdata $W/vdata_c; do
  for x in train dev test; do ln -sfn "$W/data/$x" "$D/$x"; done
  cp -f "$W/data/train.csv" "$D/train.csv"
done

log "엔진 스모크 8건"
cd $W/tools || exit 1
timeout 1800 python colab_vllm_infer.py --model Qwen/Qwen3.6-35B-A3B --data "$W/vdata" --out "$W/out_smoke" \
  --tp 1 --prompt read --view tiles2x2 --tile-scale 2 --max-model-len 16384 --max-num-seqs 64 --gpu-mem 0.90 \
  --moe-backend triton --split dev --tag-suffix SMOKE --limit 8 2>&1 | tail -4 | tee -a "$L"
[ -s "$W/out_smoke"/*SMOKE*predictions.jsonl ] 2>/dev/null || { log "스모크 실패 - 중단(본실행에 돈을 쓰지 않는다)"; exit 1; }
log "SMOKE_OK"

EVAL=$W/vdata   OUT=$W/out_verify CONFIGS="base r2v3" bash $W/run_screen_35b.sh 2>&1 | tail -5 | tee -a "$L"
EVAL=$W/vdata_c OUT=$W/out_verify CONFIGS="fs16"      bash $W/run_screen_35b.sh 2>&1 | tail -5 | tee -a "$L"
echo "$(date -u +%FT%TZ) VERIFY_DONE" >> "$W/out_verify/STATUS"
log "B_DONE"
