#!/usr/bin/env bash
# 2026-09-24 오후 — 35B 추론 모드(생각 후 답하기) 를 H408 에서 시험한다(사용자 승인).
#
# 근거: H408 v2(사용자 정답 검토) 기준 T6 실제 오답 19개 대부분이 판독이 아니라 "질문이 묻는 대상·필드를
# 잘못 고른" 선택 오류였다. 지금 파이프라인은 모두 enable_thinking=False 로 바로 답한다.
# 같은 파드·같은 조건에서 base(생각 없음)와 think1024 를 둘 다 돌려 비교를 닫는다.
set -uo pipefail
W=/workspace
L=$W/chainT.log
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

D=$W/h408data
for x in train dev test; do ln -sfn "$W/data/$x" "$D/$x"; done
cd $W/tools || exit 1
C="--model Qwen/Qwen3.6-35B-A3B --data $D --tp 1 --prompt read --view tiles2x2 --tile-scale 2 \
   --max-model-len 16384 --max-num-seqs 64 --gpu-mem 0.90 --moe-backend triton --split dev"

log "스모크: think 4건"
timeout 1800 python colab_vllm_infer.py $C --out $W/out_smoke --think 1024 --tag-suffix SMOKE --limit 4 2>&1 | tail -4 | tee -a "$L"
[ -s "$W/out_smoke"/*SMOKE*predictions.jsonl ] 2>/dev/null || { log "스모크 실패 - 중단(본실행에 돈을 쓰지 않는다)"; exit 1; }
python - <<'PY' 2>&1 | tee -a "$L"
import json, glob
r = [json.loads(l) for l in open(glob.glob('/workspace/out_smoke/*SMOKE*predictions.jsonl')[0])]
print('SMOKE think_tokens', [x.get('think_tokens') for x in r], '| reasoning 앞부분:', (r[0].get('reasoning') or '')[:200].replace('\n', ' '))
PY
log "SMOKE_OK"

for cfg in base think1024; do
  X=""; [ "$cfg" = think1024 ] && X="--think 1024"
  log "$cfg 시작"
  python colab_vllm_infer.py $C --out $W/out_think $X --tag-suffix H408 > $W/$cfg.log 2>&1 && log "$cfg 끝" || log "$cfg 실패"
done
log "T_DONE"
