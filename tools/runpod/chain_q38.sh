#!/usr/bin/env bash
# 2026-09-24 저녁 — Qwen3.8-27B 를 새 앙상블 구성원으로 시험한다(사용자 결정, 트랙 A).
#
# 근거: 1위 팀(다른 팀) 글 — Qwen3.8-27B 는 단독 Public 0.96772 로 약했지만 Qwen3.5-27B 와 50:50 으로
# 섞으면 0.97249, Gemma 까지 1/3 씩이면 0.97497. "다른 계열이 서로 다른 오답을 보완"한다는 주장.
# 우리는 35B 와 같은 read+tiles 조건으로 제로샷 추론한다. H408 먼저, 이어서 test.
# H408 결과가 나쁘면 사람이 test 를 중간에 끊는다(test 는 약 1시간, 약 $2).
set -uo pipefail
W=/workspace
L=$W/chainQ.log
M=Qwen/Qwen3.8-27B
export HF_HOME=$W/hf VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$L"; }

log "시작"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 | tee -a "$L"
python -c "import pandas, PIL" 2>/dev/null || pip install --break-system-packages -q pandas pillow 2>&1 | tail -2 | tee -a "$L"
python -c "import huggingface_hub" 2>/dev/null || pip install --break-system-packages -q huggingface_hub 2>&1 | tail -1
( python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$M', max_workers=16))" \
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

for D in $W/h408data $W/data; do for x in train dev test; do [ -e "$D/$x" ] || ln -sfn "$W/data/$x" "$D/$x"; done; done
cd $W/tools || exit 1
C="--model $M --tp 1 --prompt read --view tiles2x2 --tile-scale 2 --max-model-len 16384 --max-num-seqs 32 --gpu-mem 0.90"

log "스모크 4건"
timeout 1800 python colab_vllm_infer.py $C --data $W/h408data --split dev --out $W/out_smoke --tag-suffix SMOKE --limit 4 2>&1 | tail -4 | tee -a "$L"
[ -s "$W/out_smoke"/*SMOKE*predictions.jsonl ] 2>/dev/null || { log "스모크 실패 - 중단(본실행에 돈을 쓰지 않는다)"; exit 1; }
log "SMOKE_OK"

log "H408 시작"
python colab_vllm_infer.py $C --data $W/h408data --split dev --out $W/out_q38 --tag-suffix H408 > $W/q38_h408.log 2>&1 && log "H408 끝" || log "H408 실패"
log "test 시작"
python colab_vllm_infer.py $C --data $W/data --split test --out $W/out_q38 --tag-suffix TEST > $W/q38_test.log 2>&1 && log "test 끝" || log "test 실패"
log "Q_DONE"
