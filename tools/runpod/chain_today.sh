#!/usr/bin/env bash
# 2026-09-24 아침 실행 — vLLM 설치 → 35B 받기 → 스크리닝 5설정 → shift TTA.
#
# 어젯밤 실패에서 배운 것
#   맨 `pip install vllm` 은 **CUDA 12.9+ 용 torch** 를 끌어와 드라이버 12.8 파드에서 죽는다
#   ("The NVIDIA driver on your system is too old (found version 12080)").
#   그래서 **torch 를 고정해 pip 이 그와 호환되는 vllm 을 고르게** 한다. 버전을 추측하지 않는 방법이다.
#   이 템플릿은 runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 (torch 2.8.0 / CUDA 12.8.1)다.
#
#   설치 직후 **작은 스모크로 엔진이 실제로 뜨는지 먼저 본다.** 어젯밤에는 이걸 안 해서
#   5설정이 전부 죽은 뒤에야 알았고 파드가 3시간 놀았다.
#
# 끝나면 스스로 표시를 남긴다(TODAY_DONE). 회수는 사람이 한다 — 이 파드에는 볼륨이 없어
# 정지하면 /workspace 가 사라진다.
set -uo pipefail
W=/workspace
L=$W/chainToday.log
export HF_HOME=$W/hf VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$L"; }

log "시작. 드라이버/토치 확인"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 | tee -a "$L"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" 2>&1 | tee -a "$L"

# ---- 0) 추론기가 쓰는 파이썬 패키지 ----
# 이 템플릿은 PyTorch 개발 환경이라 pandas·pillow 가 없다. 2026-09-24 아침 첫 시도에서
# 스모크가 "No module named 'pandas'" 로 죽었다(다행히 본실행 전에 걸렸다).
python -c "import pandas, PIL" 2>/dev/null || {
  log "pandas/pillow 설치"
  pip install --break-system-packages -q pandas pillow 2>&1 | tail -3 | tee -a "$L"
}
python -c "import pandas, PIL; print('pandas', pandas.__version__)" 2>&1 | tee -a "$L"   || { log "pandas/pillow 사용 불가 - 중단"; exit 1; }

# ---- 1) vLLM — 드라이버가 감당하는 CUDA 를 보고 갈라진다 ----
#
# 2026-09-23 밤 H200 파드: 드라이버가 CUDA 12.8 까지만 지원(found version 12080)인데
#   맨 `pip install vllm` 이 CUDA 12.9+ 용 torch 를 끌어와 죽었다.
# 2026-09-24 아침 이 파드: 드라이버가 CUDA 13.2 를 지원하는데 torch 를 2.8.0 에 묶었더니
#   vLLM 이 0.11.0 으로 내려가 Qwen3_5MoeForConditionalGeneration 을 모른다고 죽었다.
#
# 고정도 방치도 답이 아니고 **드라이버에 맞춰야 한다**.
DRV=$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9.]*' | grep -o '[0-9.]*$')
log "드라이버 지원 CUDA ${DRV:-알수없음}"
NEED_NEW=$(python -c "
import sys
try:
    p=('${DRV:-0}'.split('.')+['0'])[:2]
    print('1' if (int(p[0]),int(p[1]))>=(12,9) else '0')
except Exception:
    print('0')")
if [ "$NEED_NEW" = "1" ]; then
  log "최신 vllm 설치(드라이버가 CUDA 12.9+ 지원)"
  pip install --break-system-packages -q -U vllm 2>&1 | tail -8 | tee -a "$L"
else
  log "vllm 설치, torch 2.8.0 고정(드라이버가 CUDA 12.8 까지만 지원)"
  pip install --break-system-packages -q vllm "torch==2.8.0" 2>&1 | tail -8 | tee -a "$L"
fi
python -c "import vllm, torch; print('VLLM', vllm.__version__, '| torch', torch.__version__)" 2>&1 | tee -a "$L" \
  || { log "vllm 사용 불가 - 중단"; exit 1; }

# ---- 2) 모델 ----
log "35B 다운로드"
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.6-35B-A3B', max_workers=16))" \
  2>&1 | tail -2 | tee -a "$L" || { log "다운로드 실패 - 중단"; exit 1; }

# ---- 3) 엔진 스모크 (8건) ----
for x in train dev test; do ln -sfn "$W/data/$x" "$W/screendata/$x" 2>/dev/null; done
log "엔진 스모크 8건"
cd $W/tools || exit 1
timeout 1800 python colab_vllm_infer.py --model Qwen/Qwen3.6-35B-A3B --data "$W/screendata" --out "$W/out_smoke" \
  --tp 1 --prompt read --view tiles2x2 --tile-scale 2 --max-model-len 16384 --max-num-seqs 64 --gpu-mem 0.90 \
  --split dev --tag-suffix SMOKE --limit 8 2>&1 | tail -6 | tee -a "$L"
[ -s "$W/out_smoke"/*SMOKE*predictions.jsonl ] 2>/dev/null || {
  log "스모크가 예측을 못 냈다 - 중단(본실행에 돈을 쓰지 않는다)"; exit 1; }
log "SMOKE_OK $(cat "$W/out_smoke"/*SMOKE*predictions.jsonl | wc -l)건"

# ---- 4) 스크리닝 ----
log "스크리닝 시작 ($(( $(wc -l < "$W/screendata/dev.csv") - 1 ))건)"
EVAL=$W/screendata OUT=$W/out_screen CONFIGS="${CONFIGS:-base r1v3 r1v2 fs16 r2v3}" \
  bash $W/run_screen_35b.sh 2>&1 | tail -30 | tee -a "$L"

# ---- 5) shift TTA ----
if [ -f "$W/shift_data/ids.json" ]; then
  log "shift TTA 시작"
  IDS=$W/shift_data/ids.json DATA=$W/data bash $W/run_shift_tta_35b.sh 2>&1 | tail -20 | tee -a "$L"
fi

# ---- 6) 묶기 ----
tar czf "$W/today_artifacts.tgz" -C "$W" --exclude='*.safetensors' --exclude='*.tgz' \
  out_screen out_shift_35b out_smoke 2>/dev/null && log "묶음 $(du -h "$W/today_artifacts.tgz" | cut -f1)"
log "TODAY_DONE"
