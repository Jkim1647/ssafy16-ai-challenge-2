#!/usr/bin/env bash
# 새벽 무인 큐 — R3 test 추론이 끝난 뒤 같은 파드에서 35B 스크리닝과 shift TTA 를 잇는다.
#
# 왜 이 파드인가
#   Qwen3.6-35B-A3B 가 이미 HF 캐시에 있고(다운로드 70GB 절약), train·dev·test 이미지가
#   전부 올라와 있다. 파드에 볼륨이 없어 정지하면 다 사라지므로, 살아 있는 동안 다 쓴다.
#
# 순서와 이유
#   1) R3 test 대기          GPU 를 놓고 다투지 않는다
#   2) vLLM 설치             이 파드는 LoRA 학습용이라 vllm 이 없다. 실패하면 거기서 멈춘다
#   3) 스크리닝 5설정        base / r1v2 / r1v3 / fs16 / r2v3 (2,408건 = train 2,000 + dev 408)
#      t3(타일 3배)는 뺐다 — 판독 실패가 200건 중 32건(16%)뿐이라 사정권이 가장 좁은데
#      32K context 로 가장 느리다. 새벽 예산 안에서는 우선순위가 낮다
#   4) shift TTA             마진 하위 5% 638건 x shift 0~3. 약 21분
#   5) 묶기                  파드 정지 전에 한 번에 회수할 수 있게
#
# 각 단계는 실패해도 다음으로 넘어간다(set -uo pipefail, || continue). 하나가 깨져도
# 나머지 결과는 건진다.
#
#   cd /workspace && nohup bash chain_overnight.sh > chainNight.out 2>&1 &
set -uo pipefail
W=/workspace
L=$W/chainNight.log
export HF_HOME=$W/hf VLLM_USE_FLASHINFER_SAMPLER=0
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$L"; }

# ---- 1) R3 test 대기 ----
log "R3 test(H408_DONE) 대기 시작"
for _ in $(seq 1 300); do
  grep -q H408_DONE "$W/out_27b/STATUS" 2>/dev/null && break
  sleep 30
done
grep -q H408_DONE "$W/out_27b/STATUS" 2>/dev/null || { log "R3 가 끝나지 않았다 - 중단"; exit 1; }
log "R3 완료 확인"

# ---- 2) vLLM ----
if ! python -c "import vllm" 2>/dev/null; then
  log "vllm 설치 시작"
  pip install --break-system-packages -q vllm 2>&1 | tail -5 | tee -a "$L"
fi
python -c "import vllm; print('VLLM', vllm.__version__)" 2>&1 | tee -a "$L" \
  || { log "vllm 사용 불가 - 중단"; exit 1; }

# ---- 3) 스크리닝 ----
if [ -f "$W/screendata/dev.csv" ]; then
  for x in train dev test; do ln -sfn "$W/data/$x" "$W/screendata/$x"; done
  log "스크리닝 시작 ($(( $(wc -l < "$W/screendata/dev.csv") - 1 ))건)"
  EVAL=$W/screendata OUT=$W/out_screen CONFIGS="base r1v2 r1v3 fs16 r2v3" \
    bash $W/run_screen_35b.sh 2>&1 | tail -40 | tee -a "$L"
  log "스크리닝 종료"
else
  log "screendata/dev.csv 없음 - 스크리닝 건너뜀"
fi

# ---- 4) shift TTA ----
if [ -f "$W/shift_data/ids.json" ]; then
  log "shift TTA 시작"
  IDS=$W/shift_data/ids.json DATA=$W/data bash $W/run_shift_tta_35b.sh 2>&1 | tail -30 | tee -a "$L"
  log "shift TTA 종료"
else
  log "shift_data/ids.json 없음 - 건너뜀"
fi

# ---- 5) 묶기 ----
tar czf "$W/night_artifacts.tgz" -C "$W" \
  --exclude='*.safetensors' --exclude='*.bin' --exclude='*.tgz' \
  out_screen out_shift_35b 2>/dev/null \
  && log "묶음 $(du -h "$W/night_artifacts.tgz" | cut -f1)"
log "NIGHT_DONE"
