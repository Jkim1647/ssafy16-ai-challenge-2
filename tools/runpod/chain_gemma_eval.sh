#!/usr/bin/env bash
# Gemma LoRA 학습이 끝나기를 기다렸다가 H408 채점 + test 6,714건 추론을 잇는다.
#
# 한 번의 모델 로드로 val 재로드 확인 -> H408 -> test 를 모두 낸다. 비용의 대부분이
# 로드라서 나눠 돌리면 그만큼 더 낸다.
#
# **파드에 볼륨이 없다.** 정지하면 /workspace 가 통째로 사라지므로 회수 전에는 정지하지 않는다.
set -uo pipefail
W=/workspace
L=$W/chainGemma.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$L"; }

log "GEMMA_DONE 대기 시작"
for _ in $(seq 1 480); do
  grep -q GEMMA_DONE "$W/out_gemma/STATUS" 2>/dev/null && break
  sleep 30
done
grep -q GEMMA_DONE "$W/out_gemma/STATUS" 2>/dev/null || { log "학습이 끝나지 않았다 - 중단"; exit 1; }
log "학습 완료 확인"

A=$W/out_gemma/gemma4_31b_lora/adapter
[ -d "$A" ] || { log "adapter 없음: $A - 중단"; exit 1; }

MODEL=google/gemma-4-31B-it ADAPTER=$A OUT=$W/out_gemma TAG=gemma_h408_test \
SCORE_TEST=1 bash $W/run_h408_eval.sh 2>&1 | tee -a "$L"
log "CHAIN_GEMMA_DONE"
