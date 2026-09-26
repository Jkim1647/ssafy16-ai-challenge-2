#!/usr/bin/env bash
# R3(27B LoRA + read+tiles) 학습이 끝나기를 기다렸다가 H408 채점을 잇는다.
#
# 왜 H408 인가. R3 는 train 6,714 중 5,976 을 학습해 남은 홀드아웃이 val667 뿐인데,
# 오늘 27B LoRA 에서 **val667 이 파인튜닝 모델을 과대평가한다**는 것이 확인됐다
# (val667 1위 0.9685 인데 Public 0.96187 로 꼴찌, 397B 단독보다 17문항 아래).
# dev 기반 H408 은 학습 출처가 달라 이 편향을 받지 않는다.
#
# SCORE_TEST=1 이면 같은 모델 로드에서 test 6,714건까지 채점한다(제출 후보 생성).
# 비용의 대부분이 모델 로드라 H408 과 붙여 돌리는 것이 싸다. test 이미지가 파드에 있어야 한다.
#
# **볼륨이 없는 파드다.** 회수 전에는 정지하지 않는다.
set -uo pipefail
W=/workspace
L=$W/chainR3.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$L"; }

log "R3_DONE 대기 시작"
for _ in $(seq 1 480); do
  grep -q R3_DONE "$W/out_27b/STATUS" 2>/dev/null && break
  sleep 30
done
grep -q R3_DONE "$W/out_27b/STATUS" 2>/dev/null || { log "학습이 끝나지 않았다 - 중단"; exit 1; }
log "학습 완료 확인"

if [ -n "${SCORE_TEST:-}" ]; then
  # 업로드가 학습과 동시에 돌고 있을 수 있다. 다 올라올 때까지 최대 30분 기다린다.
  for _ in $(seq 1 60); do
    [ "$(ls $W/data/test 2>/dev/null | wc -l)" -ge 6714 ] && break
    sleep 30
  done
  log "test 이미지 $(ls $W/data/test 2>/dev/null | wc -l)/6714"
fi

A=$W/out_27b/R3_27b_tiles/adapter
[ -d "$A" ] || { log "adapter 없음: $A - 중단"; exit 1; }

MODEL=Qwen/Qwen3.6-27B ADAPTER=$A OUT=$W/out_27b TAG=R3_h408 SCORE_TEST=${SCORE_TEST:-} bash $W/run_h408_eval.sh 2>&1 | tee -a "$L"
log "CHAIN_R3_DONE"
