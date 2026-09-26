#!/usr/bin/env bash
# 본 큐(스크리닝 -> shift)가 끝나면 r1v3 만 다시 돌린다.
# r1v3 는 출력 태그 맵에 v3 가 빠져 KeyError 로 9초 만에 죽었다(코드 수정 완료).
# GPU 가 하나라 본 큐와 겹치지 않게 ALL_DONE 을 기다린다.
set -uo pipefail
W=/workspace
until grep -q ALL_DONE "$W/runAll.log" 2>/dev/null; do sleep 30; done
echo "$(date -u +%FT%TZ) 본 큐 종료 확인 -> r1v3 재실행" >> "$W/out_screen/STATUS"
EVAL=$W/screendata OUT=$W/out_screen CONFIGS="r1v3" bash $W/run_screen_35b.sh
echo "$(date -u +%FT%TZ) R1V3_FOLLOWUP_DONE" >> "$W/out_screen/STATUS"
