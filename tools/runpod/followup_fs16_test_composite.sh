#!/usr/bin/env bash
# "COMPOSITE 문항은 fs16" 규칙(검증 1,908건 +6, 스크리닝 +3)을 test 에 적용하기 위해
# test COMPOSITE 2,197건만 fs16 으로 돈다. 파드 A 에서 r1v3 검증 뒤에 이어진다.
set -uo pipefail
W=/workspace
until grep -q R1V3_VERIFY_DONE $W/out_verify/STATUS 2>/dev/null; do sleep 30; done
D=$W/tdata_c; for x in train dev test; do ln -sfn $W/data/$x $D/$x; done; cp -f $W/data/train.csv $D/train.csv
EVAL=$D OUT=$W/out_test_c CONFIGS="fs16" bash $W/run_screen_35b.sh
echo "$(date -u +%FT%TZ) FS16_TESTC_DONE" >> $W/out_test_c/STATUS
