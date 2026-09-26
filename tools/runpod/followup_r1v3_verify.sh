#!/usr/bin/env bash
# r1v3 가 H408 에서 1위(368, base 363)로 올라와 검증 2,908건에 추가한다. base 는 파드 B 가 같은 vLLM 0.30.0 으로 돌렸다.
set -uo pipefail
W=/workspace
until grep -q TEST_DONE $W/out_test/STATUS 2>/dev/null; do sleep 30; done
D=$W/vdata; for x in train dev test; do ln -sfn $W/data/$x $D/$x; done; cp -f $W/data/train.csv $D/train.csv
EVAL=$D OUT=$W/out_verify CONFIGS="r1v3" bash $W/run_screen_35b.sh
echo "$(date -u +%FT%TZ) R1V3_VERIFY_DONE" >> $W/out_verify/STATUS
