#!/usr/bin/env bash
# 스크리닝 5설정을 dev179 **수정본 질문**으로 다시 돈다 → H229 와 합쳐 H408 로 채점하기 위해.
#
# 왜. 스크리닝 dev.csv 의 179행이 원본 질문이었다(내 데이터 버그, 커밋 1ce3b67). 그래서
# 지금 스크리닝은 dev 를 H229 로만 채점한다. 179건을 다시 돌리면 dev 표본이 229 → 408 로
# 커져 1문항 무게가 0.44%p → 0.25%p 가 된다. 179건 × 5설정이라 모델 로드 포함 약 20분.
#
# GPU 가 하나라 r1v3 재실행(R1V3_FOLLOWUP_DONE)이 끝난 뒤에 돈다. test 추론(followup_test.sh)은
# 사람이 TEST_CFG 를 넣어야 시작하므로, 이 스크립트가 끝난 뒤 넣으면 겹치지 않는다.
set -uo pipefail
W=/workspace
D=$W/d179data
until grep -q R1V3_FOLLOWUP_DONE "$W/out_screen/STATUS" 2>/dev/null; do sleep 30; done
for x in train dev test; do ln -sfn "$W/data/$x" "$D/$x"; done
cp -f "$W/data/train.csv" "$D/train.csv"      # fs16 예시 풀
EVAL=$D OUT=$W/out_d179 CONFIGS="${CONFIGS:-base r1v2 r1v3 fs16 r2v3}" bash $W/run_screen_35b.sh
echo "$(date -u +%FT%TZ) D179_DONE" >> "$W/out_d179/STATUS"
