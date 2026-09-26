#!/usr/bin/env bash
# 스크리닝에서 나온 후보 규칙을 **아직 안 본 train 문항**으로 검증한다(2026-09-24 사용자 승인).
#
#   fs16  COMPOSITE 에서만 +3(고침 3 / 망가뜨림 0). 결과를 보고 고른 유형이라 1,229건으로는 확정 못 한다
#         → 스크리닝 밖 COMPOSITE 1,908건(vdata_c)에서 base 와 비교
#   r2v3  전체 +4(train +3, H229 +1). 유형 무관 설정이라
#         → COMPOSITE 1,908 + 그 외 무작위 1,000 = 2,908건(vdata)에서 base 와 비교
#
# base 는 2,908건에서 한 번 돌리고 fs16 비교에는 그중 COMPOSITE 1,908건을 쓴다.
# 예전 35B train6047 결과를 base 로 쓰지 않는 이유: 설정(r2both)과 환경이 달라 train1000 에서
# 6건이 다르고 정답 수가 4 차이 난다. 기대 효과(+10 안팎)에 비해 무시할 수 없다.
#
# GPU 가 하나라 dev179 재실행(D179_DONE) 뒤에 돈다. test 추론은 사람이 TEST_CFG 를 넣어야 시작한다.
set -uo pipefail
W=/workspace
until grep -q D179_DONE "$W/out_d179/STATUS" 2>/dev/null; do sleep 30; done
for D in $W/vdata $W/vdata_c; do
  for x in train dev test; do ln -sfn "$W/data/$x" "$D/$x"; done
  cp -f "$W/data/train.csv" "$D/train.csv"    # fs16 예시 풀(평가 문항은 자동 제외)
done
EVAL=$W/vdata   OUT=$W/out_verify CONFIGS="base r2v3" bash $W/run_screen_35b.sh
EVAL=$W/vdata_c OUT=$W/out_verify CONFIGS="fs16"      bash $W/run_screen_35b.sh
echo "$(date -u +%FT%TZ) VERIFY_DONE" >> "$W/out_verify/STATUS"
