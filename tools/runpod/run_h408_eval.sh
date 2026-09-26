#!/usr/bin/env bash
# 학습이 끝난 adapter 를 재로드해 H408(dev 408건)을 채점한다.
#
# 왜 필요한가. R3·Gemma 는 train 6,714 중 5,976 을 학습했다. 남은 홀드아웃은 val667 뿐이고
# val667 은 1문항이 0.150%p 라 ±3문항이 잡음이다 — 3위 팀과의 격차(약 0.09%p)는 여기서
# 측정 자체가 불가능한 크기다. dev 는 두 모델 모두 본 적이 없으므로 H408 이 둘째 게이트가 된다.
# (train 홀드아웃 6,047 은 이들의 학습 데이터라 쓸 수 없고, holdout1500 도 R3 가 833건을 학습했다.)
#
# **파드를 정지하기 전에 돌려야 한다.** 볼륨이 없어 정지하면 adapter 가 사라지고,
# 그러면 다시 재학습(약 4시간)해야 이 숫자를 얻는다.
#
#   MODEL=Qwen/Qwen3.6-27B ADAPTER=/workspace/out_27b/R3_27b_tiles/adapter \
#   OUT=/workspace/out_27b TAG=R3_h408 bash run_h408_eval.sh
set -uo pipefail
W=/workspace
MODEL=${MODEL:?MODEL 을 지정하라}
ADAPTER=${ADAPTER:?ADAPTER 를 지정하라}
OUT=${OUT:-$W/out_h408}
TAG=${TAG:-h408}
VIEW=${VIEW:-tiles2x2}
LIMIT_VAL=${LIMIT_VAL:-50}   # 재로드가 제대로 됐는지 확인만 한다. val667 본값은 학습 산출물에 이미 있다
# SCORE_TEST=1 이면 같은 모델 로드에서 test 6,714건까지 채점한다(제출 후보 생성).
# 순서는 val 확인 -> H408 -> test. 비용의 대부분이 모델 로드라 한 번에 묶는 게 싸다.
export HF_HOME=$W/hf
st(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/STATUS"; }

[ -d "$ADAPTER" ] || { st "adapter 없음: $ADAPTER - 중단"; exit 1; }
[ -f "$W/data/dev.csv" ] || { st "dev.csv 없음 - 중단"; exit 1; }
[ -f "$W/data/hard_gold_h408.json" ] || { st "hard_gold 없음 - 중단"; exit 1; }
N=$(ls "$W/data/dev" 2>/dev/null | wc -l)
[ "$N" -ge 408 ] || { st "dev 이미지 $N 장 - 408 미만이라 중단"; exit 1; }
if [ -n "${SCORE_TEST:-}" ]; then
  NT=$(ls "$W/data/test" 2>/dev/null | wc -l)
  [ "$NT" -ge 6714 ] || { st "SCORE_TEST 인데 test 이미지가 $NT 장뿐 - 중단"; exit 1; }
  [ -f "$W/data/test.csv" ] || { st "test.csv 없음 - 중단"; exit 1; }
fi
st "H408 채점 시작 | $MODEL | $ADAPTER | view=$VIEW | dev 이미지 $N"

cd $W/tools || exit 1
python colab_lora_train.py --model "$MODEL" --data "$W/data" \
  --val-ids "$W/data/val667_ids.json" --out "$OUT" --view "$VIEW" \
  --exclude-csv "$W/data/train_exclusions_v2.csv" \
  --eval-adapter "$ADAPTER" --hard-gold "$W/data/hard_gold_h408.json" \
  --limit-val "$LIMIT_VAL" ${SCORE_TEST:+--reload-score-test} --skip-test --tag "$TAG"
rc=$?
st "H408 채점 종료 rc=$rc"
[ $rc -eq 0 ] || exit $rc

# 볼륨이 없는 파드다. 회수가 한 번에 끝나도록 묶는다.
tar czf "$OUT/${TAG}_artifacts.tgz" -C "$OUT" "$TAG" 2>/dev/null && st "묶음 $(du -h "$OUT/${TAG}_artifacts.tgz" | cut -f1)"
st "H408_DONE"
