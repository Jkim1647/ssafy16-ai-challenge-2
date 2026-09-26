#!/usr/bin/env bash
# 35B↔397B 답이 갈리는 문항만 397B로 재채점: ① 선택지 순환(shift 1~3) ② 원본 1장 ③ 타일 3배.
# 대상: val667 불일치(검증) + H184 불일치(검증) + test 불일치(적용 후보). 하나의 dev.csv로 묶어 한 번에 돈다.
#   B35_TEST=<35B test predictions.jsonl> bash run_shift_disagree.sh
set -uo pipefail  # 한 실행이 실패해도 나머지 재검증은 계속한다
M=/mnt/m; D=$M/disdata; OUT=$M/out_shift
export PATH=$M/venv/bin:$PATH HF_HUB_OFFLINE=1 HF_HOME=$M/hf XDG_CACHE_HOME=$M/cache FLASHINFER_WORKSPACE_BASE=$M/cache
mkdir -p "$D" "$OUT"
for x in train dev test; do ln -sfn "$M/ssafy-16-2-ai/$x" "$D/$x"; done
python - "$M" "${B35_TEST:?}" <<'PY'
import json, sys, pandas as pd
m, b35 = sys.argv[1:]
rd = lambda p: {json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8") if l.strip()}
t35, t397 = rd(b35), rd(f"{m}/out_test/Qwen3.5-397B-A17B-FP8_read_test_tiles2x2x2_vllm_predictions.jsonl")
dt = sorted(i for i in t397 if t35[i]["pred"] != t397[i]["pred"])
ids = set(json.load(open(f"{m}/dis_val667.json"))) | set(json.load(open(f"{m}/dis_h184.json"))) | set(dt)
src = m + "/ssafy-16-2-ai/"
cols = ["id", "path", "question", "a", "b", "c", "d"]
df = pd.concat([pd.read_csv(src + f, encoding="utf-8-sig", dtype=str, keep_default_na=False)[cols] for f in ("train.csv", "dev.csv", "test.csv")])
df = df[df.id.isin(ids)]
df.to_csv(f"{m}/disdata/dev.csv", index=False, encoding="utf-8")
json.dump(dt, open(f"{m}/dis_test.json", "w"))
print("test disagree", len(dt), "total rows", len(df))
PY
cd $M/AI_2_CHALLENGE/tools
# 기본은 shift 2만(가장 크게 섞인 순서). /mnt/m/SHIFTS 파일에 "1 2 3"을 쓰면 3회 모두 돈다.
for k in $(cat "$M/SHIFTS" 2>/dev/null || echo 2); do
  python colab_vllm_infer.py --model Qwen/Qwen3.5-397B-A17B-FP8 --data "$D" --out "$OUT" --tp 8 --prompt read --view tiles2x2 \
    --max-model-len 16384 --gpu-mem 0.90 --split dev --tag-suffix DIS --shift $k
done
# ② 원본 1장만(타일이 오히려 방해했는지 확인하는 다른 시점)
python colab_vllm_infer.py --model Qwen/Qwen3.5-397B-A17B-FP8 --data "$D" --out "$OUT" --tp 8 --prompt read --view full \
  --max-model-len 16384 --gpu-mem 0.90 --split dev --tag-suffix DIS
# ③ 타일 3배 확대(작은 글씨). 토큰이 늘어나므로 길이 상한을 올린다.
python colab_vllm_infer.py --model Qwen/Qwen3.5-397B-A17B-FP8 --data "$D" --out "$OUT" --tp 8 --prompt read --view tiles2x2 \
  --tile-scale 3 --max-model-len 32768 --gpu-mem 0.90 --split dev --tag-suffix DIS
echo "[shift] DONE"
