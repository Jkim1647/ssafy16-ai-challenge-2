#!/usr/bin/env bash
# RunPod 1-GPU(RTX Pro 6000 96GB): Qwen3.6-35B-A3B read+tiles, 2단계 지시 v1(기존) vs v2(필드·부정·정확일치 점검).
# 같은 환경·같은 문항(val667 + H331)·같은 판독문에서 2단계 지시 v1/v2를 짝비교한다. test는 STAGE=test일 때만(v2 채택 판단 후).
#   STAGE=eval bash run_35b_read2.sh     # val667+H331 v1, v2
#   STAGE=test bash run_35b_read2.sh     # test v2
set -uo pipefail
W=/workspace
REPO=$W/AI_2_CHALLENGE
DATA=$W/ssafy-16-2-ai
OUT=$W/out_35b_read2
MODEL=Qwen/Qwen3.6-35B-A3B
export HF_HOME=$W/hf
# RTX PRO 6000(sm_120)에서 FlashInfer 샘플링 JIT가 "sm75 이상 필요" 오판으로 실패 → vLLM 기본 샘플러 사용(greedy 채점엔 영향 없음)
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p "$OUT" "$W/evaldata"
status() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/STATUS"; }
cd "$REPO/tools"

# 평가셋 묶음: val667(train 행) + H331(dev 행, gold는 hard_eval_gold_v3) → 하나의 dev.csv로 한 번에 돈다.
python - "$REPO" "$DATA" "$W/evaldata" <<'PY' || { status "평가셋 준비 실패"; exit 1; }
import json, sys, os, pandas as pd
repo, data, ev = sys.argv[1:]
val = set(json.load(open(f"{repo}/data_meta/splits/val667_ids.json", encoding="utf-8-sig"))["val_groups"])
g3 = json.load(open(f"{repo}/runs/dev_validation_v1/hard_eval_gold_v3.json", encoding="utf-8-sig"))
cols = ["id", "path", "question", "a", "b", "c", "d"]
tr = pd.read_csv(f"{data}/train.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
dv = pd.read_csv(f"{data}/dev.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
df = pd.concat([tr[tr.id.isin(val)][cols], dv[dv.id.isin(g3)][cols]])
missing = [p for p in df.path if not os.path.exists(f"{data}/{p}")]
assert not missing, f"이미지 누락 {len(missing)}: {missing[:5]}"
assert len(df) == 998, len(df)
df.to_csv(f"{ev}/dev.csv", index=False, encoding="utf-8")
for x in ("train", "dev", "test"):
    if not os.path.exists(f"{ev}/{x}"): os.symlink(f"{data}/{x}", f"{ev}/{x}")
gold = {i: a for i, a in zip(tr.id, tr.answer) if i in val} | g3
json.dump(gold, open(f"{ev}/gold.json", "w"))
print("eval rows", len(df), "images ok")
PY

C="--model $MODEL --out $OUT --tp 1 --prompt read --view tiles2x2 --max-model-len 16384 --gpu-mem 0.90 --max-num-seqs 64"  # 35B Mamba 캐시 블록(783) < vLLM 기본 1024
if [ "${STAGE:-eval}" = eval ]; then
  # 같은 판독문(1단계)으로 v1·v2 2단계를 모두 채점한다(--read2 both). 먼저 8문항 스모크로 메모리·속도·저장 확인.
  status "smoke 8 시작"
  timeout 1500 python colab_vllm_infer.py $C --data "$W/evaldata" --split dev --gold-json "$W/evaldata/gold.json"     --tag-suffix SMOKE8 --read2 both --limit 8 || { status "smoke 실패"; exit 1; }
  status "smoke 8 종료"
  [ -n "${SMOKE_ONLY:-}" ] && exit 0
  status "eval both 시작"
  timeout 3600 python colab_vllm_infer.py $C --data "$W/evaldata" --split dev --gold-json "$W/evaldata/gold.json" --tag-suffix EVAL998 --read2 both
  status "eval both 종료 rc=$?"
else
  python -c "import pandas as pd,os;d=pd.read_csv('$DATA/test.csv',encoding='utf-8-sig',dtype=str);m=[p for p in d.path if not os.path.exists('$DATA/'+p)];assert not m,len(m);print('test images ok',len(d))" \
    || { status "test 이미지 누락"; exit 1; }
  status "test read2=v2 시작"
  timeout 5400 python colab_vllm_infer.py $C --data "$DATA" --split test --read2 v2
  status "test read2=v2 종료 rc=$?"
fi
status "STAGE ${STAGE:-eval} DONE"
