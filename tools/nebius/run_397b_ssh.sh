#!/usr/bin/env bash
# Nebius H100x8 자체 호스팅: Qwen3.5-397B-A17B-FP8 천장 테스트 (규칙 OK: API 아님, in-process 추론)
# 사용 전제: (1) 이 저장소가 $REPO 에 clone 됨  (2) 대회 데이터가 $DATA 에 있음(train.csv/dev.csv + train/·dev/ 이미지)
#   REPO=$HOME/AI_2_CHALLENGE  DATA=$HOME/ssafy-16-2-ai  bash run_397b_ssh.sh
# 결과: $HOME/out/*.jsonl(+meta), 요약 accuracy 출력, $HOME/nebius_397b_results.tar.gz
set -uo pipefail
REPO="${REPO:-$HOME/AI_2_CHALLENGE}"
DATA="${DATA:-$HOME/ssafy-16-2-ai}"
OUT="${OUT:-$HOME/out}"; mkdir -p "$OUT"
export HF_HOME="${HF_HOME:-$HOME/hf}"
NG=$(nvidia-smi -L | wc -l); echo "[nebius] GPUs: $NG"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$REPO/tools"
# H184(184) = 기존 35B/122B와 동일 평가셋(gold_v2)으로 사과-대-사과 비교
python3 - "$REPO" <<'PY'
import json,sys
g=json.load(open(sys.argv[1]+"/runs/dev_validation_v1/hard_eval_gold_v2.json",encoding="utf-8-sig"))
json.dump(g, open("h184_gold.json","w")); json.dump(sorted(g), open("h184_ids.json","w"))
print("[nebius] H184 gold:", len(g))
PY

echo "[nebius] installing vllm (nightly) + deps ..."
python3 -m pip install -q -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
python3 -m pip install -q pandas pillow "huggingface_hub[hf_xet]"
python3 -c "import vllm,torch;print('[nebius] vllm',vllm.__version__,'torch',torch.__version__,'cc',torch.cuda.get_device_capability(0))"

echo "[nebius] downloading Qwen3.5-397B-A17B-FP8 (~397GB) ..."
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-397B-A17B-FP8', max_workers=32)"

C="--model Qwen/Qwen3.5-397B-A17B-FP8 --data $DATA --out $OUT --tp $NG --prompt read --view tiles2x2 --max-model-len 12288 --gpu-mem 0.90"
echo "[nebius] === H184 (dev 184) ==="
python3 colab_vllm_infer.py $C --split dev --ids-json h184_ids.json --gold-json h184_gold.json --tag-suffix H184
echo "[nebius] === val667 (train split) ==="
python3 colab_vllm_infer.py $C --split val667 --split-json "$REPO/data_meta/splits/val667_ids.json"

tar czf "$HOME/nebius_397b_results.tar.gz" -C "$OUT" .
echo "[nebius] ALL DONE -> $HOME/nebius_397b_results.tar.gz"
echo "[nebius] === accuracy 요약 (아래 두 줄을 복사해 공유) ==="
grep -h "acc " "$OUT"/*H184*_meta.json 2>/dev/null || true
for f in "$OUT"/*_predictions.jsonl; do
  python3 - "$f" <<'PY'
import json,sys
r=[json.loads(l) for l in open(sys.argv[1],encoding="utf-8") if l.strip()]
ok=sum(1 for x in r if x.get("gold") and x["pred"]==x["gold"]); n=sum(1 for x in r if x.get("gold"))
print(f"{sys.argv[1].split('/')[-1]}: {ok}/{n} = {ok/n:.4f}" if n else f"{sys.argv[1].split('/')[-1]}: (no gold)")
PY
done
