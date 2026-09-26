#!/usr/bin/env bash
# Nebius: Qwen3.5-397B-A17B-FP8 read+tiles 를 검수로 문항이 수정된 179건에 돌린다.
# 목적: H229(229) + 179 = 408 로 평가셋을 키워, 앙상블 vs 397B 단독을 구별 가능한 n에서 비교한다.
# 규칙: 호스팅 API가 아니라 가중치를 직접 로드한 in-process 추론(vllm.LLM)이다.
#
# 전제
#   - 모델 디스크 c031-397b-model(930GiB, FP8 가중치 + venv)을 새 VM에 attach 해 /mnt/m 로 마운트
#   - 4×H200(564GB)면 충분하다. 지난 실행은 8×H100($30.80/h)이었지만 FP8 397GB엔 과하다.
#   - 로컬에서 만든 runs/rerun_dev179/ 를 $HOME/rerun_dev179 로 scp (20MB)
#
#   REPO=$HOME/AI_2_CHALLENGE SKIP_INSTALL=1 bash run_397b_rerun179.sh
set -uo pipefail
REPO="${REPO:-$HOME/AI_2_CHALLENGE}"
EVAL="${EVAL:-$HOME/rerun_dev179}"
MS="${MS:-$HOME/rerun_multistep21}"
OUT="${OUT:-$HOME/out_rerun179}"; mkdir -p "$OUT"
MODEL="Qwen/Qwen3.5-397B-A17B-FP8"
REV="${REV:-ea5b4f81096f3901c91dea97f81324302495781d}"   # 지난 실행과 동일 revision — 비교 조건 고정
export HF_HOME="${HF_HOME:-/mnt/m/hf}"
NG=$(nvidia-smi -L | wc -l)
echo "[nebius] GPUs: $NG, HF_HOME=$HF_HOME"

# 입력 검증 — GPU를 켠 뒤 깨지지 않게 먼저 본다.
python3 - "$EVAL" <<'PY' || { echo "[nebius] 입력 검증 실패"; exit 1; }
import json, os, sys, pandas as pd
d = sys.argv[1]
df = pd.read_csv(f"{d}/dev.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
gold = json.load(open(f"{d}/gold.json", encoding="utf-8"))
miss = [p for p in df.path if not os.path.exists(f"{d}/{p}")]
assert not miss, f"이미지 누락 {len(miss)}: {miss[:5]}"
assert len(df) == 179 and set(df.id) == set(gold), (len(df), len(gold))
assert "answer" not in df.columns
print("[ok] dev179", len(df))
PY

( while true; do echo "$(date -u +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' ')"; sleep 30; done ) > "$OUT/gpu_util.log" 2>&1 &
MON=$!

cd "$REPO/tools"
if [ -z "${SKIP_INSTALL:-}" ]; then
  python3 -m pip install -q -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
  python3 -m pip install -q pandas pillow "huggingface_hub[hf_xet]"
fi
python3 -c "import vllm,torch;print('[nebius] vllm',vllm.__version__,'torch',torch.__version__)"

# 가중치는 모델 디스크에 이미 있다. 없으면 받는다(약 397GB — 이때는 시간·비용이 크게 는다).
python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL', revision='$REV', max_workers=32))"

# 지난 실행(val667/H331)과 동일한 플래그 — 바뀌면 비교가 성립하지 않는다.
C="--model $MODEL --out $OUT --tp $NG --prompt read --view tiles2x2 --max-model-len 12288 --gpu-mem 0.90"

echo "[nebius] === smoke 8 ==="
python3 colab_vllm_infer.py $C --data "$EVAL" --split dev --gold-json "$EVAL/gold.json" \
  --tag-suffix SMOKE8 --limit 8 || { echo "[nebius] smoke 실패"; kill $MON 2>/dev/null; exit 1; }
[ -n "${SMOKE_ONLY:-}" ] && { kill $MON 2>/dev/null; exit 0; }

echo "[nebius] === dev179 ==="
python3 colab_vllm_infer.py $C --data "$EVAL" --split dev --gold-json "$EVAL/gold.json" --tag-suffix RERUN179

if [ -d "$MS" ]; then
  echo "[nebius] === multistep21 (외부 MTVQA probe — 대회 평가셋에 합치지 말 것) ==="
  python3 colab_vllm_infer.py $C --data "$MS" --split dev --gold-json "$MS/gold.json" --tag-suffix MULTISTEP21
fi

kill $MON 2>/dev/null
tar czf "$HOME/nebius_397b_rerun179.tar.gz" -C "$OUT" .
echo "[nebius] DONE — $HOME/nebius_397b_rerun179.tar.gz 회수 후 VM을 삭제한다(디스크 detach 먼저!)"
ls -la "$OUT"
