#!/usr/bin/env bash
# RunPod 1-GPU(RTX Pro 6000 96GB 권장)에서 Qwen3.5-9B LoRA: CE-only vs CE+KD(397B teacher) 비교.
# ⚠️ 비용 발생 스크립트. 팀 승인(실행안·지출 상한) 전에는 실행하지 않는다.
#
# 준비(파드 안, /workspace):
#   AI_2_CHALLENGE/        git archive 또는 clone (tools/, data_meta/, runs/dev_validation_v1/, shared_predictions/nebius_397b_kd/)
#   ssafy-16-2-ai/         train.csv, dev.csv, train/ (6,714장), dev/ (H331 331장 이상)
# 실행:
#   DEADLINE_KST="2026-09-23 06:00" bash AI_2_CHALLENGE/tools/runpod/run_kd_compare.sh 2>&1 | tee /workspace/kd_compare.log
# 산출물: /workspace/out_kd_compare/{smoke,ce,kd}/ (adapter, meta.json, val/hard predictions) + STATUS 파일
# 재개: 단계별 out/<stage>/meta.json이 있으면 그 단계를 건너뛴다(학습 도중 중단분은 처음부터 다시).
# 종료: 성공·실패·시간 초과 모두 마지막에 runpodctl로 파드를 정지한다(프로세스 종료만으로는 과금이 멈추지 않는다).
set -uo pipefail
W=/workspace
REPO=$W/AI_2_CHALLENGE
DATA=$W/ssafy-16-2-ai
OUT=$W/out_kd_compare
MODEL=Qwen/Qwen3.5-9B
KD=$REPO/shared_predictions/nebius_397b_kd/Qwen3.5-397B-A17B-FP8_read_val6714KDtrain_tiles2x2x2_vllm_predictions.jsonl
export HF_HOME=$W/hf
mkdir -p "$OUT"
DEADLINE=$(date -d "${DEADLINE_KST:-2026-09-23 06:00} KST" +%s)
left() { echo $(( DEADLINE - $(date +%s) )); }
status() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/STATUS"; }
stop_pod() {
  status "stopping pod (reason: $1)"
  sync
  [ -n "${NO_STOP:-}" ] && { status "NO_STOP 설정: 파드 정지는 오케스트레이터가 한다"; return; }
  if command -v runpodctl >/dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then runpodctl stop pod "$RUNPOD_POD_ID"; fi
  status "runpodctl 없음/실패 시 콘솔에서 직접 Stop 필요"
}
trap 'stop_pod "script exit"' EXIT

# Qwen3.5 아키텍처는 최신 transformers가 필요하다. 실제 설치 버전은 아래 [env] 줄로 STATUS에 남는다.
if [ -z "${SKIP_PIP:-}" ]; then  # 이미 준비된 환경(예: vLLM과 공존)에서는 SKIP_PIP=1로 재설치를 막는다
  pip install -q -U transformers "peft>=0.17" pandas pillow "huggingface_hub[hf_xet]" || { status "pip 실패"; exit 1; }
fi
python -c "import torch,transformers,peft;print('[env]',torch.__version__,torch.cuda.get_device_name(0),transformers.__version__,peft.__version__)" | tee -a "$OUT/STATUS"
REV=$(python -c "from huggingface_hub import model_info;print(model_info('$MODEL').sha)")
python -c "from huggingface_hub import snapshot_download;snapshot_download('$MODEL',revision='$REV')" || { status "모델 다운로드 실패"; exit 1; }
status "model $MODEL @ $REV"

cd "$REPO/tools"
COMMON="--model $MODEL --revision $REV --data $DATA --out $OUT --val-ids $REPO/data_meta/splits/val667_ids.json
  --hard-gold $REPO/runs/dev_validation_v1/hard_eval_gold_v3.json --exclude-csv $REPO/data_meta/splits/kd_train_exclusions_v1.csv
  --r 8 --alpha 16 --lr 1e-4 --epochs 1 --grad-accum 16 --seed 1 --view full --skip-test"

run_stage() {  # name, max_seconds, extra args...
  local name=$1 max=$2; shift 2
  if [ -f "$OUT/$name/meta.json" ]; then status "$name: 이미 완료 → 건너뜀"; return 0; fi
  local budget=$(( $(left) - 600 ))  # 정지 여유 10분
  [ "$budget" -lt 300 ] && { status "$name: 마감까지 시간 부족 → 중단"; return 2; }
  [ "$max" -gt "$budget" ] && max=$budget
  status "$name: 시작 (제한 ${max}s)"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 30 > "$OUT/${name}_gpu_mem.log" 2>&1 &
  local mon=$!
  timeout "$max" python colab_lora_train.py $COMMON --tag "$name" "$@"
  local rc=$?
  kill $mon 2>/dev/null
  status "$name: 종료 rc=$rc (124=시간 초과)"
  return $rc
}

# 1) 스모크: 16문항 학습 → 저장 → 8문항 채점 → 새 프로세스 재로드 채점 일치 확인. 실패하면 본실험을 하지 않는다.
run_stage smoke 1800 --limit-train 16 --limit-val 8 --grad-accum 4 --kd-jsonl "$KD" --kd-alpha 0.5 --kd-temp 2 || { status "스모크 실패 → 본실험 안 함"; exit 1; }
timeout 900 python colab_lora_train.py $COMMON --tag smoke_reload --limit-val 8 --eval-adapter "$OUT/smoke/adapter" \
  || { status "재로드 실패 → 본실험 안 함"; exit 1; }
python - "$OUT" <<'PY' || { status "재로드 예측 불일치 → 본실험 안 함"; exit 1; }
import json, sys
o = sys.argv[1]
a = {json.loads(l)["id"]: json.loads(l)["pred"] for l in open(f"{o}/smoke/val_predictions.jsonl")}
b = {json.loads(l)["id"]: json.loads(l)["pred"] for l in open(f"{o}/smoke_reload/val_reload.jsonl")}
assert a == b, (a, b)
print("[smoke] reload preds identical", len(a))
PY

# 2) 본실험: 같은 데이터·입력·순서·스텝. 차이는 KD 항뿐이다.
run_stage ce 5400 || status "ce 실패/시간초과"
run_stage kd 5400 --kd-jsonl "$KD" --kd-alpha 0.5 --kd-temp 2 --kd-filter all || status "kd 실패/시간초과"
status "ALL DONE"
