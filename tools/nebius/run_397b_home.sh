#!/usr/bin/env bash
# Nebius H100x8: Qwen3.5-397B-A17B-FP8 read+tiles 채점 (gold_v3 331 → val667) + manifest.json 기록.
# 규칙: 호스팅 API가 아니라 가중치를 직접 로드한 in-process 추론(vllm.LLM)이다.
#   REPO=$HOME/AI_2_CHALLENGE DATA=$HOME/ssafy-16-2-ai bash run_397b_home.sh
# 산출물: $OUT/*predictions.jsonl, *_meta.json, manifest.json, gpu_util.log → $HOME/nebius_397b_results.tar.gz
set -uo pipefail
REPO="${REPO:-$HOME/AI_2_CHALLENGE}"
DATA="${DATA:-$HOME/ssafy-16-2-ai}"
OUT="${OUT:-$HOME/out}"; mkdir -p "$OUT"
MODEL="Qwen/Qwen3.5-397B-A17B-FP8"
REV="${REV:-ea5b4f81096f3901c91dea97f81324302495781d}"
RUN_ID="nebius_397b_fp8_readtiles_$(date -u +%Y%m%dT%H%M%SZ)"
export HF_HOME="${HF_HOME:-$HOME/hf}"
START_UTC=$(date -u +%FT%TZ)
NG=$(nvidia-smi -L | wc -l); echo "[nebius] run $RUN_ID, GPUs: $NG"

# 멈춤(hang) 판정용: GPU 사용률·메모리를 30초마다 기록한다.
( while true; do echo "$(date -u +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' ')"; sleep 30; done ) > "$OUT/gpu_util.log" 2>&1 &
MON=$!

cd "$REPO/tools"
python3 - "$REPO" <<'PY'
import json, sys
g = json.load(open(sys.argv[1] + "/runs/dev_validation_v1/hard_eval_gold_v3.json", encoding="utf-8-sig"))
json.dump(g, open("h331_gold.json", "w")); json.dump(sorted(g), open("h331_ids.json", "w"))
print("[nebius] gold_v3:", len(g))
PY

# SKIP_INSTALL=1: 준비 VM에서 영구 디스크의 venv에 이미 설치했으면 GPU 시간에 재설치하지 않는다.
if [ -z "${SKIP_INSTALL:-}" ]; then
  echo "[nebius] installing vllm (nightly) + deps ..."
  python3 -m pip install -q -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
  python3 -m pip install -q pandas pillow "huggingface_hub[hf_xet]"
fi
python3 -c "import vllm,torch;print('[nebius] vllm',vllm.__version__,'torch',torch.__version__,'cc',torch.cuda.get_device_capability(0))"

echo "[nebius] downloading $MODEL @ $REV ..."
python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL', revision='$REV', max_workers=32))"

C="--model $MODEL --data $DATA --out $OUT --tp $NG --prompt read --view tiles2x2 --max-model-len 12288 --gpu-mem 0.90"
echo "[nebius] === gold_v3 (dev 331) ==="
python3 colab_vllm_infer.py $C --split dev --ids-json h331_ids.json --gold-json h331_gold.json --tag-suffix H331
echo "[nebius] === val667 (train split) ==="
python3 colab_vllm_infer.py $C --split val667 --split-json "$REPO/data_meta/splits/val667_ids.json"
kill $MON 2>/dev/null

# manifest.json (산출물 관리 규칙): 코드·환경·모델 revision·데이터 checksum·split·출력 파일 해시
python3 - "$REPO" "$DATA" "$OUT" "$RUN_ID" "$START_UTC" "$MODEL" "$REV" "$NG" <<'PY'
import hashlib, json, subprocess, sys, datetime, importlib.metadata as md
from pathlib import Path
repo, data, out, run_id, start, model, rev, ng = sys.argv[1:]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ver(p):
    try: return md.version(p)
    except Exception: return None
def git(*a):
    # 비공개 저장소라 VM에는 git archive로 올린다(.git 없음). 그때는 COMMIT 파일의 해시를 쓴다.
    r = subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True)
    if r.returncode == 0:
        return r.stdout.strip()
    c = Path(repo, "COMMIT")
    return {"rev-parse": c.read_text().strip() if c.exists() else None}.get(a[0])
outs = sorted(p for p in Path(out).iterdir() if p.is_file() and p.name != "manifest.json")
acc = {}
for p in outs:
    if p.name.endswith("_predictions.jsonl"):
        r = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
        n = sum(1 for x in r if x.get("gold")); ok = sum(1 for x in r if x.get("gold") and x["pred"] == x["gold"])
        acc[p.name] = {"n_rows": len(r), "n_gold": n, "correct": ok, "acc": round(ok / n, 4) if n else None}
g2 = json.load(open(repo + "/runs/dev_validation_v1/hard_eval_gold_v2.json", encoding="utf-8-sig"))
for p in outs:
    if "H331" in p.name and p.name.endswith("_predictions.jsonl"):
        r = {json.loads(l)["id"]: json.loads(l) for l in p.open(encoding="utf-8") if l.strip()}
        ok = sum(1 for i in g2 if i in r and r[i]["pred"] == g2[i])
        acc["H184_subset_of_" + p.name] = {"n_gold": len(g2), "correct": ok, "acc": round(ok / len(g2), 4)}
m = {
    "run_id": run_id, "start_utc": start, "end_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "purpose": "397B ceiling test vs 35B read+tiles (0.9655/H184 0.837) and 122B BF16 (0.9580/0.8478)",
    "code": {"repo": "https://github.com/Jkim1647/AI_2_CHALLENGE", "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
             "commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
             "script": "tools/nebius/run_397b_home.sh", "infer": "tools/colab_vllm_infer.py"},
    "environment": {"python": sys.version.split()[0], "vllm": ver("vllm"), "torch": ver("torch"),
                    "transformers": ver("transformers"), "gpus": int(ng),
                    "nvidia_smi": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().splitlines()},
    "model": {"id": model, "revision": rev, "processor_revision": rev, "quantization": "FP8 checkpoint (as published)"},
    "inference": {"engine": "vllm in-process", "prompt": "read (2-stage transcribe then a-d logprob)", "view": "tiles2x2",
                  "tile_scale": 2.0, "read_tokens": 80, "max_model_len": 12288, "tp": int(ng), "temperature": 0.0,
                  "scoring": "next-token logprob restricted to a-d", "enable_thinking": False},
    "data": {"dir": data, "sha256": {f: sha(Path(data) / f) for f in ["train.csv", "dev.csv"]},
             "splits": {"val667_ids.json": sha(repo + "/data_meta/splits/val667_ids.json"),
                        "hard_eval_gold_v3.json(331)": sha(repo + "/runs/dev_validation_v1/hard_eval_gold_v3.json"),
                        "hard_eval_gold_v2.json(184)": sha(repo + "/runs/dev_validation_v1/hard_eval_gold_v2.json")},
             "id_mapping": "id column of train.csv/dev.csv; image = <data>/<path column>; letters a-d = columns a,b,c,d"},
    "training": None, "checkpoint": None, "trainer_state": None,
    "outputs": {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in outs},
    "accuracy": acc,
    "submission": None,
}
Path(out, "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
print("[nebius] manifest written"); print(json.dumps(acc, ensure_ascii=False, indent=1))
PY

tar czf "$HOME/nebius_397b_results.tar.gz" -C "$OUT" .
echo "[nebius] ALL DONE -> $HOME/nebius_397b_results.tar.gz"
