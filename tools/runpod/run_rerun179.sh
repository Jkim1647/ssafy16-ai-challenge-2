#!/usr/bin/env bash
# RunPod 1-GPU(RTX PRO 6000 96GB): Qwen3.6-35B-A3B read+tiles 를 검수로 문항이 수정된 179건에 돌린다.
# 목적: H229(229) + 이 179 = 408 로 평가셋을 키워 "모든 모델 차이가 잡음"인 상태를 완화한다.
#
# 사전 업로드 (로컬에서 scp/runpodctl):
#   runs/rerun_dev179/      -> $W/rerun_dev179/      (dev.csv, dev/*.jpg, gold.json)
#   runs/rerun_multistep21/ -> $W/rerun_multistep21/ (선택, 외부 MTVQA probe)
# 두 폴더는 tools/build_rerun200.py 로 로컬에서 만든다. 대회 이미지라 git에 없다.
#
#   bash run_rerun179.sh                # 스모크 8 → dev179 (+ 있으면 multistep21)
#   SMOKE_ONLY=1 bash run_rerun179.sh   # 스모크만
set -uo pipefail
W=/workspace
REPO=$W/AI_2_CHALLENGE
OUT=$W/out_rerun179
MODEL=Qwen/Qwen3.6-35B-A3B
export HF_HOME=$W/hf
# RTX PRO 6000(sm_120): FlashInfer 샘플링 JIT가 "sm75 이상 필요"로 오판해 실패 → 기본 샘플러(greedy 채점엔 영향 없음)
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p "$OUT"
status() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/STATUS"; }
cd "$REPO/tools"

# 입력 검증 — 유료 GPU를 켠 뒤에 깨지지 않게 먼저 확인한다.
python - "$W" <<'PY' || { status "입력 검증 실패"; exit 1; }
import json, os, sys
w = sys.argv[1]
import pandas as pd
for name, n in (("rerun_dev179", 179), ("rerun_multistep21", 21)):
    d = f"{w}/{name}"
    if not os.path.isdir(d):
        print(f"[skip] {name} 없음"); continue
    df = pd.read_csv(f"{d}/dev.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    gold = json.load(open(f"{d}/gold.json", encoding="utf-8"))
    miss = [p for p in df.path if not os.path.exists(f"{d}/{p}")]
    assert not miss, f"{name} 이미지 누락 {len(miss)}: {miss[:5]}"
    assert len(df) == n, f"{name} 행수 {len(df)} != {n}"
    assert set(df.id) == set(gold), f"{name} id/gold 불일치"
    assert "answer" not in df.columns, f"{name} dev.csv에 정답 열이 있으면 안 된다"
    print(f"[ok] {name}: {len(df)}행, 이미지 {len(df)}장, gold {len(gold)}")
PY

# 기존 실행과 동일한 설정 — 비교가 성립하려면 바뀌면 안 된다.
# (run_35b_read2.sh 의 EVAL998 과 같은 플래그. --max-num-seqs 64 는 35B Mamba 캐시 블록 783 < vLLM 기본 1024 때문)
C="--model $MODEL --out $OUT --tp 1 --prompt read --view tiles2x2 --max-model-len 16384 --gpu-mem 0.90 --max-num-seqs 64"

status "smoke 8 시작"
timeout 1800 python colab_vllm_infer.py $C --data "$W/rerun_dev179" --split dev \
  --gold-json "$W/rerun_dev179/gold.json" --tag-suffix SMOKE8 --read2 both --limit 8 \
  || { status "smoke 실패"; exit 1; }
status "smoke 8 종료"
[ -n "${SMOKE_ONLY:-}" ] && exit 0

status "dev179 시작"
timeout 3600 python colab_vllm_infer.py $C --data "$W/rerun_dev179" --split dev \
  --gold-json "$W/rerun_dev179/gold.json" --tag-suffix RERUN179 --read2 both
status "dev179 종료 rc=$?"

if [ -d "$W/rerun_multistep21" ]; then
  # 외부 MTVQA probe. 대회 평가셋에 합치지 말 것 — 별도 보고용.
  status "multistep21 시작"
  timeout 1200 python colab_vllm_infer.py $C --data "$W/rerun_multistep21" --split dev \
    --gold-json "$W/rerun_multistep21/gold.json" --tag-suffix MULTISTEP21 --read2 both
  status "multistep21 종료 rc=$?"
fi

status "DONE — $OUT 를 로컬로 회수한 뒤 파드를 정지한다"
ls -la "$OUT"
