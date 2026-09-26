#!/usr/bin/env bash
# 선택적 선택지 순환(shift) TTA — 35B, 마진 하위 5% 문항에 shift 0~3.
#
# 근거. val667(35B, ours)에서 전 문항 4순서 평균의 이득 +5 가 마진 하위 5% 밴드에서 전부 나왔다
# (나머지 세 구간 Δ정확히 0). train6047 앙상블에서도 하위 5%(302건)에 오답 94/178 = 53% 가 몰린다.
# 원인은 위치 편향 — 정답이 a 일 때 오답률 3.69%·d 3.72% 인데 c 는 1.86% 다.
# 자세한 설계는 reports/shift_tta_selective_20260923.md.
#
# 2단계로 나눈다. 1단계(train 302)로 게이트를 먼저 보고, 통과할 때만 2단계(test 336)를 돌린다.
# 실패하면 test 이미지 336장을 파드로 올리는 일이 통째로 불필요해진다.
#
# shift0 도 같이 돌린다. 기존 base 는 RTX PRO 6000 + 그때의 vLLM 에서 나왔다. GPU·엔진이 바뀌면
# 같은 가중치라도 로그확률 말단이 달라질 수 있고, shift 쪽만 환경을 바꾸면 그 차이가 이득으로
# 오인된다. 네 순서를 같은 환경에서 뽑아 비교를 닫는다. 302건 3.5분치라 비용은 무시할 수준이다.
#
#   IDS=/workspace/shift_data/ids.json bash run_shift_tta_35b.sh
#   SHIFTS="1 2 3" 로 순서를 줄일 수 있다.
set -uo pipefail
W=/workspace
DATA=${DATA:-$W/data}          # train.csv 와 train/ 이미지가 있는 곳
D=$W/shift_data; OUT=$W/out_shift_35b
IDS=${IDS:-$D/ids.json}
export HF_HOME=$W/hf
# RTX PRO 6000(sm_120)에서 FlashInfer 샘플링 JIT 가 "sm75 이상 필요" 오판으로 실패했다.
# 다른 GPU 에서도 켜 둘 이유가 없다(greedy 채점엔 영향 없음).
export VLLM_USE_FLASHINFER_SAMPLER=0
# vLLM 0.30.0 이 EngineCore 를 fork 로 띄우는데 부모에서 CUDA 가 이미 초기화돼 있으면
# "Cannot re-initialize CUDA in forked subprocess" 로 죽는다(2026-09-23 새벽 큐 전멸 원인).
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# RTX PRO 6000(sm_120)에서 vLLM 0.30 의 FlashInfer **MoE 커널** JIT 가 컴파일되지 않는다
# ("No supported CUDA architectures found for major versions [12]").
# 기존에 끈 VLLM_USE_FLASHINFER_SAMPLER 는 샘플러 경로라 다른 문제였다. MoE 는 triton 으로 돌린다.
export VLLM_USE_FLASHINFER_MOE_FP8=0 VLLM_USE_FLASHINFER_MOE_FP4=0
mkdir -p "$D" "$OUT"
st(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/STATUS"; }

[ -f "$IDS" ] || { st "ids.json 없음: $IDS - 중단"; exit 1; }

# 1) 대상 CSV 를 파드에서 만든다. 이미지 존재를 여기서 확인해 본실행에서 0행이 나오는 것을 막는다
#    (2026-09-22 에 이미지가 없는 채로 돌려 GPU 13분을 버린 적이 있다).
python - "$DATA" "$D" "$IDS" <<'PY' || { st "대상 CSV 준비 실패 - 중단"; exit 1; }
import json, os, sys
import pandas as pd
data, d, ids_path = sys.argv[1:]
want = set(json.load(open(ids_path, encoding="utf-8")))
cols = ["id", "path", "question", "a", "b", "c", "d"]
frames = [pd.read_csv(os.path.join(data, f), encoding="utf-8-sig", dtype=str, keep_default_na=False)[cols]
          for f in ("train.csv", "dev.csv", "test.csv") if os.path.exists(os.path.join(data, f))]
df = pd.concat(frames, ignore_index=True)
df = df[df.id.isin(want)].drop_duplicates(subset="id")
missing_rows = want - set(df.id)
assert not missing_rows, f"CSV 에 없는 id {len(missing_rows)}건: {sorted(missing_rows)[:5]}"
for x in ("train", "dev", "test"):
    src, dst = os.path.join(data, x), os.path.join(d, x)
    if os.path.isdir(src) and not os.path.exists(dst):
        os.symlink(src, dst)
missing_img = [p for p in df.path if not os.path.exists(os.path.join(d, p))]
assert not missing_img, f"이미지 누락 {len(missing_img)}건: {missing_img[:5]}"
df.to_csv(os.path.join(d, "dev.csv"), index=False, encoding="utf-8")
print("rows", len(df), "images ok")
PY

# 2) vLLM. 이 파드는 LoRA 학습용으로 구성돼 vllm 이 없다. 이미 있으면 건너뛴다.
if ! python -c "import vllm" 2>/dev/null; then
  st "vllm 설치 시작"
  pip install --break-system-packages -q vllm || { st "vllm 설치 실패 - 중단"; exit 1; }
fi
python -c "import vllm; print('VLLM', vllm.__version__)" | tee -a "$OUT/STATUS"

# 3) 추론. base(shift0) 를 포함해 네 순서를 같은 환경에서 뽑는다.
cd $W/tools || exit 1
N=$(( $(wc -l < "$D/dev.csv") - 1 ))
st "대상 $N 건 / shift ${SHIFTS:-0 1 2 3}"
for k in ${SHIFTS:-0 1 2 3}; do
  st "shift $k 시작"
  python colab_vllm_infer.py --model Qwen/Qwen3.6-35B-A3B --data "$D" --out "$OUT" \
    --tp 1 --prompt read --view tiles2x2 --tile-scale 2 --max-model-len 16384 --max-num-seqs 64 \
    --gpu-mem 0.90 --moe-backend triton --split dev --tag-suffix SHIFT --shift $k || { st "shift $k 실패"; continue; }
  st "shift $k 끝"
done

# 이 파드에는 볼륨이 없다. 정지하면 /workspace 가 사라지므로 회수가 한 번에 끝나도록 묶어 둔다.
tar czf "$OUT/shift35b_artifacts.tgz" -C "$OUT" --exclude='*.tgz' . 2>/dev/null \
  && st "묶음 $(du -h "$OUT/shift35b_artifacts.tgz" | cut -f1)"
st "SHIFT35B_DONE"
ls -la "$OUT"
