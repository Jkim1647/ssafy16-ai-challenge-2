#!/usr/bin/env bash
# 27B LoRA 재학습 — 기존 단일 최고 모델(val667 0.9685)을 우리 최종 입력 형식에 맞춘다.
#
# 왜 이게 1순위인가. 기존 27B LoRA 의 meta.json 을 열어보니 args 에 `view` 키 자체가 없다.
# 구버전 스크립트로 돌아서 **full 고정**이었고 `exclude_csv` 도 없었다. 즉 우리 단일 최고
# 모델은 최종 파이프라인이 쓰는 read+tiles 와 **한 번도 결합된 적이 없다**.
#   기존: fit 6,047 / val667 / view=full   / 결함 제외 없음 -> 0.9685
#   이번: fit 5,976 / val667 / view=tiles  / 결함 71건 제외
# 학습량은 오히려 71건 줄어든다(데이터 확대가 아니다). 바뀌는 것은 입력 형식과 결함 제외뿐이라
# val667 에서 기존 0.9685 와 직접 비교하면 그 효과만 분리된다.
#
# 덤으로 이 실행이 **잃어버린 가중치를 복구**한다. 기존 27B 어댑터는 Colab 쪽에만 있고
# 저장소·교육장 PC 어디에도 없다(HANDOFF 00000005).
#
# VIEW 는 C안(35B+타일) 결과를 보고 넘긴다. AUG 는 R1·R2 를 보고 정한다.
#   예) VIEW=tiles2x2 bash run_27b_3.sh
set -uo pipefail
export HF_HOME=/workspace/hf
OUT=/workspace/out_27b
MODEL=Qwen/Qwen3.6-27B
VIEW="${VIEW:-tiles2x2}"
AUG="${AUG:-}"
mkdir -p "$OUT"
st() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/STATUS"; }

[ "$(ls /workspace/data/train 2>/dev/null | wc -l)" -ge 6714 ] || { st "이미지 부족 - 중단"; exit 1; }
st "27B 다운로드 시작"
python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL', max_workers=16)" \
  || { st "다운로드 실패 - 중단"; exit 1; }

cd /workspace/tools || exit 1
BASE="--model $MODEL --data /workspace/data --out $OUT --view $VIEW $AUG \
  --val-ids /workspace/data/val667_ids.json \
  --exclude-csv /workspace/data/train_exclusions_v2.csv \
  --r 8 --alpha 16 --lr 1e-4 --epochs 1 --grad-accum 16 --seed 1 --skip-test"

# 27B dense 는 35B-A3B(active 3B)보다 토큰당 연산이 많다. 실제 초/샘플을 먼저 재고
# 저장->재로드까지 확인한다. 여기서 죽으면 본학습에 돈을 쓰지 않는다.
st "스모크 시작 (48/40) | VIEW=$VIEW AUG=${AUG:-없음}"
timeout 7200 python colab_lora_train.py $BASE --limit-train 48 --limit-val 40 --tag smoke27 \
  || { st "스모크 실패 - 중단"; exit 1; }
st "SMOKE27_END"

st "R3 본학습 시작 (학습 5976 / 평가 667)"
timeout 64800 python colab_lora_train.py $BASE --tag R3_27b_tiles
rc=$?
st "R3 종료 rc=$rc"

# 이 파드에는 볼륨이 없다. 정지하면 /workspace 가 통째로 사라지므로 회수가 한 번에
# 끝나도록 미리 묶어 둔다. 집에서는 scp 한 줄이면 된다(HANDOFF 참조).
tar czf "$OUT/R3_artifacts.tgz" -C "$OUT" R3_27b_tiles 2>/dev/null   && st "묶음 생성 $(du -h "$OUT/R3_artifacts.tgz" | cut -f1)"   || st "묶음 실패 - R3_27b_tiles/ 를 직접 회수할 것"
st "R3_DONE"
ls -la "$OUT"
