#!/usr/bin/env bash
# 선택적 shift TTA — 397B, 마진 하위 5% 638건(train 302 + test 336)에 shift 1~3.
# 근거·설계는 tools/runpod/run_shift_tta_35b.sh 주석과 tools/select_low_margin.py 참조.
#
# 비용. 397B 실측 0.56초/문항, 로드 약 200초(8xH100 TP8).
#   1,914 문항 -> 약 18분 + 로드 3.5분 = 약 25분.
#   8xH100 $30.81/h 기준 약 $13 / 4xH200 $18/h 기준 약 $8.
#
# 지난 실행과 같은 revision 을 쓴다 — 바뀌면 base 와의 비교가 깨진다.
set -uo pipefail
M=${MNT:-/mnt/m}; D=$M/shift_data; OUT=$M/out_shift_397b
export PATH=$M/venv/bin:$PATH HF_HUB_OFFLINE=1 HF_HOME=$M/hf XDG_CACHE_HOME=$M/cache FLASHINFER_WORKSPACE_BASE=$M/cache
REV=ea5b4f81096f3901c91dea97f81324302495781d
mkdir -p "$OUT"
st(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/STATUS"; }

[ -f "$D/dev.csv" ] || { st "dev.csv 없음 - 중단"; exit 1; }
for x in train dev test; do ln -sfn "$M/ssafy-16-2-ai/$x" "$D/$x"; done
st "대상 $(( $(wc -l < "$D/dev.csv") - 1 )) 건 / shift 1,2,3 / rev $REV"

# 추론기에는 --revision 플래그가 없다. 지난 실행들과 같은 방식으로 HF 캐시에서 revision 을 고정한다.
# 디스크에 이미 받아 둔 스냅샷이 있으면 오프라인으로 그대로 잡힌다.
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-397B-A17B-FP8', revision='$REV', local_files_only=True))"   || { st "revision $REV 스냅샷이 디스크에 없다 - 중단"; exit 1; }

cd $M/AI_2_CHALLENGE/tools || exit 1
# 8건 스모크를 먼저 — 여기서 죽으면 본실행에 돈을 쓰지 않는다
python colab_vllm_infer.py --model Qwen/Qwen3.5-397B-A17B-FP8 --data "$D" --out "$OUT" \
  --tp 8 --prompt read --view tiles2x2 --max-model-len 16384 --gpu-mem 0.90 \
  --split dev --tag-suffix SMOKE8 --shift 1 --limit 8 || { st "스모크 실패 - 중단"; exit 1; }
st "SMOKE8_END"

# 지난 base 와 같은 8xH100 TP8 구성이면 shift0 은 생략해도 된다(SHIFTS="1 2 3").
# GPU 구성을 바꾸면 0 부터 같이 돌려 비교를 닫는다.
for k in ${SHIFTS:-1 2 3}; do
  st "shift $k 시작"
  python colab_vllm_infer.py --model Qwen/Qwen3.5-397B-A17B-FP8 --data "$D" --out "$OUT" \
    --tp 8 --prompt read --view tiles2x2 --max-model-len 16384 --gpu-mem 0.90 \
    --split dev --tag-suffix SHIFT --shift $k || { st "shift $k 실패"; continue; }
  st "shift $k 끝"
done

tar czf "$M/shift397b_artifacts.tgz" -C "$OUT" . 2>/dev/null && st "묶음 $(du -h "$M/shift397b_artifacts.tgz" | cut -f1)"
st "SHIFT397B_DONE"
