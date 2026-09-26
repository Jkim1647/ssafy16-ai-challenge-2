#!/usr/bin/env bash
# 스크리닝에서 이긴 설정으로 test 6,714건을 추론한다(2026-09-24 사용자 합의).
#
# 왜 파드를 끄기 전에 하나. 35B 와 test 이미지가 이미 올라와 있다. 새 파드를 띄우면
# 설치·다운로드에 20~30분이 다시 든다.
#
# 어떤 설정을 돌릴지는 **사람이 채점을 보고 정한다.** 이 스크립트는 r1v3 재실행이 끝나기를
# 기다린 뒤 $W/TEST_CFG 파일이 생기기를 기다린다. 파일 내용이 설정 이름이다(공백 구분으로 여러 개 가능).
# "none" 이면 아무것도 돌리지 않고 끝난다. 2시간 안에 파일이 없으면 GPU 를 놀리지 않도록 그냥 끝난다.
#
# 설정 → 인자 매핑은 run_screen_35b.sh 와 같아야 한다. 파드에서 그 스크립트가 도는 중에
# 파일을 고치면 bash 가 깨지므로 여기 따로 둔다.
set -uo pipefail
W=/workspace
OUT=$W/out_test
MODEL=${MODEL:-Qwen/Qwen3.6-35B-A3B}
# 동시 처리 수. 스크리닝(64)보다 올린다 — 2026-09-24 실측 KV 캐시 738K 토큰에 문항당 약 5K 토큰이라
# 약 140개까지 들어가고, GPU 전력이 600W 중 445W 로 여유가 있었다. 같은 GPU 에서 비용 없이 빨라진다.
MAXSEQ=${MAXSEQ:-128}
export HF_HOME=$W/hf VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_MOE_FP8=0 VLLM_USE_FLASHINFER_MOE_FP4=0
mkdir -p "$OUT"
st(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/STATUS"; }

until grep -q R1V3_FOLLOWUP_DONE "$W/out_screen/STATUS" 2>/dev/null; do sleep 30; done
st "스크리닝 종료 확인 -> TEST_CFG 대기"
for _ in $(seq 1 240); do [ -s "$W/TEST_CFG" ] && break; sleep 30; done
[ -s "$W/TEST_CFG" ] || { st "TEST_CFG 없음(2시간) - 종료"; st "TEST_DONE"; exit 0; }
CFGS=$(cat "$W/TEST_CFG")
[ "$CFGS" = "none" ] && { st "none - 돌리지 않음"; st "TEST_DONE"; exit 0; }

cd $W/tools || exit 1
C="--model $MODEL --data $W/data --out $OUT --tp 1 --prompt read --view tiles2x2 --tile-scale 2 \
   --max-model-len 16384 --max-num-seqs $MAXSEQ --gpu-mem 0.90 --moe-backend triton --split test"
for cfg in $CFGS; do
  case $cfg in
    base)   X="" ;;
    r1v2)   X="--read1 v2 --read-tokens 220" ;;
    r1v3)   X="--read1 v3 --read-tokens 260" ;;
    fs16)   X="--fewshot 16 --fewshot-seed 0" ;;
    r2v3)   X="--read2 v3" ;;
    *) st "알 수 없는 설정 $cfg - 건너뜀"; continue ;;
  esac
  st "test $cfg 시작"
  python colab_vllm_infer.py $C $X --tag-suffix "TEST_${cfg}" || { st "test $cfg 실패"; continue; }
  st "test $cfg 끝 $(cat "$OUT"/*TEST_${cfg}*predictions.jsonl 2>/dev/null | wc -l)건"
done
tar czf "$W/test_artifacts.tgz" -C "$OUT" . 2>/dev/null && st "묶음 $(du -h "$W/test_artifacts.tgz" | cut -f1)"
st "TEST_DONE"
