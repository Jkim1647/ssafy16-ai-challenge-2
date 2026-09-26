#!/usr/bin/env bash
# 2026-09-24 저녁 파드 E — 1~10번 중 5·6·8번을 한 파드·한 모델(35B)로 묶는다(사용자 승인 "1~10번까지 진행해").
#
#   5 유형별 입력 분기 : 35B 를 전체 이미지(view full)로 돌린다. 위치·층·방향형 문항은 타일에서 손해라는
#                        Discussion(742915) 단서. 분기 규칙은 키워드로 미리 고정하고 H408 은 확인만 한다.
#   6 해상도           : t3(타일 3배) · mp2304 · mp1536 을 스크리닝 1,408(train1000 + H408)에서.
#   8 r1v3 test        : H408 에서 35B 변형 1위(368)였던 r1v3 를 test 6,714 에 돌린다.
#
# 대기열은 $W/QUEUE 파일이다. 한 줄에 "<셋> <설정>" 하나. 셋은 scr(1,408) / test(6,714) / testsub(3,369).
# 한 줄 끝낼 때마다 파일을 다시 읽으므로, 도는 중에 사람이 줄을 **뒤에 덧붙일** 수 있다(앞줄은 고치지 않는다).
# 대기열을 다 비우고 20분 동안 새 줄이 없으면 E_DONE 을 남기고 끝난다(GPU 를 놀리지 않는다 — 정지는 사람이).
set -uo pipefail
W=/workspace
L=$W/chainE.log
M=Qwen/Qwen3.6-35B-A3B
export HF_HOME=$W/hf VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
# RTX PRO 6000(sm_120)에서 FlashInfer MoE JIT 가 안 된다 — triton 으로 돌린다(run_screen_35b.sh 참고).
export VLLM_USE_FLASHINFER_MOE_FP8=0 VLLM_USE_FLASHINFER_MOE_FP4=0
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$L"; }

log "시작"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 | tee -a "$L"
python -c "import pandas, PIL" 2>/dev/null || pip install --break-system-packages -q pandas pillow 2>&1 | tail -2 | tee -a "$L"
python -c "import huggingface_hub" 2>/dev/null || pip install --break-system-packages -q huggingface_hub 2>&1 | tail -1
( python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$M', max_workers=16))" \
    > $W/dl.log 2>&1; echo "DL_EXIT $?" >> $W/dl.log ) &
DLPID=$!
DRV=$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9.]*' | grep -o '[0-9.]*$')
log "드라이버 지원 CUDA ${DRV:-알수없음}"
NEED_NEW=$(python -c "
try:
    p=('${DRV:-0}'.split('.')+['0'])[:2]; print('1' if (int(p[0]),int(p[1]))>=(12,9) else '0')
except Exception: print('0')")
if [ "$NEED_NEW" = "1" ]; then
  pip install --break-system-packages -q -U vllm 2>&1 | tail -3 | tee -a "$L"
else
  pip install --break-system-packages -q vllm "torch==2.8.0" 2>&1 | tail -3 | tee -a "$L"
fi
python -c "import vllm, torch; print('VLLM', vllm.__version__, '| torch', torch.__version__)" 2>&1 | tee -a "$L" \
  || { log "vllm 사용 불가 - 중단"; exit 1; }
wait $DLPID
grep -q "DL_EXIT 0" $W/dl.log || { log "다운로드 실패 - 중단"; exit 1; }

# 이미지는 설치·다운로드와 겹쳐서 올린다. 사람이 압축을 다 풀고 IMAGES_READY 를 만들 때까지 기다린다(최대 1시간).
for _ in $(seq 1 120); do [ -f $W/IMAGES_READY ] && break; sleep 30; done
# 이미지 개수 확인 —09-24 오전 파드에 test 이미지가 336장뿐인 것을 추론 직전에 발견한 적이 있다.
for x in train dev test; do log "이미지 $x $(ls $W/data/$x 2>/dev/null | wc -l)장"; done
[ "$(ls $W/data/test | wc -l)" -ge 6714 ] || { log "test 이미지 부족 - 중단"; exit 1; }
S=$W/screendata
for x in train dev test; do ln -sfn "$W/data/$x" "$S/$x"; done

cd $W/tools || exit 1
C="--model $M --tp 1 --prompt read --view tiles2x2 --tile-scale 2 --max-model-len 16384 --gpu-mem 0.90 --moe-backend triton"

log "스모크: full 4건"
timeout 1800 python colab_vllm_infer.py $C --view full --max-num-seqs 64 --data $S --split dev --out $W/out_smoke \
  --tag-suffix SMOKE --limit 4 2>&1 | tail -4 | tee -a "$L"
[ -s "$(ls $W/out_smoke/*SMOKE*predictions.jsonl 2>/dev/null | head -1)" ] || { log "스모크 실패 - 중단(본실행에 돈을 쓰지 않는다)"; exit 1; }
log "SMOKE_OK"

args(){  # 설정 -> 추가 인자. run_screen_35b.sh 와 같은 매핑 + full
  case $1 in
    base)   echo "" ;;
    full)   echo "--view full" ;;
    r1v3)   echo "--read1 v3 --read-tokens 260" ;;
    r2v3)   echo "--read2 v3" ;;
    t3)     echo "--tile-scale 3 --max-model-len 32768" ;;
    # 단위는 픽셀 수(토큰 1개 = 32x32). 09-24 에 토큰 수를 넣어 이미지가 40x40 으로 줄었다(chain_e2.sh 참고)
    mp2304) echo "--min-pixels 2359296 --max-pixels 3145728" ;;
    mp1536) echo "--min-pixels 1572864 --max-pixels 2097152" ;;
    *)      echo "UNKNOWN" ;;
  esac
}

n=0; idle=0
while true; do
  line=$(sed -n "$((n+1))p" $W/QUEUE 2>/dev/null | tr -d '\r')
  if [ -z "$line" ]; then
    idle=$((idle+1)); [ $idle -ge 40 ] && break; sleep 30; continue
  fi
  idle=0; n=$((n+1))
  set -- $line; set_=$1; cfg=$2; X=$(args "$cfg")
  [ "$X" = UNKNOWN ] && { log "알 수 없는 설정 $cfg - 건너뜀"; continue; }
  case $set_ in
    scr)  D="--data $S --split dev --max-num-seqs 64";  tag="SCR_${cfg}" ;;
    test) D="--data $W/data --split test --max-num-seqs 128"; tag="TEST_${cfg}" ;;
    # 위치·층·방향 키워드 또는 OBJECT 유형 test 3,369건(testsub_ids.json, 로컬에서 미리 고정) — 5번 분기용
    testsub) D="--data $W/data --split test --max-num-seqs 128 --ids-json $W/testsub_ids.json"; tag="TESTSUB_${cfg}" ;;
    *) log "알 수 없는 셋 $set_ - 건너뜀"; continue ;;
  esac
  log "$set_ $cfg 시작"
  python colab_vllm_infer.py $C $D $X --out $W/out_e --tag-suffix "$tag" > $W/e_${set_}_${cfg}.log 2>&1 \
    && log "$set_ $cfg 끝 $(cat $W/out_e/*${tag}*predictions.jsonl 2>/dev/null | wc -l)건" \
    || log "$set_ $cfg 실패"
  # 볼륨이 없는 파드다. 한 설정 끝날 때마다 묶어 두어 언제 끊겨도 회수할 수 있게 한다.
  tar czf $W/e_artifacts.tgz -C $W/out_e . 2>/dev/null
done
log "E_DONE"
