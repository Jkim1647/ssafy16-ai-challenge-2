#!/usr/bin/env bash
# 미시험 레버 스크리닝 — 35B read+tiles 에서 한 번의 모델 로드로 여러 설정을 연달아 돌린다.
#
# 왜 35B 인가. 실측 0.673초/문항, 로드 83초로 가장 싸다. 여기서 이득이 안 나오면
# 큰 모델에서도 기대하기 어렵다. 이득이 나오면 그때 397B 로 올린다.
#
# 왜 한 스크립트에 묶나. 비용의 대부분이 모델 로드다. 설정마다 따로 띄우면 그만큼 더 낸다.
#
# 돌리는 설정 (CONFIGS 로 고를 수 있다)
# 우선순위 근거 (2026-09-23 Codex 가 test 불일치 200건을 눈으로 분류한 결과 + 우리 실험 이력)
#   판독 실패가 관여한 문항은 32/200 = 16% 뿐이고 **84% 가 판독 이후 단계**(연결·필드·선택·관례)다.
#   그래서 해상도 계열(min_pixels·타일)의 사정권은 32건이 상한이고, 판독 이후를 겨냥하는
#   read1 v2 / few-shot / read2 v3 가 먼저다.
#
#   read2 v2 는 **빼 두었다** — 이미 돌려서 졌다(train6047 5,836 -> 5,835, H408 364 -> 362).
#   Codex 는 우리 실험 이력을 몰라 이걸 1순위(사정권 77)로 올렸다.
#
#   AMBIGUOUS_OPTIONS(200건 중 54건)는 손댈 수 없음을 따로 확인했다. train 에서 보기끼리
#   부분문자열 관계인 337건(5.02%)의 정답이 긴 쪽 46.7% / 짧은 쪽 53.3% 로 규칙이 없다.
#
#   base    기준선. 같은 환경에서 다시 뽑아 비교를 닫는다(기존 base 는 다른 GPU·vLLM 에서 나왔다)
#   r1v2    1단계 판독을 "관련될 수 있는 항목을 빠짐없이" 로 바꾼다 — **누락**을 줄이는 축.
#           read-tokens 도 같이 올려야 의미가 있다
#   r1v3    판독문을 `위치 | 라벨 | 값` 한 줄씩으로 내보내게 한다 — **짝 보존**을 하는 축.
#           v2 와 다른 문제를 푼다. 판독 실패는 200건 중 32건(16%)뿐이고 84%가 판독 이후인데,
#           그중 가장 넓은 것이 소유권 상실이다(메뉴명과 가격, 간판과 층, 요일과 시간이
#           판독문에 다 있는데 어느 것이 어느 것의 짝인지 잃는다).
#           타일 위치 자체는 이미 TILES_NOTE 로 알려주고 있다 — 빠진 것은 **출력 형식**이었다
#   fs16    few-shot 16개(텍스트 전용). 미시험. 가르칠 것이 "어떻게 보느냐"가 아니라
#           "비슷한 보기 중 무엇을 고르느냐"라 이미지 없이 보여줄 수 있다
#   r2v3    2단계 라벨 타이브레이크. 동점일 때만 "크게 적힌 것 > 위쪽·중앙 > 로마자".
#           LABEL_CONVENTION 이 주·보조로 등장하는 32건(16%)을 겨냥
#   t3      타일 3배 확대(현재 2배). 순수 판독 실패 32건이 직접 사정권이지만, 공간 분리에도
#           도움이 될 수 있어 32건이 엄밀한 상한은 아니다. 다만 32K context 라 가장 느려
#           새벽 큐에서는 뺀다(CONFIGS 로 켤 수 있다)
#   mp2304  min_pixels 2304 / max_pixels 3072. t3 와 같은 축이라 둘 중 하나만 돌린다.
#           다른 참가자(38위) 실측: max_pixels 만 올리면 작은 원본이 확대되지 않아 안 오른다
#   mp1536  같은 축의 보수적인 값(Qwen3-VL-8B 최적이 1536..2048)
#
# 사용
#   EVAL=/workspace/evaldata CONFIGS="base fs16 mp2304" bash run_screen_35b.sh
#   EVAL 디렉터리에는 dev.csv 와 train/dev/test 이미지 심볼릭 링크가 있어야 한다.
set -uo pipefail
W=/workspace
MODEL=${MODEL:-Qwen/Qwen3.6-35B-A3B}
EVAL=${EVAL:-$W/screendata}   # dev.csv(train 2,000 + dev 408) + 이미지 링크
OUT=${OUT:-$W/out_screen}
CONFIGS=${CONFIGS:-"base r1v2 r1v3 fs16 r2v3"}
export HF_HOME=$W/hf VLLM_USE_FLASHINFER_SAMPLER=0
# vLLM 0.30.0 이 EngineCore 를 fork 로 띄우는데, 부모 프로세스에서 CUDA 가 이미 초기화돼 있으면
# "Cannot re-initialize CUDA in forked subprocess" 로 죽는다. 2026-09-23 밤 새벽 큐가 5설정 전부
# 이걸로 6~25초 만에 실패했다. spawn 으로 띄우면 자식이 깨끗한 상태에서 시작한다.
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# RTX PRO 6000(sm_120)에서 vLLM 0.30 의 FlashInfer **MoE 커널** JIT 가 컴파일되지 않는다
# ("No supported CUDA architectures found for major versions [12]").
# 기존에 끈 VLLM_USE_FLASHINFER_SAMPLER 는 샘플러 경로라 다른 문제였다. MoE 는 triton 으로 돌린다.
export VLLM_USE_FLASHINFER_MOE_FP8=0 VLLM_USE_FLASHINFER_MOE_FP4=0
mkdir -p "$OUT"
st(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/STATUS"; }

[ -f "$EVAL/dev.csv" ] || { st "dev.csv 없음: $EVAL - 중단"; exit 1; }
st "대상 $(( $(wc -l < "$EVAL/dev.csv") - 1 ))건 | 설정: $CONFIGS"

cd $W/tools || exit 1
C="--model $MODEL --data $EVAL --out $OUT --tp 1 --prompt read --view tiles2x2 --tile-scale 2 \
   --max-model-len 16384 --max-num-seqs 64 --gpu-mem 0.90 --moe-backend triton --split dev"

for cfg in $CONFIGS; do
  case $cfg in
    base)   X="" ;;
    r1v2)   X="--read1 v2 --read-tokens 220" ;;   # recall 강화 판독은 기본 80 토큰으로는 잘린다
    r1v3)   X="--read1 v3 --read-tokens 260" ;;   # 구조 보존(위치|라벨|값)은 줄이 길어 더 필요하다
    fs16)   X="--fewshot 16 --fewshot-seed 0" ;;
    t3)     X="--tile-scale 3 --max-model-len 32768" ;;
    mp2304) X="--min-pixels 2359296 --max-pixels 3145728" ;;   # 단위는 픽셀 수(토큰 1개 = 32x32). 09-24 토큰 수를 넣는 버그 수정
    mp1536) X="--min-pixels 1572864 --max-pixels 2097152" ;;
    r2v3)   X="--read2 v3" ;;
    *) st "알 수 없는 설정 $cfg - 건너뜀"; continue ;;
  esac
  st "$cfg 시작"
  # 설정마다 별도 tag 로 저장해 서로 덮어쓰지 않게 한다
  python colab_vllm_infer.py $C $X --tag-suffix "SCREEN_${cfg}" || { st "$cfg 실패"; continue; }
  st "$cfg 끝"
done

# 볼륨이 없는 파드다. 회수가 한 번에 끝나도록 묶는다.
tar czf "$OUT/screen35b_artifacts.tgz" -C "$OUT" --exclude='*.tgz' . 2>/dev/null \
  && st "묶음 $(du -h "$OUT/screen35b_artifacts.tgz" | cut -f1)"
st "SCREEN35B_DONE"
