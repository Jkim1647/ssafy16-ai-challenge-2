#!/usr/bin/env bash
# 파드 B — 검증이 끝나면 R3·Gemma 를 **수정본 dev179 가 들어간 H408** 로 다시 채점한다(2026-09-24 사용자 승인).
#
# 왜. 어제 R3·Gemma 의 H408(0.7304 / 0.7402)은 dev179 행을 원본 질문으로 넣은 내 데이터 버그로 무효다
# (커밋 1ce3b67). 이 둘을 다시 채점하면 T4 구성원 4개 + 35B r2v3 = 5개(홀수)가 모두 H408 을 갖게 되어
# **앙상블 결합 방식을 Public 이 아니라 H408 로 고를 수 있다.**
# 27B LoRA·9B LoRA 는 adapter 가중치가 팀 Drive 에만 있어 여기서는 못 한다.
#
# 순서와 디스크. 컨테이너 디스크가 200GB 라 35B(67GB)+27B(54GB)+Gemma(62GB)를 다 둘 수 없다.
#   27B 는 지금(검증 중) 미리 받는다 → 검증이 끝나면 35B 캐시를 지우고 Gemma 를 받으며 R3 를 돈다.
# 재로드 확인. run_h408_eval.sh 와 같이 val667 앞 50건을 먼저 채점한다 — 어제 값(R3 0.98, Gemma 0.96)과
# 비슷해야 adapter 가 제대로 붙은 것이다.
set -uo pipefail
W=/workspace
L=$W/chainH408.log
export HF_HOME=$W/hf
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$L"; }
GEMMA_REV=842da3794eaa0b77d5f08bae87a17459d91ff475     # 학습 때 쓴 커밋(RUN_META.json)

log "27B 미리 받기 시작"
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.6-27B', max_workers=8)" \
  > $W/dl27.log 2>&1 && log "27B 다운로드 완료" || log "27B 다운로드 실패"

python -c "import peft" 2>/dev/null || pip install --break-system-packages -q peft 2>&1 | tail -1 | tee -a "$L"
python -c "import torch, transformers, peft; print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__)" 2>&1 | tee -a "$L"

# 어댑터 풀기(tgz 는 로컬에서 올렸다)
mkdir -p $W/adapters
for t in R3_artifacts gemma_artifacts; do tar xzf $W/imgup/$t.tgz -C $W/adapters 2>&1 | tail -1; done
ls $W/adapters/*/adapter/adapter_model.safetensors 2>&1 | tee -a "$L"

until grep -q VERIFY_DONE "$W/out_verify/STATUS" 2>/dev/null; do sleep 30; done
log "검증 종료 확인 -> 35B 캐시 삭제, Gemma 받기 시작"
rm -rf $W/hf/hub/models--Qwen--Qwen3.6-35B-A3B
( python -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-4-31B-it', revision='$GEMMA_REV', max_workers=8)" \
    > $W/dlg.log 2>&1; echo "DL_EXIT $?" >> $W/dlg.log ) &
DLG=$!

cd $W/tools || exit 1
run(){  # $1 모델 $2 adapter $3 tag $4 revision(선택)
  log "$3 시작"
  python colab_lora_train.py --model "$1" ${4:+--revision $4} --data "$W/data" \
    --val-ids "$W/data/val667_ids.json" --out "$W/out_h408ft" --view tiles2x2 \
    --exclude-csv "$W/data/train_exclusions_v2.csv" \
    --eval-adapter "$2" --hard-gold "$W/data/hard_gold_h408.json" \
    --limit-val 50 --skip-test --tag "$3" > $W/$3.log 2>&1
  log "$3 끝 rc=$? $(grep -ao 'hard[^,]*acc[^,]*' $W/$3.log | tail -1)"
}
run Qwen/Qwen3.6-27B $W/adapters/R3_27b_tiles/adapter R3_h408v2
rm -rf $W/hf/hub/models--Qwen--Qwen3.6-27B
wait $DLG
grep -q "DL_EXIT 0" $W/dlg.log || { log "Gemma 다운로드 실패 - 중단"; exit 1; }
run google/gemma-4-31B-it $W/adapters/gemma4_31b_lora/adapter Gemma_h408v2 $GEMMA_REV
tar czf $W/h408ft_artifacts.tgz -C $W/out_h408ft . 2>/dev/null && log "묶음 $(du -h $W/h408ft_artifacts.tgz | cut -f1)"
log "H408FT_DONE"
