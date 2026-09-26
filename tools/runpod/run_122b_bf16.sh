#!/usr/bin/env bash
# RunPod B300 x2: Qwen3.5-122B-A10B BF16 + 판독(read) + 2x2 타일, H184 -> val667 순서.
#   tar -xf rp_bundle.tar -C /workspace && bash /workspace/rp/run_122b_bf16.sh
# 결과: /workspace/out/*_predictions.jsonl, *_meta.json, run.log
set -euo pipefail
W=/workspace; RP=$W/rp; OUT=$W/out; mkdir -p $OUT
export HF_HOME=$W/hf PIP_CACHE_DIR=$W/pipcache
exec > >(tee -a $OUT/run.log) 2>&1
echo "[$(date +%T)] start"; nvidia-smi --query-gpu=name,memory.total --format=csv

# 가중치 다운로드(약 244GB)와 vLLM 설치를 동시에
pip -q install -U "huggingface_hub[hf_xet]"
( hf download Qwen/Qwen3.5-122B-A10B --max-workers 32 > $OUT/download.log 2>&1; echo "[$(date +%T)] download done" ) &
DL=$!
pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly > $OUT/pip.log 2>&1
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, torch.cuda.get_device_capability())"
wait $DL

cd $RP
COMMON="--model Qwen/Qwen3.5-122B-A10B --data $RP/data --out $OUT --tp 2 --prompt read --view tiles2x2 --max-model-len 12288 --gpu-mem 0.90"
echo "[$(date +%T)] H184"
python colab_vllm_infer.py $COMMON --split dev --ids-json h184_ids.json --gold-json h184_gold.json --tag-suffix H184
echo "[$(date +%T)] val667"
python colab_vllm_infer.py $COMMON --split val667 --split-json val667_ids.json
echo "[$(date +%T)] ALL DONE"; ls -la $OUT
