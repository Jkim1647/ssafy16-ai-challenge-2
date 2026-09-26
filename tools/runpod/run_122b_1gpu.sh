#!/usr/bin/env bash
# RunPod B300 x2: 122B BF16, GPU0=H184 / GPU1=val667 병렬(각 tp=1, NCCL 회피).
# sm_103(B300) 대응: cu13 nvcc로 JIT, Triton MoE/attention 백엔드, flashinfer sampler off.
CU13=/usr/local/lib/python3.12/dist-packages/nvidia/cu13
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
export CUDA_HOME=$CU13 PATH=$CU13/bin:$PATH
export NCCL_NVLS_ENABLE=0 VLLM_USE_FLASHINFER_SAMPLER=0
OUT=/workspace/out; RP=/workspace/rp; cd $RP
C="--model Qwen/Qwen3.5-122B-A10B --data $RP/data --out $OUT --tp 1 --prompt read --view tiles2x2 --max-model-len 12288 --gpu-mem 0.95 --max-num-seqs 16 --moe-backend triton --attention-backend TRITON_ATTN"
CUDA_VISIBLE_DEVICES=0 python colab_vllm_infer.py $C --split dev --ids-json h184_ids.json --gold-json h184_gold.json --tag-suffix H184 > $OUT/h184.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 python colab_vllm_infer.py $C --split val667 --split-json val667_ids.json > $OUT/val667.log 2>&1 &
P1=$!
wait $P0 $P1; echo "[$(date +%T)] ALL DONE"
