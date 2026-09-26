#!/usr/bin/env bash
# RunPod B300 x2: Qwen3.5-397B-A17B-GPTQ-Int4 (236GB), pipeline-parallel(레이어 분할).
# 주의: 이 B300x2 컨테이너에서 vLLM 다중 GPU 워커 init이 "vLLM is using nccl" 직후 행업(tp=2도 pp=2도 동일).
#       단일 GPU는 236GB OOM. 다음 시도: --distributed-executor-backend ray, 또는 다른 vLLM/torch, 또는 SGLang.
CU13=/usr/local/lib/python3.12/dist-packages/nvidia/cu13
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
export CUDA_HOME=$CU13 PATH=$CU13/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
OUT=/workspace/out; RP=/workspace/rp; cd $RP
C="--model Qwen/Qwen3.5-397B-A17B-GPTQ-Int4 --quantization moe_wna16 --data $RP/data --out $OUT --tp 1 --pp 2 --prompt read --view tiles2x2 --max-model-len 12288 --gpu-mem 0.92 --max-num-seqs 16 --enforce-eager --attention-backend TRITON_ATTN"
python colab_vllm_infer.py $C --split dev --ids-json h184_ids.json --gold-json h184_gold.json --tag-suffix H184 > $OUT/h184_397.log 2>&1
python colab_vllm_infer.py $C --split val667 --split-json val667_ids.json > $OUT/val667_397.log 2>&1
echo "[$(date +%T)] 397B ALL DONE"
