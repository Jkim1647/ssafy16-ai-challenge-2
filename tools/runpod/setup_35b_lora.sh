#!/usr/bin/env bash
# 35B LoRA 학습 환경 준비. 끝나면 SETUP_DONE 을 남긴다(run_35b_lora_AC.sh 가 이걸 기다린다).
#
# 커널 두 개를 같이 깐다. Qwen3.6-35B-A3B 는 gated delta rule 계층을 쓰는데,
# 없으면 transformers 가 "much slower" 경고와 함께 참조 PyTorch 구현으로 돈다.
#   - flash-linear-attention : chunk_gated_delta_rule (Triton 기반, 대부분의 GPU에서 설치됨)
#   - causal-conv1d          : CUDA 확장이라 빌드가 필요. 실패해도 학습은 된다(느릴 뿐)
# 둘 다 실패해도 중단하지 않고 FLA_FAIL / CONV1D_FAIL 을 남겨 로그로 확인할 수 있게 한다.
set -x
export HF_HOME=/workspace/hf
P="pip install --break-system-packages -q"

$P -U transformers accelerate peft bitsandbytes datasets 'huggingface_hub[hf_xet]' pandas pillow \
  || { echo PIP_FAIL; exit 1; }

if $P flash-linear-attention && python -c 'import fla; print("FLA_VER", getattr(fla, "__version__", "?"))'; then
  echo KERNEL_FLA_OK
else
  echo KERNEL_FLA_FAIL
fi

if $P causal-conv1d && python -c 'import causal_conv1d'; then
  echo KERNEL_CONV1D_OK
else
  echo KERNEL_CONV1D_FAIL
fi

python -c 'import torch, transformers, peft; print("ENV", torch.__version__, transformers.__version__, peft.__version__, torch.cuda.get_device_capability(0))' \
  || { echo ENV_FAIL; exit 1; }

python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("Qwen/Qwen3.6-35B-A3B", max_workers=16))' \
  || { echo DOWNLOAD_FAIL; exit 1; }

echo SETUP_DONE
