#!/usr/bin/env bash
# Nebius 준비 VM(저가 1-GPU)에서 영구 데이터 디스크에 397B 실행 환경을 미리 만든다.
# GPU 8장 VM에서는 가중치 다운로드·패키지 설치 없이 추론만 돌리기 위함이다.
#   DEV=/dev/disk/by-id/virtio-<device id> bash prep_disk.sh
# 결과: /mnt/m/{venv,hf,AI_2_CHALLENGE,ssafy-16-2-ai}
set -euo pipefail
M=/mnt/m
DEV="${DEV:?set DEV to the data disk block device}"
MODEL="Qwen/Qwen3.5-397B-A17B-FP8"
REV="${REV:-ea5b4f81096f3901c91dea97f81324302495781d}"

# 새 디스크일 때만 포맷한다(이미 ext4면 그대로 마운트).
if ! sudo blkid "$DEV" >/dev/null 2>&1; then sudo mkfs.ext4 -q -L c031model "$DEV"; fi
sudo mkdir -p "$M" && (mountpoint -q "$M" || sudo mount "$DEV" "$M") && sudo chown "$USER" "$M"

sudo apt-get update -qq && sudo apt-get install -y -qq python3.12-venv python3-pip >/dev/null
[ -x "$M/venv/bin/python" ] || python3 -m venv "$M/venv"
PY="$M/venv/bin/python"
"$PY" -m pip install -q -U pip
"$PY" -m pip install -q -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
"$PY" -m pip install -q pandas pillow "huggingface_hub[hf_xet]"
"$PY" -c "import vllm,torch;print('[prep] vllm',vllm.__version__,'torch',torch.__version__,'cuda',torch.cuda.is_available())"

# 코드·데이터(로컬에서 scp한 tar). 비공개 저장소라 VM에서 clone하지 않는다.
[ -f "$HOME/repo.tar" ] && tar xf "$HOME/repo.tar" -C "$M"
[ -f "$HOME/data.tar" ] && mkdir -p "$M/ssafy-16-2-ai" && tar xf "$HOME/data.tar" -C "$M/ssafy-16-2-ai"

export HF_HOME="$M/hf"
"$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL', revision='$REV', max_workers=32))"
# 추론 코드는 revision 없이 모델 ID로 로드한다. 오프라인(HF_HUB_OFFLINE=1)에서 ID가 고정 revision을 가리키도록 refs/main을 맞춘다.
mkdir -p "$HF_HOME/hub/models--${MODEL//\//--}/refs" && echo -n "$REV" > "$HF_HOME/hub/models--${MODEL//\//--}/refs/main"
du -sh "$M/hf" && df -h "$M"
echo "[prep] DONE"
