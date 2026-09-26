"""주말(09/19~09/20)에 미리 돌려서 9B/2B 가중치를 로컬 HF 캐시에 받아둔다.

로컬 5060 Ti에서 실제로 실행 가능한 모델만 받는다 — 9B와 2B뿐이다.
35B(NF4 약 23GB)·122B(약 65GB)·397B(약 203GB)는 로컬 카드(16GB)에 애초에
안 들어가므로 여기서 받지 않는다. 받아봤자 디스크만 채우고 쓸 수 없다
(35B/122B/397B는 RunPod 클라우드 전용 — configs/resources/runpod_*.yaml 참고).

실행:
    python tools/prefetch_weights.py

이미 완전히 캐시돼 있으면 네트워크를 다시 건드리지 않고 즉시 통과한다.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# 기본 10초 타임아웃은 교육장 wifi에서 계속 끊긴다 — snapshot_download를 부르기
# 전에 반드시 먼저 설정해야 한다(huggingface_hub가 import 시점에 값을 읽는다).
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

# (name, hf_id, 예상 디스크 사용량 GB)
# 9B: configs/models/qwen35_9b.yaml의 hf_id를 그대로 썼다.
# 2B: 이 저장소에 아직 configs/models/*2b*.yaml이 없다 — Qwen3.5 라인업의
#     명명 규칙("Qwen/Qwen3.5-<size>B")을 따라 가정한 값이다. 실제 2B 모델
#     카드가 다른 id로 확인되면 이 줄만 고치면 된다.
MODELS_TO_PREFETCH = [
    ("9B", "Qwen/Qwen3.5-9B", 19),
    ("2B (가정 — 실제 hf_id 확인 필요)", "Qwen/Qwen3.5-2B", 5),
]

TOTAL_ESTIMATE_GB = sum(size for _, _, size in MODELS_TO_PREFETCH)


def _cached_and_complete(hf_id: str) -> bool:
    """이미 완전히 받아져 있으면 True. .lock/.incomplete가 남아 있으면 재시도 대상으로 본다."""
    from huggingface_hub import scan_cache_dir

    try:
        cache_info = scan_cache_dir()
    except Exception:
        return False
    for repo in cache_info.repos:
        if repo.repo_id == hf_id:
            return True
    return False


def main() -> int:
    from huggingface_hub import snapshot_download

    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    # disk_usage는 경로가 실제로 존재해야 한다 — 첫 실행이면 HF_HOME이 아직 없다.
    Path(hf_home).mkdir(parents=True, exist_ok=True)
    disk_free_gb = shutil.disk_usage(hf_home).free / 1e9
    print(f"HF_HOME: {hf_home}")
    print(f"디스크 여유: {disk_free_gb:.1f} GB / 필요 예상: {TOTAL_ESTIMATE_GB} GB (9B+2B)")

    if disk_free_gb < TOTAL_ESTIMATE_GB:
        print(f"\n*** 디스크 여유가 부족합니다 ({disk_free_gb:.1f}GB < {TOTAL_ESTIMATE_GB}GB). ***")
        print("*** 공간을 확보한 뒤 다시 실행하세요. ***")
        return 1

    for label, hf_id, size_gb in MODELS_TO_PREFETCH:
        print(f"\n=== {label}: {hf_id} (예상 {size_gb}GB) ===")
        if _cached_and_complete(hf_id):
            print("  이미 캐시에 있습니다 — 건너뜁니다 (네트워크 안 건드림).")
            continue

        print("  다운로드 시작 (시간 제한 없음)...")
        t0 = time.time()
        try:
            snapshot_download(repo_id=hf_id)
        except Exception as exc:
            print(f"  실패: {exc}")
            print("  네트워크가 끊겼다면 그냥 다시 실행하세요 — 받은 shard는 이어받습니다.")
            return 1
        elapsed_min = (time.time() - t0) / 60
        print(f"  완료 ({elapsed_min:.1f}분 소요).")

    print("\n전부 완료. 9B/2B 가중치가 로컬 캐시에 있습니다.")
    print("35B/122B/397B는 로컬에서 실행할 수 없으므로 이 스크립트에서 받지 않습니다 —")
    print("RunPod 등 클라우드에서 실행할 때 day1_cloud_model_probe.ipynb의 2절이 영속 캐시로 받습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
