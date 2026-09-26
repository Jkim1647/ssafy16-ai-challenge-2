"""GPU·VRAM 실측과 자원 게이트.

"이 설정이 이 카드에 들어가는가"를 실행 전에 답하고, 실행 중 실제 peak를 기록한다.
설정의 estimates는 planning 값일 뿐이라 여기서 나온 실측이 항상 우선한다.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import threading
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Any

try:
    import torch
except ImportError:  # 문서 생성·CI 환경에서는 torch가 없을 수 있다
    torch = None  # type: ignore[assignment]


@dataclass
class GpuInfo:
    index: int
    name: str
    total_gb: float
    free_gb: float


def gpu_info() -> list[GpuInfo]:
    if torch is not None and torch.cuda.is_available():
        out = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            out.append(GpuInfo(i, props.name, total / 1e9, free / 1e9))
        return out
    return _gpu_info_smi()


def _gpu_info_smi() -> list[GpuInfo]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    out = []
    for line in res.stdout.strip().splitlines():
        idx, name, total, free = (p.strip() for p in line.split(","))
        out.append(GpuInfo(int(idx), name, float(total) / 1024, float(free) / 1024))
    return out


def snapshot() -> dict[str, Any]:
    gpus = gpu_info()
    return {
        "gpu_count": len(gpus),
        "gpus": [asdict(g) for g in gpus],
        "torch": getattr(torch, "__version__", None),
        "cuda": getattr(getattr(torch, "version", None), "cuda", None) if torch else None,
    }


def reset_peak() -> None:
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_gb() -> float | None:
    if torch is not None and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return None


@contextmanager
def measure_peak(tracker, label: str):
    """with measure_peak(tracker, 'forward'): ... 로 구간 peak VRAM을 남긴다."""
    reset_peak()
    try:
        yield
    finally:
        peak = peak_gb()
        if peak is not None:
            tracker.metric(**{f"peak_vram_gb.{label}": round(peak, 2)})


def _nvidia_smi_used_gb(index: int = 0) -> float | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        lines = res.stdout.strip().splitlines()
        return float(lines[index]) / 1024
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


class NvidiaSmiPeakSampler:
    """nvidia-smi memory.used를 폴링해 실제 물리 VRAM peak를 잰다.

    torch.cuda.max_memory_allocated()는 PyTorch 캐싱 allocator 내부 집계라 CUDA
    context·드라이버 예약분을 반영하지 않는다 — 카드가 실제로 버틸 수 있는지는
    이 값과 nvidia-smi 값을 나란히 봐야 안다. with 블록 동안 별도 스레드로 폴링한다.
    """

    def __init__(self, interval: float = 0.2, index: int = 0):
        self.interval = interval
        self.index = index
        self.peak: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            v = _nvidia_smi_used_gb(self.index)
            if v is not None:
                self.peak = v if self.peak is None else max(self.peak, v)
            self._stop.wait(self.interval)

    def __enter__(self) -> "NvidiaSmiPeakSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


def kernel_impl_snapshot() -> dict[str, Any]:
    """flash-linear-attention/triton/causal_conv1d 실제 사용 여부를 잰다.

    Qwen3.5(qwen3_5, qwen3_5_moe)는 32층 중 24층이 Gated DeltaNet이라 이 커널이
    없으면 pure PyTorch fallback으로 도는데 — 09/17 로그에서 "falling back to its
    reference PyTorch implementation... much slower" 경고로 처음 발견했다. 어느 값을
    쓰는지는 transformers가 해당 모듈을 처음 import하는 시점에 고정된다(패키지가
    나중에 설치돼도 같은 프로세스에서는 안 바뀐다 — 새 프로세스로 다시 확인해야 한다).
    이 함수는 그 import 시점 결정을 inspect로 들여다봐서 fallback인지 실제 커널인지
    구분한다. 5대 중 일부만 이 커널이 깔리면 sec/step·VRAM이 서로 비교 불가능해지므로
    측정 결과마다 반드시 같이 남긴다.
    """
    import importlib
    import inspect

    def _pkg_version(name: str) -> str | None:
        try:
            mod = importlib.import_module(name)
            return getattr(mod, "__version__", "installed")
        except ImportError:
            return None

    info: dict[str, Any] = {
        "fla_version": _pkg_version("fla"),
        "triton_version": _pkg_version("triton"),
        "causal_conv1d_version": _pkg_version("causal_conv1d"),
    }

    for target, module_path, func_name in [
        ("gated_delta_rule_impl", "transformers.models.qwen3_5.modeling_qwen3_5", "torch_chunk_gated_delta_rule"),
        ("causal_conv1d_impl", "transformers.models.qwen3_5.modeling_qwen3_5", "causal_conv1d_fn"),
    ]:
        try:
            mod = importlib.import_module(module_path)
            func = getattr(mod, func_name)
            nonlocals = inspect.getclosurevars(func).nonlocals
            is_new = bool(nonlocals.get("is_new_implementation"))
            info[target] = "kernel" if is_new else "torch_fallback"
        except Exception as exc:  # noqa: BLE001 — 진단이 실패해도 측정 자체는 계속되어야 한다
            info[target] = f"unknown({type(exc).__name__})"

    return info


def _windows_gpu_shared_usage_gb(pid: int) -> float | None:
    """이 프로세스가 WDDM sysmem fallback으로 시스템 RAM에 흘려보낸 GPU 메모리량(GB).

    09/17 실측으로 확인된 문제: dedicated VRAM이 부족하면 CUDA는 OOM을 던지지 않고
    Windows WDDM이 조용히 시스템 RAM으로 넘긴다(Linux에는 없는 동작 — Colab·RunPod은
    해당 없음). torch.cuda.max_memory_allocated()도 nvidia-smi도 이 상태를 못 본다 —
    dedicated 메모리만 보고하므로 "16GB 안에 들어갔다"는 판정이 이미 sysmem fallback을
    쓰고 있었을 수 있다(성공처럼 보이지만 PCIe 대역폭에 발목 잡혀 수십 배 느리다).
    Windows 성능 카운터(`GPU Process Memory\\Shared Usage`)로 직접 잰다.
    """
    if platform.system() != "Windows":
        return None
    try:
        script = (
            f"(Get-Counter '\\GPU Process Memory(*)\\Shared Usage' -ErrorAction SilentlyContinue)."
            f"CounterSamples | Where-Object {{ $_.Path -like '*pid_{pid}_*' }} | "
            f"Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum"
        )
        res = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, timeout=15,
        )
        val = res.stdout.strip()
        return float(val) / 1e9 if val else 0.0
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


class WindowsSharedMemPeakSampler:
    """WDDM sysmem fallback 사용량의 peak를 잰다(Windows 전용, 그 외 OS는 항상 None).

    PowerShell 기동 비용 때문에 nvidia-smi 폴링보다 간격을 길게 둔다(기본 2초) —
    그래도 peak 확인 목적에는 충분하다.
    """

    def __init__(self, interval: float = 2.0, pid: int | None = None):
        self.interval = interval
        self.pid = pid or os.getpid()
        self.peak: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            v = _windows_gpu_shared_usage_gb(self.pid)
            if v is not None:
                self.peak = v if self.peak is None else max(self.peak, v)
            self._stop.wait(self.interval)

    def __enter__(self) -> "WindowsSharedMemPeakSampler":
        if platform.system() == "Windows":
            self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=3)


def preflight(cfg, tracker) -> dict[str, Any]:
    """실행 전 자원 점검. 막지는 않고 경고만 남긴다 — 추정치가 틀릴 수 있기 때문이다."""
    snap = snapshot()
    tracker.event("resource_snapshot", **snap)

    quant = str(cfg.get("model_config.load.quantization", "none")).lower()
    need = cfg.get(f"model_config.estimates.vram_infer_gb.{quant}")
    have = sum(g["total_gb"] for g in snap["gpus"]) or None

    warnings: list[str] = []
    if need and have and float(need) > have:
        warnings.append(
            f"추정 필요 VRAM {need}GB > 감지된 총 VRAM {have:.0f}GB. "
            "device_map=auto / 더 낮은 정밀도 / 더 작은 모델을 검토해라."
        )
    if not snap["gpus"]:
        warnings.append("GPU를 감지하지 못했다. CPU 실행은 실질적으로 불가능하다.")

    for w in warnings:
        tracker.event("resource_warning", message=w)
    return {"snapshot": snap, "warnings": warnings, "estimated_need_gb": need}
