"""Docker(WSL2) vs 베어메탈 환경 비교 probe — 09/19, 대회 D-3.

vram_shutdown_probe.ipynb의 measure_infer/measure_train 로직을 그대로 재사용하되
헤드리스 스크립트로 옮기고 다음을 추가한다:

- env("bare_windows"/"docker_wsl2")과 torch_version을 결과 CSV 컬럼에 추가.
  베어메탈과 Docker 결과가 한 파일에 섞이면 나중에 못 가리므로 파일명에도 env를 넣는다.
- (b) 학습 Full 1024가 베어메탈 기준(15.215 GB / 2.323 sec/step)과 크게(20%+) 다르면
  (c) Full 1536으로 넘어가지 않고 즉시 중단한다 — 대조군이 어긋나면 (c) 해석 기준이 없다.
- WindowsSharedMemPeakSampler(WDDM sysmem fallback 감지)는 Linux 컨테이너에서 항상
  None을 돌려준다(정상 동작 — Windows 전용 기능). 그걸 빈 칸/0으로 남기면 "폴백 없음"
  으로 오독되므로, docker 환경에서는 CSV에 명시적으로 라벨을 남긴다.

실행: docker compose exec lab python tools/docker_env_probe.py
"""

from __future__ import annotations

import csv
import datetime
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)

# ── 환경 판별 ────────────────────────────────────────────────
def detect_env() -> str:
    if Path("/.dockerenv").exists() or os.environ.get("AI2_FORCE_ENV") == "docker_wsl2":
        return "docker_wsl2"
    return "bare_windows"


ENV = detect_env()
IS_DOCKER = ENV == "docker_wsl2"
SYSMEM_UNAVAILABLE_LABEL = "N/A(non-Windows container — WDDM sysmem fallback 감지 불가)"

# 컨테이너 안 hostname은 컨테이너 ID라 5대 비교 관례(hostname 기준)가 깨진다.
# 실제 호스트명을 넘겨받아 덮어쓴다: docker compose exec -e AI2_REAL_HOSTNAME=<hostname>
HOSTNAME = os.environ.get("AI2_REAL_HOSTNAME") or socket.gethostname()
START_TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

print(f"env: {ENV}")
print(f"hostname: {HOSTNAME}")
print(f"start_ts: {START_TS}")

import torch  # noqa: E402

TORCH_VERSION = torch.__version__
print(f"torch: {TORCH_VERSION}")
print(f"cuda available: {torch.cuda.is_available()}")


def nvidia_smi_query(fields: str) -> list[list[str]]:
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=10,
    ).stdout.strip()
    return [line.split(", ") for line in out.splitlines()]


GPU_NAME = nvidia_smi_query("name")[0][0]
print(f"gpu: {GPU_NAME}")

# ── 전력 로거 (1초 간격, 매 줄 flush+fsync) ────────────────────
POWER_LOG_PATH = REPO_ROOT / f"power_log_{HOSTNAME}_{ENV}_{START_TS}.csv"
POWER_LOG_FIELDS = ["timestamp", "power_draw_w", "temperature_c", "utilization_pct", "memory_used_mib"]
_power_log_stop = threading.Event()


def _power_log_loop() -> None:
    with open(POWER_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(POWER_LOG_FIELDS)
            f.flush()
            os.fsync(f.fileno())
        while not _power_log_stop.is_set():
            try:
                rows = nvidia_smi_query("power.draw,temperature.gpu,utilization.gpu,memory.used")
                ts = datetime.datetime.now().isoformat()
                for row in rows:
                    writer.writerow([ts, *row])
                f.flush()
                os.fsync(f.fileno())
            except Exception as exc:  # noqa: BLE001
                print("power log 폴링 실패(계속 시도함):", exc)
            _power_log_stop.wait(1.0)


_power_log_thread = threading.Thread(target=_power_log_loop, daemon=True)
_power_log_thread.start()
print(f"전력 로거 시작: {POWER_LOG_PATH}")

# ── 결과 CSV ────────────────────────────────────────────────
RESULTS_PATH = REPO_ROOT / f"results_{HOSTNAME}_{ENV}.csv"
RESULTS_FIELDS = [
    "hostname", "env", "gpu_name", "torch_version", "mode", "long_side", "crops", "crop_long_side",
    "visual_tokens", "peak_vram_torch", "peak_vram_smi", "peak_shared_mem_gb_windows",
    "sec_per_step_steady", "sec_per_step_warmup", "status",
    "fla_version", "triton_version", "gated_delta_rule_impl", "causal_conv1d_impl",
]


def _write_result_row(row: dict) -> None:
    is_new = not RESULTS_PATH.exists()
    with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in RESULTS_FIELDS})
        f.flush()
        os.fsync(f.fileno())


print(f"결과 CSV: {RESULTS_PATH}")


def _run_subprocess(args: list[str], timeout_sec: int = 900) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "src.run", *args],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=timeout_sec,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _classify_status(returncode: int, output: str) -> str:
    if returncode == 0:
        return "ok"
    if "OutOfMemoryError" in output or "CUDA out of memory" in output:
        return "OOM"
    return f"ERROR(exit={returncode})"


def _apply_kernel_impl(row: dict, kernel_impl: dict | None) -> None:
    if not kernel_impl:
        return
    row["fla_version"] = kernel_impl.get("fla_version")
    row["triton_version"] = kernel_impl.get("triton_version")
    row["gated_delta_rule_impl"] = kernel_impl.get("gated_delta_rule_impl")
    row["causal_conv1d_impl"] = kernel_impl.get("causal_conv1d_impl")


def _base_row(mode: str, long_side, crops, crop_long_side) -> dict:
    return {
        "hostname": HOSTNAME, "env": ENV, "gpu_name": GPU_NAME, "torch_version": TORCH_VERSION,
        "mode": mode, "long_side": long_side, "crops": crops, "crop_long_side": crop_long_side,
    }


def measure_infer(name: str, long_side=None, crops=0, crop_long_side=None, n=4) -> dict:
    args = ["smoke", "-c", "9b_base_1024", "--synthetic", "-n", str(n)]
    if long_side is not None:
        args += ["--set", f"input.long_side={long_side}"]
    args += ["--set", f"input.crops={crops}"]
    if crop_long_side is not None:
        args += ["--set", f"input.crop_long_side={crop_long_side}"]

    row = _base_row("infer", long_side, crops, crop_long_side)
    print(f"=== [infer] {name} ===")
    try:
        rc, output = _run_subprocess(args)
        status = _classify_status(rc, output)
        summary_path = REPO_ROOT / "runs" / "smoke_9b_base_1024" / "summary.json"
        if status == "ok" and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            row["visual_tokens"] = summary.get("visual_tokens")
            row["peak_vram_torch"] = summary.get("peak_vram_gb")
            _apply_kernel_impl(row, summary.get("kernel_impl"))
        row["status"] = status
        print(f"  -> {status}  peak_vram_torch={row.get('peak_vram_torch')}  visual_tokens={row.get('visual_tokens')}")
        if status != "ok":
            print(output[-2000:])
    except subprocess.TimeoutExpired:
        row["status"] = "TIMEOUT"
        print("  -> TIMEOUT")
    except Exception as exc:  # noqa: BLE001
        row["status"] = f"ERROR({type(exc).__name__})"
        print("  -> ERROR:", exc)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _write_result_row(row)
    return row


def measure_train(name: str, n_steps=12, grad_checkpointing=True,
                   crops=0, crop_long_side=None, long_side=1024) -> dict:
    args = ["train", "-c", "9b_qlora_r8_1024", "--synthetic", "-n", str(n_steps),
            "--set", f"input.long_side={long_side}",
            "--set", f"input.crops={crops}",
            "--set", f"train.grad_checkpointing={'true' if grad_checkpointing else 'false'}"]
    if crop_long_side is not None:
        args += ["--set", f"input.crop_long_side={crop_long_side}"]

    row = _base_row("train", long_side, crops, crop_long_side)
    print(f"=== [train] {name} ===")
    t_start = time.time()
    try:
        rc, output = _run_subprocess(args)
        elapsed = time.time() - t_start
        status = _classify_status(rc, output)
        summary_path = REPO_ROOT / "runs" / "trainprobe_9b_qlora_r8_1024" / "summary.json"
        if status == "ok" and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            probe = summary.get("train_probe", {})
            row["peak_vram_torch"] = probe.get("peak_vram_gb_torch")
            row["peak_vram_smi"] = probe.get("peak_vram_gb_nvidia_smi")
            row["sec_per_step_steady"] = probe.get("steady_state_sec_per_step_median")
            row["sec_per_step_warmup"] = probe.get("warmup_sec_per_step")
            sysmem = probe.get("peak_shared_mem_gb_windows")
            row["peak_shared_mem_gb_windows"] = (
                SYSMEM_UNAVAILABLE_LABEL if (IS_DOCKER and sysmem is None) else sysmem
            )
            _apply_kernel_impl(row, probe.get("kernel_impl"))
        elif IS_DOCKER:
            row["peak_shared_mem_gb_windows"] = SYSMEM_UNAVAILABLE_LABEL
        row["status"] = status
        row["_elapsed_sec"] = round(elapsed, 1)
        print(f"  -> {status}  elapsed={elapsed:.1f}s  peak_vram_torch={row.get('peak_vram_torch')}  "
              f"steady sec/step={row.get('sec_per_step_steady')}")
        if status != "ok":
            print(output[-2000:])
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t_start
        row["status"] = "TIMEOUT"
        row["_elapsed_sec"] = round(elapsed, 1)
        if IS_DOCKER:
            row["peak_shared_mem_gb_windows"] = SYSMEM_UNAVAILABLE_LABEL
        print(f"  -> TIMEOUT (elapsed={elapsed:.1f}s)")
    except Exception as exc:  # noqa: BLE001
        row["status"] = f"ERROR({type(exc).__name__})"
        print("  -> ERROR:", exc)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _write_result_row(row)
    return row


# ── 베어메탈 기준값과의 drift 체크 ──────────────────────────────
BASELINE_INFER_FULL1024_VRAM_GB = 14.0148
BASELINE_INFER_FULL1024_TOKENS = 768
BASELINE_TRAIN_FULL1024_VRAM_GB = 15.215
BASELINE_TRAIN_FULL1024_SEC_PER_STEP = 2.323
DRIFT_THRESHOLD = 0.20  # 20% 이상 벗어나면 대조군 실패로 간주


def _pct_diff(a: float, b: float) -> float:
    return abs(a - b) / b if b else float("inf")


def main() -> None:
    print("\n" + "=" * 60)
    print("[2](a) 추론 Full 1024 — 대조군")
    print("=" * 60)
    row_a = measure_infer("Full 1024 (대조군)", long_side=1024, crops=0)

    print("\n" + "=" * 60)
    print("[2](b) 학습 Full 1024 — 대조군")
    print("=" * 60)
    row_b = measure_train("Full 1024 (대조군)", grad_checkpointing=True, crops=0, long_side=1024)

    drift_report = []
    if row_a.get("status") == "ok" and row_a.get("peak_vram_torch"):
        d = _pct_diff(float(row_a["peak_vram_torch"]), BASELINE_INFER_FULL1024_VRAM_GB)
        drift_report.append(f"infer VRAM drift: {d*100:.1f}% "
                             f"(docker={row_a['peak_vram_torch']} vs bare={BASELINE_INFER_FULL1024_VRAM_GB})")
    if row_b.get("status") == "ok" and row_b.get("peak_vram_torch") and row_b.get("sec_per_step_steady"):
        d_vram = _pct_diff(float(row_b["peak_vram_torch"]), BASELINE_TRAIN_FULL1024_VRAM_GB)
        d_speed = _pct_diff(float(row_b["sec_per_step_steady"]), BASELINE_TRAIN_FULL1024_SEC_PER_STEP)
        drift_report.append(f"train VRAM drift: {d_vram*100:.1f}% "
                             f"(docker={row_b['peak_vram_torch']} vs bare={BASELINE_TRAIN_FULL1024_VRAM_GB})")
        drift_report.append(f"train sec/step drift: {d_speed*100:.1f}% "
                             f"(docker={row_b['sec_per_step_steady']} vs bare={BASELINE_TRAIN_FULL1024_SEC_PER_STEP})")

        print("\n--- 대조군 drift ---")
        for line in drift_report:
            print(" ", line)

        if d_vram > DRIFT_THRESHOLD or d_speed > DRIFT_THRESHOLD:
            print(f"\n*** 대조군이 베어메탈과 {DRIFT_THRESHOLD*100:.0f}% 넘게 다릅니다 ***")
            print("*** (c) Full 1536은 건너뜁니다 — 이 상태로는 결과를 해석할 기준이 없습니다 ***")
            _finish()
            return
    else:
        print("\n*** 대조군(b) 학습 Full 1024가 ok로 끝나지 않았습니다 — 여기서 중단합니다 ***")
        _finish()
        return

    print("\n" + "=" * 60)
    print("[2](c) 학습 Full 1536 — 미측정 핵심 값")
    print("=" * 60)
    measure_train("Full 1536 (핵심 미측정값)", grad_checkpointing=True, crops=0, long_side=1536)

    print("\n" + "=" * 60)
    print("[2](d) 학습 crops=4 @768 — 베어메탈은 TIMEOUT(폴백)")
    print("=" * 60)
    measure_train("crops=4, crop_long_side=768", grad_checkpointing=True, crops=4, crop_long_side=768, long_side=1024)

    _finish()


def _finish() -> None:
    _power_log_stop.set()
    _power_log_thread.join(timeout=5)
    print("\n전력 로거 정지.")
    print("결과 파일:", RESULTS_PATH.resolve())
    print("전력 로그:", POWER_LOG_PATH.resolve())


if __name__ == "__main__":
    main()
