"""RunPod 제어와 Budget Guard.

설계 전제: API Key는 Jetson control plane에만 둔다(Restricted Key 우선). 이 모듈은
환경변수에서만 키를 읽고 어디에도 기록하지 않는다.

Budget Guard는 파드를 띄우기 전에 강제된다 — MAX_COST / MAX_GPU_COUNT /
MAX_RUNTIME 중 하나라도 넘으면 생성 자체를 거부한다. 큰 GPU를 계속 바쁘게 하는 게
목표가 아니라, 다음 비싼 실험의 불확실성을 줄이는 것이 목표다.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

API_BASE = os.environ.get("RUNPOD_ENDPOINT", "https://rest.runpod.io/v1")


class BudgetExceeded(RuntimeError):
    pass


class RunPodError(RuntimeError):
    pass


@dataclass
class BudgetGuard:
    max_cost_krw: float
    max_gpu_count: int
    max_runtime_hours: float
    usd_krw: float = 1370.0
    spent_krw: float = 0.0

    @classmethod
    def from_env(cls) -> "BudgetGuard":
        return cls(
            max_cost_krw=float(os.environ.get("AI2_MAX_COST_KRW", 1_000_000)),
            max_gpu_count=int(os.environ.get("AI2_MAX_GPU_COUNT", 2)),
            max_runtime_hours=float(os.environ.get("AI2_MAX_RUNTIME_HOURS", 12)),
            usd_krw=float(os.environ.get("AI2_USD_KRW", 1370)),
        )

    @classmethod
    def from_config(cls, cfg) -> "BudgetGuard":
        guard = cls.from_env()
        res = cfg.get("resource_config.budget_guard") or {}
        if v := res.get("max_cost_krw"):
            guard.max_cost_krw = min(guard.max_cost_krw, float(v))
        if v := res.get("max_runtime_hours"):
            guard.max_runtime_hours = min(guard.max_runtime_hours, float(v))
        if v := cfg.get("resource_config.gpu.count"):
            guard.max_gpu_count = min(guard.max_gpu_count, int(v))
        return guard

    def estimate_krw(self, usd_per_hour: float, gpu_count: int, hours: float) -> float:
        return usd_per_hour * gpu_count * hours * self.usd_krw

    def check(self, usd_per_hour: float, gpu_count: int, hours: float) -> dict[str, Any]:
        cost = self.estimate_krw(usd_per_hour, gpu_count, hours)
        if gpu_count > self.max_gpu_count:
            raise BudgetExceeded(f"GPU {gpu_count}장 > 상한 {self.max_gpu_count}장")
        if hours > self.max_runtime_hours:
            raise BudgetExceeded(f"런타임 {hours}h > 상한 {self.max_runtime_hours}h")
        if self.spent_krw + cost > self.max_cost_krw:
            raise BudgetExceeded(
                f"예상 누적 {self.spent_krw + cost:,.0f}원 > 상한 {self.max_cost_krw:,.0f}원"
            )
        return {"estimated_krw": round(cost), "remaining_krw": round(self.max_cost_krw - self.spent_krw - cost)}

    def commit(self, krw: float) -> None:
        self.spent_krw += krw


class RunPodClient:
    def __init__(self, api_key: str | None = None, timeout: int = 30):
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        self.timeout = timeout
        if not self.api_key:
            raise RunPodError("RUNPOD_API_KEY가 없다. .env를 채우거나 Jetson에서 실행해라.")

    def _request(self, method: str, path: str, **kwargs) -> Any:
        import requests

        resp = requests.request(
            method,
            f"{API_BASE}{path}",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            timeout=self.timeout,
            **kwargs,
        )
        if resp.status_code >= 400:
            raise RunPodError(f"{method} {path} → {resp.status_code}: {resp.text[:400]}")
        return resp.json() if resp.content else None

    def list_pods(self) -> list[dict[str, Any]]:
        data = self._request("GET", "/pods")
        return data if isinstance(data, list) else data.get("pods", [])

    def create_pod(self, spec: dict[str, Any], guard: BudgetGuard,
                   usd_per_hour: float, gpu_count: int, hours: float) -> dict[str, Any]:
        """Budget Guard를 통과해야만 실제 생성 요청을 보낸다."""
        estimate = guard.check(usd_per_hour, gpu_count, hours)
        pod = self._request("POST", "/pods", json=spec)
        guard.commit(estimate["estimated_krw"])
        return {"pod": pod, **estimate}

    def stop_pod(self, pod_id: str) -> Any:
        return self._request("POST", f"/pods/{pod_id}/stop")

    def terminate_pod(self, pod_id: str) -> Any:
        return self._request("DELETE", f"/pods/{pod_id}")

    def wait_ready(self, pod_id: str, timeout_sec: int = 900, poll_sec: int = 15) -> dict[str, Any]:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            pod = self._request("GET", f"/pods/{pod_id}")
            if str(pod.get("desiredStatus", "")).upper() == "RUNNING":
                return pod
            time.sleep(poll_sec)
        raise RunPodError(f"{pod_id}가 {timeout_sec}초 안에 RUNNING이 되지 않았다")


def auto_downshift(cfg, reason: str) -> str | None:
    """예산·재고 문제로 비싼 자원을 못 쓸 때 대체 자원 이름을 돌려준다.

    조용히 내려가지 않는다 — 호출한 쪽이 반드시 로그로 남기고, 내려간 자원에서
    나온 수치는 원래 자원의 수치와 같은 표에 섞지 않는다.
    """
    fallback = cfg.get("resource_config.budget_guard.auto_downshift_to")
    if not fallback:
        return None
    return str(fallback)


def stop_all_idle(client: RunPodClient, idle_minutes: int = 20) -> list[str]:
    """데이터 분석·코드 작성 중에 파드를 켜둔 채로 두지 않기 위한 안전장치."""
    stopped: list[str] = []
    for pod in client.list_pods():
        if str(pod.get("desiredStatus", "")).upper() != "RUNNING":
            continue
        idle = pod.get("lastStatusChange") or 0
        if isinstance(idle, (int, float)) and (time.time() - idle) > idle_minutes * 60:
            client.stop_pod(pod["id"])
            stopped.append(pod["id"])
    return stopped
