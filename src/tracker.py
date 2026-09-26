"""run 디렉터리와 로그 파일 관리.

runs/<run_id>/
  resolved_config.yaml   해석이 끝난 설정 전체
  events.jsonl           사람이 읽는 실행 로그 (시작, 로드, OOM, 중단 등)
  metrics.jsonl          step/epoch 단위 수치
  predictions.jsonl      샘플별 예측 + logit (앙상블 재탐색용)
  summary.json           최종 요약 — compare.py가 이것만 읽는다
  checkpoints/           adapter / merged weight
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from .config import Config


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def make_run_id(cfg: Config) -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{cfg.name}"


class Tracker:
    """한 번의 실행에 대응한다. with 블록으로 쓴다."""

    def __init__(self, cfg: Config, run_id: str | None = None):
        self.cfg = cfg
        self.run_id = run_id or make_run_id(cfg)
        self.dir = cfg.path("tracker.runs_dir", "runs") / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "checkpoints").mkdir(exist_ok=True)
        self._files: dict[str, TextIO] = {}
        self._start = time.time()
        self._summary: dict[str, Any] = {
            "run_id": self.run_id,
            "experiment": cfg.name,
            "task": cfg.task,
            "model": cfg.get("model"),
            "resource": cfg.get("resource"),
            "seed": cfg.seed,
            "git_commit": _git_commit(),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

    def __enter__(self) -> "Tracker":
        self.cfg.dump(self.dir / "resolved_config.yaml")
        self.event("run_start", python=platform.python_version(), host=platform.node())
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.event("run_failed", error=f"{exc_type.__name__}: {exc}")
            self._summary["status"] = "failed"
            self._summary["error"] = f"{exc_type.__name__}: {exc}"
        else:
            self._summary.setdefault("status", "ok")
        self._summary["elapsed_sec"] = round(time.time() - self._start, 2)
        self._summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        (self.dir / "summary.json").write_text(
            json.dumps(self._summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for fh in self._files.values():
            fh.close()
        self._files.clear()

    def _fh(self, name: str) -> TextIO:
        if name not in self._files:
            self._files[name] = (self.dir / f"{name}.jsonl").open("a", encoding="utf-8")
        return self._files[name]

    def _write(self, name: str, payload: dict[str, Any]) -> None:
        fh = self._fh(name)
        fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        fh.flush()

    def event(self, kind: str, **fields: Any) -> None:
        self._write("events", {"ts": datetime.now(timezone.utc).isoformat(), "event": kind, **fields})

    def metric(self, step: int | None = None, **values: Any) -> None:
        self._write("metrics", {"ts": time.time(), "step": step, **values})

    def prediction(self, **fields: Any) -> None:
        """샘플 하나의 예측. logit을 반드시 함께 남긴다."""
        self._write("predictions", fields)

    def set_summary(self, **fields: Any) -> None:
        self._summary.update(fields)

    def artifact(self, name: str) -> Path:
        return self.dir / name


def load_summary(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "summary.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def iter_runs(runs_dir: Path):
    if not runs_dir.exists():
        return
    for child in sorted(runs_dir.iterdir()):
        if child.is_dir() and not child.name.startswith("_"):
            yield child
