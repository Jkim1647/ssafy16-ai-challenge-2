"""Cross-model 비교표 생성 (Notion 11번 문서의 원천).

runs/*/summary.json + predictions.jsonl 을 읽어 한 장의 표로 만든다.
컬럼: Accuracy / 유형별 Accuracy / NetGain / GPU-hours / Cost / Throughput.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .evaluate import Prediction, load_predictions, net_gain, per_type_accuracy, summarize
from .tracker import iter_runs, load_summary


@dataclass
class RunRow:
    run_id: str
    experiment: str
    model: str
    resource: str
    task: str
    status: str
    accuracy: float | None
    macro_accuracy: float | None
    per_type: dict[str, Any]
    gpu_hours: float | None
    cost_krw: float | None
    sec_per_sample: float | None
    peak_vram_gb: float | None
    net_gain: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment": self.experiment,
            "model": self.model,
            "resource": self.resource,
            "task": self.task,
            "status": self.status,
            "accuracy": self.accuracy,
            "macro_accuracy": self.macro_accuracy,
            "gpu_hours": self.gpu_hours,
            "cost_krw": self.cost_krw,
            "sec_per_sample": self.sec_per_sample,
            "peak_vram_gb": self.peak_vram_gb,
            "per_type": self.per_type,
            "net_gain": self.net_gain,
        }


def _resource_rate(run_dir: Path) -> float:
    """resolved_config.yaml에서 시간당 원화 단가를 읽는다. 로컬은 0이다."""
    cfg_path = run_dir / "resolved_config.yaml"
    if not cfg_path.exists():
        return 0.0
    try:
        import yaml
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        cost = (raw.get("resource_config") or {}).get("cost") or {}
        rate = cost.get("krw_per_hour")
        count = ((raw.get("resource_config") or {}).get("gpu") or {}).get("count", 1)
        return float(rate or 0.0) * (1 if rate else float(count or 1) * 0)
    except Exception:  # noqa: BLE001 — 비교표가 설정 파싱 때문에 죽으면 안 된다
        return 0.0


def collect(runs_dir: Path) -> list[RunRow]:
    rows: list[RunRow] = []
    for run_dir in iter_runs(runs_dir):
        summary = load_summary(run_dir)
        if not summary:
            continue

        preds = []
        pred_path = run_dir / "predictions.jsonl"
        if pred_path.exists():
            preds = load_predictions(pred_path)

        report = summarize(preds) if preds else {}
        elapsed = float(summary.get("elapsed_sec") or 0.0)
        gpu_hours = round(elapsed / 3600, 4) if elapsed else None
        rate = _resource_rate(run_dir)
        cost = round(gpu_hours * rate) if (gpu_hours and rate) else (0 if gpu_hours else None)

        block = summary.get("infer") or summary.get("train") or {}
        rows.append(
            RunRow(
                run_id=summary.get("run_id", run_dir.name),
                experiment=summary.get("experiment", ""),
                model=summary.get("model", ""),
                resource=summary.get("resource", ""),
                task=summary.get("task", ""),
                status=summary.get("status", "unknown"),
                accuracy=report.get("accuracy") or block.get("accuracy") or block.get("best_accuracy"),
                macro_accuracy=report.get("macro_accuracy"),
                per_type=report.get("per_type_accuracy", {}),
                gpu_hours=gpu_hours,
                cost_krw=cost,
                sec_per_sample=block.get("sec_per_sample"),
                peak_vram_gb=block.get("peak_vram_gb"),
            )
        )
    return rows


def add_net_gain(rows: list[RunRow], runs_dir: Path, baseline_experiment: str) -> list[RunRow]:
    """baseline 실험 대비 NetGain을 각 행에 붙인다."""
    base_row = next((r for r in rows if r.experiment == baseline_experiment), None)
    if base_row is None:
        return rows
    base_preds = load_predictions(runs_dir / base_row.run_id / "predictions.jsonl")
    base_cost = base_row.cost_krw or 0

    for row in rows:
        if row.run_id == base_row.run_id:
            continue
        path = runs_dir / row.run_id / "predictions.jsonl"
        if not path.exists():
            continue
        extra = max(0, (row.cost_krw or 0) - base_cost)
        row.net_gain = net_gain(base_preds, load_predictions(path), extra or None)
    return rows


def to_markdown(rows: Sequence[RunRow], types: Sequence[str] = ()) -> str:
    """Notion 11번 문서에 그대로 붙일 수 있는 표."""
    head = ["실험", "모델", "자원", "Accuracy", "Macro", "NetGain", "GPU-h", "Cost(원)", "sec/sample", "Peak VRAM"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in sorted(rows, key=lambda x: (x.accuracy or -1), reverse=True):
        ng = r.net_gain.get("net_correct") if r.net_gain else None
        lines.append("| " + " | ".join([
            r.experiment or r.run_id,
            r.model or "—",
            r.resource or "—",
            f"{r.accuracy:.4f}" if r.accuracy is not None else "—",
            f"{r.macro_accuracy:.4f}" if r.macro_accuracy is not None else "—",
            str(ng) if ng is not None else "—",
            f"{r.gpu_hours:.2f}" if r.gpu_hours is not None else "—",
            f"{r.cost_krw:,}" if r.cost_krw else "0",
            f"{r.sec_per_sample:.3f}" if r.sec_per_sample is not None else "—",
            f"{r.peak_vram_gb:.1f}" if r.peak_vram_gb is not None else "—",
        ]) + " |")

    if types:
        lines += ["", "### 유형별 Accuracy", ""]
        head2 = ["실험", *types]
        lines += ["| " + " | ".join(head2) + " |", "|" + "---|" * len(head2)]
        for r in rows:
            cells = [f"{r.per_type[t]['accuracy']:.4f}" if t in r.per_type else "—" for t in types]
            lines.append("| " + " | ".join([r.experiment or r.run_id, *cells]) + " |")
    return "\n".join(lines)


def run(runs_dir: Path, out_dir: Path, baseline: str | None = None,
        types: Sequence[str] = ()) -> dict[str, Any]:
    rows = collect(runs_dir)
    if baseline:
        rows = add_net_gain(rows, runs_dir, baseline)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"baseline": baseline, "n_runs": len(rows), "rows": [r.to_dict() for r in rows]}
    (out_dir / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "comparison.md").write_text(to_markdown(rows, types), encoding="utf-8")
    return payload
