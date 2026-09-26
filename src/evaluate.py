"""채점: Accuracy, 유형별 Accuracy, NetGain, disagreement.

NetGain을 여기에 두는 이유: "accuracy가 올랐는가"가 아니라 "추가 GPU 비용당 몇 문제를
더 맞혔는가"가 자원 배분의 실제 판단 기준이기 때문이다.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass
class Prediction:
    id: str
    pred: str
    gold: str | None = None
    confidence: float | None = None
    logprobs: list[float] | None = None
    qtype: str | None = None

    @property
    def correct(self) -> bool | None:
        if self.gold is None:
            return None
        return self.pred.strip().upper() == self.gold.strip().upper()


def load_predictions(path: Path) -> list[Prediction]:
    out: list[Prediction] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out.append(
                Prediction(
                    id=str(row.get("id")),
                    pred=str(row.get("pred", "")),
                    gold=row.get("gold"),
                    confidence=row.get("confidence"),
                    logprobs=row.get("logprobs"),
                    qtype=row.get("type"),
                )
            )
    return out


def accuracy(preds: Sequence[Prediction]) -> float:
    scored = [p for p in preds if p.gold is not None]
    if not scored:
        return 0.0
    return sum(1 for p in scored if p.correct) / len(scored)


def per_type_accuracy(preds: Sequence[Prediction]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[Prediction]] = defaultdict(list)
    for p in preds:
        if p.gold is not None:
            buckets[p.qtype or "unknown"].append(p)
    return {
        qtype: {"n": len(group), "accuracy": round(accuracy(group), 6)}
        for qtype, group in sorted(buckets.items())
    }


def macro_accuracy(preds: Sequence[Prediction]) -> float:
    per_type = per_type_accuracy(preds)
    if not per_type:
        return 0.0
    return sum(v["accuracy"] for v in per_type.values()) / len(per_type)


def confidence_stats(preds: Sequence[Prediction]) -> dict[str, Any]:
    scored = [p for p in preds if p.confidence is not None and p.gold is not None]
    if not scored:
        return {}
    correct = [p.confidence for p in scored if p.correct]
    wrong = [p.confidence for p in scored if not p.correct]
    mean = lambda xs: round(sum(xs) / len(xs), 6) if xs else None  # noqa: E731
    return {
        "mean_confidence": mean([p.confidence for p in scored]),
        "mean_confidence_correct": mean(correct),
        "mean_confidence_wrong": mean(wrong),
        # 이 값이 0에 가까우면 confidence 기반 선택적 앙상블·재추론의 근거가 약해진다
        "separation": round((mean(correct) or 0) - (mean(wrong) or 0), 6),
    }


def net_gain(
    baseline: Sequence[Prediction],
    candidate: Sequence[Prediction],
    extra_cost_krw: float | None = None,
) -> dict[str, Any]:
    """baseline 대비 candidate가 추가로 맞힌 문제 수와, 비용당 이득."""
    base = {p.id: p for p in baseline if p.gold is not None}
    cand = {p.id: p for p in candidate if p.gold is not None}
    shared = sorted(set(base) & set(cand))

    fixed = sum(1 for i in shared if not base[i].correct and cand[i].correct)
    broken = sum(1 for i in shared if base[i].correct and not cand[i].correct)
    net = fixed - broken

    out: dict[str, Any] = {
        "n_compared": len(shared),
        "baseline_accuracy": round(accuracy([base[i] for i in shared]), 6),
        "candidate_accuracy": round(accuracy([cand[i] for i in shared]), 6),
        "fixed": fixed,
        "broken": broken,
        "net_correct": net,
        "changed_ratio": round(sum(1 for i in shared if base[i].pred != cand[i].pred) / len(shared), 6)
        if shared else 0.0,
    }
    if extra_cost_krw:
        out["extra_cost_krw"] = extra_cost_krw
        out["net_gain_per_100k_krw"] = round(net / (extra_cost_krw / 100_000), 4)
    return out


def disagreement(runs: dict[str, Sequence[Prediction]]) -> dict[str, Any]:
    """모델 간 예측 불일치. 선택적 고해상도 재추론 대상을 고르는 입력이다."""
    by_id: dict[str, dict[str, str]] = defaultdict(dict)
    for run_name, preds in runs.items():
        for p in preds:
            by_id[p.id][run_name] = p.pred

    full = {i: v for i, v in by_id.items() if len(v) == len(runs)}
    disagreed = [i for i, v in full.items() if len(set(v.values())) > 1]
    return {
        "n_compared": len(full),
        "n_disagree": len(disagreed),
        "disagree_ratio": round(len(disagreed) / len(full), 6) if full else 0.0,
        "ids": sorted(disagreed),
    }


def select_retry_targets(
    preds: Sequence[Prediction],
    confidence_threshold: float = 0.6,
    disagree_ids: Iterable[str] = (),
    types: Sequence[str] = (),
) -> list[str]:
    """선택적 고해상도 재추론 대상. 전체 test를 다시 돌리지 않기 위한 필터다."""
    want_types = {t.lower() for t in types}
    disagreed = set(disagree_ids)
    out: list[str] = []
    for p in preds:
        if p.id in disagreed:
            out.append(p.id)
        elif p.confidence is not None and p.confidence < confidence_threshold:
            out.append(p.id)
        elif want_types and (p.qtype or "").lower() in want_types:
            out.append(p.id)
    return sorted(set(out))


def public_overfit_flags(
    local_scores: dict[str, float],
    previous_local: dict[str, float],
    public_delta: float,
    changed_ratio: float,
    per_type_now: dict[str, dict[str, Any]],
    per_type_prev: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Public 과적합 판단. 하나라도 걸리면 그 제출은 의심 대상이다."""
    dropped = [k for k, v in local_scores.items() if v < previous_local.get(k, v)]
    type_drops = [
        k for k, v in per_type_now.items()
        if k in per_type_prev and v["accuracy"] < per_type_prev[k]["accuracy"] - 0.03
    ]
    flags = {
        "public_up_local_down": public_delta > 0 and len(dropped) >= 2,
        "too_many_changes": changed_ratio > 0.15,
        "type_accuracy_collapse": bool(type_drops),
    }
    return {
        "flags": flags,
        "suspect": any(flags.values()),
        "dropped_validations": dropped,
        "dropped_types": type_drops,
    }


def summarize(preds: Sequence[Prediction]) -> dict[str, Any]:
    return {
        "n": len(preds),
        "n_scored": sum(1 for p in preds if p.gold is not None),
        "accuracy": round(accuracy(preds), 6),
        "macro_accuracy": round(macro_accuracy(preds), 6),
        "per_type_accuracy": per_type_accuracy(preds),
        "confidence": confidence_stats(preds),
    }


def main(run_dir: Path, baseline_dir: Path | None = None) -> dict[str, Any]:
    preds = load_predictions(run_dir / "predictions.jsonl")
    report = summarize(preds)
    if baseline_dir:
        base = load_predictions(baseline_dir / "predictions.jsonl")
        report["net_gain_vs_baseline"] = net_gain(base, preds)
    out_path = run_dir / "evaluation.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


__all__ = [
    "Prediction", "accuracy", "macro_accuracy", "per_type_accuracy", "confidence_stats",
    "net_gain", "disagreement", "select_retry_targets", "public_overfit_flags",
    "summarize", "load_predictions", "main", "asdict",
]
