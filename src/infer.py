"""추론 실행. logit을 전부 저장해 재학습 없이 앙상블을 재탐색할 수 있게 한다."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

from tqdm import tqdm

from . import resources
from .config import Config
from .data import Sample, build_views, load_named_split, subset
from .models import build_adapter
from .tracker import Tracker


def _ocr_text(sample: Sample, cfg: Config) -> str | None:
    if not cfg.get("input.ocr.enabled"):
        return None
    cache_dir = cfg.path("input.ocr.cache", "runs/_ocr")
    path = cache_dir / f"{sample.image_path.stem}.json"
    if not path.exists():
        return None
    try:
        blocks = json.loads(path.read_text(encoding="utf-8")).get("blocks", [])
    except json.JSONDecodeError:
        return None
    # OCR은 최종 답 생성기가 아니라 perception 보조다. confidence는 그대로 넘기지 않는다.
    return "\n".join(b.get("text", "") for b in blocks if b.get("text"))


def run(cfg: Config, tracker: Tracker, split: str = "dev",
        samples: Sequence[Sample] | None = None, adapter=None) -> dict[str, Any]:
    if samples is None:
        samples, schema = load_named_split(cfg, split)
        tracker.event("schema_resolved", split=split, **schema.to_dict())
        samples = subset(
            list(samples),
            cfg.get("subset.size"),
            cfg.seed,
            cfg.get("subset.stratify_by"),
        )

    own_adapter = adapter is None
    if own_adapter:
        adapter = build_adapter(cfg)
        resources.preflight(cfg, tracker)
        t0 = time.time()
        adapter.ensure_loaded(for_training=False)
        tracker.event("model_loaded", seconds=round(time.time() - t0, 2),
                      vram_gb=resources.peak_gb())

    method = str(cfg.get("decode.method", "choice_loglikelihood"))
    save_logits = bool(cfg.get("infer.save_logits", True))

    n_correct = n_scored = 0
    visual_tokens_seen: list[int] = []
    t_start = time.time()

    resources.reset_peak()
    for sample in tqdm(samples, desc=f"infer:{split}", unit="sample"):
        views = build_views(sample, cfg)
        row: dict[str, Any] = {"id": sample.id, "type": sample.qtype, "gold": sample.answer}

        if method == "generate":
            text = adapter.generate(sample, views, _ocr_text(sample, cfg))
            row["pred"] = _normalize(text, adapter.choice_letters(sample))
            row["raw"] = text
        else:
            scored = adapter.score_choices(sample, views, _ocr_text(sample, cfg))
            row["pred"] = scored.predicted_letter
            row["confidence"] = scored.confidence
            if save_logits:
                row["logprobs"] = [round(v, 6) for v in scored.logprobs]
            if scored.visual_tokens:
                row["visual_tokens"] = scored.visual_tokens
                visual_tokens_seen.append(scored.visual_tokens)

        if sample.answer is not None:
            n_scored += 1
            n_correct += int(row["pred"].strip().upper() == sample.answer.strip().upper())

        tracker.prediction(**row)

    elapsed = time.time() - t_start
    result = {
        "split": split,
        "n": len(samples),
        "n_scored": n_scored,
        "accuracy": round(n_correct / n_scored, 6) if n_scored else None,
        "sec_per_sample": round(elapsed / max(1, len(samples)), 4),
        "elapsed_sec": round(elapsed, 2),
        "peak_vram_gb": resources.peak_gb(),
    }
    if visual_tokens_seen:
        # 계산 규칙 검증의 핵심: config 계산값과 processor 실측값이 같은지 확인한다
        result["visual_tokens"] = {
            "min": min(visual_tokens_seen),
            "median": sorted(visual_tokens_seen)[len(visual_tokens_seen) // 2],
            "max": max(visual_tokens_seen),
        }
    tracker.metric(**result)
    tracker.set_summary(infer=result)
    return result


def _normalize(text: str, letters: Sequence[str]) -> str:
    """자유 생성 출력에서 선택지 기호를 뽑아낸다."""
    upper = text.strip().upper()
    for letter in letters:
        if upper.startswith(letter.upper()):
            return letter
    for letter in letters:
        if letter.upper() in upper:
            return letter
    return letters[0] if letters else ""


def write_submission(run_dir: Path, out_path: Path, id_col: str = "id",
                     answer_col: str = "answer") -> Path:
    """predictions.jsonl → 제출 CSV. 누락·중복·형식을 여기서 검사한다."""
    import pandas as pd

    rows = []
    with (run_dir / "predictions.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                rows.append({id_col: row["id"], answer_col: row["pred"]})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("예측이 비어 있다")
    if df[id_col].duplicated().any():
        dupes = df[df[id_col].duplicated()][id_col].tolist()[:10]
        raise ValueError(f"중복 ID가 있다: {dupes}")
    if df[answer_col].isna().any() or (df[answer_col].astype(str).str.strip() == "").any():
        raise ValueError("빈 답이 있다")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Kaggle 제출: BOM 금지. utf-8-sig로 바꾸지 말 것 — 채점기가 BOM을 첫 ID의
    # 일부로 읽어 행이 통째로 매칭 실패할 수 있다. 제출은 사람이 여는 파일이 아니다.
    df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path
