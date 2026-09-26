"""Join 3B/9B local scores and prioritize still-unreviewed dev rows."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    root = here / "generated" / "dev_split_markers_v1" / "mine" / "local_vlm_drafts"
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewed", type=Path, default=root / "codex_review" / "mine_codex_reviewed.csv")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=root / "codex_review" / "multimodel_queue")
    parser.add_argument(
        "--regenerated",
        type=Path,
        default=root.parent / "regenerated_questions" / "pilot_20.csv",
        help="Optional regenerated-question CSV whose source_id rows should leave the pending queue.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.reviewed.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    predictions: dict[str, dict] = {}
    with args.predictions.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                predictions[row["id"]] = row
    ids = {row["id"] for row in rows}
    if ids != set(predictions):
        raise SystemExit(
            f"prediction id mismatch: missing={len(ids - set(predictions))}, extra={len(set(predictions) - ids)}"
        )

    regenerated_sources: set[str] = set()
    if args.regenerated.is_file():
        with args.regenerated.open(encoding="utf-8-sig", newline="") as handle:
            regenerated_sources = {
                row["source_id"] for row in csv.DictReader(handle) if row.get("source_id")
            }
        unknown_sources = regenerated_sources - ids
        if unknown_sources:
            raise SystemExit(f"unknown regenerated source ids: {sorted(unknown_sources)[:10]}")

    enriched: list[dict[str, str]] = []
    for row in rows:
        item = dict(row)
        pred = predictions[row["id"]]
        p9 = str(pred["pred"]).lower()
        c9 = float(pred["confidence"])
        p3 = row["model_candidate_answer"].lower()
        c3 = float(row["model_confidence_score"])
        agreed = p3 == p9
        floor = min(c3, c9)
        if not agreed:
            priority = "P0_disagree"
        elif floor < 0.50:
            priority = "P1_agree_low"
        elif floor < 0.75:
            priority = "P2_agree_medium"
        else:
            priority = "P3_agree_high"
        item.update(
            qwen9b_candidate_answer=p9,
            qwen9b_confidence_score=f"{c9:.6f}",
            qwen9b_choice_logprobs=json.dumps(pred.get("logprobs", []), ensure_ascii=False),
            model_agreement=str(agreed).lower(),
            multimodel_confidence_floor=f"{floor:.6f}",
            multimodel_review_priority=priority,
            regenerated_question=str(row["id"] in regenerated_sources).lower(),
        )
        enriched.append(item)

    order = {"P0_disagree": 0, "P1_agree_low": 1, "P2_agree_medium": 2, "P3_agree_high": 3}
    pending = [
        row
        for row in enriched
        if row["reviewer"] != "codex-assisted-visual" and row["id"] not in regenerated_sources
    ]
    pending.sort(
        key=lambda row: (
            order[row["multimodel_review_priority"]],
            float(row["multimodel_confidence_floor"]),
            row["id"],
        )
    )
    for rank, row in enumerate(pending, start=1):
        row["multimodel_review_rank"] = str(rank)
    for row in enriched:
        row.setdefault("multimodel_review_rank", "")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(enriched[0])

    def write(path: Path, selected: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected)

    write(args.output_dir / "all_with_9b.csv", enriched)
    write(args.output_dir / "pending_priority.csv", pending)
    summary = {
        "rows": len(enriched),
        "already_reviewed": sum(row["reviewer"] == "codex-assisted-visual" for row in enriched),
        "regenerated_sources": len(regenerated_sources),
        "handled_sources": len(enriched) - len(pending),
        "pending": len(pending),
        "agreement": sum(row["model_agreement"] == "true" for row in enriched),
        "disagreement": sum(row["model_agreement"] == "false" for row in enriched),
        "pending_priority_counts": dict(Counter(row["multimodel_review_priority"] for row in pending)),
        "automatic_approval": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
