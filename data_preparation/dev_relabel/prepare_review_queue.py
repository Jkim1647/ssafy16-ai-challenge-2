"""Prioritize local-VLM drafts for direct image review without approving them."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_dir = here / "generated" / "dev_split_markers_v1" / "mine" / "local_vlm_drafts"
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=default_dir / "mine_model_draft.csv")
    parser.add_argument("--output-dir", type=Path, default=default_dir / "review_queue")
    return parser.parse_args()


def probability_metrics(row: dict[str, str]) -> tuple[float, float]:
    probabilities = [float(value) for value in json.loads(row["model_choice_probabilities"]).values()]
    ordered = sorted(probabilities, reverse=True)
    margin = ordered[0] - ordered[1]
    entropy = -sum(value * math.log(value) for value in probabilities if value > 0.0)
    return margin, entropy


def main() -> None:
    args = parse_args()
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("input CSV is empty")
    if any(row["assignment"] != "mine" for row in rows):
        raise SystemExit("input contains rows outside assignment=mine")
    if any(row["human_answer"].strip() for row in rows):
        raise SystemExit("draft queue unexpectedly contains nonblank human_answer")

    priority = {"low": 0, "medium": 1, "high": 2}
    enriched: list[dict[str, str]] = []
    for row in rows:
        item = dict(row)
        margin, entropy = probability_metrics(row)
        item["model_margin"] = f"{margin:.6f}"
        item["model_entropy"] = f"{entropy:.6f}"
        item["review_priority"] = {
            "low": "P0_low",
            "medium": "P1_medium",
            "high": "P2_high",
        }[row["model_confidence"]]
        enriched.append(item)

    enriched.sort(
        key=lambda row: (
            priority[row["model_confidence"]],
            float(row["model_confidence_score"]),
            float(row["model_margin"]),
            row["id"],
        )
    )
    for rank, row in enumerate(enriched, start=1):
        row["review_rank"] = str(rank)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(enriched[0])

    def write_csv(path: Path, selected: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected)

    write_csv(args.output_dir / "review_queue_all.csv", enriched)
    for band in ("low", "medium", "high"):
        write_csv(
            args.output_dir / f"review_queue_{band}.csv",
            [row for row in enriched if row["model_confidence"] == band],
        )

    summary = {
        "rows": len(enriched),
        "assignment": "mine",
        "approval_state": "needs_review_only",
        "confidence_counts": dict(Counter(row["model_confidence"] for row in enriched)),
        "candidate_answer_counts": dict(Counter(row["model_candidate_answer"] for row in enriched)),
        "review_order": ["P0_low", "P1_medium", "P2_high"],
        "human_answer_nonblank": sum(bool(row["human_answer"].strip()) for row in enriched),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
