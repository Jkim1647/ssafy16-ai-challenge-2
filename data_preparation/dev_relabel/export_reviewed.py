"""Export only fully reviewed, clear dev rows as VQA training candidates."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ANSWER_KEYS = {"a", "b", "c", "d"}
TRAIN_FIELDS = ("id", "path", "question", "a", "b", "c", "d", "human_answer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"출력 디렉터리가 비어 있지 않습니다: {output}")

    rows: list[dict[str, str]] = []
    source_by_id: dict[str, Path] = {}
    for input_path in args.input:
        path = input_path.resolve()
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                sample_id = row["id"].strip()
                if sample_id in source_by_id:
                    raise SystemExit(
                        f"입력 사이에 중복 ID가 있습니다: {sample_id} "
                        f"({source_by_id[sample_id]}, {path})"
                    )
                source_by_id[sample_id] = path
                rows.append({key: (value or "").strip() for key, value in row.items()})

    pending = [row["id"] for row in rows if row.get("review_status") not in {"approved", "needs_review", "rejected"}]
    if pending:
        raise SystemExit(
            f"미완료 review_status가 {len(pending)}건 있습니다. 전체 검수 후 내보내세요: {pending[:10]}"
        )

    approved: list[dict[str, str]] = []
    holdout: list[dict[str, str]] = []
    for row in rows:
        accepted = (
            row.get("review_status") == "approved"
            and row.get("readability") == "clear"
            and row.get("confidence") in {"high", "medium"}
            and row.get("human_answer") in ANSWER_KEYS
            and row.get("data_usage") == "vqa_train"
        )
        if accepted:
            approved.append(row)
        else:
            holdout.append(row)

    output.mkdir(parents=True, exist_ok=True)
    approved_path = output / "approved_dev.csv"
    with approved_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRAIN_FIELDS)
        writer.writeheader()
        for row in approved:
            writer.writerow({field: row[field] for field in TRAIN_FIELDS})

    holdout_path = output / "quality_holdout.csv"
    holdout_fields = list(rows[0]) if rows else []
    with holdout_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=holdout_fields)
        writer.writeheader()
        writer.writerows(holdout)

    manifest = {
        "input_rows": len(rows),
        "approved_vqa_rows": len(approved),
        "quality_holdout_rows": len(holdout),
        "holdout_readability": dict(sorted(Counter(row.get("readability", "") for row in holdout).items())),
        "holdout_quality_reason": dict(sorted(Counter(row.get("quality_reason", "") for row in holdout).items())),
        "gate": {
            "review_status": "approved",
            "readability": "clear",
            "confidence": ["high", "medium"],
            "human_answer": sorted(ANSWER_KEYS),
            "data_usage": "vqa_train",
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
