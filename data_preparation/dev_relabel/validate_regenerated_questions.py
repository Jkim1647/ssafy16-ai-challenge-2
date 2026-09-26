"""Validate train-ready dev question-regeneration CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


REQUIRED = (
    "id",
    "image_id",
    "path",
    "source_id",
    "question",
    "a",
    "b",
    "c",
    "d",
    "answer",
    "confidence",
    "readability",
    "data_usage",
    "evidence",
    "reviewer",
    "review_status",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--require-balanced-answers", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in REQUIRED if name not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"missing columns: {missing}")
        rows = list(reader)

    errors: list[str] = []
    ids: set[str] = set()
    answer_counts: Counter[str] = Counter()
    for line_number, row in enumerate(rows, start=2):
        prefix = f"line {line_number} ({row['id'] or 'missing id'})"
        if row["id"] in ids:
            errors.append(f"{prefix}: duplicate id")
        ids.add(row["id"])
        status = row["review_status"]
        choices = [row[letter].strip() for letter in "abcd"]
        if status == "approved":
            if row["answer"] not in "abcd":
                errors.append(f"{prefix}: approved answer must be a-d")
            else:
                answer_counts[row["answer"]] += 1
            if any(not value for value in choices):
                errors.append(f"{prefix}: approved row has blank choice")
            if len(set(choices)) != 4:
                errors.append(f"{prefix}: approved row has duplicate choices")
            if not row["question"].strip() or not row["evidence"].strip():
                errors.append(f"{prefix}: approved row has blank question or evidence")
            if row["readability"] != "clear":
                errors.append(f"{prefix}: approved row must be clear")
            if row["confidence"] not in {"high", "medium"}:
                errors.append(f"{prefix}: approved row confidence must be high/medium")
            if row["data_usage"] != "vqa_train":
                errors.append(f"{prefix}: approved row must use vqa_train")
        elif status == "rejected":
            if row["answer"].strip() or any(choices) or row["question"].strip():
                errors.append(f"{prefix}: rejected row must not contain trainable labels")
            if row["data_usage"] != "exclude":
                errors.append(f"{prefix}: rejected row must use exclude")
            if not row["evidence"].strip():
                errors.append(f"{prefix}: rejected row needs exclusion evidence")
        else:
            errors.append(f"{prefix}: review_status must be approved/rejected")
        if Path(row["path"]).name != row["source_id"]:
            errors.append(f"{prefix}: source_id must match path filename")
        if args.image_root and not (args.image_root / row["path"]).is_file():
            errors.append(f"{prefix}: image not found at {args.image_root / row['path']}")

    if args.require_balanced_answers and answer_counts:
        counts = [answer_counts[letter] for letter in "abcd"]
        if max(counts) - min(counts) > 1:
            errors.append(f"answer positions are not balanced: {dict(answer_counts)}")

    summary = {
        "rows": len(rows),
        "unique_ids": len(ids),
        "unique_images": len({row["image_id"] for row in rows}),
        "answer_counts": {letter: answer_counts[letter] for letter in "abcd"},
        "approved": sum(row["review_status"] == "approved" for row in rows),
        "errors": errors,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
