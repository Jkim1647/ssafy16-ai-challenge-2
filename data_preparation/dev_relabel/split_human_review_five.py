"""Split the sorted human-review CSV into five disjoint reviewer files."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "generated/dev_split_markers_v1/mine/regenerated_questions/human_review_sorted.csv"
OUTPUT_DIR = HERE / "generated/dev_split_markers_v1/mine/regenerated_questions/human_review_5way"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields or "source_id" not in fields or "photo_no" not in fields:
        parser.error("source_id와 photo_no 열이 필요합니다")
    if len(rows) < 5:
        parser.error("5명에게 나누려면 최소 5행이 필요합니다")
    numbers = [int(row["photo_no"]) for row in rows]
    if numbers != sorted(numbers) or len(numbers) != len(set(numbers)):
        parser.error("입력 사진 번호가 정렬되지 않았거나 중복되었습니다")
    if len({row["source_id"] for row in rows}) != len(rows):
        parser.error("source_id가 중복되었습니다")

    base, remainder = divmod(len(rows), 5)
    sizes = [base + (index < remainder) for index in range(5)]
    parts: list[tuple[Path, list[dict[str, str]]]] = []
    offset = 0
    for index, size in enumerate(sizes, start=1):
        slice_rows = rows[offset : offset + size]
        first = int(slice_rows[0]["photo_no"])
        last = int(slice_rows[-1]["photo_no"])
        name = f"reviewer_{index}_{first:04d}-{last:04d}.csv"
        parts.append((args.output_dir / name, slice_rows))
        offset += size
    if offset != len(rows):
        raise AssertionError("row allocation mismatch")

    full_path = args.output_dir / f"full_{len(rows)}.csv"
    targets = [full_path, *(path for path, _ in parts)]
    existing = [path for path in targets if path.exists()]
    if existing:
        parser.error("기존 검수 결과를 덮어쓰지 않습니다: " + ", ".join(str(path) for path in existing))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.input, full_path)
    print(f"full: {full_path.name}: {len(rows)} rows")
    for path, slice_rows in parts:
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(slice_rows)
        issues = sum(bool(row.get("semantic_review_issue")) for row in slice_rows)
        print(f"reviewer: {path.name}: {len(slice_rows)} rows, {issues} flagged")


if __name__ == "__main__":
    main()
