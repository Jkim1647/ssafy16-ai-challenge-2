"""Create a non-destructive corrected human-review set for reported dev items.

The output remains a review queue: every reported ID is withheld from training
until a human confirms the image, question, four choices, and answer.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPORT = HERE / "human_reported_errors_2026-09-22.json"
SOURCE = HERE / "generated/dev_split_markers_v1/mine/regenerated_questions/human_review_5way/full_1317.csv"
OUTPUT = HERE / "generated/dev_split_markers_v1/mine/regenerated_questions/human_review_5way_corrected_20260922"

# Change only image-grounded errors. The remaining reports retain their current
# question/choices but are marked for re-review because the reporter's CSV may
# differ from this local version.
CORRECTIONS = {
    "dev_0621.jpg": {"d": "명가수산"},
    "dev_0642.jpg": {"d": "송영서치과"},
    "dev_1027.jpg": {
        "question": "냉장 진열대 맨 위 가운데 '김밥愛 칼리스타' 포장 바로 아래 가격표는 얼마인가요?",
        "a": "6,280원", "b": "4,680원", "c": "7,980원", "d": "8,982원", "answer": "c",
    },
    "dev_1712.jpg": {"b": "15,000원"},
}


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    reports = json.loads(REPORT.read_text(encoding="utf-8"))
    report_by_id = {item["source_id"]: item for item in reports}
    if len(reports) != 16 or len(report_by_id) != 16:
        raise RuntimeError("Expected 16 unique human reports")
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if len(rows) != 1317 or len({row["source_id"] for row in rows}) != len(rows):
        raise RuntimeError("Unexpected source row count or duplicate source_id")
    if [int(row["photo_no"]) for row in rows] != sorted(int(row["photo_no"]) for row in rows):
        raise RuntimeError("Source is not in photo-number order")
    source_by_id = {row["source_id"]: row for row in rows}
    if set(report_by_id) - set(source_by_id):
        raise RuntimeError(f"Missing reported IDs: {set(report_by_id) - set(source_by_id)}")
    for item_id in CORRECTIONS:
        if source_by_id[item_id]["review_status"] != "approved":
            raise RuntimeError(f"Correction assumption changed for {item_id}")

    changes = []
    for row in rows:
        item_id = row["source_id"]
        report = report_by_id.get(item_id)
        if report is None:
            continue
        before = dict(row)
        row.update(CORRECTIONS.get(item_id, {}))
        row["review_status"] = "rejected" if item_id == "dev_1670.jpg" else "needs_review"
        row["human_review_status"] = "excluded" if item_id == "dev_1670.jpg" else "needs_review"
        row["data_usage"] = "exclude" if item_id == "dev_1670.jpg" else "review_only"
        row["semantic_review_issue"] = report["classification"]
        note = f"2026-09-22 사람 신고 재점검: {report['finding']} 다음: {report['action']}"
        row["human_review_note"] = " | ".join(x for x in (before["human_review_note"], note) if x)
        row["review_note"] = " | ".join(x for x in (before["review_note"], "사람 신고에 따른 정정 초안; 최종 승인 전 학습 제외") if x)
        choices = [row[key].strip() for key in "abcd"]
        if item_id != "dev_1670.jpg" and (
            not row["question"].strip() or not row["answer"] in "abcd"
            or len(set(choices)) != 4 or any(not choice for choice in choices)
        ):
            raise RuntimeError(f"Invalid corrected QA: {item_id}")
        changes.append({"source_id": item_id,
                        "changed_fields": [key for key in fields if before[key] != row[key]],
                        "classification": report["classification"]})

    OUTPUT.mkdir(parents=True, exist_ok=True)
    base, remainder = divmod(len(rows), 5)
    sizes = [base + (i < remainder) for i in range(5)]
    parts = []
    offset = 0
    for index, size in enumerate(sizes, 1):
        subset = rows[offset:offset + size]
        parts.append((OUTPUT / f"reviewer_{index}_{int(subset[0]['photo_no']):04d}-{int(subset[-1]['photo_no']):04d}.csv", subset))
        offset += size
    targets = [OUTPUT / "full_1317.csv", *(p for p, _ in parts), OUTPUT / "correction_log.json"]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise RuntimeError("Refusing to overwrite existing review results: " + ", ".join(existing))
    write_csv(targets[0], fields, rows)
    for path, subset in parts:
        write_csv(path, fields, subset)
    targets[-1].write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "reported": len(changes),
                      "corrected_qa": list(CORRECTIONS), "excluded": ["dev_1670.jpg"],
                      "part_sizes": sizes, "output": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
