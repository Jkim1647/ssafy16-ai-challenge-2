"""Merge Codex-authored regeneration decisions without touching source CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


FIELDS = (
    "id",
    "image_id",
    "path",
    "source_id",
    "source_auto_type",
    "question_type",
    "question",
    "a",
    "b",
    "c",
    "d",
    "answer",
    "confidence",
    "readability",
    "quality_reason",
    "data_usage",
    "evidence",
    "reviewer",
    "llm_assistance",
    "review_status",
    "review_note",
)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    root = here / "generated" / "dev_split_markers_v1" / "mine" / "regenerated_questions"
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=root / "pilot_20.csv")
    parser.add_argument("--decisions", type=Path, default=root / "codex_decisions.jsonl")
    parser.add_argument("--assignment", type=Path, default=root / "codex_sheets" / "assignment.csv")
    parser.add_argument("--output", type=Path, default=root / "codex_regenerated_all.csv")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    base = read_csv(args.base)
    assignment = {row["source_id"]: row for row in read_csv(args.assignment)}
    decisions: list[dict[str, str]] = []
    if args.decisions.is_file():
        with args.decisions.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise SystemExit(f"decision line {line_number} is not an object")
                    decisions.append({key: str(item) for key, item in value.items()})

    seen = {row["source_id"] for row in base}
    output_rows: list[dict[str, str]] = []
    for row in base:
        item = {field: row.get(field, "") for field in FIELDS}
        item["id"] = f"dev_regen_{Path(row['source_id']).stem.removeprefix('dev_')}"
        output_rows.append(item)

    for index, decision in enumerate(decisions, start=1):
        source_id = decision.get("source_id", "")
        if source_id in seen:
            raise SystemExit(f"duplicate decision source_id at line {index}: {source_id}")
        if source_id not in assignment:
            raise SystemExit(f"unknown source_id at line {index}: {source_id}")
        seen.add(source_id)
        assigned = assignment[source_id]
        status = decision.get("review_status", "approved")
        answer = decision.get("answer", "").lower()
        if status == "approved":
            if answer not in "abcd":
                raise SystemExit(f"approved {source_id} has invalid answer {answer!r}")
            if answer != assigned["target_answer"]:
                raise SystemExit(
                    f"{source_id} answer {answer} does not match assigned balanced position "
                    f"{assigned['target_answer']}"
                )
            choices = [decision.get(letter, "").strip() for letter in "abcd"]
            if any(not choice for choice in choices) or len(set(choices)) != 4:
                raise SystemExit(f"approved {source_id} has blank or duplicate choices")
            if not decision.get("question", "").strip() or not decision.get("evidence", "").strip():
                raise SystemExit(f"approved {source_id} lacks question/evidence")
        item = {
            "id": f"dev_regen_{Path(source_id).stem.removeprefix('dev_')}",
            "image_id": Path(source_id).stem,
            "path": assigned["path"],
            "source_id": source_id,
            "source_auto_type": assigned["source_auto_type"],
            "question_type": decision.get("question_type", ""),
            "question": decision.get("question", ""),
            **{letter: decision.get(letter, "") for letter in "abcd"},
            "answer": answer,
            "confidence": decision.get("confidence", "high" if status == "approved" else "low"),
            "readability": decision.get("readability", "clear" if status == "approved" else "unreadable"),
            "quality_reason": decision.get("quality_reason", "none" if status == "approved" else "other"),
            "data_usage": decision.get("data_usage", "vqa_train" if status == "approved" else "exclude"),
            "evidence": decision.get("evidence", ""),
            "reviewer": "codex-assisted-visual",
            "llm_assistance": "Codex direct image review and question regeneration",
            "review_status": status,
            "review_note": decision.get("review_note", "direct image question regeneration"),
        }
        output_rows.append(item)

    output_rows.sort(key=lambda row: int(Path(row["source_id"]).stem.removeprefix("dev_")))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "rows": len(output_rows),
        "approved": sum(row["review_status"] == "approved" for row in output_rows),
        "rejected": sum(row["review_status"] == "rejected" for row in output_rows),
        "answer_counts": dict(
            Counter(row["answer"] for row in output_rows if row["review_status"] == "approved")
        ),
        "remaining": len(assignment) - len(decisions),
        "output": str(args.output),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
