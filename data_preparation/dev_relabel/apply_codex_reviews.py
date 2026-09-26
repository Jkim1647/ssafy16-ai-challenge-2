"""Apply audited Codex visual-review decisions without mutating the draft CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


EDITABLE = {
    "human_answer",
    "confidence",
    "readability",
    "quality_reason",
    "data_usage",
    "ambiguity_reason",
    "evidence",
    "review_status",
    "review_note",
}
ANSWERS = {"a", "b", "c", "d"}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    root = here / "generated" / "dev_split_markers_v1" / "mine" / "local_vlm_drafts"
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=root / "mine_model_draft.csv")
    parser.add_argument("--decisions", type=Path, default=root / "codex_review" / "decisions.jsonl")
    parser.add_argument("--output", type=Path, default=root / "codex_review" / "mine_codex_reviewed.csv")
    return parser.parse_args()


def validate(decision: dict[str, str]) -> None:
    unknown = set(decision) - EDITABLE - {"id"}
    if unknown:
        raise SystemExit(f"decision {decision.get('id')} has unsupported fields: {sorted(unknown)}")
    status = decision.get("review_status")
    if status not in {"approved", "needs_review", "rejected"}:
        raise SystemExit(f"decision {decision.get('id')} has invalid review_status: {status}")
    if status == "approved":
        required = {
            "human_answer": ANSWERS,
            "confidence": {"high", "medium"},
            "readability": {"clear"},
            "quality_reason": {"none"},
            "data_usage": {"vqa_train"},
        }
        for field, allowed in required.items():
            if decision.get(field) not in allowed:
                raise SystemExit(
                    f"approved decision {decision.get('id')} has invalid {field}: {decision.get(field)!r}"
                )
        if not decision.get("evidence", "").strip():
            raise SystemExit(f"approved decision {decision.get('id')} has no evidence")
    elif decision.get("human_answer", "").strip():
        raise SystemExit(f"non-approved decision {decision.get('id')} must not contain human_answer")


def main() -> None:
    args = parse_args()
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    forbidden = {f"answer{i}" for i in range(1, 6)} & set(fieldnames)
    if forbidden:
        raise SystemExit(f"forbidden response columns found: {sorted(forbidden)}")

    decisions: dict[str, dict[str, str]] = {}
    with args.decisions.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            decision = json.loads(line)
            validate(decision)
            sample_id = decision["id"]
            if sample_id in decisions:
                raise SystemExit(f"duplicate decision id: {sample_id}")
            decisions[sample_id] = decision

    known = {row["id"] for row in rows}
    if missing := set(decisions) - known:
        raise SystemExit(f"decision ids not present in source: {sorted(missing)[:10]}")

    for row in rows:
        decision = decisions.get(row["id"])
        if decision is None:
            continue
        for field in EDITABLE:
            if field in decision:
                row[field] = decision[field]
        row["reviewer"] = "codex-assisted-visual"
        row["llm_assistance"] = "Codex direct image review; local Qwen2.5 draft retained separately"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    reviewed = [row for row in rows if row["id"] in decisions]
    summary = {
        "source_rows": len(rows),
        "decision_rows": len(decisions),
        "status_counts": dict(Counter(row["review_status"] for row in reviewed)),
        "approved_answers": dict(
            Counter(row["human_answer"] for row in reviewed if row["review_status"] == "approved")
        ),
        "remaining_needs_review": sum(
            row["review_status"] == "needs_review" and row["id"] not in decisions for row in rows
        ),
        "output": str(args.output),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
