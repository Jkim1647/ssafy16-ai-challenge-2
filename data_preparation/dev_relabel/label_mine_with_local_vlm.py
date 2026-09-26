"""Create resumable local-VLM drafts for the assigned dev relabeling batch.

This tool never reads the original dev answer1..answer5 columns and never marks a
row approved.  Its output is only a review queue for direct image inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


MODEL_COLUMNS = [
    "model_candidate_answer",
    "model_confidence",
    "model_confidence_score",
    "model_choice_probabilities",
    "model_readability",
    "model_quality_reason",
    "model_evidence",
    "model_ambiguity",
    "model_raw_output",
    "model_status",
]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_root = here / "generated" / "dev_split_markers_v1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=default_root / "mine" / "labels.csv")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=default_root / "mine" / "local_vlm_drafts")
    parser.add_argument("--limit", type=int, default=0, help="0 means all remaining rows")
    parser.add_argument("--max-pixels", type=int, default=1_048_576)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def prompt_for(row: dict[str, str]) -> str:
    choices = "\n".join(f"{letter}. {row[letter]}" for letter in "abcd")
    return f"""이미지를 보고 객관식 질문에 답하시오.
이미지에 실제로 보이는 정보만 사용하시오.
질문: {row['question']}
{choices}
정답 기호 하나만 출력하시오."""


def confidence_band(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.50:
        return "medium"
    return "low"


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[str(record["id"])] = record
    return records


def materialize(rows: list[dict[str, str]], drafts: dict[str, dict[str, Any]], output: Path) -> None:
    fieldnames = list(rows[0]) + MODEL_COLUMNS
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for source in rows:
            row = dict(source)
            draft = drafts.get(source["id"], {})
            for column in MODEL_COLUMNS:
                row[column] = str(draft.get(column, ""))
            if draft:
                row["llm_assistance"] = "Qwen2.5-VL-3B-Instruct-local"
                row["review_status"] = "needs_review"
                row["review_note"] = "local VLM draft only; direct image review required"
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    journal_path = args.output_dir / "drafts.jsonl"
    csv_path = args.output_dir / "mine_model_draft.csv"

    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("input CSV is empty")
    forbidden = {f"answer{i}" for i in range(1, 6)} & set(rows[0])
    if forbidden:
        raise SystemExit(f"forbidden response columns found: {sorted(forbidden)}")
    if {row["assignment"] for row in rows} != {"mine"}:
        raise SystemExit("input contains rows outside assignment=mine")

    drafts = load_existing(journal_path)
    pending = [row for row in rows if row["id"] not in drafts]
    if args.limit > 0:
        pending = pending[: args.limit]

    processor = AutoProcessor.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        min_pixels=3_136,
        max_pixels=args.max_pixels,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    ).eval()
    tokenizer = processor.tokenizer
    letter_ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in "abcd"]
    if any(len(ids) != 1 for ids in letter_ids):
        raise SystemExit(f"choice letters are not single tokens: {letter_ids}")
    letter_tensor = torch.tensor([ids[0] for ids in letter_ids], device=model.device)

    with journal_path.open("a", encoding="utf-8", buffering=1) as journal:
        for index, row in enumerate(pending, start=1):
            started = time.perf_counter()
            image_path = args.image_root / Path(row["path"])
            try:
                with Image.open(image_path) as opened:
                    image = opened.convert("RGB")
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_for(row)},
                    ],
                }]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
                inputs = inputs.to(model.device)
                with torch.inference_mode():
                    logits = model(**inputs).logits[0, -1, :].float()
                    selected = logits.index_select(0, letter_tensor)
                    probabilities = torch.softmax(selected, dim=0).cpu().tolist()
                best = max(range(4), key=lambda choice: probabilities[choice])
                score = float(probabilities[best])
                probability_map = {
                    letter: round(float(probability), 6)
                    for letter, probability in zip("abcd", probabilities)
                }
                draft = {
                    "model_candidate_answer": "abcd"[best],
                    "model_confidence": confidence_band(score),
                    "model_confidence_score": f"{score:.6f}",
                    "model_choice_probabilities": json.dumps(probability_map, ensure_ascii=False),
                    # Model confidence is not an image-quality approval.
                    "model_readability": "",
                    "model_quality_reason": "",
                    "model_evidence": "",
                    "model_ambiguity": "" if score >= 0.75 else "choice probability requires review",
                    "model_raw_output": "",
                    "model_status": "draft",
                }
            except Exception as exc:  # Keep the journal resumable after a bad image/runtime row.
                draft = {
                    "model_candidate_answer": "",
                    "model_confidence": "low",
                    "model_confidence_score": "0.000000",
                    "model_choice_probabilities": "",
                    "model_readability": "unreadable",
                    "model_quality_reason": "other",
                    "model_evidence": "",
                    "model_ambiguity": f"runtime_error: {type(exc).__name__}: {exc}",
                    "model_raw_output": "",
                    "model_status": "runtime_error",
                }
            record = {
                "id": row["id"],
                **draft,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
            journal.write(json.dumps(record, ensure_ascii=False) + "\n")
            drafts[row["id"]] = record
            if index == 1 or index == len(pending) or index % max(args.log_every, 1) == 0:
                print(json.dumps({"progress": f"{index}/{len(pending)}", **record}, ensure_ascii=False))

    materialize(rows, drafts, csv_path)
    summary = {
        "source_rows": len(rows),
        "draft_rows": len(drafts),
        "remaining_rows": len(rows) - len(drafts),
        "output": str(csv_path),
        "approval_policy": "all drafts remain needs_review until direct image review",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
