"""Render Codex-only image sheets for direct replacement-question generation.

The sheets expose only the image, source id, broad source type, and a balanced
target answer position.  Original questions, choices, dev responses, and local
model predictions are intentionally omitted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    mine = here / "generated" / "dev_split_markers_v1" / "mine"
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=mine / "labels.csv")
    parser.add_argument("--completed", type=Path, default=mine / "regenerated_questions" / "pilot_20.csv")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=mine / "regenerated_questions" / "codex_sheets")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means every remaining row")
    parser.add_argument("--per-sheet", type=int, choices=(4, 6, 12), default=12)
    return parser.parse_args()


def font(size: int):
    for path in (
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("/fonts/malgun.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ):
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def balanced_positions(rows: list[dict[str, str]], seed: int) -> dict[str, str]:
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest(),
    )
    return {row["id"]: "abcd"[index % 4] for index, row in enumerate(ordered)}


def layout(per_sheet: int) -> tuple[int, int, tuple[int, int]]:
    if per_sheet == 4:
        return 2, 2, (960, 940)
    if per_sheet == 6:
        return 3, 2, (720, 900)
    return 3, 4, (680, 510)


def render_card(
    row: dict[str, str], image_root: Path, target: str, size: tuple[int, int]
) -> Image.Image:
    width, height = size
    card = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(card)
    title_font = font(22)
    meta_font = font(18)
    draw.rectangle((0, 0, width - 1, height - 1), outline="#666666", width=2)
    draw.text((12, 8), f"{row['id']}  |  target={target.upper()}", fill="black", font=title_font)
    draw.text((12, 38), f"type: {row.get('auto_type', 'UNKNOWN')}", fill="#333333", font=meta_font)

    image_path = image_root / row["path"]
    with Image.open(image_path) as opened:
        picture = ImageOps.exif_transpose(opened).convert("RGB")
    picture.thumbnail((width - 24, height - 76), Image.Resampling.LANCZOS)
    x = (width - picture.width) // 2
    y = 68 + (height - 76 - picture.height) // 2
    card.paste(picture, (x, y))
    return card


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    if not rows:
        raise SystemExit("input CSV is empty")
    forbidden = {f"answer{i}" for i in range(1, 6)} & set(rows[0])
    if forbidden:
        raise SystemExit(f"forbidden response columns found: {sorted(forbidden)}")
    completed_ids = {row["source_id"] for row in read_rows(args.completed)}
    pending = [row for row in rows if row["id"] not in completed_ids]
    positions = balanced_positions(pending, args.seed)
    selected = pending[args.start : args.start + args.limit if args.limit else None]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns, rows_per_sheet, card_size = layout(args.per_sheet)
    manifest: list[dict[str, object]] = []
    for offset in range(0, len(selected), args.per_sheet):
        group = selected[offset : offset + args.per_sheet]
        canvas = Image.new(
            "RGB",
            (columns * card_size[0], rows_per_sheet * card_size[1]),
            "#dddddd",
        )
        for index, row in enumerate(group):
            card = render_card(row, args.image_root, positions[row["id"]], card_size)
            canvas.paste(
                card,
                ((index % columns) * card_size[0], (index // columns) * card_size[1]),
            )
        sheet_number = args.start // args.per_sheet + offset // args.per_sheet + 1
        sheet_name = f"sheet_{sheet_number:04d}.jpg"
        canvas.save(args.output_dir / sheet_name, quality=92, subsampling=0)
        manifest.append(
            {
                "sheet": sheet_name,
                "rows": [
                    {
                        "source_id": row["id"],
                        "path": row["path"],
                        "source_auto_type": row.get("auto_type", ""),
                        "target_answer": positions[row["id"]],
                    }
                    for row in group
                ],
            }
        )

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "assignment.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("source_id", "path", "source_auto_type", "target_answer"),
        )
        writer.writeheader()
        for row in pending:
            writer.writerow(
                {
                    "source_id": row["id"],
                    "path": row["path"],
                    "source_auto_type": row.get("auto_type", ""),
                    "target_answer": positions[row["id"]],
                }
            )
    print(
        json.dumps(
            {
                "source_rows": len(rows),
                "already_completed": len(completed_ids),
                "pending": len(pending),
                "rendered": len(selected),
                "sheets": len(manifest),
                "output": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
