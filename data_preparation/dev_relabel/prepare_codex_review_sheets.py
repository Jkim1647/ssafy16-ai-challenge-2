"""Render blind image/question sheets for direct Codex-assisted dev review.

The sheet intentionally omits model candidates and original dev responses so the
visual decision can be made before comparing local-model drafts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    drafts = here / "generated" / "dev_split_markers_v1" / "mine" / "local_vlm_drafts"
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=drafts / "review_queue" / "review_queue_low.csv")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=drafts / "codex_review" / "sheets_low")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--per-sheet", type=int, choices=(1, 2, 4), default=4)
    return parser.parse_args()


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("/fonts/malgun.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def wrap(draw: ImageDraw.ImageDraw, text: str, selected_font, width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for char in text:
        candidate = current + char
        if current and draw.textbbox((0, 0), candidate, font=selected_font)[2] > width:
            lines.append(current)
            current = char
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def render_card(row: dict[str, str], image_root: Path, size: tuple[int, int]) -> Image.Image:
    card_w, card_h = size
    card = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(card)
    title_font = font(25)
    body_font = font(21)
    image_path = image_root / Path(row["path"])
    with Image.open(image_path) as source:
        picture = ImageOps.exif_transpose(source).convert("RGB")
    picture.thumbnail((card_w - 32, 600), Image.Resampling.LANCZOS)
    x = (card_w - picture.width) // 2
    y = 48 + (600 - picture.height) // 2
    card.paste(picture, (x, y))
    draw.rectangle((0, 0, card_w - 1, card_h - 1), outline="#777777", width=2)
    rank = row.get("review_rank") or row.get("multimodel_review_rank") or "?"
    draw.text((16, 10), f"rank {rank} | {row['id']}", fill="black", font=title_font)

    cursor = 660
    text_items = [f"Q. {row['question']}"] + [f"{letter.upper()}. {row[letter]}" for letter in "abcd"]
    for item in text_items:
        for line in wrap(draw, item, body_font, card_w - 32):
            draw.text((16, cursor), line, fill="black", font=body_font)
            cursor += 30
        cursor += 3
    return card


def main() -> None:
    args = parse_args()
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    forbidden = {f"answer{i}" for i in range(1, 6)} & set(rows[0])
    if forbidden:
        raise SystemExit(f"forbidden response columns found: {sorted(forbidden)}")
    selected = rows[args.start : args.start + args.limit if args.limit else None]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.per_sheet == 1:
        columns, rows_per_sheet = 1, 1
    elif args.per_sheet == 2:
        columns, rows_per_sheet = 1, 2
    else:
        columns, rows_per_sheet = 2, 2
    card_size = (1000, 980)
    manifest: list[dict[str, object]] = []
    for offset in range(0, len(selected), args.per_sheet):
        group = selected[offset : offset + args.per_sheet]
        canvas = Image.new(
            "RGB",
            (columns * card_size[0], rows_per_sheet * card_size[1]),
            "#dddddd",
        )
        for index, row in enumerate(group):
            card = render_card(row, args.image_root, card_size)
            canvas.paste(card, ((index % columns) * card_size[0], (index // columns) * card_size[1]))
        sheet_number = args.start // args.per_sheet + offset // args.per_sheet + 1
        sheet_name = f"sheet_{sheet_number:04d}.jpg"
        canvas.save(args.output_dir / sheet_name, quality=92, subsampling=0)
        manifest.append({"sheet": sheet_name, "ids": [row["id"] for row in group]})

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"rows": len(selected), "sheets": len(manifest), "output": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
