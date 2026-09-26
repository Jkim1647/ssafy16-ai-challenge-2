"""Create two leakage-safe dev relabeling packets plus a shared calibration set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


SEED = 20260921
SAFE_SOURCE_COLUMNS = ("id", "path", "question", "a", "b", "c", "d")
LABEL_COLUMNS = (
    "human_answer",
    "confidence",
    "readability",
    "quality_reason",
    "data_usage",
    "ambiguity_reason",
    "evidence",
    "reviewer",
    "llm_assistance",
    "review_status",
    "review_note",
)
RULES: dict[str, tuple[str, ...]] = {
    "PRICE": (r"가격", r"얼마", r"요금", r"금액", r"원가", r"비용"),
    "TIMEDATE": (r"영업\s*시간", r"몇\s*시", r"날짜", r"기간", r"요일", r"언제", r"영업일"),
    "SPELLING": (r"정확한\s*표기", r"철자", r"이름.{0,3}(뭐|무엇)", r"뭐라고\s*(적혀|쓰여|쓰인)", r"글자"),
    "NUMBER": (r"몇\s*개", r"번호", r"전화번호", r"수량", r"개수", r"몇\s*명", r"몇\s*번"),
    "TABLE": (r"표에서", r"목록에서", r"표\s*안", r"항목"),
    "LOGO": (r"로고", r"브랜드", r"상표"),
    "OBJECT": (r"무엇", r"물건", r"사물", r"있(나요|습니까|어)"),
    "RELATION": (r"왼쪽", r"오른쪽", r"위에", r"아래", r"옆에", r"순서", r"비교", r"보다"),
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    workspace = here.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, default=workspace / "ssafy-16-2-ai" / "dev.csv")
    parser.add_argument("--source-root", type=Path, default=workspace / "ssafy-16-2-ai")
    parser.add_argument("--output", type=Path, default=here / "generated" / "dev_split_markers_v1")
    parser.add_argument("--calibration-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-zip", action="store_true")
    parser.add_argument(
        "--include-images",
        action="store_true",
        help="Copy images into each packet. Default is marker/metadata CSV only.",
    )
    return parser.parse_args()


def stable_digest(*parts: object) -> str:
    value = "\0".join(str(part) for part in parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def classify(question: str) -> str:
    matched = [
        type_name
        for type_name, patterns in RULES.items()
        if any(re.search(pattern, question) for pattern in patterns)
    ]
    if not matched:
        return "UNKNOWN"
    if len(matched) == 1:
        return matched[0]
    return "COMPOSITE"


def read_dev(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])
        missing = [column for column in SAFE_SOURCE_COLUMNS if column not in headers]
        if missing:
            raise SystemExit(f"dev CSV 필수 열이 없습니다: {missing}")
        rows = []
        for source in reader:
            row = {column: source[column].strip() for column in SAFE_SOURCE_COLUMNS}
            row["auto_type"] = classify(row["question"])
            rows.append(row)
    if len({row["id"] for row in rows}) != len(rows):
        raise SystemExit("dev CSV id가 고유하지 않습니다.")
    return rows, headers


def choose_calibration(rows: list[dict[str, str]], size: int, seed: int) -> list[dict[str, str]]:
    by_type: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_type[row["auto_type"]].append(row)
    for type_name, bucket in by_type.items():
        bucket.sort(key=lambda row: stable_digest(seed, "calibration", type_name, row["id"]))

    chosen: list[dict[str, str]] = []
    type_names = sorted(by_type)
    while len(chosen) < size:
        progress = False
        for type_name in type_names:
            if by_type[type_name] and len(chosen) < size:
                chosen.append(by_type[type_name].pop(0))
                progress = True
        if not progress:
            break
    return chosen


def split_remaining(
    rows: list[dict[str, str]], calibration_ids: set[str], seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    by_type: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["id"] not in calibration_ids:
            by_type[row["auto_type"]].append(row)

    mine: list[dict[str, str]] = []
    teammate: list[dict[str, str]] = []
    for type_name in sorted(by_type):
        bucket = sorted(
            by_type[type_name],
            key=lambda row: stable_digest(seed, "assignment", type_name, row["id"]),
        )
        for row in bucket:
            if len(mine) <= len(teammate):
                mine.append(row)
            else:
                teammate.append(row)
    return mine, teammate


def ensure_empty_output(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"출력 디렉터리가 비어 있지 않습니다: {output}")
    output.mkdir(parents=True, exist_ok=True)


def output_row(
    row: dict[str, str], assignment: str, calibration: bool, include_images: bool
) -> dict[str, str]:
    result = {column: row[column] for column in SAFE_SOURCE_COLUMNS}
    if include_images:
        result["path"] = f"images/{Path(row['path'].replace('\\', '/')).name}"
    result.update(
        {
            "auto_type": row["auto_type"],
            "assignment": assignment,
            "is_calibration": "true" if calibration else "false",
        }
    )
    result.update({column: "" for column in LABEL_COLUMNS})
    return result


def write_packet(
    *,
    packet_dir: Path,
    rows: list[dict[str, str]],
    assignment: str,
    calibration: bool,
    source_root: Path,
    include_images: bool,
) -> None:
    packet_dir.mkdir(parents=True, exist_ok=True)
    image_dir = packet_dir / "images" if include_images else None
    if image_dir:
        image_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [*SAFE_SOURCE_COLUMNS, "auto_type", "assignment", "is_calibration", *LABEL_COLUMNS]
    with (packet_dir / "labels.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if include_images:
                source_image = source_root.joinpath(*row["path"].replace("\\", "/").split("/"))
                if not source_image.is_file():
                    raise SystemExit(f"dev 이미지가 없습니다: {source_image}")
                assert image_dir is not None
                shutil.copy2(source_image, image_dir / source_image.name)
            writer.writerow(output_row(row, assignment, calibration, include_images))


def csv_headers(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle).fieldnames or [])


def make_teammate_zip(output: Path, guide_path: Path, handoff_path: Path) -> Path:
    zip_path = output / "teammate_handoff.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(guide_path, "LABELING_GUIDE_DRAFT.md")
        archive.write(handoff_path, "TEAMMATE_HANDOFF.md")
        archive.write(output / "manifest.json", "manifest.json")
        for packet_name in ("calibration", "teammate"):
            packet_dir = output / packet_name
            for path in sorted(packet_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(output).as_posix())
    return zip_path


def distribution(rows: list[dict[str, str]]) -> dict[str, int]:
    return dict(sorted(Counter(row["auto_type"] for row in rows).items()))


def main() -> int:
    args = parse_args()
    source_csv = args.source_csv.resolve()
    source_root = args.source_root.resolve()
    output = args.output.resolve()
    if args.calibration_size <= 0:
        raise SystemExit("--calibration-size는 양수여야 합니다.")
    if not source_csv.is_file() or not source_root.is_dir():
        raise SystemExit("dev CSV 또는 데이터 루트를 찾을 수 없습니다.")
    ensure_empty_output(output)

    rows, original_headers = read_dev(source_csv)
    if args.calibration_size >= len(rows):
        raise SystemExit("calibration 크기가 dev 전체 행 수보다 작아야 합니다.")
    calibration = choose_calibration(rows, args.calibration_size, args.seed)
    calibration_ids = {row["id"] for row in calibration}
    mine, teammate = split_remaining(rows, calibration_ids, args.seed)

    write_packet(
        packet_dir=output / "calibration",
        rows=calibration,
        assignment="shared_calibration",
        calibration=True,
        source_root=source_root,
        include_images=args.include_images,
    )
    write_packet(
        packet_dir=output / "mine",
        rows=mine,
        assignment="mine",
        calibration=False,
        source_root=source_root,
        include_images=args.include_images,
    )
    write_packet(
        packet_dir=output / "teammate",
        rows=teammate,
        assignment="teammate",
        calibration=False,
        source_root=source_root,
        include_images=args.include_images,
    )

    all_ids = calibration_ids | {row["id"] for row in mine} | {row["id"] for row in teammate}
    overlap = {row["id"] for row in mine} & {row["id"] for row in teammate}
    if len(all_ids) != len(rows) or overlap:
        raise AssertionError("dev 분할에서 누락 또는 mine/teammate 중복이 발생했습니다.")

    output_csvs = [
        output / "calibration" / "labels.csv",
        output / "mine" / "labels.csv",
        output / "teammate" / "labels.csv",
    ]
    forbidden_headers = {f"answer{index}" for index in range(1, 6)}
    if any(forbidden_headers & set(csv_headers(path)) for path in output_csvs):
        raise AssertionError("출력 CSV에 answer1~answer5가 포함됐습니다.")

    source_sha256 = hashlib.sha256(source_csv.read_bytes()).hexdigest()
    manifest = {
        "version": 1,
        "seed": args.seed,
        "source_csv": str(source_csv),
        "source_csv_sha256": source_sha256,
        "source_rows": len(rows),
        "original_headers": original_headers,
        "retained_source_headers": list(SAFE_SOURCE_COLUMNS),
        "removed_source_headers": [header for header in original_headers if header not in SAFE_SOURCE_COLUMNS],
        "counts": {
            "calibration": len(calibration),
            "mine": len(mine),
            "teammate": len(teammate),
            "total_unique": len(all_ids),
            "mine_teammate_overlap": len(overlap),
        },
        "type_distribution": {
            "calibration": distribution(calibration),
            "mine": distribution(mine),
            "teammate": distribution(teammate),
        },
        "label_status": "blank_pending_human_review",
        "images_included": args.include_images,
        "path_contract": "original dataset-relative path" if not args.include_images else "packet-relative path",
    }
    zip_path = output / "teammate_handoff.zip" if not args.no_zip else None
    manifest["teammate_zip"] = str(zip_path) if zip_path else None
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if zip_path:
        guide_dir = Path(__file__).resolve().parent
        make_teammate_zip(
            output,
            guide_dir / "LABELING_GUIDE_DRAFT.md",
            guide_dir / "TEAMMATE_HANDOFF.md",
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
