"""ZIP 중앙 디렉터리 메타데이터((file_size, crc32) 쌍)로 중복 이미지 후보를 찾는다.

이미지 내용을 열지 않고도(암호화된 ZIP이라 못 연다) 중앙 디렉터리에 노출된
file_size·crc32만으로 중복 후보를 잡을 수 있다 — CRC32가 우연히 같은 (file_size,
crc32) 조합을 가질 확률은 사실상 0에 가깝다(09/17 실측: 기대값 0.03건인데 45건
발견 — 1,489배).

09/21에 실제 데이터가 풀리면 이 스크립트를 그대로 재사용해 SHA-256으로 재검증한다
(notebooks/day1_dataset_audit.ipynb 3절이 그 역할). 그러니 **하드코딩하지 않는다** —
매번 zip_inventory.csv에서 다시 계산한다.

실행:
    python tools/extract_duplicates.py

출력: data_meta/known_duplicates.csv
  (group_id, split_a, stem_a, split_b, stem_b, file_size, crc32, cross_split)
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

# 09/17 실측 당시 reports/research/에 두기로 했었으나 실제로는 tools/zip_stats.py가
# 저장소 루트에 생성한다 — 둘 다 확인해서 있는 쪽을 쓴다.
CANDIDATE_INVENTORY_PATHS = [
    REPO_ROOT / "zip_inventory.csv",
    REPO_ROOT / "reports" / "research" / "zip_inventory.csv",
]
OUT_PATH = REPO_ROOT / "data_meta" / "known_duplicates.csv"
FIELDS = ["group_id", "split_a", "stem_a", "split_b", "stem_b", "file_size", "crc32", "cross_split"]


def find_inventory() -> Path:
    for p in CANDIDATE_INVENTORY_PATHS:
        if p.exists():
            return p
    raise SystemExit(
        "zip_inventory.csv를 못 찾았습니다. 후보 경로: "
        f"{[str(p) for p in CANDIDATE_INVENTORY_PATHS]}\n"
        "tools/zip_stats.py로 먼저 생성하세요."
    )


def load_entries(inventory_path: Path) -> list[dict]:
    with open(inventory_path, encoding="utf-8-sig") as f:
        return [row for row in csv.DictReader(f) if row.get("extension", "").lower() == ".jpg"]


def build_duplicate_groups(entries: list[dict]) -> dict[tuple[str, str], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in entries:
        key = (row["file_size"], row["crc32"])
        groups[key].append(row)
    return {k: v for k, v in groups.items() if len(v) > 1}


def main() -> int:
    inventory_path = find_inventory()
    print(f"인벤토리: {inventory_path}")
    entries = load_entries(inventory_path)
    print(f"jpg 항목: {len(entries)}개")

    groups = build_duplicate_groups(entries)
    print(f"중복 후보 그룹: {len(groups)}개 (기준: 같은 file_size + crc32)")

    rows: list[dict] = []
    cross_split_counts: dict[str, int] = defaultdict(int)
    for group_id, ((file_size, crc32), members) in enumerate(sorted(groups.items()), start=1):
        for a, b in combinations(sorted(members, key=lambda r: r["stem"]), 2):
            cross_split = a["split"] != b["split"]
            rows.append({
                "group_id": group_id,
                "split_a": a["split"], "stem_a": a["stem"],
                "split_b": b["split"], "stem_b": b["stem"],
                "file_size": file_size, "crc32": crc32,
                "cross_split": cross_split,
            })
            if cross_split:
                pair = tuple(sorted([a["split"] or "(none)", b["split"] or "(none)"]))
                cross_split_counts[f"{pair[0]}<->{pair[1]}"] += 1
            else:
                cross_split_counts[f"{a['split'] or '(none)'} 내부"] += 1

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8-sig") as f:  # BOM: 사람이 Excel로 여는 산출물
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n저장됨: {OUT_PATH} ({len(rows)}쌍, {len(groups)}그룹)")

    print("\n분류별 쌍 개수:")
    for label, n in sorted(cross_split_counts.items()):
        print(f"  {label}: {n}")

    train_test = sum(1 for r in rows if {r["split_a"], r["split_b"]} == {"train", "test"})
    if train_test:
        print(f"\n*** Train<->Test 중복 {train_test}쌍 — 09/21 스키마 확정 직후 크게 보고할 것 ***")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
