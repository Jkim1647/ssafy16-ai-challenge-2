"""어려운 평가셋 확장용 dev 라벨링 패킷 생성 (1인 100문항, 각자 PC에서 작업).

    baseline/Scripts/python.exe tools/build_label_packets.py            # 5명 x 100
    baseline/Scripts/python.exe tools/build_label_packets.py --people 4 --per 100

출력: label_packets/<batch>/pN/ (index.html + images/ + README.txt) 와 pN.zip.
패킷은 파이썬 없이 index.html 을 Chrome/Edge로 열면 된다. 이미지·라벨이 들어 있으므로
label_packets/ 는 git에 올리지 않는다(.gitignore).

선정 기준 (재현 가능, seed 고정):
  - H184(hard_eval_gold_v2.json)와 이미 사람이 판정한 문항 제외
  - 무결성·train/test 중복 격리 문항 제외
  - 9B != 35B 이거나 35B 확신도 < 0.9 인 '어려운' 문항에서 무작위 추출
  - auto_type 별로 섞은 뒤 사람별로 번갈아 배정해 유형 분포를 맞춘다
모델 답·초안은 패킷에 넣지 않는다(평가용 정답이므로 가린 판정).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "runs/dev_validation_v1/dev_audit_all_2683.csv"
GOLD = REPO / "runs/dev_validation_v1/hard_eval_gold_v2.json"
DATA = REPO / "ssafy-16-2-ai"
EXCLUDE_USE = {"hard_eval_visual_verified", "quarantine_integrity", "quarantine_overlap"}
COLS = ["id", "path", "question", "a", "b", "c", "d", "auto_type"]


def existing_ids(batch: str) -> set[str]:
    """이미 만들어 둔 패킷들(items_pN.csv)이 담은 문항 id — 새 패킷이 겹치지 않게 제외한다."""
    ids: set[str] = set()
    for f in (REPO / "label_packets" / batch).glob("p*/items_p*.csv"):
        ids |= set(pd.read_csv(f, encoding="utf-8-sig", dtype=str)["id"])
    return ids


def select(people: int, per: int, seed: int, exclude: set[str] | None = None) -> list[pd.DataFrame]:
    a = pd.read_csv(AUDIT, encoding="utf-8-sig", dtype=str).fillna("")
    gold = set(json.loads(GOLD.read_text(encoding="utf-8-sig")))
    conf = pd.to_numeric(a["b35_confidence"], errors="coerce").fillna(0)
    hard = (a["nine_answer"] != a["b35_answer"]) | (conf < 0.9)
    pool = a[
        hard
        & ~a["id"].isin(gold)
        & ~a["id"].isin(exclude or set())
        & ~a["use"].isin(EXCLUDE_USE)
        & (a["human_status"] == "")
        & (a["human_answer"] == "")
        & (a["test_overlap_ids"] == "")
        & (a["train_overlap_ids"] == "")
    ]
    need = people * per
    if len(pool) < need:
        raise SystemExit(f"후보 {len(pool)}개 < 필요 {need}개")
    rng = random.Random(seed)
    ids = sorted(pool["id"])
    rng.shuffle(ids)
    chosen = pool.set_index("id").loc[ids[:need]].reset_index()
    # 유형별로 모아 순서대로 번갈아 배정 -> 사람마다 유형 분포가 비슷해진다
    chosen = chosen.sort_values("auto_type", kind="stable").reset_index(drop=True)
    buckets: list[list[int]] = [[] for _ in range(people)]
    for i in range(need):
        buckets[i % people].append(i)
    out = []
    for b in buckets:
        part = chosen.loc[b, COLS].copy()
        order = list(part.index)
        rng.shuffle(order)  # 작업 순서는 섞는다(유형이 몰려 나오지 않게)
        out.append(part.loc[order].reset_index(drop=True))
    return out


def build(batch: str, parts: list[pd.DataFrame], start: int = 1) -> Path:
    root = REPO / "label_packets" / batch
    root.mkdir(parents=True, exist_ok=True)
    tmpl = (REPO / "tools/label_packet_template.html").read_text(encoding="utf-8")
    readme = (REPO / "tools/label_packet_README.txt").read_text(encoding="utf-8")
    for n, part in enumerate(parts, start):
        pid = f"p{n}"
        d = root / pid
        if d.exists():
            shutil.rmtree(d)
        (d / "images").mkdir(parents=True)
        rows = []
        for r in part.to_dict("records"):
            src = DATA / r["path"]
            if not src.is_file():
                raise SystemExit(f"이미지 없음: {src}")
            shutil.copy2(src, d / "images" / src.name)
            rows.append({**r, "img": f"images/{src.name}"})
        meta = {"packet": f"{batch}_{pid}", "batch": batch, "pid": pid, "n": len(rows)}
        html = tmpl.replace("/*__META__*/null", json.dumps(meta, ensure_ascii=False)).replace(
            "/*__ROWS__*/[]", json.dumps(rows, ensure_ascii=False)
        )
        (d / "index.html").write_text(html, encoding="utf-8")
        (d / "README.txt").write_text(readme.replace("{PID}", pid).replace("{BATCH}", batch), encoding="utf-8-sig")
        part.to_csv(d / f"items_{pid}.csv", index=False, encoding="utf-8-sig")
        zpath = root / f"{batch}_{pid}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
            for f in sorted(d.rglob("*")):
                if f.is_file():
                    z.write(f, Path(pid) / f.relative_to(d))
        print(f"{pid}: {len(rows)}문항  유형 {part['auto_type'].value_counts().to_dict()}  -> {zpath.name} ({zpath.stat().st_size/1e6:.1f}MB)")
    return root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", type=int, default=5)
    ap.add_argument("--per", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--batch", default="hard_ext_20260922")
    ap.add_argument("--add", type=int, default=0,
                    help="기존 패킷은 두고 추가 패킷 N개 생성(이미 배정된 문항 제외). 예: --add 1 이면 p6 하나 추가")
    args = ap.parse_args()
    if args.add:
        exclude = existing_ids(args.batch)
        nums = sorted(int(p.name[1:]) for p in (REPO / "label_packets" / args.batch).glob("p*") if p.name[1:].isdigit())
        start = (max(nums) + 1) if nums else 1
        # 추가분은 기존과 다른 seed로 뽑아 재현성은 유지하되 겹치지 않게 한다
        parts = select(args.add, args.per, args.seed + start, exclude=exclude)
        root = build(args.batch, parts, start=start)
        print(f"추가 패킷 p{start}..p{start + args.add - 1}, 기존 제외 {len(exclude)}개")
    else:
        parts = select(args.people, args.per, args.seed)
        root = build(args.batch, parts)
    print("출력:", root)


if __name__ == "__main__":
    main()
