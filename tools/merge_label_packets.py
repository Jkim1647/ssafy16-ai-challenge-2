"""돌려받은 라벨링 패킷 CSV를 검증·병합해 어려운 평가셋 v3를 만든다.

    baseline/Scripts/python.exe tools/merge_label_packets.py
    baseline/Scripts/python.exe tools/merge_label_packets.py --batch hard_ext_20260922

입력: label_packets/<batch>/returned/*.csv  (팀원이 보낸 labels_<batch>_pN_이름.csv 를 여기에 둔다)
출력: runs/dev_validation_v1/
  hard_eval_gold_v3.json          H184(v2) + 이번 승인분 {id: answer}  (기계용, BOM 없음)
  hard_ext_20260922_merged.csv    전체 행 + 검증 결과 (사람용, BOM)
  hard_ext_20260922_report.md     패킷별 완료·승인·보류·거절·오류 수
승인(approved)이면서 검증을 통과한 행만 정답셋에 들어간다. 보류·거절·오류 행은 제외한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from dev_relabel_app import validate  # noqa: E402  같은 판정 규칙을 쓴다

OUT = REPO / "runs/dev_validation_v1"
GOLD_V2 = OUT / "hard_eval_gold_v2.json"
FIXED = ["id", "path", "question", "a", "b", "c", "d", "auto_type"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="hard_ext_20260922")
    args = ap.parse_args()
    root = REPO / "label_packets" / args.batch
    files = sorted((root / "returned").glob("*.csv"))
    if not files:
        raise SystemExit(f"반환 CSV 없음: {root / 'returned'}")

    frames, lines = [], ["| 파일 | 패킷 | 행 | 완료 | 승인 | 보류 | 거절 | 오류 |", "|---|---|---|---|---|---|---|---|"]
    for f in files:
        df = pd.read_csv(f, encoding="utf-8-sig", dtype=str).fillna("")
        pid = next((p for p in sorted(root.glob("p*")) if p.is_dir() and p.name in f.stem.split("_")), None)
        if pid is None:
            raise SystemExit(f"파일명에서 패킷 번호(pN)를 찾지 못함: {f.name}")
        items = pd.read_csv(pid / f"items_{pid.name}.csv", encoding="utf-8-sig", dtype=str).fillna("")
        # 원본 열·행이 바뀌지 않았는지 확인
        if list(df["id"]) != list(items["id"]) or not (df[FIXED].values == items[FIXED].values).all():
            raise SystemExit(f"{f.name}: id·질문·보기가 원본 패킷과 다르다")
        df["packet"] = pid.name
        df["source_file"] = f.name
        df["error"] = [validate(r) or "" if r["review_status"] else "" for r in df.to_dict("records")]
        frames.append(df)
        done = df["review_status"] != ""
        ok = done & (df["error"] == "")
        c = df.loc[ok, "review_status"].value_counts()
        lines.append(f"| {f.name} | {pid.name} | {len(df)} | {ok.sum()} | {c.get('approved', 0)} | "
                     f"{c.get('needs_review', 0)} | {c.get('rejected', 0)} | {(done & ~ok).sum()} |")

    m = pd.concat(frames, ignore_index=True)
    dup = m["id"][m["id"].duplicated()]
    if len(dup):
        raise SystemExit(f"같은 문항이 여러 파일에 있다(같은 패킷을 두 번 받았는지 확인): {sorted(set(dup))[:5]}")
    gold = json.loads(GOLD_V2.read_text(encoding="utf-8-sig"))
    new = m[(m["review_status"] == "approved") & (m["error"] == "")]
    overlap = set(new["id"]) & set(gold)
    if overlap:
        raise SystemExit(f"H184와 겹치는 문항: {sorted(overlap)[:5]}")
    v3 = {**gold, **dict(zip(new["id"], new["human_answer"]))}
    (OUT / "hard_eval_gold_v3.json").write_text(json.dumps(v3, ensure_ascii=False, indent=0), encoding="utf-8")
    m.to_csv(OUT / f"{args.batch}_merged.csv", index=False, encoding="utf-8-sig")
    lines += ["", f"- 기존 H184 {len(gold)} + 신규 승인 {len(new)} = **{len(v3)}건** → `hard_eval_gold_v3.json`",
              f"- 신규 승인 유형: {new['auto_type'].value_counts().to_dict()}",
              f"- 신규 승인 정답 분포: {new['human_answer'].value_counts().sort_index().to_dict()}"]
    (OUT / f"{args.batch}_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
