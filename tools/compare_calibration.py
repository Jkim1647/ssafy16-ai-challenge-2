"""calibration 50건: 두 작업자의 판정을 비교해 합의할 목록을 뽑는다.

    python tools/compare_calibration.py 내파일.csv 상대파일.csv
    python tools/compare_calibration.py teammate_handoff/calibration/labels.csv 석웅/labels.csv --names 진영 석웅

출력:
  - 콘솔: 항목별 일치율, 불일치 건수
  - <첫 파일 폴더>/calibration_compare.csv: 불일치 행만, 두 사람 값을 나란히 (Excel용 utf-8-sig)

두 파일의 id·question·선택지가 같은지 먼저 검사한다(다른 패킷이면 비교하지 않는다).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

KEYS = ["human_answer", "review_status", "readability", "quality_reason", "data_usage", "confidence"]
NOTES = ["ambiguity_reason", "evidence", "review_note"]
PROTECTED = ["id", "path", "question", "a", "b", "c", "d"]


def load(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mine")
    ap.add_argument("other")
    ap.add_argument("--names", nargs=2, default=["A", "B"])
    args = ap.parse_args()
    na, nb = args.names

    a, b = load(args.mine), load(args.other)
    if len(a) != len(b) or set(a.id) != set(b.id):
        print(f"id 집합이 다르다: {len(a)}행 vs {len(b)}행")
        return 1
    b = b.set_index("id").loc[a.id].reset_index()
    if not a[PROTECTED].equals(b[PROTECTED]):
        print("질문·선택지가 다르다 — 같은 calibration 패킷이 아니다")
        return 1

    a["no"] = range(1, len(a) + 1)
    print(f"{len(a)}건 비교 ({na} vs {nb})")
    print("항목 | 일치율 | 불일치")
    for k in KEYS:
        same = (a[k] == b[k])
        print(f"{k:15s} | {same.mean():6.1%} | {int((~same).sum())}")

    # 정답 판정: 둘 다 정답을 넣은 행에서의 일치율과, 한쪽만 넣은 행을 분리해서 본다
    both = (a.human_answer != "") & (b.human_answer != "")
    one = (a.human_answer != "") ^ (b.human_answer != "")
    print(f"\n둘 다 정답을 낸 {int(both.sum())}건 중 정답 일치 {int((both & (a.human_answer == b.human_answer)).sum())}건"
          f" / 한쪽만 정답을 낸 행 {int(one.sum())}건")
    ct = pd.crosstab(a.review_status.replace("", "(빈칸)"), b.review_status.replace("", "(빈칸)"),
                     rownames=[f"{na} 상태"], colnames=[f"{nb} 상태"])
    print("\n검토 상태 교차표\n" + ct.to_string())

    diff = a[KEYS].ne(b[KEYS]).any(axis=1)
    rows = []
    for i in a.index[diff]:
        row = {"no": a.at[i, "no"], "id": a.at[i, "id"], "question": a.at[i, "question"],
               "다른 항목": ", ".join(k for k in KEYS if a.at[i, k] != b.at[i, k])}
        # 정답이 갈린 행이 가장 먼저 합의할 대상
        row["우선순위"] = 1 if a.at[i, "human_answer"] != b.at[i, "human_answer"] else (
            2 if a.at[i, "review_status"] != b.at[i, "review_status"] else 3)
        for k in KEYS + NOTES:
            row[f"{na}_{k}"] = a.at[i, k]
            row[f"{nb}_{k}"] = b.at[i, k]
        row["합의_결과"] = ""
        rows.append(row)
    out = Path(args.mine).resolve().with_name("calibration_compare.csv")
    pd.DataFrame(rows).sort_values(["우선순위", "no"]).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n불일치 {len(rows)}행 → {out}  (우선순위 1=정답 불일치, 2=상태 불일치, 3=기타)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
