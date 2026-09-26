"""dev 전체(2,683건): 9B · Claude · (있으면) 사람 재라벨 · 원래 라벨러 5명 응답을 비교한다.

    python tools/compare_full_dev.py

입력 (없는 것은 건너뛴다)
  teammate_handoff/teammate/{llm_draft,claude_draft,labels}.csv      진영 담당 1,316
  teammate_handoff/full_dev/{llm_draft_rest,claude_draft_rest}.csv   나머지 1,367
  ssafy-16-2-ai/dev.csv                                              answer1~5 (원래 라벨러 응답)

원칙: 재라벨 작업자는 answer1~5를 보면 안 된다(TEAMMATE_HANDOFF.md 입력 경계).
그래서 라벨러 응답은 **집계 수치로만** 보고서에 쓰고, 행별 다수결은 어떤 파일에도 쓰지 않는다.
보고서: teammate_handoff/full_dev/compare_report.md
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TM = REPO / "teammate_handoff" / "teammate"
FULL = REPO / "teammate_handoff" / "full_dev"
LETTERS = ["a", "b", "c", "d"]


def read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def pct(n: int, d: int) -> str:
    return f"{n / d:.1%} ({n}/{d})" if d else "-"


def main() -> int:
    dev = read(REPO / "ssafy-16-2-ai" / "dev.csv")
    nine = pd.concat([read(TM / "llm_draft.csv"), read(FULL / "llm_draft_rest.csv")])
    claude = pd.concat([read(TM / "claude_draft.csv"), read(FULL / "claude_draft_rest.csv")])
    human = read(TM / "labels.csv")

    df = dev[["id"]].copy()
    df["segment"] = "other"
    df.loc[df.id.isin(set(read(TM / "labels.csv").id)), "segment"] = "jinyoung"
    df.loc[df.id.isin(set(read(REPO / "teammate_handoff" / "calibration" / "labels.csv").id)), "segment"] = "calibration"
    df = df.merge(nine[["id", "llm_answer", "llm_prob"]], on="id", how="left")
    df = df.merge(claude[["id", "answer", "review_status", "confidence"]].rename(
        columns={"answer": "c_answer", "review_status": "c_status", "confidence": "c_conf"}), on="id", how="left")
    if not human.empty:
        h = human[human.review_status != ""][["id", "human_answer", "review_status"]].rename(
            columns={"review_status": "h_status"})
        df = df.merge(h, on="id", how="left")
    df = df.fillna("")

    # 라벨러 응답: 메모리 안에서만 집계에 쓴다(파일로 내보내지 않는다)
    votes = dev.set_index("id")[[f"answer{i}" for i in range(1, 6)]]
    maj, maj_n, n_votes = {}, {}, {}
    for i, row in votes.iterrows():
        v = [x.strip().lower() for x in row if x.strip().lower() in LETTERS]
        n_votes[i] = len(v)
        if v:
            (top, k), = Counter(v).most_common(1)
            tie = sum(1 for c in Counter(v).values() if c == k) > 1
            maj[i], maj_n[i] = ("" if tie else top), k
    df["_maj"] = df.id.map(maj).fillna("")
    df["_maj_n"] = df.id.map(maj_n).fillna(0).astype(int)

    p = pd.to_numeric(df.llm_prob, errors="coerce")
    has9, hasC = df.llm_answer != "", df.c_status != ""
    cans = df.c_answer != ""
    lines = ["# dev 전체 비교 (9B · Claude · 라벨러 집계)", ""]
    lines += [f"- 대상 {len(df)}건 · 9B 초안 {int(has9.sum())}건 · Claude 판정 {int(hasC.sum())}건",
              "- 원래 라벨러 응답(answer1~5)은 **집계로만** 사용했다. 행별 다수결은 저장하지 않았다.", ""]

    lines += ["## 1. Claude 판정 분포", "", "| 구간 | 건수 | 승인 | 재검토 | 거절 |", "|---|---|---|---|---|"]
    for name, m in [("전체", hasC), ("진영 담당", hasC & (df.segment == "jinyoung")),
                    ("나머지(석웅)", hasC & (df.segment == "other")), ("calibration", hasC & (df.segment == "calibration")),
                    ("9B ≥0.9", hasC & (p >= 0.9)), ("9B 0.7~0.9", hasC & (p >= 0.7) & (p < 0.9)), ("9B <0.7", hasC & (p < 0.7))]:
        s = df[m]
        if len(s):
            vc = s.c_status.value_counts()
            lines.append(f"| {name} | {len(s)} | {vc.get('approved', 0)/len(s):.0%} | "
                         f"{vc.get('needs_review', 0)/len(s):.0%} | {vc.get('rejected', 0)/len(s):.0%} |")

    lines += ["", "## 2. 9B ↔ Claude 정답 일치", "", "Claude가 정답을 낸(승인) 문항 기준.", ""]
    m = has9 & cans
    lines.append(f"- 전체: {pct(int((df[m].llm_answer == df[m].c_answer).sum()), int(m.sum()))}")
    for name, mm in [("9B ≥0.9", p >= 0.9), ("9B 0.7~0.9", (p >= 0.7) & (p < 0.9)), ("9B <0.7", p < 0.7)]:
        s = df[m & mm]
        lines.append(f"- {name}: {pct(int((s.llm_answer == s.c_answer).sum()), len(s))}")

    lines += ["", "## 3. 원래 라벨러 다수결과의 일치 (집계)", "",
              "dev는 5명 중 4명 이상 합의하지 못한 문항이다. 다수결은 '가장 많이 나온 답'이며 동률은 제외했다.", ""]
    dist = Counter(df._maj_n)
    lines.append("- 다수결 표 수 분포: " + ", ".join(f"{k}표 {v}건" for k, v in sorted(dist.items())))
    hm = df._maj != ""
    lines += ["", "| 비교 | 전체 | 다수결 3표 | 다수결 2표 |", "|---|---|---|---|"]
    for name, col, mask in [("9B", "llm_answer", has9), ("Claude(승인만)", "c_answer", cans)] + (
            [("사람 재라벨(승인만)", "human_answer", df.get("human_answer", pd.Series("", index=df.index)) != "")]
            if "human_answer" in df else []):
        cells = []
        for k in [None, 3, 2]:
            s = df[mask & hm & ((df._maj_n == k) if k else True)]
            cells.append(pct(int((s[col] == s._maj).sum()), len(s)))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    both = has9 & cans & hm
    agree = both & (df.llm_answer == df.c_answer)
    s = df[agree]
    lines.append(f"| 9B=Claude 일치 문항 | {pct(int((s.c_answer == s._maj).sum()), len(s))} | | |")
    s = df[both & ~agree]
    lines += [f"| 9B≠Claude 문항: 9B가 다수결 | {pct(int((s.llm_answer == s._maj).sum()), len(s))} | | |",
              f"| 9B≠Claude 문항: Claude가 다수결 | {pct(int((s.c_answer == s._maj).sum()), len(s))} | | |"]

    lines += ["", "## 4. Claude가 보류/거절한 문항에서 라벨러 합의 정도", "",
              "라벨러도 의견이 크게 갈린 문항에 Claude 보류가 몰려 있는지 본다.", "",
              "| Claude 상태 | 건수 | 다수결 3표 이상 비율 | 다수결 2표 이하 비율 |", "|---|---|---|---|"]
    for st in ["approved", "needs_review", "rejected"]:
        s = df[df.c_status == st]
        if len(s):
            lines.append(f"| {st} | {len(s)} | {(s._maj_n >= 3).mean():.0%} | {(s._maj_n <= 2).mean():.0%} |")

    out = FULL / "compare_report.md"
    FULL.mkdir(parents=True, exist_ok=True)
    # 사람이 여는 보고서라 utf-8-sig (CLAUDE.md 인코딩 규칙 2)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print("\n".join(lines))
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
