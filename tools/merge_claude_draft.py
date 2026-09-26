"""Claude 1차 판정(묶음별 JSONL)을 teammate/claude_draft.csv 하나로 합친다.

    python tools/merge_claude_draft.py --src <claude_out 폴더>
    python tools/merge_claude_draft.py --src <폴더> --items teammate_handoff/full_dev/rest_items.csv
        --nine teammate_handoff/full_dev/llm_draft_rest.csv --out teammate_handoff/full_dev/claude_draft_rest.csv

- labels.csv는 건드리지 않는다. 사람이 라벨링 화면에서 "Claude 초안 적용" 후 검수·저장한다.
- 각 행을 라벨링 화면과 같은 규칙(dev_relabel_app.validate)으로 검사해 위반을 rule_error에 남긴다.
- 9B 초안(llm_draft.csv)과 정답이 다르면 disagree_9b=1. 검수자가 먼저 볼 행이다.
  (Claude가 approved로 정답을 냈는데 9B가 다른 답을 고른 경우만 센다)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from dev_relabel_app import validate  # noqa: E402

TEAMMATE = REPO / "teammate_handoff" / "teammate"
MODEL_TAG = "claude-opus-5"
FIELDS = ["id", "answer", "confidence", "readability", "quality_reason", "data_usage",
          "review_status", "evidence", "ambiguity_reason"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="batch_*.jsonl 폴더")
    ap.add_argument("--items", default=str(TEAMMATE / "labels.csv"), help="대상 id 목록 CSV")
    ap.add_argument("--nine", default=str(TEAMMATE / "llm_draft.csv"), help="9B 초안 CSV")
    ap.add_argument("--out", default=str(TEAMMATE / "claude_draft.csv"))
    args = ap.parse_args()

    labels = pd.read_csv(args.items, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    valid_ids = set(labels.id)
    rows, problems = [], []
    for f in sorted(Path(args.src).glob("batch_*.jsonl")):
        # 읽기는 항상 utf-8-sig (CLAUDE.md 인코딩 규칙 1). BOM이 없으면 utf-8과 동일하고,
        # 있으면 벗겨 준다 — utf-8로 읽으면 첫 줄 json.loads가 BOM 때문에 실패한다.
        for n, line in enumerate(f.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                problems.append(f"{f.name}:{n} JSON 오류 {exc}")
                continue
            row = {k: str(obj.get(k, "") or "").strip() for k in FIELDS}
            if row["id"] not in valid_ids:
                continue          # 다른 대상 목록의 묶음(같은 폴더 공유)
            rows.append(row)

    df = pd.DataFrame(rows, columns=FIELDS).drop_duplicates("id", keep="last")
    # 형식만 어긋난 행은 가이드대로 맞춘다: partial/unreadable은 정답을 비우고(추정 답은 이유에 남김)
    # 학습 후보에서 뺀다. 판정 자체(상태)는 바꾸지 않는다.
    weak = df.readability.isin(["partial", "unreadable"])
    guess = weak & (df.answer != "")
    df.loc[guess, "ambiguity_reason"] = (df.loc[guess, "ambiguity_reason"] + " (추정 답: "
                                         + df.loc[guess, "answer"] + ")").str.strip()
    df.loc[guess, "answer"] = ""
    df.loc[weak & (df.data_usage == "vqa_train"), "data_usage"] = "vision_auxiliary_candidate"
    df.loc[weak & (df.review_status == "approved"), "review_status"] = "needs_review"
    df["rule_error"] = [
        validate({
            "human_answer": r.answer, "confidence": r.confidence, "readability": r.readability,
            "quality_reason": r.quality_reason, "data_usage": r.data_usage, "review_status": r.review_status,
            "evidence": r.evidence, "ambiguity_reason": r.ambiguity_reason, "review_note": "",
            "reviewer": "claude", "llm_assistance": MODEL_TAG,
        }) or ""
        for r in df.itertuples()
    ]
    nine = Path(args.nine)
    if nine.exists():
        d9 = pd.read_csv(nine, encoding="utf-8-sig", dtype=str, keep_default_na=False)[["id", "llm_answer", "llm_prob"]]
        df = df.merge(d9, on="id", how="left")
        df["disagree_9b"] = ((df.answer != "") & (df.answer != df.llm_answer)).astype(int)
    df["model"] = MODEL_TAG
    out = Path(args.out)
    # 사람이 Excel로 함께 여는 파일이라 utf-8-sig (CLAUDE.md 인코딩 규칙 2)
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"{len(df)}/{len(labels)}건 → {out}")
    print("상태", df.review_status.value_counts().to_dict())
    if "disagree_9b" in df:
        print("9B와 정답 불일치", int(df.disagree_9b.sum()))
    bad = df[df.rule_error != ""]
    print("규칙 위반", len(bad))
    for r in bad.head(10).itertuples():
        print(" ", r.id, r.rule_error)
    for p in problems[:10]:
        print(" ", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
