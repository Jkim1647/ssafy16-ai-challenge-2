"""dev 재라벨: 여러 사람의 labels.csv 병합 → 9B zero-shot과 대조 → 가린 재판정 대기열 생성.

    python tools/relabel_merge_compare.py --inputs a/labels.csv b/labels.csv ... [--packet teammate]

- 입력은 teammate_handoff 패킷과 같은 21열 형식이면 어느 파일이든 된다(teammate / other).
- 병합: id 기준. 한 사람만 작성한 행은 그대로, 여러 사람이 다르게 작성했으면 conflict로 표시한다.
  **원본 labels.csv(각 패킷 폴더)는 수정하지 않는다.** 결과는 teammate_handoff/merged/에 쓴다.
- 대조 상대는 9B zero-shot(llm_draft.csv). dev를 학습하지 않았고 화면에서 기본 숨김이라 비교적 독립적이다.
  Claude 초안은 검수 때 자동 적용돼 사람 라벨과 독립적이지 않으므로 대조 기준으로 쓰지 않는다.
- 재판정 파일(recheck/labels.csv)에는 이전 판정·모델 답을 넣지 않는다(가린 재판정).
  비교용 키는 merged/recheck_key.csv에 따로 둔다 — 재판정이 끝나기 전에 열지 않는다.

answer1~5(원래 라벨러 응답)는 읽지 않는다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PACKET = REPO / "teammate_handoff"
PROTECTED = ["id", "path", "question", "a", "b", "c", "d", "auto_type", "assignment", "is_calibration"]
WRITABLE = ["human_answer", "confidence", "readability", "quality_reason", "data_usage",
            "ambiguity_reason", "evidence", "reviewer", "llm_assistance", "review_status", "review_note"]
KEY_FIELDS = ["human_answer", "review_status", "readability", "data_usage"]


def read(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="병합할 labels.csv들")
    ap.add_argument("--min-prob", type=float, default=0.0, help="이 이상 9B 확률인 불일치만 대기열에")
    args = ap.parse_args()

    out = PACKET / "merged"
    out.mkdir(parents=True, exist_ok=True)

    frames = []
    for i, p in enumerate(args.inputs):
        df = read(Path(p))
        missing = [c for c in PROTECTED + WRITABLE if c not in df.columns]
        if missing:
            print(f"{p}: 열 누락 {missing}")
            return 1
        df["_src"] = f"{i}:{Path(p).parent.name}/{Path(p).name}"
        frames.append(df[df.review_status != ""])
    filled = pd.concat(frames, ignore_index=True)

    # 같은 id를 여러 입력이 작성했으면 판정 핵심 필드가 같은지 본다
    rows, conflicts = [], []
    for rid, g in filled.groupby("id", sort=False):
        uniq = g.drop_duplicates(subset=KEY_FIELDS)
        rows.append(g.iloc[-1])
        if len(uniq) > 1:
            conflicts.append({"id": rid, "sources": " | ".join(g._src),
                              **{f: " / ".join(g[f]) for f in KEY_FIELDS + ["reviewer"]}})
    merged = pd.DataFrame(rows).drop(columns="_src")
    merged["conflict"] = merged.id.isin({c["id"] for c in conflicts}).map({True: "1", False: ""})

    # 9B 초안(teammate + other + rest 어느 쪽이든)
    nine = pd.concat([read(p) for p in [PACKET / "teammate" / "llm_draft.csv", PACKET / "other" / "llm_draft.csv"]
                      if p.exists()]).drop_duplicates("id")[["id", "llm_answer", "llm_prob"]]
    m = merged.merge(nine, on="id", how="left")
    m["llm_prob"] = pd.to_numeric(m.llm_prob, errors="coerce")

    ap_ = m[m.review_status == "approved"]
    agree = (ap_.human_answer == ap_.llm_answer)
    dis = ap_[~agree & (ap_.llm_prob >= args.min_prob)].sort_values("llm_prob", ascending=False)

    # 원본 행(보호 열)을 가져와 작성 열을 비운 재판정 패킷을 만든다
    base = pd.concat([read(p) for p in [PACKET / "teammate" / "labels.csv", PACKET / "other" / "labels.csv"]])
    base = base.drop_duplicates("id").set_index("id")
    rc = base.loc[dis.id].reset_index()
    for c in WRITABLE:
        rc[c] = ""
    rc["assignment"] = "recheck"
    (PACKET / "recheck").mkdir(exist_ok=True)
    existing = PACKET / "recheck" / "labels.csv"
    if existing.exists() and (read(existing).review_status != "").any():
        # 이미 재판정한 행이 있으면 덮어쓰지 않는다 — 사람 작업 보호
        done = read(existing)
        print(f"recheck/labels.csv에 이미 재판정 {int((done.review_status != '').sum())}건이 있어 새로 만들지 않는다. "
              "새 대기열이 필요하면 기존 파일을 다른 이름으로 옮긴 뒤 다시 실행")
        return 2
    cols = list(read(PACKET / "teammate" / "labels.csv").columns)
    # 사람이 Excel로 여는 파일이라 utf-8-sig (CLAUDE.md 인코딩 규칙 2)
    rc[cols].to_csv(PACKET / "recheck" / "labels.csv", index=False, encoding="utf-8-sig", lineterminator="\r\n")
    dis[["id", "human_answer", "reviewer", "llm_answer", "llm_prob"]].rename(
        columns={"human_answer": "first_human_answer", "reviewer": "first_reviewer"}).to_csv(
        out / "recheck_key.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(out / "labels_merged.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(conflicts).to_csv(out / "conflicts.csv", index=False, encoding="utf-8-sig")

    n = len(merged)
    print(f"병합: 입력 {len(args.inputs)}개 → 작성된 id {n}건 (충돌 {len(conflicts)}건)")
    print("상태", merged.review_status.value_counts().to_dict())
    print(f"사람 승인 {len(ap_)}건 중 9B와 정답 일치 {int(agree.sum())}건 ({agree.mean():.1%})" if len(ap_) else "승인 없음")
    if len(ap_):
        for lo, hi in [(0.9, 1.01), (0.7, 0.9), (0.0, 0.7)]:
            s = ap_[(ap_.llm_prob >= lo) & (ap_.llm_prob < hi)]
            if len(s):
                print(f"  9B 확률 {lo}~{min(hi, 1)}: {len(s)}건, 일치 {(s.human_answer == s.llm_answer).mean():.1%}")
    print(f"재판정 대기열: {len(dis)}건 → {PACKET / 'recheck' / 'labels.csv'} (9B 확신 높은 순)")
    print(f"비교 키: {out / 'recheck_key.csv'} (재판정 끝나기 전 열지 말 것)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
