"""통합 dev 라벨 마스터 생성.

    baseline/Scripts/python.exe tools/build_dev_master.py

두 출처를 한 표에 담되 섞지 않는다(질문이 서로 다르므로 별도 행, source로 구분):
  1) 원본 dev 2,683 질문 — 사람 라벨(우리 hard_ext 회수분 + gold_v2 검증분)을 붙인다.
  2) Codex 재생성 QA 1,317 — dev 이미지에 새 질문을 생성한 것. 사람 검수 대기(pending)이므로
     verified=pending으로 표시해 검수 전까지 학습에 쓰지 않는다(CLAUDE.md 생성형 증강 주의).

출력: runs/dev_validation_v1/dev_label_master.csv (utf-8-sig, 사람이 Excel로 확인)
join: Codex source_id == dev.id. 재생성 행 id는 dev_regen_XXXX.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEV = REPO / "ssafy-16-2-ai/dev.csv"
RET = REPO / "label_packets/hard_ext_20260922/returned"
GOLD2 = REPO / "runs/dev_validation_v1/hard_eval_gold_v2.json"
CODEX = REPO / "teammate_handoff/codex_dev_regen/full_1317.csv"  # 석웅 Codex 재생성 QA 정리분
OUT = REPO / "runs/dev_validation_v1/dev_label_master.csv"

COLS = ["id", "image_id", "path", "source", "question", "a", "b", "c", "d",
        "answer", "answer_source", "review_status", "verified", "evidence", "issue", "src_file"]


def original_rows() -> pd.DataFrame:
    dev = pd.read_csv(DEV, encoding="utf-8-sig", dtype=str).fillna("")
    # 우리 사람 라벨(hard_ext 회수분): id 기준 최신 검수
    ret = pd.concat([pd.read_csv(f, encoding="utf-8-sig", dtype=str).fillna("")
                     for f in glob.glob(str(RET / "*.csv"))], ignore_index=True) if list(RET.glob("*.csv")) else pd.DataFrame()
    lab = {}
    for r in ret.to_dict("records"):
        if r.get("review_status"):
            lab[r["id"]] = r
    gold2 = set(json.loads(GOLD2.read_text(encoding="utf-8-sig"))) if GOLD2.exists() else set()
    gold2ans = json.loads(GOLD2.read_text(encoding="utf-8-sig")) if GOLD2.exists() else {}
    rows = []
    for r in dev.to_dict("records"):
        i = r["id"]
        ans, asrc, rst, ver, ev, iss, sf = "", "", "unlabeled", "none", "", "", ""
        if i in lab:
            l = lab[i]
            ans, asrc = l.get("human_answer", ""), "human"
            rst = l.get("review_status", "")
            ver = "human" if rst == "approved" else "none"
            ev, sf = l.get("evidence", ""), "hard_ext_returned"
            if l.get("ambiguity_reason"):
                iss = l["ambiguity_reason"]
        elif i in gold2:
            ans, asrc, rst, ver, sf = gold2ans[i], "human", "approved", "human", "gold_v2"
        rows.append({"id": i, "image_id": i, "path": r["path"], "source": "original",
                     "question": r["question"], "a": r["a"], "b": r["b"], "c": r["c"], "d": r["d"],
                     "answer": ans, "answer_source": asrc, "review_status": rst, "verified": ver,
                     "evidence": ev, "issue": iss, "src_file": sf})
    return pd.DataFrame(rows)


def regen_rows() -> pd.DataFrame:
    if not CODEX.exists():
        print(f"[경고] Codex 파일 없음: {CODEX} — 재생성 행 생략")
        return pd.DataFrame(columns=COLS)
    cx = pd.read_csv(CODEX, encoding="utf-8-sig", dtype=str).fillna("")
    rows = []
    for r in cx.to_dict("records"):
        hrs = r.get("human_review_status", "")
        # LLM 자동승인이라도 사람 검수 전이면 verified=pending → 학습 제외
        ver = "human" if hrs == "approved" else ("pending" if hrs in ("", "pending") else hrs)
        rows.append({"id": r["id"], "image_id": r.get("image_id", ""), "path": r["path"], "source": "codex_regen",
                     "question": r["question"], "a": r["a"], "b": r["b"], "c": r["c"], "d": r["d"],
                     "answer": r.get("answer", ""), "answer_source": "llm_regen",
                     "review_status": r.get("review_status", ""), "verified": ver,
                     "evidence": r.get("evidence", ""), "issue": r.get("semantic_review_issue", ""),
                     "src_file": "codex_full_1317"})
    return pd.DataFrame(rows)


def main() -> None:
    o, g = original_rows(), regen_rows()
    m = pd.concat([o[COLS], g[COLS]], ignore_index=True)
    OUT.write_text(m.to_csv(index=False), encoding="utf-8-sig")
    print(f"dev_label_master: {len(m)}행 → {OUT}")
    print("source:", dict(m["source"].value_counts()))
    print("original verified:", dict(o[o.source == 'original']["verified"].value_counts()))
    print("regen verified:", dict(g["verified"].value_counts()) if len(g) else "{}")
    train_ok = m[(m.verified == "human") & (m.answer != "")]
    print(f"학습 사용 가능(human 검증+정답 있음): {len(train_ok)}  "
          f"(원본 {len(train_ok[train_ok.source=='original'])} / 재생성 {len(train_ok[train_ok.source=='codex_regen'])})")
    print(f"재생성 미검증 보류(pending): {len(g[g.verified=='pending']) if len(g) else 0} — 사람 검수 후 학습 편입")


if __name__ == "__main__":
    main()
