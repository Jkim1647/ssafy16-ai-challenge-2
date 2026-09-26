# -*- coding: utf-8 -*-
"""여러 제출 CSV 중 **하나라도 답이 다른 test 문항**을 한 페이지(html)와 Codex 입력(jsonl)으로 만든다.

규칙 경계는 tools/make_diff_review.py 와 같다 — 이 산출물로 사람이나 AI 가 문항별 정답을 판단해
제출을 고르면 Kaggle 기본규칙 4-b(수기 라벨링) 위반이고, 외부 호스팅 모델(Codex 등)의 판단을 답으로
쓰면 API 추론 금지에도 걸린다. 쓰임은 **어떤 유형에서 방식끼리 갈리나**의 진단뿐이다.

usage:
  python tools/make_multi_diff_review.py --sub T4=a.csv --sub T6=b.csv --sub T7=c.csv \
     --member 35B=<jsonl> ... --out reports/diff_T4_T6_T7
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ensemble_submit import load_any  # noqa: E402
from make_diff_review import lsm, read_sub  # noqa: E402

LETTERS = "abcd"
COLORS = ["#0a66c2", "#b3261e", "#1b7f3b", "#8e44ad", "#b76e00"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", action="append", required=True, help="이름=제출csv (2개 이상)")
    ap.add_argument("--member", action="append", default=[], help="이름=예측 jsonl/csv")
    ap.add_argument("--csv", default="ssafy-16-2-ai/test.csv")
    ap.add_argument("--types", default="data_meta/question_types.csv")
    ap.add_argument("--image-prefix", default="../ssafy-16-2-ai/")
    ap.add_argument("--out", required=True, help="확장자 없는 경로. .html 과 .jsonl 을 쓴다")
    a = ap.parse_args()

    subs = [(s.split("=", 1)[0], read_sub(s.split("=", 1)[1])) for s in a.sub]
    with open(a.csv, encoding="utf-8-sig", newline="") as fh:
        rows = {r["id"]: r for r in csv.DictReader(fh)}
    types = {}
    if os.path.exists(a.types):
        with open(a.types, encoding="utf-8-sig", newline="") as fh:
            types = {r["row_id"]: r["auto_type"] for r in csv.DictReader(fh)}
    mem = [(s.split("=", 1)[0], load_any(s.split("=", 1)[1])) for s in a.member]
    ids = [i for i in subs[0][1] if len({d[i] for _, d in subs}) > 1]

    # 쌍별 변경 수(요약)
    pair = []
    for x in range(len(subs)):
        for y in range(x + 1, len(subs)):
            pair.append(f"{subs[x][0]}↔{subs[y][0]} {sum(subs[x][1][i] != subs[y][1][i] for i in subs[0][1])}")

    esc = html.escape
    css = """
:root{--bg:#fff;--fg:#1d1d1f;--mut:#6e6e73;--card:#f5f5f7;--warn:#fff4e5;--wb:#b76e00}
@media (prefers-color-scheme:dark){:root{--bg:#161618;--fg:#f2f2f2;--mut:#a1a1a6;--card:#232326;--warn:#3a2c12;--wb:#ffb74d}}
body{background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,'Malgun Gothic',sans-serif;margin:0 auto;padding:16px;max-width:1100px}
.warn{background:var(--warn);border-left:4px solid var(--wb);padding:10px 14px;border-radius:6px;margin:12px 0}
.card{background:var(--card);border-radius:10px;padding:14px;margin:14px 0;display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:14px}
@media (max-width:760px){.card{grid-template-columns:1fr}}
img{width:100%;height:auto;border-radius:6px}
.ch{margin:2px 0;padding:3px 6px;border-radius:4px}
.tag{display:inline-block;padding:1px 6px;border-radius:4px;color:#fff;font-size:12px;margin-right:4px}
table{border-collapse:collapse;font-size:13px;margin-top:8px;width:100%}
td,th{padding:2px 6px;border-bottom:1px solid #8883;text-align:left}.mut{color:var(--mut)}
"""
    out = [f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>{esc(' vs '.join(n for n, _ in subs))} 차이</title><style>{css}</style></head><body>",
           f"<h1>{esc(' / '.join(n for n, _ in subs))} — 답이 하나라도 다른 test {len(ids)}문항</h1>",
           f"<p class='mut'>쌍별 변경 수: {esc(' · '.join(pair))}</p>",
           "<div class='warn'><b>규칙 4-b · API 추론 금지.</b> 이 페이지로 사람이나 AI(Codex 포함)가 문항별 정답을 판단해 "
           "<b>제출을 고르거나 답을 고치면 안 된다.</b> 보는 목적은 <b>어떤 유형에서 방식끼리 갈리는지</b>의 진단이다. "
           "제출 선택은 정답이 있는 H408·train 검증셋으로 한다.</div>"]
    jl = []
    for i in ids:
        r = rows[i]
        tag = "".join(f"<span class='tag' style='background:{COLORS[k % len(COLORS)]}'>{esc(n)}: {d[i]}</span>"
                      for k, (n, d) in enumerate(subs))
        ch = "".join(f"<div class='ch'><b>{x}.</b> {esc(r[x])} "
                     + "".join(f"<span class='tag' style='background:{COLORS[k % len(COLORS)]}'>{esc(n)}</span>"
                               for k, (n, d) in enumerate(subs) if d[i] == x)
                     + "</div>" for x in LETTERS)
        tb, mrec = "", {}
        if mem:
            tb = "<table><tr><th>구성원</th><th>선택</th>" + "".join(f"<th>{x}</th>" for x in LETTERS) + "</tr>"
            for n, d in mem:
                if i not in d:
                    continue
                p = np.exp(lsm(d[i]["logprobs"]))
                mrec[n] = {"pick": LETTERS[int(p.argmax())], "probs": [round(float(v), 3) for v in p],
                           "transcript": d[i].get("transcript")}
                tb += f"<tr><td>{esc(n)}</td><td>{LETTERS[int(p.argmax())]}</td>" + "".join(f"<td>{v:.2f}</td>" for v in p) + "</tr>"
            tb += "</table>"
        tr = "".join(f"<p class='mut'><b>{esc(n)} 판독문:</b> {esc(v['transcript'][:400])}</p>"
                     for n, v in mrec.items() if v.get("transcript"))
        out.append(f"<div class='card'><div><img loading='lazy' src='{esc(a.image_prefix + r['path'])}' alt='{esc(i)}'></div>"
                   f"<div><div class='mut'>{esc(i)} · 유형 {esc(types.get(i, '?'))}</div><p><b>{esc(r['question'])}</b></p>"
                   f"{ch}<p>{tag}</p>{tb}{tr}</div></div>")
        jl.append({"id": i, "image": a.image_prefix + r["path"], "type": types.get(i), "question": r["question"],
                   "choices": {x: r[x] for x in LETTERS}, "answers": {n: d[i] for n, d in subs}, "members": mrec})
    out.append("</body></html>")
    with open(a.out + ".html", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out))
    with open(a.out + ".jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for x in jl:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"{len(ids)}문항 -> {a.out}.html / .jsonl | {' · '.join(pair)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
