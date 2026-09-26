# -*- coding: utf-8 -*-
"""두 제출 CSV 가 **다르게 답한 test 문항**을 이미지·질문·보기·구성원별 선택과 함께 한 페이지로 보여 준다.

규칙 경계 (tools/make_test_review.py 와 같다)
  Kaggle 기본규칙 4-b 는 test 레코드에 대한 사람의 라벨링·수기 예측을 제출에 쓰는 것을 금지한다.
  이 페이지를 보고 "이 문항은 이쪽이 맞다"를 판단해 **어느 제출을 최종으로 고를지 정하면**
  사람이 test 정답을 매긴 것과 같아진다. 허용되는 쓰임은 "어떤 종류의 문항에서 두 방식이 갈리나"를
  보는 **진단**이다. 최종 선택은 검증셋(H229/H408, train 저마진 302) 근거로 한다.

usage:
  python tools/make_diff_review.py --a T4=submissions/T4.csv --b T5=submissions/T5.csv \
     --member R3=<jsonl> --member 35B=<jsonl> ... --out reports/diff_T4_T5.html
"""
from __future__ import annotations

import argparse
import csv
import html
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ensemble_submit import load_any  # noqa: E402

LETTERS = "abcd"


def lsm(v):
    a = np.clip(np.array(v, float), -1e4, None)
    a = a - a.max()
    return a - np.log(np.exp(a).sum())


def read_sub(p):
    with open(p, encoding="utf-8-sig", newline="") as fh:
        return {r["id"]: r["answer"] for r in csv.DictReader(fh)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="이름=제출csv")
    ap.add_argument("--b", required=True, help="이름=제출csv")
    ap.add_argument("--member", action="append", default=[], help="이름=예측(jsonl/csv) — 구성원별 선택 표시")
    ap.add_argument("--csv", default="ssafy-16-2-ai/test.csv")
    ap.add_argument("--image-prefix", default="../ssafy-16-2-ai/")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    na, pa = a.a.split("=", 1)
    nb, pb = a.b.split("=", 1)
    A, B = read_sub(pa), read_sub(pb)
    with open(a.csv, encoding="utf-8-sig", newline="") as fh:
        rows = {r["id"]: r for r in csv.DictReader(fh)}
    mem = [(s.split("=", 1)[0], load_any(s.split("=", 1)[1])) for s in a.member]
    diff = [i for i in A if A[i] != B.get(i)]

    esc = html.escape
    css = """
:root{--bg:#fff;--fg:#1d1d1f;--mut:#6e6e73;--card:#f5f5f7;--a:#0a66c2;--b:#b3261e;--warn:#fff4e5;--wb:#b76e00}
@media (prefers-color-scheme:dark){:root{--bg:#161618;--fg:#f2f2f2;--mut:#a1a1a6;--card:#232326;--a:#6cb4ff;--b:#ff8a80;--warn:#3a2c12;--wb:#ffb74d}}
body{background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,'Malgun Gothic',sans-serif;margin:0;padding:16px;max-width:1100px;margin:auto}
.warn{background:var(--warn);border-left:4px solid var(--wb);padding:10px 14px;border-radius:6px;margin:12px 0}
.card{background:var(--card);border-radius:10px;padding:14px;margin:14px 0;display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:14px}
@media (max-width:760px){.card{grid-template-columns:1fr}}
img{width:100%;height:auto;border-radius:6px}
.ch{margin:2px 0;padding:3px 6px;border-radius:4px}
.ca{outline:2px solid var(--a)} .cb{outline:2px dashed var(--b)}
table{border-collapse:collapse;font-size:13px;margin-top:8px;width:100%}
td,th{padding:2px 6px;border-bottom:1px solid #8883;text-align:left}
.tagA{color:var(--a);font-weight:600}.tagB{color:var(--b);font-weight:600}.mut{color:var(--mut)}
"""
    out = [f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>{esc(na)} vs {esc(nb)} 차이</title><style>{css}</style></head><body>",
           f"<h1>{esc(na)} vs {esc(nb)} — 답이 다른 test {len(diff)}문항</h1>",
           "<div class='warn'><b>규칙 4-b.</b> 이 페이지를 보고 사람이 정답을 판단해 <b>어느 제출을 최종으로 고를지</b> 정하면 "
           "test 레코드에 대한 사람의 예측을 제출에 쓴 것이 된다. 여기서는 <b>어떤 종류의 문항에서 두 방식이 갈리는지</b>만 본다. "
           "우리에게 test 정답은 없다 — 보이는 것은 ‘갈렸다’이지 ‘틀렸다’가 아니다.</div>",
           f"<p class='mut'><span class='tagA'>실선 = {esc(na)}</span> · <span class='tagB'>점선 = {esc(nb)}</span></p>"]
    for i in diff:
        r = rows[i]
        ch = "".join(
            f"<div class='ch {'ca' if A[i]==x else ''} {'cb' if B[i]==x else ''}'><b>{x}.</b> {esc(r[x])}</div>"
            for x in LETTERS)
        tb = ""
        if mem:
            tb = "<table><tr><th>구성원</th><th>선택</th>" + "".join(f"<th>{x}</th>" for x in LETTERS) + "</tr>"
            for n, d in mem:
                if i not in d:
                    continue
                p = np.exp(lsm(d[i]["logprobs"]))
                tb += f"<tr><td>{esc(n)}</td><td>{LETTERS[int(p.argmax())]}</td>" + "".join(f"<td>{v:.2f}</td>" for v in p) + "</tr>"
            tb += "</table>"
        out.append(f"<div class='card'><div><img loading='lazy' src='{esc(a.image_prefix + r.get('path', 'test/' + i))}' alt='{esc(i)}'></div>"
                   f"<div><div class='mut'>{esc(i)}</div><p><b>{esc(r['question'])}</b></p>{ch}"
                   f"<p><span class='tagA'>{esc(na)}: {A[i]}</span> · <span class='tagB'>{esc(nb)}: {B[i]}</span></p>{tb}</div></div>")
    out.append("</body></html>")
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out))
    print(f"{len(diff)}문항 -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
