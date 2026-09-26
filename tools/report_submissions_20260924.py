# -*- coding: utf-8 -*-
"""2026-09-24 제출별 H408 오답·test 변경을 한 번에 정리한다. GPU 미사용.

산출
  reports/submissions_errors_20260924.md    제출별 H408 점수·유형별 정확도·오답 id·T6 대비 test 변경(유형별)
  reports/h408_errors_by_submission.html     H408 에서 한 제출이라도 틀린 문항 — 이미지·질문·보기·정답·제출별 답·구성원 확률
H408 은 dev 문항(정답 있음)이라 문항을 눈으로 봐도 규칙과 무관하다. test 는 정답이 없어 '바뀐 문항과 유형'만 센다.
T5(shift TTA)는 test 저마진 336건에만 적용한 것이라 H408 값이 없다(= T4).
"""
import csv, html, json, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explore_ens_20260924 as E  # noqa: E402

pr, t6, t7, fam = E.pr, E.t6, E.t7, E.fam
T = E.T


def soft4(D, i):
    return np.exp(np.mean([np.log(np.maximum(pr(D, n, i), 1e-300)) for n in ['R3', '35B', '397B', 'Gemma']], 0))


def t8a(D, i):
    if T.get(i) == 'COMPOSITE':
        return (pr(D, 'R3', i) + fam(D, i, 1, .5, .5) + pr(D, '397B', i) + pr(D, 'Gemma', i)) / 4
    return t6(D, i)


SUBS = {  # 이름: (결합 함수, Public)
    'T4': (soft4, 0.97348), 'T6': (t6, 0.97438), 'T7': (t7, 0.97408), 'T8A': (t8a, 0.97438),
    'T9': (E.CANDS['T6·T7 혼합 0.25:0.75'], 0.97378), 'T10': (E.grid[(1.5, 1.5, 1, 1)], 0.97408),
}
L = 'abcd'
ids, g = E.ids, E.g
H = {k: {i: L[int(np.argmax(f(E.M, i)))] for i in ids} for k, (f, _) in SUBS.items()}
TP = {k: {i: L[int(np.argmax(f(E.TEST, i)))] for i in E.tids} for k, (f, _) in SUBS.items()}
types = [t for t, _ in collections.Counter(T.get(i, '?') for i in ids).most_common()]

md = ['# 제출별 H408 오답·test 변경 (2026-09-24)', '',
      '`tools/report_submissions_20260924.py`. H408 = H229 + 수정본 dev179(정답 있음). test 는 정답이 없어 T6 대비 바뀐 문항 수만 센다.', '',
      '## 1. 점수와 유형별 정확도 (H408)', '',
      '| 제출 | Public | H408 | ' + ' | '.join(f'{t}({sum(T.get(i, "?") == t for i in ids)})' for t in types) + ' |',
      '|---|---|---|' + '---|' * len(types)]
for k, (_, pub) in SUBS.items():
    row = [f'{sum(H[k][i] == g[i] for i in ids if T.get(i, "?") == t)}' for t in types]
    md.append(f'| {k} | {pub} | **{sum(H[k][i] == g[i] for i in ids)}** | ' + ' | '.join(row) + ' |')
md += ['', '유형 뒤 괄호는 H408 문항 수. **유형별 오답률**(T6 기준): ' +
       ', '.join(f'{t} {sum(H["T6"][i] != g[i] for i in ids if T.get(i, "?") == t)}/{sum(T.get(i, "?") == t for i in ids)}' for t in types), '']

md += ['## 2. H408 오답 문항 (제출별)', '']
allwrong = sorted({i for k in SUBS for i in ids if H[k][i] != g[i]})
common = [i for i in allwrong if all(H[k][i] != g[i] for k in SUBS)]
md.append(f'- 모든 제출이 틀린 문항: **{len(common)}** / 한 제출이라도 틀린 문항: {len(allwrong)}')
for k in SUBS:
    w = [i for i in ids if H[k][i] != g[i]]
    only = [i for i in w if i not in common]
    md.append(f'- **{k}** 오답 {len(w)} — 공통 외: ' + (', '.join(f'{i}({T.get(i, "?")})' for i in only) or '없음'))
md.append('- 공통 오답: ' + ', '.join(f'{i}({T.get(i, "?")})' for i in common))

md += ['', '## 3. test 에서 T6 대비 바뀐 문항 (유형별)', '',
       '| 제출 | 변경 수 | ' + ' | '.join(types) + ' |', '|---|---|' + '---|' * len(types)]
for k in SUBS:
    ch = [i for i in E.tids if TP[k][i] != TP['T6'][i]]
    c = collections.Counter(T.get(i, '?') for i in ch)
    md.append(f'| {k} | {len(ch)} | ' + ' | '.join(str(c.get(t, 0)) for t in types) + ' |')
for k in ['T8A', 'T9', 'T10', 'T7', 'T4']:
    ch = [i for i in E.tids if TP[k][i] != TP['T6'][i]]
    md.append(f'- {k} 변경 id: ' + ', '.join(ch[:60]) + (' …' if len(ch) > 60 else ''))

open('reports/submissions_errors_20260924.md', 'w', encoding='utf-8-sig', newline='\n').write('\n'.join(md) + '\n')

# ---- H408 오답 페이지 ----
rows = {}
for p in ['runs/h408_pack/dev.csv']:
    for r in csv.DictReader(open(p, encoding='utf-8-sig')):
        rows[r['id']] = r
esc = html.escape
out = ["<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
       "<title>H408 제출별 오답</title><style>:root{--bg:#fff;--fg:#1d1d1f;--mut:#6e6e73;--card:#f5f5f7;--ok:#1b7f3b;--no:#b3261e}"
       "@media (prefers-color-scheme:dark){:root{--bg:#161618;--fg:#f2f2f2;--mut:#a1a1a6;--card:#232326;--ok:#6fd08c;--no:#ff8a80}}"
       "body{background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,'Malgun Gothic',sans-serif;margin:0 auto;padding:16px;max-width:1100px}"
       ".card{background:var(--card);border-radius:10px;padding:14px;margin:14px 0;display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:14px}"
       "@media (max-width:760px){.card{grid-template-columns:1fr}}img{width:100%;height:auto;border-radius:6px}"
       ".ok{color:var(--ok);font-weight:600}.no{color:var(--no);font-weight:600}.mut{color:var(--mut)}"
       "table{border-collapse:collapse;font-size:13px;width:100%;margin-top:6px}td,th{padding:2px 6px;border-bottom:1px solid #8883;text-align:left}</style></head><body>",
       f"<h1>H408 에서 한 제출이라도 틀린 {len(allwrong)}문항</h1>",
       f"<p class='mut'>정답이 있는 dev 문항이다. 모든 제출 공통 오답 {len(common)}건을 먼저, 나머지는 뒤에 둔다. "
       "유형은 자동 분류(data_meta/question_types.csv).</p>"]
for i in common + [x for x in allwrong if x not in common]:
    r = rows[i]
    ans = ' · '.join(f"<span class='{'ok' if H[k][i] == g[i] else 'no'}'>{k}:{H[k][i]}</span>" for k in SUBS)
    ch = ''.join(f"<div>{'✅ ' if x == g[i] else ''}<b>{x}.</b> {esc(r[x])}</div>" for x in L)
    tb = "<table><tr><th>구성원</th><th>선택</th>" + ''.join(f'<th>{x}</th>' for x in L) + '</tr>'
    for n in ['R3', '35B', '397B', 'Gemma', '35Br2v3', '35Bfs16']:
        if i in E.M[n]:
            p = pr(E.M, n, i)
            tb += f"<tr><td>{n}</td><td class='{'ok' if L[int(p.argmax())] == g[i] else 'no'}'>{L[int(p.argmax())]}</td>" + ''.join(f'<td>{v:.2f}</td>' for v in p) + '</tr>'
    tb += '</table>'
    tag = '모든 제출 오답' if i in common else '일부 제출만 오답'
    out.append(f"<div class='card'><div><img loading='lazy' src='../ssafy-16-2-ai/{esc(r['path'])}' alt='{esc(i)}'></div>"
               f"<div><div class='mut'>{esc(i)} · 유형 {esc(T.get(i, '?'))} · {tag}</div><p><b>{esc(r['question'])}</b></p>{ch}"
               f"<p>정답 <b>{g[i]}</b> | {ans}</p>{tb}</div></div>")
out.append('</body></html>')
open('reports/h408_errors_by_submission.html', 'w', encoding='utf-8', newline='\n').write('\n'.join(out))
print('\n'.join(md))
