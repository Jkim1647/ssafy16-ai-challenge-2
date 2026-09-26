# -*- coding: utf-8 -*-
"""한 구성원이 H408 v2 에서 틀린 문항을 이미지와 함께 보여 준다(라벨 검토용). GPU 미사용.
usage: python tools/member_errors_page.py 397B
dev 문항(정답 있음)이라 사람이 보고 라벨을 검토해도 규칙과 무관하다.
"""
import csv, html, io, contextlib, json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import explore_ens_20260924 as E  # noqa: E402
who = sys.argv[1] if len(sys.argv) > 1 else '397B'
L = 'abcd'
g = E.g
rows = {r['id']: r for r in csv.DictReader(open('runs/h408_pack/dev.csv', encoding='utf-8-sig'))}
reviewed = {x['id'] for x in json.load(open('reports/h408_errors_20260924.jsonl', encoding='utf-8'))}
T = E.T
pick = lambda n, i: L[int(np.argmax(E.pr(E.M, n, i)))]
t6 = lambda i: L[int(np.argmax(E.t6(E.M, i)))]
wrong = [i for i in E.ids if pick(who, i) != g[i]]
new = [i for i in wrong if i not in reviewed]
esc = html.escape
out = ["<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
       f"<title>{esc(who)} H408 오답</title><style>:root{{--bg:#fff;--fg:#1d1d1f;--mut:#6e6e73;--card:#f5f5f7;--ok:#1b7f3b;--no:#b3261e;--new:#0a66c2}}"
       "@media (prefers-color-scheme:dark){:root{--bg:#161618;--fg:#f2f2f2;--mut:#a1a1a6;--card:#232326;--ok:#6fd08c;--no:#ff8a80;--new:#6cb4ff}}"
       "body{background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,'Malgun Gothic',sans-serif;margin:0 auto;padding:16px;max-width:1100px}"
       ".card{background:var(--card);border-radius:10px;padding:14px;margin:14px 0;display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:14px}"
       "@media (max-width:760px){.card{grid-template-columns:1fr}}img{width:100%;height:auto;border-radius:6px}"
       ".ok{color:var(--ok);font-weight:600}.no{color:var(--no);font-weight:600}.mut{color:var(--mut)}.new{color:var(--new);font-weight:700}"
       "table{border-collapse:collapse;font-size:13px;width:100%;margin-top:6px}td,th{padding:2px 6px;border-bottom:1px solid #8883;text-align:left}</style></head><body>",
       f"<h1>{esc(who)} 가 H408 v2 에서 틀린 {len(wrong)}문항</h1>",
       f"<p class='mut'>새로 볼 문항 {len(new)}개를 먼저, 오후에 이미 검토하신 {len(wrong) - len(new)}개는 뒤에 둔다. "
       "✅ = 현재 정답(v2). 중복정답이면 두 보기에 ✅.</p>"]
for i in new + [x for x in wrong if x in reviewed]:
    r = rows[i]
    ch = ''.join(f"<div>{'✅ ' if x == g[i] else ''}<b>{x}.</b> {esc(r[x])}</div>" for x in L)
    tb = "<table><tr><th>모델</th><th>선택</th>" + ''.join(f'<th>{x}</th>' for x in L) + '</tr>'
    for n in ['397B', 'R3', '35B', 'Gemma']:
        p = E.pr(E.M, n, i)
        tb += f"<tr><td>{n}</td><td class='{'ok' if L[int(p.argmax())] == g[i] else 'no'}'>{L[int(p.argmax())]}</td>" + ''.join(f'<td>{v:.2f}</td>' for v in p) + '</tr>'
    tb += '</table>'
    tr = (E.M[who][i].get('transcript') or '')[:400]
    tag = "<span class='new'>새로 볼 문항</span>" if i not in reviewed else '이미 검토함'
    out.append(f"<div class='card'><div><img loading='lazy' src='../ssafy-16-2-ai/{esc(r['path'])}' alt='{esc(i)}'></div>"
               f"<div><div class='mut'>{esc(i)} · 유형 {esc(T.get(i, '?'))} · {tag}</div><p><b>{esc(r['question'])}</b></p>{ch}"
               f"<p>정답 <b>{g[i]}</b> · {esc(who)} <span class='no'>{pick(who, i)}</span> · T6 "
               f"<span class='{'ok' if t6(i) == g[i] else 'no'}'>{t6(i)}</span></p>{tb}"
               f"<p class='mut'><b>{esc(who)} 판독문:</b> {esc(tr)}</p></div></div>")
out.append('</body></html>')
fn = f'reports/h408_errors_{who}.html'
open(fn, 'w', encoding='utf-8', newline='\n').write('\n'.join(out))
print(f'{who} 오답 {len(wrong)} (새로 볼 문항 {len(new)}, 이미 검토 {len(wrong) - len(new)}) -> {fn}')
print('새로 볼 문항:', ', '.join(f"{i[4:8]}({T.get(i)})" for i in new))
