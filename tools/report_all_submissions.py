# -*- coding: utf-8 -*-
"""T1~T15 제출 종합 비교 — 무엇이 바뀌었고, 바뀐 문항이 '어려운 문항'인지 '앙상블 실수'인지, 과적합인지 본다. GPU 미사용.

산출
  reports/all_submissions_20260924.md     제출별 요약(Public·H408 v2·test 변경·잡음 판정·다수 의견 뒤집기) + H408 오답 분류
  reports/all_submissions_20260924.html   카드 페이지: (1) test 에서 T6 와 답이 다른 문항 (2) H408 v2 오답 문항
                                          — 이미지·질문·보기·제출별 답·구성원 확률

판정 기준
  - Public 차이(문항) = (Public - T6 Public) x 3357. Public 은 test 의 약 절반(0.00030 = 1문항)이다.
  - 잡음 폭 = sqrt(K/2), K = T6 대비 test 에서 바뀐 답 수(그중 절반쯤이 Public 에 들어가고 각각 +-1).
    |차이| <= 잡음 폭 이면 '잡음', 2배 이내면 '약한 신호', 그 밖은 '유의'.
  - 다수 의견 뒤집기: 바뀐 test 문항에서 새 답을 고른 기본 구성원(R3·35B·397B·Gemma) 수가 T6 답을 고른 수보다
    적으면 '다수 → 소수'. 이런 변경이 많으면 결합 방식이 소수 의견을 과하게 키운 것이다.
  - H408 오답 분류(v2 정답 기준, 기본 구성원 4개의 단독 답): 0개 맞음 = 전원 오답(어렵거나 라벨 문제),
    1개 맞음 = 소수만 맞음, 2개 이상 맞음 = 앙상블이 다수(또는 절반)를 놓친 실수.

규칙 4-b: test 문항은 정답을 판단하거나 손으로 고치지 않는다. 이 페이지의 test 부분은 '어떤 구조의 문항이 바뀌는가'를 보는 용도다.
"""
import csv, html, io, contextlib, math, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import explore_ens_20260924 as E  # noqa: E402
    import eval_type_branch as TB  # noqa: E402  (E.M/E.TEST['35Bfull'])
    import eval_r1v3 as R1  # noqa: E402  (E.TEST['35Br1v3'])
    import eval_ft_shift_tta as SH  # noqa: E402  (R3·Gemma 회전)
from ensemble_submit import load_any  # noqa: E402
import glob  # noqa: E402

pr, L, T = E.pr, 'abcd', E.T
BASE = ['R3', '35B', '397B', 'Gemma']
QD = 'shared_predictions/runpod_q38_20260924/'
E.M['Q38'] = E.load(*glob.glob(QD + '*H408*predictions.jsonl'))
E.TEST['Q38'] = load_any(glob.glob(QD + '*testTEST*predictions.jsonl')[0])
PUB_N = 3357
# 기준 제출(기본 T6). REF=T14 처럼 바꾸면 모든 차이·카드가 그 제출 기준이 된다. 산출 파일 이름에 _vs{REF} 가 붙는다
REF = os.environ.get('REF', 'T6')
SUF = '' if REF == 'T6' else f'_vs{REF}'


def soft4(D, i):
    return np.exp(np.mean([np.log(np.maximum(pr(D, n, i), 1e-300)) for n in BASE], 0))


def t8a(D, i):
    if T.get(i) == 'COMPOSITE':
        return (pr(D, 'R3', i) + E.fam(D, i, 1, .5, .5) + pr(D, '397B', i) + pr(D, 'Gemma', i)) / 4
    return E.t6(D, i)


def t11(D, i):
    return (pr(D, 'R3', i) + pr(D, '35B', i) + 1.5 * pr(D, '397B', i) + 1.5 * pr(D, 'Gemma', i)) / 5


def t15(D, i):
    return (sum(pr(D, n, i) for n in BASE) + 0.5 * pr(D, 'Q38', i)) / 4.5


def t16(D, i):  # 사용자 배합(2026-09-24 밤): T14 50% + 397B·Gemma 1/6 + 35B·R3 1/12
    return (0.5 * SUBS_T14(D, i) + pr(D, '397B', i) / 6 + pr(D, 'Gemma', i) / 6
            + pr(D, '35B', i) / 12 + pr(D, 'R3', i) / 12)


SUBS_T14 = SH.make(('Gemma',), lambda D, i: True)

# 이름: (파일, Public, H408 결합 함수 또는 None, 한 줄 설명)
SUBS = {
    'T1': ('T1_5soft_gemma_test.csv', 0.97170, None, '5개 로그확률 평균(27B LoRA full·35B·397B·9B LoRA·Gemma)'),
    'T2': ('T2_5hard_gemma_test.csv', 0.97140, None, 'T1 과 같은 5개, 다수결'),
    'T3': ('T3_5soft_R3_test.csv', 0.97319, None, 'T1 에서 27B LoRA(full) -> R3(27B LoRA tiles)'),
    'T4': ('T4_4soft_no9b_test.csv', 0.97348, soft4, 'T3 에서 9B LoRA 뺌 — R3·35B·397B·Gemma 로그확률 평균'),
    'T5': ('T5_T4_shift35b_test.csv', 0.97348, soft4, 'T4 + 35B 보기 회전(test 저마진 336). H408 은 T4 와 같음'),
    'T6': ('T6_4pmean_test.csv', 0.97438, E.t6, 'T4 구성원, 확률 산술평균(pmean) — 기준'),
    'T7': ('T7_5pmean_test.csv', 0.97408, E.t7, 'T6 + 35B r2v3 5번째'),
    'T8A': ('T8A_comp35fam_test.csv', 0.97438, t8a, 'COMPOSITE 만 35B 계열(35B·r2v3·fs16) 평균'),
    'T9': ('T9_t6t7mix025_test.csv', 0.97378, E.CANDS['T6·T7 혼합 0.25:0.75'], '0.25·T6 + 0.75·T7'),
    'T10': ('T10_w1515_test.csv', 0.97408, E.grid[(1.5, 1.5, 1, 1)], '가중치 R3 1.5 / 35B계열 1.5 / 397B 1 / Gemma 1'),
    'T11': ('T11_w397g15_test.csv', 0.97378, t11, '가중치 397B·Gemma 1.5'),
    'T12': ('T12_t6_posfull_test.csv', 0.97438, TB.t6b(TB.RULES['POS']), '위치 키워드 문항만 35B 전체 이미지'),
    'T13': ('T13_t6_r1v3swap_test.csv', 0.97140, R1.CANDS['35B -> r1v3 교체'], '35B 를 r1v3(구조화 판독)로 교체'),
    'T14': ('T14_t6_gemmarot_test.csv', 0.97378, SUBS_T14, 'Gemma 보기 회전 4개 평균'),
    'T15': ('T15_t6_q38half_test.csv', 0.97467, t15, 'T6 + Qwen3.8-27B 제로샷 가중 0.5 (트랙 A)'),
    'T16': ('T16_t14half_w_test.csv', 0.97378, t16, '0.5·T14 + 397B·Gemma 1/6 + 35B·R3 1/12'),
}

rows = {}
for p in ['runs/h408_pack/dev.csv', 'ssafy-16-2-ai/test.csv']:
    for r in csv.DictReader(open(p, encoding='utf-8-sig')):
        rows[r['id']] = r
TP = {k: {r['id']: r['answer'] for r in csv.DictReader(open('submissions/' + f, encoding='utf-8-sig'))} for k, (f, *_ ) in SUBS.items()}
ids, g = E.ids, E.g
H = {k: {i: L[int(np.argmax(fn(E.M, i)))] for i in ids} for k, (_, _, fn, _) in SUBS.items() if fn}
# 제출 파일이 결합 함수로 재현되는지(T5 는 test 336건만 다른 방식이라 제외)
REPRO = {k: sum(L[int(np.argmax(fn(E.TEST, i)))] != TP[k][i] for i in E.tids)
         for k, (_, _, fn, _) in SUBS.items() if fn and k != 'T5'}
pick = lambda D, n, i: L[int(np.argmax(pr(D, n, i)))]  # noqa: E731
t6h, t6t = H[REF], TP[REF]
t6pub = SUBS[REF][1]

md = ['# T1~T15 제출 종합 비교 (2026-09-24 밤)', '',
      f'`tools/report_all_submissions.py`. 기준은 {REF}. H408 은 v2(401문항, 사용자 정답 검토 반영). '
      'Public 1문항 = 0.00030, 잡음 폭 = sqrt(K/2).', '',
      '## 1. 제출별 요약 — 과적합·잡음 판정', '',
      '| 제출 | 무엇을 바꿨나 | Public | Public 차이(문항) | test 변경 K | 잡음 폭 | 판정 | H408 v2 | H408 차이 | 다수→소수 뒤집기 | 파일 재현 |',
      '|---|---|---|---|---|---|---|---|---|---|---|']
flip_stats = {}
for k, (f, pub, fn, desc) in SUBS.items():
    ch = [i for i in E.tids if TP[k][i] != t6t[i]]
    K = len(ch)
    dq = round((pub - t6pub) * PUB_N)
    nz = math.sqrt(K / 2) if K else 0
    if K == 0:
        verdict = '-'
    elif abs(dq) <= nz:
        verdict = '잡음'
    elif abs(dq) <= 2 * nz:
        verdict = '약한 신호(' + ('+' if dq > 0 else '-') + ')'
    else:
        verdict = '**유의(' + ('+' if dq > 0 else '-') + ')**'
    down = sum(sum(pick(E.TEST, n, i) == TP[k][i] for n in BASE) < sum(pick(E.TEST, n, i) == t6t[i] for n in BASE) for i in ch)
    flip_stats[k] = (K, down)
    h = sum(H[k][i] == g[i] for i in ids) if k in H else None
    dh = f'{h - sum(t6h[i] == g[i] for i in ids):+d}' if h is not None else '-'
    md.append(f'| {k} | {desc} | {pub:.5f} | {dq:+d} | {K} | {nz:.1f} | {verdict} | {h if h is not None else "-"} | '
              f'{dh} | {down}/{K} | {REPRO.get(k, "-")} |')
md += ['',
       '- **파일 재현**: 결합 함수로 test 답을 다시 계산했을 때 제출 파일과 다른 문항 수(0 이어야 이 표의 H408 이 그 제출의 값이다).',
       '- T1~T3 은 H408 v2 에 없는 구성원(27B LoRA full·9B LoRA)을 써서 H408 을 비워 두었다.', '']

# H408 v2 와 Public 의 방향이 맞았는지
md += ['## 2. H408 v2 와 Public 이 같은 방향이었나', '',
       '| 제출 | H408 차이 | Public 차이 | 같은 방향? |', '|---|---|---|---|']
for k, (f, pub, fn, desc) in SUBS.items():
    if k not in H or k == REF:
        continue
    dh = sum(H[k][i] == g[i] for i in ids) - sum(t6h[i] == g[i] for i in ids)
    dq = round((pub - t6pub) * PUB_N)
    same = '같음' if (dh > 0 and dq > 0) or (dh < 0 and dq < 0) or (dh == 0 and dq == 0) else ('H408 동점' if dh == 0 else '**반대**')
    md.append(f'| {k} | {dh:+d} | {dq:+d} | {same} |')
md.append('')

# H408 오답 분류
md += ['## 3. H408 v2 오답 분류 — 어려운 문항인가, 앙상블 실수인가', '',
       '| 제출 | 오답 | 전원 오답(0/4) | 소수만 맞음(1/4) | 절반 맞음(2/4) | 다수 맞음(3~4/4) |', '|---|---|---|---|---|---|']
ncorrect = {i: sum(pick(E.M, n, i) == g[i] for n in BASE) for i in ids}
for k in H:
    w = [i for i in ids if H[k][i] != g[i]]
    c = collections.Counter(min(ncorrect[i], 3) for i in w)
    md.append(f'| {k} | {len(w)} | {c.get(0, 0)} | {c.get(1, 0)} | {c.get(2, 0)} | {c.get(3, 0)} |')
wr6 = [i for i in ids if t6h[i] != g[i]]
md += ['', f'- {REF} 오답 {len(wr6)}개 중 **다수(3~4/4)가 맞았는데 틀린 문항**: '
       + (', '.join(i for i in wr6 if ncorrect[i] >= 3) or '없음'),
       f'- {REF} 오답 중 절반(2/4)이 맞은 문항: ' + (', '.join(i for i in wr6 if ncorrect[i] == 2) or '없음'), '']

# test 변경 유형
types = [t for t, _ in collections.Counter(T.get(i, '?') for i in E.tids).most_common(6)]
md += [f'## 4. test 에서 {REF} 대비 바뀐 문항의 유형', '',
       '| 제출 | K | ' + ' | '.join(types) + ' |', '|---|---|' + '---|' * len(types)]
for k in SUBS:
    ch = [i for i in E.tids if TP[k][i] != t6t[i]]
    c = collections.Counter(T.get(i, '?') for i in ch)
    md.append(f'| {k} | {len(ch)} | ' + ' | '.join(str(c.get(t, 0)) for t in types) + ' |')
open(f'reports/all_submissions_20260924{SUF}.md', 'w', encoding='utf-8-sig', newline='\n').write('\n'.join(md) + '\n')

# ---------- HTML ----------
esc = html.escape
EXTRA = [('Q38', 'Q38'), ('35Br2v3', 'r2v3'), ('35Br1v3', 'r1v3'), ('35Bfull', '35B full')]


def member_table(D, i, gold=None):
    tb = "<table><tr><th>구성원</th><th>답</th>" + ''.join(f'<th>{x}</th>' for x in L) + '</tr>'
    for n, lab in [(n, n) for n in BASE] + EXTRA:
        if n in D and i in D[n]:
            p = pr(D, n, i)
            a = L[int(p.argmax())]
            cls = ('ok' if a == gold else 'no') if gold else ''
            tb += f"<tr><td>{lab}</td><td class='{cls}'>{a}</td>" + ''.join(f'<td>{v:.2f}</td>' for v in p) + '</tr>'
    return tb + '</table>'


def card(i, head, answers, gold=None, D=None):
    r = rows[i]
    ch = ''.join(f"<div>{'✅ ' if gold and x == gold else ''}<b>{x}.</b> {esc(r[x])}</div>" for x in L)
    return (f"<div class='card'><div><img loading='lazy' src='../ssafy-16-2-ai/{esc(r['path'])}' alt='{esc(i)}'></div>"
            f"<div><div class='mut'>{esc(i)} · 유형 {esc(T.get(i, '?'))} · {head}</div><p><b>{esc(r['question'])}</b></p>{ch}"
            f"<p>{answers}</p>{member_table(D, i, gold)}</div></div>")


out = ["<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
       "<title>제출 종합 비교</title><style>:root{--bg:#fff;--fg:#1d1d1f;--mut:#6e6e73;--card:#f5f5f7;--ok:#1b7f3b;--no:#b3261e;--hi:#8a5a00}"
       "@media (prefers-color-scheme:dark){:root{--bg:#161618;--fg:#f2f2f2;--mut:#a1a1a6;--card:#232326;--ok:#6fd08c;--no:#ff8a80;--hi:#ffcc66}}"
       "body{background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,'Malgun Gothic',sans-serif;margin:0 auto;padding:16px;max-width:1150px}"
       ".card{background:var(--card);border-radius:10px;padding:14px;margin:14px 0;display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:14px}"
       "@media (max-width:760px){.card{grid-template-columns:1fr}}img{width:100%;height:auto;border-radius:6px}"
       ".ok{color:var(--ok);font-weight:600}.no{color:var(--no);font-weight:600}.hi{color:var(--hi);font-weight:700}.mut{color:var(--mut)}"
       "table{border-collapse:collapse;font-size:13px;width:100%;margin-top:6px}td,th{padding:2px 6px;border-bottom:1px solid #8883;text-align:left}"
       "nav a{margin-right:12px}.warn{border-left:4px solid var(--hi);padding:6px 10px;background:var(--card)}</style></head><body>",
       f"<h1>T1~T15 제출 종합 비교 — 기준 {REF}</h1><nav><a href='#sum'>요약</a><a href='#test'>test 변경 문항</a><a href='#h408'>H408 v2 오답</a></nav>",
       "<h2 id='sum'>요약</h2><table><tr><th>제출</th><th>무엇을 바꿨나</th><th>Public</th><th>차이(문항)</th><th>K</th><th>판정</th><th>H408 v2</th><th>다수→소수</th></tr>"]
for line in md[6:6 + len(SUBS)]:
    c = [x.strip() for x in line.strip('|').split('|')]
    out.append('<tr>' + ''.join(f'<td>{esc(c[j]).replace("**", "")}</td>' for j in (0, 1, 2, 3, 4, 6, 7, 9)) + '</tr>')
out.append('</table>')

# (1) test 변경 문항: T6 와 한 제출이라도 다른 문항 (T1~T3 은 구성원이 달라 제외)
TS = [k for k in SUBS if REF != 'T6' or k not in ('T1', 'T2', 'T3', 'T4', 'T5')]
chg = collections.defaultdict(list)
for k in TS:
    for i in E.tids:
        if TP[k][i] != t6t[i]:
            chg[i].append(k)
order = sorted(chg, key=lambda i: (-len(chg[i]), i))
out.append(f"<h2 id='test'>test 에서 {REF} 와 답이 다른 문항 ({len(order)}개, {TS[0]}~{TS[-1]})</h2>"
           "<p class='warn'>규칙 4-b: test 문항은 정답을 판단하거나 손으로 고치지 않는다. 여기서는 <b>어떤 구조의 문항이 "
           "바뀌는지, 새 답이 구성원 다수와 같은지</b>만 본다. 노란색은 기준 제출과 다른 답.</p>")
for i in order:
    t6a = t6t[i]
    votes = collections.Counter(pick(E.TEST, n, i) for n in BASE)
    ans = ' · '.join(f"<span class='{'hi' if TP[k][i] != t6a else 'mut'}'>{k}:{TP[k][i]}</span>" for k in TS)
    head = f"{REF} 답 {t6a} (기본 구성원 표 {dict(votes)}) · 다른 제출 {len(chg[i])}개 ({', '.join(chg[i])}) · T6 마진 {E.marg(E.t6(E.TEST, i)):.2f}"
    out.append(card(i, head, ans, None, E.TEST))

# (2) H408 v2 오답
allw = sorted({i for k in H for i in ids if H[k][i] != g[i]}, key=lambda i: (ncorrect[i], i))
out.append(f"<h2 id='h408'>H408 v2 에서 한 제출이라도 틀린 문항 ({len(allw)}개)</h2>"
           "<p class='mut'>정답이 있는 dev 문항. 기본 구성원 4개 중 맞힌 수가 적은 순(0 = 전원 오답)으로 정렬.</p>")
for i in allw:
    ans = ' · '.join(f"<span class='{'ok' if H[k][i] == g[i] else 'no'}'>{k}:{H[k][i]}</span>" for k in H)
    kind = {0: '전원 오답', 1: '소수만 맞음', 2: '절반 맞음'}.get(ncorrect[i], '다수가 맞았는데 앙상블 오답' if all(H[k][i] != g[i] for k in H) else '다수 맞음')
    out.append(card(i, f"정답 {g[i]} · 기본 구성원 {ncorrect[i]}/4 맞음 · {kind}", ans, g[i], E.M))
out.append('</body></html>')
open(f'reports/all_submissions_20260924{SUF}.html', 'w', encoding='utf-8', newline='\n').write('\n'.join(out))
print('\n'.join(md))
