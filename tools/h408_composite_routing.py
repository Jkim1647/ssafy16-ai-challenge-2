# -*- coding: utf-8 -*-
"""Codex 제안(2026-09-24): COMPOSITE 문항에서만 35B 계열을 r2v3/fs16 로 바꾸거나 섞는 라우팅을 H408 로 채점한다.

주의: 앙상블 수준 검증은 H408 뿐이다(COMPOSITE 112문항). R3·Gemma 는 train 으로 학습해 train 검증셋을 쓸 수 없다.
35B 단독 수준의 COMPOSITE 검증(스크리닝 373 / 독립 1,908)은 이미 r2v3 +3/+5, fs16 +3/+6 으로 나와 있다.
"""
import sys, os, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h408_ensemble import M, g, ids, load, S  # noqa: E402
from ensemble_methods import lsm  # noqa: E402
M['35Bfs16'] = load(S + 'fs16_predictions.jsonl', S + 'd179/fs16_predictions.jsonl')
T = {r['row_id']: r['auto_type'] for r in csv.DictReader(open('data_meta/question_types.csv', encoding='utf-8-sig'))}
P = lambda n, i: np.exp(lsm(M[n][i]['logprobs']))
def t6(i): return (P('R3', i) + P('35B', i) + P('397B', i) + P('Gemma', i)) / 4
def marg(p): s = np.sort(p); return s[-1] - s[-2]
def fam(i, w35, wr, wf):
    f = (w35 * P('35B', i) + wr * P('35Br2v3', i) + wf * P('35Bfs16', i)) / (w35 + wr + wf)
    return (P('R3', i) + f + P('397B', i) + P('Gemma', i)) / 4
comp = [i for i in ids if T.get(i) == 'COMPOSITE']
base = {i: int(np.argmax(t6(i))) for i in ids}
def score(fn, name):
    pr = {i: int(np.argmax(fn(i))) for i in ids}
    tot = sum('abcd'[pr[i]] == g[i] for i in ids); c = sum('abcd'[pr[i]] == g[i] for i in comp)
    ch = sum(pr[i] != base[i] for i in ids)
    fx = sum(pr[i] != base[i] and 'abcd'[pr[i]] == g[i] for i in ids); br = sum(pr[i] != base[i] and 'abcd'[base[i]] == g[i] for i in ids)
    print(f'| {name} | {tot} | {c}/{len(comp)} | {ch} | {fx}/{br} |')
print('| 후보 | H408 | COMPOSITE | 변경 | 고침/망가뜨림 |'); print('|---|---|---|---|---|')
score(t6, 'T6')
C = lambda alt: (lambda i: alt(i) if T.get(i) == 'COMPOSITE' else t6(i))
score(C(lambda i: fam(i, 0, 1, 0)), 'C-R2-REPLACE')
score(C(lambda i: fam(i, 1, 1, 0)), 'C-R2-FAMILY')
for th in [0.03, 0.05, 0.10, 0.15]:
    score(lambda i, th=th: fam(i, 0, 1, 0) if T.get(i) == 'COMPOSITE' and marg(t6(i)) < th else t6(i), f'C-R2-M{int(th*100):02d}')
dis = lambda i: np.argmax(P('35B', i)) != np.argmax(P('35Br2v3', i))
for th in [0.10, 1.0]:
    score(lambda i, th=th: fam(i, 0, 1, 0) if T.get(i) == 'COMPOSITE' and dis(i) and marg(t6(i)) < th else t6(i), f'불일치 게이트 교체 m<{th}')
    score(lambda i, th=th: fam(i, 1, 1, 0) if T.get(i) == 'COMPOSITE' and dis(i) and marg(t6(i)) < th else t6(i), f'불일치 게이트 family m<{th}')
for w in [(1, .5, .5), (1, 1, 1), (.5, .5, 1), (0, 1, 1), (0, 0, 1)]:
    score(C(lambda i, w=w: fam(i, *w)), f'family 35B/r2v3/fs16={w}')
score(lambda i: fam(i, 1, 1, 1) if T.get(i) == 'COMPOSITE' and marg(t6(i)) < 0.10 else t6(i), '추천구조: COMP & m<0.10 -> family(1,1,1)')
# 35B vs r2v3 가 갈린 H408 COMPOSITE 문항 표
d = [i for i in comp if dis(i)]
a35 = lambda i: 'abcd'[np.argmax(P('35B', i))] == g[i]; ar = lambda i: 'abcd'[np.argmax(P('35Br2v3', i))] == g[i]
print(f'\nH408 COMPOSITE 중 35B≠r2v3: {len(d)}건 | 35B만 정답 {sum(a35(i) and not ar(i) for i in d)} | r2v3만 정답 {sum(ar(i) and not a35(i) for i in d)} | 둘 다 오답 {sum(not a35(i) and not ar(i) for i in d)}')
