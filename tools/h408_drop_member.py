# -*- coding: utf-8 -*-
"""'틀린 답을 말하는 모델을 빼고 앙상블'을 H408 v2 로 시험한다(2026-09-24 사용자 제안). GPU 미사용.

test 에는 정답이 없어 '이 문항에서 누가 틀렸나'를 알 수 없다. 그래서 정답 없이 적용할 수 있는 규칙만 본다.
  A. 구성원을 통째로 빼기 — 4개 중 3개 / 2개 조합
  B. 문항마다 '다른 모델들과 가장 어긋나는' 구성원을 자동으로 빼기
     B1 이탈자 제거: 나머지 3개 평균과 가장 먼(총변동거리) 1개를 빼고 pmean
     B2 다수 합의: 3개 이상이 같은 답이면 그 답, 아니면 pmean(T6)
     B3 확신 이탈자 제거: 다른 3개가 모두 같은 답인데 혼자 다른 답을 확신(>0.8)하면 그 1개만 빼기
"""
import itertools, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import explore_ens_20260924 as E  # noqa: E402  (H408 v2 구성원 M, test TEST)
pr, L = E.pr, 'abcd'
BASE = ['R3', '35B', '397B', 'Gemma']


def pmean(D, i, names):
    return np.mean([pr(D, n, i) for n in names], 0)


def drop_outlier(D, i):
    P = {n: pr(D, n, i) for n in BASE}
    dist = {n: 0.5 * np.abs(P[n] - np.mean([P[m] for m in BASE if m != n], 0)).sum() for n in BASE}
    out = max(dist, key=dist.get)
    return np.mean([P[m] for m in BASE if m != out], 0)


def majority(D, i):
    picks = [int(np.argmax(pr(D, n, i))) for n in BASE]
    c = np.bincount(picks, minlength=4)
    if c.max() >= 3:
        v = np.zeros(4); v[int(c.argmax())] = 1; return v
    return pmean(D, i, BASE)


def drop_confident_lone(D, i):
    P = {n: pr(D, n, i) for n in BASE}
    picks = {n: int(np.argmax(P[n])) for n in BASE}
    for n in BASE:
        others = [picks[m] for m in BASE if m != n]
        if len(set(others)) == 1 and picks[n] != others[0] and P[n].max() > 0.8:
            return np.mean([P[m] for m in BASE if m != n], 0)
    return np.mean(list(P.values()), 0)


cands = {'T6 (4개 pmean)': lambda D, i: pmean(D, i, BASE)}
for k in (3, 2):
    for sub in itertools.combinations(BASE, k):
        cands[f'{"+".join(sub)}'] = lambda D, i, s=sub: pmean(D, i, list(s))
cands['B1 문항별 이탈자 1개 제거'] = drop_outlier
cands['B2 3개 이상 합의면 그 답'] = majority
cands['B3 혼자 확신하며 다른 답이면 제거'] = drop_confident_lone

print(f'H408 기준 {os.environ.get("H408_GOLD", "v2")} ({len(E.ids)}문항)\n')
print('| 후보 | H408 | T6 대비 test 변경 |'); print('|---|---|---|')
for k, fn in cands.items():
    h, ch, _ = E.evaluate(fn)
    print(f'| {k} | {h} | {ch} |')
