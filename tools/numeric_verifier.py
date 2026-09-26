# -*- coding: utf-8 -*-
"""숫자·가격 규칙 검증기 — 판독문에 있는 숫자와 보기의 숫자를 맞춰 본다. GPU 미사용. (2026-09-24 2번 항목)

Codex 제안: 판독문은 '01' 을 읽고도 최종 선택이 0·1·8 로 흩어진다 → 숫자를 정규화해 보기와 맞추자.
규칙
  - 보기 4개가 모두 숫자를 하나 이상 포함할 때만 적용한다(가격·수량·시간·층 등).
  - 숫자 정규화: 쉼표 제거, 앞자리 0 제거(단 '0' 자체는 유지), 소수점 유지.
  - 35B·397B 두 판독문에서 뽑은 숫자 집합과 각 보기의 숫자 집합을 비교한다.
  - 보기 중 **정확히 하나만** 숫자가 전부 판독문에 들어 있으면, 그 보기의 확률에 (1+beta) 를 곱한다.
평가: H408 v2(판독문은 스크리닝 35B base + 397B H331/rerun179), test 변경 수.
"""
import csv, io, contextlib, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import explore_ens_20260924 as E  # noqa: E402

P = 'shared_predictions/'
NUM = re.compile(r'\d[\d,]*(?:\.\d+)?')


def nums(s):
    out = set()
    for m in NUM.findall(s or ''):
        m = m.replace(',', '')
        if '.' in m:
            a, b = m.split('.', 1); a = a.lstrip('0') or '0'; out.add(a + '.' + b)
        else:
            out.add(m.lstrip('0') or '0')
    return out


rows = {}
for p in ['runs/h408_pack/dev.csv', 'ssafy-16-2-ai/test.csv']:
    for r in csv.DictReader(open(p, encoding='utf-8-sig')):
        rows[r['id']] = r
TR = {}
for n, d in (('35B', E.M['35B']), ('397B', E.M['397B']), ('35B', E.TEST['35B']), ('397B', E.TEST['397B'])):
    for i, r in d.items():
        TR.setdefault(i, []).append(r.get('transcript') or '')


def rule(i):
    """(적용 여부, 뒷받침되는 보기 index 또는 None)"""
    r = rows.get(i)
    if r is None:
        return False, None
    on = [nums(r[x]) for x in 'abcd']
    if not all(on):
        return False, None
    tn = set().union(*[nums(t) for t in TR.get(i, [])])
    hit = [k for k in range(4) if on[k] <= tn]
    return True, (hit[0] if len(hit) == 1 else None)


def make(beta):
    def fn(D, i):
        p = E.t6(D, i)
        ok, k = rule(i)
        if ok and k is not None:
            p = p.copy(); p[k] *= (1 + beta)
        return p
    return fn


if __name__ == '__main__':
    app = [i for i in E.ids if rule(i)[0]]
    sup = [i for i in app if rule(i)[1] is not None]
    print(f'H408 v2: 숫자 보기 문항 {len(app)} / 판독문이 한 보기만 뒷받침 {len(sup)} / 그 보기가 정답 {sum("abcd"[rule(i)[1]] == E.g[i] for i in sup)}')
    tapp = [i for i in E.tids if rule(i)[0]]
    print(f'test: 숫자 보기 문항 {len(tapp)} / 한 보기만 뒷받침 {sum(rule(i)[1] is not None for i in tapp)}')
    print('| beta | H408 v2 | T6 대비 test 변경 |'); print('|---|---|---|')
    for b in [0, 0.1, 0.3, 1.0, 3.0]:
        h, ch, _ = E.evaluate(make(b)); print(f'| {b} | {h} | {ch} |')
