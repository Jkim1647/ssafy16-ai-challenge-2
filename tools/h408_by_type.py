# -*- coding: utf-8 -*-
"""T4(soft)·T6(pmean)·T7(5개 pmean)을 H408 에서 유형별로 비교한다. 정답이 있는 셋이라 제출 판단 근거로 쓸 수 있다."""
import csv, json, sys, os
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h408_ensemble import M, g, ids, g229  # noqa: E402  (구성원 H408 예측 로드)
from ensemble_methods import combine, lsm  # noqa: E402
T = {r['row_id']: r['auto_type'] for r in csv.DictReader(open('data_meta/question_types.csv', encoding='utf-8-sig'))}
def pred(names, m):
    return {i: 'abcd'[int(np.argmax(combine(m, np.stack([lsm(M[n][i]['logprobs']) for n in names]))))] for i in ids}
S = {'T4 soft': pred(['R3', '35B', '397B', 'Gemma'], 'soft'),
     'T6 pmean': pred(['R3', '35B', '397B', 'Gemma'], 'pmean'),
     'T7 5pmean': pred(['R3', '35B', '397B', 'Gemma', '35Br2v3'], 'pmean')}
types = sorted(Counter(T.get(i, '?') for i in ids).items(), key=lambda x: -x[1])
print('| 유형 | n | ' + ' | '.join(S) + ' |'); print('|---|---|' + '---|' * len(S))
for t, n in types:
    print(f'| {t} | {n} | ' + ' | '.join(str(sum(S[k][i] == g[i] for i in ids if T.get(i, "?") == t)) for k in S) + ' |')
print(f'| **합계** | {len(ids)} | ' + ' | '.join(f'**{sum(S[k][i]==g[i] for i in ids)}**' for k in S) + ' |')
print()
for k in ['T6 pmean', 'T7 5pmean']:
    fx = [i for i in ids if S['T4 soft'][i] != g[i] and S[k][i] == g[i]]
    br = [i for i in ids if S['T4 soft'][i] == g[i] and S[k][i] != g[i]]
    print(f'- {k} vs T4: 고침 {len(fx)} {dict(Counter(T.get(i,"?") for i in fx))} / 망가뜨림 {len(br)} {dict(Counter(T.get(i,"?") for i in br))}')
