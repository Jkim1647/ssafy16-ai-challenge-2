# -*- coding: utf-8 -*-
"""7번 R3·Gemma 보기 회전 TTA — 회전 0(기존) + 1·2·3 의 확률을 원래 a~d 순서로 평균해 T6 안에 넣는다. GPU 미사용.

test 는 T6 1·2위 확률 차가 작은 600건(lowmargin600_ids.json, 마진 ≤ 0.934)만 회전 예측이 있다.
H408 은 408건 전부 있으므로 두 가지로 본다: 전 문항 적용 / test 와 같은 마진 조건(≤ 0.934)에서만 적용.
"""
import io, contextlib, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import explore_ens_20260924 as E  # noqa: E402
from ensemble_submit import load_any  # noqa: E402
from ensemble_methods import lsm  # noqa: E402

P = os.environ.get('SHIFT_DIR', 'shared_predictions/runpod_f_20260924/')
TH = 0.934
SH = {}
for n, d in (('R3', 'R3_shift'), ('Gemma', 'Gemma_shift')):
    for k in (1, 2, 3):
        for part, D in (('hard', E.M), ('test', E.TEST)):
            f = f'{P}{d}/{part}_predictions_shift{k}.jsonl'
            if os.path.exists(f):
                SH[(n, k, part)] = load_any(f)


def pavg(D, n, i):
    part = 'hard' if D is E.M else 'test'
    ps = [E.pr(D, n, i)] + [np.exp(lsm(SH[(n, k, part)][i]['logprobs']))
                            for k in (1, 2, 3) if (n, k, part) in SH and i in SH[(n, k, part)]]
    return np.mean(ps, 0)


def make(members, gate):
    def fn(D, i):
        use = gate(D, i)
        p = {n: (pavg(D, n, i) if (n in members and use) else E.pr(D, n, i)) for n in ('R3', 'Gemma')}
        return (p['R3'] + E.pr(D, '35B', i) + E.pr(D, '397B', i) + p['Gemma']) / 4
    return fn


if __name__ == '__main__':
    print('회전 예측 파일:', sorted(f'{n}{k}{p}:{len(v)}' for (n, k, p), v in SH.items()))
    for n in ('R3', 'Gemma'):
        if (n, 1, 'hard') not in SH:
            continue
        base = sum('abcd'[int(np.argmax(E.pr(E.M, n, i)))] == E.g[i] for i in E.ids)
        rot = {k: sum('abcd'[int(np.argmax(np.exp(lsm(SH[(n, k, 'hard')][i]['logprobs']))))] == E.g[i] for i in E.ids)
               for k in (1, 2, 3) if (n, k, 'hard') in SH}
        avg = sum('abcd'[int(np.argmax(pavg(E.M, n, i)))] == E.g[i] for i in E.ids)
        flip = sum(len({int(np.argmax(E.pr(E.M, n, i)))} | {int(np.argmax(SH[(n, k, 'hard')][i]['logprobs'])) for k in rot}) > 1
                   for i in E.ids)
        print(f'{n} 단독 H408 v2: 회전0 {base} / 회전별 {rot} / 4회전 평균 {avg} / 회전에 따라 답이 바뀐 문항 {flip}')
    allg = lambda D, i: True  # noqa: E731
    lowg = lambda D, i: E.marg(E.t6(D, i)) <= TH  # noqa: E731
    print(f'\n| 후보 | H408 v2 ({len(E.ids)}) | T6 대비 test 변경 |'); print('|---|---|---|')
    print(f'| T6 | {E.evaluate(E.t6)[0]} | 0 |')
    for mem in (('R3', 'Gemma'), ('R3',), ('Gemma',)):
        for gname, gate in (('전 문항', allg), (f'마진≤{TH}', lowg)):
            h, ch, _ = E.evaluate(make(mem, gate))
            print(f'| {"+".join(mem)} 회전평균 / {gname} | {h} | {ch} |')
