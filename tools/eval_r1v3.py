# -*- coding: utf-8 -*-
"""8번 r1v3 test — 35B 판독 형식 v3(위치|라벨|값) 변형을 T6 에 넣어 H408 v2 와 test 변경 수로 본다. GPU 미사용.

후보: 35B 자리 교체 / 5번째 구성원 추가 / 35B 계열 평균(35B + w·r1v3). H408 의 r1v3 예측은 스크리닝(09-24) 것이다.
"""
import io, contextlib, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import explore_ens_20260924 as E  # noqa: E402
from ensemble_submit import load_any  # noqa: E402

E.TEST['35Br1v3'] = load_any('shared_predictions/runpod_e_20260924/'
                             'Qwen3.6-35B-A3B_read_testTEST_r1v3_tiles2x2x2_r1v3_vllm_predictions.jsonl')
pr = E.pr
CANDS = {
    'T6': E.t6,
    '35B -> r1v3 교체': lambda D, i: (pr(D, 'R3', i) + pr(D, '35Br1v3', i) + pr(D, '397B', i) + pr(D, 'Gemma', i)) / 4,
    '+ r1v3 5번째': lambda D, i: (4 * E.t6(D, i) + pr(D, '35Br1v3', i)) / 5,
    'T7 + r1v3 6번째': lambda D, i: (5 * E.t7(D, i) + pr(D, '35Br1v3', i)) / 6,
}
for w in (0.5, 1.0):
    CANDS[f'35B계열 (35B + {w}·r1v3)'] = (lambda D, i, w=w: (pr(D, 'R3', i) + (pr(D, '35B', i) + w * pr(D, '35Br1v3', i)) / (1 + w)
                                                           + pr(D, '397B', i) + pr(D, 'Gemma', i)) / 4)

if __name__ == '__main__':
    s = lambda n: sum('abcd'[int(np.argmax(pr(E.M, n, i)))] == E.g[i] for i in E.ids)  # noqa: E731
    print(f'단독 H408 v2: 35B {s("35B")} / r1v3 {s("35Br1v3")} / r2v3 {s("35Br2v3")}')
    same = sum(int(np.argmax(pr(E.TEST, '35B', i))) == int(np.argmax(pr(E.TEST, '35Br1v3', i))) for i in E.tids)
    print(f'test 35B vs r1v3 같은 답 {same}/6714\n')
    print('| 후보 | H408 v2 | T6 대비 test 변경 |'); print('|---|---|---|')
    for k, fn in CANDS.items():
        h, ch, _ = E.evaluate(fn)
        print(f'| {k} | {h} | {ch} |')
