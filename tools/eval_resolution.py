# -*- coding: utf-8 -*-
"""6번 해상도 — 35B 스크리닝 1,408(train1000 + H408)에서 t3·mp1536px·mp2304px 를 기존 타일(2배)과 비교한다. GPU 미사용.

T6 안 H408 은 35B 자리를 해당 설정으로 바꿔 본다(test 예측이 없으므로 test 변경 수는 아직 없다).
"""
import csv, glob, io, contextlib, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import explore_ens_20260924 as E  # noqa: E402
from ensemble_submit import load_any  # noqa: E402

P = 'shared_predictions/runpod_e_20260924/'
gold = {r['id']: r['answer'] for r in csv.DictReader(open('ssafy-16-2-ai/train.csv', encoding='utf-8-sig'))}
base = load_any('shared_predictions/runpod_screen_20260924/base_predictions.jsonl')
tr = [i for i in base if i.startswith('train')]
print(f'| 설정 | train1000 | H408 v2 단독 ({len(E.ids)}) | T6 안 H408 |'); print('|---|---|---|---|')
print(f'| 타일 2배(기존) | {sum(base[i]["pred"] == gold[i] for i in tr)} | '
      f'{sum("abcd"[int(np.argmax(E.pr(E.M, "35B", i)))] == E.g[i] for i in E.ids)} | {E.evaluate(E.t6)[0]} |')
for name in ('t3', 'mp1536px', 'mp2304px'):
    fs = glob.glob(f'{P}*SCR_{name}_*predictions.jsonl')
    if not fs:
        continue
    d = load_any(fs[0])
    E.M['35Bx'] = d
    trn = sum(d[i]['pred'] == gold[i] for i in tr if i in d)
    h1 = sum('abcd'[int(np.argmax(E.pr(E.M, '35Bx', i)))] == E.g[i] for i in E.ids)
    h6 = sum('abcd'[int(np.argmax((E.pr(E.M, 'R3', i) + E.pr(E.M, '35Bx', i) + E.pr(E.M, '397B', i) + E.pr(E.M, 'Gemma', i)) / 4))] == E.g[i]
             for i in E.ids)
    print(f'| {name} | {trn} | {h1} | {h6} |')
