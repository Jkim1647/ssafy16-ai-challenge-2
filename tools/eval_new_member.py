# -*- coding: utf-8 -*-
"""새 구성원(LoRA 등) 하나를 T6·T15 앙상블에 넣어 H408 v2 와 test 변경 수로 본다. GPU 미사용.

사용
  python tools/eval_new_member.py NAME H408.jsonl TEST.jsonl [VAL667.jsonl]
  예) python tools/eval_new_member.py 9Bbase shared_predictions/runpod_h_20260924/9b_base/hard_predictions.jsonl \
        shared_predictions/runpod_h_20260924/9b_base/test_predictions.jsonl shared_predictions/runpod_h_20260924/9b_base/val_predictions.jsonl

후보: T6 + 새것(가중 0.5·1), T15 + 새것, 기본 구성원 하나를 새것으로 교체. H408 v2 는 사용자 정답 검토 반영본.
val667 은 새 구성원 단독 정확도만 본다(다른 구성원 중 397B·35B 는 val667 이 제로샷이라 앙상블 비교가 공정하지 않다).
"""
import csv, io, contextlib, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import report_all_submissions as R  # noqa: E402  (Q38·T15 준비)
from ensemble_submit import load_any  # noqa: E402

E, pr = R.E, R.E.pr
BASE = ['R3', '35B', '397B', 'Gemma']
name, hp, tp = sys.argv[1], sys.argv[2], sys.argv[3]
E.M[name] = load_any(hp)
E.TEST[name] = load_any(tp)
assert all(i in E.M[name] for i in E.ids), 'H408 누락'
assert all(i in E.TEST[name] for i in E.tids), 'test 누락'
L = 'abcd'
single = sum(L[int(np.argmax(pr(E.M, name, i)))] == E.g[i] for i in E.ids)
print(f'{name} 단독 H408 v2 {single}/{len(E.ids)}')
if len(sys.argv) > 4:
    gold = {r['id']: r['answer'] for r in csv.DictReader(open('ssafy-16-2-ai/train.csv', encoding='utf-8-sig'))}
    v = [json.loads(l) for l in open(sys.argv[4], encoding='utf-8') if l.strip()]
    print(f'{name} 단독 val667 {sum(r["pred"] == gold[r["id"]] for r in v)}/{len(v)}')
t6w = sum(1 for i in E.ids if L[int(np.argmax(E.t6(E.M, i)))] != E.g[i])
fix = sum(1 for i in E.ids if L[int(np.argmax(E.t6(E.M, i)))] != E.g[i] and L[int(np.argmax(pr(E.M, name, i)))] == E.g[i])
print(f'T6 오답 {t6w}개 중 {name} 이 맞힌 문항 {fix}')

C = {'T6': E.t6, 'T15': R.t15}
for w in (0.5, 1.0):
    C[f'T6 + {name} x{w}'] = lambda D, i, w=w: (sum(pr(D, n, i) for n in BASE) + w * pr(D, name, i)) / (4 + w)
    C[f'T15 + {name} x{w}'] = lambda D, i, w=w: (sum(pr(D, n, i) for n in BASE) + 0.5 * pr(D, 'Q38', i) + w * pr(D, name, i)) / (4.5 + w)
for n in BASE:
    C[f'{n} -> {name}'] = lambda D, i, n=n: np.mean([pr(D, m, i) for m in BASE if m != n] + [pr(D, name, i)], 0)
print('\n| 후보 | H408 v2 | T6 대비 test 변경 |'); print('|---|---|---|')
for k, fn in C.items():
    h, ch, _ = E.evaluate(fn)
    print(f'| {k} | {h} | {ch} |')
