# -*- coding: utf-8 -*-
"""5번 유형별 입력 분기 — 35B 를 전체 이미지(view full)로 돌린 결과를 타일(기존)과 비교한다. GPU 미사용.

분기 규칙은 H408 을 보기 전에 고정했다(Discussion 742915: 타일은 위치·층·방향형에서 손해).
  POS : 질문에 위치 키워드(KW)가 있으면 full, 아니면 tiles
  OBJ : auto_type 이 OBJECT 면 full
  POS|OBJ : 둘 중 하나면 full
  ALL : 전부 full (참고)
각 규칙을 35B 단독과 T6(35B 자리만 바꿈) 안에서 H408 v2 로 채점하고, test 에서 T6 대비 바뀌는 수를 센다.
test 의 full 예측은 POS|OBJ 3,369건(testsub)만 있다 — ALL 은 test 에 못 쓴다.
"""
import csv, io, contextlib, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import explore_ens_20260924 as E  # noqa: E402
from ensemble_submit import load_any  # noqa: E402

P = 'shared_predictions/runpod_e_20260924/'
KW = ['층', '위치', '방향', '지하', '왼쪽', '오른쪽', '위쪽', '아래', '옆에', '가운데', '중앙', '맨 ', '번째', '출구', '플랫폼',
      '상단', '하단', '좌측', '우측']
Q = {}
for p in ['runs/h408_pack/dev.csv', 'ssafy-16-2-ai/test.csv', 'ssafy-16-2-ai/train.csv']:
    for r in csv.DictReader(open(p, encoding='utf-8-sig')):
        Q[r['id']] = r['question']
FULL = load_any(P + 'Qwen3.6-35B-A3B_read_devSCR_full_vllm_predictions.jsonl')
TFULL_P = P + 'Qwen3.6-35B-A3B_read_testTESTSUB_full_vllm_predictions.jsonl'
TFULL = load_any(TFULL_P) if os.path.exists(TFULL_P) else {}
SCR_BASE = load_any('shared_predictions/runpod_screen_20260924/base_predictions.jsonl')
E.M['35Bfull'] = FULL
E.TEST['35Bfull'] = TFULL

pos = lambda i: any(k in Q[i] for k in KW)  # noqa: E731
obj = lambda i: E.T.get(i) == 'OBJECT'  # noqa: E731
RULES = {'POS': pos, 'OBJ': obj, 'POS|OBJ': lambda i: pos(i) or obj(i), 'ALL': lambda i: True}


def p35(D, i, rule):
    return E.pr(D, '35Bfull', i) if rule(i) and i in D['35Bfull'] else E.pr(D, '35B', i)


def single(rule):
    return lambda D, i: p35(D, i, rule)


def t6b(rule):
    return lambda D, i: (E.pr(D, 'R3', i) + p35(D, i, rule) + E.pr(D, '397B', i) + E.pr(D, 'Gemma', i)) / 4


if __name__ == '__main__':
    gold = {r['id']: r['answer'] for r in csv.DictReader(open('ssafy-16-2-ai/train.csv', encoding='utf-8-sig'))}
    tr = [i for i in FULL if i.startswith('train')]
    acc = lambda d: sum(d[i]['pred'] == gold[i] for i in tr)  # noqa: E731
    print(f'train1000 (35B 단독): tiles {acc(SCR_BASE)} / full {acc(FULL)}  (n={len(tr)})')
    for name, rule in RULES.items():
        m = [i for i in tr if rule(i)]
        print(f'  {name:8s} n={len(m):4d}  tiles {sum(SCR_BASE[i]["pred"] == gold[i] for i in m)} / full {sum(FULL[i]["pred"] == gold[i] for i in m)}')
    print(f'\nH408 v2 ({len(E.ids)}문항)  test full 예측 {len(TFULL)}건')
    print('| 규칙 | 해당 H408 | 35B 단독 H408 | T6 안 H408 | T6 대비 test 변경 |'); print('|---|---|---|---|---|')
    h35 = sum('abcd'[int(np.argmax(E.pr(E.M, '35B', i)))] == E.g[i] for i in E.ids)
    print(f'| 없음(tiles) | 0 | {h35} | {E.evaluate(E.t6)[0]} | 0 |')
    for name, rule in RULES.items():
        n = sum(rule(i) for i in E.ids)
        hs = sum('abcd'[int(np.argmax(single(rule)(E.M, i)))] == E.g[i] for i in E.ids)
        if name == 'ALL' or not TFULL:
            ht = sum('abcd'[int(np.argmax(t6b(rule)(E.M, i)))] == E.g[i] for i in E.ids); ch = '-'
        else:
            ht, ch, _ = E.evaluate(t6b(rule))
        print(f'| {name} | {n} | {hs} | {ht} | {ch} |')
