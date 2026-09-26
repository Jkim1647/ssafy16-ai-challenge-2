# -*- coding: utf-8 -*-
"""H229 + 검수수정 179 = H408에서 모델·앙상블 재채점. GPU 미사용.

왜
  H229(229건)에서는 상위 4개 모델 차이가 전부 잡음이었다(1문항 = 0.44%p).
  검수로 문항이 수정된 179건을 같은 조건으로 다시 추론해 합치면 408건이 되고
  1문항의 무게가 0.25%p로 줄어, 앙상블 vs 397B 단독을 구별할 수 있는지 다시 본다.

예측 출처 (H229 부분 ∪ 179 부분)
  35B v1/v2 : runpod_35b_read2(EVAL998, --read2 both) ∪ runpod_35b_rerun179(RERUN179)
  397B      : nebius_397b(devH331) ∪ nebius_397b_rerun179(RERUN179)   ← 179쪽은 실행 후 생김
  앙상블    : 위 두 모델 a~d 로그확률을 각각 log-softmax 후 0.5:0.5 평균(제출과 같은 규칙)
  9B CE/KD  : H229 부분만 있으므로 H229에서만 보고한다
  397B 예측이 아직 없으면 H229 구간만 계산하고 그렇게 표시한다.

usage: python tools/score_h408.py [--out reports/h408_score_20260923.md]
"""
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
P = 'shared_predictions/'
L = 'abcd'


def load(path):
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            if line.strip():
                o = json.loads(line)
                out[o['id']] = o
    return out


def norm(lp):
    m = max(lp)
    z = math.log(sum(math.exp(x - m) for x in lp)) + m
    return [x - z for x in lp]


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p, den = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - h) / den, (c + h) / den)


def paired_bootstrap(a, b, B=10000, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(B, len(d)))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return d.mean(), lo, hi


def main():
    out_md = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else 'reports/h408_score_20260923.md'
    gold229 = json.load(open('runs/dev_validation_v1/hard_eval_gold_v5_H.json', encoding='utf-8-sig'))
    gold179 = json.load(open('runs/rerun_dev179/gold.json', encoding='utf-8')) if os.path.exists('runs/rerun_dev179/gold.json') else {}
    assert not (set(gold229) & set(gold179)), '두 평가셋이 겹친다'
    gold = {**gold229, **gold179}

    e998 = load(P + 'runpod_35b_read2/Qwen3.6-35B-A3B_read_devEVAL998_tiles2x2x2_r2both_vllm_predictions.jsonl')
    e179 = load(P + 'runpod_35b_rerun179/Qwen3.6-35B-A3B_read_devRERUN179_tiles2x2x2_r2both_vllm_predictions.jsonl')
    m35 = {**{i: r for i, r in e998.items()}, **e179}
    f397 = load(P + 'nebius_397b/Qwen3.5-397B-A17B-FP8_read_devH331_tiles2x2x2_vllm_predictions.jsonl')
    r397 = load(P + 'nebius_397b_rerun179/Qwen3.5-397B-A17B-FP8_read_devRERUN179_tiles2x2x2_vllm_predictions.jsonl')
    m397 = {**f397, **r397}
    ce = load(P + 'runpod_9b_kd/ce/hard_predictions.jsonl')
    kd = load(P + 'runpod_9b_kd/kd/hard_predictions.jsonl')

    have397_179 = set(gold179) <= set(m397)
    ids408 = [i for i in gold if i in m35 and i in m397] if have397_179 else []
    ids229 = [i for i in gold229 if i in m35 and i in m397]
    scopes = {'H229': ids229}
    if have397_179:
        scopes['H408'] = sorted(ids408)
        scopes['dev179(신규)'] = sorted(gold179)

    def preds(name, ids):
        if name == '35B v1':
            return [m35[i]['pred'] for i in ids]
        if name == '35B v2':
            return [m35[i].get('v2', m35[i])['pred'] for i in ids]
        if name == '397B':
            return [m397[i]['pred'] for i in ids]
        if name == '앙상블(35Bv1+397B)':
            r = []
            for i in ids:
                s = [0.5 * a + 0.5 * b for a, b in zip(norm(m35[i]['logprobs']), norm(m397[i]['logprobs']))]
                r.append(L[s.index(max(s))])
            return r
        if name == '9B CE':
            return [ce[i]['pred'] if i in ce else None for i in ids]
        if name == '9B KD':
            return [kd[i]['pred'] if i in kd else None for i in ids]
        raise KeyError(name)

    models = ['앙상블(35Bv1+397B)', '397B', '35B v1', '35B v2', '9B CE', '9B KD']
    md = ['# H408 재채점 — H229 + 검수수정 179 (2026-09-23)', '',
          '`tools/score_h408.py`. GPU 미사용. 앙상블은 제출과 같은 규칙(a~d log-softmax 0.5:0.5).', '']
    if not have397_179:
        md += ['> ⚠️ **397B의 179건 예측이 아직 없다.** H229 구간만 계산했다. '
               '`shared_predictions/nebius_397b_rerun179/` 가 생기면 다시 돌리면 H408이 채워진다.', '']

    results = {}
    for scope, ids in scopes.items():
        md += [f'## {scope} (n={len(ids)})', '', '| 모델 | 정답 | 정확도 | 95% CI |', '|---|---|---|---|']
        for name in models:
            p = preds(name, ids)
            usable = [(x, gold[i]) for x, i in zip(p, ids) if x is not None]
            if not usable:
                continue
            k = sum(x == g for x, g in usable)
            lo, hi = wilson(k, len(usable))
            results[(scope, name)] = [int(x == gold[i]) for x, i in zip(p, ids) if x is not None]
            note = '' if len(usable) == len(ids) else f' (평가 가능 {len(usable)})'
            md += [f'| {name} | {k}{note} | {k / len(usable):.4f} | {lo:.3f}–{hi:.3f} |']
        md += ['']
        # 쌍 비교: 상위 후보만
        pairs = [('앙상블(35Bv1+397B)', '397B'), ('앙상블(35Bv1+397B)', '35B v1'), ('397B', '35B v1'), ('35B v1', '35B v2')]
        md += ['| 쌍 비교 | 평균차 | 95% CI | 판정 |', '|---|---|---|---|']
        for a, b in pairs:
            if (scope, a) not in results or (scope, b) not in results:
                continue
            va, vb = results[(scope, a)], results[(scope, b)]
            if len(va) != len(vb):
                continue
            d, lo, hi = paired_bootstrap(va, vb)
            verdict = '구별됨' if (lo > 0 or hi < 0) else '구별 안 됨(잡음)'
            md += [f'| {a} − {b} | {d:+.4f} | {lo:+.4f}–{hi:+.4f} | {verdict} |']
        md += ['']
        # 불일치 내역
        if scope != 'dev179(신규)':
            pa, pb = preds('앙상블(35Bv1+397B)', ids), preds('397B', ids)
            dis = [(i, x, y) for i, x, y in zip(ids, pa, pb) if x != y]
            aw = sum(1 for i, x, y in dis if x == gold[i])
            md += [f'- 앙상블 vs 397B 예측이 갈린 문항 **{len(dis)}**건 중 앙상블 정답 {aw} / 397B 정답 {len(dis) - aw}', '']

    # 유형별 (H408 가능하면 H408, 아니면 H229)
    qt = pd.read_csv('data_meta/question_types.csv', encoding='utf-8-sig', dtype=str)
    typ = dict(zip(qt.row_id, qt.auto_type))
    scope = 'H408' if have397_179 else 'H229'
    ids = scopes[scope]
    md += [f'## 유형별 정확도 ({scope}, auto_type = 규칙 기반 분류)', '',
           '| 유형 | n | 앙상블 | 397B | 35B v1 |', '|---|---|---|---|---|']
    by = defaultdict(list)
    for i in ids:
        by[typ.get(i, '?')].append(i)
    pa = dict(zip(ids, preds('앙상블(35Bv1+397B)', ids)))
    pb = dict(zip(ids, preds('397B', ids)))
    pc = dict(zip(ids, preds('35B v1', ids)))
    for t, tids in sorted(by.items(), key=lambda kv: -len(kv[1])):
        f = lambda d: sum(d[i] == gold[i] for i in tids) / len(tids)  # noqa: E731
        md += [f'| {t} | {len(tids)} | {f(pa):.3f} | {f(pb):.3f} | {f(pc):.3f} |']
    md += ['', f'- 정답 분포(gold, {scope}): ' + str(dict(Counter(gold[i] for i in ids)))]

    open(out_md, 'w', encoding='utf-8', newline='\n').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
