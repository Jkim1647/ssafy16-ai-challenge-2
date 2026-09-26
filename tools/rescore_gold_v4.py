# -*- coding: utf-8 -*-
"""gold_v4로 보정한 H331(=H229)에서 기존 예측 재채점 + 약점 분석. GPU 미사용.

보정 규칙
  - 검수에서 question/a~d가 바뀐 문항(rerun_required)은 모델이 본 입력과 달라졌으므로 제외
  - 판정 불가(exclude)로 분류된 문항도 제외
  - 원문 무변경 확정분(rescore_candidate)은 gold_v4 정답으로 교체
  - 검수 대상이 아니었던 문항은 gold_v3 정답 유지 (미검증이라는 한계는 리포트에 명시)
"""
import json, os
from collections import Counter

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
D = 'review_gold_final_20260923/review_gold_final_20260923/'
P = 'shared_predictions/'
OUT = 'runs/dev_validation_v1/'
L = 'abcd'
RESC = 'rescore_candidate_requires_matching_fingerprint'


def load_preds(path):
    d = {}
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                o = json.loads(line)
                d[o['id']] = o
    return d


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - h) / den, (c + h) / den)


def paired_bootstrap(a, b, B=10000, seed=0):
    d = a - b
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(B, len(d)))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return d.mean(), lo, hi


def main():
    A = pd.read_csv(D + 'audit_all_350.csv', encoding='utf-8-sig', dtype=str).fillna('')
    gold_v4 = json.load(open(OUT + 'hard_eval_gold_v4.json', encoding='utf-8'))
    gold_v3 = json.load(open(OUT + 'hard_eval_gold_v3.json', encoding='utf-8'))
    H184 = set(json.load(open(OUT + 'hard_eval_gold_v2.json', encoding='utf-8')))
    H331 = set(gold_v3)
    route = dict(zip(A['id'], A['evaluation_route_vs_original']))
    atype = dict(zip(A['id'], A['auto_type'])) if 'auto_type' in A.columns else {}

    keep_orig = sorted(H331 - set(A['id']))
    keep_resc = sorted(i for i in H331 if route.get(i) == RESC)
    drop_rerun = sorted(i for i in H331 if route.get(i) == 'rerun_required')
    drop_excl = sorted(i for i in H331 if route.get(i) == 'exclude')

    GOLD = {i: gold_v3[i] for i in keep_orig}
    GOLD.update({i: gold_v4[i] for i in keep_resc})
    ids = sorted(GOLD)
    json.dump(GOLD, open(OUT + 'hard_eval_gold_v4_H229.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, sort_keys=True)

    m35 = load_preds(P + 'runpod_35b_read2/Qwen3.6-35B-A3B_read_devEVAL998_tiles2x2x2_r2both_vllm_predictions.jsonl')
    m397 = load_preds(P + 'nebius_397b/Qwen3.5-397B-A17B-FP8_read_devH331_tiles2x2x2_vllm_predictions.jsonl')
    ce = load_preds(P + 'runpod_9b_kd/ce/hard_predictions.jsonl')
    kd = load_preds(P + 'runpod_9b_kd/kd/hard_predictions.jsonl')

    def ens(i):
        a, b = m35.get(i), m397.get(i)
        if not a or not b:
            return None
        s = [(x + y) / 2 for x, y in zip(a['logprobs'], b['logprobs'])]
        return L[int(np.argmax(s))]

    MODELS = {
        '35B v1': lambda i: (m35.get(i) or {}).get('pred'),
        '35B v2': lambda i: ((m35.get(i) or {}).get('v2') or {}).get('pred'),
        '397B FP8': lambda i: (m397.get(i) or {}).get('pred'),
        '9B CE': lambda i: (ce.get(i) or {}).get('pred'),
        '9B KD': lambda i: (kd.get(i) or {}).get('pred'),
        '앙상블 35B+397B': ens,
    }

    print('=' * 74)
    print('보정 평가셋 H229 = 미검수 유지 {} + 검수확정 {} = {}'.format(
        len(keep_orig), len(keep_resc), len(ids)))
    print('  제외: 문항·선택지 변경 {} / 판정불가 {}'.format(len(drop_rerun), len(drop_excl)))
    print('=' * 74)

    V = {n: np.array([1.0 if f(i) == GOLD[i] else 0.0 for i in ids]) for n, f in MODELS.items()}
    V3 = {n: np.array([1.0 if f(i) == gold_v3[i] else 0.0 for i in ids]) for n, f in MODELS.items()}

    rows = []
    for n in MODELS:
        k4, k3 = int(V[n].sum()), int(V3[n].sum())
        lo, hi = wilson(k4, len(ids))
        rows.append(dict(모델=n, gold_v3=f'{k3}/{len(ids)}', gold_v4=f'{k4}/{len(ids)}',
                         정확도=round(k4 / len(ids), 4), 변화=k4 - k3,
                         CI=f'[{lo:.3f},{hi:.3f}]'))
    T = pd.DataFrame(rows).sort_values('정확도', ascending=False)
    print('\n[재채점] 같은 229문항, 정답만 gold_v3 -> gold_v4')
    print(T.to_string(index=False))

    print('\n[모델 쌍 차이] paired bootstrap 10,000회')
    best = T.iloc[0]['모델']
    for other in [n for n in MODELS if n != best]:
        d, lo, hi = paired_bootstrap(V[best], V[other])
        sig = '유의' if (lo > 0 or hi < 0) else '잡음'
        print('  {:>14s} - {:<14s} d={:+6.2f}%p CI[{:+6.2f},{:+6.2f}] -> {}'.format(
            best, other, d * 100, lo * 100, hi * 100, sig))

    # 오라클 / 전원 오답
    anyok = {i for i in ids if any(f(i) == GOLD[i] for n, f in MODELS.items() if n != '앙상블 35B+397B')}
    allwrong = sorted(set(ids) - anyok)
    print('\n[상한] 5모델 오라클 {}/{} = {:.4f} | 현재 최고 {} {}'.format(
        len(anyok), len(ids), len(anyok) / len(ids), best, int(V[best].sum())))
    print('[전원 오답] {}건'.format(len(allwrong)))

    # 유형별 약점 (auto_type은 검수분에만 있으므로 question_types.csv 병행)
    qt = {}
    qtp = 'data_meta/question_types.csv'
    if os.path.exists(qtp):
        Q = pd.read_csv(qtp, encoding='utf-8-sig', dtype=str).fillna('')
        qt = dict(zip(Q['row_id'], Q['auto_type']))
        print('\n[유형 출처] {} 의 auto_type 컬럼'.format(qtp))
    for i in ids:
        qt.setdefault(i, atype.get(i, 'UNKNOWN') or 'UNKNOWN')

    print('\n[유형별] 최고 모델 {} 기준'.format(best))
    recs = []
    for t in sorted(set(qt[i] for i in ids)):
        sub = [i for i in ids if qt[i] == t]
        if len(sub) < 5:
            continue
        k = sum(1 for i in sub if MODELS[best](i) == GOLD[i])
        lo, hi = wilson(k, len(sub))
        aw = sum(1 for i in sub if i in allwrong)
        recs.append(dict(유형=t, n=len(sub), 맞음=k, 정확도=round(k / len(sub), 3),
                         CI=f'[{lo:.2f},{hi:.2f}]', 전원오답=aw))
    R = pd.DataFrame(recs).sort_values('정확도')
    print(R.to_string(index=False))

    # 검수가 드러낸 데이터 품질
    print('\n' + '=' * 74)
    print('[데이터 품질] H331 중 사람이 검수한 139건')
    print('=' * 74)
    sub = A[A['id'].isin(H331)]
    print('  문항·선택지를 고쳐야 했음 : {} ({:.0f}%)'.format(
        len(drop_rerun), 100 * len(drop_rerun) / len(sub)))
    print('  판정 불가로 제외          : {} ({:.0f}%)'.format(
        len(drop_excl), 100 * len(drop_excl) / len(sub)))
    print('  원문 그대로 정답 확정     : {} ({:.0f}%)'.format(
        len(keep_resc), 100 * len(keep_resc) / len(sub)))
    ch = Counter()
    for v in sub[sub['evaluation_route_vs_original'] == 'rerun_required']['changed_fields_vs_original']:
        for f in str(v).split(';'):
            if f:
                ch[f] += 1
    print('\n  바뀐 필드 (rerun {}건):'.format(len(drop_rerun)))
    for f, c in ch.most_common():
        print('    {:<10s} {}'.format(f, c))
    print('\n  전체 350건 기준 exclude 사유 상위:')
    ex = A[A['status'] == 'exclude']['reason'].str.slice(0, 40)
    for r, c in Counter(ex).most_common(8):
        print('    {:>2d}  {}'.format(c, r))

    return T, R, allwrong, ids


if __name__ == '__main__':
    main()
