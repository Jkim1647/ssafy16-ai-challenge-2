# -*- coding: utf-8 -*-
"""평가셋 신뢰도 진단 — GPU 미사용. 기존 예측 jsonl만 재분석한다.

산출:
  - 모델별 정확도 + Wilson 95% CI (val667 / H331 / H184)
  - paired bootstrap (모델 쌍 차이가 잡음인지)
  - 오답 겹침 행렬, 3모델 오라클 상한, 전원 오답 ID
  - 정답 위치 편향
  - 사람 검수 패킷 회수분 실태 (gold_v3 반영 여부 포함)

사용: baseline/Scripts/python.exe tools/eval_reliability.py
결과 해석: reports/eval_reliability_20260923.md
"""
import json, math, glob, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
P = 'shared_predictions/'
L = 'abcd'


def load(path):
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
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / den, (c + h) / den)


def paired_bootstrap(a, b, B=10000, seed=0):
    """a, b: 0/1/nan 정답 벡터. 둘 다 예측이 있는 항목만 쓴다."""
    m = (~np.isnan(a)) & (~np.isnan(b))
    x, y = a[m], b[m]
    if len(x) == 0:
        return None
    d = x - y
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(B, len(d)))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return dict(n=len(d), diff=d.mean(), lo=lo, hi=hi, p=min(p, 1.0),
                a_only=int(((x == 1) & (y == 0)).sum()),
                b_only=int(((x == 0) & (y == 1)).sum()))


def main():
    m35 = load(P + 'runpod_35b_read2/Qwen3.6-35B-A3B_read_devEVAL998_tiles2x2x2_r2both_vllm_predictions.jsonl')
    v397 = load(P + 'nebius_397b/Qwen3.5-397B-A17B-FP8_read_val667_tiles2x2x2_vllm_predictions.jsonl')
    h397 = load(P + 'nebius_397b/Qwen3.5-397B-A17B-FP8_read_devH331_tiles2x2x2_vllm_predictions.jsonl')
    m397 = {**v397, **h397}
    ce = {**load(P + 'runpod_9b_kd/ce/val_predictions.jsonl'),
          **load(P + 'runpod_9b_kd/ce/hard_predictions.jsonl')}
    kd = {**load(P + 'runpod_9b_kd/kd/val_predictions.jsonl'),
          **load(P + 'runpod_9b_kd/kd/hard_predictions.jsonl')}
    c35 = {}
    for f in ['colab_ai2_35b_out/Qwen3.6-35B-A3B_read_val667_tiles2x2x2_predictions.jsonl',
              'colab_ai2_35b_out/Qwen3.6-35B-A3B_read_devH184_tiles2x2x2_predictions.jsonl']:
        if os.path.exists(P + f):
            c35.update(load(P + f))

    goldv3 = json.load(open('runs/dev_validation_v1/hard_eval_gold_v3.json', encoding='utf-8'))
    goldv2 = json.load(open('runs/dev_validation_v1/hard_eval_gold_v2.json', encoding='utf-8'))
    H331, H184, val667 = set(goldv3), set(goldv2), set(v397)
    GOLD = {i: o['gold'] for i, o in v397.items()}
    GOLD.update(goldv3)

    # 앙상블: 35B + 397B 로그확률 0.5/0.5 (현재 최고 제출과 동일 규칙)
    def ens_pred(i):
        a, b = m35.get(i), m397.get(i)
        if not a or not b:
            return None
        s = [(x + y) / 2 for x, y in zip(a['logprobs'], b['logprobs'])]
        return L[int(np.argmax(s))]

    MODELS = {
        '35B v1 (vLLM)':   lambda i: (m35.get(i) or {}).get('pred'),
        '35B v2 (vLLM)':   lambda i: ((m35.get(i) or {}).get('v2') or {}).get('pred'),
        '35B (Colab,제출)': lambda i: (c35.get(i) or {}).get('pred'),
        '397B FP8':        lambda i: (m397.get(i) or {}).get('pred'),
        '9B CE':           lambda i: (ce.get(i) or {}).get('pred'),
        '9B KD':           lambda i: (kd.get(i) or {}).get('pred'),
        '앙상블 35B+397B':  ens_pred,
    }

    def vec(fn, ids):
        out = []
        for i in ids:
            p = fn(i)
            out.append(np.nan if p is None else (1.0 if p == GOLD.get(i) else 0.0))
        return np.array(out)

    PAIRS = [('397B FP8', '35B v1 (vLLM)'), ('9B CE', '9B KD'),
             ('35B v2 (vLLM)', '35B v1 (vLLM)'),
             ('앙상블 35B+397B', '35B v1 (vLLM)'), ('앙상블 35B+397B', '397B FP8')]
    CORE = ['35B v1 (vLLM)', '35B v2 (vLLM)', '397B FP8', '9B CE', '9B KD', '앙상블 35B+397B']

    for setname, idset in [('val667', val667), ('H331', H331), ('H184', H184 & H331)]:
        ids = sorted(idset)
        print('\n' + '=' * 76)
        print('### {}  (n={})'.format(setname, len(ids)))
        print('=' * 76)
        V = {name: vec(fn, ids) for name, fn in MODELS.items()}

        rows = []
        for name, v in V.items():
            n = int((~np.isnan(v)).sum())
            k = int(np.nansum(v))
            if n == 0:
                rows.append((name, '-', '-', '-', len(ids)))
                continue
            lo, hi = wilson(k, n)
            rows.append((name, '{}/{}'.format(k, n), '{:.4f}'.format(k / n),
                         '[{:.3f},{:.3f}]'.format(lo, hi), len(ids) - n))
        print(pd.DataFrame(rows, columns=['모델', '맞음/평가', '정확도',
                                          'Wilson 95% CI', '예측없음']).to_string(index=False))

        print('\n-- paired bootstrap (10,000회) --')
        for a, b in PAIRS:
            r = paired_bootstrap(V[a], V[b])
            if not r:
                continue
            sig = '유의' if (r['lo'] > 0 or r['hi'] < 0) else '잡음'
            print('{:>16s} - {:<16s} n={:4d} d={:+6.2f}%p CI[{:+6.2f},{:+6.2f}] '
                  'p={:.3f}  (A만 {:3d} / B만 {:3d}) -> {}'.format(
                      a, b, r['n'], r['diff'] * 100, r['lo'] * 100, r['hi'] * 100,
                      r['p'], r['a_only'], r['b_only'], sig))

        print('\n-- 오답 겹침 (대각선 = 총 오답 수) --')
        W = {k: {ids[j] for j in range(len(ids)) if V[k][j] == 0.0} for k in CORE}
        T = pd.DataFrame(index=CORE, columns=CORE, dtype=object)
        for a in CORE:
            for b in CORE:
                T.loc[a, b] = len(W[a]) if a == b else len(W[a] & W[b])
        print(T.to_string())

        anyok = set()
        for k in ['35B v1 (vLLM)', '397B FP8', '9B CE']:
            anyok |= {ids[j] for j in range(len(ids)) if V[k][j] == 1.0}
        allwrong = sorted(set(ids) - anyok)
        best_n, best_name = max((int(np.nansum(V[k])), k) for k in CORE)
        print('\n-- 3모델 오라클 상한: {}/{} = {:.4f}'.format(len(anyok), len(ids), len(anyok) / len(ids)))
        print('-- 현재 최고: {} {}  -> 남은 여지 {}문항'.format(best_name, best_n, len(anyok) - best_n))
        print('-- 전원 오답: {}건 ({:.1f}%)'.format(len(allwrong), len(allwrong) / len(ids) * 100))

        gd = pd.Series([GOLD[i] for i in ids]).value_counts().reindex(list(L)).fillna(0).astype(int)
        pv = [p for p in (ens_pred(i) for i in ids) if p]
        pdist = pd.Series(pv).value_counts().reindex(list(L)).fillna(0).astype(int)
        print('-- 정답 분포 {} / 앙상블 예측 {}'.format(dict(gd), dict(pdist)))
        if setname == 'H331':
            print('-- 전원 오답 ID: {}'.format(allwrong))

    # 사람 검수 패킷 실태
    print('\n' + '=' * 76)
    print('### 사람 검수 패킷 회수분')
    print('=' * 76)
    files = sorted(glob.glob('label_packets/hard_ext_20260922/returned/*.csv'))
    if not files:
        print('회수 파일 없음')
        return
    R = pd.concat([pd.read_csv(f, encoding='utf-8-sig', dtype=str).fillna('') for f in files],
                  ignore_index=True)
    print('회수 {}팩 / {}건 (유니크 {})'.format(len(files), len(R), R['id'].nunique()))
    print(R['review_status'].value_counts().to_string())
    R['has_ans'] = R['human_answer'].str.strip().isin(list(L))
    R['in_goldv3'] = R['id'].isin(goldv3)
    print('\n-- gold_v3 반영 여부 x review_status --')
    print(pd.crosstab(R['in_goldv3'], R['review_status']).to_string())
    sub = R[R['in_goldv3']]
    same = int((sub['id'].map(goldv3) == sub['human_answer'].str.strip()).sum())
    print('\ngold_v3에 있는 검수건 {}건 중 human_answer와 라벨 일치: {}'.format(len(sub), same))
    if len(sub) and same == len(sub):
        print('=> gold_v3는 이미 이 검수 결과로 만들어졌다. 재채점해도 점수 변화 없음.')
    print('미반영 신규 라벨(gold_v3 밖 + human_answer 있음): {}건'.format(
        int(((~R['in_goldv3']) & R['has_ans']).sum())))


if __name__ == '__main__':
    main()
