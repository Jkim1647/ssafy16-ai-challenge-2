# -*- coding: utf-8 -*-
"""구성원별 확률 보정(온도) 후 평균 — H408 v2 로 채점. GPU 미사용. (2026-09-24 1번 항목)

왜: T6 는 네 모델의 확률을 그대로 산술평균한다. 모델마다 확신을 표현하는 정도가 다르면 과신하는 모델이
평균을 끌고 간다. 온도 T 로 logit/T 를 맞춰 각 모델의 확신을 '실제 정답률'에 맞춘 뒤 평균한다.

과적합 방지: 온도는 H408 이 아니라 **각 모델의 자기 홀드아웃**에서 맞춘다.
  35B  : 스크리닝 train1000 (test 구성원과 같은 read+tiles, read2 v1)
  397B : train 6,714 중 R3·Gemma 학습에 쓰이지 않은 val667 을 제외한 나머지(제로샷이라 어디든 공정)
  R3·Gemma : val667 (학습에 쓰지 않은 홀드아웃)
"""
import csv, io, contextlib, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import explore_ens_20260924 as E  # noqa: E402
from ensemble_methods import lsm  # noqa: E402

P = 'shared_predictions/'
L = 'abcd'
gold = {r['id']: r['answer'] for r in csv.DictReader(open('ssafy-16-2-ai/train.csv', encoding='utf-8-sig'))}
val667 = set(json.load(open('data_meta/splits/val667_ids.json', encoding='utf-8'))['val_groups'])


def load(p):
    return {json.loads(l)['id']: json.loads(l) for l in open(p, encoding='utf-8') if l.strip()}


HOLD = {
    '35B': {k: v for k, v in load(P + 'runpod_screen_20260924/base_predictions.jsonl').items() if k.startswith('train')},
    '397B': {k: v for k, v in load(P + 'nebius_397b_kd/Qwen3.5-397B-A17B-FP8_read_val6714KDtrain_tiles2x2x2_vllm_predictions.jsonl').items()},
    'R3': load(P + 'runpod_27b_R3/R3_27b_tiles/val_predictions.jsonl'),
    'Gemma': load(P + 'runpod_gemma4_31b/gemma4_31b_lora/val_predictions.jsonl'),
}


def nll(rows, T):
    s = 0.0
    for r in rows:
        lp = lsm(np.array(r['logprobs'], float) / T)
        s -= lp[L.index(gold[r['id']])]
    return s / len(rows)


temps = {}
for n, d in HOLD.items():
    rows = [r for i, r in d.items() if i in gold and gold[i] in L]
    grid = np.exp(np.linspace(np.log(0.2), np.log(20), 121))
    best = min(grid, key=lambda T: nll(rows, T))
    acc = np.mean([r['pred'] == gold[r['id']] for r in rows])
    conf = np.mean([np.exp(lsm(np.array(r['logprobs'], float))).max() for r in rows])
    temps[n] = float(best)
    print(f'{n:6s} 홀드아웃 {len(rows):5d}건  정답률 {acc:.4f}  평균확신 {conf:.4f}  -> 온도 T={best:.3f}')

B = ['R3', '35B', '397B', 'Gemma']


def cal(D, n, i, T):
    return np.exp(lsm(np.array(D[n][i]['logprobs'], float) / T))


def make(mode):
    if mode == 'T6':
        return E.t6
    if mode == 'cal_pmean':
        return lambda D, i: np.mean([cal(D, n, i, temps[n]) for n in B], 0)
    if mode == 'cal_geo':
        return lambda D, i: np.mean([np.log(np.maximum(cal(D, n, i, temps[n]), 1e-300)) for n in B], 0)
    if mode == 'cal_T_common':   # 모든 모델 같은 온도(평균) — 보정 자체보다 '평탄화' 효과인지 확인
        Tm = float(np.mean(list(temps.values())))
        return lambda D, i: np.mean([cal(D, n, i, Tm) for n in B], 0)


print(f'\nH408 기준 v2 ({len(E.ids)}문항)')
print('| 결합 | H408 v2 | T6 대비 test 변경 |'); print('|---|---|---|')
res = {}
for m in ['T6', 'cal_pmean', 'cal_geo', 'cal_T_common']:
    h, ch, pred = E.evaluate(make(m))
    res[m] = (h, ch)
    print(f'| {m} | {h} | {ch} |')
json.dump({'temps': temps, 'results': res}, open('reports/calibration_20260924.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
