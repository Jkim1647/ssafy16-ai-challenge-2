# -*- coding: utf-8 -*-
"""gold_v4 생성 + 기존 예측 재채점 (GPU 미사용).

입력: review_gold_final_20260923/review_gold_final_20260923/
  - audit_all_350.csv : 350건 사람 검수 결과(status / 최종 문항 / 원본 대비 변경 필드 / 평가 경로)
  - gold.csv          : include 287건의 최종 정답

핵심 제약
  검수 과정에서 질문·선택지를 고친 문항은 모델이 본 입력과 달라졌다. 그런 문항에
  기존 예측 글자를 그대로 대보면 틀린 채점이 된다. 따라서 오프라인 재채점은
  `evaluation_route_vs_original == rescore_candidate_requires_matching_fingerprint`
  (= 원본 대비 question/a/b/c/d 무변경)인 문항으로만 한정하고, 그 사실을
  ssafy-16-2-ai/dev.csv 원문과 직접 대조해 한 번 더 확인한다.

출력:
  runs/dev_validation_v1/hard_eval_gold_v4.json        전체 include 정답(287)
  runs/dev_validation_v1/gold_v4_rescorable.json       오프라인 재채점 가능분만
  reports/gold_v4_rescore_20260923.md                  비교표·약점 분석
"""
import json, os, sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
D = 'review_gold_final_20260923/review_gold_final_20260923/'
P = 'shared_predictions/'
OUT = 'runs/dev_validation_v1/'
L = 'abcd'
RESCORE_ROUTE = 'rescore_candidate_requires_matching_fingerprint'


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


def main():
    A = pd.read_csv(D + 'audit_all_350.csv', encoding='utf-8-sig', dtype=str).fillna('')
    G = pd.read_csv(D + 'gold.csv', encoding='utf-8-sig', dtype=str).fillna('')

    # ---------- gold_v4 ----------
    gold_v4 = dict(zip(G['id'], G['answer']))
    assert len(gold_v4) == len(G), 'gold.csv에 id 중복'
    assert set(gold_v4.values()) <= set(L), 'a~d 외 정답 존재'

    # ---------- 오프라인 재채점 가능 집합 ----------
    resc = A[(A['status'] == 'include') & (A['evaluation_route_vs_original'] == RESCORE_ROUTE)]
    # dev.csv 원문과 직접 대조 (multistep은 대조 원본이 없어 제외)
    dev = pd.read_csv('ssafy-16-2-ai/dev.csv', encoding='utf-8-sig', dtype=str).fillna('')
    devmap = {r['id']: r for _, r in dev.iterrows()}
    verified, unverified = [], []
    for _, r in resc.iterrows():
        src = devmap.get(r['id'])
        if src is None:
            unverified.append((r['id'], 'dev.csv에 없음(multistep)'))
            continue
        same = all(str(src[c]).strip() == str(r['final_' + c]).strip()
                   for c in ['question', 'a', 'b', 'c', 'd'])
        (verified if same else unverified).append(
            (r['id'], 'ok' if same else '원문과 불일치'))
    vids = [i for i, _ in verified]
    print('[검증] rescore 후보 {}건 중 dev.csv 원문 대조 통과 {}건 / 대조불가·불일치 {}건'
          .format(len(resc), len(vids), len(unverified)))
    for i, why in unverified[:5]:
        print('        - {} : {}'.format(i, why))
    if len(unverified) > 5:
        print('        ... 외 {}건'.format(len(unverified) - 5))

    # ---------- 예측 로드 ----------
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
        '35B v1':          lambda i: (m35.get(i) or {}).get('pred'),
        '35B v2':          lambda i: ((m35.get(i) or {}).get('v2') or {}).get('pred'),
        '397B FP8':        lambda i: (m397.get(i) or {}).get('pred'),
        '9B CE':           lambda i: (ce.get(i) or {}).get('pred'),
        '9B KD':           lambda i: (kd.get(i) or {}).get('pred'),
        '앙상블 35B+397B':  ens,
    }

    scorable = sorted([i for i in vids if i in m35 and i in m397])
    print('[집합] gold_v4 include {} / 원문 검증 통과 {} / 예측까지 있는 재채점 가능 {}'
          .format(len(gold_v4), len(vids), len(scorable)))

    gold_v3 = json.load(open(OUT + 'hard_eval_gold_v3.json', encoding='utf-8'))

    # ---------- 저장 ----------
    os.makedirs(OUT, exist_ok=True)
    json.dump(gold_v4, open(OUT + 'hard_eval_gold_v4.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, sort_keys=True)
    json.dump({i: gold_v4[i] for i in scorable},
              open(OUT + 'gold_v4_rescorable.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, sort_keys=True)
    print('[저장] hard_eval_gold_v4.json ({}) / gold_v4_rescorable.json ({})'
          .format(len(gold_v4), len(scorable)))

    # ---------- 라벨 변경 ----------
    both = [i for i in scorable if i in gold_v3]
    changed = [i for i in both if gold_v3[i] != gold_v4[i]]
    print('[라벨] 재채점 집합 중 gold_v3에도 있던 {}건, 그중 정답이 바뀐 것 {}건'
          .format(len(both), len(changed)))

    # ---------- 재채점 ----------
    rows = []
    for name, fn in MODELS.items():
        k3 = sum(1 for i in both if fn(i) == gold_v3.get(i))
        k4 = sum(1 for i in scorable if fn(i) == gold_v4[i])
        lo, hi = wilson(k4, len(scorable))
        rows.append(dict(모델=name,
                         gold_v3=f'{k3}/{len(both)}' if both else '-',
                         gold_v4=f'{k4}/{len(scorable)}',
                         정확도_v4=round(k4 / len(scorable), 4) if scorable else None,
                         CI=f'[{lo:.3f},{hi:.3f}]'))
    T = pd.DataFrame(rows).sort_values('정확도_v4', ascending=False)
    print('\n[재채점] gold_v4 기준')
    print(T.to_string(index=False))

    # ---------- 약점 분석 ----------
    best = T.iloc[0]['모델']
    fn = MODELS[best]
    wrong = [i for i in scorable if fn(i) != gold_v4[i]]
    allwrong = [i for i in scorable
                if all(f(i) != gold_v4[i] for f in MODELS.values())]
    print('\n[약점] 최고 모델 {} 오답 {}건 / 전 모델 오답 {}건'
          .format(best, len(wrong), len(allwrong)))
    return dict(A=A, gold_v4=gold_v4, scorable=scorable, both=both, changed=changed,
                MODELS=MODELS, table=T, wrong=wrong, allwrong=allwrong,
                gold_v3=gold_v3, unverified=unverified, resc=resc)


if __name__ == '__main__':
    main()
