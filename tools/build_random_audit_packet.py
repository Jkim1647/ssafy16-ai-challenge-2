# -*- coding: utf-8 -*-
"""전체 결함률 추정용 '무작위' 검수 패킷 생성.

왜 필요한가
  지금까지 사람이 본 dev 743건은 전부 모델이 틀렸거나 불일치한 문항만 골라 뽑은 표본이다.
  그 표본에서 80%가 결함이었지만, 그것으로 데이터 전체의 결함률을 추정할 수는 없다.
  자동 구조 플래그도 판별력이 없었다(플래그 76% vs 비플래그 83%).
  train은 6,714건 중 사람이 본 것이 15건뿐이고, 그 위에서 학습하고 있다.

  따라서 **아무 조건 없이 균등 무작위**로 뽑아야 한다. 이미 검수된 문항도 빼지 않는다
  (빼면 표본이 다시 편향된다). 대신 `기존검수` 열로 표시해 재사용할 수 있게 한다.

  n=100이면 결함률 추정 정밀도는 약 ±10%p다. "5%인가 40%인가"를 가르기에는 충분하다.

사용
  baseline/Scripts/python.exe tools/build_random_audit_packet.py
  baseline/Scripts/python.exe tools/build_random_audit_packet.py --per 100 --seed 20260923

출력: label_packets/random_base_20260923/p1(train) p2(dev) + zip
"""
import argparse, glob, json, os, random, sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from build_label_packets import build, COLS  # 패킷 생성 기계 재사용

DATA = 'ssafy-16-2-ai/'
BATCH = 'random_base_20260923'


def human_reviewed():
    """사람이 실제로 본 id (모델 기반 티어링은 제외)."""
    H = set()
    p = 'review_gold_final_20260923/review_gold_final_20260923/audit_all_350.csv'
    if os.path.exists(p):
        H |= set(pd.read_csv(p, encoding='utf-8-sig', dtype=str)['id'])
    for f in glob.glob('label_packets/hard_ext_20260922/returned/*.csv'):
        H |= set(pd.read_csv(f, encoding='utf-8-sig', dtype=str)['id'])
    p = 'teammate_handoff/merged/dev_label_provisional.csv'
    if os.path.exists(p):
        d = pd.read_csv(p, encoding='utf-8-sig', dtype=str).fillna('')
        H |= set(d[d['tier'].isin(['T1_human_approved', 'X_human_excluded'])]['id'])
    return H


def qtypes():
    p = 'data_meta/question_types.csv'
    if not os.path.exists(p):
        return {}
    q = pd.read_csv(p, encoding='utf-8-sig', dtype=str).fillna('')
    return dict(zip(q['row_id'], q['auto_type']))


def sample(split, per, seed):
    d = pd.read_csv(DATA + f'{split}.csv', encoding='utf-8-sig', dtype=str).fillna('')
    qt = qtypes()
    H = human_reviewed()
    ids = sorted(d['id'])
    rng = random.Random(seed)
    rng.shuffle(ids)
    part = d.set_index('id').loc[ids[:per]].reset_index()
    part['auto_type'] = part['id'].map(lambda i: qt.get(i, 'UNKNOWN') or 'UNKNOWN')
    part['기존검수'] = part['id'].isin(H).map({True: 'Y', False: 'N'})
    return part


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per', type=int, default=100)
    ap.add_argument('--seed', type=int, default=20260923)
    a = ap.parse_args()

    parts, meta = [], []
    for n, split in enumerate(['train', 'dev']):
        p = sample(split, a.per, a.seed + n)
        meta.append((split, p))
        parts.append(p[COLS].copy())

    root = build(BATCH, parts)

    # 정답 대조용 키는 패킷 밖에 따로 둔다 (작업자가 보면 안 된다)
    keys = {}
    for split, p in meta:
        src = pd.read_csv(DATA + f'{split}.csv', encoding='utf-8-sig', dtype=str).fillna('')
        src = src.set_index('id')
        for i in p['id']:
            r = src.loc[i]
            keys[i] = {'split': split,
                       'dataset_answer': r['answer'] if 'answer' in src.columns else '',
                       'annotators': {c: r[c] for c in src.columns if c.startswith('answer')},
                       '기존검수': p.loc[p['id'] == i, '기존검수'].iloc[0]}
    os.makedirs('runs/random_base_20260923', exist_ok=True)
    json.dump(keys, open('runs/random_base_20260923/answer_key.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)

    print()
    for (split, p), part in zip(meta, parts):
        seen = (p['기존검수'] == 'Y').sum()
        print(f'{split}: {len(part)}건 무작위, 이미 사람이 본 것 {seen}건 '
              f'(재사용 가능) | 유형 {part["auto_type"].value_counts().to_dict()}')
    print('\n정답 키(작업자 비공개): runs/random_base_20260923/answer_key.json')
    print('출력:', root)


if __name__ == '__main__':
    main()
