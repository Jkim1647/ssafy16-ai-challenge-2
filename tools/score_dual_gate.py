# -*- coding: utf-8 -*-
"""이중 게이트 채점 — train 홀드아웃 + H408 을 같이 본다. GPU 미사용.

왜 두 개인가
  두 셋은 **서로 다른 것**을 재고, 어느 쪽도 단독으로는 test 를 대변하지 못한다.

  - train 홀드아웃(6,047) : 라벨이 test 와 **같은 방식**으로 만들어졌다(교육생 응답 중
    여러 명이 일치한 문항만 채택). 397B 가 여기서 0.9651 이고 Public 이 0.96693 이라
    난이도·분포가 맞는다. n 이 커서 1문항이 0.017%p — 오늘 H408(0.25%p)에서 전부
    잡음이던 비교가 여기서는 갈릴 수 있다.
  - H408 : 사람이 이미지를 보고 확정한 엄격한 판독 기준. n 은 작지만 라벨 품질이 높고,
    train 홀드아웃이 놓치는 실패 유형(모호·복수정답)을 잡는다.

  Public 과 Private 의 분포가 다를 수 있으므로 한쪽만 보고 채택하지 않는다.

채택 규칙 (기본값)
  train 홀드아웃에서 **유의하게 개선**되고 H408 에서 **유의하게 나빠지지 않으면** 채택.
  둘 다 개선이면 강한 채택. train 에서 잡음이면 보류(H408 단독 개선으로는 채택하지 않는다 —
  n=408 에서 1~3문항 차이는 오늘 내내 잡음이었다).

사용
  python tools/score_dual_gate.py --cand <후보 predictions.jsonl> [--cand ...]
  후보를 주지 않으면 현재 보유 모델(35B/397B/앙상블)만 비교한다.
"""
import argparse, json, math, os
from collections import Counter

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
P = 'shared_predictions/'
L = 'abcd'


def load(path):
    d = {}
    if not os.path.exists(path):
        return d
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                o = json.loads(line)
                d[o['id']] = o
    return d


def lsm(v):
    a = np.clip(np.array(v, float), -1e4, None)
    m = a.max()
    e = np.exp(a - m)
    return np.log(np.maximum(e / e.sum(), 1e-300))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / den, (c + h) / den)


def paired(a, b, B=10000, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(B, len(d)))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return d.mean() * 100, lo * 100, hi * 100, int((d != 0).sum())


def build_sets():
    """(이름, {id: gold}, {모델: preds}) 목록."""
    sets = {}

    # ---- train 홀드아웃 ----
    t397 = load(P + 'nebius_397b_kd/Qwen3.5-397B-A17B-FP8_read_val6714KDtrain_tiles2x2x2_vllm_predictions.jsonl')
    t35 = load(P + 'runpod_35b_train6047/Qwen3.6-35B-A3B_read_val6047TRAIN6047_tiles2x2x2_r2both_vllm_predictions.jsonl')
    if t397:
        gold = {i: o['gold'] for i, o in t397.items()}
        sets['train홀드아웃'] = (gold, {'397B': t397, **({'35B v1': t35} if t35 else {})})

    # ---- H408 ----
    h35 = {**load(P + 'runpod_35b_read2/Qwen3.6-35B-A3B_read_devEVAL998_tiles2x2x2_r2both_vllm_predictions.jsonl'),
           **load(P + 'runpod_35b_rerun179/Qwen3.6-35B-A3B_read_devRERUN179_tiles2x2x2_r2both_vllm_predictions.jsonl')}
    h397 = {**load(P + 'nebius_397b/Qwen3.5-397B-A17B-FP8_read_devH331_tiles2x2x2_vllm_predictions.jsonl'),
            **load(P + 'nebius_397b_rerun179/Qwen3.5-397B-A17B-FP8_read_devRERUN179_tiles2x2x2_vllm_predictions.jsonl')}
    g229 = json.load(open('runs/dev_validation_v1/hard_eval_gold_v5_H.json', encoding='utf-8'))
    g179 = json.load(open('runs/rerun_dev179/gold.json', encoding='utf-8'))
    g = {**g229, **g179}
    sets['H408'] = (g, {'35B v1': h35, '397B': h397})

    # ---- 파인튜닝 모델은 H229 에서만 본다 ----
    # **2026-09-24 정정.** 전날 "파인튜닝 모델은 dev179(질문을 다시 쓴 179건)에서만 0.55 로 무너진다
    # (4/4 재현) -> train 표현 관례를 외운다" 고 적었는데 **틀렸다. 데이터 버그였다.**
    # runs/h408_pack/dev.csv 를 만들 때 dev179 행을 수정본(runs/rerun_dev179/dev.csv)이 아니라
    # 원본 dev.csv 에서 가져왔다. 질문 134건·보기 125건이 수정본과 다른데 gold 는 수정본 기준이었다.
    # 결정적 증거: **제로샷 35B 도 같은 버그 데이터를 주면 dev179 가 0.888 -> 0.520 으로 무너진다.**
    # 파인튜닝 모델(R3 0.553, Gemma 0.570)은 오히려 제로샷보다 약간 낫다.
    #
    # h408_pack/dev.csv 는 고쳤다. 하지만 R3·Gemma·9B 의 dev179 예측은 버그 데이터로 나왔으므로
    # **다시 돌리기 전까지 H229(버그 영향 없음, 제로샷 35B 가 204 vs 205 로 재현)만 쓴다.**
    # 파인튜닝 모델의 H408 산출물(있는 것만). 이들에게 공정한 게이트는 H229 다.
    ft = {}
    for name, path in (('Gemma4 31B LoRA', P + 'runpod_gemma4_31b/gemma_h408_test/hard_predictions.jsonl'),
                       ('27B R3(tiles)', P + 'runpod_27b_R3/R3_h408/hard_predictions.jsonl'),
                       ('9B CE', P + 'runpod_9b_kd/ce/hard_predictions.jsonl'),
                       ('9B KD', P + 'runpod_9b_kd/kd/hard_predictions.jsonl')):
        d = load(path)
        if set(g229) <= set(d):      # H229 를 전부 덮는 것만 넣는다
            ft[name] = d
    sets['H229(파인튜닝용)'] = ({i: v for i, v in g.items() if i in g229},
                             {'35B v1': h35, '397B': h397, **ft})

    # ---- val667 (파인튜닝 모델 전용 게이트) ----
    # 위 'train홀드아웃' 6,047 은 우리 LoRA 들이 **학습에 쓴** 부분이다
    # (train 6,714 = fit 6,047 + val667 667, 교집합 0 확인). 제로샷 모델(397B·35B base)에는
    # 미지의 셋이라 유효하지만, 파인튜닝 모델을 거기서 채점하면 자기 학습 데이터를 재는 것이다.
    # holdout1500 도 공통 게이트가 못 된다 — R3(27B, fit 5,976)가 그 중 833 건을 학습했다.
    # 네 모델 모두에게 미지인 것은 val667 하나뿐이라 파인튜닝 후보는 여기서만 비교한다.
    vids = json.load(open('data_meta/splits/val667_ids.json', encoding='utf-8-sig'))
    vids = set(vids['val_groups'] if isinstance(vids, dict) else vids)
    vmodels = {}
    for name, path in (
            ('27B LoRA full(기존 최고)', P + 'colab_ai2_35b_out/lora/27b_lora_r8/val_predictions.jsonl'),
            ('27B read+tiles', P + 'colab_ai2_35b_out/Qwen3.6-27B_read_val667_tiles2x2x2_lora_predictions.jsonl'),
            ('35B LoRA A(full)', P + 'runpod_35b_lora_A/A_35b_full/val_predictions.jsonl'),
            ('35B LoRA C(tiles)', P + 'runpod_35b_lora_C/C_35b_tiles/val_predictions.jsonl'),
            ('27B R3(tiles+제외)', P + 'runpod_27b_R3/R3_27b_tiles/val_predictions.jsonl'),
            ('Gemma4 31B LoRA(tiles)', P + 'runpod_gemma4_31b/gemma4_31b_lora/val_predictions.jsonl')):
        m = {i: o for i, o in load(path).items() if i in vids and 'gold' in o}
        if len(m) == len(vids):
            vmodels[name] = m
    if vmodels:
        sets['val667'] = ({i: o['gold'] for i, o in next(iter(vmodels.values())).items()}, vmodels)
    return sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cand', action='append', default=[],
                    help='후보 predictions.jsonl. 같은 이름이 두 셋에 다 있으면 양쪽에서 채점된다')
    ap.add_argument('--baseline', default='앙상블', help='비교 기준 모델명')
    a = ap.parse_args()

    for cand in a.cand:
        assert os.path.exists(cand), cand

    md = ['# 이중 게이트 채점 (train 홀드아웃 + H408)', '',
          '`tools/score_dual_gate.py`. GPU 미사용.', '',
          '> 두 셋은 서로 다른 것을 잰다. train 홀드아웃은 **test 와 같은 방식으로 만든 합의 라벨**이라',
          '> 분포가 맞고 n 이 크다. H408 은 **사람이 확정한 엄격한 판독 기준**이라 라벨 품질이 높다.',
          '> Public/Private 분포가 다를 수 있으므로 한쪽만 보고 채택하지 않는다.', '']

    for sname, (gold, models) in build_sets().items():
        # 후보 붙이기
        for c in a.cand:
            preds = load(c)
            if set(preds) & set(gold):
                # 우리 LoRA 산출물은 파일명이 전부 val_predictions.jsonl 이라 상위 폴더까지 이름에 넣는다
                label = '/'.join(c.replace(chr(92), '/').rstrip('/').split('/')[-2:])[:44]
                models[label] = preds

        ids = sorted(i for i in gold if all(i in m for m in models.values()))
        if not ids:
            md += [f'## {sname}', '', '공통 id 없음 — 건너뜀', '']
            continue

        # 앙상블(제출 규칙: 35B v1 + 397B, log-softmax 0.5:0.5)
        vecs = {}
        for name, m in models.items():
            vecs[name] = [int(m[i]['pred'] == gold[i]) for i in ids]
        if '35B v1' in models and '397B' in models:
            s35 = {i: lsm(models['35B v1'][i]['logprobs']) for i in ids}
            s97 = {i: lsm(models['397B'][i]['logprobs']) for i in ids}
            vecs['앙상블'] = [int(L[int(np.argmax(0.5 * s35[i] + 0.5 * s97[i]))] == gold[i]) for i in ids]

        md += [f'## {sname} (n={len(ids)})', '',
               '| 모델 | 정답 | 정확도 | 95% CI |', '|---|---|---|---|']
        for name, v in sorted(vecs.items(), key=lambda x: -sum(x[1])):
            k = sum(v)
            lo, hi = wilson(k, len(v))
            md.append(f'| {name} | {k} | {k/len(v):.4f} | {lo:.4f}–{hi:.4f} |')
        md += ['', f'- 1문항 = {100/len(ids):.3f}%p']

        base = a.baseline if a.baseline in vecs else max(vecs, key=lambda x: sum(vecs[x]))
        md += ['', f'### {base} 대비 짝비교 (paired bootstrap 10,000회)', '',
               '| 비교 | 평균차 | 95% CI | 예측 변경 | 판정 |', '|---|---|---|---|---|']
        for name, v in vecs.items():
            if name == base:
                continue
            d, lo, hi, ch = paired(v, vecs[base])
            sig = '**유의**' if (lo > 0 or hi < 0) else '잡음'
            md.append(f'| {name} − {base} | {d:+.2f}%p | {lo:+.2f}–{hi:+.2f} | {ch} | {sig} |')
        md.append('')

    md += ['## 채택 규칙', '',
           '1. **train 홀드아웃에서 유의 개선** + **H408에서 유의 악화 없음** → 채택',
           '2. 둘 다 유의 개선 → 강한 채택 (test 배포 우선순위)',
           '3. train 에서 잡음 → 보류. H408 단독 개선만으로는 채택하지 않는다',
           '   (n=408 에서 1~3문항 차이는 2026-09-23 내내 전부 잡음이었다)',
           '4. 두 셋의 방향이 반대면 채택하지 않고 원인을 먼저 본다',
           '5. **파인튜닝 모델은 val667 과 H229 로 본다.** train 홀드아웃 6,047 은 그들의 학습 데이터고',
           '   (train 6,714 = fit 6,047 + val667 667), holdout1500 도 R3 가 833 건을 학습했다.',
           '   파인튜닝 모델의 dev179 예측은 **데이터 버그로 무효**다(원본 질문 vs 수정본 gold). 재실행 전까지 쓰지 않는다.',
           '6. val667 은 파인튜닝의 **상한**, H229 는 **하한**으로 본다. 둘 다 n 이 작으므로',
           '   후보를 미리 정해 Public(n=3,357)에서 한 번만 읽는 것이 실질적으로 가장 큰 자다.', '']

    out = 'reports/dual_gate_20260923.md'
    open(out, 'w', encoding='utf-8', newline='\n').write('\n'.join(md) + '\n')
    try:
        print('\n'.join(md))
    except UnicodeEncodeError:  # Windows cp949 콘솔. 파일은 위에서 이미 저장됐다
        pass
    print(f'\n저장: {out}')


if __name__ == '__main__':
    main()
