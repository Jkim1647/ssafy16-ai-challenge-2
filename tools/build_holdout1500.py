# -*- coding: utf-8 -*-
"""파인튜닝 비교용 홀드아웃 1,500건을 만든다 (GPU 미사용).

왜 필요한가
  기존 LoRA 모델(27B r8 등)은 train 6,047건을 학습에 썼으므로 남은 평가셋이 val667(667건)
  하나뿐이다. 1문항이 0.15%p 라 상위 모델들의 2~5문항 차이를 구별하지 못한다.
  (2026-09-23 val667 짝비교: 앙상블 648 / 27B LoRA 646 / 35B 644 / 397B 643 — 전부 잡음)

설계
  - **val667 을 포함하는 상위집합**으로 만든다. 기존 val667 숫자(27B LoRA 646, 앙상블 648 …)가
    그대로 비교 가능하고, 측정력만 2.25배가 된다(1문항 0.15%p → 0.067%p).
  - **이미지 group split**. 같은 이미지(또는 중복 이미지)가 학습과 평가에 나뉘어 들어가면
    누수다. data_meta/known_duplicates.csv 의 중복군을 한 덩어리로 묶어 배정한다.
  - 구조 결함·중복·누수 문항(data_meta/train_exclusions_v2.csv)은 **양쪽 모두에서 제외**한다.
  - 정답 글자 분포를 train 전체와 맞춘다(층화 추출).

출력
  data_meta/splits/holdout1500.json   {holdout: [...], fit: [...], ...}
"""
import hashlib, json, os, random
from collections import Counter, defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
SEED = 20260923
N_HOLDOUT = 1500


def main():
    tr = pd.read_csv('ssafy-16-2-ai/train.csv', encoding='utf-8-sig', dtype=str).fillna('')
    val667 = set(json.load(open('data_meta/splits/val667_ids.json', encoding='utf-8'))['val_groups'])

    excl = set()
    p = 'data_meta/train_exclusions_v2.csv'
    if os.path.exists(p):
        excl = set(pd.read_csv(p, encoding='utf-8-sig', dtype=str)['id'])
    print(f'제외 {len(excl)}건 (구조 결함 + 중복·누수)')

    # ---- 이미지 group: 중복 이미지는 한 덩어리로 ----
    group = {i: i for i in tr.id}          # 기본은 자기 자신
    dup = 'data_meta/known_duplicates.csv'
    if os.path.exists(dup):
        d = pd.read_csv(dup, encoding='utf-8-sig', dtype=str).fillna('')
        cols = [c for c in d.columns if 'id' in c.lower()]
        if len(cols) >= 2:
            for _, r in d.iterrows():
                a, b = r[cols[0]], r[cols[1]]
                if a in group and b in group:
                    group[b] = group[a]     # 같은 대표로 묶는다
    members = defaultdict(list)
    for i in tr.id:
        if i not in excl:
            members[group[i]].append(i)
    print(f'이미지 그룹 {len(members)}개 / 유효 문항 {sum(len(v) for v in members.values())}')

    # ---- val667 이 속한 그룹은 통째로 홀드아웃 ----
    hold_groups = {group[i] for i in val667 if i in group}
    hold = {i for gp in hold_groups for i in members.get(gp, [])}
    print(f'val667 그룹 확장: {len(val667)} → {len(hold)}건')

    # ---- 정답 분포를 맞춰 부족분을 채운다 ----
    ans = dict(zip(tr.id, tr.answer))
    target = Counter(ans[i] for i in tr.id if i not in excl)
    tot = sum(target.values())
    want = {k: round(N_HOLDOUT * v / tot) for k, v in target.items()}
    have = Counter(ans[i] for i in hold)

    rng = random.Random(SEED)
    cand = [gp for gp in members if gp not in hold_groups]
    rng.shuffle(cand)
    for gp in cand:
        if len(hold) >= N_HOLDOUT:
            break
        ids = members[gp]
        # 이 그룹을 넣으면 어떤 글자가 늘어나는지 — 가장 모자란 글자를 우선 채운다
        gain = Counter(ans[i] for i in ids)
        short = {k: want.get(k, 0) - have.get(k, 0) for k in 'abcd'}
        if all(short.get(k, 0) <= 0 for k in gain):
            continue
        hold |= set(ids)
        have += gain
    # 그래도 모자라면 분포 무시하고 채운다
    for gp in cand:
        if len(hold) >= N_HOLDOUT:
            break
        if not (set(members[gp]) & hold):
            hold |= set(members[gp])

    hold = set(sorted(hold)[:N_HOLDOUT]) if len(hold) > N_HOLDOUT else hold
    fit = [i for i in sorted(tr.id) if i not in hold and i not in excl]
    hold = sorted(hold)

    # ---- 검증 ----
    assert not (set(hold) & set(fit)), '홀드아웃과 학습셋이 겹친다'
    assert not (set(hold) & excl) and not (set(fit) & excl), '제외 문항이 남아 있다'
    leak = {group[i] for i in hold} & {group[i] for i in fit}
    assert not leak, f'이미지 그룹 누수 {len(leak)}개'
    missing_val = val667 - set(hold)
    assert not missing_val, f'val667 중 {len(missing_val)}건이 홀드아웃에 없다'

    obj = {'holdout': hold, 'fit': fit, 'n_holdout': len(hold), 'n_fit': len(fit),
           'seed': SEED, 'built': '2026-09-23',
           'rule': 'val667을 포함하는 상위집합. 이미지 group split, train_exclusions_v2 제외, 정답 분포 층화',
           'contains_val667': True, 'n_val667': len(val667),
           'excluded': len(excl),
           'answer_dist_holdout': dict(Counter(ans[i] for i in hold)),
           'answer_dist_fit': dict(Counter(ans[i] for i in fit)),
           'holdout_sha256': hashlib.sha256('\n'.join(hold).encode()).hexdigest()}
    out = 'data_meta/splits/holdout1500.json'
    json.dump(obj, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    print()
    print(f'홀드아웃 {len(hold)}  /  학습셋 {len(fit)}  /  제외 {len(excl)}')
    print(f'  합계 검증: {len(hold)+len(fit)+len(excl)} == {len(tr)} ->',
          len(hold) + len(fit) + len(excl) == len(tr))
    print(f'  val667 포함: {len(val667 & set(hold))}/{len(val667)}')
    print(f'  정답 분포 홀드아웃 {obj["answer_dist_holdout"]}')
    print(f'  정답 분포 학습셋   {obj["answer_dist_fit"]}')
    print(f'  1문항 = {100/len(hold):.3f}%p  (val667 은 {100/len(val667):.3f}%p)')
    print(f'저장: {out}')


if __name__ == '__main__':
    main()
