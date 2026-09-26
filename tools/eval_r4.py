# -*- coding: utf-8 -*-
"""트랙 B(2026-09-24): R4 = R3 설정 + --aug-shift(보기 순서 순환) 27B LoRA 를 H408 v2 + val667 합산으로 본다. GPU 미사용.

R4 는 R3 와 같은 계열(같은 모델·데이터·설정)이라 5번째 구성원보다 R3 교체·R3 와 반씩이 자연스러운 후보다.
val667 은 R3·R4·Gemma 학습에 쓰지 않은 홀드아웃이다. test 예측이 있으면 T6 대비 test 변경 수도 적는다.
사용: python tools/eval_r4.py <R4 출력 폴더(val_predictions.jsonl, hard_predictions.jsonl[, test_predictions.jsonl])>
"""
import csv, io, contextlib, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import explore_ens_20260924 as E  # noqa: E402
from ensemble_submit import load_any  # noqa: E402

D = sys.argv[1] if len(sys.argv) > 1 else 'shared_predictions/colab_R4_20260924/'
P = 'shared_predictions/'
E.M['R4'] = E.load(os.path.join(D, 'hard_predictions.jsonl'))
tp = os.path.join(D, 'test_predictions.jsonl')
has_test = os.path.exists(tp) and len(load_any(tp)) == len(E.tids)
if has_test:
    E.TEST['R4'] = load_any(tp)
pr, g, ids, L = E.pr, E.g, E.ids, 'abcd'
missing = [i for i in ids if i not in E.M['R4']]
assert not missing, f'R4 H408 누락 {len(missing)}'

VAL = {'R3': E.load(P + 'runpod_27b_R3/R3_27b_tiles/val_predictions.jsonl'),
       'Gemma': E.load(P + 'runpod_gemma4_31b/gemma4_31b_lora/val_predictions.jsonl'),
       '397B': E.load(P + 'nebius_397b/Qwen3.5-397B-A17B-FP8_read_val667_tiles2x2x2_vllm_predictions.jsonl'),
       '35B': E.load(P + 'colab_ai2_35b_out/Qwen3.6-35B-A3B_read_val667_tiles2x2x2_predictions.jsonl'),
       'R4': E.load(os.path.join(D, 'val_predictions.jsonl'))}
vgold = {r['id']: r['answer'] for r in csv.DictReader(open('ssafy-16-2-ai/train.csv', encoding='utf-8-sig'))}
vids = sorted(set.intersection(*[set(d) for d in VAL.values()]))
assert len(vids) == 667, len(vids)

pick = lambda D_, fn, i: L[int(np.argmax(fn(D_, i)))]
single = lambda n: (lambda D_, i: pr(D_, n, i))
for n in ('R3', 'R4'):
    print(f'{n} 단독: H408 v2 {sum(pick(E.M, single(n), i) == g[i] for i in ids)} / val667 '
          f'{sum(pick(VAL, single(n), i) == vgold[i] for i in vids)}')
t6 = {i: pick(E.M, E.t6, i) for i in ids}
r4 = {i: pick(E.M, single('R4'), i) for i in ids}
print(f'T6 오답 중 R4 정답 {sum(t6[i] != g[i] and r4[i] == g[i] for i in ids)} / '
      f'T6 정답 중 R4 오답 {sum(t6[i] == g[i] and r4[i] != g[i] for i in ids)}')

BASE = ['R3', '35B', '397B', 'Gemma']
cands = {'T6': E.t6,
         'R3 -> R4 교체': lambda D_, i: np.mean([pr(D_, n, i) for n in ['R4', '35B', '397B', 'Gemma']], 0),
         'R3+R4 반씩': lambda D_, i: (0.5 * pr(D_, 'R3', i) + 0.5 * pr(D_, 'R4', i)
                                    + pr(D_, '35B', i) + pr(D_, '397B', i) + pr(D_, 'Gemma', i)) / 4,
         'T6 + R4 (5개)': lambda D_, i: np.mean([pr(D_, n, i) for n in BASE + ['R4']], 0)}

print(f'\n| 후보 | H408 v2 ({len(ids)}) | val667 | 합산 | T6 대비 test 변경 |'); print('|---|---|---|---|---|')
for k, fn in cands.items():
    h = sum(pick(E.M, fn, i) == g[i] for i in ids)
    v = sum(pick(VAL, fn, i) == vgold[i] for i in vids)
    ch = E.evaluate(fn)[1] if has_test else '-'
    print(f'| {k} | {h} | {v} | {h + v} | {ch} |')
