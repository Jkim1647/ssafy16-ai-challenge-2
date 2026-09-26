# -*- coding: utf-8 -*-
"""트랙 A(2026-09-24): Qwen3.8-27B 제로샷(read+tiles)을 T6 앙상블 구성원으로 넣을 수 있는지 본다. GPU 미사용.

선택 기준(사용자 지시): H408 v2 + val667 합산 정답 수. val667 은 R3·Gemma 학습에 쓰지 않은 홀드아웃이다.
1위 팀 단서는 '다른 계열이 서로 다른 오답을 보완'이라, T6 가 틀린 문항 중 Q38 이 맞힌 수도 함께 본다.
val667·test 예측 파일이 아직 없으면 그 열은 '-' 로 둔다.
"""
import csv, glob, io, contextlib, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import explore_ens_20260924 as E  # noqa: E402
from ensemble_submit import load_any  # noqa: E402

P = 'shared_predictions/'
D = P + 'runpod_q38_20260924/'
E.M['Q38'] = E.load(*glob.glob(D + '*H408*predictions.jsonl'))
tp = glob.glob(D + '*TEST*predictions.jsonl')
has_test = bool(tp) and len(load_any(tp[0])) == len(E.tids)
if has_test:
    E.TEST['Q38'] = load_any(tp[0])
pr, g, ids, L = E.pr, E.g, E.ids, 'abcd'
missing = [i for i in ids if i not in E.M['Q38']]
assert not missing, f'Q38 H408 누락 {len(missing)}'

# val667: 네 구성원은 test 구성원과 같은 조건의 기존 예측(calibrate_members.py 와 같은 파일).
VAL = {'R3': E.load(P + 'runpod_27b_R3/R3_27b_tiles/val_predictions.jsonl'),
       'Gemma': E.load(P + 'runpod_gemma4_31b/gemma4_31b_lora/val_predictions.jsonl'),
       '397B': E.load(P + 'nebius_397b/Qwen3.5-397B-A17B-FP8_read_val667_tiles2x2x2_vllm_predictions.jsonl'),
       '35B': E.load(P + 'colab_ai2_35b_out/Qwen3.6-35B-A3B_read_val667_tiles2x2x2_predictions.jsonl')}
vp = glob.glob(D + '*val667*predictions.jsonl')
if vp:
    VAL['Q38'] = E.load(vp[0])
vgold = {r['id']: r['answer'] for r in csv.DictReader(open('ssafy-16-2-ai/train.csv', encoding='utf-8-sig'))}
vids = sorted(set.intersection(*[set(d) for d in VAL.values()]))
has_val = 'Q38' in VAL and len(vids) == 667

pick = lambda D_, fn, i: L[int(np.argmax(fn(D_, i)))]
q1 = lambda D_, i: pr(D_, 'Q38', i)
q = {i: pick(E.M, q1, i) for i in ids}
t6 = {i: pick(E.M, E.t6, i) for i in ids}
print(f'H408 v2 {len(ids)}문항 | Q38 단독 {sum(q[i] == g[i] for i in ids)} | T6 {sum(t6[i] == g[i] for i in ids)}')
print(f'T6 오답 중 Q38 정답 {sum(t6[i] != g[i] and q[i] == g[i] for i in ids)} / '
      f'T6 정답 중 Q38 오답 {sum(t6[i] == g[i] and q[i] != g[i] for i in ids)}')
if has_val:
    print(f'val667 | Q38 단독 {sum(pick(VAL, q1, i) == vgold[i] for i in vids)} | T6 {sum(pick(VAL, E.t6, i) == vgold[i] for i in vids)}')

BASE = ['R3', '35B', '397B', 'Gemma']
cands = {'T6': E.t6}
for w in (0.5, 1.0):
    cands[f'T6 + Q38 x{w}'] = lambda D_, i, w=w: (sum(pr(D_, n, i) for n in BASE) + w * pr(D_, 'Q38', i)) / (4 + w)
for n in BASE:
    cands[f'{n} -> Q38 교체'] = lambda D_, i, n=n: np.mean([pr(D_, m, i) for m in BASE if m != n] + [pr(D_, 'Q38', i)], 0)
cands['R3+Q38 반씩'] = lambda D_, i: (0.5 * pr(D_, 'R3', i) + 0.5 * pr(D_, 'Q38', i)
                                   + pr(D_, '35B', i) + pr(D_, '397B', i) + pr(D_, 'Gemma', i)) / 4

print(f'\n| 후보 | H408 v2 ({len(ids)}) | val667 | 합산 | T6 대비 test 변경 |'); print('|---|---|---|---|---|')
for k, fn in cands.items():
    h = sum(pick(E.M, fn, i) == g[i] for i in ids)
    v = sum(pick(VAL, fn, i) == vgold[i] for i in vids) if has_val else None
    ch = E.evaluate(fn)[1] if has_test else '-'
    print(f'| {k} | {h} | {"-" if v is None else v} | {"-" if v is None else h + v} | {ch} |')
