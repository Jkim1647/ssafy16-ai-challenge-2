# -*- coding: utf-8 -*-
"""H408(=H229+수정본 dev179)에서 구성원 조합 x 결합 방식을 채점한다. GPU 미사용.

2026-09-24 R3·Gemma 를 수정본 H408 로 다시 채점해(runpod_h408ft_20260924) 처음으로
T4 구성원 전부가 올바른 H408 예측을 갖게 됐다. 결합 방식은 tools/ensemble_methods.py 와 같다.
35B 는 test 구성원과 같은 read2 v1 조건인 오늘 스크리닝 base(H229) + dev179 재실행을 쓴다.
"""
import json, itertools, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ensemble_methods import combine, lsm  # noqa: E402

P = 'shared_predictions/'
S = P + 'runpod_screen_20260924/'
def load(*paths):
    d = {}
    for p in paths:
        for l in open(p, encoding='utf-8'):
            if l.strip():
                o = json.loads(l); d[o['id']] = o
    return d
class Gold(str):
    """정답 글자(중복정답이면 'cd' 처럼 여러 글자). `pred == g[i]` 가 '정답 중 하나인가'로 동작한다."""
    def __eq__(self, o):
        return isinstance(o, str) and len(o) == 1 and o in str(self)
    def __ne__(self, o):
        return not self.__eq__(o)
    __hash__ = str.__hash__


# 2026-09-24 부터 평가 기준은 H408 v2(김진영 이미지 검토: 정답 수정 5 / 중복정답 3 / 애매 4 제외, 404문항).
# 옛 기준으로 보려면 환경변수 H408_GOLD=v1.
if os.environ.get('H408_GOLD', 'v2') == 'v1':
    g = {k: Gold(v) for k, v in json.load(open('runs/h408_pack/hard_gold_h408.json', encoding='utf-8')).items()}
else:
    g = {k: Gold(v) for k, v in json.load(open('runs/h408_pack/hard_gold_h408_v2.json', encoding='utf-8'))['gold'].items()}
g229 = set(json.load(open('runs/dev_validation_v1/hard_eval_gold_v5_H.json', encoding='utf-8')))
M = {
    'R3': load(P + 'runpod_h408ft_20260924/R3/hard_predictions.jsonl'),
    'Gemma': load(P + 'runpod_h408ft_20260924/Gemma/hard_predictions.jsonl'),
    '397B': load(P + 'nebius_397b/Qwen3.5-397B-A17B-FP8_read_devH331_tiles2x2x2_vllm_predictions.jsonl',
                 P + 'nebius_397b_rerun179/Qwen3.5-397B-A17B-FP8_read_devRERUN179_tiles2x2x2_vllm_predictions.jsonl'),
    '35B': load(S + 'base_predictions.jsonl', S + 'd179/base_predictions.jsonl'),
    '35Br2v3': load(S + 'r2v3_predictions.jsonl', S + 'd179/r2v3_predictions.jsonl'),
    '35Br1v3': load(S + 'r1v3_predictions.jsonl', S + 'd179/r1v3_predictions.jsonl'),
}
ids = [i for i in g if all(i in d for d in M.values())]
assert len(ids) == len(g), (len(ids), len(g))
L = 'abcd'
def score(names, method):
    ok = 0; h = 0
    for i in ids:
        pr = L[int(np.argmax(combine(method, np.stack([lsm(M[n][i]['logprobs']) for n in names]))))]
        ok += pr == g[i]; h += (pr == g[i]) and i in g229
    return ok, h
print(f'H408 기준: {os.environ.get("H408_GOLD", "v2")} ({len(ids)}문항)')
print(f'H408 기준 {os.environ.get("H408_GOLD", "v2")} ({len(ids)}문항)')
print('| 구성 | 방식 | H408 | H229 | dev179 |'); print('|---|---|---|---|---|')
for n in M:
    o, h = score([n], 'soft'); print(f'| {n} 단독 | - | {o} | {h} | {o-h} |')
combos = {
    'T4 (R3,35B,397B,Gemma)': ['R3', '35B', '397B', 'Gemma'],
    'T6 교체 (R3,35Br2v3,397B,Gemma)': ['R3', '35Br2v3', '397B', 'Gemma'],
    '5개 (T4+35Br2v3)': ['R3', '35B', '397B', 'Gemma', '35Br2v3'],
    '5개 (T6+35Br1v3)': ['R3', '35Br2v3', '397B', 'Gemma', '35Br1v3'],
}
for cname, names in combos.items():
    for m in ['soft', 'pmean', 'hard', 'borda', 'trim']:
        if m == 'hard' and len(names) % 2 == 0:
            continue
        o, h = score(names, m); print(f'| {cname} | {m} | {o} | {h} | {o-h} |')
