# -*- coding: utf-8 -*-
"""앙상블 후보를 val667 + H408 v2 합산으로 채점한다(사용자 지시: 두 셋을 합쳐 본다). GPU 미사용.

val667 공정성: R3·Gemma·Q35(Qwen3.5-27B LoRA)는 val667 을 학습에 쓰지 않았고, 35B·397B·Q38 은 제로샷이다.
따라서 이 여섯 구성원의 조합은 val667 에서도 처음 보는 문항으로 채점된다. (27B·9B LoRA 등 다른 구성원은 제외)
합산은 1,068문항(val667 667 + H408 v2 401). 1문항 = 0.094%p.
"""
import csv, io, contextlib, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import report_all_submissions as R  # noqa: E402
from ensemble_submit import load_any  # noqa: E402
from ensemble_methods import lsm  # noqa: E402

E = R.E
L = 'abcd'
P = 'shared_predictions/'
E.M['Q35'] = load_any(P + 'runpod_g_20260924/hard_predictions.jsonl')
V = {'R3': load_any(P + 'runpod_27b_R3/R3_27b_tiles/val_predictions.jsonl'),
     'Gemma': load_any(P + 'runpod_gemma4_31b/gemma4_31b_lora/val_predictions.jsonl'),
     '35B': load_any(P + 'colab_ai2_35b_out/Qwen3.6-35B-A3B_read_val667_tiles2x2x2_predictions.jsonl'),
     '397B': load_any(P + 'nebius_397b/Qwen3.5-397B-A17B-FP8_read_val667_tiles2x2x2_vllm_predictions.jsonl'),
     'Q38': load_any(P + 'runpod_q38_20260924/Qwen3.8-27B_read_val667_tiles2x2x2_vllm_predictions.jsonl'),
     'Q35': load_any(P + 'runpod_g_20260924/val_predictions.jsonl')}
gold = {r['id']: r['answer'] for r in csv.DictReader(open('ssafy-16-2-ai/train.csv', encoding='utf-8-sig'))}
vids = sorted(set.intersection(*[set(d) for d in V.values()]))
assert len(vids) == 667, len(vids)


def p(D, n, i):
    return np.exp(lsm(D[n][i]['logprobs']))


def mix(ws):
    """ws: {구성원: 가중치} -> (val667 정답, H408 v2 정답)"""
    tot = sum(ws.values())
    v = sum(L[int(np.argmax(sum(w * p(V, n, i) for n, w in ws.items()) / tot))] == gold[i] for i in vids)
    h = sum(L[int(np.argmax(sum(w * p(E.M, n, i) for n, w in ws.items()) / tot))] == E.g[i] for i in E.ids)
    return v, h


CANDS = {
    'T6 (R3·35B·397B·Gemma)': {'R3': 1, '35B': 1, '397B': 1, 'Gemma': 1},
    'T15 (T6 + Q38 0.5)': {'R3': 1, '35B': 1, '397B': 1, 'Gemma': 1, 'Q38': .5},
    'T6 + Q35 0.5': {'R3': 1, '35B': 1, '397B': 1, 'Gemma': 1, 'Q35': .5},
    'T6 + Q35 1': {'R3': 1, '35B': 1, '397B': 1, 'Gemma': 1, 'Q35': 1},
    'T15 + Q35 0.5': {'R3': 1, '35B': 1, '397B': 1, 'Gemma': 1, 'Q38': .5, 'Q35': .5},
    'T15 + Q35 1': {'R3': 1, '35B': 1, '397B': 1, 'Gemma': 1, 'Q38': .5, 'Q35': 1},
    'R3 -> Q35': {'Q35': 1, '35B': 1, '397B': 1, 'Gemma': 1},
    'R3 -> Q35 + Q38 0.5': {'Q35': 1, '35B': 1, '397B': 1, 'Gemma': 1, 'Q38': .5},
    '6개 균등': {'R3': 1, '35B': 1, '397B': 1, 'Gemma': 1, 'Q38': 1, 'Q35': 1},
    '1위 팀식 Q35·Q38·Gemma': {'Q35': 1, 'Q38': 1, 'Gemma': 1},
    'Q35·Q38·Gemma·397B': {'Q35': 1, 'Q38': 1, 'Gemma': 1, '397B': 1},
}

if __name__ == '__main__':
    print(f'| 구성원 | val667 단독 | H408 v2 단독 |'); print('|---|---|---|')
    for n in V:
        print(f'| {n} | {sum(L[int(np.argmax(p(V, n, i)))] == gold[i] for i in vids)} | '
              f'{sum(L[int(np.argmax(p(E.M, n, i)))] == E.g[i] for i in E.ids)} |')
    print(f'\n| 후보 | val667 (667) | H408 v2 ({len(E.ids)}) | 합산 |'); print('|---|---|---|---|')
    for k, ws in CANDS.items():
        v, h = mix(ws)
        print(f'| {k} | {v} | {h} | **{v + h}** |')
