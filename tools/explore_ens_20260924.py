# -*- coding: utf-8 -*-
"""2026-09-24 오후 추가 탐색 — 기존 확률만으로 앙상블 후보를 H408 과 test 변경 수로 비교한다. GPU 미사용.

후보군(사용자 지시):
  1. T6·T7 확률 혼합 (0.75/0.5/0.25)
  2. 35B 계열 = (35B + w·r2v3)/(1+w), 전 문항 (w = 0.25/0.5/0.75)
  3. COMPOSITE 이고 T6 마진 < th 일 때만 35B 계열(1:.5:.5) (th = 0.05/0.10/0.15/0.20)
  4. 구성원별 거친 가중치 {0.5,1,1.5}^4 (R3, 35B계열, 397B, Gemma) — 81개. 결과를 보고 고르면 H408 과적합이므로
     상위 결과는 참고로만 적고, 제출 조건(H408≥380, T6 대비 test 변경≥5)을 함께 본다.
"""
import csv, itertools, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ensemble_methods import lsm  # noqa: E402
from ensemble_submit import load_any  # noqa: E402
from h408_ensemble import M, g, ids, load, S  # noqa: E402

M['35Bfs16'] = load(S + 'fs16_predictions.jsonl', S + 'd179/fs16_predictions.jsonl')
P = 'shared_predictions/'
TEST = {'R3': load_any(P + 'runpod_27b_R3/R3_h408/test_predictions.jsonl'),
        '35B': load_any(P + 'colab_ai2_35b_out/Qwen3.6-35B-A3B_read_test_tiles2x2x2_predictions.jsonl'),
        '397B': load_any(P + 'nebius_397b_test/Qwen3.5-397B-A17B-FP8_read_test_tiles2x2x2_vllm_predictions.jsonl'),
        'Gemma': load_any(P + 'runpod_gemma4_31b/gemma_h408_test/test_predictions.jsonl'),
        '35Br2v3': load_any(P + 'runpod_test_r2v3_20260924/Qwen3.6-35B-A3B_read_testTEST_r2v3_tiles2x2x2_r2v3_vllm_predictions.jsonl'),
        '35Bfs16': load_any(P + 'runpod_fs16_testc_20260924/Qwen3.6-35B-A3B_read_devSCREEN_fs16_tiles2x2x2_vllm_predictions.jsonl')}
T = {r['row_id']: r['auto_type'] for r in csv.DictReader(open('data_meta/question_types.csv', encoding='utf-8-sig'))}
tids = [f'test_{i:04d}.jpg' for i in range(1, 6715)]


def pr(D, n, i):
    return np.exp(lsm(D[n][i]['logprobs']))


def marg(p):
    s = np.sort(p); return s[-1] - s[-2]


def t6(D, i):
    return (pr(D, 'R3', i) + pr(D, '35B', i) + pr(D, '397B', i) + pr(D, 'Gemma', i)) / 4


def t7(D, i):
    return (4 * t6(D, i) + pr(D, '35Br2v3', i)) / 5


def fam(D, i, w35=1, wr=0, wf=0):
    num = w35 * pr(D, '35B', i) + wr * pr(D, '35Br2v3', i)
    den = w35 + wr
    if wf and i in D['35Bfs16']:
        num = num + wf * pr(D, '35Bfs16', i); den += wf
    return num / den


CANDS = {}
for a in (0.75, 0.5, 0.25):
    CANDS[f'T6·T7 혼합 {a}:{1-a}'] = lambda D, i, a=a: a * t6(D, i) + (1 - a) * t7(D, i)
for w in (0.25, 0.5, 0.75):
    CANDS[f'35B계열 r2v3 {w} (전 문항)'] = lambda D, i, w=w: (pr(D, 'R3', i) + fam(D, i, 1, w) + pr(D, '397B', i) + pr(D, 'Gemma', i)) / 4
for th in (0.05, 0.10, 0.15, 0.20):
    CANDS[f'COMPOSITE & 마진<{th} → 계열(1:.5:.5)'] = (
        lambda D, i, th=th: (pr(D, 'R3', i) + fam(D, i, 1, .5, .5) + pr(D, '397B', i) + pr(D, 'Gemma', i)) / 4
        if T.get(i) == 'COMPOSITE' and marg(t6(D, i)) < th else t6(D, i))
grid = {}
for wR, wF, w3, wG in itertools.product((0.5, 1, 1.5), repeat=4):
    grid[(wR, wF, w3, wG)] = (lambda D, i, w=(wR, wF, w3, wG): (w[0] * pr(D, 'R3', i) + w[1] * fam(D, i, 1, .5)
                                                              + w[2] * pr(D, '397B', i) + w[3] * pr(D, 'Gemma', i)))


def evaluate(fn):
    h = sum('abcd'[int(np.argmax(fn(M, i)))] == g[i] for i in ids)
    base = {i: int(np.argmax(t6(TEST, i))) for i in tids}
    pred = {i: int(np.argmax(fn(TEST, i))) for i in tids}
    return h, sum(pred[i] != base[i] for i in tids), pred


if __name__ == '__main__':
    print('| 후보 | H408 | T6 대비 test 변경 |'); print('|---|---|---|')
    print(f'| T6 | {evaluate(t6)[0]} | 0 |')
    print(f'| T7 | {evaluate(t7)[0]} | {evaluate(t7)[1]} |')
    for k, fn in CANDS.items():
        h, ch, _ = evaluate(fn); print(f'| {k} | {h} | {ch} |')
    res = sorted(((evaluate(fn)[:2], w) for w, fn in grid.items()), key=lambda x: (-x[0][0], x[0][1]))
    print('\n거친 가중치 81개 중 H408 상위 8 (R3, 35B계열, 397B, Gemma):')
    for (h, ch), w in res[:8]:
        print(f'  {w}  H408 {h}  변경 {ch}')
    print('  H408 분포:', {h: sum(1 for (hh, _), _w in res if hh == h) for h in sorted({r[0][0] for r in res}, reverse=True)})


def write_sub(fn_pred, path):
    """검사(6,714행·ID 순서·a~d)를 거쳐 제출 CSV 를 쓴다. BOM 없음."""
    import hashlib
    order = [r['id'] for r in csv.DictReader(open('ssafy-16-2-ai/sample_submission.csv', encoding='utf-8-sig'))]
    assert order == tids and len(order) == 6714
    pred = {i: int(np.argmax(fn_pred(TEST, i))) for i in tids}
    with open(path, 'w', encoding='utf-8', newline='') as fh:
        fh.write('id,answer\n' + ''.join(f'{i},{"abcd"[pred[i]]}\n' for i in order))
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()
