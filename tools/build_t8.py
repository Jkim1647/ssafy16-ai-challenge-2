# -*- coding: utf-8 -*-
"""T8 — COMPOSITE 문항만 35B 계열(35B·r2v3·fs16)을 **한 표로 묶어** T6 에 넣는다. GPU 미사용.

    COMPOSITE:  p = mean(R3, fam, 397B, Gemma),  fam = (w35*35B + wr*r2v3 + wf*fs16) / (w35+wr+wf)
    그 외:       p = T6 = mean(R3, 35B, 397B, Gemma)          (확률 산술평균)

변형(2026-09-24 Codex 제안): A (1, .5, .5) / B (.5, .5, 1) / C (1, 0, 1) / D (0, 0, 1)
제출 전 체크리스트(Codex 2026-09-24): 필수 1·2·3·4·5·6·9 는 하나라도 실패하면 파일을 쓰지 않는다.
7(H408 재현)은 기대값이 있는 변형(A)에만 적용하고 그 변형만 막는다. 8(변경 수)·10(SHA256)은 기록용 — 변경 0 이면 쓰지 않는다.

usage: python tools/build_t8.py --fs16-test <out_test_c predictions.jsonl> --fs16-meta <meta.json>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ensemble_methods import lsm  # noqa: E402
from ensemble_submit import load_any  # noqa: E402

P = 'shared_predictions/'
TEST = {
    'R3': P + 'runpod_27b_R3/R3_h408/test_predictions.jsonl',
    '35B': P + 'colab_ai2_35b_out/Qwen3.6-35B-A3B_read_test_tiles2x2x2_predictions.jsonl',
    '397B': P + 'nebius_397b_test/Qwen3.5-397B-A17B-FP8_read_test_tiles2x2x2_vllm_predictions.jsonl',
    'Gemma': P + 'runpod_gemma4_31b/gemma_h408_test/test_predictions.jsonl',
    'r2v3': P + 'runpod_test_r2v3_20260924/Qwen3.6-35B-A3B_read_testTEST_r2v3_tiles2x2x2_r2v3_vllm_predictions.jsonl',
}
VARIANTS = {'A': (1, .5, .5), 'B': (.5, .5, 1), 'C': (1, 0, 1), 'D': (0, 0, 1)}
H408_EXPECT = {'A': 381}          # 같은 코드로 H408 을 다시 계산했을 때 나와야 하는 값(체크 7)


def probs(d, i):
    v = np.array(d[i]['logprobs'], float)
    assert v.shape == (4,) and np.all(np.isfinite(v)), f'NaN/형상 오류 {i}'
    return np.exp(lsm(v))


def combine(get, i, comp, w):
    base = [get('R3', i), get('35B', i), get('397B', i), get('Gemma', i)]
    if not comp:
        return np.mean(base, 0)
    w35, wr, wf = w
    fam = (w35 * get('35B', i) + wr * get('r2v3', i) + wf * get('fs16', i)) / (w35 + wr + wf)
    return np.mean([base[0], fam, base[2], base[3]], 0)


def h408_score(w):
    from h408_ensemble import M, g, ids, load, S  # noqa: E402
    M.setdefault('35Bfs16', load(S + 'fs16_predictions.jsonl', S + 'd179/fs16_predictions.jsonl'))
    alias = {'r2v3': '35Br2v3', 'fs16': '35Bfs16'}
    T = types()
    get = lambda n, i: probs(M[alias.get(n, n)], i)
    return sum('abcd'[int(np.argmax(combine(get, i, T.get(i) == 'COMPOSITE', w)))] == g[i] for i in ids)


def types():
    with open('data_meta/question_types.csv', encoding='utf-8-sig', newline='') as fh:
        return {r['row_id']: r['auto_type'] for r in csv.DictReader(fh)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--fs16-test', required=True)
    ap.add_argument('--fs16-meta', required=True)
    ap.add_argument('--variants', default='A,B,C,D')
    a = ap.parse_args()

    ok = True
    def check(n, cond, msg):
        nonlocal ok
        print(f'[{"통과" if cond else "실패"}] {n}. {msg}')
        ok &= bool(cond)

    D = {k: load_any(v) for k, v in TEST.items()}
    raw = [json.loads(l) for l in open(a.fs16_test, encoding='utf-8') if l.strip()]
    D['fs16'] = {r['id']: r for r in raw}
    T = types()
    with open('ssafy-16-2-ai/sample_submission.csv', encoding='utf-8-sig', newline='') as fh:
        order = [r['id'] for r in csv.DictReader(fh)]
    comp = {i for i in order if T.get(i) == 'COMPOSITE'}

    check(1, len(comp) == 2197 and set(D['fs16']) == comp, f'COMPOSITE {len(comp)}건, fs16 이 정확히 그 집합을 덮음({len(set(D["fs16"]) & comp)})')
    dup = len(raw) - len(D['fs16'])
    nan = sum(not all(math.isfinite(x) for x in r['logprobs']) or len(r['logprobs']) != 4 for r in raw)
    check(3, dup == 0 and nan == 0, f'fs16 중복 {dup} / NaN·형상 오류 {nan}')
    meta = json.load(open(a.fs16_meta, encoding='utf-8'))
    check(4, meta.get('choice_shift', 0) == 0, f'보기 순환 없음(choice_shift={meta.get("choice_shift")}) → logprobs 가 a,b,c,d 원래 순서')
    # meta 의 prompt 필드는 모드 이름이 아니라 1단계 판독 지시 원문이다(검증 run 과 같은 형식).
    check(6, meta.get('fewshot') == 16 and 'Do not answer yet' in str(meta.get('prompt')) and meta.get('n') == 2197,
          f'fs16 버그 재발 없음(meta fewshot={meta.get("fewshot")}, read 프롬프트, n={meta.get("n")})')
    pp = [probs(D['fs16'], i) for i in comp]
    check(5, all(np.all(p >= 0) and abs(p.sum() - 1) < 1e-4 for p in pp), '확률이 모두 0 이상, 합이 1 에서 1e-4 이내')
    for k in ['R3', '35B', '397B', 'Gemma', 'r2v3']:
        check('0', all(i in D[k] for i in order), f'{k} test 6,714건 전부 있음')

    get = lambda n, i: probs(D[n], i)
    t6 = {i: int(np.argmax(combine(get, i, False, None))) for i in order}
    ref = {}
    with open('submissions/T6_4pmean_test.csv', encoding='utf-8-sig', newline='') as fh:
        ref = {r['id']: r['answer'] for r in csv.DictReader(fh)}
    check('0', all('abcd'[t6[i]] == ref[i] for i in order), '이 코드의 T6 가 제출한 T6 와 6,714건 모두 같다')

    out = {}
    for v in a.variants.split(','):
        w = VARIANTS[v]
        pred = {i: int(np.argmax(combine(get, i, i in comp, w))) for i in order}
        same_non = all(pred[i] == t6[i] for i in order if i not in comp)
        check(2, same_non, f'[{v}] 비COMPOSITE {len(order) - len(comp)}건이 T6 와 동일')
        h = h408_score(w)
        ch = sum(pred[i] != t6[i] for i in order)
        blocked = False
        if v in H408_EXPECT and h != H408_EXPECT[v]:
            print(f'[실패] 7. [{v}] H408 재현 {h} (기대 {H408_EXPECT[v]}) — 이 변형만 제외'); blocked = True
        elif v in H408_EXPECT:
            print(f'[통과] 7. [{v}] H408 재현 {h}')
        if ch == 0:
            print(f'    [{v}] T6 와 같은 제출 — 제외'); blocked = True
        print(f'    [{v}] w(35B,r2v3,fs16)={w}  H408 {h}  T6 대비 변경 {ch}건 (체크 8)')
        if not blocked:
            out[v] = (pred, h, ch)

    if not ok:
        print('체크 실패 — 제출 파일을 쓰지 않는다')
        return 1
    shas = {}
    for v, (pred, h, ch) in out.items():
        fn = f'submissions/T8{v}_comp35fam_test.csv'
        with open(fn, 'w', encoding='utf-8', newline='') as fh:   # 제출 CSV 는 BOM 없이
            fh.write('id,answer\n' + ''.join(f'{i},{"abcd"[pred[i]]}\n' for i in order))
        rows = list(csv.DictReader(open(fn, encoding='utf-8', newline='')))
        check(9, [r['id'] for r in rows] == order, f'[{v}] ID 순서가 sample_submission 과 같다')
        shas[fn] = hashlib.sha256(open(fn, 'rb').read()).hexdigest()
    shas[a.fs16_test] = hashlib.sha256(open(a.fs16_test, 'rb').read()).hexdigest()
    with open('submissions/T8_sha256.json', 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(shas, fh, ensure_ascii=False, indent=1)
    check(10, True, 'SHA256 저장: submissions/T8_sha256.json')
    # 변형끼리 서로 얼마나 다른가(사실상 같은 후보 제거용)
    vs = list(out)
    for x in range(len(vs)):
        for y in range(x + 1, len(vs)):
            print(f'    {vs[x]}↔{vs[y]} 다른 문항 {sum(out[vs[x]][0][i] != out[vs[y]][0][i] for i in order)}')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
