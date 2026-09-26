# -*- coding: utf-8 -*-
"""확정 결함을 학습 split과 평가 gold에서 제외한다 (GPU 미사용).

제외 대상은 `tools/build_defect_queue.py`가 규칙으로 확정한 것만이다.
  - 보기중복  : 정규화 후 보기 둘이 같다      -> 문제로 성립하지 않음
  - 깨진_보기 : 보기가 '?'/U+FFFD로 손상      -> 판정 불가

**정답⊂보기는 제외하지 않는다.** 사람 판정이 아직 없고, 상당수는 정상이다
(예: `행복마트` vs `행복마트 본점 식품관`은 명확히 구분된다).
판정이 끝나면 `--drop-substring <판정완료.csv>`로 `복수정답`만 추가 제외한다.

출력
  data_meta/splits/train_ids_v2_clean.json          학습 ID (v1 - 확정결함)
  runs/dev_validation_v1/hard_eval_gold_v5.json     gold_v4 - 확정결함
  runs/dev_validation_v1/hard_eval_gold_v5_H.json   H229  - 확정결함  (모델 비교용)
  reports/defect_exclusion_20260923.md              제외 내역과 재채점 델타
"""
import argparse, hashlib, json, os, sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from build_defect_queue import dup_pairs, broken_opts  # 동일 규칙 재사용

DATA = 'ssafy-16-2-ai/'
SPL = 'data_meta/splits/'
OUT = 'runs/dev_validation_v1/'
P = 'shared_predictions/'
L = 'abcd'


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False)
                          .encode('utf-8')).hexdigest()


AUDIT = 'review_gold_final_20260923/review_gold_final_20260923/audit_all_350.csv'


def _scan(d):
    d = d.copy()
    d['_dup'] = d.apply(dup_pairs, axis=1)
    d['_brk'] = d.apply(broken_opts, axis=1)
    bad = d[(d['_dup'] != '') | (d['_brk'] != '')]
    return {r['id']: ';'.join(filter(None, [
        f"보기중복({r['_dup']})" if r['_dup'] else '',
        f"깨진_보기({r['_brk']})" if r['_brk'] else ''])) for _, r in bad.iterrows()}


def confirmed(split):
    """원본 CSV 기준 결함."""
    return _scan(pd.read_csv(DATA + f'{split}.csv', encoding='utf-8-sig', dtype=str).fillna(''))


def confirmed_final():
    """검수에서 고친 문항은 '고친 뒤' 보기로 검사해야 한다.
    audit_all_350.csv의 final_* 컬럼을 a~d로 매핑해 다시 스캔한다."""
    if not os.path.exists(AUDIT):
        return {}
    A = pd.read_csv(AUDIT, encoding='utf-8-sig', dtype=str).fillna('')
    F = A[['id']].copy()
    for c in L:
        F[c] = A['final_' + c]
    return _scan(F)


def load_preds(p):
    d = {}
    for line in open(p, encoding='utf-8'):
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--drop-substring', default=None,
                    help='정답⊂보기 판정 완료 CSV. 판정==복수정답 인 행만 추가 제외한다.')
    a = ap.parse_args()

    tr_bad = confirmed('train')
    dv_orig = confirmed('dev')
    dv_final = confirmed_final()          # 검수에서 고친 뒤 기준
    reviewed = set()
    if os.path.exists(AUDIT):
        reviewed = set(pd.read_csv(AUDIT, encoding='utf-8-sig', dtype=str)['id'])
    # 검수된 문항은 고친 결과로, 나머지는 원본으로 판정한다
    dv_bad = {k: v for k, v in dv_orig.items() if k not in reviewed}
    dv_bad.update(dv_final)
    print(f'[dev 결함] 원본기준 {len(dv_orig)} / 검수 후 재검출 {len(dv_final)} '
          f'-> 적용 {len(dv_bad)} (검수로 고쳐져 빠진 것 {len(set(dv_orig) & reviewed - set(dv_final))})')
    extra = set()
    if a.drop_substring:
        s = pd.read_csv(a.drop_substring, encoding='utf-8-sig', dtype=str).fillna('')
        extra = set(s[s['판정'].str.strip() == '복수정답']['id'])
        print(f'[추가 제외] 정답⊂보기 판정 "복수정답" {len(extra)}건')

    lines = ['# 확정 결함 제외 반영 (2026-09-23)', '',
             '`tools/apply_defect_exclusions.py` 산출. 규칙으로 확정된 결함만 뺐고 '
             '정답⊂보기(사람 판정 대기)는 그대로 두었다.', '']

    # ---------- 학습 split ----------
    v1 = json.load(open(SPL + 'kd_train_ids_v1.json', encoding='utf-8'))
    ids1 = v1['ids'] if isinstance(v1, dict) and 'ids' in v1 else list(v1)
    drop_tr = sorted((set(ids1) & set(tr_bad)) | (set(ids1) & extra))
    ids2 = sorted(set(ids1) - set(drop_tr))
    obj = {'ids': ids2, 'n': len(ids2),
           'rule': 'kd_train_ids_v1 minus rule-confirmed structural defects: options that are '
                   'identical after NFKC + lowercase + whitespace removal + trailing-decoration '
                   'strip (internal punctuation kept, so 4.0 != 40), or options mangled into '
                   "'?'/U+FFFD. See tools/build_defect_queue.py and tests/test_defect_rules.py.",
           'parent': 'kd_train_ids_v1.json', 'parent_n': len(ids1),
           'dropped': {i: tr_bad.get(i, '정답⊂보기:복수정답') for i in drop_tr},
           'id_sha256': sha(ids2), 'built': '2026-09-23'}
    json.dump(obj, open(SPL + 'train_ids_v2_clean.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f'[학습] {len(ids1)} -> {len(ids2)}  (제외 {len(drop_tr)})')

    # colab_lora_train.py --exclude-csv 로 바로 쓸 수 있는 형태.
    # 기존 중복 제외(train_duplicate_exclusions.csv)와 합쳐 한 파일로 만든다.
    rows_ex = [{'id': i, 'reason': 'structural_defect', 'detail': tr_bad.get(i, 'substring:복수정답')}
               for i in drop_tr]
    dup_p = 'data_meta/train_duplicate_exclusions.csv'
    if os.path.exists(dup_p):
        for _, r in pd.read_csv(dup_p, encoding='utf-8-sig', dtype=str).fillna('').iterrows():
            rows_ex.append({'id': r['id'], 'reason': r.get('reason', 'duplicate'),
                            'detail': r.get('duplicate_of', '')})
    ex = pd.DataFrame(rows_ex).drop_duplicates(subset='id')
    ex.to_csv('data_meta/train_exclusions_v2.csv', index=False, encoding='utf-8-sig')
    print(f'[제외 CSV] data_meta/train_exclusions_v2.csv  {len(ex)}건 '
          f'(구조결함 {len(drop_tr)} + 중복 {len(ex) - len(drop_tr)})')
    lines += ['## 학습 split', '',
              f'- `kd_train_ids_v1.json` {len(ids1)} → **`train_ids_v2_clean.json` {len(ids2)}** '
              f'(제외 {len(drop_tr)})',
              f'- id_sha256 `{obj["id_sha256"][:16]}…`', '']

    # val667 오염 확인
    try:
        v = json.load(open(SPL + 'val667_ids.json', encoding='utf-8'))
        vids = set(v['ids'] if isinstance(v, dict) and 'ids' in v else v)
        lines.append(f'- val667 안의 확정 결함: **{len(vids & set(tr_bad))}건**')
    except Exception:
        pass
    lines.append('')

    # ---------- gold ----------
    g4 = json.load(open(OUT + 'hard_eval_gold_v4.json', encoding='utf-8'))
    g4c = {k: v for k, v in g4.items() if k not in dv_bad and k not in extra}
    json.dump(g4c, open(OUT + 'hard_eval_gold_v5.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, sort_keys=True)

    gH = json.load(open(OUT + 'hard_eval_gold_v4_H229.json', encoding='utf-8'))
    gHc = {k: v for k, v in gH.items() if k not in dv_bad and k not in extra}
    json.dump(gHc, open(OUT + 'hard_eval_gold_v5_H.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, sort_keys=True)
    print(f'[gold] v4 {len(g4)} -> v5 {len(g4c)} | H229 {len(gH)} -> H{len(gHc)}')
    lines += ['## 평가 gold', '',
              f'- `hard_eval_gold_v4.json` {len(g4)} → **`hard_eval_gold_v5.json` {len(g4c)}** '
              f'(제외 {len(g4) - len(g4c)})',
              f'- `hard_eval_gold_v4_H229.json` {len(gH)} → **`hard_eval_gold_v5_H.json` {len(gHc)}** '
              f'(제외 {len(gH) - len(gHc)})', '']

    # ---------- 재채점 델타 ----------
    m35 = load_preds(P + 'runpod_35b_read2/Qwen3.6-35B-A3B_read_devEVAL998_tiles2x2x2_r2both_vllm_predictions.jsonl')
    m397 = load_preds(P + 'nebius_397b/Qwen3.5-397B-A17B-FP8_read_devH331_tiles2x2x2_vllm_predictions.jsonl')
    ce = load_preds(P + 'runpod_9b_kd/ce/hard_predictions.jsonl')
    kd = load_preds(P + 'runpod_9b_kd/kd/hard_predictions.jsonl')

    def ens(i):
        x, y = m35.get(i), m397.get(i)
        if not x or not y:
            return None
        return L[int(np.argmax([(p + q) / 2 for p, q in zip(x['logprobs'], y['logprobs'])]))]

    M = {'앙상블 35B+397B': ens,
         '35B v1': lambda i: (m35.get(i) or {}).get('pred'),
         '397B FP8': lambda i: (m397.get(i) or {}).get('pred'),
         '35B v2': lambda i: ((m35.get(i) or {}).get('v2') or {}).get('pred'),
         '9B KD': lambda i: (kd.get(i) or {}).get('pred'),
         '9B CE': lambda i: (ce.get(i) or {}).get('pred')}

    rows = []
    for n, f in M.items():
        kH = sum(1 for i in gH if f(i) == gH[i])
        kC = sum(1 for i in gHc if f(i) == gHc[i])
        lo, hi = wilson(kC, len(gHc))
        rows.append({'모델': n, f'이전(H{len(gH)})': f'{kH}/{len(gH)}',
                     f'이후(H{len(gHc)})': f'{kC}/{len(gHc)}',
                     '정확도': round(kC / len(gHc), 4), 'CI': f'[{lo:.3f},{hi:.3f}]'})
    T = pd.DataFrame(rows).sort_values('정확도', ascending=False)
    print('\n[재채점] 확정 결함 제외 후')
    print(T.to_string(index=False))

    lines += [f'## 재채점 — H229 → H{len(gHc)}', '', '| ' + ' | '.join(T.columns) + ' |',
              '|' + '---|' * len(T.columns)]
    for _, r in T.iterrows():
        lines.append('| ' + ' | '.join(str(x) for x in r.values) + ' |')
    def nrows(p):
        return len(pd.read_csv(p, encoding='utf-8-sig', dtype=str)) if os.path.exists(p) else '?'

    q = 'reports/review_queue_20260923/'
    lines += ['', '## 제외하지 않은 것', '',
              f'- **정답⊂보기** train {nrows(q + "train_answer_substring.csv")} / '
              f'dev {nrows(q + "dev_answer_substring.csv")} — 사람 판정 대기. '
              '상당수는 정상이라 자동 제외하지 않았다. 판정 후 `--drop-substring <csv>`로 '
              '`복수정답`만 추가 제외한다.',
              '- **동일질문_다른정답** train 784 — 이미지가 다르면 정상이므로 이미지 대조가 필요하다.', '']
    open('reports/defect_exclusion_20260923.md', 'w', encoding='utf-8',
         newline='\n').write('\n'.join(lines) + '\n')
    print('\n리포트: reports/defect_exclusion_20260923.md')


if __name__ == '__main__':
    main()
