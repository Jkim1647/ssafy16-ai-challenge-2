# -*- coding: utf-8 -*-
"""구조적 결함 검수 대기열 생성 (GPU 미사용).

reports/broken_*_structural.csv를 그대로 믿지 않고 train.csv/dev.csv 원본에서 직접
다시 계산한다. 각 행에 판정 근거를 붙여 사람이 바로 확인할 수 있게 한다.

검사 (기존 broken_*_structural.csv의 태그 의미를 재현하되 정의를 명시했다)
  보기중복   : NFKC 정규화 후 공백·문장부호·기호를 제거하면 보기 둘이 같다   -> 규칙 확정
               예: "잘해주는 치과" == "잘 해 주는 치과"
  깨진_보기  : 보기가 '?'·U+FFFD로 깨져 있다(비공백 문자의 절반 이상)        -> 규칙 확정
               예: "??????" — 인코딩 손상이라 정답 여부를 판정할 수 없다
  정답⊂보기  : 정답 텍스트가 다른 보기의 진부분문자열(정규화 후)             -> 사람 확인
               예: 정답 "3,000원", 오답 "13,000원" -> 판독에 따라 둘 다 맞아 보인다

출력 (utf-8-sig. 대회 문항 텍스트를 포함하므로 git에 커밋하지 않는다)
  reports/review_queue_20260923/{train,dev}_confirmed_defects.csv
  reports/review_queue_20260923/{train,dev}_answer_substring.csv
  reports/review_queue_20260923/SUMMARY.md
"""
import json, os, re, unicodedata

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
DATA = 'ssafy-16-2-ai/'
OUTD = 'reports/review_queue_20260923/'
L = list('abcd')


def canon(s):
    """비교용 표준형.

    공백·대소문자 차이와 꼬리 장식만 없앤다. 내부 문장부호는 **남긴다** —
    소수점·마이너스·하이픈은 의미를 바꾸기 때문이다:
      `4.0` vs `40`, `1.5km` vs `15km`, `log(1-y)` vs `-log(1-y)`
    이들을 같다고 보면 멀쩡한 문항을 결함으로 잘못 뺀다.
    """
    s = unicodedata.normalize('NFKC', str(s))
    s = re.sub(r'\s', '', s).lower()
    return s.rstrip('!?~.,·…"\'')


def is_mojibake(s):
    """'?'·U+FFFD가 비공백 문자의 절반 이상이면 인코딩 손상으로 본다."""
    t = re.sub(r'\s', '', str(s))
    if not t:
        return True
    q = t.count('?') + t.count('�')
    return q > 0 and q >= len(t) * 0.5


def dup_pairs(r):
    c = {x: canon(r[x]) for x in L}
    return ';'.join(f'{x}={y}' for i, x in enumerate(L) for y in L[i + 1:]
                    if c[x] and c[x] == c[y])


def broken_opts(r):
    return ';'.join(x for x in L if is_mojibake(r[x]))


def answer_substring(r, gold):
    g = canon(r.get(gold, '')) if gold in L else ''
    if not g:
        return ''
    return ';'.join(f'{gold}⊂{x}' for x in L
                    if x != gold and g != canon(r[x]) and g in canon(r[x]))


def training_ids():
    p = 'data_meta/splits/kd_train_ids_v1.json'
    if not os.path.exists(p):
        return set()
    o = json.load(open(p, encoding='utf-8'))
    return set(o['ids']) if isinstance(o, dict) and 'ids' in o else set(o)


def dev_gold():
    g = {}
    for p in ['runs/dev_validation_v1/hard_eval_gold_v4.json',
              'runs/dev_validation_v1/hard_eval_gold_v3.json']:
        if os.path.exists(p):
            for k, v in json.load(open(p, encoding='utf-8')).items():
                g.setdefault(k, v)
    return g


def defects(df, inuse_ids, inuse_col, gold_of):
    df = df.copy()
    df['_dup'] = df.apply(dup_pairs, axis=1)
    df['_brk'] = df.apply(broken_opts, axis=1)
    df['_sub'] = df.apply(lambda r: answer_substring(r, gold_of(r)), axis=1)
    conf = df[(df['_dup'] != '') | (df['_brk'] != '')].copy()
    conf['결함'] = [';'.join(filter(None, [
        f'보기중복({d})' if d else '', f'깨진_보기({b})' if b else '']))
        for d, b in zip(conf['_dup'], conf['_brk'])]
    conf[inuse_col] = conf['id'].isin(inuse_ids).map({True: 'Y', False: 'N'})
    sub = df[df['_sub'] != ''].copy()
    sub['근거'] = sub['_sub']
    sub[inuse_col] = sub['id'].isin(inuse_ids).map({True: 'Y', False: 'N'})
    sub['판정'] = ''      # 사람이 채움: ok / 복수정답 / 모호
    sub['메모'] = ''
    return conf, sub


def main():
    os.makedirs(OUTD, exist_ok=True)
    rows = []

    tr = pd.read_csv(DATA + 'train.csv', encoding='utf-8-sig', dtype=str).fillna('')
    tids = training_ids()
    tconf, tsub = defects(tr, tids, '학습에_포함', lambda r: str(r['answer']).strip())
    tconf['조치'] = '학습에서 제외'
    tconf[['id', 'path', '결함', '학습에_포함', '조치', 'question'] + L + ['answer']].to_csv(
        OUTD + 'train_confirmed_defects.csv', index=False, encoding='utf-8-sig')
    tsub[['id', 'path', '근거', '학습에_포함', '판정', '메모', 'question'] + L + ['answer']].to_csv(
        OUTD + 'train_answer_substring.csv', index=False, encoding='utf-8-sig')
    rows.append(('train 확정 결함 (규칙, 사람 불필요)', len(tconf), int((tconf['학습에_포함'] == 'Y').sum())))
    rows.append(('train 정답⊂보기 (사람 확인)', len(tsub), int((tsub['학습에_포함'] == 'Y').sum())))

    dv = pd.read_csv(DATA + 'dev.csv', encoding='utf-8-sig', dtype=str).fillna('')
    devg = dev_gold()
    dconf, dsub = defects(dv, set(devg), '평가셋_포함', lambda r: devg.get(r['id'], ''))
    dconf['조치'] = '평가셋에서 제외'
    dconf[['id', 'path', '결함', '평가셋_포함', '조치', 'question'] + L].to_csv(
        OUTD + 'dev_confirmed_defects.csv', index=False, encoding='utf-8-sig')
    dsub['gold'] = dsub['id'].map(devg)
    dsub[['id', 'path', '근거', 'gold', '평가셋_포함', '판정', '메모', 'question'] + L].to_csv(
        OUTD + 'dev_answer_substring.csv', index=False, encoding='utf-8-sig')
    rows.append(('dev 확정 결함 (규칙, 사람 불필요)', len(dconf), int((dconf['평가셋_포함'] == 'Y').sum())))
    rows.append(('dev 정답⊂보기 (gold 있는 문항만)', len(dsub), int((dsub['평가셋_포함'] == 'Y').sum())))

    out = ['# 구조적 결함 검수 대기열 (2026-09-23)', '',
           '`tools/build_defect_queue.py` 산출. `ssafy-16-2-ai/{train,dev}.csv` 원본에서 직접 재계산했다.',
           '**대회 문항 텍스트를 포함하므로 git에 커밋하지 않는다.**', '',
           '| 대기열 | 건수 | 학습·평가에 실제 포함된 수 |', '|---|---|---|']
    out += [f'| {n} | {c} | {u} |' for n, c, u in rows]
    out += ['', '## 판정 규칙', '',
            '- **보기중복** — NFKC 정규화 후 공백·문장부호·기호를 제거하면 보기 둘이 같다. '
            '예: `잘해주는 치과` = `잘 해 주는 치과`. 선택지가 실질 3개 이하가 되므로 확정 결함이다.',
            '- **깨진_보기** — 보기가 `?`·U+FFFD로 깨져 있다(비공백 문자의 절반 이상). '
            '예: `??????`. 인코딩 손상이라 정답 여부를 판정할 수 없다. '
            '팀 합의 규칙("문자가 `?`로 깨져 있으면 rejected")과 같은 판정이다.',
            '- **정답⊂보기** — 정답이 다른 보기의 진부분문자열. 예: 정답 `3,000원`인데 오답에 `13,000원`. '
            '자동 확정하지 않고 사람이 `판정` 열에 `ok` / `복수정답` / `모호`를 적는다.', '',
            '## 처리 순서', '',
            '1. `*_confirmed_defects.csv` — 확인 없이 학습·평가에서 제외',
            '2. `train_answer_substring.csv` — 사람이 `판정`·`메모`를 채운 뒤 `복수정답`만 제외',
            '3. dev 쪽은 gold가 있는 문항에만 ⊂ 검사가 가능하다(dev는 단일 정답 없이 answer1~5)']
    open(OUTD + 'SUMMARY.md', 'w', encoding='utf-8', newline='\n').write('\n'.join(out) + '\n')

    print('출력:', OUTD)
    for n, c, u in rows:
        print(f'  {n:<38s} {c:5d}   (사용중 {u})')


if __name__ == '__main__':
    main()
