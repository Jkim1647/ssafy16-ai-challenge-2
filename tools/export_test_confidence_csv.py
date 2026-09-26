# -*- coding: utf-8 -*-
"""test 6,714건에 대해 모델 16개와 제출들의 답·a~d 확률을 CSV 한 개로 정리한다. GPU 미사용(저장된 확률만 쓴다).

사용: python tools/export_test_confidence_csv.py
출력: reports/test_confidence_models_subs_T28_20260926.csv (utf-8-sig, Excel 용)

열: id, 유형, 질문, 보기 a~d, 요약(모델 답 종류 수, T18 과 같은 모델 수),
    제출별 [답, 확신, a, b, c, d], 모델별 [답, 확신, a, b, c, d]
확신 = 고른 보기의 확률. 제출 확률은 결합 함수로 다시 계산하고, 제출 파일과 6,714건 모두 같은 답일 때만 넣는다.
결합 함수가 없거나 재현되지 않는 제출(T1~T3·S1~S4 등)은 답만 넣는다.
T22·T23·T24 결합식은 Kaggle 제출 설명(2026-09-25), T25~T28 은 커밋 메시지(2026-09-26)에서 가져왔다.
다수결 제출(T25·T26·T27)은 확률 대신 득표율(표/5)을 넣는다.
T29 는 제출 답끼리의 규칙이라 확률 없이 답과 출처(T27/T28)만 넣는다.

규칙 4-b: test 문항은 정답을 판단하거나 손으로 고쳐 제출하지 않는다. 모델이 어디서 얼마나 확신하는지 보는 분석용이다.
"""
import csv, io, contextlib, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import export_test_models_xlsx_v2 as V  # noqa: E402  (모델 16개 확률, T12~T20 결합 함수)
from ensemble_submit import load_any  # noqa: E402

R, E, L, pr = V.R, V.E, 'abcd', V.E.pr
E.TEST['35Bsw'] = load_any(V.MODELS['35B_zs_sw'][0])

t18 = V.SUBS['T18'][2]
t21 = lambda D, i: (19 * R.t16(D, i) + t18(D, i)) / 20  # noqa: E731
FN = {k: v[2] for k, v in R.SUBS.items() if v[2]}
FN.update({k: v[2] for k, v in V.SUBS.items()})
FN['T21'] = t21
FN['T22'] = lambda D, i: 0.832 * t21(D, i) + 0.166 * R.t15(D, i) + 0.002 * pr(D, '9Bdora', i)
FN['T23'] = lambda D, i: 0.970 * t21(D, i) + 0.030 * pr(D, '35Bsw', i)
FN['T24'] = lambda D, i: 0.65 * t18(D, i) + 0.35 * R.t15(D, i)
q35 = lambda D, i: pr(D, 'Q35', i)  # noqa: E731
dora = lambda D, i: pr(D, '9Bdora', i)  # noqa: E731
FN['T28'] = lambda D, i: (FN['T22'](D, i) + FN['T9'](D, i) + FN['T14'](D, i) + q35(D, i) + pr(D, 'Gemma', i)) / 5


def vote(voters, tie):
    """5개 다수결. 반환은 a~d 득표율(표/5)에 동점 해소용 아주 작은 값을 더한 것 — argmax 가 제출 답이 된다.
    tie='order': 동점이면 voters 앞쪽 투표자의 답, tie='probsum': 동점이면 투표자 확률 합이 큰 보기."""
    def fn(D, i):
        ps = [f(D, i) for f in voters]
        v = np.zeros(4)
        for p in ps:
            v[int(np.argmax(p))] += 1
        top = [j for j in range(4) if v[j] == v.max()]
        if tie == 'order':
            win = next(int(np.argmax(p)) for p in ps if int(np.argmax(p)) in top)
        else:
            s = sum(ps)
            win = max(top, key=lambda j: s[j])
        v[win] += 1e-6
        return v / len(voters)
    return fn


# 다수결 제출: 확신·a~d 열이 확률이 아니라 득표율이다
VOTE = {
    'T25': vote([FN['T22'], FN['T11'], FN['T19'], q35, dora], 'order'),
    'T26': vote([FN['T22'], FN['T11'], FN['T19'], q35, dora], 'probsum'),
    'T27': vote([FN['T24'], FN['T22'], FN['T19'], q35, dora], 'probsum'),
}
FN.update(VOTE)

# 이름: (제출 파일, Public). 최종 선택 T28·T27 을 앞에 둔다.
SUBS = {
    'T28': ('T28_soft5_t22t9t14q35gemma_test.csv', 0.97706), 'T27': ('T27_vote5_t24t22t19q35dora_test.csv', 0.97467),
    'T25': ('T25_vote5_t22t11t19q35dora_test.csv', 0.97408), 'T26': ('T26_vote5_probsum_tie_test.csv', 0.97408),
    'T18': ('T18_t6_q35half_test.csv', 0.97497), 'T24': ('T24_t18t15_65_35_test.csv', 0.97497),
    'T15': ('T15_t6_q38half_test.csv', 0.97467), 'T6': ('T6_4pmean_test.csv', 0.97438),
    'T8A': ('T8A_comp35fam_test.csv', 0.97438), 'T12': ('T12_t6_posfull_test.csv', 0.97438),
    'T20': ('T20_t14_q35half_test.csv', 0.97438), 'T22': ('T22_A_t21t15dora_test.csv', 0.97438),
    'T7': ('T7_5pmean_test.csv', 0.97408), 'T10': ('T10_w1515_test.csv', 0.97408),
    'T17': ('T17_t6_9bdora05_test.csv', 0.97408), 'T21': ('T21_t16t18_19to1_test.csv', 0.97408),
    'T23': ('T23_B_t21sw_test.csv', 0.97408), 'T9': ('T9_t6t7mix025_test.csv', 0.97378),
    'T11': ('T11_w397g15_test.csv', 0.97378), 'T14': ('T14_t6_gemmarot_test.csv', 0.97378),
    'T16': ('T16_t14half_w_test.csv', 0.97378), 'T4': ('T4_4soft_no9b_test.csv', 0.97348),
    'T5': ('T5_T4_shift35b_test.csv', 0.97348), 'T3': ('T3_5soft_R3_test.csv', 0.97319),
    'T19': ('T19_r3toq35_test.csv', 0.97319), 'T1': ('T1_5soft_gemma_test.csv', 0.97170),
    'T2': ('T2_5hard_gemma_test.csv', 0.97140), 'T13': ('T13_t6_r1v3swap_test.csv', 0.97140),
    'S4': ('S4_27blora_35b_397b_9blora_test.csv', 0.97110), 'S3': ('S3_27blora_35b_397b_test.csv', 0.96961),
    'S2': ('S2_27blora_397b_test.csv', 0.96842), 'S1': ('S1_27blora_test.csv', 0.96187),
}

tids = E.tids
SA, SP = {}, {}
for k, (f, _) in SUBS.items():
    SA[k] = {r['id']: r['answer'] for r in csv.DictReader(open('submissions/' + f, encoding='utf-8-sig'))}
    assert len(SA[k]) == len(tids) and all(i in SA[k] for i in tids), f'{k}: 제출 파일 id 불일치'
    if k in FN:
        p = {i: FN[k](E.TEST, i) for i in tids}
        bad = sum(L[int(np.argmax(p[i]))] != SA[k][i] for i in tids)
        if bad == 0:
            SP[k] = p
        else:
            print(f'  {k}: 결합 함수가 제출 파일과 {bad}문항 달라 답만 넣는다')

# T29 = 제출 답끼리의 규칙(확률 없음): T26 == T27 이고 T27 != T24 이면 T27 답, 아니면 T28 답. Public 0.97616
T29_FILE = 'T29_rule_t26t27_else_t28_test.csv'
T29_SRC = {i: 'T27' if SA['T26'][i] == SA['T27'][i] != SA['T24'][i] else 'T28' for i in tids}
T29 = {i: SA[T29_SRC[i]][i] for i in tids}
t29_file = {r['id']: r['answer'] for r in csv.DictReader(open('submissions/' + T29_FILE, encoding='utf-8-sig'))}
assert all(T29[i] == t29_file[i] for i in tids), 'T29 규칙이 제출 파일과 다르다'

names = list(V.MODELS)
PR, ans = V.PR, V.ans
r4 = lambda v: round(float(v), 4)  # noqa: E731

head = ['id', '유형', '질문', 'a', 'b', 'c', 'd', f'모델 답 종류 수(/{len(names)})', f'T18과 같은 모델 수(/{len(names)})',
        'T29 답', 'T29 출처']
for k in SUBS:
    if k in VOTE:
        head += [f'{k} 답', f'{k} 득표율'] + [f'{k} {x} 득표율' for x in L]
    else:
        head += [f'{k} 답', f'{k} 확신'] + [f'{k} {x}' for x in L]
for n in names:
    head += [f'{n} 답', f'{n} 확신'] + [f'{n} {x}' for x in L]

out = 'reports/test_confidence_models_subs_T29_20260926.csv'
with open(out, 'w', encoding='utf-8-sig', newline='') as fh:  # 사람이 Excel 로 여는 분석 산출물
    w = csv.writer(fh)
    w.writerow(head)
    for i in tids:
        r = V.rows[i]
        line = [i, V.T.get(i, '?'), r['question'], r['a'], r['b'], r['c'], r['d'],
                len({ans[n][i] for n in names}), sum(ans[n][i] == SA['T18'][i] for n in names), T29[i], T29_SRC[i]]
        for k in SUBS:
            if k in SP:
                p = SP[k][i]
                if k in VOTE:
                    p = np.round(p, 1)  # 동점 해소용 1e-6 제거 -> 0, 0.2, ..., 1.0
                line += [SA[k][i], r4(p.max())] + [r4(v) for v in p]
            else:
                line += [SA[k][i]] + [''] * 5
        for n in names:
            p = PR[n][i]
            line += [ans[n][i], r4(p.max())] + [r4(v) for v in p]
        w.writerow(line)

print(f'{out}: {len(tids)}행, {len(head)}열 / 모델 {len(names)}개, 제출 {len(SUBS)}개(확률 있음 {len(SP)}개)')
print('확률 없음(답만):', [k for k in SUBS if k not in SP])
