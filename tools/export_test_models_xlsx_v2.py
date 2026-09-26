# -*- coding: utf-8 -*-
"""test 6,714건 예측이 모두 있는 모델 16개와 제출 T15~T19 를 T14 와 비교해 Excel 로 정리한다. GPU 미사용(저장된 확률만 쓴다).

tools/export_test_models_xlsx.py(13개 모델, 2026-09-24 밤) 의 갱신판. 원본은 다른 작업이 함께 쓰고 있어 건드리지 않고 새 파일로 만든다.
추가된 모델: Q35_LoRA_shuf(파드 G), 9B_LoRA_tiles 없음 -> 9B_LoRA_base·9B_DoRA(파드 H, 전체 이미지).
추가된 제출: T12(위치 문항 35B 전체 이미지), T13(35B->r1v3), T15(T6+Q38x0.5), T16, T17(T6+9B DoRA x0.5), T18(T6+Q35 x0.5),
T19(R3->Q35), T20(T14+Q35 x0.5). 결합 함수가 제출 파일을 그대로 재현하는지 먼저 검사한다.
일부 문항만 test 예측이 있는 재료(35B 전체 이미지 3,369건, R3·Gemma 보기 회전 600건)는 모델 열에는 넣지 않고,
그 재료를 쓴 제출(T12·T14·T20) 열로 들어간다.
R4(Qwen3.6-27B LoRA + 보기 셔플, Colab)는 test 추론을 하지 않아 넣지 않는다.

시트
  설명        읽는 법과 규칙 4-b 주의
  모델요약    모델별 H408 v2·val667·단독 Public, T14 와 같은 답/다른 답 수, 평균 확신도
  제출요약    제출별 Public·H408 v2·T14 와 다른 답 수
  다른문항    모델 16개·제출 5개 중 하나라도 T14 와 답이 다른 문항 — 질문·보기, 제출별 (답, 확신), 모델별 (답, 확신, T14답 확률)
  전체문항    6,714건 전부 — 질문·보기, T14·제출별 답과 a~d 확률, 모델 16개의 답과 a~d 확률(T14와 다른 답은 노란 칸)

규칙 4-b: test 문항은 정답을 판단하거나 손으로 고치지 않는다. 이 표는 모델끼리 어디서 갈리는지 보는 용도다.
"""
import csv, io, contextlib, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import report_all_submissions as R  # noqa: E402  (T14·T15·T16 결합 함수, H408 v2 구성원)
from ensemble_submit import load_any  # noqa: E402
from ensemble_methods import lsm  # noqa: E402
from openpyxl import Workbook  # noqa: E402
from openpyxl.styles import Alignment, Font, PatternFill  # noqa: E402
from openpyxl.utils import get_column_letter  # noqa: E402

E = R.E
L = 'abcd'
P = 'shared_predictions/'
G, H = P + 'runpod_g_20260924/', P + 'runpod_h_20260924/'
# 이름: (test 파일, 설명, H408 v2 키 또는 H408 파일 또는 None, val667(수 또는 파일), 단독 Public)
MODELS = {
    '397B': (P + 'nebius_397b_test/Qwen3.5-397B-A17B-FP8_read_test_tiles2x2x2_vllm_predictions.jsonl', 'Qwen3.5-397B-A17B 제로샷 판독+타일', '397B', 643, 0.96693),
    'Q35_LoRA_shuf': (G + 'test_predictions.jsonl', 'Qwen3.5-27B LoRA 타일 + 보기 셔플(파드 G, 9번) [새]', G + 'hard_predictions.jsonl', G + 'val_predictions.jsonl', None),
    '35B_r2v3': (P + 'runpod_test_r2v3_20260924/Qwen3.6-35B-A3B_read_testTEST_r2v3_tiles2x2x2_r2v3_vllm_predictions.jsonl', 'Qwen3.6-35B-A3B 제로샷 + 라벨 규칙(r2v3)', '35Br2v3', None, None),
    '35B_r1v3': (P + 'runpod_e_20260924/Qwen3.6-35B-A3B_read_testTEST_r1v3_tiles2x2x2_r1v3_vllm_predictions.jsonl', 'Qwen3.6-35B-A3B 제로샷 + 구조화 판독(r1v3)', '35Br1v3', None, None),
    '35B': (P + 'colab_ai2_35b_out/Qwen3.6-35B-A3B_read_test_tiles2x2x2_predictions.jsonl', 'Qwen3.6-35B-A3B 제로샷 판독+타일', '35B', 644, 0.96455),
    'Gemma': (P + 'runpod_gemma4_31b/gemma_h408_test/test_predictions.jsonl', 'Gemma4-31B LoRA 타일', 'Gemma', 647, None),
    'R3': (P + 'runpod_27b_R3/R3_h408/test_predictions.jsonl', 'Qwen3.6-27B LoRA 타일(R3)', 'R3', 646, None),
    'Q38': (P + 'runpod_q38_20260924/Qwen3.8-27B_read_testTEST_tiles2x2x2_vllm_predictions.jsonl', 'Qwen3.8-27B 제로샷 판독+타일(트랙 A)', 'Q38', 636, None),
    '27B_LoRA_full': (P + 'colab_ai2_35b_out/lora/27b_lora_r8/test_predictions.jsonl', 'Qwen3.6-27B LoRA 전체 이미지', 341, 646, 0.96187),
    '9B_LoRA_base': (H + '9b_base/test_predictions.jsonl', 'Qwen3.5-9B LoRA 전체 이미지(파드 H, 10번 기준) [새]', H + '9b_base/hard_predictions.jsonl', H + '9b_base/val_predictions.jsonl', None),
    '9B_DoRA': (H + '9b_dora/test_predictions.jsonl', 'Qwen3.5-9B DoRA 전체 이미지(파드 H, 10번) [새]', H + '9b_dora/hard_predictions.jsonl', H + '9b_dora/val_predictions.jsonl', None),
    '9B_LoRA': (P + 'colab_ai2_35b_out/lora/Qwen3.5-9B_lora_r8_lr0.0001_ep1/test_predictions.jsonl', 'Qwen3.5-9B LoRA 전체 이미지(Colab)', 321, None, None),
    '27B_zs': (P + 'colab_ai2_35b_out/Qwen3.6-27B_ours_test_predictions.jsonl', 'Qwen3.6-27B 제로샷(우리 프롬프트, 전체 이미지)', None, 637, None),
    '35B_zs_ours': (P + 'colab_ai2_35b_out/Qwen3.6-35B-A3B_ours_test_predictions.jsonl', 'Qwen3.6-35B-A3B 제로샷(우리 프롬프트)', None, 630, None),
    '35B_zs_sw': (P + '20260921/test_35b_seokwoong.jsonl', 'Qwen3.6-35B-A3B 제로샷(석웅 프롬프트)', None, None, 0.95144),
    '9B_zs': (P + '20260921/test_9b_ours.jsonl', 'Qwen3.5-9B NF4 제로샷', None, None, 0.94221),
}
BASE = ['R3', '35B', '397B', 'Gemma']
pr = E.pr
E.M['Q35'], E.TEST['Q35'] = load_any(G + 'hard_predictions.jsonl'), load_any(G + 'test_predictions.jsonl')
E.M['9Bdora'], E.TEST['9Bdora'] = load_any(H + '9b_dora/hard_predictions.jsonl'), load_any(H + '9b_dora/test_predictions.jsonl')


def plus_half(extra):
    return lambda D, i: (sum(pr(D, n, i) for n in BASE) + 0.5 * pr(D, extra, i)) / 4.5


# 이름: (제출 파일, Public, 결합 함수, 한 줄 설명)
SUBS = {
    'T12': ('T12_t6_posfull_test.csv', 0.97438, R.SUBS['T12'][2], 'T6 + 위치 키워드 문항만 35B 전체 이미지'),
    'T13': ('T13_t6_r1v3swap_test.csv', 0.97140, R.SUBS['T13'][2], 'T6 에서 35B -> r1v3(구조화 판독)'),
    'T15': ('T15_t6_q38half_test.csv', 0.97467, R.t15, 'T6 + Qwen3.8-27B 제로샷 x0.5'),
    'T16': ('T16_t14half_w_test.csv', 0.97378, R.t16, '0.5·T14 + 397B·Gemma 1/6 + 35B·R3 1/12'),
    'T17': ('T17_t6_9bdora05_test.csv', 0.97408, plus_half('9Bdora'), 'T6 + 9B DoRA x0.5'),
    'T18': ('T18_t6_q35half_test.csv', 0.97497, plus_half('Q35'), 'T6 + Qwen3.5-27B LoRA 셔플 x0.5 (팀 최고)'),
    'T19': ('T19_r3toq35_test.csv', 0.97319, lambda D, i: np.mean([pr(D, n, i) for n in ['Q35', '35B', '397B', 'Gemma']], 0),
            'T6 에서 R3 -> Qwen3.5-27B LoRA 셔플'),
    # T14(Gemma 보기 회전 평균) 위에 T18 과 같은 Q35 x0.5 — 사용자 제안(2026-09-25)
    'T20': ('T20_t14_q35half_test.csv', 0.97438,
            lambda D, i: (pr(D, 'R3', i) + pr(D, '35B', i) + pr(D, '397B', i) + R.SH.pavg(D, 'Gemma', i) + 0.5 * pr(D, 'Q35', i)) / 4.5,
            'T14 + Qwen3.5-27B LoRA 셔플 x0.5'),
}
SN = list(SUBS)

tids = E.tids
gold_train = {r['id']: r['answer'] for r in csv.DictReader(open('ssafy-16-2-ai/train.csv', encoding='utf-8-sig'))}
PR = {}
for n, (f, *_rest) in MODELS.items():
    d = load_any(f)
    assert all(i in d for i in tids), f'{n}: test 누락'
    PR[n] = {i: np.exp(lsm(d[i]['logprobs'])) for i in tids}
names = list(MODELS)
t14p = {i: R.SUBS['T14'][2](E.TEST, i) for i in tids}
t14a = {i: R.TP['T14'][i] for i in tids}
assert all(L[int(np.argmax(t14p[i]))] == t14a[i] for i in tids), 'T14 결합 함수가 제출 파일과 다르다'
SP, SA = {}, {}
for k, (f, _, fn, _) in SUBS.items():
    SA[k] = {r['id']: r['answer'] for r in csv.DictReader(open('submissions/' + f, encoding='utf-8-sig'))}
    SP[k] = {i: fn(E.TEST, i) for i in tids}
    bad = sum(L[int(np.argmax(SP[k][i]))] != SA[k][i] for i in tids)
    assert bad == 0, f'{k} 결합 함수가 제출 파일과 {bad}문항 다르다'
ans = {n: {i: L[int(np.argmax(PR[n][i]))] for i in tids} for n in names}
rows = R.rows
T = E.T
g, ids = E.g, E.ids


def h408(key):
    if key is None or isinstance(key, int):
        return key
    D = {'_': load_any(key)} if key.endswith('.jsonl') else E.M
    k = '_' if key.endswith('.jsonl') else key
    return sum(L[int(np.argmax(pr(D, k, i)))] == g[i] for i in ids)


def val667(v):
    if v is None or isinstance(v, int):
        return v
    d = load_any(v)
    return sum(L[int(np.argmax(np.array(d[i]['logprobs'])))] == gold_train[i] for i in d)


wb = Workbook()
bold = Font(bold=True)
yellow = PatternFill('solid', fgColor='FFF2CC')
green = PatternFill('solid', fgColor='E2EFDA')
blue = PatternFill('solid', fgColor='DDEBF7')
wrap = Alignment(wrap_text=True, vertical='top')

ws = wb.active
ws.title = '설명'
for line in [
    f'test 예측이 6,714건 모두 있는 모델 {len(names)}개와 제출 {", ".join(SN)} 을 T14(제출)와 비교한 표입니다. (이전 판: 13개 모델)',
    '새로 추가: 모델 Q35_LoRA_shuf·9B_LoRA_base·9B_DoRA(설명에 [새]), 제출 T12·T13·T15~T20. R4(27B LoRA 셔플, Colab)는 test 추론이 없어 빠졌습니다.',
    '일부 문항만 test 예측이 있는 재료(35B 전체 이미지 3,369건, Gemma 보기 회전 600건)는 그 재료를 쓴 제출 T12·T14·T20 열로 보면 됩니다.',
    '확신도 = 그 모델의 a~d 확률(로그확률을 softmax) 중 고른 보기의 확률. 1에 가까울수록 확신.',
    '전체문항 시트: test 6,714건 전부. 제출·모델마다 [답, a, b, c, d 확률] 5칸, T14와 다른 답은 노란 칸.',
    '다른문항 시트: 모델·제출 중 하나라도 T14 와 다른 답을 고른 문항. 노란 칸 = T14 와 다른 답, 초록 칸 = T14 와 같은 답.',
    '"T14답 확률" = 그 모델이 T14 가 고른 보기에 준 확률. 낮을수록 그 모델은 T14 답에 반대한다.',
    'T14 = T6(R3·35B·397B·Gemma 확률 평균)에서 Gemma 를 보기 회전 4개 평균으로 바꾼 것(Public 0.97378).',
    'T18 = T6 + Qwen3.5-27B LoRA(보기 셔플) 가중 0.5 — Public 0.97497 로 현재 팀 최고.',
    'T20 = T14 + 같은 Q35 가중 0.5 — Public 0.97438 (T18 -2, T14 +2). 바탕 T6->T14 차이와 Q35 효과가 각각 같은 폭으로 나왔다.',
    '제출요약의 H408 v2 = 사용자 정답 검토 반영 401문항. 결합 함수는 제출 파일과 6,714건 모두 같은 답인지 검사했다.',
    '규칙 4-b: test 문항은 정답을 판단하거나 손으로 고쳐 제출하지 않는다. 모델이 어디서 갈리는지 보는 분석용이다.',
    '만든 도구: tools/export_test_models_xlsx_v2.py (저장된 예측만 사용, GPU 없음).']:
    ws.append([line])
ws.column_dimensions['A'].width = 130

ws = wb.create_sheet('모델요약')
ws.append(['모델', '설명', 'H408 v2(401)', 'val667', '단독 Public', 'T14와 같은 답', 'T14와 다른 답', '평균 확신도(전체)',
           '평균 확신도(T14와 다른 문항)', 'T14답에 준 평균 확률(다른 문항)'])
for c in ws[1]:
    c.font = bold
for n in names:
    f, desc, hk, v, pub = MODELS[n]
    diff = [i for i in tids if ans[n][i] != t14a[i]]
    conf = [PR[n][i].max() for i in tids]
    ws.append([n, desc, h408(hk), val667(v), pub, len(tids) - len(diff), len(diff), round(float(np.mean(conf)), 4),
               round(float(np.mean([PR[n][i].max() for i in diff])), 4) if diff else None,
               round(float(np.mean([PR[n][i][L.index(t14a[i])] for i in diff])), 4) if diff else None])
    if '[새]' in desc:
        for c in ws[ws.max_row]:
            c.fill = blue
for j, w in enumerate([16, 50, 12, 9, 11, 13, 13, 16, 22, 24], 1):
    ws.column_dimensions[get_column_letter(j)].width = w

ws = wb.create_sheet('제출요약')
ws.append(['제출', '설명', 'Public', 'H408 v2(401)', 'T14와 다른 답'])
for c in ws[1]:
    c.font = bold
ws.append(['T14', R.SUBS['T14'][3], R.SUBS['T14'][1], sum(L[int(np.argmax(R.SUBS['T14'][2](E.M, i)))] == g[i] for i in ids), 0])
for k in SN:
    f, pub, fn, desc = SUBS[k]
    ws.append([k, desc, pub, sum(L[int(np.argmax(fn(E.M, i)))] == g[i] for i in ids), sum(SA[k][i] != t14a[i] for i in tids)])
for j, w in enumerate([8, 50, 10, 13, 13], 1):
    ws.column_dimensions[get_column_letter(j)].width = w

ws = wb.create_sheet('다른문항')
diff_ids = [i for i in tids if any(SA[k][i] != t14a[i] for k in SN) or any(ans[n][i] != t14a[i] for n in names)]
diff_ids.sort(key=lambda i: (sum(ans[n][i] == t14a[i] for n in names), i))
head = ['id', '유형', '질문', 'a', 'b', 'c', 'd', 'T14 답', 'T14 확신']
for k in SN:
    head += [f'{k} 답', f'{k} 확신']
head += [f'T14와 같은 모델 수(/{len(names)})', '가장 많이 나온 다른 답']
for n in names:
    head += [f'{n} 답', f'{n} 확신', f'{n} T14답 확률']
ws.append(head)
for c in ws[1]:
    c.font = bold
    c.alignment = wrap
for i in diff_ids:
    r = rows[i]
    same = sum(ans[n][i] == t14a[i] for n in names)
    other = collections.Counter(ans[n][i] for n in names if ans[n][i] != t14a[i]).most_common(1)
    line = [i, T.get(i, '?'), r['question'], r['a'], r['b'], r['c'], r['d'], t14a[i], round(float(t14p[i].max()), 3)]
    for k in SN:
        line += [SA[k][i], round(float(SP[k][i].max()), 3)]
    line += [same, f'{other[0][0]} ({other[0][1]}개)' if other else '']
    for n in names:
        line += [ans[n][i], round(float(PR[n][i].max()), 3), round(float(PR[n][i][L.index(t14a[i])]), 3)]
    ws.append(line)
    rr = ws.max_row
    for s, k in enumerate(SN):
        if SA[k][i] != t14a[i]:
            ws.cell(row=rr, column=10 + 2 * s).fill = yellow
    for k, n in enumerate(names):
        cell = ws.cell(row=rr, column=len(head) - 3 * (len(names) - k) + 1)
        cell.fill = yellow if ans[n][i] != t14a[i] else green
ws.freeze_panes = 'D2'
for j, w in enumerate([15, 11, 50, 18, 18, 18, 18, 8, 9] + [8, 9] * len(SN) + [12, 14] + [8, 8, 10] * len(names), 1):
    ws.column_dimensions[get_column_letter(j)].width = w
ws.auto_filter.ref = ws.dimensions

ws = wb.create_sheet('전체문항')
head = ['id', '유형', '질문', 'a', 'b', 'c', 'd', 'T14 답', 'T14 a', 'T14 b', 'T14 c', 'T14 d']
for k in SN:
    head += [f'{k} 답'] + [f'{k} {x}' for x in L]
head += [f'T14와 같은 모델 수(/{len(names)})']
m0 = len(head) + 1  # 첫 모델 답 열
for n in names:
    head += [f'{n} 답'] + [f'{n} {x}' for x in L]
ws.append(head)
for c in ws[1]:
    c.font = bold
    c.alignment = wrap
for i in tids:
    r = rows[i]
    line = [i, T.get(i, '?'), r['question'], r['a'], r['b'], r['c'], r['d'], t14a[i]] + [round(float(v), 3) for v in t14p[i]]
    for k in SN:
        line += [SA[k][i]] + [round(float(v), 3) for v in SP[k][i]]
    line += [sum(ans[n][i] == t14a[i] for n in names)]
    for n in names:
        line += [ans[n][i]] + [round(float(v), 3) for v in PR[n][i]]
    ws.append(line)
    rr = ws.max_row
    for s, k in enumerate(SN):
        if SA[k][i] != t14a[i]:
            ws.cell(row=rr, column=13 + 5 * s).fill = yellow
    for k, n in enumerate(names):
        if ans[n][i] != t14a[i]:
            ws.cell(row=rr, column=m0 + 5 * k).fill = yellow
ws.freeze_panes = 'D2'
for j, w in enumerate([15, 11, 50, 18, 18, 18, 18, 8, 7, 7, 7, 7] + [8, 7, 7, 7, 7] * len(SN) + [12] + [7, 6, 6, 6, 6] * len(names), 1):
    ws.column_dimensions[get_column_letter(j)].width = w
ws.auto_filter.ref = ws.dimensions

out = f'reports/test_{len(names)}models_vs_T14_20260925.xlsx'
wb.save(out)
print(f'{out}: 다른문항 {len(diff_ids)} / 6714')
for k in SN:
    print(f'  제출 {k}: T14와 다른 답 {sum(SA[k][i] != t14a[i] for i in tids)}')
for n in names:
    print(f'  {n:14s} T14와 다른 답 {sum(ans[n][i] != t14a[i] for i in tids)}')
