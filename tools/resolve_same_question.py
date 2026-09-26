# -*- coding: utf-8 -*-
"""동일질문_다른정답 대기열을 이미지 대조로 기계 판정한다(사람 검수 불필요분 분리).

왜
  train 784건은 "질문이 같은데 정답이 다르다"로 묶인 것이고, 앞선 기록대로
  **이미지가 다르면 정상**이다("이 간판의 전화번호는?" 같은 질문은 사진마다 답이 다르다).
  그래서 질문만으로는 결함이 아니고, 같은 질문 + 같은 이미지인데 정답이 다를 때만 모순이다.

판정 (이미지 동일성은 파일 바이트가 아니라 픽셀·지각 해시로 본다)
  - 같은 질문 + 같은 이미지 + 다른 정답  -> confirmed_conflict  (확정 결함, 사람 불필요)
  - 같은 질문 + 같은 이미지 + 같은 정답  -> exact_duplicate     (중복 행, 학습 중복 가중만 유발)
  - 같은 질문 + 다른 이미지              -> normal              (대기열에서 제거)
  dhash가 같고 pixel_sha256이 다르면 near_duplicate로 따로 표시한다(리사이즈·재압축 흔적).

출력
  reports/review_queue_20260923/train_same_question_resolved.csv  (대회 문항 텍스트 포함 -> git 제외)
  data_meta/train_exclusions_v3.csv                                (id·사유만 -> git 추적 가능)
  reports/same_question_resolution_20260923.md                     (집계, 문항 텍스트 없음)
usage: python tools/resolve_same_question.py
"""
import json
import os
import re
import unicodedata
from collections import defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
OUTD = 'reports/review_queue_20260923/'
L = list('abcd')


def canon(s):
    """공백·대소문자·꼬리 장식만 무시한다. 숫자·기호는 보존(2026-09-23 정규화 버그 재발 방지)."""
    s = unicodedata.normalize('NFKC', str(s)).strip().lower()
    return re.sub(r'\s+', ' ', s).rstrip(' .?!:;')


def main():
    inv = pd.read_csv('runs/dev_validation_v1/image_inventory.csv', encoding='utf-8-sig', dtype=str)
    px = dict(zip(inv.id, inv.pixel_sha256))
    dh = dict(zip(inv.id, inv.dhash))
    tr = pd.read_csv('ssafy-16-2-ai/train.csv', encoding='utf-8-sig', dtype=str, keep_default_na=False)
    train_ids = set(json.load(open('data_meta/splits/train_ids_v2_clean.json', encoding='utf-8-sig'))['ids'])

    groups = defaultdict(list)
    for r in tr.itertuples():
        groups[canon(r.question)].append(r)

    rows, excl = [], {}
    stat = defaultdict(int)
    for q, members in groups.items():
        if len(members) < 2:
            continue
        by_img = defaultdict(list)
        for r in members:
            by_img[px.get(r.id)].append(r)
        for key, same_img in by_img.items():
            answers = {str(r.answer).strip().lower() for r in same_img}
            if len(same_img) < 2:
                continue
            kind = 'confirmed_conflict' if len(answers) > 1 else 'exact_duplicate'
            for r in same_img:
                stat[kind] += 1
                rows.append({'id': r.id, '판정': kind, '그룹크기': len(same_img), '정답들': '|'.join(sorted(answers)),
                             '학습에_포함': r.id in train_ids, 'path': r.path, 'question': r.question,
                             **{c: getattr(r, c) for c in L}, 'answer': r.answer})
                if kind == 'confirmed_conflict' and r.id in train_ids:
                    excl[r.id] = 'same_question_same_image_conflicting_answer'
        # 픽셀은 다르지만 지각 해시가 같은 쌍(리사이즈·재압축) — 같은 사진일 가능성
        seen = defaultdict(list)
        for r in members:
            seen[dh.get(r.id)].append(r)
        for key, grp in seen.items():
            if len(grp) < 2 or len({px.get(r.id) for r in grp}) < 2:
                continue
            if len({str(r.answer).strip().lower() for r in grp}) > 1:
                for r in grp:
                    stat['near_dup_conflict'] += 1
                    rows.append({'id': r.id, '판정': 'near_dup_conflict', '그룹크기': len(grp),
                                 '정답들': '|'.join(sorted({str(x.answer).strip().lower() for x in grp})),
                                 '학습에_포함': r.id in train_ids, 'path': r.path, 'question': r.question,
                                 **{c: getattr(r, c) for c in L}, 'answer': r.answer})

    same_q_rows = sum(len(m) for m in groups.values() if len(m) > 1)
    flagged = {r['id'] for r in rows}
    os.makedirs(OUTD, exist_ok=True)
    df = pd.DataFrame(rows).drop_duplicates(subset=['id', '판정'])
    df.to_csv(OUTD + 'train_same_question_resolved.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([{'id': i, 'reason': v} for i, v in sorted(excl.items())]).to_csv(
        'data_meta/train_exclusions_v3.csv', index=False, encoding='utf-8-sig')

    md = [
        '# 동일질문_다른정답 대기열 기계 판정 (2026-09-23)',
        '',
        '`tools/resolve_same_question.py`. 질문이 같아도 **이미지가 다르면 정상**이므로, 같은 질문 안에서 이미지까지 같은 경우만 모순으로 본다.',
        '문항 텍스트가 든 표는 `reports/review_queue_20260923/train_same_question_resolved.csv`(git 제외)에 있고 여기에는 집계만 적는다.',
        '',
        f'- 질문이 2건 이상 겹치는 train 행: **{same_q_rows}**건',
        f'- 이 중 이미지까지 같아 판정 대상이 된 행: **{len(flagged)}**건',
        '',
        '| 판정 | 행 수 | 뜻 | 조치 |',
        '|---|---|---|---|',
        f'| confirmed_conflict | {stat["confirmed_conflict"]} | 같은 질문·같은 이미지인데 정답이 다르다 | 학습 제외(`data_meta/train_exclusions_v3.csv`) |',
        f'| exact_duplicate | {stat["exact_duplicate"]} | 같은 질문·같은 이미지·같은 정답 | 결함 아님. 중복 가중만 주의 |',
        f'| near_dup_conflict | {stat["near_dup_conflict"]} | dhash는 같고 픽셀은 다른데 정답이 다르다 | 사람 확인 필요(리사이즈본일 수 있음) |',
        '',
        f'- 학습(5,975)에 실제 포함된 confirmed_conflict: **{len(excl)}**건 → `--exclude-csv data_meta/train_exclusions_v3.csv`',
        '- 나머지 동일질문 행은 이미지가 달라 **사람 검수가 필요 없다**. 대기열에서 내린다.',
    ]
    open('reports/same_question_resolution_20260923.md', 'w', encoding='utf-8', newline='\n').write('\n'.join(md) + '\n')
    print('\n'.join(md[4:]))


if __name__ == '__main__':
    main()
