# -*- coding: utf-8 -*-
"""검수에서 문항이 수정된 항목(rerun_required)을 추론용 데이터 폴더로 만든다.

왜
  gold_v4 287건 중 200건은 검수자가 질문·선택지를 고쳤다. 모델이 본 입력과 달라져
  저장된 예측을 대보면 잘못된 채점이 된다. 다시 추론해야 점수가 생긴다.
  이 항목들은 전부 사람이 확정한 라벨이라 H229의 미검증 192건보다 질이 높고,
  평가셋을 키워 "모든 차이가 잡음"인 현재 상태를 직접 완화한다.

두 가지 함정을 피한다
  1. **검수 패킷 이미지는 축소본이다.** 긴 변 760px로, dev 원본(중앙값 960)보다 작다.
     179건 중 170건이 축소돼 있어 그대로 쓰면 H229(원본 해상도)와 같은 척도가 아니다.
     -> dev 문항은 `ssafy-16-2-ai/dev/` **원본**을 쓴다. 검수는 문항 텍스트만 고쳤고
        이미지는 건드리지 않았으므로 원본을 쓰는 것이 맞다.
  2. **multistep은 대회 데이터가 아니다.** `multistep_50_images/manifest.json`이
     `source_id: external_mtvqa_*`로 외부 MTVQA 벤치마크임을 밝힌다. 언어는 ja/en/zh로
     한국어가 0건이고, 이미지 18장에 문항 50개라 문항끼리 독립적이지도 않다.
     -> 대회 평가셋에 **섞지 않는다.** 별도 폴더로 빼서 외부 probe로만 쓴다.

출력 (대회 문항·이미지 포함 -> git 제외)
  runs/rerun_dev179/{dev.csv, dev/*.jpg, gold.json, manifest.json}     대회 평가셋에 합칠 것
  runs/rerun_multistep21/{dev.csv, dev/*.jpg, gold.json, manifest.json} 외부 probe(선택)
"""
import hashlib, json, os, shutil

import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
SRC = 'review_gold_final_20260923/review_gold_final_20260923/'
MS = 'multistep_50_images/'
L = list('abcd')


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for c in iter(lambda: fh.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def emit(out, sel, img_of, note):
    os.makedirs(out + 'dev', exist_ok=True)
    rows, gold, sizes, missing = [], {}, [], []
    for _, r in sel.iterrows():
        src = img_of(r)
        if not src or not os.path.isfile(src):
            missing.append(r['id'])
            continue
        name = os.path.basename(src)
        shutil.copy2(src, out + 'dev/' + name)
        sizes.append(max(Image.open(src).size))
        rows.append({'id': r['id'], 'path': 'dev/' + name, 'question': r['final_question'],
                     **{c: r['final_' + c] for c in L}})
        gold[r['id']] = r['final_answer']
    if missing:
        print(f'  ⚠ 이미지 없음 {len(missing)}: {missing[:5]}')
    if not rows:
        # multistep 원본(multistep_50_images/)이 없는 PC에서도 dev 쪽은 계속 만들어야 한다.
        print(f'  [skip] {out} — 이미지가 하나도 없어 생성하지 않는다')
        return set()
    df = pd.DataFrame(rows)[['id', 'path', 'question'] + L]
    assert df['id'].is_unique and set(gold.values()) <= set(L)
    # dev split은 엔진이 answer 열을 비우므로 CSV에 정답을 넣지 않는다(유출 방지)
    df.to_csv(out + 'dev.csv', index=False, encoding='utf-8-sig')
    json.dump(gold, open(out + 'gold.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, sort_keys=True)
    import statistics
    man = {'built': '2026-09-23', 'n': len(df), 'note': note,
           'source': SRC + 'audit_all_350.csv',
           'rule': "status == 'include' and evaluation_route_vs_original == 'rerun_required'",
           'image_long_side': {'median': statistics.median(sizes), 'max': max(sizes),
                               'min': min(sizes)},
           'dev_csv_sha256': sha256_file(out + 'dev.csv'),
           'gold_sha256': sha256_file(out + 'gold.json'),
           'gold_letter_dist': pd.Series(list(gold.values())).value_counts().to_dict()}
    json.dump(man, open(out + 'manifest.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    mb = sum(os.path.getsize(out + 'dev/' + f) for f in os.listdir(out + 'dev')) / 1e6
    print(f'  {out}  {len(df)}건 / 이미지 {len(os.listdir(out + "dev"))}장 {mb:.1f}MB '
          f'| 긴 변 중앙값 {man["image_long_side"]["median"]} | 정답 {man["gold_letter_dist"]}')
    return set(df['id'])


def main():
    A = pd.read_csv(SRC + 'audit_all_350.csv', encoding='utf-8-sig', dtype=str).fillna('')
    sel = A[(A['status'] == 'include') &
            (A['evaluation_route_vs_original'] == 'rerun_required')]
    print(f'rerun_required·include {len(sel)}건 → {sel["dataset"].value_counts().to_dict()}')

    print('\n[대회 평가셋에 합칠 것] dev — 원본 해상도 사용')
    dev = emit('runs/rerun_dev179/', sel[sel['dataset'] == 'dev'],
               lambda r: 'ssafy-16-2-ai/dev/' + r['id'],
               'dev 원본 이미지 사용(검수 패킷은 760px 축소본이라 쓰지 않음). H229와 합쳐 평가셋 확장')

    print('\n[외부 probe — 대회 평가셋에 섞지 말 것] multistep')
    ms = emit('runs/rerun_multistep21/', sel[sel['dataset'] == 'multistep'],
              lambda r: MS + 'images/' + r['id'] + '.jpg',
              '외부 MTVQA(ja/en/zh, 한국어 0). 이미지 18장에 문항 50개라 독립 표본 아님. '
              '대회 분포와 다르므로 별도 보고만 한다')

    # 합쳤을 때 평가셋 크기
    H = set(json.load(open('runs/dev_validation_v1/hard_eval_gold_v5_H.json', encoding='utf-8')))
    print(f'\nH229 {len(H)} + rerun_dev {len(dev)} = 합계 {len(H | dev)} '
          f'(겹침 {len(H & dev)})')


if __name__ == '__main__':
    main()
