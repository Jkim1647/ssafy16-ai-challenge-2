# -*- coding: utf-8 -*-
"""H408 추론용 데이터 폴더를 만든다 (⑤ 조건 스크리닝용).

H408 = H229(원본 문항) + rerun179(검수에서 수정된 문항). 문항 텍스트 출처가 다르므로
하나의 dev.csv로 합쳐 둬야 한 번의 추론으로 408건을 돌릴 수 있다.

  - H229 : ssafy-16-2-ai/dev.csv 의 원문 그대로. gold 는 hard_eval_gold_v5_H.json
  - 179  : runs/rerun_dev179/dev.csv 의 수정된 문항. gold 는 runs/rerun_dev179/gold.json
  - 이미지는 둘 다 ssafy-16-2-ai/dev/ 원본을 쓴다 (검수는 문항만 고쳤다)

출력: runs/eval_h408/{dev.csv, gold.json, manifest.json}  (이미지는 복사하지 않는다 —
      추론할 때 dev/ 를 ssafy-16-2-ai/dev 로 링크하거나 그대로 업로드한다)
"""
import hashlib, json, os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
OUT = 'runs/eval_h408/'
L = list('abcd')
COLS = ['id', 'path', 'question'] + L


def main():
    os.makedirs(OUT, exist_ok=True)
    g229 = json.load(open('runs/dev_validation_v1/hard_eval_gold_v5_H.json', encoding='utf-8'))
    g179 = json.load(open('runs/rerun_dev179/gold.json', encoding='utf-8'))
    assert not (set(g229) & set(g179)), '두 셋이 겹친다'

    dev = pd.read_csv('ssafy-16-2-ai/dev.csv', encoding='utf-8-sig', dtype=str).fillna('')
    part229 = dev[dev.id.isin(g229)][COLS].copy()
    assert len(part229) == len(g229), (len(part229), len(g229))

    r179 = pd.read_csv('runs/rerun_dev179/dev.csv', encoding='utf-8-sig', dtype=str).fillna('')
    part179 = r179[COLS].copy()
    # rerun 쪽 path 는 'dev/<파일명>' 이라 원본 dev.csv 와 같은 형식이다 — 그대로 쓴다.

    df = pd.concat([part229, part179], ignore_index=True).sort_values('id').reset_index(drop=True)
    assert df.id.is_unique and len(df) == 408, (len(df), df.id.is_unique)
    miss = [p for p in df.path if not os.path.exists('ssafy-16-2-ai/' + p)]
    assert not miss, f'이미지 누락 {len(miss)}: {miss[:5]}'

    gold = {**g229, **g179}
    # 엔진이 dev split 에서 answer 열을 비우므로 CSV 에는 정답을 넣지 않는다.
    df.to_csv(OUT + 'dev.csv', index=False, encoding='utf-8-sig')
    json.dump(gold, open(OUT + 'gold.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, sort_keys=True)

    def sha(p):
        return hashlib.sha256(open(p, 'rb').read()).hexdigest()

    json.dump({'built': '2026-09-23', 'n': len(df),
               'parts': {'H229(원본 문항)': len(part229), 'rerun179(검수 수정 문항)': len(part179)},
               'gold_sources': ['runs/dev_validation_v1/hard_eval_gold_v5_H.json',
                                'runs/rerun_dev179/gold.json'],
               'images': 'ssafy-16-2-ai/dev (원본)',
               'dev_csv_sha256': sha(OUT + 'dev.csv'), 'gold_sha256': sha(OUT + 'gold.json'),
               'note': '⑤ 조건 스크리닝(판독 recall 강화 등)을 408건 한 번에 돌리기 위한 합본'},
              open(OUT + 'manifest.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    print(f'{OUT}  {len(df)}건 (H229 {len(part229)} + rerun179 {len(part179)})')
    print('정답 분포:', pd.Series([gold[i] for i in df.id]).value_counts().to_dict())


if __name__ == '__main__':
    main()
