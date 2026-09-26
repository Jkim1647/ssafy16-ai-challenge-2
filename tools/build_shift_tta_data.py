# -*- coding: utf-8 -*-
"""선택적 shift TTA 용 부분 데이터셋(dev.csv)을 만든다. GPU 미사용.

왜 한 파일로 묶나
  비용의 대부분이 모델 로드다(397B 는 로드 201초 / 문항 0.56초). train 검증분과 test 적용분을
  따로 돌리면 로드를 두 번 낸다. 하나의 dev.csv 로 묶어 한 번만 로드한다.
  나중에 결합할 때는 id 로 갈라내면 된다(train_* / test_* 접두가 다르다).

usage: python tools/build_shift_tta_data.py --ids runs/shift_tta/train6047/ids.json \
         --ids runs/shift_tta/test/ids.json --src ssafy-16-2-ai --out runs/shift_tta/data
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

COLS = ["id", "path", "question", "a", "b", "c", "d"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", action="append", required=True)
    ap.add_argument("--src", default="ssafy-16-2-ai")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    want = set()
    for p in a.ids:
        want |= set(json.load(open(p, encoding="utf-8")))

    frames = []
    for f in ("train.csv", "dev.csv", "test.csv"):
        fp = os.path.join(a.src, f)
        if os.path.exists(fp):
            frames.append(pd.read_csv(fp, encoding="utf-8-sig", dtype=str, keep_default_na=False)[COLS])
    df = pd.concat(frames, ignore_index=True)
    df = df[df.id.isin(want)].drop_duplicates(subset="id")

    missing = want - set(df.id)
    assert not missing, f"원본 CSV 에서 못 찾은 id {len(missing)}건: {sorted(missing)[:5]}"

    os.makedirs(a.out, exist_ok=True)
    # 기계가 읽는 파일이라 BOM 을 붙이지 않는다(CLAUDE.md 인코딩 규칙 3)
    df.to_csv(os.path.join(a.out, "dev.csv"), index=False, encoding="utf-8")
    print(f"{len(df)}행 저장: {os.path.join(a.out, 'dev.csv')}")
    print("split 별:", {k: int(v) for k, v in df.id.str.split("_").str[0].value_counts().items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
