# -*- coding: utf-8 -*-
"""스크리닝 결과 채점 — train 2,000 과 dev(H408) 을 **따로** 본다. GPU 미사용.

왜 두 부분을 나눠 보나
  오늘 27B LoRA 에서 확인한 것: **train 출처 평가셋은 train 을 활용한 기법을 과대평가한다.**
  val667 에서 1위(0.9685)였던 모델이 Public 에서 꼴찌(0.96187, 397B 단독보다 17문항 아래)였다.
  few-shot 은 train 에서 예시를 뽑으므로 정확히 같은 편향을 받는다 — train 2,000 에서
  좋아 보여도 그게 test 로 간다는 보장이 없다.

  그래서 채택 규칙을 이렇게 둔다.
    train2000 개선 + dev408 악화 없음  -> 채택 후보 (전체 6,047 로 확인)
    train2000 만 개선, dev408 악화      -> **기각**. train 특성을 외운 것이다
    dev408 만 개선                      -> 보류. n=408 은 1문항이 0.245%p 라 잡음이 크다

usage: python tools/score_screen.py <out_dir> [--gold runs/screen/gold.json]
  out_dir 안의 *_predictions.jsonl 을 전부 읽어 설정별로 비교한다.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import sys

import numpy as np


def load(p):
    d = {}
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                o = json.loads(line)
                d[o["id"]] = o
    return d


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p, den = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - h) / den, (c + h) / den)


def paired(a, b, B=10000, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(B, len(d)))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return d.mean() * 100, lo * 100, hi * 100, int((d != 0).sum())


def main() -> int:
    out_dir = sys.argv[1]
    gold_path = sys.argv[sys.argv.index("--gold") + 1] if "--gold" in sys.argv else "runs/screen/gold.json"
    gold = json.load(open(gold_path, encoding="utf-8"))
    # 표본 크기는 실제로 채점된 문항 수로 붙인다(스크리닝셋을 2,408 -> 1,408 로 줄인 적이 있다)
    parts = {"train": [i for i in gold if i.startswith("train")],
             "dev": [i for i in gold if i.startswith("dev")]}

    runs = {}
    for p in sorted(glob.glob(os.path.join(out_dir, "*_predictions.jsonl"))):
        m = re.search(r"SCREEN_([A-Za-z0-9]+)", os.path.basename(p))
        runs[m.group(1) if m else os.path.basename(p)[:28]] = load(p)
    if not runs:
        print(f"{out_dir} 에 *_predictions.jsonl 이 없다")
        return 1

    md = ["# 미시험 레버 스크리닝 결과", "",
          "`tools/score_screen.py`. GPU 미사용.", "",
          "> train2000 은 train 출처라 train 을 쓰는 기법(few-shot)을 과대평가한다. "
          "dev408 에서 악화가 없어야 채택 후보로 본다 — 오늘 27B LoRA 가 val667 1위인데 "
          "Public 꼴찌였던 것과 같은 함정이다.", ""]
    for part, ids in parts.items():
        ids = [i for i in ids if all(i in r for r in runs.values())]
        if not ids:
            continue
        md += [f"## {part}{len(ids)} (n={len(ids)})", "", "| 설정 | 정답 | 정확도 | 95% CI |", "|---|---|---|---|"]
        vec = {}
        for name, r in runs.items():
            vec[name] = [int(r[i]["pred"] == gold[i]) for i in ids]
        for name in sorted(vec, key=lambda x: -sum(vec[x])):
            k = sum(vec[name])
            lo, hi = wilson(k, len(ids))
            md.append(f"| {name} | {k} | {k/len(ids):.4f} | {lo:.4f}–{hi:.4f} |")
        base = "base" if "base" in vec else max(vec, key=lambda x: sum(vec[x]))
        md += ["", f"- 1문항 = {100/len(ids):.3f}%p", "",
               f"### {base} 대비 (paired bootstrap 10,000회)", "",
               "| 비교 | 평균차 | 95% CI | 예측 변경 | 판정 |", "|---|---|---|---|---|"]
        for name in vec:
            if name == base:
                continue
            d, lo, hi, ch = paired(vec[name], vec[base])
            md.append(f"| {name} − {base} | {d:+.2f}%p | {lo:+.2f}–{hi:+.2f} | {ch} | "
                      f"{'**유의**' if (lo > 0 or hi < 0) else '잡음'} |")
        md += [""]

    text = "\n".join(md) + "\n"
    open("reports/screen_result.md", "w", encoding="utf-8", newline="\n").write(text)
    try:
        print(text)
    except UnicodeEncodeError:   # Windows cp949 콘솔
        pass
    print("저장: reports/screen_result.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
