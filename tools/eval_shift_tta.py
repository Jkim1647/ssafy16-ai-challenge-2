# -*- coding: utf-8 -*-
"""선택적 선택지 순환(shift) TTA 결합·채점. GPU 미사용.

무엇을 하나
  `tools/select_low_margin.py` 가 고른 저마진 문항에 대해서만 shift1~3 재추론 결과를 받아
  결합 규칙 여러 개를 같은 자로 비교한다. 선택되지 않은 문항은 base 를 그대로 쓴다.

결합 규칙 (어느 것이 맞는지 미정이라 전부 비교한다)
  base   재추론 안 함(기준선)
  mean4  base + s1~3 의 a~d 로그확률을 log-softmax 후 평균   ← val667 에서 +5 를 낸 방식.
         정규화 확률의 **기하평균**에 해당한다
  pmean4 같은 네 view 의 **확률 산술평균**. mean4 와 다른 결합이다 — 기하평균은 한 view 가
         0 에 가까운 확률을 주면 그 보기를 강하게 눌러 버리고, 산술평균은 덜 누른다.
         어느 쪽이 맞는지는 데이터가 정한다(팀원 제안, reports/tta_ensemble_strategy_20260923.md)
  mean3  s1~3 만 평균(base 를 버린다)                        ← 다른 참가자(38위) "교체" 방식.
         그쪽 n=200 에서 교체 97.0% > 평균 96.0% 였다. 우리 데이터에서는 미확인
  vote   네 순서의 argmax 다수결, 동률이면 base

  mean3 과 mean4 가 갈리는 이유가 있다. base 는 "모델이 원래 순서에서 헷갈린 그 답"이라
  위치 편향을 그대로 담고 있다. 편향을 지우려면 빼야 한다는 주장과, 4개 표본 중 하나를
  버리면 분산이 커진다는 주장이 둘 다 성립한다. 재어 보는 수밖에 없다.

왜 저마진만인가 — `tools/select_low_margin.py` 의 주석 참조(고마진 구간 Δ는 정확히 0이었다).

산출
  --emit-dir 를 주면 모델별로 결합된 predictions.jsonl 을 쓴다(원래 a~d 순서).
  그대로 tools/ensemble_submit.py 에 넘기면 제출 CSV 가 나온다.

사용
  python tools/eval_shift_tta.py --ids runs/shift_tta/train6047/ids.json \
    --model "35B=base.jsonl,s1.jsonl,s2.jsonl,s3.jsonl" \
    --model "397B=base397.jsonl,s1_397.jsonl,s2_397.jsonl,s3_397.jsonl"
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

LETTERS = "abcd"


def load(path):
    d = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                o = json.loads(line)
                d[o["id"]] = o
    return d


def lsm(v):
    a = np.clip(np.array(v, float), -1e4, None)
    m = a.max()
    e = np.exp(a - m)
    return np.log(np.maximum(e / e.sum(), 1e-300))


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


def combine(rule, views):
    """views[0] 이 base, 나머지가 shift1~3 의 a~d 로그확률(원래 순서로 되돌려진 값)."""
    if rule == "base" or len(views) == 1:
        return views[0]
    if rule == "mean4":
        return np.mean(views, axis=0)
    if rule == "pmean4":
        # 확률 산술평균. 뒤에서 argmax 만 쓰므로 로그로 돌려 줘도 순서는 같다
        return np.log(np.maximum(np.mean([np.exp(v) for v in views], axis=0), 1e-300))
    if rule == "mean3":
        return np.mean(views[1:], axis=0)
    if rule == "vote":
        picks = [int(np.argmax(v)) for v in views]
        cnt = np.bincount(picks, minlength=4)
        best = int(cnt.argmax())
        if (cnt == cnt[best]).sum() > 1:       # 동률이면 base
            return views[0]
        out = np.full(4, -20.0)
        out[best] = 0.0
        return out
    raise KeyError(rule)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    help='"이름=base.jsonl[,s1.jsonl,s2.jsonl,s3.jsonl]". shift 파일은 없어도 된다')
    ap.add_argument("--ids", default=None, help="TTA 를 적용할 id 목록 json. 없으면 전 문항")
    ap.add_argument("--gold-json", default=None, help="없으면 base 의 gold 필드를 쓴다")
    ap.add_argument("--rules", default="base,mean4,pmean4,mean3,vote")
    ap.add_argument("--weights", default=None, help="모델 결합 가중치. 기본 균등")
    ap.add_argument("--emit-dir", default=None)
    ap.add_argument("--out", default=None, help="마크다운 저장 경로")
    a = ap.parse_args()

    models = {}
    for spec in a.model:
        name, paths = spec.split("=", 1)
        files = [p for p in paths.split(",") if p]
        for p in files:
            assert os.path.exists(p), p
        models[name] = [load(p) for p in files]

    ids_pool = sorted(set.intersection(*[set(v[0]) for v in models.values()]))
    sel = set(json.load(open(a.ids, encoding="utf-8"))) & set(ids_pool) if a.ids else set(ids_pool)
    if a.gold_json:
        gold = json.load(open(a.gold_json, encoding="utf-8"))
    else:
        gold = {i: models[next(iter(models))][0][i].get("gold") for i in ids_pool}
    ids = [i for i in ids_pool if gold.get(i)]
    assert ids, "gold 가 있는 공통 id 가 없다"

    # shift 파일이 실제로 덮는 선택 문항 수를 보고한다(부분 실행 대비)
    for name, vs in models.items():
        if len(vs) > 1:
            cov = min(len(set(v) & sel) for v in vs[1:])
            print(f"{name}: shift 파일 {len(vs)-1}개, 선택 {len(sel)}건 중 {cov}건 덮음")
        else:
            print(f"{name}: shift 파일 없음 — base 로만 계산")

    w = [float(x) for x in a.weights.split(",")] if a.weights else [1.0 / len(models)] * len(models)
    rules = a.rules.split(",")
    vecs, merged = {}, {}
    for rule in rules:
        per_model = {}
        for name, vs in models.items():
            per_model[name] = {}
            for i in ids:
                views = [lsm(vs[0][i]["logprobs"])]
                if i in sel:
                    views += [lsm(v[i]["logprobs"]) for v in vs[1:] if i in v]
                per_model[name][i] = combine(rule, views) if len(views) > 1 else views[0]
        merged[rule] = per_model
        vecs[rule] = [int(LETTERS[int(np.argmax(sum(wi * per_model[n][i] for wi, n in zip(w, models))))] == gold[i])
                      for i in ids]

    md = ["# 선택적 shift TTA 결합 비교", "",
          f"`tools/eval_shift_tta.py`. GPU 미사용. 전체 {len(ids)}건 중 TTA 적용 **{len(sel & set(ids))}**건.", "",
          "| 규칙 | 정답 | 정확도 | 95% CI |", "|---|---|---|---|"]
    for rule in rules:
        k = sum(vecs[rule])
        lo, hi = wilson(k, len(ids))
        md.append(f"| {rule} | {k} | {k/len(ids):.4f} | {lo:.4f}–{hi:.4f} |")
    md += ["", f"- 1문항 = {100/len(ids):.3f}%p", "",
           "### base 대비 짝비교 (paired bootstrap 10,000회)", "",
           "| 비교 | 평균차 | 95% CI | 예측 변경 | 판정 |", "|---|---|---|---|---|"]
    for rule in rules:
        if rule == "base":
            continue
        d, lo, hi, ch = paired(vecs[rule], vecs["base"])
        md.append(f"| {rule} − base | {d:+.2f}%p | {lo:+.2f}–{hi:+.2f} | {ch} | "
                  f"{'**유의**' if (lo > 0 or hi < 0) else '잡음'} |")

    # 선택 구간만 따로 — 이득이 정말 거기서 나는지 확인
    si = [n for n, i in enumerate(ids) if i in sel]
    if si and len(si) < len(ids):
        md += ["", f"### TTA 적용 구간만 (n={len(si)})", "", "| 규칙 | 정답 |", "|---|---|"]
        for rule in rules:
            md.append(f"| {rule} | {sum(vecs[rule][n] for n in si)} |")

    if a.emit_dir:
        os.makedirs(a.emit_dir, exist_ok=True)
        for rule in rules:
            if rule == "base":
                continue
            for name in models:
                p = os.path.join(a.emit_dir, f"{name}_{rule}_predictions.jsonl")
                with open(p, "w", encoding="utf-8", newline="\n") as fh:
                    for i in ids_pool:
                        v = merged[rule][name].get(i)
                        if v is None:
                            v = lsm(models[name][0][i]["logprobs"])
                        fh.write(json.dumps({"id": i, "pred": LETTERS[int(np.argmax(v))],
                                             "logprobs": [round(float(x), 6) for x in v]},
                                            ensure_ascii=False) + "\n")
        md += ["", f"- 결합 예측 저장: `{a.emit_dir}` (tools/ensemble_submit.py 에 그대로 넘긴다)"]

    text = "\n".join(md) + "\n"
    if a.out:
        open(a.out, "w", encoding="utf-8", newline="\n").write(text)
    try:
        print(text)
    except UnicodeEncodeError:   # Windows cp949 콘솔
        pass
    if a.out:
        print(f"저장: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
