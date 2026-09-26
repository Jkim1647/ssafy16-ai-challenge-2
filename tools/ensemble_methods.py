# -*- coding: utf-8 -*-
"""같은 구성원으로 **결합 방식만 바꾼** 제출 파일을 여러 개 만든다. GPU 미사용.

왜 필요한가
  구성원을 바꾸는 실험(T1->T3->T4)은 끝났고, 7개 구성원(홀수)이 되면 결합 방식 자체를 비교할 수 있다.
  방식마다 T4 대비 몇 문항이 바뀌는지 먼저 보고, Public 은 √(K/2) 잡음 규칙으로 읽는다.
  **로컬 채점은 제한적이다** — 27B LoRA·9B LoRA 는 H229 전체 예측이 없어, 모든 구성원이 있는
  공통 문항(--gold 와 교집합)에서만 채점한다. 파인튜닝 모델은 train 으로 학습해 train 채점도 못 한다.

방식
  soft     구성원별 log-softmax 평균(기하평균). 지금까지의 기본(T4)
  pmean    확률 산술평균. 한 모델이 0 에 가까운 확률을 줘도 덜 누른다
  hard     다수결. 동률이면 soft 로 가른다(홀수여도 3-2-2 같은 동률이 생긴다)
  borda    순위 점수(1위 3점, 2위 2점, 3위 1점) 합. 확신도 크기를 무시하고 순서만 쓴다
  trim     보기마다 구성원 로그확률의 최댓값·최솟값을 빼고 평균. 한 모델의 극단값에 덜 흔들린다
  gate     기준 앙상블(--gate-base, 기본 T4 구성원 soft)을 쓰되, 그 마진이 하위 --gate-pct% 인
           문항만 전체 구성원 hard 로 바꾼다. 확신 있는 문항은 건드리지 않는다

usage:
  python tools/ensemble_methods.py --member R3=<jsonl> --member 35B=<jsonl> ... \
      --gate-base R3,35B,397B,Gemma --prefix T7 [--gold <json>] [--methods soft,pmean,...]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ensemble_submit import load_any  # noqa: E402

LETTERS = "abcd"


def lsm(v):
    a = np.clip(np.array(v, float), -1e4, None)
    a = a - a.max()
    return a - np.log(np.exp(a).sum())


def combine(method, M):
    """M: (모델 수, 4) log-softmax. 반환: 4개 점수(클수록 좋음)."""
    if method == "soft":
        return M.mean(0)
    if method == "pmean":
        return np.log(np.exp(M).mean(0))
    if method == "hard":
        votes = np.bincount(M.argmax(1), minlength=4).astype(float)
        return votes + 1e-3 * M.mean(0) / (np.abs(M.mean(0)).max() + 1e-9)   # 동률만 soft 로
    if method == "borda":
        ranks = np.argsort(np.argsort(-M, axis=1), axis=1)       # 0 = 1위
        pts = (3 - ranks).clip(0).sum(0).astype(float)
        return pts + 1e-3 * M.mean(0) / (np.abs(M.mean(0)).max() + 1e-9)
    if method == "trim":
        if len(M) < 3:
            return M.mean(0)
        S = np.sort(M, axis=0)
        return S[1:-1].mean(0)
    raise KeyError(method)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", action="append", required=True, help="이름=경로")
    ap.add_argument("--methods", default="soft,pmean,hard,borda,trim,gate")
    ap.add_argument("--gate-base", default=None, help="gate 기준 구성원(쉼표). 기본은 앞의 4개")
    ap.add_argument("--gate-pct", type=float, default=5.0)
    ap.add_argument("--ref", default="submissions/T4_4soft_no9b_test.csv", help="변경 수 비교 기준 제출")
    ap.add_argument("--gold", default=None, help="로컬 채점용 {id: 정답} json(공통 문항만 채점)")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--outdir", default="submissions")
    a = ap.parse_args()

    names, data = [], {}
    for spec in a.member:
        n, p = spec.split("=", 1)
        names.append(n)
        data[n] = load_any(p)
    gate_base = a.gate_base.split(",") if a.gate_base else names[:4]

    test_ids = [f"test_{i:04d}.jpg" for i in range(1, 6715)]
    for n in names:
        miss = [i for i in test_ids if i not in data[n]]
        assert not miss, f"{n}: test {len(miss)}건 없음"
    ref = {}
    if a.ref and os.path.exists(a.ref):
        with open(a.ref, encoding="utf-8-sig", newline="") as fh:
            ref = {r["id"]: r["answer"] for r in csv.DictReader(fh)}
    gold = json.load(open(a.gold, encoding="utf-8")) if a.gold else {}

    def mat(i, who):
        return np.stack([lsm(data[n][i]["logprobs"]) for n in who])

    def predict(ids, method):
        out = {}
        if method == "gate":
            base = {i: mat(i, gate_base).mean(0) for i in ids}
            marg = {i: np.sort(np.exp(v))[-1] - np.sort(np.exp(v))[-2] for i, v in base.items()}
            cut = np.percentile(list(marg.values()), a.gate_pct)
            for i in ids:
                v = combine("hard", mat(i, names)) if marg[i] <= cut else base[i]
                out[i] = LETTERS[int(np.argmax(v))]
            return out
        for i in ids:
            out[i] = LETTERS[int(np.argmax(combine(method, mat(i, names))))]
        return out

    methods = a.methods.split(",")
    gids = [i for i in gold if all(i in data[n] for n in names)]
    rows = ["| 방식 | 파일 | T4 대비 변경 | 로컬 정답(n) | 답 분포 a/b/c/d |", "|---|---|---|---|---|"]
    for m in methods:
        pred = predict(test_ids, m)
        fn = os.path.join(a.outdir, f"{a.prefix}_{len(names)}{m}_test.csv")
        with open(fn, "w", encoding="utf-8", newline="") as fh:   # 제출 CSV 는 BOM 없이(채점기 ID 매칭)
            fh.write("id,answer\n")
            for i in test_ids:
                fh.write(f"{i},{pred[i]}\n")
        ch = sum(pred[i] != ref[i] for i in test_ids) if ref else "-"
        loc = "-"
        if gids and m != "gate":
            lp = predict(gids, m)
            loc = f"{sum(lp[i] == gold[i] for i in gids)} ({len(gids)})"
        dist = "/".join(str(sum(1 for i in test_ids if pred[i] == x)) for x in LETTERS)
        rows.append(f"| {m} | {os.path.basename(fn)} | {ch} | {loc} | {dist} |")
    if gids:
        base_loc = predict(gids, "soft") if set(gate_base) == set(names) else None
        gb = {i: LETTERS[int(np.argmax(mat(i, gate_base).mean(0)))] for i in gids}
        rows.append(f"| (기준) {'+'.join(gate_base)} soft | - | - | {sum(gb[i] == gold[i] for i in gids)} ({len(gids)}) | - |")
    text = "\n".join([f"구성원({len(names)}): {', '.join(names)}", ""] + rows)
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((text + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
