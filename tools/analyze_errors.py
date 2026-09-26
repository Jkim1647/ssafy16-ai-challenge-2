# -*- coding: utf-8 -*-
"""남은 오답의 구조를 본다 — 무엇이 회수 가능하고 무엇이 바닥인가. GPU 미사용.

무엇을 답하려는 도구인가
  "어느 문제를 틀렸나"만으로는 다음 수가 안 나온다. 오답을 **회수 가능성**으로 갈라야 한다.

    전원 오답      아무 모델도 못 맞힌다. 결합을 바꿔도 안 나온다.
                   -> 라벨 결함이거나, 새 모델·새 입력이 필요한 구간
    앙상블만 오답  누군가는 맞혔는데 결합이 떨어뜨렸다.
                   -> **결합·가중치·신뢰도 라우팅으로 회수 가능한 상한**
    오라클         문항마다 맞힌 모델이 하나라도 있으면 정답 처리. 현재 모델 구성의 천장.

  그다음 회수 가능한 오답이 어디에 몰려 있는지 본다(마진 / 정답 글자 / 유형 / 모델 조합).
  마진 하위 구간에 몰려 있으면 shift TTA·재추론이 듣고, 특정 모델만 맞히는 문항이 많으면
  그 모델의 가중치를 올릴 근거가 된다(단, 유형별 가중치는 이미 검증해 졌다 —
  reports/type_routing_20260923.md).

usage:
  python tools/analyze_errors.py --gold runs/screen/gold.json \
    --model "35B=<jsonl>" --model "397B=<jsonl>" [--model ...] [--out reports/error_structure.md]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from collections import Counter, defaultdict

import numpy as np

LETTERS = "abcd"


def load(path):
    d = {}
    if path.lower().endswith(".csv"):
        import csv
        with open(path, encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                key = r.get("id") or r.get("row_id")
                d[key] = {"pred": (r.get("prediction") or r.get("pred") or "").strip().lower(),
                          "logprobs": [float(r[f"logit_{c}"]) for c in LETTERS]}
        return d
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--model", action="append", required=True,
                    help='"이름=경로" 또는 "이름=경로1,경로2" (여러 파일을 합친다. '
                         'H408 처럼 평가셋이 두 실행에 나뉘어 있을 때 쓴다)')
    ap.add_argument("--out", default="reports/error_structure.md")
    ap.add_argument("--types", default="data_meta/question_types.csv")
    a = ap.parse_args()

    gold = json.load(open(a.gold, encoding="utf-8"))
    models = {}
    for spec in a.model:
        name, paths = spec.split("=", 1)
        merged = {}
        for path in paths.split(","):
            assert os.path.exists(path), path
            merged.update(load(path))
        models[name] = merged

    ids = sorted(i for i in gold if all(i in m for m in models.values()))
    assert ids, "공통 id 가 없다"

    # 균등 앙상블(제출 규칙과 같은 로그확률 평균)
    S = {n: {i: lsm(m[i]["logprobs"]) for i in ids} for n, m in models.items()}
    ens = {i: LETTERS[int(np.argmax(sum(S[n][i] for n in models) / len(models)))] for i in ids}

    ok = {n: {i: models[n][i]["pred"] == gold[i] for i in ids} for n in models}
    ok_ens = {i: ens[i] == gold[i] for i in ids}
    any_ok = {i: any(ok[n][i] for n in models) for i in ids}

    n = len(ids)
    md = ["# 남은 오답의 구조", "", "`tools/analyze_errors.py`. GPU 미사용.", "",
          f"대상 {n}건, 모델 {len(models)}개 ({', '.join(models)})", "",
          "## 1. 회수 가능성으로 나눈 오답", "", "| 구분 | 건수 | 비율 |", "|---|---|---|"]
    err_ens = [i for i in ids if not ok_ens[i]]
    core = [i for i in ids if not any_ok[i]]
    recov = [i for i in err_ens if any_ok[i]]
    md += [f"| 앙상블 정답 | {n - len(err_ens)} | {(n-len(err_ens))/n:.4f} |",
           f"| **앙상블 오답 중 회수 가능**(누군가는 맞힘) | **{len(recov)}** | {len(recov)/n:.4f} |",
           f"| 전원 오답(바닥) | {len(core)} | {len(core)/n:.4f} |",
           "", f"- 오라클(문항별 최선) = **{n - len(core)}/{n} = {(n-len(core))/n:.4f}** — 현재 구성의 천장",
           f"- 앙상블 = {n - len(err_ens)}/{n} = {(n-len(err_ens))/n:.4f}",
           f"- **결합만으로 더 얻을 수 있는 최대치 = {len(recov)}문항 ({len(recov)/n*100:.2f}%p)**", ""]

    md += ["## 2. 모델별", "", "| 모델 | 정답 | 정확도 | 이 모델만 맞힌 문항 |", "|---|---|---|---|"]
    for name in sorted(models, key=lambda x: -sum(ok[x].values())):
        k = sum(ok[name].values())
        only = sum(1 for i in ids if ok[name][i] and not any(ok[o][i] for o in models if o != name))
        md.append(f"| {name} | {k} | {k/n:.4f} | {only} |")
    md += ["", "- '이 모델만 맞힌 문항'이 크면 그 모델을 빼면 안 된다. 단독 점수가 낮아도 마찬가지다.", ""]

    md += ["## 3. 오답 겹침 (두 모델이 같이 틀린 문항 수)", "", "| 쌍 | 같이 틀림 | 자카드 |", "|---|---|---|"]
    for x, y in itertools.combinations(sorted(models), 2):
        ex = {i for i in ids if not ok[x][i]}
        ey = {i for i in ids if not ok[y][i]}
        inter = len(ex & ey)
        md.append(f"| {x} ∩ {y} | {inter} | {inter/max(len(ex|ey),1):.3f} |")
    md += ["", "- 자카드가 낮을수록 서로 다른 것을 틀린다 = 섞을 값어치가 크다.", ""]

    # 4. 회수 가능 오답이 어디에 몰려 있나 — 마진
    ensp = {}
    for i in ids:
        s = sum(S[nm][i] for nm in models) / len(models)
        p = np.exp(s - s.max())
        p /= p.sum()
        t = np.sort(p)[::-1]
        ensp[i] = t[0] - t[1]
    order = sorted(ids, key=lambda i: ensp[i])
    md += ["## 4. 회수 가능 오답이 어디에 있나 (앙상블 마진 구간별)", "",
           "| 구간 | n | 앙상블 오답 | 그중 회수 가능 |", "|---|---|---|---|"]
    bands = [(0.0, 0.05, "하위 5%"), (0.05, 0.10, "5~10%"), (0.10, 0.25, "10~25%"), (0.25, 1.0, "상위 75%")]
    for lo, hi, label in bands:
        sel = order[int(n * lo):int(n * hi)]
        e = [i for i in sel if not ok_ens[i]]
        r = [i for i in e if any_ok[i]]
        md.append(f"| {label} | {len(sel)} | {len(e)} | {len(r)} |")
    md += ["", "- 오답과 회수 가능분이 저마진에 몰리면 **저마진만 재추론**하는 것이 싸게 먹힌다.", ""]

    # 5. 정답 글자별 (위치 편향)
    md += ["## 5. 정답 글자별 오답률 (위치 편향)", "", "| 정답 | n | 앙상블 오답 | 오답률 |", "|---|---|---|---|"]
    by = defaultdict(list)
    for i in ids:
        by[gold[i]].append(i)
    for g in LETTERS:
        s = by.get(g, [])
        if not s:
            continue
        e = sum(1 for i in s if not ok_ens[i])
        md.append(f"| {g} | {len(s)} | {e} | {e/len(s):.4f} |")
    md += ["", "- a·d 가 c 보다 뚜렷이 높으면 선택지 순환(shift) 평균이 듣는다.", ""]

    # 6. 유형별 (진단용)
    if os.path.exists(a.types):
        import csv
        typ = {}
        with open(a.types, encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                typ[r["row_id"]] = r["auto_type"]
        md += ["## 6. 유형별 (진단용 — 가중치 근거로는 쓰지 않는다)", "",
               "| 유형 | n | 앙상블 오답 | 전원 오답 |", "|---|---|---|---|"]
        tb = defaultdict(list)
        for i in ids:
            tb[typ.get(i, "?")].append(i)
        for t, s in sorted(tb.items(), key=lambda kv: -len(kv[1])):
            e = sum(1 for i in s if not ok_ens[i])
            c = sum(1 for i in s if not any_ok[i])
            md.append(f"| {t} | {len(s)} | {e} | {c} |")
        md += [""]

    # 7. 회수 가능 오답 목록(사람 검수용, 상위 40건)
    md += ["## 7. 회수 가능 오답 (마진 낮은 순 40건)", "",
           "| id | 정답 | 앙상블 | 맞힌 모델 | 마진 |", "|---|---|---|---|---|"]
    for i in sorted(recov, key=lambda x: ensp[x])[:40]:
        who = ",".join(nm for nm in models if ok[nm][i])
        md.append(f"| {i} | {gold[i]} | {ens[i]} | {who} | {ensp[i]:.4f} |")
    md += ["", f"- 전체 회수 가능 {len(recov)}건. 전원 오답 {len(core)}건은 라벨 검수 대상이다.", ""]

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    text = "\n".join(md) + "\n"
    open(a.out, "w", encoding="utf-8", newline="\n").write(text)
    try:
        print(text)
    except UnicodeEncodeError:   # Windows cp949 콘솔
        pass
    print(f"저장: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
