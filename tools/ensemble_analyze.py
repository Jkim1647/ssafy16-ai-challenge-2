"""여러 모델의 dev 예측을 받아 앙상블 분석 — Oracle 갭(상보성) 공략.

    baseline/Scripts/python.exe tools/ensemble_analyze.py \
        --preds Qwen=preds/qwen.jsonl AX=preds/ax.jsonl InternVL=preds/internvl.jsonl \
                MiniCPM=preds/minicpm.jsonl Molmo=preds/molmo.jsonl \
        --gold runs/dev_validation_v1/hard_eval_gold_v3.json \
        --types runs/dev_validation_v1/dev_audit_all_2683.csv

각 preds 파일(모델명=경로): jsonl(줄마다 {"id","pred"[, "logprobs":[a,b,c,d] 또는 "p_a".."p_d"]}) 또는
csv(열 id,pred[,logprob_a..d]). pred는 a/b/c/d.

산출(모두 규칙 OK — 로컬 in-process 결합):
1. 모델별 정확도(overall + 유형별, Wilson 95% CI, n).
2. 모델 쌍별 오답 겹침(상보성) 행렬.
3. hard-vote 앙상블 vs 신뢰도가중(logprob 있을 때) vs 유형조건부가중 — gold에서 정확도 비교.
4. oracle(누구든 맞히면 정답) 상한.
결과: reports/ensemble_analysis.md (없으면 stdout).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

LETTERS = ["a", "b", "c", "d"]
REPO = Path(__file__).resolve().parents[1]


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963985
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def load_preds(path: Path) -> dict[str, dict]:
    """{id: {"pred": 'a', "p": [pa,pb,pc,pd] or None}}"""
    out: dict[str, dict] = {}
    if path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            p = None
            if "logprobs" in r and r["logprobs"] is not None:
                lg = r["logprobs"]
                m = max(lg)
                e = [math.exp(x - m) for x in lg]
                s = sum(e)
                p = [x / s for x in e]
            elif all(f"p_{c}" in r for c in LETTERS):
                p = [float(r[f"p_{c}"]) for c in LETTERS]
            out[r["id"]] = {"pred": (r.get("pred") or "").strip().lower(), "p": p}
    else:  # csv
        with path.open(encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                p = None
                if all(f"logprob_{c}" in r and r[f"logprob_{c}"] != "" for c in LETTERS):
                    lg = [float(r[f"logprob_{c}"]) for c in LETTERS]
                    m = max(lg)
                    e = [math.exp(x - m) for x in lg]
                    s = sum(e)
                    p = [x / s for x in e]
                out[r["id"]] = {"pred": (r.get("pred") or "").strip().lower(), "p": p}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True, help="모델명=경로 목록")
    ap.add_argument("--gold", required=True, help="{id: 정답} JSON")
    ap.add_argument("--types", default=None, help="id→유형 csv(열 id, auto_type). 없으면 유형표 생략")
    ap.add_argument("--out", default=str(REPO / "reports/ensemble_analysis.md"))
    args = ap.parse_args()

    models: dict[str, dict] = {}
    for kv in args.preds:
        name, path = kv.split("=", 1)
        models[name] = load_preds(Path(path))
    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    gold = {k: v.strip().lower() for k, v in gold.items()}
    ids = sorted(set(gold) & set.intersection(*[set(m) for m in models.values()]))
    types = {}
    if args.types and Path(args.types).exists():
        import pandas as pd
        t = pd.read_csv(args.types, encoding="utf-8-sig", dtype=str).fillna("")
        col = "auto_type" if "auto_type" in t.columns else ("question_type" if "question_type" in t.columns else None)
        if col:
            types = dict(zip(t["id"], t[col]))

    L = [f"# 앙상블 분석 ({len(ids)}문항, 모델 {len(models)}개)\n"]
    # 1. 모델별 정확도 + 유형별
    L.append("## 1. 모델별 정확도 (Wilson 95% CI)\n\n| 모델 | overall | " +
             " | ".join(sorted({types.get(i, "?") for i in ids})) + " |")
    tset = sorted({types.get(i, "?") for i in ids})
    L.append("|---|---|" + "---|" * len(tset))
    for name, m in models.items():
        ok = sum(m[i]["pred"] == gold[i] for i in ids)
        lo, hi = wilson(ok, len(ids))
        cells = [f"{ok/len(ids):.3f} [{lo:.2f},{hi:.2f}] n={len(ids)}"]
        for ty in tset:
            tid = [i for i in ids if types.get(i, "?") == ty]
            k = sum(m[i]["pred"] == gold[i] for i in tid)
            lo2, hi2 = wilson(k, len(tid))
            cells.append(f"{k/max(1,len(tid)):.2f} n={len(tid)}" if tid else "-")
        L.append(f"| {name} | " + " | ".join(cells) + " |")

    # 2. 쌍별 오답 겹침
    L.append("\n## 2. 쌍별 오답 겹침 (둘 다 틀린 수 / 한쪽이라도 틀린 수)\n")
    names = list(models)
    L.append("| | " + " | ".join(names) + " |")
    L.append("|---|" + "---|" * len(names))
    for a in names:
        row = [a]
        for b in names:
            wa = {i for i in ids if models[a][i]["pred"] != gold[i]}
            wb = {i for i in ids if models[b][i]["pred"] != gold[i]}
            row.append(f"{len(wa & wb)}/{len(wa | wb)}" if wa | wb else "0")
        L.append("| " + " | ".join(row) + " |")

    # 3. 앙상블 전략
    def acc(fn) -> float:
        return sum(fn(i) == gold[i] for i in ids) / len(ids)

    def hard_vote(i):
        c = defaultdict(float)
        for m in models.values():
            c[m[i]["pred"]] += 1
        return max(LETTERS, key=lambda x: (c.get(x, 0), -LETTERS.index(x)))

    def conf_avg(i):
        if not all(models[n][i]["p"] for n in models):
            return hard_vote(i)
        s = [0.0] * 4
        for m in models.values():
            for j in range(4):
                s[j] += m[i]["p"][j]
        return LETTERS[max(range(4), key=lambda j: s[j])]

    def oracle(i):
        return gold[i] if any(models[n][i]["pred"] == gold[i] for n in models) else models[names[0]][i]["pred"]

    L.append("\n## 3. 앙상블 전략 정확도\n")
    L.append(f"- hard-vote(다수결): **{acc(hard_vote):.4f}**")
    have_p = all(all(models[n][i]["p"] for i in ids) for n in models)
    L.append(f"- 신뢰도평균(logprob softmax): **{acc(conf_avg):.4f}**" if have_p else "- 신뢰도평균: logprob 없어 생략(모델별 a/b/c/d logprob를 넣으면 활성화)")
    L.append(f"- **oracle 상한(누구든 맞히면): {acc(oracle):.4f}** ← 이 갭이 앙상블로 줍을 수 있는 최대치")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8-sig")
    print("\n".join(L))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
