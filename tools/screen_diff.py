# -*- coding: utf-8 -*-
"""스크리닝 설정이 base 대비 **어느 문항을 고치고 어느 문항을 망가뜨렸는지** 뽑는다. GPU 미사용.

score_screen.py 는 "몇 개 올랐나"만 답한다. 다음 수를 정하려면 "무엇이 고쳐졌나"가 필요하다.
test 불일치 분석(reports/test_review.*)은 정답이 없어 "갈렸다"만 볼 수 있었지만, 여기는
train1000 + H229 정답이 있어 **실제로 맞았는지 틀렸는지**로 본다. 수동 검토·Codex 분류는
train·dev 문항이라 대회 규칙 4-b(test 수기 라벨링 금지)와 무관하다.

산출물
  reports/screen_diff.md     설정별 고침/망가뜨림 수, 유형별 순이득, 설정 간 겹침
  reports/screen_diff.jsonl  바뀐 문항 전부(질문·보기·정답·양쪽 예측·확신도·판독문) — Codex 입력용

usage:
  python tools/screen_diff.py --base base=<jsonl> --cfg r1v2=<jsonl> [--cfg ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter

LETTERS = "abcd"


def load_preds(path):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r
    return out


def load_csv(path, key):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return {r[key]: r for r in csv.DictReader(fh)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="name=path")
    ap.add_argument("--cfg", action="append", required=True, help="name=path (여러 번)")
    ap.add_argument("--gold", default="runs/screen/gold1408_valid.json")
    ap.add_argument("--data", default="runs/screen/dev1408.csv", help="질문·보기 원문(수정본 반영)")
    ap.add_argument("--types", default="data_meta/question_types.csv")
    ap.add_argument("--out", default="reports/screen_diff")
    args = ap.parse_args()

    gold = json.load(open(args.gold, encoding="utf-8"))
    rows = load_csv(args.data, "id")
    types = load_csv(args.types, "row_id") if os.path.exists(args.types) else {}
    bname, bpath = args.base.split("=", 1)
    base = load_preds(bpath)
    cfgs = [c.split("=", 1) for c in args.cfg]

    md = [f"# 스크리닝 설정별 고침·망가뜨림 (base = {bname})", "",
          f"`tools/screen_diff.py`. 채점 {len(gold)}건(`{args.gold}`). GPU 미사용.", "",
          "| 설정 | 채점 | base 정답 | 설정 정답 | 고침 | 망가뜨림 | 순이득 | train | dev(H229) |",
          "|---|---|---|---|---|---|---|---|---|"]
    items, fixed_sets = [], {}
    type_net = {}
    for name, path in cfgs:
        pr = load_preds(path)
        ids = [i for i in gold if i in base and i in pr]
        fixed, broken = [], []
        for i in ids:
            b_ok, c_ok = base[i]["pred"] == gold[i], pr[i]["pred"] == gold[i]
            if b_ok != c_ok:
                (fixed if c_ok else broken).append(i)
        net = Counter()
        for i in fixed:
            net[(types.get(i, {}).get("auto_type") or "?")] += 1
        for i in broken:
            net[(types.get(i, {}).get("auto_type") or "?")] -= 1
        type_net[name] = net
        fixed_sets[name] = set(fixed)
        part = lambda lst, p: sum(1 for i in lst if i.startswith(p))
        md.append(f"| {name} | {len(ids)} | {sum(base[i]['pred'] == gold[i] for i in ids)} | "
                  f"{sum(pr[i]['pred'] == gold[i] for i in ids)} | {len(fixed)} | {len(broken)} | "
                  f"**{len(fixed) - len(broken):+d}** | {part(fixed, 'train') - part(broken, 'train'):+d} | "
                  f"{part(fixed, 'dev') - part(broken, 'dev'):+d} |")
        for i in fixed + broken:
            r = rows.get(i, {})
            split = "train" if i.startswith("train") else "dev"
            items.append({
                "config": name, "status": "fixed" if i in fixed else "broken", "id": i,
                "image": f"../ssafy-16-2-ai/{split}/{i}", "type": types.get(i, {}).get("auto_type"),
                "question": r.get("question"), "choices": {x: r.get(x) for x in LETTERS},
                "gold": gold[i],
                "base": {"pred": base[i]["pred"], "conf": base[i].get("confidence"),
                         "transcript": base[i].get("transcript")},
                "cfg": {"pred": pr[i]["pred"], "conf": pr[i].get("confidence"),
                        "transcript": pr[i].get("transcript")},
            })

    md += ["", "- 순이득이 ±√(고침+망가뜨림) 안이면 잡음과 구분되지 않는다.",
           "- 채택 조건은 score_screen.py 와 같다: train 순이득 + dev(H229) 악화 없음.", "",
           "## 유형별 순이득", ""]
    all_types = sorted({t for n in type_net.values() for t in n})
    md.append("| 유형 | " + " | ".join(n for n, _ in cfgs) + " |")
    md.append("|---|" + "---|" * len(cfgs))
    for t in all_types:
        md.append(f"| {t} | " + " | ".join(f"{type_net[n][t]:+d}" for n, _ in cfgs) + " |")

    if len(cfgs) > 1:
        md += ["", "## 설정 간 고친 문항 겹침 (둘 다 고친 수 / 합집합)", ""]
        names = [n for n, _ in cfgs]
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                A, B = fixed_sets[names[a]], fixed_sets[names[b]]
                md.append(f"- {names[a]} × {names[b]}: {len(A & B)} / {len(A | B)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".md", "w", encoding="utf-8-sig", newline="\n") as fh:
        fh.write("\n".join(md) + "\n")
    with open(args.out + ".jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    try:
        print("\n".join(md))
    except UnicodeEncodeError:
        sys.stdout.buffer.write(("\n".join(md) + "\n").encode("utf-8"))
    print(f"-> {args.out}.md / .jsonl ({len(items)}건)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
