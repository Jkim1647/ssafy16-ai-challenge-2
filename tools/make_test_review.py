# -*- coding: utf-8 -*-
"""test 에서 **모델이 갈리거나 확신이 낮은 문항**을 눈으로 보게 만든다. GPU 미사용.

규칙 경계를 먼저 분명히 한다
  Kaggle 기본규칙 4-b 는 **validation·test 레코드에 대한 사람의 라벨링·수기 예측을
  제출에 쓰는 것을 금지한다**(CLAUDE.md 「확정으로 기록한 대회 규칙」). 이 페이지를 보고
  "이 문항은 c 가 맞다"고 판단해 제출 CSV 를 손으로 고치면 규칙 위반이다.

  허용되는 쓰임은 **실패 유형 진단**이다. "저마진 불일치 대부분이 표의 어느 행을 가리키는지
  헷갈린 것"이라는 패턴을 찾으면 프롬프트·타일·해상도를 바꿔 **전 문항에 균일하게** 적용할
  수 있다. 그건 모델 개선이지 수기 라벨링이 아니다.

  또 test 정답은 우리에게 없다. 여기서 보이는 것은 "모델이 갈렸다 / 확신이 낮다"이지
  "틀렸다"가 아니다. 리더보드 점수 차로 개별 문항 정답을 역산하는 것(leaderboard probing)도
  하지 않는다 — 일 20회로는 불가능하고, Public 에만 맞추게 되어 Private 에서 깨진다.

무엇을 고르나
  --mode disagree : 모델들의 argmax 가 갈린 문항
  --mode margin   : 앙상블 1·2위 확률 차가 낮은 문항
  --mode both     : 둘 중 하나라도 해당(기본)
  --mode softhard : 소프트(로그확률 평균)와 하드(다수결)의 결론이 갈린 문항만

소프트/하드 비교는 **모델 수가 홀수일 때만 의미가 있다.** 짝수면 2:2 동률이 자주 나고
동률은 소프트 점수로 가르므로 하드가 소프트로 수렴한다 — 실제로 4모델에서는 6,714 중
21건, val667·train6047 에서는 **1건**만 갈렸다.

usage:
  python tools/make_test_review.py --pred A.jsonl --pred B.jsonl --csv ssafy-16-2-ai/test.csv \
    --limit 150 --out reports/test_review.html
"""
from __future__ import annotations

import argparse
import collections
import csv
import html
import io
import json
import math
import os
import re
import sys

LETTERS = "abcd"


def load_preds(path):
    """id -> (a~d 로그확률 또는 로짓, 판독문). jsonl 과 csv 를 모두 받는다."""
    out = {}
    if path.lower().endswith(".csv"):
        for r in csv.DictReader(open(path, encoding="utf-8-sig")):
            try:
                out[r["id"]] = ([float(r["logit_" + x]) for x in LETTERS], "")
            except (KeyError, ValueError):
                continue
        return out
    for line in open(path, encoding="utf-8-sig"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        lp = r.get("logprobs")
        if lp and len(lp) == 4:
            out[r["id"]] = (lp, r.get("transcript", "") or "")
    return out


def norm(lp):
    """log-softmax. 397B 파일은 상위 20 밖 글자에 -1e4 표시값을 쓰므로 최댓값-30 으로 누른다."""
    m = max(lp)
    lp = [max(x, m - 30) for x in lp]
    m = max(lp)
    lse = m + math.log(sum(math.exp(x - m) for x in lp))
    return [x - lse for x in lp]


def canon(x):
    return re.sub(r"\s", "", str(x)).lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", action="append", required=True, help="예측 파일(jsonl 또는 csv)")
    ap.add_argument("--name", action="append", default=None, help="모델 표시 이름(--pred 순서대로)")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--csv", default="ssafy-16-2-ai/test.csv")
    ap.add_argument("--image-prefix", default="../ssafy-16-2-ai/")
    ap.add_argument("--out", default="reports/test_review.html")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--low-margin", type=float, default=0.30,
                    help="1·2위 확률 차가 이 값 미만이면 저마진으로 본다")
    ap.add_argument("--mode", default="both", choices=["disagree", "margin", "both", "softhard"],
                    help="softhard: 소프트(로그확률 평균)와 하드(다수결)의 결론이 갈린 문항만. "
                         "두 결합 방식 중 무엇을 쓸지는 이 문항들에서만 결정된다")
    ap.add_argument("--jsonl", default=None,
                    help="같은 항목을 기계가 읽기 쉬운 jsonl 로도 쓴다. 다른 도구·에이전트에게 "
                         "HTML 을 파싱시키지 않기 위한 동반 파일이다(이미지 경로 포함)")
    a = ap.parse_args()

    rows = {r["id"]: r for r in csv.DictReader(open(a.csv, encoding="utf-8-sig"))}
    models = [load_preds(p) for p in a.pred]
    names = a.name or [os.path.basename(p)[:24] for p in a.pred]
    assert len(names) == len(models), "--name 개수가 --pred 개수와 다르다"
    w = ([float(x) for x in a.weights.split(",")] if a.weights else [1.0 / len(models)] * len(models))

    ids = sorted(set.intersection(*[set(m) for m in models]) & set(rows))
    print(f"공통 문항 {len(ids)}건, 모델 {len(models)}개")

    items = []
    for i in ids:
        per = [norm(m[i][0]) for m in models]
        ens = [sum(wi * p[j] for wi, p in zip(w, per)) for j in range(4)]
        pred = LETTERS[max(range(4), key=lambda j: ens[j])]
        probs = [math.exp(v) for v in ens]
        srt = sorted(probs)
        margin = srt[-1] - srt[-2]
        singles = [LETTERS[max(range(4), key=lambda j: p[j])] for p in per]
        # 하드 다수결: 모델 가중치로 표를 세고, 동률이면 소프트 점수로 가른다(ensemble_submit --vote 와 같은 규칙)
        votes = [0.0] * 4
        for wi, sg in zip(w, singles):
            votes[LETTERS.index(sg)] += wi
        top = max(votes)
        tied = [j for j in range(4) if votes[j] == top]
        hard = LETTERS[tied[0] if len(tied) == 1 else max(tied, key=lambda j: ens[j])]
        dis = len(set(singles)) > 1
        low = margin < a.low_margin
        sh = hard != pred
        if a.mode == "disagree" and not dis:
            continue
        if a.mode == "margin" and not low:
            continue
        if a.mode == "softhard" and not sh:
            continue
        if a.mode == "both" and not (dis or low):
            continue
        tags = []
        if sh:
            tags.append(f"소프트{pred}≠하드{hard}")
        if dis:
            tags.append("모델불일치 " + "/".join(singles))
        if low:
            tags.append("저마진")
        # 보기끼리 부분문자열이면 중의적 선지다(디스커션 실측 train 4.97% / test 5.00%)
        cs = [canon(rows[i][x]) for x in LETTERS]
        if any(cs[p] and cs[q] and (cs[p] in cs[q] or cs[q] in cs[p])
               for p in range(4) for q in range(p + 1, 4)):
            tags.append("중의적선지")
        items.append(dict(id=i, margin=margin, pred=pred, hard=hard, probs=probs, tags=tags, singles=singles,
                          per=[[math.exp(v) for v in p] for p in per],
                          trs=[(n, m[i][1][:260]) for n, m in zip(names, models) if m[i][1]]))

    items.sort(key=lambda x: x["margin"])
    total = len(items)
    if a.limit:
        items = items[:a.limit]
    esc = html.escape

    out = [
        '<!doctype html><meta charset="utf-8"><title>test 검토(모델 불일치·저마진)</title>',
        '<style>body{font-family:system-ui,"Malgun Gothic",sans-serif;margin:24px;background:#fafafa;color:#111}',
        '.it{background:#fff;border:1px solid #ddd;border-radius:8px;padding:14px;margin:18px 0;display:grid;',
        'grid-template-columns:340px 1fr;gap:16px}img{max-width:340px;border:1px solid #ccc;border-radius:4px}',
        '.q{font-weight:600;margin-bottom:8px}.o{padding:3px 6px;margin:2px 0;border-radius:4px}',
        '.pred{background:#ddeeff;font-weight:600}',
        '.tag{display:inline-block;background:#eee;border-radius:10px;padding:1px 8px;font-size:12px;margin-right:4px}',
        '.tr{font-size:12px;color:#555;white-space:pre-wrap;background:#f6f6f6;padding:6px;border-radius:4px;margin-top:6px}',
        '.mv{font-size:12px;color:#444;margin-top:6px}h1{font-size:20px}</style>',
        f'<h1>test 검토 — 모델이 갈리거나 확신이 낮은 {len(items)}건 (해당 전체 {total}건, 마진 낮은 순)</h1>',
        '<p style="color:#b00"><b>정답은 우리에게 없다.</b> 파란색은 앙상블이 고른 답일 뿐 정답이 아니다. '
        '여기서 얻을 것은 <b>실패 유형</b>이며, 개별 문항의 답을 사람이 정해 제출에 반영하는 것은 '
        'Kaggle 기본규칙 4-b 위반이다.</p>',
        f'<p style="color:#555">이미지는 <code>{esc(a.image_prefix)}</code> 를 참조하므로 '
        '대회 데이터가 있는 PC 에서 <code>reports/</code> 안에 두고 열 것.</p>',
    ]
    for k, it in enumerate(items, 1):
        r = rows[it["id"]]
        out.append(f'<div class="it"><div><img src="{esc(a.image_prefix + r["path"])}" loading="lazy">'
                   f'<div style="font-size:12px;color:#666;margin-top:4px">{k}. {esc(it["id"])} · '
                   f'마진 {it["margin"]:.3f}</div></div><div>')
        out.append("".join(f'<span class="tag">{esc(t)}</span>' for t in it["tags"]))
        out.append(f'<div class="q">{esc(r["question"])}</div>')
        for j, x in enumerate(LETTERS):
            mark = ("소프트" if x == it["pred"] else "") + ("하드" if x == it["hard"] else "")
            cls = "pred o" if (x == it["pred"] or x == it["hard"]) else "o"
            out.append(f'<div class="{cls}">{x}. {esc(str(r[x]))} '
                       f'<span style="color:#06c;font-size:11px">{mark}</span> '
                       f'<span style="color:#888;font-size:12px">{it["probs"][j] * 100:.1f}%</span></div>')
        out.append('<div class="mv">모델별: ' + " · ".join(
            f'{esc(n)} <b>{s}</b> ({max(p) * 100:.0f}%)'
            for n, s, p in zip(names, it["singles"], it["per"])) + '</div>')
        for nm, tr in it["trs"]:
            out.append(f'<div class="tr"><b>{esc(nm)}</b>: {esc(tr)}</div>')
        out.append('</div></div>')

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    # 브라우저가 읽는 파일이라 BOM 없이 utf-8 로 쓰고 charset 을 meta 로 선언한다.
    io.open(a.out, "w", encoding="utf-8", newline="\n").write("\n".join(out))

    if a.jsonl:
        # 기계가 읽는 산출물이라 BOM 없이 utf-8 (CLAUDE.md 인코딩 규칙 3)
        with io.open(a.jsonl, "w", encoding="utf-8", newline="\n") as fh:
            for it in items:
                r = rows[it["id"]]
                fh.write(json.dumps({
                    "id": it["id"],
                    "image": a.image_prefix + r["path"],
                    "question": r["question"],
                    "choices": {x: r[x] for x in LETTERS},
                    "ensemble_pick_soft": it["pred"],
                    "ensemble_pick_hard": it["hard"],
                    "ensemble_probs": {x: round(it["probs"][j], 4) for j, x in enumerate(LETTERS)},
                    "margin": round(it["margin"], 4),
                    "model_picks": dict(zip(names, it["singles"])),
                    "model_confidence": {n: round(max(p), 4) for n, p in zip(names, it["per"])},
                    "tags": it["tags"],
                    "transcripts": {n: t for n, t in it["trs"]},
                }, ensure_ascii=False) + "\n")
        print(f"동반 jsonl: {a.jsonl}")
    print(f"생성: {a.out} ({len(items)}건, {os.path.getsize(a.out) // 1024}KB)")
    print("태그 분포:", dict(collections.Counter(t.split()[0] for it in items for t in it["tags"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
