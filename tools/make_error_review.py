"""앙상블 오답 검토 페이지를 만든다.

왜 필요한가. 1등 팀(다른 참가자) 조언이 "앙상블만 쓰지 않고 어떤 유형이 틀리나 계속 보면서
어이없게 틀리는 걸 보정했다" 였다. 그걸 하려면 오답을 눈으로 볼 수 있어야 한다.
자동 검사로는 저수준 오류가 거의 안 잡혔지만(문자 유형 불일치 3.4%, 숫자 질문 오류 0건),
사람 눈에는 코드로 만들지 못한 범주가 보일 수 있다.

설계에서 정한 것 둘.

1. **이미지를 base64 로 박지 않고 상대 경로로 참조한다.** 박으면 10MB 를 넘고 대회 이미지가
   저장소에 들어간다. 경로 참조면 파일이 수백 KB 로 끝나고, 대회 데이터가 있는 PC 에서
   `reports/` 안에서 열면 그대로 보인다.
2. **판독문(transcript)을 같이 보여준다.** "읽기 실패"와 "선택 실패"를 눈으로 가르기
   위해서다. 우리 진단으로는 둘 다 틀린 건의 86%가 판독은 맞고 선택이 어긋난 경우다.
   예: 판독문에는 `LIFE` 가 정확히 적혀 있는데 모델은 철자가 비슷한 `LIFT` 를 골랐다.

사용 예 (R3·Gemma 합류 후 다시 뽑을 때):

    python tools/make_error_review.py \
      --pred shared_predictions/runpod_35b_train6047/...predictions.jsonl \
      --pred shared_predictions/nebius_397b_kd/...predictions.jsonl \
      --out reports/ensemble_errors_review.html

`--pred` 를 여러 번 주면 균등 가중으로 결합한다. `--weights 0.5,0.5` 로 바꿀 수 있다.
우리 jsonl 형식과 팀원 Gemma 의 csv(`logit_a..d`) 형식을 모두 읽는다.
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
        # 팀원 Gemma 형식: id,prediction,logit_a..d,prob_a..d
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
    """log-softmax 정규화. 397B 파일은 상위 20 밖 글자에 -1e4 표시값을 쓰므로
    최댓값-30 으로 눌러 언더플로를 막는다."""
    m = max(lp)
    lp = [max(x, m - 30) for x in lp]
    m = max(lp)
    lse = m + math.log(sum(math.exp(x - m) for x in lp))
    return [x - lse for x in lp]


def canon(x):
    return re.sub(r"\s", "", str(x)).lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", action="append", required=True,
                    help="예측 파일. 여러 번 주면 결합한다(jsonl 또는 csv)")
    ap.add_argument("--weights", default=None, help="쉼표 구분. 생략하면 균등")
    ap.add_argument("--csv", default="ssafy-16-2-ai/train.csv",
                    help="정답·질문·보기가 있는 원본 csv")
    ap.add_argument("--image-prefix", default="../ssafy-16-2-ai/",
                    help="HTML 에서 이미지를 참조할 접두. reports/ 에 두는 것이 기본값")
    ap.add_argument("--out", default="reports/ensemble_errors_review.html")
    ap.add_argument("--limit", type=int, default=None, help="상위 N건만(마진 낮은 순)")
    ap.add_argument("--low-margin", type=float, default=0.10,
                    help="1·2위 확률 차가 이 값 미만이면 '저마진' 태그")
    args = ap.parse_args()

    rows = {r["id"]: r for r in csv.DictReader(open(args.csv, encoding="utf-8-sig"))}
    models = [load_preds(p) for p in args.pred]
    names = [os.path.basename(p)[:28] for p in args.pred]
    w = ([float(x) for x in args.weights.split(",")] if args.weights
         else [1.0 / len(models)] * len(models))
    assert len(w) == len(models), "weights 개수가 pred 개수와 다르다"

    ids = sorted(set.intersection(*[set(m) for m in models]) & set(rows))
    print(f"공통 문항 {len(ids)}건, 모델 {len(models)}개, 가중치 {w}")

    items = []
    for i in ids:
        per = [norm(m[i][0]) for m in models]
        ens = [sum(wi * p[j] for wi, p in zip(w, per)) for j in range(4)]
        pred = LETTERS[max(range(4), key=lambda j: ens[j])]
        goldletter = rows[i]["answer"].strip().lower()
        if pred == goldletter:
            continue
        probs = [math.exp(v) for v in ens]
        srt = sorted(probs)
        margin = srt[-1] - srt[-2]
        singles = [LETTERS[max(range(4), key=lambda j: p[j])] for p in per]
        tags = ["전원틀림" if all(s != goldletter for s in singles) else "일부만틀림"]
        cp, cg = canon(rows[i][pred]), canon(rows[i][goldletter])
        if cp and (cp in cg or cg in cp):
            tags.append("부분문자열")
        if margin < args.low_margin:
            tags.append("저마진")
        items.append(dict(id=i, margin=margin, pred=pred, gold=goldletter, probs=probs,
                          tags=tags, singles=singles,
                          trs=[(n, m[i][1][:260]) for n, m in zip(names, models) if m[i][1]]))

    items.sort(key=lambda x: x["margin"])
    if args.limit:
        items = items[:args.limit]
    esc = html.escape

    out = [
        '<!doctype html><meta charset="utf-8"><title>앙상블 오답 검토</title>',
        '<style>body{font-family:system-ui,"Malgun Gothic",sans-serif;margin:24px;background:#fafafa;color:#111}',
        '.it{background:#fff;border:1px solid #ddd;border-radius:8px;padding:14px;margin:18px 0;display:grid;',
        'grid-template-columns:320px 1fr;gap:16px}img{max-width:320px;border:1px solid #ccc;border-radius:4px}',
        '.q{font-weight:600;margin-bottom:8px}.o{padding:3px 6px;margin:2px 0;border-radius:4px}',
        '.gold{background:#d8f5d8;font-weight:600}.pred{background:#ffd9d9;font-weight:600}',
        '.tag{display:inline-block;background:#eee;border-radius:10px;padding:1px 8px;font-size:12px;margin-right:4px}',
        '.tr{font-size:12px;color:#555;white-space:pre-wrap;background:#f6f6f6;padding:6px;border-radius:4px;',
        'margin-top:6px}h1{font-size:20px}</style>',
        f'<h1>앙상블 오답 {len(items)}건 — 마진 낮은 순</h1>',
        '<p style="color:#555">초록=정답, 빨강=모델이 고른 답. 마진이 낮을수록 모델이 헷갈린 문항이다. '
        f'이미지는 <code>{esc(args.image_prefix)}</code> 를 참조하므로 대회 데이터가 있는 PC 에서 열 것.</p>',
    ]
    for k, it in enumerate(items, 1):
        r = rows[it["id"]]
        out.append(f'<div class="it"><div><img src="{esc(args.image_prefix + r["path"])}" loading="lazy">'
                   f'<div style="font-size:12px;color:#666;margin-top:4px">{k}. {esc(it["id"])} · '
                   f'마진 {it["margin"]:.3f} · 단일예측 {esc("/".join(it["singles"]))}</div></div><div>')
        out.append("".join(f'<span class="tag">{esc(t)}</span>' for t in it["tags"]))
        out.append(f'<div class="q">{esc(r["question"])}</div>')
        for j, x in enumerate(LETTERS):
            cls = "gold" if x == it["gold"] else ("pred" if x == it["pred"] else "o")
            out.append(f'<div class="{cls} o">{x}. {esc(str(r[x]))} '
                       f'<span style="color:#888;font-size:12px">{it["probs"][j] * 100:.1f}%</span></div>')
        for nm, tr in it["trs"]:
            out.append(f'<div class="tr"><b>{esc(nm)}</b>: {esc(tr)}</div>')
        out.append('</div></div>')

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # 브라우저가 읽는 파일이라 BOM 없이 utf-8 로 쓰고 charset 을 meta 로 선언한다.
    io.open(args.out, "w", encoding="utf-8", newline="\n").write("\n".join(out))
    print(f"생성: {args.out} ({len(items)}건, {os.path.getsize(args.out) // 1024}KB)")
    print("태그 분포:", dict(collections.Counter(t for it in items for t in it["tags"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
