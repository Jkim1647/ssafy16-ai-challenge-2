# -*- coding: utf-8 -*-
"""선택적 shift TTA 대상 고르기 — 앙상블 마진 하위 k%. GPU 미사용.

왜 전 문항을 돌리지 않는가
  val667(35B, ours)에서 전 문항 4순서 평균의 이득 +5 가 **마진 하위 5% 밴드에서 전부** 나왔다.
  나머지 세 구간은 정확히 Δ0 이다.

    | 구간 | n | shift0 | 4순서평균 | Δ |
    | 하위 5% | 33 | 14 | 19 | +5 |
    | 5~10%   | 33 | 24 | 24 | +0 |
    | 10~25%  |100 | 94 | 94 | +0 |
    | 상위 75%|501 |498 |498 | +0 |

  train6047 앙상블에서도 같은 구조다 — 마진 하위 5%(302건)에 오답 94/178 = 53% 가 몰린다.
  세 지표(p1-p2, p1/p2, 엔트로피)가 같은 302건을 거의 같게 고르므로 지표 선택은 중요하지 않다.

  따라서 추가 추론은 전 문항 3배가 아니라 **선택분 3배 = 전체의 0.15배**면 된다.

왜 이득이 나는가 (위치 편향)
  train6047 에서 정답이 a 일 때 오답률 3.69%, d 3.72% 인데 c 는 1.86% 다. 모델이 a·d 를
  덜 고른다. 선택지를 순환시켜 평균하면 이 편향이 상쇄된다. 모델이 진짜로 확신하는
  문항은 순서를 바꿔도 답이 안 바뀌므로 고마진 구간에서 Δ0 인 것도 같은 설명이다.

주의
  위 val667 근거는 `ours` 프롬프트 35B(0.945) 에서 나왔다. 최종 구성은 read+tiles(0.9655)라
  **높은 정확도 구간에서도 같은 이득이 나오는지는 미확인**이다. 그걸 재는 것이 이 실험이다.

사용
  python tools/select_low_margin.py --pred A.jsonl --pred B.jsonl --pct 5 --out runs/shift_tta_train6047
  결과: <out>/ids.json (추론기 --ids-json 에 그대로 넣는다) + manifest.json + 요약표
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np

LETTERS = "abcd"


def load_preds(path):
    """id -> {logprobs, pred, gold?}. 우리 jsonl 과 팀원 Gemma csv(logit_a..d) 둘 다 읽는다."""
    out = {}
    if path.lower().endswith(".csv"):
        import csv as _csv
        with open(path, encoding="utf-8-sig", newline="") as fh:
            for r in _csv.DictReader(fh):
                key = r.get("id") or r.get("row_id") or r.get("image")
                lp = [float(r[f"logit_{c}"]) for c in LETTERS]
                out[key] = {"logprobs": lp, "pred": r.get("pred"), "gold": r.get("gold") or r.get("answer")}
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                o = json.loads(line)
                out[o["id"]] = o
    return out


def lsm(v):
    """a~d 안에서만 정규화한 로그확률. -1e4 는 '상위 20 밖' 표시값이라 잘라낸다."""
    a = np.clip(np.array(v, float), -1e4, None)
    m = a.max()
    e = np.exp(a - m)
    return np.log(np.maximum(e / e.sum(), 1e-300))


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", action="append", required=True,
                    help="예측 파일. 여러 번 주면 로그확률을 log-softmax 후 가중 평균(제출 규칙과 같다)")
    ap.add_argument("--weights", default=None, help="예: 0.5,0.5 (기본: 균등)")
    ap.add_argument("--pct", type=float, default=5.0, help="마진 하위 몇 %% 를 고를지")
    ap.add_argument("--n", type=int, default=None, help="개수로 직접 지정(--pct 보다 우선)")
    ap.add_argument("--metric", default="margin", choices=["margin", "ratio", "entropy"],
                    help="margin=p1-p2 (기본), ratio=p1/p2, entropy=a~d 엔트로피")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    preds = [load_preds(p) for p in a.pred]
    w = [float(x) for x in a.weights.split(",")] if a.weights else [1.0 / len(preds)] * len(preds)
    assert len(w) == len(preds), "--weights 개수가 --pred 개수와 다르다"
    ids = sorted(set.intersection(*[set(p) for p in preds]))
    assert ids, "예측 파일들의 공통 id 가 없다"

    rows = []
    for i in ids:
        s = sum(wi * lsm(p[i]["logprobs"]) for wi, p in zip(w, preds))
        prob = np.exp(s - s.max())
        prob = prob / prob.sum()
        top = np.sort(prob)[::-1]
        score = {"margin": top[0] - top[1],
                 "ratio": top[0] / max(top[1], 1e-300),
                 "entropy": -float((prob * np.log(prob + 1e-300)).sum())}[a.metric]
        # 엔트로피는 클수록 불확실하므로 정렬 방향을 뒤집는다
        rows.append((i, LETTERS[int(np.argmax(s))], -score if a.metric == "entropy" else score))

    rows.sort(key=lambda r: r[2])
    k = a.n if a.n is not None else int(round(len(rows) * a.pct / 100.0))
    sel = rows[:k]
    sel_ids = sorted(r[0] for r in sel)

    os.makedirs(a.out, exist_ok=True)
    ids_path = os.path.join(a.out, "ids.json")
    with open(ids_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(sel_ids, fh, ensure_ascii=False, indent=0)

    # gold 가 있으면 "이 선택이 오답을 얼마나 담는지"를 같이 보고한다
    gold = {i: preds[0][i].get("gold") for i in ids if preds[0][i].get("gold")}
    cover = {}
    if len(gold) == len(ids):
        err_all = sum(r[1] != gold[r[0]] for r in rows)
        err_sel = sum(r[1] != gold[r[0]] for r in sel)
        cover = {"errors_total": int(err_all), "errors_in_selection": int(err_sel),
                 "coverage": round(err_sel / err_all, 4) if err_all else None}

    man = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": "tools/select_low_margin.py",
        "metric": a.metric,
        "pct": a.pct, "n_selected": len(sel_ids), "n_pool": len(ids),
        "weights": w,
        "inputs": [{"path": p, "sha256": sha256(p), "n": len(d)} for p, d in zip(a.pred, preds)],
        "ids_sha256": hashlib.sha256("\n".join(sel_ids).encode()).hexdigest(),
        "margin_cutoff": float(sel[-1][2]) if sel else None,
        **cover,
    }
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(man, fh, ensure_ascii=False, indent=2)

    print(f"선택 {len(sel_ids)} / 전체 {len(ids)} ({len(sel_ids)/len(ids):.1%}), 지표 {a.metric}")
    if cover:
        print(f"이 선택이 담는 오답: {cover['errors_in_selection']} / {cover['errors_total']} "
              f"= {cover['coverage']:.0%}")
    print(f"추가 추론량: 선택 {len(sel_ids)} x 3순서 = {len(sel_ids)*3} 회 "
          f"(전 문항 3배 대비 {len(sel_ids)*3/(len(ids)*3):.1%})")
    print(f"저장: {ids_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
