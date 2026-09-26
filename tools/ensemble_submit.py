"""Equal-weight probability-average ensemble of prediction files -> submission csv.

usage: python tools/ensemble_submit.py out.csv a.jsonl b.csv [...] [--weights .5,.5] [--logprob] [--vote] [--allow-partial]
jsonl 각 행은 id와 logprobs(a~d 원래 순서)를 가져야 한다. 행 순서는 sample_submission.csv를 따른다.

  --logprob       확률 대신 정규화 로그확률(기하평균)을 가중합
  --vote          하드 다수결. 동률이면 로그확률 합으로 가른다
  --allow-partial 일부 id 가 없는 모델을 허용한다(그 id 에서는 빼고 남은 가중치를 재정규화).
                  팀원 Gemma test 예측이 test_0001~1254 (18.7%) 뿐이라 필요하다.
                  **이때 점수 차이는 덮은 구간에서만 나오므로 그만큼 희석돼 읽힌다.**

팀원 Gemma 형식(csv: id, prediction, logit_a..d, prob_a..d)도 읽는다.
"""
import json
import math
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SAMPLE = next((p for p in (REPO / "ssafy-16-2-ai/sample_submission.csv", REPO / "data/sample_submission.csv") if p.exists()),
              REPO / "data/sample_submission.csv")


def load_any(f):
    """jsonl(id, logprobs) 과 팀원 Gemma csv(id, logit_a..d) 를 모두 id -> {pred, logprobs} 로 읽는다."""
    path = Path(f)
    if path.suffix.lower() == ".csv":
        import csv as _csv
        out = {}
        with path.open(encoding="utf-8-sig", newline="") as fh:
            for r in _csv.DictReader(fh):
                key = r.get("id") or r.get("row_id")
                lp = [float(r[f"logit_{c}"]) for c in "abcd"]
                out[key] = {"pred": (r.get("prediction") or r.get("pred") or "").strip().lower(), "logprobs": lp}
        return out
    return {r["id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r}


def softmax(lp):
    m = max(lp)
    e = [math.exp(x - m) for x in lp]
    z = sum(e)
    return [x / z for x in e]


def main():
    args = sys.argv[1:]
    weights = None
    # --logprob: 확률 대신 정규화 로그확률(기하평균)을 가중합한다. 35B+397B에서 H184가 156→158로 더 좋았다.
    use_logprob = "--logprob" in args
    use_vote = "--vote" in args
    allow_partial = "--allow-partial" in args
    args = [a for a in args if a not in ("--logprob", "--vote", "--allow-partial")]
    if "--weights" in args:
        i = args.index("--weights")
        weights = [float(w) for w in args[i + 1].split(",")]
        args = args[:i] + args[i + 2:]
    out, files = Path(args[0]), args[1:]
    weights = weights or [1 / len(files)] * len(files)
    preds = [load_any(f) for f in files]
    sample = pd.read_csv(SAMPLE, dtype=str)
    answers, missing = [], 0
    partial_ids = 0
    for i in sample.id:
        have = [(w, pr[i]) for w, pr in zip(weights, preds) if i in pr]
        if len(have) < len(preds):
            (partial_ids := partial_ids + 1) if allow_partial else None
            missing += len(preds) - len(have)
        if not have:
            raise SystemExit(f"{i}: 어느 모델에도 예측이 없다")
        # 빠진 모델이 있으면 남은 가중치를 재정규화한다. 안 그러면 그 id 만 전체 신뢰도가 낮아진다
        z = sum(w for w, _ in have)
        p = [0.0] * 4
        votes = [0.0] * 4
        for w, r in have:
            q = softmax(r["logprobs"])
            p = [a + (w / z) * (math.log(max(b, 1e-300)) if use_logprob else b) for a, b in zip(p, q)]
            votes[max(range(4), key=lambda k: q[k])] += w / z
        if use_vote:
            top = max(votes)
            tied = [k for k in range(4) if votes[k] == top]
            best = tied[0] if len(tied) == 1 else max(tied, key=lambda k: p[k])  # 동률은 로그확률 합으로
            answers.append("abcd"[best])
        else:
            answers.append("abcd"[p.index(max(p))])
    if missing and not allow_partial:
        raise SystemExit(f"missing {missing} predictions (부분 예측을 쓰려면 --allow-partial)")
    if partial_ids:
        print(f"부분 커버: {partial_ids}/{len(sample)} 행에서 일부 모델이 빠졌다(가중치 재정규화)")
    sub = pd.DataFrame({"id": sample.id, "answer": answers})
    assert len(sub) == len(sample) and sub.answer.isin(list("abcd")).all()
    sub.to_csv(out, index=False, encoding="utf-8")  # 제출 CSV는 BOM 없이(채점기 ID 매칭)
    for f, pr in zip(files, preds):
        common = [(i, a) for i, a in zip(sample.id, answers) if i in pr]
        agree = sum(pr[i]["pred"] == a for i, a in common)
        print(f"{Path(f).parent.name}/{Path(f).name}: agrees {agree}/{len(common)}")
    print("answer dist", sub.answer.value_counts().to_dict(), "->", out)


if __name__ == "__main__":
    main()
