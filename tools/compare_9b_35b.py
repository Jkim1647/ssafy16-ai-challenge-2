"""val667 9B vs 35B paired comparison + dev relabel filter with 35B.

usage: python tools/compare_9b_35b.py <35b_val667.jsonl or -> <35b_dev.jsonl or ->
dev 결과는 입력 파일 옆 dev_label_provisional_35b.csv로 저장한다(대회 데이터 포함 — git 미추적 위치에 둘 것).
"""
import json
import math
import random
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
NINE_VAL = REPO / "runs/20260921-165120_9b_zs_orig_g667/predictions.jsonl"
DEV_LABELS = REPO / "teammate_handoff/merged/dev_label_provisional.csv"


def load(p):
    return {r["id"]: r for r in map(json.loads, Path(p).read_text(encoding="utf-8").splitlines()) if r}


def softmax(lp):
    m = max(lp)
    e = [math.exp(x - m) for x in lp]
    z = sum(e)
    return [x / z for x in e]


def val_compare(b35_path):
    n9, n35 = load(NINE_VAL), load(b35_path)
    ids = sorted(set(n9) & set(n35))
    print(f"[val667] 9B {len(n9)} / 35B {len(n35)} / common {len(ids)}")
    c9 = [n9[i]["pred"] == n9[i]["gold"] for i in ids]
    c35 = [n35[i]["pred"] == n9[i]["gold"] for i in ids]
    n = len(ids)
    print(f"  acc 9B {sum(c9)/n:.4f} ({sum(c9)}/{n})  35B {sum(c35)/n:.4f} ({sum(c35)}/{n})")
    new_correct = [i for i, a, b in zip(ids, c9, c35) if b and not a]
    new_wrong = [i for i, a, b in zip(ids, c9, c35) if a and not b]
    both_wrong = [i for i, a, b in zip(ids, c9, c35) if not a and not b]
    print(f"  New Correct(35B O,9B X) {len(new_correct)}  New Wrong(35B X,9B O) {len(new_wrong)}  both wrong {len(both_wrong)}")
    diffs = [int(b) - int(a) for a, b in zip(c9, c35)]
    rng = random.Random(0)
    boots = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(5000))
    print(f"  delta {sum(diffs)/n:+.4f}  paired bootstrap 95% CI [{boots[125]:+.4f}, {boots[4875]:+.4f}]")
    for w in (0.3, 0.5, 0.7):
        ens = []
        for i in ids:
            p9, p35 = softmax(n9[i]["logprobs"]), softmax(n35[i]["logprobs"])
            p = [(1 - w) * a + w * b for a, b in zip(p9, p35)]
            ens.append("abcd"[p.index(max(p))] == n9[i]["gold"])
        print(f"  prob-avg ensemble w35={w}: {sum(ens)/n:.4f}  (참고: val667에서 w를 고르면 과적합)")
    agree = sum(n9[i]["pred"] == n35[i]["pred"] for i in ids)
    agree_ok = sum(n9[i]["pred"] == n35[i]["pred"] == n9[i]["gold"] for i in ids)
    print(f"  두 모델 일치 {agree}/{n}, 일치 시 정확도 {agree_ok/agree:.4f}")
    return {"new_correct": new_correct, "new_wrong": new_wrong, "both_wrong": both_wrong}


def dev_filter(b35_path):
    n35 = load(b35_path)
    df = pd.read_csv(DEV_LABELS, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df["b35_answer"] = df.id.map(lambda i: n35.get(i, {}).get("pred", ""))
    df["b35_conf"] = df.id.map(lambda i: n35.get(i, {}).get("confidence", ""))
    print(f"[dev] labels {len(df)} / 35B preds {len(n35)} / matched {(df.b35_answer != '').sum()}")
    hold = df.provisional_use == "hold_until_35b"
    same = hold & (df.b35_answer == df.final_answer)
    print(f"  hold_until_35b {hold.sum()}: 35B 동일 {same.sum()} -> train_ok, 불일치 {(hold & ~same).sum()} -> exclude")
    ok = df.provisional_use == "train_ok"
    print(f"  기존 train_ok {ok.sum()} 중 35B 동일 {(ok & (df.b35_answer == df.final_answer)).sum()}")
    ex = df.provisional_use == "exclude"
    tri = ex & (df.claude_answer != "") & (df.claude_answer == df.nine_answer) & (df.nine_answer == df.b35_answer)
    duo = ex & (df.claude_answer == "") & (df.nine_answer == df.b35_answer)
    print(f"  exclude {ex.sum()}: Claude·9B·35B 3자 일치 {tri.sum()}, (Claude 답 없음) 9B·35B 일치 {duo.sum()}")
    df["provisional_use_35b"] = df.provisional_use
    df.loc[same, "provisional_use_35b"] = "train_ok"
    df.loc[hold & ~same, "provisional_use_35b"] = "exclude_35b_disagree"
    print(df.provisional_use_35b.value_counts().to_string())
    return df


if __name__ == "__main__":
    out = {}
    if sys.argv[1] != "-":
        out.update(val_compare(sys.argv[1]))
    if len(sys.argv) > 2 and sys.argv[2] != "-":
        df = dev_filter(sys.argv[2])
        df.to_csv(Path(sys.argv[2]).with_name("dev_label_provisional_35b.csv"), index=False, encoding="utf-8-sig")
    Path(sys.argv[1] if sys.argv[1] != "-" else sys.argv[2]).with_name("val667_diff_ids.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
