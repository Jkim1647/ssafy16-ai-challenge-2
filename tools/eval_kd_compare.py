"""9B LoRA CE-only vs CE+KD 비교 + 현재 최고 앙상블(35B Colab + 397B) 보완성 평가.

규칙(결과 보기 전 고정): 3모델 동일가중(1/3) 로그확률 앙상블(35B Colab + 397B + 9B-KD)이
현재 제출(35B+397B 0.5:0.5, val667 648 / H184 158)보다 val667과 H184 둘 다 높을 때만 제출 후보. 가중치 탐색 없음.
usage: python tools/eval_kd_compare.py <out_kd_compare 디렉터리> [--out result.json]
"""
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
P = REPO / "shared_predictions"
RULE = "3모델 1/3 로그확률 앙상블(35B Colab+397B+9B-KD)이 val667·H184 둘 다 현재 제출(648/158)보다 높으면 제출 후보"


def load(p):
    return {r["id"]: r for r in map(json.loads, Path(p).read_text(encoding="utf-8").splitlines()) if r}


def norm(lp):
    m = max(lp)
    z = math.log(sum(math.exp(x - m) for x in lp)) + m
    return [x - z for x in lp]


def comb(*lps):
    s = [sum(norm(lp)[c] for lp in lps) / len(lps) for c in range(4)]
    return "abcd"[s.index(max(s))]


def main():
    d = Path(sys.argv[1])
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    g3 = json.loads((REPO / "runs/dev_validation_v1/hard_eval_gold_v3.json").read_text(encoding="utf-8-sig"))
    g2 = json.loads((REPO / "runs/dev_validation_v1/hard_eval_gold_v2.json").read_text(encoding="utf-8-sig"))
    v397 = load(P / "nebius_397b/Qwen3.5-397B-A17B-FP8_read_val667_tiles2x2x2_vllm_predictions.jsonl")
    h397 = load(P / "nebius_397b/Qwen3.5-397B-A17B-FP8_read_devH331_tiles2x2x2_vllm_predictions.jsonl")
    c35 = load(P / "colab_ai2_35b_out/Qwen3.6-35B-A3B_read_val667_tiles2x2x2_predictions.jsonl") | \
        load(P / "colab_ai2_35b_out/Qwen3.6-35B-A3B_read_devH184_tiles2x2x2_predictions.jsonl")
    gold = {i: v397[i]["gold"] for i in v397} | g3
    b397 = v397 | h397
    runs = {}
    for name in ("ce", "kd"):
        runs[name] = load(d / name / "val_predictions.jsonl") | load(d / name / "hard_predictions.jsonl")
    sets = {"val667": sorted(v397), "H331": sorted(g3), "H184": sorted(g2), "H331_minus_H184": sorted(set(g3) - set(g2))}
    res = {"rule": RULE, "sets": {}}
    for sname, ids in sets.items():
        ce, kd = runs["ce"], runs["kd"]
        assert all(i in ce and i in kd for i in ids), f"{sname}: 예측 누락"
        acc = lambda f: sum(f(i) == gold[i] for i in ids)  # noqa: E731
        nc = sorted(i for i in ids if ce[i]["pred"] != gold[i] and kd[i]["pred"] == gold[i])
        nw = sorted(i for i in ids if ce[i]["pred"] == gold[i] and kd[i]["pred"] != gold[i])
        r = {"n": len(ids), "9B_ce": acc(lambda i: ce[i]["pred"]), "9B_kd": acc(lambda i: kd[i]["pred"]),
             "kd_vs_ce_NC": len(nc), "kd_vs_ce_NW": len(nw), "kd_vs_ce_net": len(nc) - len(nw), "NC_ids": nc, "NW_ids": nw}
        if all(i in c35 for i in ids):  # 35B Colab 예측이 있는 셋(val667, H184)
            base = {i: comb(c35[i]["logprobs"], b397[i]["logprobs"]) for i in ids}
            tri = {i: comb(c35[i]["logprobs"], b397[i]["logprobs"], kd[i]["logprobs"]) for i in ids}
            r |= {"current_ens": acc(lambda i: base[i]), "tri_ens_with_kd": acc(lambda i: tri[i]),
                  "tri_NC": sum(base[i] != gold[i] and tri[i] == gold[i] for i in ids),
                  "tri_NW": sum(base[i] == gold[i] and tri[i] != gold[i] for i in ids),
                  "kd_right_where_current_wrong": sum(base[i] != gold[i] and kd[i]["pred"] == gold[i] for i in ids)}
        res["sets"][sname] = r
        print(sname, {k: v for k, v in r.items() if not k.endswith("_ids")})
    s = res["sets"]
    res["submit_candidate"] = (s["val667"]["tri_ens_with_kd"] > s["val667"]["current_ens"]
                               and s["H184"]["tri_ens_with_kd"] > s["H184"]["current_ens"])
    print("RULE:", RULE, "→ submit_candidate =", res["submit_candidate"])
    for name in ("ce", "kd"):
        m = json.loads((d / name / "meta.json").read_text(encoding="utf-8"))
        res[f"{name}_meta"] = {k: m.get(k) for k in ("train_sec", "val_acc", "hard_acc", "fit", "steps", "kd", "versions")}
    if out:
        Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
