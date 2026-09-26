"""35B 2단계 지시 v1/v2(같은 판독문) 비교와 397B 앙상블 효과 평가.

입력: --read2 both 출력 jsonl(행 = v1, 행["v2"] = v2), 397B val667/H331 예측, 35B 기존(Colab) 예측, gold.
출력: 표준출력 표 + JSON(--out). 판정 규칙은 실행 전에 고정했다(아래 RULE).
usage: python tools/eval_read2_compare.py <both.jsonl> --out reports/.../read2_compare.json
"""
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
P = REPO / "shared_predictions"
RULE = ("앙상블(0.5·35B + 0.5·397B, 정규화 로그확률)에서 v2가 v1 대비 val667 NetGain>0 이고 H331 NetGain>=0, "
        "보조지표 val667∖진단15(652)·H331∖H184(147)에서 하락 없음 → 채택")


def load(p):
    return {r["id"]: r for r in map(json.loads, Path(p).read_text(encoding="utf-8").splitlines()) if r}


def norm(lp):
    m = max(lp)
    z = math.log(sum(math.exp(x - m) for x in lp)) + m
    return [x - z for x in lp]


def ens(a, b):
    s = [0.5 * x + 0.5 * y for x, y in zip(norm(a), norm(b))]
    return "abcd"[s.index(max(s))]


def main():
    both = load(sys.argv[1])
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    v397 = load(P / "nebius_397b/Qwen3.5-397B-A17B-FP8_read_val667_tiles2x2x2_vllm_predictions.jsonl")
    h397 = load(P / "nebius_397b/Qwen3.5-397B-A17B-FP8_read_devH331_tiles2x2x2_vllm_predictions.jsonl")
    old35 = load(P / "colab_ai2_35b_out/Qwen3.6-35B-A3B_read_val667_tiles2x2x2_predictions.jsonl") | \
        load(P / "colab_ai2_35b_out/Qwen3.6-35B-A3B_read_devH184_tiles2x2x2_predictions.jsonl")
    g3 = json.loads((REPO / "runs/dev_validation_v1/hard_eval_gold_v3.json").read_text(encoding="utf-8-sig"))
    g2 = json.loads((REPO / "runs/dev_validation_v1/hard_eval_gold_v2.json").read_text(encoding="utf-8-sig"))
    diag = {l.split(",")[1] for l in (REPO / "reports/nebius_397b/common_errors_31_reviewed.csv").read_text(encoding="utf-8-sig").splitlines()[1:]}
    gold = {i: v397[i]["gold"] for i in v397} | g3
    b397 = v397 | h397
    sets = {"val667": sorted(v397), "val667_minus_diag15": sorted(set(v397) - diag),
            "H331": sorted(g3), "H184": sorted(g2), "H331_minus_H184": sorted(set(g3) - set(g2))}
    missing = sorted(i for ids in sets.values() for i in ids if i not in both)
    assert not missing, f"both 파일에 없는 평가 문항 {len(missing)}: {missing[:5]}"
    res = {"rule": RULE, "n_both_rows": len(both), "sets": {}}
    for name, ids in sets.items():
        p1 = {i: both[i]["pred"] for i in ids}
        p2 = {i: both[i]["v2"]["pred"] for i in ids}
        e1 = {i: ens(both[i]["logprobs"], b397[i]["logprobs"]) for i in ids}
        e2 = {i: ens(both[i]["v2"]["logprobs"], b397[i]["logprobs"]) for i in ids}
        acc = lambda d: sum(d[i] == gold[i] for i in ids)  # noqa: E731
        nc = sorted(i for i in ids if e1[i] != gold[i] and e2[i] == gold[i])
        nw = sorted(i for i in ids if e1[i] == gold[i] and e2[i] != gold[i])
        snc = sorted(i for i in ids if p1[i] != gold[i] and p2[i] == gold[i])
        snw = sorted(i for i in ids if p1[i] == gold[i] and p2[i] != gold[i])
        r = {"n": len(ids), "35B_v1": acc(p1), "35B_v2": acc(p2), "ens_v1": acc(e1), "ens_v2": acc(e2),
             "single_NC": len(snc), "single_NW": len(snw), "ens_NC": len(nc), "ens_NW": len(nw), "ens_net": len(nc) - len(nw),
             "ens_NC_ids": nc, "ens_NW_ids": nw, "single_changed": sum(p1[i] != p2[i] for i in ids)}
        old_ids = [i for i in ids if i in old35]
        if old_ids:  # 재현성: 같은 v1 설정, Colab(HF) vs RunPod(vLLM)
            r["repro_v1_vs_colab_same_pred"] = f"{sum(old35[i]['pred'] == p1[i] for i in old_ids)}/{len(old_ids)}"
            r["colab_35B_acc"] = f"{sum(old35[i]['pred'] == gold[i] for i in old_ids)}/{len(old_ids)}"
        res["sets"][name] = r
        print(f"{name:20s} n={len(ids):3d} 35B v1 {r['35B_v1']:3d} v2 {r['35B_v2']:3d} (NC {len(snc)} NW {len(snw)}) | "
              f"ens v1 {r['ens_v1']:3d} v2 {r['ens_v2']:3d} (NC {len(nc)} NW {len(nw)} net {len(nc) - len(nw):+d})"
              + (f" | colab재현 {r.get('repro_v1_vs_colab_same_pred')} colab정답 {r.get('colab_35B_acc')}" if old_ids else ""))
    s = res["sets"]
    res["adopt_v2"] = (s["val667"]["ens_net"] > 0 and s["H331"]["ens_net"] >= 0
                       and s["val667_minus_diag15"]["ens_net"] >= 0 and s["H331_minus_H184"]["ens_net"] >= 0)
    print("RULE:", RULE, "→ adopt_v2 =", res["adopt_v2"])
    if out:
        Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
