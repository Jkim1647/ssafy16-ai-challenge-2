"""35B+397B 앙상블에서 두 모델 답이 갈린 문항만 397B 추가 시점(선택지 순환·원본·3배 타일)으로 보강했을 때의 효과를 잰다.

usage: python tools/eval_disagree_tta.py [--write-test out.csv --combo NAME]
불일치 문항 외에는 기본 앙상블과 동일하다. 조합마다 val667·H184 정답 수를 출력한다.
"""
import itertools
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
P = REPO / "shared_predictions"
DIS = P / "nebius_397b_dis"
VARIANTS = {"s1": "tiles2x2x2_shift1", "s2": "tiles2x2x2_shift2", "s3": "tiles2x2x2_shift3", "full": None, "t3": "tiles2x2x3"}


def load(p):
    return {r["id"]: r for r in map(json.loads, Path(p).read_text(encoding="utf-8").splitlines()) if r}


def norm(lp):
    m = max(lp)
    z = math.log(sum(math.exp(x - m) for x in lp)) + m
    return [x - z for x in lp]


def variant_file(key):
    tag = VARIANTS[key]
    name = f"Qwen3.5-397B-A17B-FP8_read_devDIS_{tag}_vllm_predictions.jsonl" if tag else "Qwen3.5-397B-A17B-FP8_read_devDIS_vllm_predictions.jsonl"
    return DIS / name


def decide(i, b35, b397, extra, keys):
    """397B 점수 = 기본 397B와 선택한 추가 시점의 정규화 로그확률 평균. 그다음 35B와 0.5:0.5."""
    views = [norm(b397[i]["logprobs"])] + [norm(extra[k][i]["logprobs"]) for k in keys if i in extra[k]]
    s397 = [sum(v[c] for v in views) / len(views) for c in range(4)]
    s = [0.5 * a + 0.5 * b for a, b in zip(norm(b35[i]["logprobs"]), s397)]
    return "abcd"[s.index(max(s))]


def main():
    extra = {k: load(variant_file(k)) for k in VARIANTS if variant_file(k).exists()}
    v35 = load(P / "colab_ai2_35b_out/Qwen3.6-35B-A3B_read_val667_tiles2x2x2_predictions.jsonl")
    v397 = load(P / "nebius_397b/Qwen3.5-397B-A17B-FP8_read_val667_tiles2x2x2_vllm_predictions.jsonl")
    h35 = load(P / "colab_ai2_35b_out/Qwen3.6-35B-A3B_read_devH184_tiles2x2x2_predictions.jsonl")
    h397 = load(P / "nebius_397b/Qwen3.5-397B-A17B-FP8_read_devH331_tiles2x2x2_vllm_predictions.jsonl")
    g2 = json.loads((REPO / "runs/dev_validation_v1/hard_eval_gold_v2.json").read_text(encoding="utf-8-sig"))
    vg = {i: v397[i]["gold"] for i in v397}
    avail = sorted(extra)
    print("available variants:", avail, {k: len(v) for k, v in extra.items()})
    combos = [()]
    for r in range(1, len(avail) + 1):
        combos += list(itertools.combinations(avail, r))
    results = {}
    for keys in combos:
        va = sum(decide(i, v35, v397, extra, keys) == vg[i] for i in v397)
        ha = sum(decide(i, h35, h397, extra, keys) == g2[i] for i in g2)
        name = "+".join(keys) or "base"
        results[name] = keys
        print(f"{name:22s} val667 {va}/{len(v397)} ({va / len(v397):.4f})  H184 {ha}/{len(g2)} ({ha / len(g2):.4f})")
    if "--write-test" in sys.argv:
        out = sys.argv[sys.argv.index("--write-test") + 1]
        keys = results[sys.argv[sys.argv.index("--combo") + 1]]
        import pandas as pd
        t35 = load(P / "colab_ai2_35b_out/Qwen3.6-35B-A3B_read_test_tiles2x2x2_predictions.jsonl")
        t397 = load(P / "nebius_397b_test/Qwen3.5-397B-A17B-FP8_read_test_tiles2x2x2_vllm_predictions.jsonl")
        sample = pd.read_csv(REPO / "ssafy-16-2-ai/sample_submission.csv", dtype=str)
        ans = [decide(i, t35, t397, extra, keys) for i in sample.id]
        pd.DataFrame({"id": sample.id, "answer": ans}).to_csv(out, index=False, encoding="utf-8")  # 제출 CSV는 BOM 없이
        print("wrote", out, "keys", keys)


if __name__ == "__main__":
    main()
