"""불일치 재검증(①선택지 순환 ②원본 1장 ③타일 3배) 결과를 재사용 가능한 산출물로 정리한다(추가 추론 없음).

입력: shared_predictions/nebius_397b_dis/*_predictions.jsonl (VM 원본 출력, logprobs는 원래 a~d 순서로 되돌린 값)
출력: shared_predictions/nebius_397b_dis/derived/
  - {variant}_enriched.jsonl : 원 행 + gold(있을 때만) + 입력 조건 + (순환이면) 화면 순서 logprobs·복원 매핑
  - rotation_combined.jsonl  : 순환 s1~s3(+기본 s0) 로그확률 평균·확률 평균을 모두 기록
  - comparison.json          : val667/H184/test별 New Correct·New Wrong·NetGain·변경 ID
  - meta_{variant}.json      : 모델 revision, 프롬프트 hash, 입력 설정, ID hash 규칙, 코드 commit, 성공/누락 수
usage: python tools/build_dis_artifacts.py
"""
import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from colab_35b_infer import READ_1, READ_2, TILES_NOTE  # noqa: E402  실제 실행에 쓰인 프롬프트 상수

P = REPO / "shared_predictions"
DIS = P / "nebius_397b_dis"
OUT = DIS / "derived"
MISSING = -1e4  # colab_vllm_infer.py: 상위 20 logprobs 밖이면 -1e4로 기록(추정값 아님, '없음' 표시)
REV = "ea5b4f81096f3901c91dea97f81324302495781d"
VARIANTS = {
    "s1": dict(file="tiles2x2x2_shift1", view="tiles2x2", tile_scale=2.0, shift=1, label="① 선택지 순환 shift1"),
    "s2": dict(file="tiles2x2x2_shift2", view="tiles2x2", tile_scale=2.0, shift=2, label="① 선택지 순환 shift2"),
    "s3": dict(file="tiles2x2x2_shift3", view="tiles2x2", tile_scale=2.0, shift=3, label="① 선택지 순환 shift3"),
    "full": dict(file=None, view="full", tile_scale=None, shift=0, label="② 원본 1장"),
    "t3": dict(file="tiles2x2x3", view="tiles2x2", tile_scale=3.0, shift=0, label="③ 타일 3배"),
}


def load(p):
    return {r["id"]: r for r in map(json.loads, Path(p).read_text(encoding="utf-8").splitlines()) if r}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def id_hash(ids):
    # colab_vllm_infer.py와 같은 규칙: id 오름차순 정렬 → "\n" 연결 → UTF-8 → SHA-256
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def norm(lp):
    m = max(lp)
    z = math.log(sum(math.exp(x - m) for x in lp)) + m
    return [x - z for x in lp]


def vfile(key):
    tag = VARIANTS[key]["file"]
    return DIS / (f"Qwen3.5-397B-A17B-FP8_read_devDIS_{tag}_vllm_predictions.jsonl" if tag
                  else "Qwen3.5-397B-A17B-FP8_read_devDIS_vllm_predictions.jsonl")


def main():
    OUT.mkdir(exist_ok=True)
    v397 = load(P / "nebius_397b/Qwen3.5-397B-A17B-FP8_read_val667_tiles2x2x2_vllm_predictions.jsonl")
    h397 = load(P / "nebius_397b/Qwen3.5-397B-A17B-FP8_read_devH331_tiles2x2x2_vllm_predictions.jsonl")
    t397 = load(P / "nebius_397b_test/Qwen3.5-397B-A17B-FP8_read_test_tiles2x2x2_vllm_predictions.jsonl")
    v35 = load(P / "colab_ai2_35b_out/Qwen3.6-35B-A3B_read_val667_tiles2x2x2_predictions.jsonl")
    h35 = load(P / "colab_ai2_35b_out/Qwen3.6-35B-A3B_read_devH184_tiles2x2x2_predictions.jsonl")
    t35 = load(P / "colab_ai2_35b_out/Qwen3.6-35B-A3B_read_test_tiles2x2x2_predictions.jsonl")
    g2 = json.loads((REPO / "runs/dev_validation_v1/hard_eval_gold_v2.json").read_text(encoding="utf-8-sig"))
    gold = {i: v397[i]["gold"] for i in v397} | g2  # test 정답은 없으므로 넣지 않는다
    subset = {"val667": set(v397), "H184": set(g2), "test": set(t397)}
    base397 = v397 | h397 | t397
    base35 = v35 | h35 | t35
    runs = {k: load(vfile(k)) for k in VARIANTS if vfile(k).exists()}
    prompt_text = {"read_stage1": READ_1, "read_stage2": READ_2, "tiles_note": TILES_NOTE}

    for k, rows in runs.items():
        cfg = VARIANTS[k]
        with (OUT / f"{k}_enriched.jsonl").open("w", encoding="utf-8") as fh:
            for i, r in rows.items():
                row = dict(r)
                row["gold"] = gold.get(i)
                row["set"] = next((s for s, ids in subset.items() if i in ids), None)
                row["input"] = {"variant": cfg["label"], "view": cfg["view"], "tile_scale": cfg["tile_scale"], "choice_shift": cfg["shift"]}
                row["logprobs_scope"] = "full-vocabulary next-token logprob (not renormalized); -1e4 = letter outside top-20"
                row["confidence_def"] = "max softmax over the 4 letter logprobs (renormalized within a-d)"
                if cfg["shift"]:
                    s = cfg["shift"]
                    row["shown_order"] = [f"{'abcd'[j]}<-{'abcd'[(j + s) % 4]}" for j in range(4)]
                    # 저장값 logprobs[o]는 화면 위치 (o-s)%4의 값이다 → 화면 순서 원값을 역산(순열이라 손실 없음)
                    row["logprobs_shown_order"] = [r["logprobs"][(j + s) % 4] for j in range(4)]
                    row["logprobs_original_order"] = r["logprobs"]
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        ids = list(rows)
        miss = [i for i in ids if MISSING in rows[i]["logprobs"]]
        meta_src = json.loads(Path(str(vfile(k)).replace("_predictions.jsonl", "_meta.json")).read_text(encoding="utf-8"))
        meta = {
            "variant": k, "label": cfg["label"], "model": "Qwen/Qwen3.5-397B-A17B-FP8", "revision": REV,
            "engine": "vllm 0.30.0 in-process, TP=8, 8x H100 80GB (Nebius eu-north1), temperature 0",
            "prompt": prompt_text, "prompt_source": "tools/colab_35b_infer.py (READ_1/READ_2/TILES_NOTE)",
            "prompt_source_sha256": sha(REPO / "tools/colab_35b_infer.py"),
            "input": {"view": cfg["view"], "tile_scale": cfg["tile_scale"], "choice_shift": cfg["shift"], "read_tokens": 80,
                      "max_model_len": meta_src.get("max_model_len")},
            "code": {"infer": "tools/colab_vllm_infer.py", "commit": "228d039 (VM 파일 sha256 앞16자 9c139d59703eb077로 일치 확인)",
                     "driver": "tools/nebius/run_shift_disagree.sh"},
            "ids": {"n": len(ids), "sha256": id_hash(ids), "rule": "sorted ascending ids joined by '\\n', UTF-8, SHA-256",
                    "source": "35B read+tiles vs 397B read+tiles pred 불일치: val667 20 + H184 26 + test 177"},
            "counts": {"success": len(ids), "missing_letter_logprob_rows": len(miss), "missing_ids": miss, "failed": 0},
            "vm_meta": meta_src, "raw_predictions": str(vfile(k).relative_to(REPO)), "raw_sha256": sha(vfile(k)),
        }
        (OUT / f"meta_{k}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # 순환 결합: s0(기본 397B) + 있는 shift들. 로그확률 평균(정규화 후)과 확률 평균을 모두 기록한다.
    rot = [k for k in ("s1", "s2", "s3") if k in runs]
    with (OUT / "rotation_combined.jsonl").open("w", encoding="utf-8") as fh:
        for i in runs[rot[0]]:
            views = {"s0": base397[i]["logprobs"]} | {k: runs[k][i]["logprobs"] for k in rot}
            nl = {k: norm(v) for k, v in views.items()}
            lp_mean = [sum(v[c] for v in nl.values()) / len(nl) for c in range(4)]
            pr_mean = [sum(math.exp(v[c]) for v in nl.values()) / len(nl) for c in range(4)]
            fh.write(json.dumps({"id": i, "gold": gold.get(i), "views_original_order": views,
                                 "mean_normalized_logprob": [round(x, 6) for x in lp_mean],
                                 "mean_prob": [round(x, 6) for x in pr_mean],
                                 "pred_logprob_mean": "abcd"[lp_mean.index(max(lp_mean))],
                                 "pred_prob_mean": "abcd"[pr_mean.index(max(pr_mean))]}, ensure_ascii=False) + "\n")

    # 비교: baseline = 35B+397B 로그확률 0.5:0.5(제출 ens_35b_397b_logprob_test.csv와 동일 규칙)
    def ens(i, extra_keys):
        v = [norm(base397[i]["logprobs"])] + [norm(runs[k][i]["logprobs"]) for k in extra_keys if i in runs[k]]
        s397 = [sum(x[c] for x in v) / len(v) for c in range(4)]
        s = [0.5 * a + 0.5 * b for a, b in zip(norm(base35[i]["logprobs"]), s397)]
        return "abcd"[s.index(max(s))]

    dis_ids = set(runs[next(iter(runs))])
    comp = {"baseline": {"rule": "per-model log-softmax over a-d, 0.5*35B + 0.5*397B, argmax",
                         "files": {"35B": "colab_ai2_35b_out/Qwen3.6-35B-A3B_read_{val667,devH184,test}_tiles2x2x2_predictions.jsonl",
                                   "397B": "nebius_397b/…_read_{val667,devH331}…, nebius_397b_test/…_read_test…"}},
            "variant_rule": "불일치 문항에서만 397B 점수 = 기본 397B와 해당 변형들의 정규화 로그확률 평균 → 35B와 0.5:0.5. 불일치 밖은 baseline과 동일",
            "results": {}}
    combos = [(k,) for k in runs] + [tuple(rot), ("full", "t3"), tuple(runs)]
    for keys in combos:
        name = "+".join(keys)
        res = {}
        for sname, ids in subset.items():
            sub = sorted(ids & dis_ids)
            base = {i: ens(i, ()) for i in ids}
            new = {i: ens(i, keys) for i in ids}
            changed = sorted(i for i in ids if base[i] != new[i])
            r = {"n_eval": len(ids), "n_disagreement_subset": len(sub), "changed": len(changed), "changed_ids": changed}
            if sname != "test":
                nc = sorted(i for i in ids if base[i] != gold[i] and new[i] == gold[i])
                nw = sorted(i for i in ids if base[i] == gold[i] and new[i] != gold[i])
                r |= {"new_correct": len(nc), "new_wrong": len(nw), "net_gain": len(nc) - len(nw),
                      "new_correct_ids": nc, "new_wrong_ids": nw,
                      "full_set_acc": f"{sum(new[i] == gold[i] for i in ids)}/{len(ids)}",
                      "baseline_full_set_acc": f"{sum(base[i] == gold[i] for i in ids)}/{len(ids)}",
                      "subset_acc_397B_variant_alone": f"{sum(('abcd'[max(range(4), key=lambda c: sum(norm(x)[c] for x in [base397[i]['logprobs']] + [runs[k][i]['logprobs'] for k in keys]))]) == gold[i] for i in sub)}/{len(sub)}",
                      "subset_acc_397B_base": f"{sum(base397[i]['pred'] == gold[i] for i in sub)}/{len(sub)}"}
            res[sname] = r
        comp["results"][name] = res
    (OUT / "comparison.json").write_text(json.dumps(comp, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, res in comp["results"].items():
        print(name, {s: {k: v for k, v in r.items() if not k.endswith("_ids")} for s, r in res.items()})


if __name__ == "__main__":
    main()
