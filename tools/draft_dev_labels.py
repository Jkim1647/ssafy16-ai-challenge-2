"""dev 재라벨링 초안: 로컬 Qwen3.5-9B로 각 문항의 후보 답과 확신도를 뽑는다.

teammate_handoff/TEAMMATE_HANDOFF.md의 "LLM 사용 규칙"을 따른다.
  - 모델에는 이미지·질문·a~d만 준다(answer1~5는 패킷에 없다).
  - 결과는 초안이다. 반환 파일 labels.csv는 건드리지 않고 옆에 llm_draft.csv로 따로 쓴다.
    모델 답이 human_answer 열에 섞이면 사람 판정과 구분할 수 없게 되기 때문이다.
  - calibration은 두 작업자가 기준을 맞추는 단계라 기본적으로 초안을 만들지 않는다.
  - 호스팅 API가 아니라 로컬 모델이다(대회 규칙의 API 추론 금지와 무관).

채점은 파이프라인과 같은 선택지 로그확률 방식(enable_thinking=false)이다.

실행:
    python tools/draft_dev_labels.py                       # teammate 1,316건
    python tools/draft_dev_labels.py --labels teammate_handoff/calibration/labels.csv --allow-calibration

중간에 끊겨도 다시 실행하면 이미 쓴 id는 건너뛴다.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MODEL_TAG = "Qwen3.5-9B-NF4-local"
FIELDS = ["id", "llm_answer", "llm_prob", "llm_margin", "p_a", "p_b", "p_c", "p_d", "llm_model"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="teammate_handoff/teammate/labels.csv")
    ap.add_argument("--out", default=None, help="기본: labels.csv 옆 llm_draft.csv")
    ap.add_argument("--config", default="9b_zs_orig", help="모델·입력 설정(원본 해상도)")
    ap.add_argument("--allow-calibration", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import pandas as pd

    from src.config import load_config
    from src.data import Sample, build_views
    from src.models import build_adapter
    from src.models.base import softmax

    labels_path = (REPO / args.labels).resolve()
    out_path = Path(args.out).resolve() if args.out else labels_path.with_name("llm_draft.csv")

    df = pd.read_csv(labels_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    if (df["is_calibration"].str.lower() == "true").any() and not args.allow_calibration:
        print("calibration 행이 있다. 기준 맞추기 단계라 초안을 만들지 않는다 "
              "(필요하면 --allow-calibration).")
        return 2

    cfg = load_config(args.config, [])
    dev_dir = cfg.path("data.root")
    done: set[str] = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8-sig", newline="") as fh:
            done = {r["id"] for r in csv.DictReader(fh)}

    samples = [
        Sample(id=r["id"], image_path=dev_dir / r["path"], question=r["question"],
               choices=[r["a"], r["b"], r["c"], r["d"]])
        for _, r in df.iterrows() if r["id"] not in done
    ]
    if args.limit:
        samples = samples[: args.limit]
    missing = [s.id for s in samples if not s.image_path.exists()]
    if missing:
        print(f"이미지 없음 {len(missing)}건: {missing[:5]}")
        return 1
    print(f"대상 {len(samples)}건 (이미 완료 {len(done)}건) → {out_path}")
    if not samples:
        return 0

    adapter = build_adapter(cfg)
    adapter.ensure_loaded(for_training=False)

    new_file = not out_path.exists()
    t0 = time.time()
    # 사람이 Excel로 함께 여는 파일이라 utf-8-sig (CLAUDE.md 인코딩 규칙 2)
    with out_path.open("a", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for i, s in enumerate(samples, 1):
            scored = adapter.score_choices(s, build_views(s, cfg))
            probs = softmax(scored.logprobs)      # 4개 기호에 대해 재정규화한 확률
            top = sorted(probs, reverse=True)
            w.writerow({
                "id": s.id,
                "llm_answer": scored.predicted_letter,
                "llm_prob": f"{top[0]:.4f}",
                "llm_margin": f"{top[0] - top[1]:.4f}",
                **{f"p_{k}": f"{p:.4f}" for k, p in zip("abcd", probs)},
                "llm_model": MODEL_TAG,
            })
            fh.flush()
            if i % 50 == 0 or i == len(samples):
                rate = (time.time() - t0) / i
                print(f"{i}/{len(samples)}  {rate:.2f}s/건  남은 약 {math.ceil(rate * (len(samples) - i) / 60)}분",
                      flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
