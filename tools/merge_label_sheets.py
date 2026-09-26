"""Day 1 — 구글시트에서 내려받은 라벨 감사 CSV를 집계해 label noise·규칙 품질을 계산한다.

사용:
  1. 구글시트에서 파일 > 다운로드 > 쉼표로 구분된 값(.csv) 으로 내려받는다.
  2. data_meta/label_audit/label_audit_500_filled.csv 로 저장한다.
  3. python tools/merge_label_sheets.py [--csv 경로]

출력: data_meta/label_audit.csv (검증을 통과한 전체 행) + 콘솔 리포트.

**label noise 추정치는 무작위 표본(baseline300 + train_random)으로만 계산한다.** supplement·forced는
희소 유형·의심 이미지를 일부러 모은 편향 표본이라 noise 비율에 넣으면 Accuracy 상한이 왜곡된다.
dev(=검증셋)와 train의 noise는 따로 낸다 — dev noise가 검증 accuracy의 상한이고, train noise는 학습 품질이다.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")  # Windows 콘솔(cp949)에서 한글 깨짐 방지

DEFAULT_CSV = REPO_ROOT / "data_meta" / "label_audit" / "label_audit_500_filled.csv"
OUT_PATH = REPO_ROOT / "data_meta" / "label_audit.csv"

LABEL_OPTIONS = {"맞음", "틀림", "애매"}
READ_OPTIONS = {"선명", "흐림", "판독불가"}
NOISE_LABELS = {"틀림", "애매"}
RANDOM_SOURCES = {"baseline300", "train_random"}
FORBIDDEN_SPLITS = {"test"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """이항 비율의 Wilson 95% 신뢰구간. 표본이 작을 때 0%·100%를 단정하지 않기 위해 병기한다."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _fmt_rate(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:.2f}%  (95% CI {100 * lo:.1f}~{100 * hi:.1f}%)" if n else "표본 없음"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"{args.csv} 가 없습니다. 구글시트를 CSV로 내려받아 이 경로에 저장하세요.")

    with open(args.csv, encoding="utf-8-sig", newline="") as f:
        all_rows = list(csv.DictReader(f))
    if not all_rows:
        raise SystemExit("행이 없습니다.")
    print(f"{args.csv.name}: {len(all_rows)}행")

    # Kaggle 4-b 방어 — 시트가 어떻게 편집됐든 test가 있으면 멈춘다
    bad = [r for r in all_rows if r.get("split") in FORBIDDEN_SPLITS or r.get("row_id", "").startswith("test_")]
    if bad:
        raise SystemExit(f"test 행이 {len(bad)}개 있습니다 — Kaggle 4-b 위반. 예: {[b['row_id'] for b in bad[:5]]}")

    invalid = [r for r in all_rows
               if (r.get("라벨정확") and r["라벨정확"] not in LABEL_OPTIONS)
               or (r.get("판독가능") and r["판독가능"] not in READ_OPTIONS)]
    if invalid:
        print(f"\n경고: 허용값이 아닌 입력 {len(invalid)}행 (집계에서 제외) — 오타나 드롭다운 미적용 확인:")
        for r in invalid[:10]:
            print(f"  {r['row_id']} 라벨정확={r.get('라벨정확')!r} 판독가능={r.get('판독가능')!r} 담당={r.get('담당')}")
    valid_ids = {id(r) for r in invalid}
    rows = [r for r in all_rows if id(r) not in valid_ids]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8-sig") as f:  # BOM: 사람이 Excel로 여는 산출물
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"저장됨: {OUT_PATH}")

    labeled = [r for r in rows if r.get("라벨정확")]
    print(f"\n진행률: {len(labeled)}/{len(all_rows)} ({100 * len(labeled) / len(all_rows):.1f}%)")
    by_member_total = Counter(r.get("담당", "?") for r in all_rows)
    by_member_done = Counter(r.get("담당", "?") for r in labeled)
    for m, total in by_member_total.most_common():
        print(f"  {m}: {by_member_done[m]}/{total}")

    # ── label noise: 무작위 표본만, split별로 ─────────────────────────
    random_rows = [r for r in labeled if r.get("source") in RANDOM_SOURCES]
    print("\n=== Label noise 추정치 (무작위 표본: baseline300 + train_random) ===")
    for split in ("dev", "train"):
        part = [r for r in random_rows if r["split"] == split]
        k = sum(1 for r in part if r["라벨정확"] in NOISE_LABELS)
        print(f"[{split}] 틀림+애매 {_fmt_rate(k, len(part))}")
        if part:
            n_wrong = sum(1 for r in part if r["라벨정확"] == "틀림")
            print(f"       (그중 확실히 틀림 {n_wrong}, 애매 {k - n_wrong})")
    dev_part = [r for r in random_rows if r["split"] == "dev"]
    if dev_part:
        k = sum(1 for r in dev_part if r["라벨정확"] in NOISE_LABELS)
        ratio = k / len(dev_part)
        print(f"\n*** dev noise {100 * ratio:.2f}% → 검증 Accuracy 상한은 약 {100 * (1 - ratio):.2f}%. "
              f"「06」 9절과 목표 Accuracy 재조정의 근거로 쓰세요. ***")
        print("    (\"애매\"는 사람마다 기준이 달라 하한~상한 구간이 넓다. CI와 함께 읽는다.)")

    biased = [r for r in labeled if r.get("source") not in RANDOM_SOURCES]
    if biased:
        k = sum(1 for r in biased if r["라벨정확"] in NOISE_LABELS)
        print(f"\n[참고·편향 표본: supplement/forced] 틀림+애매 {k}/{len(biased)} — noise 추정에 쓰지 않는다")
        by_reason: dict[str, Counter] = defaultdict(Counter)
        for r in biased:
            for reason in (r.get("forced_reason") or r["source"]).split("+"):
                by_reason[reason][r["라벨정확"]] += 1
        for reason, counts in sorted(by_reason.items()):
            print(f"  {reason:10} {dict(counts)}")

    # ── 자동 분류 정확도 ─────────────────────────────────────────────
    confirmed = [r for r in labeled if r.get("유형확정") and r.get("auto_type")]
    if confirmed:
        n_match = sum(1 for r in confirmed if r["유형확정"] == r["auto_type"])
        print(f"\n=== 자동 분류 정확도 ===\n유형확정 == auto_type: {_fmt_rate(n_match, len(confirmed))}")
        mismatch_by_rule: Counter = Counter()
        for r in confirmed:
            if r["유형확정"] != r["auto_type"] and r.get("matched_rule"):
                for rule in str(r["matched_rule"]).split(";;"):
                    mismatch_by_rule[rule] += 1
        if mismatch_by_rule:
            print("\n=== 가장 많이 틀린 규칙 Top 5 (tools/classify_questions.py에서 고칠 자리) ===")
            for rule, n in mismatch_by_rule.most_common(5):
                print(f"  {n:4}회  {rule}")

    # ── 유형별 판독가능 분포 ─────────────────────────────────────────
    readable_by_type: dict[str, Counter] = defaultdict(Counter)
    for r in labeled:
        if r.get("판독가능"):
            readable_by_type[r.get("유형확정") or r.get("auto_type") or "UNKNOWN"][r["판독가능"]] += 1
    if readable_by_type:
        print("\n=== 유형별 판독가능 분포 ===")
        for type_name, counts in sorted(readable_by_type.items()):
            total_t = sum(counts.values())
            print(f"  {type_name:10} " + ", ".join(f"{k}={v}({100 * v / total_t:.0f}%)" for k, v in counts.items()))

    incomplete = [r for r in all_rows if not r.get("라벨정확")]
    if incomplete:
        print(f"\n=== 미완료 행 {len(incomplete)}개 ===")
        for m, n in Counter(r.get("담당", "?") for r in incomplete).most_common():
            print(f"  {m}: {n}개 남음")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
