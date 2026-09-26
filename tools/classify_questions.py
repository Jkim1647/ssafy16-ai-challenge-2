"""Day 1 오전 — 질문 유형 자동 분류 (규칙 기반, GPU/모델 없음).

Kaggle 기본규칙 4-b는 validation·test의 **수동** 라벨링을 금지하지만, 프로그램에 의한
자동 분류는 막지 않는다. 그래서 이 스크립트는 train·dev·test 전부 처리한다 —
사람이 직접 눈으로 보는 건 tools/build_label_sheet.py 몫이고, 거기서는 test를
코드로 강제 차단한다.

실행:
    python tools/classify_questions.py

출력: data_meta/question_types.csv (row_id, split, auto_type, matched_rule, match_count)
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")  # Windows 콘솔(cp949)에서 한글/이모지 깨짐 방지

from src.config import load_config  # noqa: E402
from src.data import load_split  # noqa: E402

# ── 규칙: 전부 여기 한 곳에 모은다 — 검수 후 고칠 자리다 ──────────────
# 「02」 taxonomy. configs/base.yaml의 evaluate.types(OCR/Scene 포함, PRICE/SPELLING/
# TIMEDATE 없음)와는 다른 체계다 — 이 taxonomy가 최신이면 evaluate.types도 나중에
# 맞춰야 한다(별도 작업).
RULES: dict[str, list[str]] = {
    "PRICE": [r"가격", r"얼마", r"요금", r"금액", r"원가", r"비용"],
    "TIMEDATE": [r"영업\s*시간", r"몇\s*시", r"날짜", r"기간", r"요일", r"언제", r"영업일"],
    "SPELLING": [r"정확한\s*표기", r"철자", r"이름.{0,3}(뭐|무엇)", r"뭐라고\s*(적혀|쓰여|쓰인)", r"글자"],
    "NUMBER": [r"몇\s*개", r"번호", r"전화번호", r"수량", r"개수", r"몇\s*명", r"몇\s*번"],
    "TABLE": [r"표에서", r"목록에서", r"표\s*안", r"항목"],
    "LOGO": [r"로고", r"브랜드", r"상표"],
    "OBJECT": [r"무엇", r"물건", r"사물", r"있(나요|습니까|어)"],
    "RELATION": [r"왼쪽", r"오른쪽", r"위에", r"아래", r"옆에", r"순서", r"비교", r"보다"],
}

CONFIG_NAME = "9b_base_1024"  # data.columns 스키마만 빌려 쓴다 — 모델/GPU는 안 건드린다
UNKNOWN_WARN_RATIO = 0.30
SPLIT_DRIFT_WARN_PP = 5.0  # percentage point


def classify(question: str) -> tuple[str, list[str], int]:
    """질문 하나를 (auto_type, matched_rule 목록, match_count)로 분류한다."""
    matched: list[str] = []
    for type_name, patterns in RULES.items():
        for pat in patterns:
            if re.search(pat, question):
                matched.append(f"{type_name}:{pat}")
                break  # 이 유형은 최초 1개 패턴만 기록 (유형별 1회)
    matched_types = sorted({m.split(":", 1)[0] for m in matched})
    count = len(matched_types)
    if count == 0:
        auto_type = "UNKNOWN"
    elif count == 1:
        auto_type = matched_types[0]
    else:
        auto_type = "COMPOSITE"
    return auto_type, matched, count


def main() -> int:
    cfg = load_config(CONFIG_NAME)
    out_path = REPO_ROOT / "data_meta" / "question_types.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    per_split_counts: dict[str, Counter] = defaultdict(Counter)
    unknown_samples: list[tuple[str, str, str]] = []

    for split in ("train", "dev", "test"):
        try:
            samples, _schema = load_split(cfg, split)
        except Exception as exc:  # noqa: BLE001 — 한 split이 없어도 나머지는 계속한다
            print(f"[{split}] 로드 실패, 건너뜀: {exc}")
            continue
        for s in samples:
            auto_type, matched, count = classify(s.question)
            rows.append({
                "row_id": s.id,
                "split": split,
                "auto_type": auto_type,
                "matched_rule": ";;".join(matched),
                "match_count": count,
            })
            per_split_counts[split][auto_type] += 1
            if auto_type == "UNKNOWN" and len(unknown_samples) < 20:
                unknown_samples.append((split, s.id, s.question))

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:  # BOM: 사람이 Excel로 여는 산출물
        writer = csv.DictWriter(f, fieldnames=["row_id", "split", "auto_type", "matched_rule", "match_count"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"저장됨: {out_path} ({len(rows)}행)")

    if not rows:
        print("행이 하나도 없습니다 — CSV 경로/스키마를 확인하세요.")
        return 1

    total = len(rows)
    n_unknown = sum(1 for r in rows if r["auto_type"] == "UNKNOWN")
    unknown_ratio = n_unknown / total
    print(f"\nUNKNOWN 비율: {unknown_ratio*100:.1f}% ({n_unknown}/{total})")
    if unknown_ratio > UNKNOWN_WARN_RATIO:
        print(f"*** 경고: UNKNOWN이 {UNKNOWN_WARN_RATIO*100:.0f}%를 넘습니다 — 규칙이 실패했습니다. ***")
        print("*** 아래 샘플을 보고 RULES를 보강하세요. ***")
    if unknown_samples:
        print("\nUNKNOWN 샘플 (최대 20개):")
        for split, rid, q in unknown_samples:
            print(f"  [{split}] {rid}: {q}")

    print("\n유형별 비율 (split별, %):")
    all_types = sorted({t for c in per_split_counts.values() for t in c})
    splits_present = [s for s in ("train", "dev", "test") if s in per_split_counts]
    header = "type".ljust(12) + "".join(s.rjust(10) for s in splits_present)
    print(header)
    drift_warnings: list[str] = []
    for t in all_types:
        ratios: dict[str, float] = {}
        line = t.ljust(12)
        for split in splits_present:
            n = per_split_counts[split][t]
            total_s = sum(per_split_counts[split].values())
            pct = 100 * n / total_s if total_s else 0.0
            ratios[split] = pct
            line += f"{pct:9.1f}%"
        print(line)
        if len(ratios) > 1 and (max(ratios.values()) - min(ratios.values())) > SPLIT_DRIFT_WARN_PP:
            drift_warnings.append(t)

    if drift_warnings:
        print(f"\n*** 경고: 다음 유형은 split 간 비율 차이가 {SPLIT_DRIFT_WARN_PP:.0f}%p를 넘습니다 "
              f"— 배포 편향 가능성, 그 자체가 중요한 정보입니다: {drift_warnings} ***")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
