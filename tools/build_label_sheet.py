"""Day 1 오전 — 사람이 검수할 라벨 시트(엑셀) 생성.

**Kaggle 기본규칙 4-b: validation·test 레코드의 수동 라벨링 금지.** 그래서 이 스크립트는
`dev` split에서만 뽑는다. 코드 레벨에서도 test가 한 줄이라도 섞이면 경고가 아니라
**에러로 멈춘다** — `_forbid_disallowed_split()`이 매 단계 호출된다.

샘플 구성:
  1. baseline 300개 — day1_baseline_9b.ipynb의 `--set subset.size=300
     --set subset.stratify_by=type`과 **완전히 같은 함수(src.data.subset)를 같은
     seed로 호출**해서 반드시 같은 300개가 나오게 한다. accuracy 계산에 쓰는 subset이라
     여기서 비율을 왜곡하면 편향된다.
  2. 희소 유형 보충분 — tools/classify_questions.py가 만든 auto_type 기준으로
     baseline 300개 안에 10개 미만인 유형을 최소 10개까지 채운다. 감사 전용이고
     accuracy 계산에는 안 쓴다(`is_supplement=True`로 구분).
  3. 강제 포함 (09/17 ZIP metadata 분석, 「06」 25~30절) — 무작위 300개에는 안 걸릴 수
     있지만 존재 여부 자체가 Accuracy 상한에 직결되는 의심 이미지들. train·dev만
     (test는 Kaggle 4-b상 제외):
       - 압축률 이상치 7장 (JPEG인데 ZIP 압축률 0.89 미만 — 거의 단색/손상 의심)
       - `data_meta/known_duplicates.csv`(tools/extract_duplicates.py 산출물)의
         중복 그룹에 속한 train·dev 이미지 전부
       - 픽셀 크기 최소/최대 3장씩 (train·dev 각각)
     `forced_reason` 컬럼으로 왜 강제 포함됐는지 표시한다. 전부 감사 전용이라
     accuracy 계산에는 안 쓴다(`is_supplement=True`).

실행:
    python tools/build_label_sheet.py
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")  # Windows 콘솔(cp949)에서 한글/이모지 깨짐 방지

from PIL import Image  # noqa: E402
from openpyxl import Workbook  # noqa: E402
from openpyxl.drawing.image import Image as XLImage  # noqa: E402
from openpyxl.styles import Alignment, Font, PatternFill, Protection  # noqa: E402
from openpyxl.utils import get_column_letter  # noqa: E402
from openpyxl.worksheet.datavalidation import DataValidation  # noqa: E402
from openpyxl.worksheet.protection import SheetProtection  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data import load_split, subset  # noqa: E402

CONFIG_NAME = "9b_base_1024"
SOURCE_SPLIT = "dev"  # Kaggle 4-b — 사람이 보는 건 dev(또는 train)만. test 절대 금지.
ALLOWED_SPLITS = {"train", "dev"}
BASELINE_N = 300
MIN_PER_TYPE = 10
SIZE_EXTREME_N = 3  # split별 최소/최대 각 3장

# 담당자는 여기 하나만 고치면 된다 — 인원/조 편성이 또 바뀔 수 있다.
# 각 튜플이 한 조(짝)다. 파일 번호는 이 리스트를 펼친 순서로 01, 02, 03...이 매겨진다.
PAIRS: list[tuple[str, str]] = [
    ("진영", "건순"),
    ("현호", "민우"),
]
TEAM_MEMBERS: list[str] = [name for pair in PAIRS for name in pair]
PARTNER_OF: dict[str, str] = {a: b for a, b in PAIRS} | {b: a for a, b in PAIRS}

# 09/17 ZIP metadata 분석(「06」 25~30절): JPEG인데 ZIP 압축률 0.89 미만 — 거의
# 단색이거나 손상 의심. 하드코딩이지만 이건 그 분석의 결론 자체라 재계산 대상이
# 아니다(재계산하려면 tools/extract_duplicates.py처럼 zip_inventory.csv의
# compression_ratio 컬럼을 다시 스캔하면 된다).
FORCED_COMPRESSION_OUTLIERS = {
    "train_2395", "train_1614", "dev_1915", "train_0238",
    "train_5079", "train_6208", "train_0704",
}

TYPES = ["PRICE", "TIMEDATE", "SPELLING", "NUMBER", "TABLE", "LOGO", "OBJECT",
         "RELATION", "COMPOSITE", "UNKNOWN"]
LABEL_ACCURACY_OPTIONS = ["맞음", "틀림", "애매"]
READABILITY_OPTIONS = ["선명", "흐림", "판독불가"]

TYPE_DEFINITIONS = {
    "PRICE": ("가격·금액·요금을 묻는 질문", ["(예) 가격표·영수증에서 금액을 찾는 질문", "이용 요금이 표시된 부분을 보고 답하세요"]),
    "TIMEDATE": ("영업시간·날짜·기간을 묻는 질문", ["영업 시작 시간은 몇 시인가요?", "(예) 행사·할인 기간의 마지막 날짜를 묻는 질문"]),
    "SPELLING": ("정확한 표기·철자·이름을 묻는 질문", ["간판에 적힌 정확한 상호명은?", "이 단어의 철자가 맞습니까?"]),
    "NUMBER": ("개수·번호·수치를 묻는 질문", ["사진 속 사람은 몇 명인가요?", "전화번호의 마지막 네 자리는?"]),
    "TABLE": ("표·목록에서 값을 찾는 질문", ["표에서 3번째 항목의 값은?", "목록 중 가장 위에 있는 항목은?"]),
    "LOGO": ("로고·브랜드·상표를 식별하는 질문", ["이 로고는 어느 브랜드인가요?", "상표에 그려진 도형은 무엇인가요?"]),
    "OBJECT": ("사물 식별·존재 여부를 묻는 질문", ["사진에 우산이 있나요?", "테이블 위에 놓인 물건은 무엇인가요?"]),
    "RELATION": ("위치관계·비교·순서를 묻는 질문", ["문 왼쪽에 있는 것은?", "가장 큰 상자는 어느 것인가요?"]),
    "COMPOSITE": ("두 개 이상의 판단이 결합된 질문", ["왼쪽 표에서 가장 비싼 항목의 가격은?"]),
    "UNKNOWN": ("규칙이 유형을 특정하지 못한 질문", ["규칙 보강이 필요한 질문들"]),
}


class ForbiddenSplitError(RuntimeError):
    """Kaggle 기본규칙 4-b 위반 — test(또는 미허용) split이 라벨 시트에 섞였다."""


def _forbid_disallowed_split(rows: list[dict[str, Any]], context: str) -> None:
    bad = [r for r in rows if r.get("split") not in ALLOWED_SPLITS]
    if bad:
        raise ForbiddenSplitError(
            f"[{context}] test(또는 미허용) split이 {len(bad)}행 섞여 있습니다 — "
            f"Kaggle 기본규칙 4-b(validation/test 수동 라벨링 금지) 위반입니다. "
            f"예: {[b['sample'].id for b in bad[:5]]}"
        )


def _load_question_types() -> dict[str, dict[str, str]]:
    path = REPO_ROOT / "data_meta" / "question_types.csv"
    if not path.exists():
        raise SystemExit(f"{path} 가 없습니다. 먼저 python tools/classify_questions.py 를 실행하세요.")
    out: dict[str, dict[str, str]] = {}
    with open(path, encoding="utf-8-sig") as f:  # utf-8-sig: classify_questions가 BOM으로 쓴다
        for row in csv.DictReader(f):
            out[row["row_id"]] = row
    return out


def _pick_supplement(remaining: list, question_types: dict, already: set[str], seed: int) -> list[dict]:
    """baseline 300 안에서 MIN_PER_TYPE 미만인 유형을 remaining(dev, 300 제외)에서 채운다."""
    from collections import Counter

    baseline_type_counts = Counter(
        question_types.get(rid, {}).get("auto_type", "UNKNOWN") for rid in already
    )
    by_type: dict[str, list] = {}
    for sample in remaining:
        row = question_types.get(sample.id)
        if row is None:
            continue
        by_type.setdefault(row["auto_type"], []).append(sample)

    def sort_key(sample):
        return hashlib.sha256(f"{seed}:supplement:{sample.id}".encode()).hexdigest()

    picked: list[dict] = []
    for type_name in TYPES:
        have = baseline_type_counts.get(type_name, 0)
        need = max(0, MIN_PER_TYPE - have)
        if need == 0:
            continue
        pool = sorted(by_type.get(type_name, []), key=sort_key)[:need]
        for sample in pool:
            row = question_types[sample.id]
            picked.append({"sample": sample, "auto_type": row["auto_type"],
                           "matched_rule": row["matched_rule"], "split": SOURCE_SPLIT,
                           "is_supplement": True})
    return picked


def _load_forced_duplicate_stems() -> set[str]:
    """train/dev에 속한 중복 그룹 이미지 stem 전부 (test는 4-b상 제외)."""
    path = REPO_ROOT / "data_meta" / "known_duplicates.csv"
    if not path.exists():
        print(f"경고: {path} 없음 — 중복 강제포함을 건너뜁니다. 먼저 "
              f"python tools/extract_duplicates.py 를 실행하세요.")
        return set()
    stems: set[str] = set()
    with open(path, encoding="utf-8-sig") as f:  # utf-8-sig: extract_duplicates가 BOM으로 쓴다
        for row in csv.DictReader(f):
            if row["split_a"] in ALLOWED_SPLITS:
                stems.add(row["stem_a"])
            if row["split_b"] in ALLOWED_SPLITS:
                stems.add(row["stem_b"])
    return stems


def _size_extremes(samples: list, n: int = SIZE_EXTREME_N) -> list:
    """픽셀 크기(긴 변) 기준 최소 n장 + 최대 n장. 헤더만 읽으므로 빠르다."""
    sized = []
    for s in samples:
        try:
            with Image.open(s.image_path) as im:
                sized.append((max(im.size), s))
        except Exception:  # noqa: BLE001 — 이미지 하나 못 열어도 나머지는 계속한다
            continue
    sized.sort(key=lambda t: t[0])
    return [s for _, s in sized[:n]] + [s for _, s in sized[-n:]]


def _build_forced_rows(train_samples: list, dev_samples: list, question_types: dict) -> list[dict]:
    """강제 포함 대상(압축률 이상치·중복·크기극단)을 실제 Sample로 해석한다."""
    by_id: dict[str, tuple[Any, str]] = {}
    for s in train_samples:
        by_id[s.id] = (s, "train")
    for s in dev_samples:
        by_id[s.id] = (s, "dev")

    forced: dict[str, dict] = {}

    def add(stem: str, reason: str) -> None:
        found = by_id.get(stem)
        if found is None:
            print(f"경고: 강제포함 대상 {stem} 을 train/dev에서 찾을 수 없습니다 — 건너뜁니다.")
            return
        sample, split = found
        if stem in forced:
            forced[stem]["forced_reason"] = forced[stem]["forced_reason"] + "+" + reason
            return
        qrow = question_types.get(stem, {})
        forced[stem] = {
            "sample": sample, "split": split,
            "auto_type": qrow.get("auto_type", "UNKNOWN"),
            "matched_rule": qrow.get("matched_rule", ""),
            "is_supplement": True, "forced_reason": reason,
        }

    for stem in FORCED_COMPRESSION_OUTLIERS:
        add(stem, "압축률이상")
    for stem in _load_forced_duplicate_stems():
        add(stem, "중복")
    for split_samples in (train_samples, dev_samples):
        for s in _size_extremes(split_samples):
            add(s.id, "크기극단")

    return list(forced.values())


def _autosize_and_lock(ws, editable_cols: set[int], n_data_rows: int, header_row: int = 1) -> None:
    ws.protection = SheetProtection(sheet=True, password=None)
    for col in range(1, ws.max_column + 1):
        for row in range(1, ws.max_row + 1):
            cell = ws.cell(row=row, column=col)
            cell.protection = Protection(locked=(col not in editable_cols or row <= header_row))


def _build_workbook(member: str, partner: str, rows: list[dict[str, Any]], out_path: Path) -> None:
    wb = Workbook()

    # ── 시트1: 작업지침 ──────────────────────────────────────────
    ws1 = wb.active
    ws1.title = "작업지침"
    instructions = [
        ("Day 1 라벨 검수 작업지침", True, None),
        ("", False, None),
        ("*** test 데이터는 이 시트에 없습니다. 있으면 안 됩니다. ***", True, "FFCCCC"),
        ("이 시트는 dev split에서만 뽑았습니다 (Kaggle 기본규칙 4-b: validation/test 수동 라벨링 금지).", False, None),
        ("", False, None),
        ("조 편성", True, None),
        (f"당신은 {member}입니다. 짝은 {partner}입니다 — 두 사람이 인접한 문항 블록을 나눠 맡습니다.", False, "DDEBF7"),
        ("판단이 갈리거나 애매하면 옆에서 짝과 바로 맞춰보세요. 비슷한 유형·순서를 보고 있어서 대화가 됩니다.", False, None),
        ("", False, None),
        ("무엇을 판단하나요?", True, None),
        ("각 행의 질문·이미지·제공된 정답을 보고 4가지를 판단합니다:", False, None),
        ("  1) 유형확정 — 자동분류(auto_type)가 맞는지. 맞으면 그대로 두고, 틀리면 드롭다운에서 고치세요.", False, None),
        ("  2) 라벨정확 — 제공된 정답이 이미지·질문과 맞는지 (맞음/틀림/애매).", False, None),
        ("  3) 판독가능 — 이미지에서 답을 읽어낼 수 있는지 (선명/흐림/판독불가).", False, None),
        ("  4) 메모 — 자유롭게 기록.", False, None),
        ("", False, None),
        ("애매하면 어떻게 하나요?", True, None),
        ("확신이 안 서면 '유형확정'은 UNKNOWN으로 두고, '라벨정확'은 '애매'를 고른 뒤", False, None),
        ("메모에 왜 애매한지 한 줄만 남겨주세요. 억지로 확정하지 않아도 됩니다.", False, None),
        ("", False, None),
        ("예상 소요시간", True, None),
        (f"1인당 문항 수: {len(rows)}개. 문항당 약 10~15초 예상 → 약 {len(rows)*12//60}분 내외.", False, None),
        ("", False, None),
        ("이미지 확인 방법", True, None),
        ("셀에 삽입된 썸네일은 훑어보기용입니다(작은 글씨는 안 보일 수 있음).", False, None),
        ("'원본 보기' 링크를 클릭하면 실제 크기 이미지가 열립니다 — 확신이 안 서면 꼭 열어보세요.", False, None),
    ]
    for i, (text, bold, fill) in enumerate(instructions, start=1):
        cell = ws1.cell(row=i, column=1, value=text)
        cell.font = Font(bold=bold, size=13 if i == 1 else 11)
        if fill:
            cell.fill = PatternFill("solid", fgColor=fill)
    ws1.column_dimensions["A"].width = 100
    for i in range(1, len(instructions) + 1):
        ws1.cell(row=i, column=1).alignment = Alignment(wrap_text=True, vertical="top")

    # ── 시트2: 라벨링 ────────────────────────────────────────────
    ws2 = wb.create_sheet("라벨링")
    locked_headers = ["row_id", "split", "image_id", "질문", "선택지1", "선택지2", "선택지3", "선택지4",
                       "제공된정답", "auto_type", "matched_rule", "is_supplement", "forced_reason",
                       "썸네일", "원본"]
    editable_headers = ["유형확정", "라벨정확", "판독가능", "메모"]
    all_headers = locked_headers + editable_headers
    for col, h in enumerate(all_headers, start=1):
        cell = ws2.cell(row=1, column=col, value=h)
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="DCE6F1" if h in locked_headers else "FFF2CC")
    ws2.freeze_panes = "A2"

    THUMB_COL = locked_headers.index("썸네일") + 1
    LINK_COL = locked_headers.index("원본") + 1
    TYPE_CONFIRM_COL = len(locked_headers) + 1

    for r_idx, row in enumerate(rows, start=2):
        sample = row["sample"]
        choices = list(sample.choices) + [""] * (4 - len(sample.choices))
        values = [
            sample.id, row["split"], sample.image_path.name, sample.question,
            choices[0], choices[1], choices[2], choices[3],
            sample.answer or "", row["auto_type"], row["matched_rule"],
            "예" if row.get("is_supplement") else "",
            row.get("forced_reason", ""),
        ]
        for c_idx, v in enumerate(values, start=1):
            ws2.cell(row=r_idx, column=c_idx, value=v)

        # 썸네일 삽입 (이미지가 실제로 있을 때만 — 없으면 조용히 건너뛴다)
        if sample.image_path.exists():
            try:
                img = XLImage(str(sample.image_path))
                long_side = max(img.width, img.height)
                scale = 400 / long_side if long_side else 1
                img.width = int(img.width * scale)
                img.height = int(img.height * scale)
                anchor = f"{get_column_letter(THUMB_COL)}{r_idx}"
                ws2.add_image(img, anchor)
                ws2.row_dimensions[r_idx].height = max(ws2.row_dimensions[r_idx].height or 15,
                                                        img.height * 0.75)
            except Exception as exc:  # noqa: BLE001 — 썸네일 하나 실패해도 시트 전체는 만든다
                ws2.cell(row=r_idx, column=THUMB_COL, value=f"(썸네일 실패: {exc})")
            link_cell = ws2.cell(row=r_idx, column=LINK_COL, value="원본 보기")
            link_cell.hyperlink = sample.image_path.resolve().as_uri()
            link_cell.font = Font(color="0563C1", underline="single")
        else:
            ws2.cell(row=r_idx, column=THUMB_COL, value="(이미지 없음)")

        # 유형확정 기본값 = auto_type (맞으면 그냥 넘어가게)
        ws2.cell(row=r_idx, column=TYPE_CONFIRM_COL, value=row["auto_type"])

    ws2.column_dimensions[get_column_letter(THUMB_COL)].width = 58
    for h, w in [("질문", 45), ("선택지1", 20), ("선택지2", 20), ("선택지3", 20), ("선택지4", 20),
                 ("matched_rule", 30), ("메모", 30)]:
        if h in all_headers:
            ws2.column_dimensions[get_column_letter(all_headers.index(h) + 1)].width = w

    last_row = len(rows) + 1
    dv_type = DataValidation(type="list", formula1=f'"{",".join(TYPES)}"', allow_blank=True)
    dv_acc = DataValidation(type="list", formula1=f'"{",".join(LABEL_ACCURACY_OPTIONS)}"', allow_blank=True)
    dv_read = DataValidation(type="list", formula1=f'"{",".join(READABILITY_OPTIONS)}"', allow_blank=True)
    for dv, col in [(dv_type, TYPE_CONFIRM_COL), (dv_acc, TYPE_CONFIRM_COL + 1), (dv_read, TYPE_CONFIRM_COL + 2)]:
        ws2.add_data_validation(dv)
        dv.add(f"{get_column_letter(col)}2:{get_column_letter(col)}{last_row}")

    editable_cols = {TYPE_CONFIRM_COL, TYPE_CONFIRM_COL + 1, TYPE_CONFIRM_COL + 2, TYPE_CONFIRM_COL + 3}
    _autosize_and_lock(ws2, editable_cols, len(rows))

    # ── 시트3: 유형정의 ──────────────────────────────────────────
    ws3 = wb.create_sheet("유형정의")
    ws3.append(["유형", "정의", "예시1", "예시2"])
    for cell in ws3[1]:
        cell.font = Font(bold=True)
    for type_name in TYPES:
        definition, examples = TYPE_DEFINITIONS[type_name]
        examples = (examples + ["", ""])[:2]
        ws3.append([type_name, definition, *examples])
    for col, w in zip("ABCD", [14, 45, 40, 40]):
        ws3.column_dimensions[col].width = w

    # ── 시트4: 집계 (수식) ────────────────────────────────────────
    ws4 = wb.create_sheet("집계")
    type_confirm_letter = get_column_letter(TYPE_CONFIRM_COL)
    label_acc_letter = get_column_letter(TYPE_CONFIRM_COL + 1)
    readable_letter = get_column_letter(TYPE_CONFIRM_COL + 2)
    auto_type_letter = get_column_letter(locked_headers.index("auto_type") + 1)

    ws4["A1"] = "진행률 (라벨정확 채운 행 / 전체)"
    ws4["B1"] = f'=COUNTA(라벨링!{label_acc_letter}2:{label_acc_letter}{last_row})/{len(rows)}'
    ws4["A2"] = "불일치율 (유형확정 ≠ auto_type)"
    ws4["B2"] = (f'=SUMPRODUCT((라벨링!{type_confirm_letter}2:{type_confirm_letter}{last_row}'
                 f'<>라벨링!{auto_type_letter}2:{auto_type_letter}{last_row})*1)/{len(rows)}')

    ws4["A4"] = "라벨정확 분포"
    for i, opt in enumerate(LABEL_ACCURACY_OPTIONS):
        ws4.cell(row=5 + i, column=1, value=opt)
        ws4.cell(row=5 + i, column=2,
                 value=f'=COUNTIF(라벨링!{label_acc_letter}2:{label_acc_letter}{last_row},"{opt}")')

    ws4["A9"] = "판독가능 분포"
    for i, opt in enumerate(READABILITY_OPTIONS):
        ws4.cell(row=10 + i, column=1, value=opt)
        ws4.cell(row=10 + i, column=2,
                 value=f'=COUNTIF(라벨링!{readable_letter}2:{readable_letter}{last_row},"{opt}")')

    for row in ws4.iter_rows(min_row=1, max_row=13, min_col=1, max_col=1):
        row[0].font = Font(bold=True)
    ws4.column_dimensions["A"].width = 35
    ws4.column_dimensions["B"].width = 15

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)
    print(f"저장됨: {out_path} ({len(rows)}행, 담당: {member})")


def main() -> int:
    cfg = load_config(CONFIG_NAME)
    samples, _schema = load_split(cfg, SOURCE_SPLIT)
    train_samples, _ = load_split(cfg, "train")

    baseline = subset(list(samples), BASELINE_N, cfg.seed, "type")
    baseline_ids = {s.id for s in baseline}

    label_sample_path = REPO_ROOT / "data_meta" / "label_sample_300.csv"
    label_sample_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_sample_path, "w", newline="", encoding="utf-8-sig") as f:  # BOM: 사람이 Excel로 여는 산출물
        w = csv.writer(f)
        w.writerow(["row_id", "split"])
        for s in baseline:
            w.writerow([s.id, SOURCE_SPLIT])
    print(f"baseline 300 저장됨: {label_sample_path}")

    question_types = _load_question_types()
    remaining = [s for s in samples if s.id not in baseline_ids]
    supplement = _pick_supplement(remaining, question_types, baseline_ids, cfg.seed)
    print(f"보충분: {len(supplement)}행")

    combined: list[dict[str, Any]] = []
    combined_by_id: dict[str, dict] = {}
    for s in baseline:
        row = question_types.get(s.id, {})
        entry = {
            "sample": s, "split": SOURCE_SPLIT,
            "auto_type": row.get("auto_type", "UNKNOWN"),
            "matched_rule": row.get("matched_rule", ""),
            "is_supplement": False, "forced_reason": "",
        }
        combined.append(entry)
        combined_by_id[s.id] = entry
    for row in supplement:
        row.setdefault("forced_reason", "")
        combined.append(row)
        combined_by_id[row["sample"].id] = row

    forced_rows = _build_forced_rows(train_samples, samples, question_types)
    new_forced_ids: set[str] = set()
    for row in forced_rows:
        existing = combined_by_id.get(row["sample"].id)
        if existing is not None:
            # 이미 baseline/보충분에 있으면 이유만 덧붙인다 — 행을 중복시키지 않는다
            existing["forced_reason"] = (existing["forced_reason"] + "+" + row["forced_reason"]
                                          if existing["forced_reason"] else row["forced_reason"])
        else:
            combined.append(row)
            combined_by_id[row["sample"].id] = row
            new_forced_ids.add(row["sample"].id)
    print(f"강제 포함: {len(forced_rows)}건 대상, 신규 {len(new_forced_ids)}행 추가"
          f"(나머지는 이미 baseline/보충분에 있어 forced_reason만 추가)")

    # Kaggle 4-b 방어: 여기서 test가 한 줄이라도 섞였으면 즉시 멈춘다
    _forbid_disallowed_split(combined, "라벨 시트 후보 전체")

    n_members = len(TEAM_MEMBERS)

    # 본문(forced_reason 없는 나머지)은 row_id 순으로 정렬해 인접 블록으로 나눈다 —
    # 같은 조가 이웃한 문항을 받아야 옆에서 "이거 어떻게 찍었어?"가 대화가 된다.
    # **forced_reason이 붙은 행은 전부**(신규 추가분뿐 아니라 원래 baseline/보충분에
    # 있다가 forced_reason만 덧붙여진 것까지) 따로 떼어 라운드로빈으로 고르게
    # 흩뿌린다 — 한 사람에게 몰리면 그 사람 시트만 이상한 이미지 투성이가 되어
    # "원래 이런 데이터인가" 오판하게 된다.
    ordinary_rows = sorted((r for r in combined if not r.get("forced_reason")),
                           key=lambda r: r["sample"].id)
    forced_flagged_rows = sorted((r for r in combined if r.get("forced_reason")),
                                  key=lambda r: r["sample"].id)

    chunks: list[list[dict]] = [[] for _ in range(n_members)]
    base_size, remainder = divmod(len(ordinary_rows), n_members)
    start = 0
    for i in range(n_members):
        size = base_size + (1 if i < remainder else 0)
        chunks[i] = ordinary_rows[start:start + size]
        start += size
    for i, row in enumerate(forced_flagged_rows):
        chunks[i % n_members].append(row)

    out_dir = REPO_ROOT / "data_meta" / "label_sheets"
    for i, (member, chunk) in enumerate(zip(TEAM_MEMBERS, chunks), start=1):
        _forbid_disallowed_split(chunk, f"{member} 파일")
        out_path = out_dir / f"label_sheet_{i:02d}_{member}.xlsx"
        _build_workbook(member, PARTNER_OF[member], chunk, out_path)

    forced_per_file = [sum(1 for r in chunk if r.get("forced_reason")) for chunk in chunks]
    print(f"\n총 {len(combined)}행을 {n_members}명에게 분배했습니다 (1인당 약 {len(combined)//n_members}행).")
    print(f"파일별 강제포함 행 수 (고르게 분산됐는지 확인): {dict(zip(TEAM_MEMBERS, forced_per_file))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
