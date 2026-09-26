"""Create a number-sorted, image-checked copy for final human review.

The generated source CSV is never edited. Run this once before human edits;
rerunning it after review would replace the review copy.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_DIR = HERE / "generated/dev_split_markers_v1/mine/regenerated_questions"
EXTRA_FIELDS = ("photo_no", "semantic_review_issue", "human_review_status", "human_review_note")


# Only corrections supported by the original image. At 0411, avoid relying on
# the unclear two-character text and ask about the legible vertical sign.
CORRECTIONS: dict[int, dict[str, str]] = {
    109: {
        "b": "메타그로스",
        "evidence": "카드 상단에 성호의 메타그로스 ex라고 적혀 있고 그림도 메타그로스임",
        "review_note": "원본 이미지 재확인: 루기아를 메타그로스로 수정",
    },
    135: {
        "question": "건물의 큰 파란 벽면에 적힌 전문 가전의 이름은 무엇인가요?",
        "c": "딤채",
        "evidence": "파란 벽면에 딤채 전문 가전이라고 크게 적혀 있음",
        "review_note": "원본 이미지 재확인: 침대 오독 수정",
    },
    411: {
        "question": "노란 기둥 간판의 큰 세로 문구가 안내하는 시설은 무엇인가요?",
        "a": "편의점",
        "b": "카페",
        "c": "식당",
        "d": "노래연습장",
        "evidence": "노란 간판의 큰 세로 글씨에 노래연습장이 선명하게 적혀 있음",
        "review_note": "상단 두 글자의 판독에 의존하지 않고 선명한 세로 문구로 재작성",
    },
    634: {
        "question": "스시웨이 가게 간판 아래 그림은 어떤 동물 캐릭터인가요?",
        "a": "생선",
        "b": "고양이",
        "c": "토끼",
        "d": "강아지",
        "evidence": "스시웨이 글씨 아래 웃는 생선 캐릭터가 보임",
        "review_note": "초밥 그림으로 오인한 문항을 눈에 보이는 캐릭터 질문으로 수정",
    },
    919: {
        "question": "쪼그려 앉은 사람 앞에 있는 주황색 강아지는 몇 마리인가요?",
        "evidence": "사람 앞에 몸집이 다른 주황색 강아지 두 마리가 보임",
        "review_note": "고양이를 강아지로 수정",
    },
    1026: {
        "a": "밝은 크림색",
        "evidence": "투명 케이스 안 조각 케이크의 맨 위 층은 밝은 크림색임",
        "review_note": "분홍색 가운데 층과 구분하여 맨 위 층 색상 수정",
    },
    1401: {
        "question": "빨간 노점 앞면에서 가장 크게 보이는 음식 사진은 어떤 요리인가요?",
        "c": "대장 요리",
        "evidence": "노점 앞면의 큰 음식 사진 옆에 卤大肠과 大肠이 적혀 있음",
        "review_note": "옆면 닭발 광고와 앞면 대장 요리 사진을 구분",
    },
    1621: {
        "question": "버스 정류장 오른쪽 광고 하단에서 안내하는 진료 분야는 무엇인가요?",
        "review_note": "문항 본문에 정답인 피부과가 노출되지 않도록 수정",
    },
    1632: {
        "question": "건물 오른쪽의 큰 흰색 간판에는 어떤 공인중개사 상호가 적혀 있나요?",
        "evidence": "건물 오른쪽 큰 흰색 간판에 행운 공인중개사가 적혀 있음",
        "review_note": "노란 간판이라는 잘못된 지칭을 흰색 큰 간판으로 수정",
    },
    1694: {
        "question": "붉은 모집 안내판에 모집한다고 적힌 부문은 무엇인가요?",
        "c": "학생부",
        "evidence": "붉은 안내판에 학생부모집이라고 적혀 있음",
        "review_note": "학생복 오독을 학생부로 수정",
    },
    1766: {
        "question": "기내 화면에서 헬싱키에서 런던으로 가는 구간의 도착 게이트는 몇 번인가요?",
        "answer": "c",
        "evidence": "HEL-LHR 구간 도착 게이트는 45번 탑승구로 표시됨",
        "review_note": "헬싱키 경유 구간의 44번 게이트와 최종 구간 45번을 구분",
    },
}


def photo_number(value: str) -> int:
    name = Path(value).name
    matches = re.findall(r"\d+", Path(name).stem)
    if not matches:
        raise ValueError(f"사진 번호가 없는 파일명: {value}")
    return int(matches[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_DIR / "codex_regenerated_all.csv")
    parser.add_argument("--flags", type=Path, default=DEFAULT_DIR / "semantic_review_flags.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_DIR / "human_review_sorted.csv")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing review copy")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        parser.error(f"검수 파일이 이미 있습니다: {args.output} (--force 없이 덮어쓰지 않음)")
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or ())
        rows = list(reader)
    with args.flags.open(encoding="utf-8-sig", newline="") as handle:
        flags = {row["source_id"]: row for row in csv.DictReader(handle)}
    if not rows or not fields or "source_id" not in fields:
        parser.error("원본 CSV에 source_id 열 또는 행이 없습니다")
    numbers: set[int] = set()
    for row in rows:
        number = photo_number(row["source_id"])
        if number in numbers:
            parser.error(f"중복 사진 번호: {number}")
        numbers.add(number)
        row["photo_no"] = str(number)
        correction = CORRECTIONS.get(number, {})
        row.update(correction)
        flag = flags.get(row["source_id"])
        row["semantic_review_issue"] = flag["issue_type"] if flag else ""
        row["human_review_status"] = "needs_review" if flag else "pending"
        row["human_review_note"] = flag["review_note"] if flag else ""
        if row["review_status"] == "approved":
            choices = [row[key].strip() for key in "abcd"]
            if row["answer"] not in "abcd" or any(not choice for choice in choices) or len(set(choices)) != 4:
                parser.error(f"승인 행의 선지/정답 오류: {row['source_id']}")
    if set(CORRECTIONS) - numbers:
        parser.error(f"수정 대상이 입력에 없음: {sorted(set(CORRECTIONS) - numbers)}")
    rows.sort(key=lambda row: (int(row["photo_no"]), row["source_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + list(EXTRA_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    print(f"created {args.output}: {len(rows)} rows, {len(CORRECTIONS)} corrections, {len(flags)} flagged")


if __name__ == "__main__":
    main()
