"""CSV 인코딩 정책 회귀 테스트 (CLAUDE.md 「인코딩 규칙」 강제).

대회 원본 CSV에 BOM이 없어 한글 Windows의 Excel이 cp949로 읽고 한글이 깨진다.
우리가 생성하는 파일만 고친다:
  1. 읽기는 항상 utf-8-sig (BOM 있든 없든 안전)
  2. 사람이 여는 산출물(CSV·한글 txt)은 utf-8-sig
  3. 기계가 읽는 산출물(Kaggle 제출)은 utf-8, BOM 금지

실행: python -m unittest discover -s tests
"""
from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 사람이 Excel/메모장으로 여는 산출물을 쓰는 파일 — 여기의 모든 텍스트 쓰기/읽기는
# encoding이 지정돼 있으면 utf-8-sig여야 한다.
HUMAN_FACING_FILES = {
    "tools/classify_questions.py",
    "tools/merge_label_sheets.py",
    "tools/extract_duplicates.py",
    "tools/build_label_sheet.py",
    "tools/build_audit_package.py",
    "tools/zip_stats.py",
    "tools/draft_dev_labels.py",
    "tools/dev_relabel_app.py",
    "tools/compare_calibration.py",
    "tools/merge_claude_draft.py",
    "tools/compare_full_dev.py",
    "tools/relabel_merge_compare.py",
}
# Kaggle 제출을 쓰는 파일 — utf-8이어야 하고 절대 BOM(utf-8-sig)을 붙이면 안 된다.
SUBMISSION_FILE = "src/infer.py"


def _encoding_kwarg(node: ast.Call) -> str | None | object:
    """encoding= 리터럴 값을 돌려준다. 없으면 MISSING, 리터럴이 아니면 UNKNOWN."""
    for kw in node.keywords:
        if kw.arg == "encoding":
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                return kw.value.value
            return UNKNOWN
    return MISSING


MISSING = object()
UNKNOWN = object()


def _mode_of_open(node: ast.Call) -> str:
    """open(...) / X.open(...) 의 mode 문자열. 기본은 'r'."""
    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
        return str(node.args[1].value)
    for kw in node.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            return str(kw.value.value)
    return "r"


def _to_csv_writes_file(node: ast.Call) -> bool:
    """to_csv 가 실제로 파일을 쓰는 호출인지.

    pandas `to_csv(path_or_buf=None, ...)` 는 경로를 주지 않으면 **문자열을 반환**하고
    파일을 만들지 않는다. 이때 encoding 인자는 아무 일도 하지 않으므로 정책 대상이 아니다.
    경로를 바깥에서 인코딩 지정해 쓰는 패턴이 실제로 쓰인다:
        OUT.write_text(df.to_csv(index=False), encoding="utf-8-sig")
    이런 호출까지 위반으로 잡으면 오탐이다. 경로가 주어진 경우에만 파일 쓰기로 본다.
    """
    if node.args:  # to_csv(path, ...)
        first = node.args[0]
        return not (isinstance(first, ast.Constant) and first.value is None)
    for kw in node.keywords:
        if kw.arg == "path_or_buf":
            return not (isinstance(kw.value, ast.Constant) and kw.value.value is None)
    return False


def _iter_text_io_calls(py_path: Path):
    """(lineno, is_write, encoding) 를 yield.

    대상: open() / Path.open() / DataFrame.to_csv(경로 있을 때) / Path.write_text / Path.read_text.
    write_text·read_text 도 플랫폼 기본 인코딩으로 떨어지므로 같은 정책을 받는다.
    """
    tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_open = (isinstance(func, ast.Name) and func.id == "open") or (
            isinstance(func, ast.Attribute) and func.attr == "open"
        )
        is_to_csv = isinstance(func, ast.Attribute) and func.attr == "to_csv"
        is_path_text = isinstance(func, ast.Attribute) and func.attr in ("write_text", "read_text")
        if is_path_text:
            yield node.lineno, func.attr == "write_text", _encoding_kwarg(node)
        elif is_open:
            mode = _mode_of_open(node)
            if "b" in mode:  # 바이너리는 encoding 대상이 아니다
                continue
            yield node.lineno, ("w" in mode or "a" in mode or "x" in mode), _encoding_kwarg(node)
        elif is_to_csv and _to_csv_writes_file(node):
            yield node.lineno, True, _encoding_kwarg(node)


def _py_files():
    for base in ("src", "tools"):
        for p in (REPO_ROOT / base).rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            yield p


class TestStaticEncodingPolicy(unittest.TestCase):
    def test_no_text_write_without_encoding(self):
        """CSV/텍스트 쓰기에 encoding 인자가 빠지면 플랫폼 기본(cp949)으로 깨진다."""
        violations = []
        for p in _py_files():
            rel = p.relative_to(REPO_ROOT).as_posix()
            for lineno, is_write, enc in _iter_text_io_calls(p):
                if is_write and enc is MISSING:
                    violations.append(f"{rel}:{lineno} 쓰기에 encoding= 가 없다")
        self.assertEqual(violations, [], "\n" + "\n".join(violations))

    def test_human_facing_files_use_bom(self):
        """사람이 여는 산출물 파일의 텍스트 IO는 encoding이 있으면 utf-8-sig여야 한다."""
        violations = []
        for p in _py_files():
            rel = p.relative_to(REPO_ROOT).as_posix()
            if rel not in HUMAN_FACING_FILES:
                continue
            for lineno, _is_write, enc in _iter_text_io_calls(p):
                if isinstance(enc, str) and enc != "utf-8-sig":
                    violations.append(f"{rel}:{lineno} encoding={enc!r} (utf-8-sig 여야 함)")
        self.assertEqual(violations, [], "\n" + "\n".join(violations))

    def test_submission_writer_is_bomless(self):
        """제출 파일 writer는 utf-8이어야 하고 utf-8-sig(BOM)이면 안 된다."""
        p = REPO_ROOT / SUBMISSION_FILE
        found_to_csv = False
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "to_csv":
                found_to_csv = True
                enc = _encoding_kwarg(node)
                self.assertEqual(enc, "utf-8",
                                 f"{SUBMISSION_FILE}:{node.lineno} 제출은 utf-8이어야 한다 (BOM 금지)")
        self.assertTrue(found_to_csv, "src/infer.py에서 제출 to_csv를 찾지 못했다")


class TestCheckerItself(unittest.TestCase):
    """정적 검사기의 오탐/누락을 고정한다 (2026-09-23).

    `df.to_csv(index=False)` 는 경로가 없어 문자열만 돌려준다. 파일을 만들지 않으므로
    encoding 인자가 의미가 없는데, 이전 검사기는 이것도 위반으로 잡아
    `OUT.write_text(df.to_csv(index=False), encoding="utf-8-sig")`(정상 코드)를 실패시켰다.
    """

    def _scan(self, src: str):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.py"
            p.write_text(src, encoding="utf-8")
            return list(_iter_text_io_calls(p))

    def test_to_csv_without_path_is_not_a_file_write(self):
        self.assertEqual(self._scan("df.to_csv(index=False)\n"), [])
        self.assertEqual(self._scan("df.to_csv(None, index=False)\n"), [])
        self.assertEqual(self._scan("df.to_csv(path_or_buf=None)\n"), [])

    def test_wrapped_write_text_is_accepted(self):
        """실제로 실패했던 패턴 — 바깥 write_text가 encoding을 지정하므로 위반이 아니다."""
        got = self._scan('OUT.write_text(df.to_csv(index=False), encoding="utf-8-sig")\n')
        self.assertEqual([(w, e) for _l, w, e in got], [(True, "utf-8-sig")])

    def test_to_csv_with_path_still_requires_encoding(self):
        """완화가 과해지지 않았는지 — 경로를 주면 여전히 잡아야 한다."""
        got = self._scan('df.to_csv("out.csv", index=False)\n')
        self.assertEqual([(w, e) for _l, w, e in got], [(True, MISSING)])
        got = self._scan('df.to_csv(path_or_buf=p)\n')
        self.assertEqual([(w, e) for _l, w, e in got], [(True, MISSING)])

    def test_plain_open_write_still_requires_encoding(self):
        got = self._scan('open("x.txt", "w")\n')
        self.assertEqual([(w, e) for _l, w, e in got], [(True, MISSING)])

    def test_path_text_helpers_are_covered(self):
        """write_text/read_text 도 플랫폼 기본 인코딩으로 떨어지므로 검사 대상이다."""
        self.assertEqual([(w, e) for _l, w, e in self._scan('p.write_text(s)\n')],
                         [(True, MISSING)])
        self.assertEqual([(w, e) for _l, w, e in self._scan('p.read_text()\n')],
                         [(False, MISSING)])


class TestReaderHandlesBom(unittest.TestCase):
    def test_bom_corrupts_first_field_without_sig(self):
        """실패 모드 문서화: stdlib csv로 BOM'd CSV를 utf-8로 읽으면 첫 필드명이 오염된다.

        src/data.py:69와 tools/의 csv.DictReader 경로가 이 함정에 빠진다(pandas는 이
        버전에서 BOM을 자동으로 벗기지만 stdlib csv는 남긴다 — 그래서 utf-8-sig가 필요).
        """
        import csv
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.csv"
            p.write_text("id,answer\ntest_0001,A\n", encoding="utf-8-sig")  # BOM 포함
            with open(p, encoding="utf-8") as f:
                cols_utf8 = next(csv.reader(f))
            with open(p, encoding="utf-8-sig") as f:
                cols_sig = next(csv.reader(f))
            self.assertTrue(cols_utf8[0].startswith("﻿"))
            self.assertEqual(cols_sig[0], "id")

    def test_load_split_reads_bom_csv(self):
        """load_split이 BOM 붙은 대회 CSV를 컬럼 오염 없이 읽는다."""
        from src.config import load_config
        from src.data import load_split

        with tempfile.TemporaryDirectory() as d:
            csv_path = Path(d) / "dev.csv"
            # BOM 포함 + 09/21 확정 스키마(configs/base.yaml data.columns)
            csv_path.write_text(
                "id,path,question,a,b,c,d,answer\n"
                "dev_0001,dev/dev_0001.jpg,얼마인가?,100,200,300,400,a\n",
                encoding="utf-8-sig",
            )
            cfg = load_config("9b_base_1024", [f"data.dev_csv={csv_path.as_posix()}"])
            samples, schema = load_split(cfg, "dev")
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].id, "dev_0001")
            self.assertFalse(schema.id.startswith("﻿"),
                             "BOM이 컬럼명에 새어 들어갔다 — utf-8-sig 읽기가 안 됐다")


class TestSubmissionNoBom(unittest.TestCase):
    def test_write_submission_has_no_bom(self):
        from src.infer import write_submission

        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d)
            (run_dir / "predictions.jsonl").write_text(
                json.dumps({"id": "test_0001", "pred": "A"}) + "\n"
                + json.dumps({"id": "test_0002", "pred": "B"}) + "\n",
                encoding="utf-8",
            )
            out = run_dir / "submission.csv"
            write_submission(run_dir, out)
            self.assertNotEqual(out.read_bytes()[:3], b"\xef\xbb\xbf",
                                "제출 CSV에 BOM이 있으면 Kaggle 채점기가 첫 ID를 깨뜨린다")


if __name__ == "__main__":
    unittest.main()
