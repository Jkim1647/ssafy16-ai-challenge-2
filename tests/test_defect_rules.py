# -*- coding: utf-8 -*-
"""구조적 결함 판정 규칙 회귀 테스트.

2026-09-23에 실제로 낸 버그를 고정한다: 비교용 정규화가 문장부호·기호를 전부
제거하는 바람에 숫자·수식이 뭉개져 멀쩡한 문항이 '보기중복'으로 잡혔다.
  4.0 == 40 / 1.5km == 15km / log(1-y) == -log(1-y)
이 상태로 제외를 돌리면 정상 평가 문항이 빠진다.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))

from build_defect_queue import broken_opts, canon, dup_pairs  # noqa: E402


def same(a, b):
    return canon(a) == canon(b)


class TestCanonMergesOnlyNoise:
    """공백·대소문자·꼬리 장식만 같은 것으로 본다."""

    def test_whitespace(self):
        assert same('잘해주는 치과', '잘 해 주는 치과')

    def test_case(self):
        assert same('Sunny 영어학원', 'SUNNY 영어학원')

    def test_trailing_decoration(self):
        assert same('가스 잠그기', '가스 잠그기!')
        assert same('오늘도 좋은 하루 & ~', '오늘도 좋은 하루 &')


class TestCanonKeepsMeaning:
    """의미를 바꾸는 내부 문장부호는 남긴다 (2026-09-23 버그)."""

    def test_decimal_point(self):
        assert not same('4.0', '40')
        assert not same('1.5km', '15km')

    def test_minus_sign(self):
        assert not same('log(1−ŷ)', '−log(1−ŷ)')

    def test_internal_hyphen(self):
        assert not same('B-11-6', 'B116')


class TestDupPairs:
    def test_detects_duplicate(self):
        r = {'a': '가스 잠그기', 'b': '치갑 폰', 'c': '다른 것', 'd': '가스 잠그기!'}
        assert dup_pairs(r) == 'a=d'

    def test_no_false_positive_on_numbers(self):
        r = {'a': '4,000', 'b': '0.4', 'c': '4.0', 'd': '40'}
        assert dup_pairs(r) == ''

    def test_empty_options_not_paired(self):
        """빈 문자열끼리는 '중복'으로 세지 않는다 (깨진_보기 쪽에서 잡는다)."""
        r = {'a': '', 'b': '', 'c': '가', 'd': '나'}
        assert dup_pairs(r) == ''


class TestBrokenOpts:
    def test_detects_mojibake(self):
        assert broken_opts({'a': '美食??', 'b': '정상', 'c': '정상2', 'd': '정상3'}) == 'a'

    def test_detects_all_question_marks(self):
        assert 'c' in broken_opts({'a': '태국식당', 'b': '방콕식당', 'c': '??????', 'd': '타이'})

    def test_keeps_normal_text_with_one_question_mark(self):
        """물음표가 섞였다고 다 깨진 건 아니다 — 비공백의 절반 이상일 때만."""
        assert broken_opts({'a': '정말로 그런가요?', 'b': '나', 'c': '다', 'd': '라'}) == ''
