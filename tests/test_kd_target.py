"""KD 목표분포(kd_target)와 손실 수식의 순수 파이썬 회귀 테스트.

검증 범위: 선택지 순환(--aug-shift) 매핑, 온도·재정규화, 글자 누락 처리, KL 방향, alpha=1 동일성.
torch 텐서 연산·역전파·메모리는 이 테스트로 검증되지 않는다(GPU 환경에서 스모크 필요).
실행: python -m unittest discover -s tests
"""
from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import colab_lora_train as m  # noqa: E402

L = m.LETTERS


def log_softmax(xs):
    mx = max(xs)
    z = math.log(sum(math.exp(x - mx) for x in xs)) + mx
    return [x - z for x in xs]


def kd_loss_py(student_letter_logits, target, ce, alpha, temp):
    """colab_lora_train.py 학습 루프의 KD 항을 그대로 옮긴 순수 파이썬 판."""
    s_logp = log_softmax([x / temp for x in student_letter_logits])
    kl = sum(p * (math.log(max(p, 1e-12)) - q) for p, q in zip(target, s_logp))
    return alpha * ce + (1 - alpha) * temp ** 2 * kl, kl


class KDTargetTest(unittest.TestCase):
    def test_shift_mapping_matches_gold_position(self):
        rng = random.Random(0)
        for _ in range(2000):
            gold = rng.choice(L)
            t_orig = [-0.05 if x == gold else -4.0 - rng.random() for x in L]
            k = rng.randrange(4)
            gold_shown = (L.index(gold) - k) % 4  # 학습 루프와 같은 규칙
            tgt = m.kd_target(t_orig, k, 2.0)
            self.assertAlmostEqual(sum(tgt), 1.0, places=12)
            self.assertEqual(tgt.index(max(tgt)), gold_shown)

    def test_missing_letter_returns_none(self):
        self.assertIsNone(m.kd_target([-0.1, m.KD_MISSING, -3.0, -4.0], 0, 2.0))

    def test_temperature_one_is_renormalized_teacher(self):
        t = [-0.2, -1.8, -3.0, -5.0]
        z = sum(math.exp(x) for x in t)
        for a, b in zip(m.kd_target(t, 0, 1.0), t):
            self.assertAlmostEqual(a, math.exp(b) / z, places=12)

    def test_kl_zero_when_student_matches_teacher_and_positive_otherwise(self):
        t = [-0.3, -1.5, -2.5, -4.0]
        tgt = m.kd_target(t, 0, 2.0)
        _, kl_same = kd_loss_py(t, tgt, ce=0.0, alpha=0.5, temp=2.0)  # 학생 logit = teacher logprob(+상수 무관)
        self.assertAlmostEqual(kl_same, 0.0, places=12)
        _, kl_diff = kd_loss_py([-3.0, -0.2, -2.0, -4.0], tgt, ce=0.0, alpha=0.5, temp=2.0)
        self.assertGreater(kl_diff, 0.0)

    def test_alpha_one_equals_ce(self):
        tgt = m.kd_target([-0.3, -1.5, -2.5, -4.0], 1, 2.0)
        loss, _ = kd_loss_py([-3.0, -0.2, -2.0, -4.0], tgt, ce=1.234, alpha=1.0, temp=2.0)
        self.assertAlmostEqual(loss, 1.234, places=12)


if __name__ == "__main__":
    unittest.main()
