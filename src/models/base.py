"""모델 어댑터 공통 계층.

파이프라인은 구체 모델을 모른다. 이 인터페이스만 안다:
  load()                  가중치·processor 로드
  build_prompt(sample)    질문 + 선택지 → 채팅 메시지
  score_choices(...)      선택지별 log-likelihood (주 경로)
  generate(...)           자유 생성 (비교용)
  training_example(...)   loss masking이 적용된 input_ids/labels

왜 score_choices가 주 경로인가: 답이 A/B/C/D 한 글자인 객관식에서 자유 생성은
형식 위반(빈 답, 문장으로 답하기)이라는 별개의 실패 모드를 만든다. 선택지 조건부
확률로 argmax를 고르면 그 실패 모드가 구조적으로 사라지고, logit을 그대로 저장해
재학습 없이 앙상블 가중치를 재탐색할 수 있다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

from PIL import Image

from ..config import Config, ConfigError
from ..data import Sample

CHOICE_PROMPT = (
    "이미지를 보고 질문에 답하시오.\n"
    "질문: {question}\n"
    "{choices}\n"
    "정답 기호 하나만 출력하시오."
)

# 석웅 35B GGUF 1,000건 실행(commands.txt)과 같은 문구. prompt.template=seokwoong_ko 로 선택.
SEOKWOONG_PROMPT = (
    "이미지를 보고 객관식 문제를 푸세요. 설명이나 추론은 출력하지 말고 정답인 소문자 a, b, c, d 중 "
    "하나만 출력하세요. 질문: {question} 선택지: {choices} 정답:"
)


@dataclass
class ChoiceScores:
    sample_id: str
    logprobs: list[float]          # 선택지별 평균 log-likelihood
    predicted_index: int
    predicted_letter: str
    confidence: float              # softmax 후 최댓값
    visual_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.sample_id,
            "logprobs": [round(v, 6) for v in self.logprobs],
            "pred_index": self.predicted_index,
            "pred": self.predicted_letter,
            "confidence": round(self.confidence, 6),
            "visual_tokens": self.visual_tokens,
        }


def softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    top = max(values)
    exps = [pow(2.718281828459045, v - top) for v in values]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


class ModelAdapter(ABC):
    """구체 모델 하나를 감싼다."""

    family: str = "base"

    # processor가 배치 축 없이 "여러 이미지의 patch를 이미 이어붙인" 형태로 주는 키.
    # text 계열 키(input_ids 등)는 배치 크기 1의 앞차원을 갖지만 이 키들은 그렇지 않다 —
    # crop을 쓰면(view가 2개 이상) 여기 `v[0]`을 걸거나 나중에 `unsqueeze(0)`을 걸면
    # 첫 이미지 것만 남고 나머지 crop의 patch·grid 정보가 통째로 잘려나간다
    # (09/17 실측: image_grid_thw가 (5,3)이어야 할 게 (3,)으로 잘리는 것을 확인).
    IMAGE_CONCAT_KEYS: frozenset[str] = frozenset({
        "pixel_values", "image_grid_thw",
        "pixel_values_videos", "video_grid_thw", "second_per_grid_ts",
    })

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.mc = cfg.model
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._loaded = False

    # --- 로드 ---

    @abstractmethod
    def load(self, for_training: bool = False) -> None:
        ...

    def ensure_loaded(self, for_training: bool = False) -> None:
        if not self._loaded:
            self.load(for_training=for_training)
            self._loaded = True

    # --- 프롬프트 ---

    def choice_letters(self, sample: Sample) -> list[str]:
        prefixes = self.cfg.get("data.columns.choice_prefixes") or ["A", "B", "C", "D"]
        return [str(p) for p in prefixes[: len(sample.choices)]]

    def build_prompt(self, sample: Sample, ocr_text: str | None = None) -> str:
        letters = self.choice_letters(sample)
        lines = [f"{letter}. {text}" for letter, text in zip(letters, sample.choices)]
        if self.cfg.get("prompt.template") == "seokwoong_ko":
            prompt = SEOKWOONG_PROMPT.format(question=sample.question, choices=" ".join(lines))
        else:
            prompt = CHOICE_PROMPT.format(question=sample.question, choices="\n".join(lines))
        if ocr_text and self.cfg.get("prompt.include_ocr_text"):
            prompt = f"이미지에서 추출한 텍스트:\n{ocr_text}\n\n{prompt}"
        if system := self.cfg.get("prompt.system"):
            prompt = f"{system}\n\n{prompt}"
        return prompt

    def build_messages(self, sample: Sample, views: Sequence[Image.Image],
                       ocr_text: str | None = None) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "image", "image": v} for v in views]
        content.append({"type": "text", "text": self.build_prompt(sample, ocr_text)})
        return [{"role": "user", "content": content}]

    # --- 추론 ---

    @abstractmethod
    def score_choices(self, sample: Sample, views: Sequence[Image.Image],
                      ocr_text: str | None = None) -> ChoiceScores:
        ...

    @abstractmethod
    def generate(self, sample: Sample, views: Sequence[Image.Image],
                 ocr_text: str | None = None) -> str:
        ...

    # --- 학습 ---

    @abstractmethod
    def training_example(self, sample: Sample, views: Sequence[Image.Image]) -> dict[str, Any]:
        ...

    def peft_target_modules(self) -> list[str]:
        targets = self.mc.get("peft_targets", {})
        base = list(targets.get("default", []))
        if str(self.cfg.get("train.target", "default")) == "with_experts":
            base += list(targets.get("moe_experts", []))
        if not base:
            raise ConfigError(f"{self.mc.get('name')}에 peft_targets.default가 비어 있다")
        return base

    def count_visual_tokens(self, inputs: Any) -> int | None:
        """processor가 실제로 만든 visual token 수. 계산 규칙 검증의 핵심 항목이다.

        09/17: crop sweep에서 config 계산값과 어긋나는 원인으로 이 함수의
        `sum-then-floor`(전체 patch 수를 먼저 합하고 마지막에 한 번만 merge^2로
        나누는 방식)를 의심했다. image_grid_thw를 직접 찍어 개별 이미지별
        floor 후 합산과 비교했더니 완전히 동일했다(각 이미지의 patch grid는
        processor의 smart-resize로 이미 merge_size의 배수라 floor 순서가
        결과에 영향을 주지 않는다) — **이 함수는 버그가 아니다.** 실제 원인은
        `Config.visual_tokens()`가 원본 픽셀 크기를 그대로 썼던 것이었다(수정함).
        """
        grid = getattr(inputs, "image_grid_thw", None)
        if grid is None and isinstance(inputs, dict):
            grid = inputs.get("image_grid_thw")
        if grid is None:
            return None
        try:
            merge = int(self.mc.get("vision", {}).get("spatial_merge_size", 2))
            return int(grid.prod(dim=-1).sum().item()) // (merge * merge)
        except (AttributeError, TypeError, ValueError):
            return None
