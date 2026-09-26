"""Qwen3.5 계열 어댑터 (9B / 122B-A10B / 397B-A17B).

Qwen3.5 이후 본체는 네이티브 멀티모달이라 별도 VL 브랜치를 쓰지 않는다.
Qwen3.6은 아키텍처가 동일해 qwen36.py가 이 클래스를 상속한다.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import torch
from PIL import Image

from ..config import ConfigError
from ..data import Sample
from .base import ChoiceScores, ModelAdapter, softmax

log = logging.getLogger(__name__)

IGNORE_INDEX = -100
_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class Qwen35Adapter(ModelAdapter):
    family = "qwen35"

    # --- 로드 ---

    def _quant_config(self):
        quant = str(self.mc.get("load", {}).get("quantization", "none")).lower()
        if quant in {"none", "", "null"}:
            return None
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ConfigError("양자화에는 transformers + bitsandbytes가 필요하다") from exc

        compute = _DTYPES[str(self.mc.get("load", {}).get("dtype", "bfloat16"))]
        if quant == "nf4":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,   # 고정 설정
                bnb_4bit_compute_dtype=compute,
            )
        if quant == "int8":
            return BitsAndBytesConfig(load_in_8bit=True)
        raise ConfigError(
            f"지원하지 않는 quantization: {quant}. bitsandbytes는 NF4·INT8만 지원하고 "
            "네이티브 FP8은 지원하지 않는다."
        )

    def _model_cls(self):
        import transformers
        for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"):
            cls = getattr(transformers, name, None)
            if cls is not None:
                return cls
        raise ConfigError("transformers에서 멀티모달 모델 클래스를 찾지 못했다")

    def load(self, for_training: bool = False) -> None:
        from transformers import AutoProcessor

        load_cfg = self.mc.get("load", {})
        hf_id = self.mc.get("hf_id")
        if not hf_id:
            raise ConfigError(f"{self.mc.get('name')}에 hf_id가 없다")

        vis = self.mc.get("vision", {})
        self.processor = AutoProcessor.from_pretrained(
            hf_id,
            trust_remote_code=bool(load_cfg.get("trust_remote_code", True)),
            min_pixels=int(vis.get("min_pixels", 313_600)),
            max_pixels=int(vis.get("max_pixels", 7_840_000)),
            use_fast=True,          # 고정. fast/slow processor는 출력이 다를 수 있다
        )
        self.tokenizer = getattr(self.processor, "tokenizer", None) or self.processor

        kwargs: dict[str, Any] = {
            "dtype": _DTYPES[str(load_cfg.get("dtype", "bfloat16"))],
            "trust_remote_code": bool(load_cfg.get("trust_remote_code", True)),
            "device_map": load_cfg.get("device_map", "auto"),
        }
        if (qc := self._quant_config()) is not None:
            kwargs["quantization_config"] = qc
        if attn := load_cfg.get("attn_implementation"):
            kwargs["attn_implementation"] = attn

        cls = self._model_cls()
        try:
            self.model = cls.from_pretrained(hf_id, **kwargs)
        except (ImportError, ValueError) as exc:
            # flash-attn 미설치 환경에서 흔하다. 수치가 미세하게 달라지므로 로그로 남긴다.
            if "attn_implementation" not in kwargs:
                raise
            log.warning("attn_implementation=%s 실패(%s) — sdpa로 폴백한다", kwargs["attn_implementation"], exc)
            kwargs["attn_implementation"] = "sdpa"
            self.model = cls.from_pretrained(hf_id, **kwargs)

        if for_training and self.cfg.get("train.grad_checkpointing", True):
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False
        else:
            self.model.eval()

    # --- 입력 만들기 ---

    def _encode(self, sample: Sample, views: Sequence[Image.Image],
                answer_text: str | None = None, ocr_text: str | None = None):
        messages = self.build_messages(sample, views, ocr_text)
        # Qwen3.5 템플릿은 기본이 thinking 모드라 generation prompt가 "<think>\n"로 끝난다.
        # 그대로 두면 선택지 글자 확률을 '생각 블록 첫 토큰' 위치에서 읽게 된다(09/21 확인).
        # enable_thinking=False면 "<think>\n\n</think>\n\n" 뒤, 즉 답변 위치에서 채점한다.
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=bool(self.cfg.get("prompt.enable_thinking", False)),
        )
        if answer_text is not None:
            text = text + answer_text
        inputs = self.processor(
            text=[text], images=list(views), return_tensors="pt", padding=True
        )
        return inputs.to(self.model.device), text

    def _letter_token_ids(self, letters: Sequence[str]) -> list[int] | None:
        """각 기호가 단일 토큰이면 forward 1회로 모든 선택지를 채점할 수 있다."""
        ids: list[int] = []
        for letter in letters:
            enc = self.tokenizer.encode(letter, add_special_tokens=False)
            if len(enc) != 1:
                return None
            ids.append(enc[0])
        return ids

    # --- 추론 ---

    @torch.no_grad()
    def score_choices(self, sample: Sample, views: Sequence[Image.Image],
                      ocr_text: str | None = None) -> ChoiceScores:
        if not sample.choices:
            raise ConfigError(f"sample {sample.id}에 선택지가 없다")
        letters = self.choice_letters(sample)

        fast_ids = self._letter_token_ids(letters)
        if fast_ids is not None:
            inputs, _ = self._encode(sample, views, ocr_text=ocr_text)
            out = self.model(**inputs)
            logits = out.logits[0, -1, :].float()
            logprobs = torch.log_softmax(logits, dim=-1)
            scores = [logprobs[i].item() for i in fast_ids]
            n_visual = self.count_visual_tokens(inputs)
        else:
            # 선택지 본문 전체를 채점해야 하는 경우 (기호가 단일 토큰이 아닐 때)
            scores, n_visual = [], None
            for letter, choice in zip(letters, sample.choices):
                answer = f"{letter}. {choice}"
                inputs, _ = self._encode(sample, views, answer_text=answer, ocr_text=ocr_text)
                n_answer = len(self.tokenizer.encode(answer, add_special_tokens=False))
                out = self.model(**inputs)
                lp = torch.log_softmax(out.logits[0, :-1, :].float(), dim=-1)
                tgt = inputs["input_ids"][0, 1:]
                picked = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[-n_answer:]
                scores.append(picked.mean().item())    # 길이 보정을 위해 평균을 쓴다
                n_visual = n_visual or self.count_visual_tokens(inputs)

        probs = softmax(scores)
        best = max(range(len(scores)), key=lambda i: scores[i])
        return ChoiceScores(
            sample_id=sample.id,
            logprobs=scores,
            predicted_index=best,
            predicted_letter=letters[best],
            confidence=probs[best] if probs else 0.0,
            visual_tokens=n_visual,
        )

    @torch.no_grad()
    def generate(self, sample: Sample, views: Sequence[Image.Image],
                 ocr_text: str | None = None) -> str:
        inputs, _ = self._encode(sample, views, ocr_text=ocr_text)
        out = self.model.generate(
            **inputs,
            max_new_tokens=int(self.cfg.get("decode.max_new_tokens", 4)),
            do_sample=False,                  # temperature=0, greedy 고정
        )
        trimmed = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(trimmed, skip_special_tokens=True).strip()

    # --- 학습 ---

    def training_example(self, sample: Sample, views: Sequence[Image.Image]) -> dict[str, Any]:
        """loss masking: 질문·선택지 토큰은 IGNORE_INDEX, 정답 토큰에만 loss를 건다."""
        if sample.answer is None:
            raise ConfigError(f"sample {sample.id}에 정답이 없다 — 학습에 쓸 수 없다")

        letters = self.choice_letters(sample)
        answer = sample.answer.strip()
        if answer not in letters:
            # 정답이 기호가 아니라 본문으로 들어온 경우 기호로 되돌린다
            try:
                answer = letters[[c.strip() for c in sample.choices].index(answer)]
            except ValueError as exc:
                raise ConfigError(
                    f"sample {sample.id}의 정답 {sample.answer!r}이 선택지·기호 어디에도 없다"
                ) from exc

        prompt_inputs, prompt_text = self._encode(sample, views)
        prompt_len = prompt_inputs["input_ids"].shape[1]

        full_inputs, _ = self._encode(sample, views, answer_text=answer)
        input_ids = full_inputs["input_ids"][0]

        labels = input_ids.clone()
        if self.cfg.get("train.loss_masking", True):
            labels[:prompt_len] = IGNORE_INDEX
        # IMAGE_CONCAT_KEYS는 배치 축이 없다 — v[0]을 걸면 crop(여러 이미지) 중
        # 첫 장만 남고 나머지가 잘려나간다(base.py 주석 참조).
        example = {
            k: (v if k in self.IMAGE_CONCAT_KEYS else v[0])
            for k, v in full_inputs.items()
        }
        example["labels"] = labels
        example["_prompt_len"] = prompt_len
        example["_visual_tokens"] = self.count_visual_tokens(full_inputs)
        del prompt_text
        return example
