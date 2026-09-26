"""모델 어댑터 레지스트리."""

from __future__ import annotations

from ..config import Config, ConfigError
from .base import ChoiceScores, ModelAdapter
from .qwen35 import Qwen35Adapter
from .qwen36 import Qwen36Adapter

ADAPTERS: dict[str, type[ModelAdapter]] = {
    "qwen35": Qwen35Adapter,
    "qwen36": Qwen36Adapter,
}


def build_adapter(cfg: Config) -> ModelAdapter:
    key = str(cfg.get("model_config.adapter", "")).lower()
    if key not in ADAPTERS:
        raise ConfigError(f"알 수 없는 adapter: {key!r}. 가능한 값: {sorted(ADAPTERS)}")
    return ADAPTERS[key](cfg)


__all__ = ["ADAPTERS", "ChoiceScores", "ModelAdapter", "build_adapter"]
