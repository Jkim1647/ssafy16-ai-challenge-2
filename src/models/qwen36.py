"""Qwen3.6 계열 어댑터 (27B dense / 35B-A3B MoE).

3.6은 3.5와 아키텍처·파라미터 구성·context·라이선스가 동일하다. 달라지는 것은
가중치와 벤치마크뿐이라 로딩·채점 경로를 그대로 상속하고, 3.6에서만 다시 확인해야
하는 항목을 여기서 명시한다.
"""

from __future__ import annotations

import logging

from ..config import ConfigError
from .qwen35 import Qwen35Adapter

log = logging.getLogger(__name__)


class Qwen36Adapter(Qwen35Adapter):
    family = "qwen36"

    def load(self, for_training: bool = False) -> None:
        super().load(for_training=for_training)

        # 3.5에서 통과한 smoke test 결과를 3.6에 그대로 옮기지 않는다.
        # expert weight는 fused 3D 텐서이고, vLLM #38520(미해결)이 우리 스택에서도
        # 재현되는지는 실제 환경에서만 알 수 있다.
        if for_training and str(self.cfg.get("train.target", "default")) == "with_experts":
            missing = self._missing_targets(self.mc.get("peft_targets", {}).get("moe_experts", []))
            if missing:
                raise ConfigError(
                    "expert target module을 모델에서 찾지 못했다: "
                    f"{missing}. vLLM #38520과 같은 이름 불일치일 수 있다 — "
                    "train.target=default(non-expert Linear만)로 먼저 통과시켜라."
                )

    def _missing_targets(self, names: list[str]) -> list[str]:
        if self.model is None or not names:
            return []
        present = {n for n, _ in self.model.named_modules()}
        return [n for n in names if not any(n in p for p in present)]
