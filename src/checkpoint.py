"""checkpoint 저장·선택·재로드.

원칙 두 가지:
1. best checkpoint는 validation loss가 아니라 validation Accuracy로 고른다.
   loss가 내려가도 accuracy가 내려가는 구간이 실제로 있다.
2. 저장한 adapter는 "저장 → 프로세스 재시작 → 재로드 → 추론 → merge"까지
   통과해야 쓸 수 있다. MoE fused expert weight에서 이 경로가 깨진 선례가 있다.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class CheckpointRecord:
    step: int
    accuracy: float
    loss: float | None
    path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CheckpointManager:
    def __init__(self, run_dir: Path, keep_last: int = 2, metric: str = "accuracy"):
        self.dir = run_dir / "checkpoints"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self.metric = metric
        self.history: list[CheckpointRecord] = []
        self.best: CheckpointRecord | None = None

    # --- 저장 ---

    def save(self, model, step: int, accuracy: float, loss: float | None = None,
             processor=None) -> CheckpointRecord:
        target = self.dir / f"step-{step:06d}"
        target.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(target)          # PEFT면 adapter만 저장된다
        if processor is not None:
            processor.save_pretrained(target)

        record = CheckpointRecord(step=step, accuracy=accuracy, loss=loss, path=str(target))
        self.history.append(record)

        if self.best is None or accuracy > self.best.accuracy:
            self.best = record
            self._write_best()

        self._prune()
        self._write_index()
        return record

    def _write_best(self) -> None:
        if self.best is None:
            return
        (self.dir / "best.json").write_text(
            json.dumps(self.best.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _write_index(self) -> None:
        (self.dir / "index.json").write_text(
            json.dumps(
                {
                    "metric": self.metric,
                    "best": self.best.to_dict() if self.best else None,
                    "history": [r.to_dict() for r in self.history],
                },
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )

    def _prune(self) -> None:
        """best는 절대 지우지 않고, 나머지는 최근 keep_last개만 남긴다."""
        protected = {self.best.path} if self.best else set()
        removable = [r for r in self.history if r.path not in protected]
        for record in removable[: max(0, len(removable) - self.keep_last)]:
            path = Path(record.path)
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)

    # --- 선택 ---

    def best_path(self) -> Path | None:
        if self.best:
            return Path(self.best.path)
        return read_best(self.dir.parent)


def read_best(run_dir: Path) -> Path | None:
    best_file = run_dir / "checkpoints" / "best.json"
    if not best_file.exists():
        return None
    try:
        return Path(json.loads(best_file.read_text(encoding="utf-8"))["path"])
    except (json.JSONDecodeError, KeyError):
        return None


def load_adapter(model, adapter_path: Path):
    """저장한 PEFT adapter를 base 모델 위에 다시 올린다."""
    from peft import PeftModel

    if not adapter_path.exists():
        raise FileNotFoundError(f"adapter 경로가 없다: {adapter_path}")
    return PeftModel.from_pretrained(model, str(adapter_path))


def merge_adapter(model, output_dir: Path):
    """adapter를 base에 병합해 저장한다. 추론 배포용."""
    merged = model.merge_and_unload()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir)
    return merged


def smoke_test_roundtrip(adapter, sample, views, tracker) -> dict[str, Any]:
    """저장 → 재로드 → 추론이 실제로 도는지 확인한다.

    프로세스 재시작까지는 이 함수가 대신할 수 없다. run.py의 `smoke` 명령이
    별도 프로세스로 재로드 단계를 한 번 더 돌린다.
    """
    result: dict[str, Any] = {"stage": None, "ok": False}
    try:
        result["stage"] = "score_before_save"
        before = adapter.score_choices(sample, views)
        result["pred_before"] = before.predicted_letter
        result["visual_tokens"] = before.visual_tokens

        result["stage"] = "save"
        path = tracker.artifact("checkpoints") / "smoke"
        adapter.model.save_pretrained(path)

        result["stage"] = "reload"
        from peft import PeftModel

        current = adapter.model
        # adapter.model은 이미 학습용으로 LoRA가 올라간 PeftModel이다. 그 위에 그대로
        # PeftModel.from_pretrained를 또 호출하면 이중 래핑(base_model.model.base_model...)이
        # 되어 새로 붙는 adapter가 항상 항등(identity) 상태라 재로드 검증이 무의미해진다.
        # 저장한 adapter를 내려 순수 base로 되돌린 뒤 그 위에 다시 올려야 실제 재로드를 검증한다.
        base = current.base_model.unload() if isinstance(current, PeftModel) else current
        reloaded = load_adapter(base, path)
        result["stage"] = "score_after_reload"
        adapter.model, original = reloaded, adapter.model
        after = adapter.score_choices(sample, views)
        adapter.model = original
        result["pred_after"] = after.predicted_letter
        result["match"] = before.predicted_letter == after.predicted_letter
        result["ok"] = True
    except Exception as exc:  # noqa: BLE001 — smoke test는 실패 사유 자체가 결과다
        result["error"] = f"{type(exc).__name__}: {exc}"
    tracker.event("peft_smoke_test", **result)
    return result
