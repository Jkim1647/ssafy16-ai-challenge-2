"""실험 설정 로딩과 해석.

configs/base.yaml  ← configs/models/<model>.yaml + configs/resources/<resource>.yaml
                   ← configs/experiments/<name>.yaml  ← CLI --set override

해석된 설정(resolved config)은 run 디렉터리에 resolved_config.yaml로 그대로 저장한다.
나중에 "이 숫자가 어느 설정에서 나왔는가"를 되짚을 수 있어야 하기 때문이다.
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"


class ConfigError(RuntimeError):
    pass


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"설정 파일이 없다: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"최상위가 매핑이 아니다: {path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """override가 이긴다. 양쪽 다 dict인 키만 재귀적으로 합친다."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce(text: str) -> Any:
    """--set 값의 YAML 스칼라 해석. 'null' → None, '1e-5' → float, 'a,b' → list."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """--set train.lr=5e-5 --set input.crops=4 형태를 적용한다."""
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"--set 은 key=value 형식이어야 한다: {item!r}")
        dotted, raw = item.split("=", 1)
        node = out
        parts = dotted.strip().split(".")
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[parts[-1]] = _coerce(raw.strip())
    return out


@dataclass
class Config:
    """해석이 끝난 설정. dict 접근과 점 표기 헬퍼를 함께 제공한다."""

    raw: dict[str, Any]
    experiment_path: Path | None = None
    overrides: list[str] = field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel or value is None:
            raise ConfigError(f"필수 설정이 비어 있다: {dotted}")
        return value

    # --- 자주 쓰는 파생값 ---

    @property
    def name(self) -> str:
        return self.get("name", "unnamed")

    @property
    def task(self) -> str:
        return self.get("task", "infer")

    @property
    def seed(self) -> int:
        return int(self.get("seed", 1))

    @property
    def model(self) -> dict[str, Any]:
        return self.get("model_config", {})

    @property
    def resource(self) -> dict[str, Any]:
        return self.get("resource_config", {})

    def path(self, dotted: str, default: str | None = None) -> Path:
        """설정의 상대 경로를 저장소 루트 기준 절대 경로로 바꾼다."""
        value = self.get(dotted, default)
        if value is None:
            raise ConfigError(f"경로 설정이 비어 있다: {dotted}")
        p = Path(str(value)).expanduser()
        return p if p.is_absolute() else (REPO_ROOT / p)

    def visual_tokens(self, width: int, height: int) -> int:
        """이 모델의 patch/merge 규칙으로 visual token 수를 계산한다.

        09/17 실측(long_side·crops=4/9·crop_long_side=768/512 sweep 총 5가지 조합)으로
        원인을 확인했다: AutoProcessor는 원본 픽셀을 그대로 세지 않고 Qwen2-VL 계열의
        `smart_resize`를 그대로 쓴다 — 각 변을 patch*merge(이 모델은 16×2=32px)의
        배수로 반올림하고, 그 결과 넓이가 [min_pixels, max_pixels] 밖이면 비율을 유지한
        채 ceil/floor로 다시 그 범위 안으로 맞춘다. crop_long_side를 512처럼 작게
        주면 crop 넓이가 min_pixels(313,600px)보다 작아져 **업스케일되면서 종횡비까지
        살짝 바뀐다**(512×384 → 512×672) — 그래서 처음엔 단순 반올림만으로는
        crop_long_side=512 조합에서 또 어긋났다. min/max_pixels 클램프까지 넣고 나서야
        5개 조합 전부(long_side 1024/1536/2048, crop_long_side 768/512) processor
        실측과 정확히 일치했다. `count_visual_tokens()`의 `sum-then-floor`는 이 문제와
        무관하다 — image_grid_thw 직접 비교로 이미 정확함을 확인했다(그쪽 함수 주석 참조).
        """
        vis = self.model.get("vision", {})
        patch = int(vis.get("patch_size", 16))
        merge = int(vis.get("spatial_merge_size", 2))
        factor = patch * merge
        min_pixels = int(vis.get("min_pixels", factor * factor))
        max_pixels = int(vis.get("max_pixels", 7_840_000))

        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = math.floor(height / beta / factor) * factor
            w_bar = math.floor(width / beta / factor) * factor
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor

        per_token = int(vis.get("pixels_per_token", 1024))
        return (h_bar * w_bar) // per_token

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.raw)

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.raw, fh, allow_unicode=True, sort_keys=False)


def _resolve_ref(kind: str, ref: str) -> dict[str, Any]:
    """'qwen36_35b_a3b' 또는 'configs/models/qwen36_35b_a3b.yaml' 둘 다 받는다."""
    candidate = Path(ref)
    if candidate.suffix in {".yaml", ".yml"}:
        path = candidate if candidate.is_absolute() else REPO_ROOT / candidate
    else:
        path = CONFIGS_DIR / kind / f"{ref}.yaml"
    return _read_yaml(path)


def load_config(experiment: str, overrides: list[str] | None = None) -> Config:
    """experiment 이름 또는 경로를 받아 완전히 해석된 Config를 돌려준다."""
    overrides = list(overrides or [])

    exp_path = Path(experiment)
    if exp_path.suffix not in {".yaml", ".yml"}:
        exp_path = CONFIGS_DIR / "experiments" / f"{experiment}.yaml"
    elif not exp_path.is_absolute():
        exp_path = REPO_ROOT / exp_path

    exp = _read_yaml(exp_path)
    merged = deep_merge(_read_yaml(CONFIGS_DIR / "base.yaml"), exp)

    # override를 먼저 적용한다. --set resource=local_5060ti 처럼 참조 자체를 바꾸는
    # 경우가 있어서, 참조를 해석하기 전에 최종 model/resource 이름을 확정해야 한다.
    merged = apply_overrides(merged, overrides)

    model_ref = merged.get("model")
    if not model_ref:
        raise ConfigError(f"{exp_path.name}에 model 키가 없다")
    merged["model_config"] = _resolve_ref("models", str(model_ref))

    resource_ref = merged.get("resource")
    merged["resource_config"] = _resolve_ref("resources", str(resource_ref)) if resource_ref else {}

    # 참조를 펼친 뒤 한 번 더 적용한다 — model_config.* / resource_config.* 를
    # 직접 덮어쓰는 override가 파일 값에 지워지지 않게 하기 위해서다.
    merged = apply_overrides(merged, overrides)
    merged = _apply_env(merged)

    cfg = Config(raw=merged, experiment_path=exp_path, overrides=overrides)
    _validate(cfg)
    return cfg


def _apply_env(cfg: dict[str, Any]) -> dict[str, Any]:
    """.env / 셸 환경변수로 경로만 덮어쓴다. 비밀값은 설정에 넣지 않는다."""
    out = copy.deepcopy(cfg)
    if data_dir := os.environ.get("AI2_DATA_DIR"):
        out.setdefault("data", {})["root"] = data_dir
    if runs_dir := os.environ.get("AI2_RUNS_DIR"):
        out.setdefault("tracker", {})["runs_dir"] = runs_dir
    return out


def _validate(cfg: Config) -> None:
    """실행 전에 잡을 수 있는 모순만 걸러낸다."""
    method = str(cfg.get("train.method", "qlora")).lower()
    if method not in {"lora", "qlora", "dora", "qdora", "fft"}:
        raise ConfigError(f"알 수 없는 train.method: {method}")

    if cfg.task == "train" and method == "fft":
        res = cfg.resource
        if res and not res.get("limits", {}).get("allow_fft", False):
            need = cfg.model.get("estimates", {}).get("fft_state_gb", "?")
            raise ConfigError(
                f"{res.get('name')}에서 FFT는 허용되지 않는다 "
                f"(학습 상태 약 {need}GB). PEFT를 쓰거나 자원을 키워라."
            )

    quant = str(cfg.get("model_config.load.quantization", "none")).lower()
    if method in {"qlora", "qdora"} and quant not in {"nf4", "int8"}:
        raise ConfigError(f"{method}는 양자화된 base가 필요하다 (현재: {quant})")

    crops = int(cfg.get("input.crops", 0) or 0)
    if crops not in {0, 4, 9}:
        raise ConfigError(f"input.crops는 0 / 4 / 9 중 하나여야 한다 (현재: {crops})")

    if cfg.get("input.allow_upscale"):
        raise ConfigError(
            "전체 이미지 업스케일은 금지다. 정보를 만들지 못하면서 visual token만 늘리고, "
            "생성형 SR은 판독 불가 글자를 '선명하지만 틀린 글자'로 바꾼다."
        )

    max_b = cfg.get("resource_config.limits.max_model_params_b")
    total_b = cfg.get("model_config.architecture.total_params_b")
    if max_b and total_b and float(total_b) > float(max_b):
        raise ConfigError(
            f"{cfg.get('model')}({total_b}B)는 {cfg.get('resource')}의 상한 {max_b}B를 넘는다"
        )
