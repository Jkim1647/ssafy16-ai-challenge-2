"""데이터 로딩, 스키마 추론, view(full/crop) 생성.

CSV 스키마는 암호 해제(09/21) 전까지 확정되지 않았다. 그래서 컬럼명을 설정으로 받되,
비어 있으면 헤더에서 추론하고 추론 결과를 run 디렉터리에 기록한다. 암호가 풀리면
configs/base.yaml의 data.columns를 실제 이름으로 고정하고 추론 경로를 끈다.
"""

from __future__ import annotations

import csv
import functools
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from .config import REPO_ROOT, Config, ConfigError

RESAMPLE = {
    "bicubic": Image.BICUBIC,
    "bilinear": Image.BILINEAR,
    "lanczos": Image.LANCZOS,
    "nearest": Image.NEAREST,
}

# 헤더 자동 추론에 쓰는 후보. 왼쪽이 우선한다.
_CANDIDATES = {
    "id": ["id", "sample_id", "qid", "question_id", "index", "idx"],
    "image": ["image", "image_path", "img", "img_path", "filename", "file_name", "image_id"],
    "question": ["question", "query", "prompt", "text", "질문"],
    "answer": ["answer", "label", "target", "gt", "correct", "정답"],
    "choices": ["choices", "options", "candidates", "선택지"],
    "type": ["type", "category", "question_type", "qtype", "유형"],
}


@functools.lru_cache(maxsize=1)
def _duplicate_group_map() -> dict[str, str]:
    """tools/extract_duplicates.py가 만든 known_duplicates.csv를 union-find로 묶는다.

    (file_size, crc32)가 같아 내용이 같다고 확인된 두 이미지는 파일명이 달라도
    같은 그룹으로 취급해야 한다 — 그래야 group split에서 서로 다른 fold로
    갈리는 걸 막을 수 있다. 파일이 아직 없으면(추출 전) 빈 매핑을 돌려주고
    group_key는 평소처럼 파일명 단위로 동작한다.
    """
    path = REPO_ROOT / "data_meta" / "known_duplicates.csv"
    if not path.exists():
        return {}

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    with open(path, encoding="utf-8-sig") as f:  # utf-8-sig: BOM 붙은 중복 목록도 안전하게 읽는다
        for row in csv.DictReader(f):
            union(row["stem_a"], row["stem_b"])

    return {stem: find(stem) for stem in parent}


@dataclass
class Sample:
    id: str
    image_path: Path
    question: str
    choices: list[str]
    answer: str | None = None
    qtype: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def group_key(self) -> str:
        """Group Split 기준. 같은 이미지가 train/val에 동시에 들어가지 않게 한다.

        known_duplicates.csv에 있는 이미지는 파일명이 달라도 같은 그룹으로 묶는다 —
        내용이 같은 두 이미지가 서로 다른 fold에 갈리면 그 자체가 누수다.
        """
        canonical = _duplicate_group_map().get(self.image_path.stem)
        return canonical if canonical else self.image_path.name


@dataclass
class Schema:
    id: str
    image: str
    question: str
    choices: list[str]          # 단일 컬럼이면 길이 1, 보기별 컬럼이면 길이 N
    choices_mode: str           # "single" | "columns"
    answer: str | None
    qtype: str | None
    inferred: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "image": self.image,
            "question": self.question,
            "choices": self.choices,
            "choices_mode": self.choices_mode,
            "answer": self.answer,
            "type": self.qtype,
            "inferred": self.inferred,
        }


def _pick(columns: Sequence[str], names: Iterable[str]) -> str | None:
    lowered = {c.lower().strip(): c for c in columns}
    for want in names:
        if want in lowered:
            return lowered[want]
    for want in names:
        for low, orig in lowered.items():
            if want in low:
                return orig
    return None


def infer_schema(df: pd.DataFrame, configured: dict[str, Any] | None = None) -> Schema:
    """설정에 있는 컬럼명을 우선 쓰고, 없는 것만 헤더에서 추론한다."""
    configured = configured or {}
    cols = list(df.columns)

    def resolve(key: str) -> str | None:
        given = configured.get(key)
        if given:
            if given not in cols:
                raise ConfigError(f"data.columns.{key}={given!r} 컬럼이 CSV에 없다. 헤더: {cols}")
            return given
        return _pick(cols, _CANDIDATES[key])

    id_col = resolve("id")
    image_col = resolve("image")
    question_col = resolve("question")
    # 정답 컬럼은 split마다 있을 수도 없을 수도 있다(test는 없음, dev는 answer1~5).
    # 설정된 이름이 없는 split은 추론으로 엉뚱한 컬럼(answer1)을 잡지 말고 None으로 둔다.
    given_answer = configured.get("answer")
    answer_col = (given_answer if given_answer in cols else None) if given_answer else resolve("answer")
    type_col = resolve("type")

    # 선택지: 단일 컬럼(choices) 또는 보기별 컬럼(A/B/C/D, choice_1..4, option_a..)
    choice_cols: list[str] = []
    mode = "single"
    given_choices = configured.get("choices")
    if given_choices:
        choice_cols = [given_choices] if isinstance(given_choices, str) else list(given_choices)
        mode = "single" if len(choice_cols) == 1 else "columns"
    else:
        single = _pick(cols, _CANDIDATES["choices"])
        if single:
            choice_cols, mode = [single], "single"
        else:
            pat = re.compile(r"^(choice|option|answer)?[_\-\s]?([a-dA-D]|[1-4])$")
            found = [c for c in cols if pat.match(c.strip())]
            if found:
                choice_cols, mode = sorted(found, key=lambda c: c.strip()[-1].lower()), "columns"

    if not (image_col and question_col):
        raise ConfigError(
            "image/question 컬럼을 추론하지 못했다. configs/base.yaml의 data.columns를 "
            f"직접 지정해라. 헤더: {cols}"
        )
    if not id_col:
        id_col = image_col

    return Schema(
        id=id_col,
        image=image_col,
        question=question_col,
        choices=choice_cols,
        choices_mode=mode,
        answer=answer_col,
        qtype=type_col,
        inferred=not configured.get("image"),
    )


def _split_choices(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value]
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed]
        except json.JSONDecodeError:
            pass
    for sep in ("|", "\t", ";", "^"):
        if sep in text:
            return [p.strip() for p in text.split(sep) if p.strip()]
    return [p.strip() for p in text.split(",") if p.strip()]


def load_split(cfg: Config, split: str) -> tuple[list[Sample], Schema]:
    """train / dev / test 하나를 Sample 리스트로 읽는다."""
    csv_path = cfg.path(f"data.{split}_csv")
    if not csv_path.exists():
        raise ConfigError(
            f"{csv_path} 가 없다. 데이터 ZIP은 암호화돼 있고 암호는 대회 시작일 공개된다."
        )
    # utf-8-sig: BOM이 있든 없든 안전. BOM 붙은 CSV를 그냥 utf-8로 읽으면
    # 첫 컬럼명이 '﻿id'가 되어 infer_schema가 id 컬럼을 못 찾는다.
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    schema = infer_schema(df, cfg.get("data.columns") or {})
    image_dir = cfg.path(f"data.image_dirs.{split}", f"data/{split}")

    samples: list[Sample] = []
    for _, row in df.iterrows():
        raw_image = str(row[schema.image]).strip()
        image_path = Path(raw_image)
        if not image_path.is_absolute():
            image_path = image_dir / image_path.name

        if schema.choices_mode == "columns":
            choices = [str(row[c]).strip() for c in schema.choices]
        elif schema.choices:
            choices = _split_choices(row[schema.choices[0]])
        else:
            choices = []

        samples.append(
            Sample(
                id=str(row[schema.id]),
                image_path=image_path,
                question=str(row[schema.question]).strip(),
                choices=choices,
                answer=str(row[schema.answer]).strip() if schema.answer else None,
                qtype=str(row[schema.qtype]).strip() if schema.qtype else None,
            )
        )
    return samples, schema


def fixed_split(cfg: Config, samples: list[Sample]) -> tuple[list[Sample], list[Sample]] | None:
    """팀 공용 분할(data.split.file)로 train을 fit/holdout으로 나눈다. 파일 미지정이면 None.

    dev는 라벨러 합의 미달 문항만 모인 편향 표본이라 검증셋으로 쓰지 않는다
    (reports/train_val_split_report.md). 검증은 이 파일의 val(train holdout)로 한다.
    """
    rel = cfg.get("data.split.file")
    if not rel:
        return None
    path = cfg.path("data.split.file")
    if not path.exists():
        raise ConfigError(f"data.split.file={rel} 이 없다")

    # 제외 목록(예: test와 겹치는 train 중복 45건) — fit·val 양쪽에서 뺀다
    excluded: set[str] = set()
    if cfg.get("data.split.exclude_file"):
        ex_path = cfg.path("data.split.exclude_file")
        if not ex_path.exists():
            raise ConfigError(f"data.split.exclude_file={ex_path} 이 없다")
        excluded = set(pd.read_csv(ex_path, encoding="utf-8-sig")["id"].astype(str))
        samples = [s for s in samples if s.id not in excluded and s.image_path.name not in excluded]

    if path.suffix == ".json":
        # group split JSON (석웅 브랜치 형식): {"sample_count", "val_groups": [...]}. group = 이미지 파일명
        payload = json.loads(path.read_text(encoding="utf-8"))
        val_keys = set(map(str, payload["val_groups"]))
        expected = payload.get("sample_count")
        if expected is not None and expected != len(samples):
            raise ConfigError(f"split JSON sample_count={expected}인데 제외 후 표본은 {len(samples)}개 — "
                              "data.split.exclude_file이 split을 만든 기준과 같은지 확인")
        fit = [s for s in samples if s.group_key not in val_keys and s.image_path.name not in val_keys]
        val = [s for s in samples if s.group_key in val_keys or s.image_path.name in val_keys]
        return fit, val

    df = pd.read_csv(path, encoding="utf-8-sig")
    val_keys = set(df.loc[df["split"] == "val", "id"].astype(str))
    known = set(df["id"].astype(str)) - excluded

    # 분할 CSV의 id가 CSV id인지 이미지 파일명인지 둘 다 받아준다
    def key(s: Sample) -> str:
        return s.id if s.id in known else s.image_path.name

    missing = [s.id for s in samples if key(s) not in known]
    if missing:
        raise ConfigError(f"분할 파일에 없는 train 샘플 {len(missing)}개: {missing[:5]}")
    fit = [s for s in samples if key(s) not in val_keys]
    val = [s for s in samples if key(s) in val_keys]
    return fit, val


def load_named_split(cfg: Config, split: str) -> tuple[list[Sample], Schema]:
    """train/dev/test에 더해 val(팀 holdout)·train_fit(holdout 제외 train)을 지원한다."""
    if split not in {"val", "train_fit"}:
        return load_split(cfg, split)
    samples, schema = load_split(cfg, "train")
    parts = fixed_split(cfg, samples)
    if parts is None:
        raise ConfigError("split=val/train_fit은 data.split.file이 설정돼 있어야 한다")
    return (parts[1] if split == "val" else parts[0]), schema


def group_split(
    samples: list[Sample], val_fraction: float, seed: int, cache: Path | None = None
) -> tuple[list[Sample], list[Sample]]:
    """이미지 단위 Group Split. 해시 기반이라 순서·seed가 같으면 항상 같은 결과다.

    한 번 만든 split은 파일로 고정한다 — 매번 새로 나누면 실험 간 비교가 불가능해진다.
    """
    if cache and cache.exists():
        held = set(json.loads(cache.read_text(encoding="utf-8"))["val_groups"])
    else:
        groups = sorted({s.group_key for s in samples})
        scored = sorted(
            groups,
            key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest(),
        )
        n_val = max(1, int(round(len(scored) * val_fraction)))
        held = set(scored[:n_val])
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(
                json.dumps({"seed": seed, "val_fraction": val_fraction, "val_groups": sorted(held)},
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    train = [s for s in samples if s.group_key not in held]
    val = [s for s in samples if s.group_key in held]
    return train, val


def check_leakage(train: list[Sample], val: list[Sample]) -> dict[str, Any]:
    """split 후 그룹 누수가 없는지 확인한다. 학습 전에 반드시 호출한다."""
    tg = {s.group_key for s in train}
    vg = {s.group_key for s in val}
    overlap = sorted(tg & vg)
    return {
        "train_groups": len(tg),
        "val_groups": len(vg),
        "overlap_count": len(overlap),
        "overlap_sample": overlap[:10],
        "clean": not overlap,
    }


# --- view 생성 -------------------------------------------------------------

def _resize_long_side(img: Image.Image, long_side: int | None, resample: int) -> Image.Image:
    """긴 변을 목표 크기로 맞춘다. 업스케일은 하지 않는다(정보를 만들지 못한다)."""
    if not long_side:
        return img
    w, h = img.size
    cur = max(w, h)
    if cur <= long_side:
        return img
    scale = long_side / cur
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), resample)


def _grid_crops(img: Image.Image, n: int, overlap: float) -> list[Image.Image]:
    if n == 0:
        return []
    side = 2 if n == 4 else 3
    w, h = img.size
    cw, ch = w / side, h / side
    ox, oy = cw * overlap, ch * overlap
    out: list[Image.Image] = []
    for r in range(side):
        for c in range(side):
            left = max(0, int(c * cw - ox))
            top = max(0, int(r * ch - oy))
            right = min(w, int((c + 1) * cw + ox))
            bottom = min(h, int((r + 1) * ch + oy))
            out.append(img.crop((left, top, right, bottom)))
    return out


def build_views(sample: Sample, cfg: Config) -> list[Image.Image]:
    """모델에 넣을 이미지 리스트를 만든다. [full] 또는 [full, crop...] 순서다.

    허용되는 것은 crop(원본 픽셀을 잘라 쓰기)과 비생성형 보정뿐이다.
    전체 업스케일과 AI super-resolution은 config._validate에서 이미 막힌다.
    """
    resample = RESAMPLE[str(cfg.get("input.interpolation", "bicubic")).lower()]
    with Image.open(sample.image_path) as raw:
        # EXIF Orientation을 픽셀에 적용한다. 단 대회 데이터셋은 EXIF가 전부 제거돼 있어
        # (16,111장 전수 확인: Orientation 태그 부재, 짧은 변 720px 강제 리사이즈) 여기서는
        # 실효가 없는 no-op다. 외부 데이터(태그가 남은 휴대폰 사진 등)가 유입될 때를 대비한
        # 방어 코드로 남긴다 — 방향 태그가 있으면 옆으로/거꾸로 들어가는 것을 막는다.
        img = ImageOps.exif_transpose(raw).convert("RGB")
    # 비생성형 회전 뷰(뒤집힌 글씨 대응). 원본 뷰와 확신도로 결합하는 것은 평가 쪽에서 한다.
    if rot := int(cfg.get("input.rotate", 0) or 0):
        img = img.rotate(rot, expand=True)

    views = [_resize_long_side(img, cfg.get("input.long_side"), resample)]

    n_crops = int(cfg.get("input.crops", 0) or 0)
    if n_crops:
        crop_side = cfg.get("input.crop_long_side")
        for crop in _grid_crops(img, n_crops, float(cfg.get("input.crop_overlap", 0.1))):
            views.append(_resize_long_side(crop, crop_side, resample))
    return views


def make_synthetic(cfg: Config, n: int = 4) -> list[Sample]:
    """대회 데이터 없이 파이프라인을 점검하기 위한 합성 샘플.

    암호 해제(09/21) 전에는 실제 CSV·이미지가 없다. 그래도 확인해야 하는 것이 있다 —
    가중치가 로드되는가, processor가 도는가, visual token이 몇 개 나오는가, PEFT
    저장/재로드가 통과하는가. 이 함수는 그 경로만 열어준다.

    여기서 나온 accuracy는 아무 의미가 없다. 의미 있는 것은 "터지지 않는가"와
    실측 VRAM·visual_tokens뿐이다.
    """
    out_dir = cfg.path("tracker.runs_dir", "runs") / "_synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 실제 휴대폰 원본급(약 12MP, 4:3) 고정 크기로 만든다 — cfg의 input.long_side에
    # 맞춰 만들면 안 된다. build_views._resize_long_side는 업스케일을 하지 않으므로,
    # 캐시된 합성 이미지가 이미 목표 long_side보다 작으면 --set input.long_side=1536/2048로
    # sweep을 돌려도 리사이즈가 전혀 일어나지 않아 항상 첫 실행 크기로 채점하는
    # 조용한 실패가 난다(실측: 첫 sweep에서 1024/1536 결과가 동일했다). 원본을 sweep
    # 대상 중 가장 큰 값보다 확실히 크게 고정해 매 실행이 실제로 리사이즈를 타게 한다.
    long_side = 4032
    letters = [str(p) for p in (cfg.get("data.columns.choice_prefixes") or ["A", "B", "C", "D"])]

    samples: list[Sample] = []
    for i in range(n):
        path = out_dir / f"synthetic_{i:03d}.jpg"
        if not path.exists():
            # 글자가 있는 이미지를 만든다 — 빈 단색 이미지는 vision encoder 경로를
            # 통과시키긴 해도 OCR 계열 동작을 전혀 건드리지 않는다.
            img = Image.new("RGB", (long_side, int(long_side * 0.75)), (245, 245, 240))
            draw = ImageDraw.Draw(img)
            draw.rectangle([40, 40, long_side - 40, int(long_side * 0.75) - 40], outline=(30, 30, 30), width=3)
            draw.text((80, 80), f"TEST {i:03d}\n가나다라 12,345원", fill=(10, 10, 10))
            img.save(path, quality=92)
        samples.append(
            Sample(
                id=f"synthetic_{i:03d}",
                image_path=path,
                question="이미지에 적힌 금액은 얼마인가?",
                choices=["12,345원", "23,456원", "34,567원", "45,678원"],
                answer=letters[0],
                qtype="Number",
            )
        )
    return samples


def subset(samples: list[Sample], size: int | None, seed: int, stratify_by: str | None = None) -> list[Sample]:
    """대표 subset 추출. 유형 컬럼이 있으면 유형 비율을 유지한다."""
    if not size or size >= len(samples):
        return samples

    def key(s: Sample) -> str:
        return hashlib.sha256(f"{seed}:{s.id}".encode()).hexdigest()

    if stratify_by == "type" and any(s.qtype for s in samples):
        buckets: dict[str, list[Sample]] = {}
        for s in samples:
            buckets.setdefault(s.qtype or "unknown", []).append(s)
        out: list[Sample] = []
        for qtype, group in sorted(buckets.items()):
            take = max(1, round(size * len(group) / len(samples)))
            out.extend(sorted(group, key=key)[:take])
        return sorted(out, key=key)[:size]

    return sorted(samples, key=key)[:size]
