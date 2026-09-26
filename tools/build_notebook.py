"""ipynb를 손으로 쓰지 않고 코드로 조립하기 위한 최소 헬퍼.

markdown()/code()로 셀을 만들고 write_notebook()으로 nbformat v4 JSON을 저장한다.
Day 1 노트북들(day1_dataset_audit / day1_baseline_9b / day1_cloud_model_probe)과
full1536_crop_probe_docker.ipynb가 전부 이 방식으로 만들어졌다 — 노트북 JSON을
직접 타이핑하면 source 배열의 개행 처리가 쉽게 깨지므로 항상 이 헬퍼를 거친다.
"""

from __future__ import annotations

import json
from pathlib import Path


def _split_source(lines: tuple[str, ...]) -> list[str]:
    if not lines:
        return []
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def markdown(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _split_source(lines)}


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _split_source(lines),
    }


def build_notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(path: str | Path, cells: list[dict]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_notebook(cells), ensure_ascii=False, indent=1), encoding="utf-8")
    return out
