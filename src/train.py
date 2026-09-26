"""PEFT / FFT 학습 루프.

고정 원칙 세 가지가 코드로 강제된다:
  - loss masking: 질문·선택지는 마스킹하고 정답 토큰에만 loss (models/*.training_example)
  - best checkpoint는 validation Accuracy 기준 (checkpoint.CheckpointManager)
  - Group Split 후 누수 검사를 통과해야 학습이 시작된다 (data.check_leakage)
"""

from __future__ import annotations

import math
import random
import time
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from . import resources
from .checkpoint import CheckpointManager, smoke_test_roundtrip
from .config import Config, ConfigError
from .data import Sample, build_views, check_leakage, fixed_split, group_split, load_split, subset
from .infer import run as run_infer
from .models import build_adapter
from .tracker import Tracker


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class VQADataset(Dataset):
    """training_example을 lazy하게 만든다 — 이미지를 미리 다 열면 메모리가 터진다."""

    def __init__(self, samples: list[Sample], adapter, cfg: Config):
        self.samples = samples
        self.adapter = adapter
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        views = build_views(sample, self.cfg)
        return self.adapter.training_example(sample, views)


def make_collate(image_concat_keys: frozenset[str]):
    """microbatch=1이 기본이라 대부분 그대로 통과한다. 메타 키는 떼어낸다.

    image_concat_keys(pixel_values·image_grid_thw 등)는 배치 축이 없다 — crop을
    쓰면(view 2개 이상) 여기에 unsqueeze(0)을 걸면 model이 기대하는 "여러 이미지가
    이어붙은" 모양이 깨진다(models/base.py의 IMAGE_CONCAT_KEYS 주석 참조).
    """
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        if len(batch) == 1:
            item = batch[0]
            return {
                k: (v if k in image_concat_keys else v.unsqueeze(0))
                for k, v in item.items()
                if not k.startswith("_") and torch.is_tensor(v)
            }
        raise NotImplementedError(
            "microbatch > 1은 아직 지원하지 않는다. 고해상도에서는 1이 기본이고, "
            "batch sweep은 추론(infer.batch_size)에서 한다."
        )
    return collate


def build_peft_model(adapter, cfg: Config):
    method = str(cfg.get("train.method", "qlora")).lower()
    if method == "fft":
        return adapter.model

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    quant = str(cfg.get("model_config.load.quantization", "none")).lower()
    if quant in {"nf4", "int8"}:
        adapter.model = prepare_model_for_kbit_training(
            adapter.model,
            use_gradient_checkpointing=bool(cfg.get("train.grad_checkpointing", True)),
        )

    lora_cfg = LoraConfig(
        r=int(cfg.get("train.r", 8)),
        lora_alpha=int(cfg.get("train.alpha", 16)),
        lora_dropout=float(cfg.get("train.dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=adapter.peft_target_modules(),
        use_dora=method in {"dora", "qdora"},
    )
    return get_peft_model(adapter.model, lora_cfg)


def _trainable_summary(model) -> dict[str, Any]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 4) if total else 0.0,
    }


def run(cfg: Config, tracker: Tracker) -> dict[str, Any]:
    set_seed(cfg.seed)

    # --- 데이터와 split ---
    samples, schema = load_split(cfg, "train")
    tracker.event("schema_resolved", split="train", **schema.to_dict())

    parts = fixed_split(cfg, samples)
    if parts is None:
        cache = cfg.path("data.split.cache", "runs/_splits") / f"{cfg.seed}_group.json"
        parts = group_split(
            samples, float(cfg.get("data.split.val_fraction", 0.1)), cfg.seed, cache
        )
    train_samples, val_samples = parts
    leak = check_leakage(train_samples, val_samples)
    # probe 학습: 방향 탐색 단계는 전체 대신 고정 subset으로 돌린다(같은 seed → 같은 표본)
    train_samples = subset(train_samples, cfg.get("train.max_samples"), cfg.seed)
    val_samples = subset(val_samples, cfg.get("train.eval_size"), cfg.seed)
    tracker.event("split", n_train=len(train_samples), n_val=len(val_samples),
                  split_file=cfg.get("data.split.file"), **leak)
    if not leak["clean"]:
        raise ConfigError(f"그룹 누수가 있다: {leak['overlap_sample']}")

    # --- 모델 ---
    resources.preflight(cfg, tracker)
    adapter = build_adapter(cfg)
    t0 = time.time()
    adapter.ensure_loaded(for_training=True)
    tracker.event("model_loaded", seconds=round(time.time() - t0, 2))
    tracker.event("kernel_impl", **resources.kernel_impl_snapshot())

    adapter.model = build_peft_model(adapter, cfg)
    tracker.event("peft_ready", method=cfg.get("train.method"),
                  targets=adapter.peft_target_modules(), **_trainable_summary(adapter.model))

    # 본학습 전에 저장→재로드→추론이 도는지 먼저 확인한다.
    # MoE fused expert weight에서 이 경로가 깨진 선례(vLLM #38520)가 있다.
    if val_samples:
        probe = val_samples[0]
        smoke_test_roundtrip(adapter, probe, build_views(probe, cfg), tracker)

    # --- 옵티마이저 ---
    loader = DataLoader(
        VQADataset(train_samples, adapter, cfg),
        batch_size=int(cfg.get("train.microbatch", 1)),
        shuffle=True,
        collate_fn=make_collate(adapter.IMAGE_CONCAT_KEYS),
        num_workers=0,          # 이미지 처리는 processor가 하므로 워커를 늘리지 않는다
    )
    grad_accum = int(cfg.get("train.grad_accum", 16))
    epochs = int(cfg.get("train.epochs", 2))
    total_steps = max(1, math.ceil(len(loader) / grad_accum) * epochs)
    warmup = int(total_steps * float(cfg.get("train.warmup_ratio", 0.03)))

    optimizer = torch.optim.AdamW(
        [p for p in adapter.model.parameters() if p.requires_grad],
        lr=float(cfg.get("train.lr", 2e-5)),
        weight_decay=float(cfg.get("train.weight_decay", 0.0)),
    )
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    ckpt = CheckpointManager(tracker.dir, metric=str(cfg.get("train.save_best_by", "accuracy")))
    eval_every = int(cfg.get("train.eval_every_steps", 50))
    patience = int(cfg.get("train.early_stopping_patience", 2))
    max_grad_norm = float(cfg.get("train.max_grad_norm", 1.0))

    tracker.event("train_start", total_steps=total_steps, warmup_steps=warmup,
                  epochs=epochs, grad_accum=grad_accum)

    step = 0
    since_improve = 0
    best_acc = -1.0
    resources.reset_peak()

    for epoch in range(epochs):
        adapter.model.train()
        running = 0.0
        for i, batch in enumerate(tqdm(loader, desc=f"train e{epoch}", unit="mb")):
            batch = {k: v.to(adapter.model.device) for k, v in batch.items()}
            loss = adapter.model(**batch).loss / grad_accum
            loss.backward()
            running += loss.item()

            if (i + 1) % grad_accum != 0:
                continue

            torch.nn.utils.clip_grad_norm_(
                [p for p in adapter.model.parameters() if p.requires_grad], max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % int(cfg.get("tracker.log_every_steps", 10)) == 0:
                tracker.metric(step=step, loss=round(running, 6),
                               lr=scheduler.get_last_lr()[0],
                               peak_vram_gb=resources.peak_gb())
            running = 0.0

            if step % eval_every == 0:
                acc = _validate(cfg, tracker, adapter, val_samples, step)
                rec = ckpt.save(adapter.model, step, acc, processor=adapter.processor)
                tracker.event("checkpoint", step=step, accuracy=acc, path=rec.path)
                if acc > best_acc:
                    best_acc, since_improve = acc, 0
                else:
                    since_improve += 1
                    if since_improve >= patience:
                        tracker.event("early_stop", step=step, best_accuracy=best_acc)
                        break
                adapter.model.train()
        else:
            continue
        break

    # 마지막 평가와 저장
    final_acc = _validate(cfg, tracker, adapter, val_samples, step)
    ckpt.save(adapter.model, step, final_acc, processor=adapter.processor)
    best = ckpt.best

    result = {
        "steps": step,
        "best_accuracy": best.accuracy if best else final_acc,
        "best_step": best.step if best else step,
        "best_checkpoint": best.path if best else None,
        "final_accuracy": final_acc,
        "peak_vram_gb": resources.peak_gb(),
        "n_train": len(train_samples),
        "n_val": len(val_samples),
    }
    tracker.set_summary(train=result)
    tracker.event("train_done", **result)
    return result


def run_synthetic_probe(cfg: Config, tracker: Tracker, n_steps: int = 12,
                        n_warmup: int = 3) -> dict[str, Any]:
    """대회 데이터 없이 forward/backward/optimizer step의 VRAM·속도만 잰다.

    `run()`의 CSV 로드·Group Split·누수 검사·validation Accuracy 채점은 전부 건너뛴다 —
    09/21 전에는 데이터가 없고, 이 probe의 목적은 "학습이 몇 GB를 먹는가"이지 실제
    학습이 아니다(PREFLIGHT_LOCAL.md 5절). best checkpoint 로직도 쓰지 않는다.

    09/17: n_steps=3 평균을 sec/step으로 썼더니 Triton JIT 컴파일(최초 shape에서만
    발생, 수십 초)이 평균에 섞여 왜곡됐다. 앞 n_warmup step은 컴파일·캐시 예열
    구간으로 따로 두고, steady-state는 그 이후 step들의 **중앙값**(평균이 아니다 —
    한두 개 튀는 값에 흔들리지 않도록)으로 낸다. 기본을 12 step으로 늘려 steady-state
    표본을 충분히 확보한다.
    """
    from .data import make_synthetic

    set_seed(cfg.seed)
    resources.preflight(cfg, tracker)

    adapter = build_adapter(cfg)
    t0 = time.time()
    adapter.ensure_loaded(for_training=True)
    tracker.event("model_loaded", seconds=round(time.time() - t0, 2))
    kernel_impl = resources.kernel_impl_snapshot()
    tracker.event("kernel_impl", **kernel_impl)

    adapter.model = build_peft_model(adapter, cfg)
    tracker.event("peft_ready", method=cfg.get("train.method"),
                  targets=adapter.peft_target_modules(), **_trainable_summary(adapter.model))

    samples = make_synthetic(cfg, n_steps + 1)
    optimizer = torch.optim.AdamW(
        [p for p in adapter.model.parameters() if p.requires_grad],
        lr=float(cfg.get("train.lr", 2e-5)),
        weight_decay=float(cfg.get("train.weight_decay", 0.0)),
    )
    max_grad_norm = float(cfg.get("train.max_grad_norm", 1.0))

    adapter.model.train()
    resources.reset_peak()
    step_metrics: list[dict[str, Any]] = []

    with resources.NvidiaSmiPeakSampler() as smi, resources.WindowsSharedMemPeakSampler() as shared_mem:
        for i in range(n_steps):
            sample = samples[i]
            views = build_views(sample, cfg)
            example = adapter.training_example(sample, views)
            batch = {
                k: (v if k in adapter.IMAGE_CONCAT_KEYS else v.unsqueeze(0)).to(adapter.model.device)
                for k, v in example.items()
                if not k.startswith("_") and torch.is_tensor(v)
            }

            t0 = time.time()
            stage = "forward"
            try:
                loss = adapter.model(**batch).loss
                stage = "backward"
                loss.backward()
                stage = "optimizer_step"
                torch.nn.utils.clip_grad_norm_(
                    [p for p in adapter.model.parameters() if p.requires_grad], max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            except torch.cuda.OutOfMemoryError as exc:
                tracker.event("train_probe_oom", step=i, stage=stage, error=str(exc))
                raise

            sec = time.time() - t0
            peak_torch = resources.peak_gb()
            metric = {
                "step": i,
                "warmup": i < n_warmup,
                "stage_reached": "done",
                "loss": round(loss.item(), 6),
                "sec": round(sec, 3),
                "peak_vram_gb_torch": round(peak_torch, 3) if peak_torch is not None else None,
            }
            step_metrics.append(metric)
            tracker.metric(step=i, loss=metric["loss"], sec_per_step=metric["sec"],
                           peak_vram_gb=metric["peak_vram_gb_torch"])

    warmup_secs = [s["sec"] for s in step_metrics if s["warmup"]]
    steady_secs = sorted(s["sec"] for s in step_metrics if not s["warmup"])
    steady_median = steady_secs[len(steady_secs) // 2] if steady_secs else None
    if steady_secs and len(steady_secs) % 2 == 0 and len(steady_secs) > 0:
        # 짝수 개면 가운데 두 값의 평균 — 표준적인 중앙값 정의를 그대로 쓴다.
        steady_median = (steady_secs[len(steady_secs) // 2 - 1] + steady_secs[len(steady_secs) // 2]) / 2

    result = {
        "n_steps": n_steps,
        "n_warmup": n_warmup,
        "grad_checkpointing": bool(cfg.get("train.grad_checkpointing", True)),
        "kernel_impl": kernel_impl,
        "steps": step_metrics,
        "peak_vram_gb_torch": round(resources.peak_gb(), 3) if resources.peak_gb() is not None else None,
        "peak_vram_gb_nvidia_smi": round(smi.peak, 3) if smi.peak is not None else None,
        # WDDM sysmem fallback(Windows 전용) — dedicated VRAM 부족 시 CUDA가 OOM
        # 대신 조용히 시스템 RAM으로 흘려보내는 양. torch/nvidia-smi 둘 다 못 본다.
        # 0보다 크면 "성공"으로 보인 peak_vram_gb가 이미 sysmem을 쓰고 있었다는 뜻이다.
        "peak_shared_mem_gb_windows": round(shared_mem.peak, 3) if shared_mem.peak is not None else None,
        "warmup_sec_per_step": warmup_secs,
        "steady_state_sec_per_step": steady_secs,
        "steady_state_sec_per_step_median": round(steady_median, 3) if steady_median is not None else None,
    }
    tracker.set_summary(train_probe=result)
    tracker.event("train_probe_done", **result)
    return result


def _validate(cfg: Config, tracker: Tracker, adapter, val_samples: list[Sample], step: int) -> float:
    """validation Accuracy. checkpoint 선택 기준은 이 값이지 loss가 아니다."""
    if not val_samples:
        return 0.0
    adapter.model.eval()
    with torch.no_grad():
        result = run_infer(cfg, tracker, split="val", samples=val_samples, adapter=adapter)
    acc = float(result.get("accuracy") or 0.0)
    tracker.metric(step=step, val_accuracy=acc)
    return acc
