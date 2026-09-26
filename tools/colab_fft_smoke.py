"""35B FFT feasibility smoke test (B300 1장) — 메모리 실측 + 학습 실제 동작 검증.

목적: "35B 전체 파인튜닝이 B300 1장에 들어가는가 + 실제로 가중치가 움직이는가"를
GPU를 켠 뒤 20~30분 안에 판정한다. 본 학습이 아니다.

    python colab_fft_smoke.py --model Qwen/Qwen3.6-35B-A3B --data /content/data \
        --overfit-csv overfit30_train.csv --steps 3 --view full --optimizer adamw8bit

핵심 구성(ChatGPT/석웅 분석 반영):
  - BF16 전체 파라미터 학습(언어 모델). 비전 타워는 기본 동결(--train-vision로 포함).
  - 8-bit AdamW(bitsandbytes) — optimizer state 8-bit, FP32 master weight 없음(로그로 확인).
  - gradient checkpointing ON, microbatch=1.
  - 단계별 피크 메모리 기록: load / forward / backward / optimizer.step.
  - 특정 레이어 weight delta로 "업데이트가 BF16 반올림에 묻히지 않는지" 확인.
  - 3-step smoke 통과 후 --overfit-steps로 20~50 step overfit(loss 실제 하강 확인).
  - --save 시 저장→(같은 프로세스에서) 재로드 sanity.

판정 기준(예): optimizer.step 피크가 B300 288GB 대비 여유가 있으면 원본1장 FFT 성립.
280GB 이상이면 타일 입력에서 위험 → LoRA로 내린다. 결과는 out/fft_smoke.json에 남는다.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from colab_lora_train import LETTERS, build_inputs  # 같은 입력·프롬프트를 쓴다


def gb(x: int) -> float:
    return round(x / 1024**3, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--data", required=True)
    ap.add_argument("--overfit-csv", required=True, help="학습 subset(id,path,question,a~d,answer). train에서 뽑은 것")
    ap.add_argument("--out", default="/workspace/out/fft_smoke")
    ap.add_argument("--steps", type=int, default=3, help="메모리 실측용 최소 step")
    ap.add_argument("--overfit-steps", type=int, default=0, help=">0이면 subset을 반복해 loss 하강 확인")
    ap.add_argument("--optimizer", default="adamw8bit", choices=["adamw8bit", "adamw"])
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--view", default="full", choices=["full", "tiles2x2"])
    ap.add_argument("--tile-scale", type=float, default=2.0)
    ap.add_argument("--train-vision", action="store_true", help="비전 타워도 학습(기본 동결)")
    ap.add_argument("--save", action="store_true", help="끝에 전체 저장→재로드 sanity(느림/디스크 큼)")
    ap.add_argument("--revision", default=None)
    args = ap.parse_args()

    import pandas as pd
    import torch
    from PIL import Image, ImageOps
    from transformers import AutoModelForImageTextToText, AutoProcessor

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rec: dict = {"model": args.model, "view": args.view, "optimizer": args.optimizer, "stages": {}, "steps": []}
    torch.cuda.reset_peak_memory_stats()

    df = pd.read_csv(args.overfit_csv, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df = df[df.answer.str.lower().isin(LETTERS)].reset_index(drop=True)
    data = Path(args.data)
    processor = AutoProcessor.from_pretrained(args.model, use_fast=True, revision=args.revision)
    tok = processor.tokenizer
    letter_ids = [tok.encode(x, add_special_tokens=False)[0] for x in LETTERS]

    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa", revision=args.revision)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    n_train = n_frozen = 0
    for name, p in model.named_parameters():
        is_vision = "visual" in name or "vision" in name
        p.requires_grad = args.train_vision or not is_vision
        (n_train := n_train + p.numel()) if p.requires_grad else (n_frozen := n_frozen + p.numel())
    # 위 walrus 트릭 대신 명시적으로 다시 센다(가독성)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    rec["trainable_params_B"] = round(n_train / 1e9, 2)
    rec["frozen_params_B"] = round(n_frozen / 1e9, 2)
    rec["param_dtype"] = str(next(model.parameters()).dtype)
    rec["stages"]["after_load"] = gb(torch.cuda.max_memory_allocated())
    rec["load_sec"] = round(time.time() - t0, 1)
    print(f"[fft] load {rec['stages']['after_load']}GB, trainable {rec['trainable_params_B']}B "
          f"frozen {rec['frozen_params_B']}B dtype {rec['param_dtype']}", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "adamw8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=args.lr, weight_decay=0.0)
    else:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    # weight delta 관찰용: 학습되는 마지막 language_model linear 하나를 고른다
    probe_name, probe = None, None
    for name, p in model.named_parameters():
        if p.requires_grad and "language_model" in name and p.dim() == 2:
            probe_name, probe = name, p
    probe0 = probe.detach().float().clone() if probe is not None else None

    load_img = lambda p: ImageOps.exif_transpose(Image.open(data / p)).convert("RGB")  # noqa: E731

    def one(i: int) -> float:
        r = df.iloc[i % len(df)]
        shown = [r[x] for x in LETTERS]
        gold = letter_ids[LETTERS.index(r["answer"].strip().lower())]
        inputs = build_inputs(processor, load_img(r["path"]), r["question"], shown, args.view, args.tile_scale).to(model.device)
        logits = model(**inputs).logits[0, -1].float()
        return torch.nn.functional.cross_entropy(logits[None], torch.tensor([gold], device=logits.device))

    model.train()
    for s in range(args.steps):
        torch.cuda.reset_peak_memory_stats()
        loss = one(s)
        rec["stages"]["after_forward"] = gb(torch.cuda.max_memory_allocated())
        loss.backward()
        rec["stages"]["after_backward"] = gb(torch.cuda.max_memory_allocated())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        rec["stages"]["after_step"] = gb(torch.cuda.max_memory_allocated())
        rec["steps"].append({"step": s, "loss": round(loss.item(), 4)})
        print(f"[fft] step {s} loss {loss.item():.4f} | fwd {rec['stages']['after_forward']} "
              f"bwd {rec['stages']['after_backward']} step {rec['stages']['after_step']} GB", flush=True)

    if probe0 is not None:
        delta = (probe.detach().float() - probe0).abs()
        rec["weight_delta"] = {"param": probe_name, "mean_abs": float(delta.mean()), "max_abs": float(delta.max()),
                               "changed": bool(delta.max() > 0)}
        print(f"[fft] weight delta {probe_name}: mean {rec['weight_delta']['mean_abs']:.2e} "
              f"max {rec['weight_delta']['max_abs']:.2e} changed={rec['weight_delta']['changed']}", flush=True)

    if args.overfit_steps:
        print(f"[fft] overfit {args.overfit_steps} steps on {len(df)} samples", flush=True)
        for s in range(args.overfit_steps):
            loss = one(s)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if s % 5 == 0 or s == args.overfit_steps - 1:
                print(f"[fft] overfit step {s} loss {loss.item():.4f}", flush=True)
                rec["steps"].append({"overfit_step": s, "loss": round(loss.item(), 4)})

    if args.save:
        sp = out / "fft_ckpt"
        t = time.time()
        model.save_pretrained(sp, safe_serialization=True)
        rec["save_sec"] = round(time.time() - t, 1)
        del model
        torch.cuda.empty_cache()
        m2 = AutoModelForImageTextToText.from_pretrained(sp, dtype=torch.bfloat16, device_map="cuda")
        rec["reload_ok"] = True
        print(f"[fft] save→reload ok ({rec['save_sec']}s)", flush=True)

    rec["total_sec"] = round(time.time() - t0, 1)
    (out / "fft_smoke.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[fft] DONE → {out / 'fft_smoke.json'}\n{json.dumps(rec['stages'], indent=2)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
