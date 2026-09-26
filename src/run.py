"""단일 진입점.

  python -m src.run train    --config 35b_qdora_r8_1536
  python -m src.run infer     --config 9b_base_1024 --split dev
  python -m src.run evaluate  --run runs/20260921-101500_9b_base_1024
  python -m src.run compare   --baseline 9b_base_1024
  python -m src.run smoke     --config 9b_base_1024 --synthetic   # 데이터 없이 점검
  python -m src.run submit    --run runs/... --out submissions/stable.csv
  python -m src.run config    --config 35b_qdora_r8_1536      # 해석 결과만 출력

--set 으로 어떤 키든 덮어쓸 수 있다:
  python -m src.run infer --config 9b_base_1024 --set input.long_side=2048 --set input.crops=4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import REPO_ROOT, Config, ConfigError, load_config
from .tracker import Tracker

log = logging.getLogger("ai2")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _load(args) -> Config:
    return load_config(args.config, args.set or [])


# --- 명령 ---

def cmd_config(args) -> int:
    cfg = _load(args)
    print(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2, default=str))
    return 0


def cmd_train(args) -> int:
    cfg = _load(args)
    if cfg.task != "train":
        log.warning("%s의 task가 %r인데 train으로 실행한다", cfg.name, cfg.task)

    if args.synthetic:
        from .train import run_synthetic_probe

        with Tracker(cfg, run_id=f"trainprobe_{cfg.name}") as tracker:
            result = run_synthetic_probe(cfg, tracker, n_steps=args.n)
            shared = result.get("peak_shared_mem_gb_windows")
            log.info(
                "train probe 통과. peak_vram_gb(torch)=%.2f nvidia-smi=%s "
                "steady-state sec/step(median)=%s%s",
                result["peak_vram_gb_torch"],
                f"{result['peak_vram_gb_nvidia_smi']:.2f}" if result["peak_vram_gb_nvidia_smi"] else "?",
                result["steady_state_sec_per_step_median"],
                f" | ⚠ WDDM sysmem fallback {shared:.2f}GB — 조용히 느려지는 중" if shared else "",
            )
        return 0

    from .train import run as train_run

    with Tracker(cfg) as tracker:
        result = train_run(cfg, tracker)
        log.info("best accuracy %.4f (step %s) → %s",
                 result["best_accuracy"], result["best_step"], tracker.dir)
    return 0


def cmd_infer(args) -> int:
    from .infer import run as infer_run

    cfg = _load(args)
    with Tracker(cfg) as tracker:
        result = infer_run(cfg, tracker, split=args.split)
        acc = result.get("accuracy")
        log.info("%s n=%d accuracy=%s sec/sample=%.3f → %s",
                 args.split, result["n"], f"{acc:.4f}" if acc is not None else "n/a",
                 result["sec_per_sample"], tracker.dir)
    return 0


def cmd_evaluate(args) -> int:
    from .evaluate import main as evaluate_main

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    baseline = Path(args.baseline_run) if args.baseline_run else None
    report = evaluate_main(run_dir, baseline)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_compare(args) -> int:
    from .compare import run as compare_run

    runs_dir = REPO_ROOT / (args.runs_dir or "runs")
    out_dir = REPO_ROOT / (args.out or "reports/comparisons")
    payload = compare_run(runs_dir, out_dir, args.baseline, args.types or [])
    log.info("%d개 run을 비교했다 → %s", payload["n_runs"], out_dir / "comparison.md")
    print((out_dir / "comparison.md").read_text(encoding="utf-8"))
    return 0


def cmd_smoke(args) -> int:
    """본학습 전 필수 관문: 로드 → 토큰 실측 → 채점 → PEFT 저장/재로드."""
    from .checkpoint import smoke_test_roundtrip
    from .data import build_views, load_named_split, make_synthetic, subset
    from .models import build_adapter
    from .train import build_peft_model
    from . import resources

    cfg = _load(args)
    with Tracker(cfg, run_id=f"smoke_{cfg.name}") as tracker:
        resources.preflight(cfg, tracker)

        if args.synthetic:
            # 암호 해제 전에도 로드·processor·PEFT 경로를 점검하기 위한 경로.
            picked = make_synthetic(cfg, args.n)
            tracker.event("synthetic_data", n=len(picked),
                          note="합성 샘플이다. accuracy는 의미 없고 VRAM·visual_tokens만 본다")
        else:
            samples, schema = load_named_split(cfg, args.split)
            tracker.event("schema_resolved", split=args.split, **schema.to_dict())
            picked = subset(list(samples), args.n, cfg.seed, "type")

        resources.reset_peak()
        adapter = build_adapter(cfg)
        adapter.ensure_loaded(for_training=True)
        kernel_impl = resources.kernel_impl_snapshot()
        tracker.event("kernel_impl", **kernel_impl)

        sample = picked[0]
        views = build_views(sample, cfg)
        scored = adapter.score_choices(sample, views)

        # 계산 규칙 검증: config 계산값 vs processor 실측값.
        # 반드시 build_views가 만든(리사이즈·crop 이후) view 크기로 계산해야 한다.
        # 원본 파일 크기로 계산하면 input.long_side가 실제로 다운스케일하는 모든 설정에서
        # 값이 어긋난다 — processor는 항상 모델에 들어간 실제 크기를 기준으로 세기 때문이다.
        sizes = [v.size for v in views]
        expected = sum(cfg.visual_tokens(w, h) for w, h in sizes) if sizes else None
        tracker.event(
            "visual_token_check",
            image_sizes=sizes,
            expected_from_config=expected,
            actual_from_processor=scored.visual_tokens,
            match=(expected == scored.visual_tokens) if expected and scored.visual_tokens else None,
        )

        adapter.model = build_peft_model(adapter, cfg)
        result = smoke_test_roundtrip(adapter, sample, views, tracker)
        # reset_peak() 이후(로드~PEFT 왕복 전체)의 peak를 잰다 — 해상도/crop sweep이 보는 값
        peak_vram_gb = resources.peak_gb()
        tracker.metric(peak_vram_gb=peak_vram_gb)
        tracker.set_summary(smoke=result, visual_tokens=scored.visual_tokens, peak_vram_gb=peak_vram_gb,
                           kernel_impl=kernel_impl)

        if not result["ok"]:
            log.error("smoke test 실패 (%s): %s", result["stage"], result.get("error"))
            return 1
        log.info("smoke test 통과. visual_tokens=%s (config 계산=%s)",
                 scored.visual_tokens, expected)
    return 0


def cmd_submit(args) -> int:
    from .infer import write_submission

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    path = write_submission(run_dir, out)
    log.info("제출 파일 검증 통과 → %s", path)
    return 0


# --- CLI ---

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ai2", description="AI 챌린지 실험 러너")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def with_config(sp):
        sp.add_argument("--config", "-c", required=True, help="configs/experiments 이름 또는 경로")
        sp.add_argument("--set", action="append", metavar="KEY=VALUE", help="설정 덮어쓰기")
        return sp

    with_config(sub.add_parser("config", help="해석된 설정 출력")).set_defaults(func=cmd_config)

    sp = with_config(sub.add_parser("train", help="학습"))
    sp.add_argument("--synthetic", action="store_true",
                    help="대회 데이터 없이 합성 샘플로 forward/backward/optimizer step VRAM만 잰다")
    sp.add_argument("-n", type=int, default=12, help="--synthetic일 때 돌릴 step 수(앞 3은 warmup)")
    sp.set_defaults(func=cmd_train)

    sp = with_config(sub.add_parser("infer", help="추론"))
    sp.add_argument("--split", default="val", choices=["train", "dev", "test", "val", "train_fit"])
    sp.set_defaults(func=cmd_infer)

    sp = with_config(sub.add_parser("smoke", help="로드·토큰·PEFT 왕복 점검"))
    sp.add_argument("--split", default="val", choices=["train", "dev", "test", "val", "train_fit"])
    sp.add_argument("-n", type=int, default=8)
    sp.add_argument("--synthetic", action="store_true",
                    help="대회 데이터 없이 합성 샘플로 점검한다 (암호 해제 전 기본 경로)")
    sp.set_defaults(func=cmd_smoke)

    sp = sub.add_parser("evaluate", help="채점")
    sp.add_argument("--run", required=True)
    sp.add_argument("--baseline-run")
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("compare", help="cross-model 비교표 생성")
    sp.add_argument("--runs-dir")
    sp.add_argument("--out")
    sp.add_argument("--baseline", help="NetGain 기준 실험 이름")
    sp.add_argument("--types", nargs="*")
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("submit", help="제출 CSV 생성 및 검증")
    sp.add_argument("--run", required=True)
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_submit)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return args.func(args)
    except ConfigError as exc:
        log.error("설정 오류: %s", exc)
        return 2
    except KeyboardInterrupt:
        log.warning("사용자 중단")
        return 130


if __name__ == "__main__":
    sys.exit(main())
