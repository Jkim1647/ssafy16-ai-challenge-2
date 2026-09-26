"""Colab 단독 실행: Qwen VLM LoRA 학습 → val 채점 → test 추론까지 한 번에.

colab_35b_infer.py와 같은 입력·채점 방식으로 학습한다. 추론 입력(add_generation_prompt=True) 그대로
마지막 위치의 다음 토큰 분포에서 정답 글자(a~d)에 cross-entropy를 건다 — loss masking과 채점 위치가
구조적으로 일치한다.

    !pip -q install "git+https://github.com/huggingface/transformers.git@ffddd25146e4225b97a6b5e5b66dd9a823bf39d1" accelerate peft pillow pandas
    !python colab_lora_train.py --model Qwen/Qwen3.5-9B --data /content/data --val-ids val667_ids.json --out /content/drive/MyDrive/ai2_35b_out/lora

- 학습: train - val(val_ids) 전체, 1 epoch 기본. --extra-csv로 dev 재라벨(id,path,question,a,b,c,d,answer) 추가 가능.
- --aug-shift: 샘플마다 보기를 무작위로 순환(정답 글자도 같이 이동). 위치 편향 완화용.
- 산출물: <out>/<tag>/val_predictions.jsonl, test_predictions.jsonl(원래 보기 순서 logprobs), adapter/, meta.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

LETTERS = ["a", "b", "c", "d"]
OURS = "이미지를 보고 질문에 답하시오.\n질문: {question}\n{choices}\n정답 기호 하나만 출력하시오."


def build_inputs(processor, img, question, shown, view="full", tile_scale=2.0):
    text = OURS.format(question=question, choices="\n".join(f"{x}. {c}" for x, c in zip(LETTERS, shown)))
    views = [img]
    if view == "tiles2x2":
        # 추론(colab_35b_infer.py --view tiles2x2)과 같은 타일·안내 문구를 쓴다. 두 파일을 같은 폴더에 둔다.
        from PIL import Image
        from colab_35b_infer import TILES_NOTE, tiles_2x2
        views += tiles_2x2(img, tile_scale, Image.BICUBIC)
        text = TILES_NOTE + text
    messages = [{"role": "user", "content": [{"type": "image", "image": v} for v in views] + [{"type": "text", "text": text}]}]
    try:
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
    except TypeError:
        # enable_thinking 은 Qwen chat template 전용 인자다. Gemma 등 다른 계열은
        # 이 인자를 모르면 TypeError 를 낸다. 무인 실행 중에 여기서 죽지 않게 한다.
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[prompt], images=views, return_tensors="pt")


def aug_image(img, rng):
    """학습 전용 비생성 증강. **평가 경로에서는 절대 호출하지 않는다.**

    정답이 이미지 속 글자 자체이므로 허용 범위가 좁다.
      - 좌우반전 금지: 글자가 뒤집히면 라벨과 어긋난다.
      - crop 금지: 가장자리 텍스트를 잘라내면 답을 못 읽는 이미지가 된다.
      - 생성형(초해상·diffusion) 금지: 없던 획을 만들어 "선명하지만 틀린 글자"가 된다.
    남는 것은 원본 픽셀을 보존하는 회전(expand=True)과 밝기·대비 보정뿐이다.
    """
    from PIL import Image, ImageEnhance
    if rng.random() < 0.5:  # 촬영 기울어짐 모사. expand로 잘림을 막고 흰 배경으로 채운다.
        img = img.rotate(rng.uniform(-4, 4), resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
    if rng.random() < 0.5:  # 조명 편차
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 1.15))
    if rng.random() < 0.5:  # 저조도·역광 모사
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.85, 1.15))
    return img


KD_MISSING = -1e4  # colab_vllm_infer.py가 상위 20 logprobs 밖 글자에 쓰는 표시값(실제 확률 아님)


def kd_target(teacher_lp_orig, shift: int, temp: float):
    """teacher의 원래 a~d 로그확률 → 학생 화면 순서(shift 적용)의 온도 T 확률분포. 글자가 빠졌으면 None.

    학생 화면 위치 j에는 원래 보기 (j+shift)%4가 보인다(아래 학습 루프의 shown과 같은 규칙).
    4글자 사이 로그확률 차이는 logit 차이와 같으므로 /T 후 softmax하면 온도 T 분포가 정확히 재현된다.
    """
    if any(x <= KD_MISSING for x in teacher_lp_orig):
        return None
    shown = [teacher_lp_orig[(j + shift) % 4] / temp for j in range(4)]
    m = max(shown)
    e = [math.exp(x - m) for x in shown]
    z = sum(e)
    return [x / z for x in e]


def versions() -> dict:
    """재현용 패키지 버전 기록."""
    import importlib.metadata as md
    out = {}
    for pkg in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "pillow"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--data", required=True)
    ap.add_argument("--val-ids", required=True, help="val667_ids.json (val_groups 키)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra-csv", default=None, help="추가 학습 행(dev 재라벨 등). path는 --data 기준")
    ap.add_argument("--train-csv", default=None,
                    help="프리빌트 학습셋(train_A/AB/ABC.csv). 주면 train.csv-val 대신 이 파일 전체를 fit으로 쓴다(누수 사전검증됨). source 컬럼 유지.")
    ap.add_argument("--source-weight", default=None,
                    help="source별 오버샘플 배수. 예: 'train_orig:1,dev_pseudo:1,codex_regen:1'. --train-csv에 source 컬럼이 있을 때만.")
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--dora", action="store_true", help="LoRA 대신 DoRA(use_dora=True). 10번 설정 비교용")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--aug-shift", action="store_true")
    ap.add_argument("--aug-image", action="store_true",
                    help="학습 이미지에만 비생성 증강(회전 ±4도, 밝기·대비 ±15%). 평가 경로에는 적용하지 않는다")
    ap.add_argument("--eval-every", type=int, default=0, help="optimizer step 간격으로 val 채점(0이면 끝에서만)")
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-val", type=int, default=None, help="smoke용. 본 실험에서는 쓰지 않는다")
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--eval-adapter", default=None,
                    help="학습 없이 저장된 adapter를 새 프로세스에서 재로드해 val만 채점(저장→재시작→재로드 검증)")
    ap.add_argument("--hard-gold", default=None,
                    help="{dev id: 정답} JSON(어려운 문제 평가셋). 학습 후 dev에서 이 문항만 추가 채점한다")
    ap.add_argument("--view", default="full", choices=["full", "tiles2x2"],
                    help="tiles2x2: 전체 1장 + 2x2 타일 4장(2배 확대)으로 학습·채점. visual token 약 5배")
    ap.add_argument("--tile-scale", type=float, default=2.0)
    ap.add_argument("--revision", default=None, help="HF 모델 commit 고정(재현용)")
    ap.add_argument("--save-steps", type=int, default=0, help="이 optimizer step마다 adapter 중간 저장(0이면 끝에서만)")
    ap.add_argument("--load-4bit", action="store_true", help="NF4 + double quant, BF16 compute (QLoRA)")
    ap.add_argument("--full-ft", action="store_true", help="LoRA 대신 언어모델 전체 파인튜닝(비전 동결). B300 1장 가능(스모크 확인)")
    ap.add_argument("--optimizer", default="adamw", choices=["adamw","adafactor"], help="FFT는 adafactor 권장(B300 8bit optim 불가)")
    ap.add_argument("--save-full", action="store_true", help="FFT 전체 가중치 저장(~70GB, 느림). 기본은 저장 안 함")
    ap.add_argument("--exclude-csv", default=None,
                    help="학습에서 뺄 id 목록 CSV(id 열). 예: data_meta/train_duplicate_exclusions.csv(test·val·dev와 같은 이미지 45건)")
    ap.add_argument("--kd-jsonl", default=None,
                    help="teacher 예측 jsonl(id, logprobs[a,b,c,d] 원래 순서). 주면 CE + KD(KL) 학습")
    ap.add_argument("--kd-alpha", type=float, default=1.0,
                    help="loss = alpha*CE + (1-alpha)*T^2*KL. 1.0이면 CE만(기존과 동일)")
    ap.add_argument("--kd-temp", type=float, default=2.0)
    ap.add_argument("--reload-score-test", action="store_true",
                    help="--eval-adapter와 함께: 재로드 val 채점 뒤 test.csv도 채점해 test_predictions.jsonl 저장")
    ap.add_argument("--eval-shifts", default="",
                    help="--eval-adapter와 함께: 보기를 k칸 순환해 다시 채점(예 '1,2,3'). 로그확률은 원래 a~d 순서로 되돌려 "
                         "hard_predictions_shift{k}.jsonl / test_predictions_shift{k}.jsonl 로 저장(보기 회전 TTA)")
    ap.add_argument("--shift-test-ids", default=None,
                    help="--eval-shifts 의 test 채점을 이 id 목록(JSON 리스트)으로 제한")
    ap.add_argument("--shifts-only", action="store_true",
                    help="--eval-shifts 와 함께: 회전 없는 H408·test 채점은 건너뛴다(이미 있는 예측 재사용)")
    ap.add_argument("--kd-filter", default="all", choices=["all", "agree"],
                    help="agree: teacher pred == gold인 문항에만 KD 항을 건다(CE는 전 문항)")
    args = ap.parse_args()

    import pandas as pd
    import torch
    from peft import LoraConfig, get_peft_model
    from PIL import Image, ImageOps
    from transformers import AutoModelForImageTextToText, AutoProcessor

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = Path(args.data)
    _kind = "fft" if args.full_ft else f"lora_r{args.r}"
    tag = args.tag or (f"{args.model.split('/')[-1]}_{_kind}_lr{args.lr:g}_ep{args.epochs:g}"
                       + ("_aug" if args.aug_shift else "") + ("_augimg" if args.aug_image else "")
                       + ("_tiles" if args.view == "tiles2x2" else ""))
    out = Path(args.out) / tag
    out.mkdir(parents=True, exist_ok=True)

    read = lambda p: pd.read_csv(p, encoding="utf-8-sig", dtype=str, keep_default_na=False)  # noqa: E731
    _stem = lambda s: Path(str(s)).stem  # canonical image id (train_0001, dev_0545). 문자열 id가 아닌 실제 이미지 기준.  # noqa: E731
    train = read(data / "train.csv")
    val_ids = set(json.loads(Path(args.val_ids).read_text(encoding="utf-8"))["val_groups"])
    val = train[train.id.isin(val_ids)].reset_index(drop=True)
    if args.train_csv:
        # 프리빌트 학습셋 전체를 fit으로. (train_A/AB/ABC.csv은 누수 사전검증됨)
        fit = read(args.train_csv)
        assert {"id", "path", "question", "a", "b", "c", "d", "answer"} <= set(fit.columns), "train-csv 컬럼 부족"
        fit = fit[fit.answer.str.lower().isin(LETTERS)].reset_index(drop=True)
        if args.source_weight and "source" in fit.columns:
            wmap = {kv.split(":")[0]: int(kv.split(":")[1]) for kv in args.source_weight.split(",")}
            fit = pd.concat([fit] + [fit[fit.source == s] for s, w in wmap.items() for _ in range(max(0, w - 1))],
                            ignore_index=True)
    else:
        fit = train[~train.id.isin(val_ids)]
        if args.extra_csv:
            extra = read(args.extra_csv)[["id", "path", "question", "a", "b", "c", "d", "answer"]]
            extra = extra[extra.answer.str.lower().isin(LETTERS)]
            fit = pd.concat([fit, extra], ignore_index=True)
    if args.exclude_csv:
        ex = set(read(args.exclude_csv).id)
        n0 = len(fit)
        fit = fit[~fit.id.isin(ex)]
        print(f"[{tag}] exclude-csv로 {n0 - len(fit)}건 제외", flush=True)
    fit = fit.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    if args.limit_train:
        fit = fit.head(args.limit_train)
    if args.limit_val:
        val = val.head(args.limit_val)
    # ---- 누수 assert: 문자열 id + canonical(파일 stem) 두 축으로 검사 ----
    fit_cid = set(fit.path.map(_stem))
    val_cid = set(val.path.map(_stem)) | {_stem(x) for x in val_ids}
    assert not (set(fit.id) & val_ids), "val 문항이 학습셋에 섞였다(id)"
    assert not (fit_cid & val_cid), "val 이미지가 학습셋에 섞였다(canonical)"
    if args.hard_gold:
        hg_raw = json.loads(Path(args.hard_gold).read_text(encoding="utf-8"))
        hard_ids = set(hg_raw)
        hard_cid = {_stem(x) for x in hard_ids}
        assert not (set(fit.id) & hard_ids), "어려운 평가셋 문항이 학습셋에 섞였다(id)"
        assert not (fit_cid & hard_cid), "어려운 평가셋 이미지가 학습셋에 섞였다(canonical)"
    teacher = {}
    if args.kd_jsonl:
        for line in Path(args.kd_jsonl).read_text(encoding="utf-8").splitlines():
            if line.strip():
                t = json.loads(line)
                teacher[t["id"]] = t
        cov = sum(1 for i in fit.id if i in teacher)
        print(f"[{tag}] KD teacher {len(teacher)}행, fit 중 teacher 있음 {cov}/{len(fit)}, "
              f"alpha {args.kd_alpha} T {args.kd_temp} filter {args.kd_filter}", flush=True)
    print(f"[{tag}] fit {len(fit)} / val {len(val)}", flush=True)
    if "source" in fit.columns:
        print("[source]", fit.source.value_counts().to_dict(), flush=True)

    processor = AutoProcessor.from_pretrained(args.model, use_fast=True, revision=args.revision)
    tok = processor.tokenizer
    letter_ids = [tok.encode(x, add_special_tokens=False) for x in LETTERS]
    assert all(len(i) == 1 for i in letter_ids), letter_ids
    letter_ids = [i[0] for i in letter_ids]
    qconf = None
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        qconf = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16,
                                   llm_int8_skip_modules=["visual", "lm_head"])
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa", quantization_config=qconf,
        revision=args.revision)
    if args.load_4bit and not args.eval_adapter:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if args.eval_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.eval_adapter)
        n_lora = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A"))
        print(f"[{tag}] adapter 재로드 {args.eval_adapter}, LoRA 모듈 {n_lora}개", flush=True)
        model.eval()

        def score_reload(df, path, shift=0):
            """재로드한 adapter 로 df 를 채점해 jsonl 로 쓰고 정답 수를 돌려준다.
            answer 가 비어 있으면 gold 를 None 으로 남기고 정답 수는 세지 않는다(test).
            shift=k 면 화면 위치 j 에 원래 보기 (j+k)%4 를 보여 주고, 점수는 원래 a~d 순서로 되돌려 쓴다."""
            ok = 0
            with torch.no_grad(), path.open("w", encoding="utf-8") as fh:
                for r in df.itertuples():
                    shown = [getattr(r, LETTERS[(j + shift) % 4]) for j in range(4)]
                    img = ImageOps.exif_transpose(Image.open(data / r.path)).convert("RGB")
                    inputs = build_inputs(processor, img, r.question, shown, args.view, args.tile_scale).to(model.device)
                    lp = torch.log_softmax(model(**inputs).logits[0, -1].float(), dim=-1)
                    sc = [0.0] * 4
                    for j, t in enumerate(letter_ids):
                        sc[(j + shift) % 4] = lp[t].item()
                    best = max(range(4), key=lambda k: sc[k])
                    m = max(sc)
                    e = [math.exp(x - m) for x in sc]
                    g = (getattr(r, "answer", "") or "").strip().lower() or None
                    ok += int(LETTERS[best] == g) if g else 0
                    fh.write(json.dumps({"id": r.id, "pred": LETTERS[best], "gold": g,
                                         "confidence": round(e[best] / sum(e), 6),
                                         "logprobs": [round(x, 6) for x in sc]}, ensure_ascii=False) + "\n")
            return ok

        ok = score_reload(val, out / "val_reload.jsonl")
        print(f"[{tag}] 재로드 val acc {ok / len(val):.4f} ({len(val)}문항)", flush=True)
        if args.hard_gold:
            # 학습 경로에만 있던 어려운 평가셋 채점을 재로드 경로에서도 낸다.
            # R3·Gemma 는 train 만 학습했으므로 dev 기반 H408 은 미지의 셋이다. val667 하나로는
            # 이중 게이트가 성립하지 않는다 — val667 은 1문항 0.150%p 라 ±3문항이 잡음이다.
            hg = json.loads(Path(args.hard_gold).read_text(encoding="utf-8"))
            hard = read(data / "dev.csv")
            hard = hard[hard.id.isin(hg)].reset_index(drop=True)
            hard["answer"] = hard.id.map(hg)
            assert len(hard) == len(hg), f"dev.csv 에 없는 평가 문항 {len(hg) - len(hard)}건"
            hok = 0 if args.shifts_only else score_reload(hard, out / "hard_predictions.jsonl")
            print(f"[{tag}] 재로드 어려운 평가셋 acc {hok / len(hard):.4f} ({len(hard)}문항)", flush=True)
        shifts = [int(x) for x in args.eval_shifts.split(",") if x.strip()]
        for k in shifts:
            if args.hard_gold:
                sok = score_reload(hard, out / f"hard_predictions_shift{k}.jsonl", shift=k)
                print(f"[{tag}] shift{k} 어려운 평가셋 acc {sok / len(hard):.4f} ({len(hard)}문항)", flush=True)
            if args.reload_score_test:
                stest = read(data / "test.csv")
                if args.shift_test_ids:
                    keep = set(json.loads(Path(args.shift_test_ids).read_text(encoding="utf-8")))
                    stest = stest[stest.id.isin(keep)].reset_index(drop=True)
                score_reload(stest, out / f"test_predictions_shift{k}.jsonl", shift=k)
                print(f"[{tag}] shift{k} test 채점 완료 {len(stest)}문항", flush=True)
        if args.reload_score_test and not args.shifts_only:
            # 저장된 adapter로 test만 채점(제출 후보용). 출력 형식은 학습 후 test_predictions.jsonl과 같다.
            test = read(data / "test.csv")
            score_reload(test, out / "test_predictions.jsonl")
            print(f"[{tag}] 재로드 test 채점 완료 {len(test)}문항", flush=True)
        # 재로드 경로도 meta 를 남긴다. 이게 없으면 "어떤 adapter 를 어떤 설정으로 채점했는지"가
        # 로그에만 남아 재현이 안 된다(산출물 규칙: 모든 run 에 manifest).
        rmeta = {"model": args.model, "tag": tag, "mode": "eval_adapter",
                 "eval_adapter": args.eval_adapter, "args": vars(args),
                 "view": args.view, "tile_scale": args.tile_scale,
                 "letter_ids": letter_ids, "versions": versions(),
                 "reload_val_acc": round(ok / max(len(val), 1), 6), "reload_val_n": len(val),
                 "scored_test": bool(args.reload_score_test)}
        if args.hard_gold:
            rmeta["hard_gold"] = args.hard_gold
            rmeta["reload_hard_acc"] = round(hok / max(len(hard), 1), 6)
            rmeta["reload_hard_n"] = len(hard)
        (out / "meta.json").write_text(json.dumps(rmeta, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if args.full_ft:
        # FFT: 비전 동결, 언어 모델 전체 파라미터 학습(LoRA 아님). B300 1장 138GB 피크(스모크 확인).
        for name, p in model.named_parameters():
            p.requires_grad = not ("visual" in name or "vision" in name)
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[{tag}] FFT 전체학습 {n_tr/1e9:.2f}B, letter ids {letter_ids}", flush=True)
    else:
        # 비전 타워는 동결하고 언어 모델 projection에만 LoRA를 건다.
        target = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
        model = get_peft_model(model, LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=0.05,
                                                 target_modules=target, task_type="CAUSAL_LM", use_dora=args.dora))
        n_lora = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A"))
        assert n_lora > 0, "LoRA가 걸린 모듈이 없다"
        model.print_trainable_parameters()
        print(f"[{tag}] LoRA 모듈 {n_lora}개, letter ids {letter_ids}", flush=True)

    load_img = lambda p: ImageOps.exif_transpose(Image.open(data / p)).convert("RGB")  # noqa: E731

    def score(df, path, shift=0):
        model.eval()
        rows, ok = [], 0
        with torch.no_grad(), path.open("w", encoding="utf-8") as fh:
            for r in df.itertuples():
                shown = [getattr(r, LETTERS[(j + shift) % 4]) for j in range(4)]
                inputs = build_inputs(processor, load_img(r.path), r.question, shown, args.view, args.tile_scale).to(model.device)
                lp = torch.log_softmax(model(**inputs).logits[0, -1].float(), dim=-1)
                s = [0.0] * 4
                for j, t in enumerate(letter_ids):
                    s[(j + shift) % 4] = lp[t].item()
                m = max(s)
                p = [math.exp(x - m) for x in s]
                best = max(range(4), key=lambda k: s[k])
                gold = (r.answer or "").strip().lower() or None
                ok += int(gold == LETTERS[best])
                row = {"id": r.id, "pred": LETTERS[best], "gold": gold, "confidence": round(p[best] / sum(p), 6),
                       "logprobs": [round(x, 6) for x in s]}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
        model.train()
        return ok / max(1, sum(1 for x in rows if x["gold"])), rows

    steps_total = math.ceil(len(fit) * args.epochs / args.grad_accum)
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "adafactor":
        # B300(sm_103)에서 bitsandbytes 8bit optim 커널 컴파일 불가 → FFT는 Adafactor(저메모리, 순수 torch).
        from transformers.optimization import Adafactor
        opt = Adafactor(params, lr=args.lr, scale_parameter=False, relative_step=False, warmup_init=False, weight_decay=0.0)
    else:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    warm = max(1, int(0.03 * steps_total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps_total)))))

    meta = {"model": args.model, "tag": tag, "args": vars(args), "fit": len(fit), "val": len(val),
            "steps": steps_total, "gpu": torch.cuda.get_device_name(0), "history": [], "versions": versions()}
    kd_stat = {"used": 0, "skipped": 0, "fallback": {}}
    meta["kd"] = {"jsonl": args.kd_jsonl, "alpha": args.kd_alpha, "temp": args.kd_temp, "filter": args.kd_filter,
                  "stat": kd_stat, "loss": "alpha*CE(full vocab) + (1-alpha)*T^2*KL(teacher_T || student_T over a-d)"}
    model.train()
    t0 = time.time()
    n_samples = math.ceil(len(fit) * args.epochs)
    step = run_loss = 0
    for i in range(n_samples):
        r = fit.iloc[i % len(fit)]
        k = random.randrange(4) if args.aug_shift else 0
        shown = [r[LETTERS[(j + k) % 4]] for j in range(4)]
        gold_shown = (LETTERS.index(r["answer"].strip().lower()) - k) % 4
        _img = load_img(r["path"])
        if args.aug_image:
            _img = aug_image(_img, random)  # 학습 경로 전용. score()는 원본 그대로 쓴다.
        inputs = build_inputs(processor, _img, r["question"], shown, args.view, args.tile_scale).to(model.device)
        logits = model(**inputs).logits[0, -1].float()
        loss = torch.nn.functional.cross_entropy(logits[None], torch.tensor([letter_ids[gold_shown]], device=logits.device))
        if teacher and args.kd_alpha < 1.0:
            t = teacher.get(r["id"])
            tgt = kd_target(t["logprobs"], k, args.kd_temp) if t else None
            why = "no_teacher" if t is None else ("missing_letter_logprob" if tgt is None else None)
            if tgt is not None and args.kd_filter == "agree" and t["pred"] != r["answer"].strip().lower():
                tgt, why = None, "teacher_disagrees_gold"
            if tgt is None:
                # 이 샘플은 CE만 쓴다. 에폭이 1을 넘으면 같은 id가 반복 기록될 수 있다.
                kd_stat["skipped"] += 1
                kd_stat["fallback"].setdefault(why, []).append(r["id"])
            else:
                kd_stat["used"] += 1
                # 학생도 a~d 4글자 logit만으로 재정규화한 분포(온도 T)와 비교한다.
                s_logp = torch.log_softmax(logits[letter_ids] / args.kd_temp, dim=-1)
                p_t = torch.tensor(tgt, device=logits.device)
                kl = torch.sum(p_t * (torch.log(p_t.clamp_min(1e-12)) - s_logp))
                loss = args.kd_alpha * loss + (1.0 - args.kd_alpha) * (args.kd_temp ** 2) * kl
        (loss / args.grad_accum).backward()
        run_loss += loss.item()
        if (i + 1) % args.grad_accum == 0 or i + 1 == n_samples:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == steps_total:
                el = time.time() - t0
                print(f"[{tag}] step {step}/{steps_total} loss {run_loss / args.grad_accum / 10:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {el / (i + 1):.2f}s/샘플 남은 {el / (i + 1) * (n_samples - i - 1) / 60:.0f}분",
                      flush=True)
                meta["history"].append({"step": step, "loss": run_loss / args.grad_accum / 10})
                run_loss = 0
            if args.save_steps and step % args.save_steps == 0 and step < steps_total:
                model.save_pretrained(out / f"adapter_step{step}")
                print(f"[{tag}] step {step} adapter 저장", flush=True)
            if args.eval_every and step % args.eval_every == 0 and step < steps_total:
                acc, _ = score(val, out / f"val_step{step}.jsonl")
                print(f"[{tag}] step {step} val acc {acc:.4f}", flush=True)
                meta["history"].append({"step": step, "val_acc": acc})

    meta["train_sec"] = round(time.time() - t0, 1)
    if not args.full_ft:
        model.save_pretrained(out / "adapter")
    elif args.save_full:
        model.save_pretrained(out / "full_model", safe_serialization=True)
    acc, _ = score(val, out / "val_predictions.jsonl")
    meta["val_acc"] = acc
    print(f"[{tag}] 최종 val acc {acc:.4f} ({len(val)}문항)", flush=True)
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.hard_gold:
        hg = json.loads(Path(args.hard_gold).read_text(encoding="utf-8"))
        hard = read(data / "dev.csv")
        hard = hard[hard.id.isin(hg)].reset_index(drop=True)
        hard["answer"] = hard.id.map(hg)
        hacc, _ = score(hard, out / "hard_predictions.jsonl")
        meta["hard_acc"] = hacc
        print(f"[{tag}] 어려운 평가셋 acc {hacc:.4f} ({len(hard)}문항)", flush=True)
        (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.skip_test:
        test = read(data / "test.csv")
        test["answer"] = ""
        t1 = time.time()
        score(test, out / "test_predictions.jsonl")
        meta["test_sec"] = round(time.time() - t1, 1)
        (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{tag}] test 추론 완료 {meta['test_sec']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
