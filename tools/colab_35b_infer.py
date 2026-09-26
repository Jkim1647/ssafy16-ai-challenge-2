"""Colab 단독 실행: Qwen3.6-35B-A3B zero-shot 채점(--prompt ours|seokwoong, --split val<N>|test|dev).

저장소 없이 이 파일 하나만 Colab에 올려 실행한다(비공개 저장소 clone 불필요).

    !pip -q install "git+https://github.com/huggingface/transformers.git@ffddd25146e4225b97a6b5e5b66dd9a823bf39d1" accelerate pillow pandas
    !python colab_35b_prompt_infer.py --data /content/drive/MyDrive/<경로>/ssafy-16-2-ai --split val100 --out /content/drive/MyDrive/<경로>/out
    !python colab_35b_prompt_infer.py --data ... --split test --out ...

- 프롬프트: 석웅 1,000건 실행(commands.txt)과 한 글자도 같게. enable_thinking=False.
- 채점: 답변 위치 다음 토큰에서 a/b/c/d 확률만 비교(= a~d 문법 제한 1토큰 greedy와 같은 답). 확률도 저장.
- 정밀도: bf16(기본). --dtype 로 바꿀 수 있다.
- 이미지: 원본 크기(업스케일 없음). visual token 상한은 processor 기본(max_pixels)을 쓴다.
- 이어하기: 출력 predictions.jsonl에 이미 있는 id는 건너뛴다(런타임이 끊겨도 다시 실행하면 이어서).
- val100: train 중 팀 group split val(data_meta/splits/1_group.json) 앞 100개. 정답이 있어 정확도 검증용.
  이 JSON이 없으면 train에서 seed 1로 100개를 뽑는다(그 경우 zero-shot 검증용으로만 의미).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

PROMPT = ("이미지를 보고 객관식 문제를 푸세요. 설명이나 추론은 출력하지 말고 정답인 소문자 a, b, c, d 중 "
          "하나만 출력하세요. 질문: {q} 선택지: a. {a} b. {b} c. {c} d. {d} 정답:")
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
LETTERS = ["a", "b", "c", "d"]
# 우리 기본 프롬프트(src/models/base.py CHOICE_PROMPT와 같은 문구). --prompt ours로 선택.
OURS = "이미지를 보고 질문에 답하시오.\n질문: {question}\n{choices}\n정답 기호 하나만 출력하시오."
# 영어 지시문 + 판독 요령. 질문·보기 원문(한국어)은 그대로 두고 지시만 영어로 준다.
EN = ("Look at the photo carefully. Text in it may be very small, blurry, tilted, rotated or upside down, "
      "and may be Korean, English, numbers or other languages. First find the text or object the question asks about, "
      "read it exactly as written, then compare it with every option.\n"
      "Question: {question}\n{choices}\nAnswer with only the option letter (a, b, c or d).")
PROMPTS = {"ours": OURS, "en": EN}
# read: 2단계. 1단계에서 질문 관련 글자를 원문 그대로 받아 적게 하고(짧은 생성),
# 2단계에서 그 판독을 대화에 남긴 채 보기 기호 확률을 잰다. 오답 대부분이 판독 실패라서 넣었다.
READ_1 = ("Question: {question}\n{choices}\n"
          "Do not answer yet. Find the part of the photo this question is about and transcribe the text there exactly "
          "as written (keep the original language, digits, units and spacing; text may be small, tilted or upside down). "
          "If the answer depends on objects rather than text, briefly describe those objects instead. Output only that.")
READ_2 = "Using the photo and your transcription, choose the correct option. Answer with only the option letter (a, b, c or d)."


TILES_NOTE = "첫 번째 이미지는 전체 사진이고, 나머지 4장은 같은 사진을 2x2로 나눠 확대한 부분(왼쪽 위, 오른쪽 위, 왼쪽 아래, 오른쪽 아래)이다.\n"


def tiles_2x2(img, scale: float, resample, overlap: float = 0.1):
    """원본 픽셀을 자른 2x2 타일(겹침 포함)을 scale배로 키운다 — 작은 글씨 영역의 visual token을 늘리는 용도."""
    w, h = img.size
    tw, th = int(w * (0.5 + overlap / 2)), int(h * (0.5 + overlap / 2))
    boxes = [(0, 0, tw, th), (w - tw, 0, w, th), (0, h - th, tw, h), (w - tw, h - th, w, h)]
    return [img.crop(b).resize((int(tw * scale), int(th * scale)), resample) for b in boxes]


def load_rows(data: Path, split: str, split_json: Path | None):
    import pandas as pd
    if split == "dev":
        # dev는 answer1~5(라벨러 응답)만 있고 정답 열이 없다 — 모델 입력에는 질문·보기만 쓴다.
        df = pd.read_csv(data / "dev.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
        df["answer"] = ""
        return df[["id", "path", "question", "a", "b", "c", "d", "answer"]]
    if split == "test":
        df = pd.read_csv(data / "test.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
        df["answer"] = ""
        return df
    df = pd.read_csv(data / "train.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    n = int(split.replace("val", "") or 100)
    if split_json and split_json.exists():
        val = json.loads(split_json.read_text(encoding="utf-8"))["val_groups"]
        keep = sorted(set(val))[:n]
        return df[df.id.isin(keep)].reset_index(drop=True)
    return df.sample(n, random_state=1).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="train.csv/test.csv와 train/·test/ 이미지가 있는 폴더")
    ap.add_argument("--split", default="val100", help="val<N> 또는 test")
    ap.add_argument("--split-json", default=None, help="data_meta/splits/1_group.json 경로(val용)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prompt", default="ours", choices=["ours", "en", "read", "seokwoong"], help="ours=9B와 같은 기본 프롬프트, en=영어 지시문+판독 요령")
    ap.add_argument("--read-tokens", type=int, default=80, help="--prompt read 1단계 판독 최대 토큰 수")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270], help="이미지를 반시계로 회전(비생성형). 회전 TTA는 결과 파일끼리 확신도로 결합한다")
    ap.add_argument("--model", default=MODEL_ID, help="코드 점검용으로 9B 등으로 바꿀 수 있다")
    ap.add_argument("--shift", type=int, default=0, choices=[0, 1, 2, 3],
                    help="보기 순환 TTA: 화면 j번째 보기 = 원래 (j+shift)%%4번째. 저장 logprobs는 원래 순서로 되돌린다")
    ap.add_argument("--view", default="full", choices=["full", "tiles2x2"],
                    help="tiles2x2: 전체 1장 + 2x2 타일 4장(겹침 10%%, --tile-scale배 확대)을 함께 넣는다")
    ap.add_argument("--tile-scale", type=float, default=2.0,
                    help="타일 확대 배율. 원본이 이미 모델에 그대로 들어가므로 타일은 확대해야 그 영역 visual token이 늘어난다")
    ap.add_argument("--ids-json", default=None, help="이 id 목록(JSON 배열)만 돌린다")
    ap.add_argument("--gold-json", default=None, help="{id: 정답} JSON. dev 등 정답 열이 없는 split의 채점용")
    ap.add_argument("--tag-suffix", default="", help="출력 파일 tag 뒤에 붙일 이름(같은 split의 부분집합 구분용)")
    ap.add_argument("--adapter", default=None, help="PEFT LoRA adapter 폴더. 붙여서 추론한다(tag에 _lora가 붙는다)")
    ap.add_argument("--max-memory-gb", default=None,
                    help="GPU별 상한(GB), 쉼표로 구분. 예: '260,260' → 397B를 2장에 레이어 분할(vLLM 회피, Accelerate device_map). None이면 auto")
    args = ap.parse_args()

    import torch
    from PIL import Image, ImageOps
    from transformers import AutoModelForImageTextToText, AutoProcessor

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = (f"{args.model.split('/')[-1]}_{args.prompt}_{args.split}{args.tag_suffix}"
           + (f"_s{args.shift}" if args.shift else "") + (f"_{args.view}x{args.tile_scale:g}" if args.view != "full" else "")
           + (f"_r{args.rotate}" if args.rotate else "") + ("_lora" if args.adapter else ""))
    pred_path = out / f"{tag}_predictions.jsonl"
    done = set()
    if pred_path.exists():
        done = {json.loads(l)["id"] for l in pred_path.read_text(encoding="utf-8").splitlines() if l.strip()}

    rows = load_rows(data, args.split, Path(args.split_json) if args.split_json else None)
    if args.ids_json:
        keep = set(json.loads(Path(args.ids_json).read_text(encoding="utf-8")))
        rows = rows[rows.id.isin(keep)].reset_index(drop=True)
    if args.gold_json:
        gold = json.loads(Path(args.gold_json).read_text(encoding="utf-8"))
        rows["answer"] = [gold.get(i, a) for i, a in zip(rows.id, rows.answer)]
    if args.limit:
        rows = rows.head(args.limit)
    todo = rows[~rows.id.isin(done)]
    print(f"[{tag}] 전체 {len(rows)} / 완료 {len(done)} / 남음 {len(todo)}", flush=True)

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model, use_fast=True)
    tok = processor.tokenizer
    mm = None
    if args.max_memory_gb:
        mm = {i: f"{g}GiB" for i, g in enumerate(args.max_memory_gb.split(","))}
    load_kw = dict(device_map="auto", attn_implementation="sdpa")
    if mm:
        load_kw["max_memory"] = mm
    # GPTQ 등 양자화 모델은 config의 quantization_config로 자동 로드된다(gptqmodel/auto-gptq 필요).
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), **load_kw)
    input_dev = "cuda:0"
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    letter_ids = [tok.encode(x, add_special_tokens=False) for x in LETTERS]
    assert all(len(i) == 1 for i in letter_ids), f"a~d가 단일 토큰이 아니다: {letter_ids}"
    letter_ids = [i[0] for i in letter_ids]
    load_sec = round(time.time() - t0, 1)
    print(f"[{tag}] 모델 로드 {load_sec}s, GPU {torch.cuda.get_device_name(0)}, letter ids {letter_ids}", flush=True)

    meta = {"model": args.model, "dtype": args.dtype, "prompt": PROMPTS.get(args.prompt, READ_1 + " || " + READ_2 if args.prompt == "read" else PROMPT), "read_tokens": args.read_tokens, "rotate": args.rotate, "enable_thinking": False,
            "scoring": "next-token logprob over a-d at answer position", "split": args.split, "shift": args.shift, "view": args.view, "tile_scale": args.tile_scale, "adapter": args.adapter,
            "n": len(rows), "id_set_sha256": hashlib.sha256("\n".join(sorted(rows.id)).encode()).hexdigest(),
            "gpu": torch.cuda.get_device_name(0), "load_sec": load_sec,
            "torch": torch.__version__}
    (out / f"{tag}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    n_ok = n_seen = 0
    t1 = time.time()
    with pred_path.open("a", encoding="utf-8") as fh, torch.no_grad():
        for i, r in enumerate(todo.itertuples(), 1):
            img = ImageOps.exif_transpose(Image.open(data / r.path)).convert("RGB")
            if args.rotate:
                img = img.rotate(args.rotate, expand=True)
            shown = [getattr(r, LETTERS[(j + args.shift) % 4]) for j in range(4)]
            choices = "\n".join(f"{x}. {c}" for x, c in zip(LETTERS, shown))
            if args.prompt == "read":
                text = READ_1.format(question=r.question, choices=choices)
            elif args.prompt in PROMPTS:
                text = PROMPTS[args.prompt].format(question=r.question, choices=choices)
            else:
                text = PROMPT.format(q=r.question, a=shown[0], b=shown[1], c=shown[2], d=shown[3])
            views = [img]
            if args.view == "tiles2x2":
                views += tiles_2x2(img, args.tile_scale, Image.BICUBIC)
                text = TILES_NOTE + text
            messages = [{"role": "user", "content": [{"type": "image", "image": v} for v in views]
                         + [{"type": "text", "text": text}]}]
            transcript = None
            if args.prompt == "read":
                p1 = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                in1 = processor(text=[p1], images=views, return_tensors="pt").to(input_dev)
                gen = model.generate(**in1, max_new_tokens=args.read_tokens, do_sample=False)
                transcript = tok.decode(gen[0, in1["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                messages += [{"role": "assistant", "content": [{"type": "text", "text": transcript}]},
                             {"role": "user", "content": [{"type": "text", "text": READ_2}]}]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                   enable_thinking=False)
            inputs = processor(text=[prompt], images=views, return_tensors="pt").to(input_dev)
            logits = model(**inputs).logits[0, -1].float()
            lp = torch.log_softmax(logits, dim=-1)
            shown_scores = [lp[t].item() for t in letter_ids]
            scores = [0.0] * 4
            for j, s in enumerate(shown_scores):
                scores[(j + args.shift) % 4] = s
            m = max(scores)
            probs = [math.exp(s - m) for s in scores]
            z = sum(probs)
            best = max(range(4), key=lambda k: scores[k])
            row = {"id": r.id, "pred": LETTERS[best], "gold": r.answer or None,
                   "confidence": round(probs[best] / z, 6), "logprobs": [round(s, 6) for s in scores],
                   "visual_tokens": int(inputs["image_grid_thw"].prod(-1).sum().item()) // 4
                   if "image_grid_thw" in inputs else None}
            if transcript is not None:
                row["transcript"] = transcript
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if r.answer:
                n_seen += 1
                n_ok += int(LETTERS[best] == r.answer.strip().lower())
            if i % 50 == 0 or i == len(todo):
                rate = (time.time() - t1) / i
                acc = f" acc {n_ok}/{n_seen}={n_ok / n_seen:.4f}" if n_seen else ""
                print(f"[{tag}] {i}/{len(todo)} {rate:.2f}s/건 남은 {rate * (len(todo) - i) / 60:.0f}분{acc}", flush=True)

    if args.split == "test":
        import pandas as pd
        preds = {json.loads(l)["id"]: json.loads(l)["pred"] for l in pred_path.read_text(encoding="utf-8").splitlines() if l.strip()}
        sample = pd.read_csv(data / "sample_submission.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
        missing = [x for x in sample.id if x not in preds]
        if missing:
            print(f"[{tag}] 제출 파일 보류: 예측 누락 {len(missing)}건 (다시 실행하면 이어서 채운다)")
            return 1
        sample["answer"] = sample.id.map(preds)
        assert sample.answer.isin(LETTERS).all() and not sample.id.duplicated().any()
        sub = out / f"{tag}_submission.csv"
        # Kaggle 제출: BOM 금지(utf-8). 채점기가 첫 id를 BOM과 붙여 읽을 수 있다.
        sample.to_csv(sub, index=False, encoding="utf-8")
        print(f"[{tag}] 제출 파일 {sub} ({len(sample)}행, 분포 {sample.answer.value_counts().to_dict()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
