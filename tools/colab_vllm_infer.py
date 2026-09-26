"""vLLM in-process 추론(HTTP 서버 없음): 대형·양자화 모델(122B GPTQ-Int4 등)용.

colab_35b_infer.py와 같은 프롬프트·채점을 쓴다. 답 위치 다음 토큰을 a~d로 제한(allowed_token_ids)하고
그 4개 로그확률을 저장한다. 출력 jsonl 형식도 같아서 tools/ensemble_submit.py로 바로 합칠 수 있다.

    !pip -q install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
    !python colab_vllm_infer.py --model Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 --quantization moe_wna16 \
        --data /content/data --split val667 --split-json val667_ids.json --out /content/drive/MyDrive/ai2_35b_out

대회 규칙: API 호출 추론은 금지다. 이 스크립트는 가중치를 같은 프로세스에서 직접 돌린다(vllm.LLM).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

from colab_35b_infer import LETTERS, OURS, READ_1, READ_2, TILES_NOTE, load_rows, tiles_2x2

# 2단계 답 선택 지시 v2 (2026-09-22 공통 오답 31건 진단에서 "판독은 맞고 선택이 틀림" 14건 대응).
# 필드 혼동(발행일↔인쇄일, 라스트오더↔영업종료), 부정문, 보기의 덧붙은 말(상위집합 함정)을 점검하게 한다.
READ_2_V2 = ("Using the photo and your transcription, choose the option that answers exactly what the question asks. "
             "Before choosing, check: (1) which specific item or field the question targets (for example which date, time, "
             "name, price, floor or row), not a nearby similar one; (2) whether the question is negative (not / 아닌 / 없는 / 않은); "
             "(3) prefer the option whose wording matches the photo exactly, without extra or missing words. "
             "Answer with only the option letter (a, b, c or d).")

# 2단계 답 선택 지시 v3 — 라벨 타이브레이크 (2026-09-23).
#
# 근거 둘이 같은 곳을 가리킨다.
#   - 우리 train 6,047 진단: 35B·397B가 같이 틀린 144건 중 86%가 **판독문은 맞는데** 선택이 달랐다.
#     예) 사진에 `소화기`가 적혀 있고 모델이 그대로 읽었는데 합의 정답은 `소화전`.
#   - 캐글 디스커션(다른 참가자 공유): 정답 후보가 여럿일 때 라벨이 ①더 크게 적힌 것 ②위쪽·중앙에 있는 것
#     ③다국어면 로마자 표기를 고르는 경향. 예) 작게 적힌 제조사명이 아니라 크게 박힌 제품명,
#     한자 표기가 아니라 로마자 표기.
#
# **가설이지 규칙이 아니다.** 표본 6건에 선택 편향이 있고 반례도 보고됐다(train_6043: 작은 쪽이 정답).
# 그래서 기본 판단을 뒤집으라고 하지 않고, **진짜로 갈릴 때만** 적용하는 동점 처리로 제한한다.
# 채택은 train 홀드아웃 6,047 + H408 두 게이트를 모두 통과할 때만.
READ_2_V3 = ("Using the photo and your transcription, choose the option that answers exactly what the question asks. "
             "Before choosing, check: (1) which specific item or field the question targets (for example which date, time, "
             "name, price, floor or row), not a nearby similar one; (2) whether the question is negative (not / 아닌 / 없는 / 않은); "
             "(3) prefer the option whose wording matches the photo exactly, without extra or missing words. "
             "If two or more options still seem equally correct after those checks, and the photo genuinely shows several "
             "candidates, break the tie by preferring the candidate that is displayed most prominently: first the one "
             "printed largest, then the one placed higher or more central in the photo, and for the same name written in "
             "several scripts the romanized form. Apply this only to break a genuine tie — never to override an option "
             "that the question and photo already single out. "
             "Answer with only the option letter (a, b, c or d).")

THINK_NOTE = ("\nBefore answering, think step by step: (1) state exactly which entity and which field the question asks "
              "about (e.g. building name vs church name, complex name vs unit number, current station vs destination, "
              "number of floors vs which floor); (2) find that exact item in the photo/transcription; (3) check each option "
              "against it, rejecting options that match a different item or only partly match. Then answer with only the letter.")


def build_fewshot(rows, k, seed, exclude):
    """few-shot 예시 블록을 만든다. **텍스트 전용이다.**

    예시마다 이미지를 붙이면 타일 때문에 예시 1개당 이미지 5장이 들어가 프롬프트가 폭발한다.
    그런데 우리 오답의 성격상 이미지가 없어도 될 수 있다 — 35B·397B 가 둘 다 틀린 144건의
    86%는 판독문이 맞고 **선택이** 어긋난 건이다. 즉 배워야 할 것은 "어떻게 보느냐"가 아니라
    "비슷한 보기 중 사람들이 무엇을 고르느냐"이고, 그건 텍스트만으로 보여줄 수 있다.

    예시는 평가 문항과 반드시 분리한다(`exclude` 에 평가 id 를 넘긴다).
    효과가 확인되면 그때 이미지 포함 버전을 비용 들여 시험한다.
    """
    import random as _random
    letters = list(LETTERS)
    pool = [r for r in rows
            if r["id"] not in exclude and str(r.get("answer", "")).strip().lower() in letters]
    _random.Random(seed).shuffle(pool)
    blocks = []
    for r in pool[:k]:
        choices = "\n".join("{}. {}".format(x, r[x]) for x in letters)
        blocks.append("Question: {}\n{}\nAnswer: {}".format(
            r["question"], choices, str(r["answer"]).strip().lower()))
    if not blocks:
        return ""
    return ("Here are solved examples from the same dataset. "
            "Follow the same answer conventions.\n\n" + "\n\n".join(blocks) + "\n\n")


# 1단계 판독 지시 v2 — recall 강화 (2026-09-23 H408 진단).
# 근거: 판독문 길이가 COMPOSITE에서 평균 17.4자로 전 유형 중 가장 짧은데 정확도도 최저(0.769)였고,
# 길이 0~20자 구간의 정확도가 0.867, 21~50자 구간은 0.952였다. 최대 290자까지 나온 걸 보면
# read_tokens 상한에 막힌 게 아니라 모델이 스스로 일찍 멈춘다. 즉 "읽고 틀린" 게 아니라 "덜 읽어서" 틀린다.
# 그래서 정답으로 보이는 한 조각이 아니라 관련될 수 있는 항목을 빠짐없이 옮기게 한다.
READ_1_V2 = ("Question: {question}\n{choices}\n"
             "Do not answer yet. Transcribe every piece of text in the photo that could bear on this question — "
             "not only the single item you believe is the answer. If the question involves a table, list, menu or "
             "schedule, transcribe each relevant row on its own line with both its label and its value. If it requires "
             "comparing or combining several items, transcribe all of them. Keep the original language, digits, units "
             "and spacing; text may be small, tilted or upside down. If the answer depends on objects rather than text, "
             "describe those objects and where they are instead. Output only the transcription.")


# 1단계 판독 지시 v3 — 구조 보존 (2026-09-23).
#
# v2 는 "빠짐없이 옮겨라"로 **누락**을 줄이는 판이다. 그런데 test 불일치 200건을 눈으로 분류한
# 결과, 판독 실패가 관여한 것은 32건(16%)뿐이고 84%가 판독 이후 단계였다. 그중 가장 넓은 것이
# **소유권 상실**이다 — 메뉴명과 가격, 간판과 층, 요일과 시간이 판독문에 다 들어 있는데
# 어느 것이 어느 것의 짝인지 잃는다.
#
# 타일 위치는 이미 TILES_NOTE 로 알려주고 있다(왼쪽 위/오른쪽 위/왼쪽 아래/오른쪽 아래).
# 빠진 것은 **출력 형식에 구조를 요구하지 않는다**는 점이다. 그래서 v3 는 한 줄에 한 항목씩
# `위치 | 라벨 | 값` 으로 내보내게 한다. 누락 줄이기(v2)와 짝 보존(v3)은 서로 다른 축이라
# 스크리닝에서 따로 비교한다.
READ_1_V3 = ("Question: {question}\n{choices}\n"
             "Do not answer yet. Transcribe every piece of text in the photo that could bear on this question — "
             "not only the single item you believe is the answer. "
             "Output one line per item in the form `where | label | value`, where "
             "`where` is which image it came from (full / top-left / top-right / bottom-left / bottom-right), "
             "`label` is what the text belongs to (the menu name, the shop name, the floor, the day, the field name) "
             "and `value` is the text itself. Keep the original language, digits, units and spacing. "
             "If a row has no separate label, repeat the nearest heading as the label. "
             "For tables, menus, schedules and building signs, emit one line per row so that each value stays "
             "attached to its own row. If the answer depends on objects rather than text, describe each object "
             "and where it is on its own line instead. Output only these lines.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantization", default=None, help="예: moe_wna16 (Qwen3.5 MoE GPTQ-Int4 모델 카드 권장)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val667")
    ap.add_argument("--split-json", default=None)
    ap.add_argument("--ids-json", default=None)
    ap.add_argument("--gold-json", default=None)
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-mem", type=float, default=0.93)
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel GPU 수(RunPod 다중 GPU)")
    ap.add_argument("--pp", type=int, default=1, help="pipeline parallel 단계 수(레이어 분할, P2P 통신). tp 행업 회피용")
    ap.add_argument("--executor", default=None, choices=[None,"mp","ray","uni"],
                    help="분산 executor. B300x2에서 기본 mp 워커 init이 행업 → ray로 우회")
    ap.add_argument("--batch", type=int, default=32, help="한 번에 generate에 넘길 요청 수")
    ap.add_argument("--kv-gb", type=float, default=None,
                    help="KV 캐시 크기를 GiB로 고정. 96GB 한 장의 122B-Int4는 자동 추정이 여유를 넘겨 OOM이 났다")
    ap.add_argument("--max-num-seqs", type=int, default=None, help="동시 처리 요청 수 상한(메모리 절약)")
    ap.add_argument("--enforce-eager", action="store_true", help="CUDA graph를 끄고 메모리를 아낀다")
    ap.add_argument("--attention-backend", default=None,
                    help="vLLM attention 커널 선택(예: TRITON_ATTN). B300에서 flashinfer TRT-LLM attention JIT가 실패해 추가")
    ap.add_argument("--moe-backend", default=None,
                    help="vLLM MoE 커널 선택(예: triton). B300(sm_103)에서 기본 flashinfer_trtllm JIT 컴파일이 실패해 추가")
    ap.add_argument("--disable-custom-ar", action="store_true",
                    help="vLLM custom all-reduce를 끈다. RunPod B300x2에서 NCCL 초기화 직후 멈춰서 추가(NCCL_NVLS_ENABLE=0과 함께)")
    ap.add_argument("--prompt", default="ours", choices=["ours", "read"],
                    help="read: 1단계 판독 생성 → 2단계 a~d 채점(colab_35b_infer.py --prompt read와 같은 문구)")
    ap.add_argument("--read-tokens", type=int, default=80)
    ap.add_argument("--fewshot", type=int, default=0,
                    help="풀어 둔 예시 K개를 프롬프트에 넣는다(텍스트 전용). build_fewshot 주석 참조. "
                         "read 프롬프트에서는 2단계(선택)에만 붙인다 — 배우게 하려는 것이 "
                         "'어떻게 보느냐'가 아니라 '비슷한 보기 중 무엇을 고르느냐'이기 때문이다")
    ap.add_argument("--fewshot-seed", type=int, default=0)
    ap.add_argument("--min-pixels", type=int, default=None,
                    help="비전 processor 의 min_pixels. 다른 참가자(38위) 실측에서 max_pixels 만 올리면 "
                         "작은 원본이 확대되지 않아 성능이 안 올랐고 min_pixels 를 올려야 올랐다"
                         "(256..2048 92.0%% -> 2304..3072 95.5%%). 우리는 한 번도 조정한 적이 없다")
    ap.add_argument("--max-pixels", type=int, default=None)
    ap.add_argument("--read1", default="v1", choices=["v1", "v2", "v3"],
                    help="read 1단계 판독 지시. v2=관련 항목을 빠짐없이 옮기게 하는 recall 강화판. "
                         "v2를 쓸 때는 --read-tokens 도 같이 올려야 의미가 있다(기본 80은 짧다). "
                         "'토큰만 늘린 효과'와 분리하려면 v1 + 같은 read-tokens 대조군을 함께 돌린다")
    ap.add_argument("--think", type=int, default=0,
                    help="2단계에서 먼저 생각하게 한다(생각 토큰 상한). 0 이면 끔. read 프롬프트에서만 쓴다")
    ap.add_argument("--read2", default="v1", choices=["v1", "v2", "v3", "both"],
                    help="read 2단계 답 선택 지시. v2=필드·부정·정확일치 점검. v3=v2 + 동점일 때 라벨 타이브레이크"
                         "(크게/위쪽/로마자). both=같은 판독문으로 v1(기본 행)과 v2(행의 'v2' 필드)를 모두 채점")
    ap.add_argument("--view", default="full", choices=["full", "tiles2x2"])
    ap.add_argument("--tile-scale", type=float, default=2.0)
    ap.add_argument("--shift", type=int, default=0, choices=[0, 1, 2, 3],
                    help="선택지 순환(choice-shift TTA): 보기 위치 j에 원래 보기 (j+shift)%%4를 보여 준다. "
                         "저장하는 logprobs·pred는 원래 a~d 순서로 되돌린 값이다")
    args = ap.parse_args()

    from PIL import Image, ImageOps
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = (f"{args.model.split('/')[-1]}_{args.prompt}_{args.split}{args.tag_suffix}"
           + (f"_{args.view}x{args.tile_scale:g}" if args.view != "full" else "")
           + (f"_shift{args.shift}" if args.shift else "")
           + {"v1": "", "v2": "_r1v2", "v3": "_r1v3"}[args.read1]
           + {"v1": "", "v2": "_r2v2", "v3": "_r2v3", "both": "_r2both"}[args.read2]
           + (f"_think{args.think}" if args.think else "") + "_vllm")
    read1 = {"v2": READ_1_V2, "v3": READ_1_V3}.get(args.read1, READ_1)
    read2 = {"v2": READ_2_V2, "v3": READ_2_V3}.get(args.read2, READ_2)
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
    todo = rows[~rows.id.isin(done)].reset_index(drop=True)
    print(f"[{tag}] 전체 {len(rows)} / 완료 {len(done)} / 남음 {len(todo)}", flush=True)

    fewshot = ""
    if args.fewshot:
        # load_rows 의 "train" 분기는 val<N> 표본용이라 split="train" 을 주면 int("train") 으로 죽는다.
        import pandas as pd
        pool = pd.read_csv(data / "train.csv", encoding="utf-8-sig", dtype=str,
                           keep_default_na=False).to_dict("records")
        fewshot = build_fewshot(pool, args.fewshot, args.fewshot_seed, set(rows.id))
        assert fewshot, "few-shot 예시를 만들지 못했다(풀이 비었거나 전부 제외됐다)"
        print(f"[{tag}] few-shot {args.fewshot}개, {len(fewshot)}자", flush=True)

    t0 = time.time()
    mmkw = {k: v for k, v in (("min_pixels", args.min_pixels), ("max_pixels", args.max_pixels)) if v}
    processor = AutoProcessor.from_pretrained(args.model, **mmkw)
    tok = processor.tokenizer
    letter_ids = [tok.encode(x, add_special_tokens=False) for x in LETTERS]
    assert all(len(i) == 1 for i in letter_ids), letter_ids
    letter_ids = [i[0] for i in letter_ids]
    llm = LLM(model=args.model, quantization=args.quantization, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, tensor_parallel_size=args.tp,
              pipeline_parallel_size=args.pp,
              **({"distributed_executor_backend": args.executor} if args.executor else {}),
              limit_mm_per_prompt={"image": 5}, enable_prefix_caching=False,
              **({"mm_processor_kwargs": mmkw} if mmkw else {}),
              enforce_eager=args.enforce_eager,
              disable_custom_all_reduce=args.disable_custom_ar,
              **({"moe_backend": args.moe_backend} if args.moe_backend else {}),
              **({"attention_backend": args.attention_backend} if args.attention_backend else {}),
              **({"kv_cache_memory_bytes": int(args.kv_gb * 1024**3)} if args.kv_gb else {}),
              **({"max_num_seqs": args.max_num_seqs} if args.max_num_seqs else {}))
    # logprobs는 제한 전 분포의 상위 k개에서 오므로 20개를 받아 a~d가 빠지는 경우를 줄인다.
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20, allowed_token_ids=letter_ids)
    sp_read = SamplingParams(max_tokens=args.read_tokens, temperature=0.0)
    # 생각 단계: </think> 에서 멈추고, 그 뒤 a~d 한 토큰을 기존과 같은 방식으로 채점한다.
    sp_think = SamplingParams(max_tokens=args.think or 1, temperature=0.0, stop=["</think>"])
    load_sec = round(time.time() - t0, 1)
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "?"
    print(f"[{tag}] 모델 로드 {load_sec}s, GPU {gpu}, letter ids {letter_ids}", flush=True)
    meta = {"model": args.model, "quantization": args.quantization, "engine": "vllm",
            "prompt": read1 + " || " + read2 if args.prompt == "read" else OURS,
            "read1": args.read1, "read2": args.read2, "read_tokens": args.read_tokens,
            **({"prompt_v2_stage2": READ_2_V2} if args.read2 == "both" else {}), "view": args.view,
            "tile_scale": args.tile_scale, "choice_shift": args.shift,
            "fewshot": args.fewshot, "fewshot_seed": args.fewshot_seed,
            "min_pixels": args.min_pixels, "max_pixels": args.max_pixels,
            "enable_thinking": bool(args.think), "think_budget": args.think, "think_note": THINK_NOTE if args.think else None, "scoring": "next-token logprob restricted to a-d", "split": args.split,
            "n": len(rows), "id_set_sha256": hashlib.sha256("\n".join(sorted(rows.id)).encode()).hexdigest(),
            "gpu": gpu, "tp": args.tp, "load_sec": load_sec, "max_model_len": args.max_model_len}
    (out / f"{tag}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    n_ok = n_seen = 0
    t1 = time.time()
    with pred_path.open("a", encoding="utf-8") as fh:
        for s in range(0, len(todo), args.batch):
            chunk = todo.iloc[s:s + args.batch]
            reqs, convs = [], []
            for r in chunk.itertuples():
                img = ImageOps.exif_transpose(Image.open(data / r.path)).convert("RGB")
                views = [img] + (tiles_2x2(img, args.tile_scale, Image.BICUBIC) if args.view == "tiles2x2" else [])
                choices = "\n".join(f"{x}. {getattr(r, LETTERS[(j + args.shift) % 4])}" for j, x in enumerate(LETTERS))
                text = (read1 if args.prompt == "read" else OURS).format(question=r.question, choices=choices)
                if fewshot and args.prompt != "read":
                    text = fewshot + text
                if args.view == "tiles2x2":
                    text = TILES_NOTE + text
                messages = [{"role": "user", "content": [{"type": "image"} for _ in views] + [{"type": "text", "text": text}]}]
                convs.append(messages)
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                       enable_thinking=False)
                reqs.append({"prompt": prompt, "multi_modal_data": {"image": views if len(views) > 1 else img}})
            transcripts = None
            outs_v2 = None
            reasons = None
            if args.prompt == "read":
                gens = llm.generate(reqs, sp_read, use_tqdm=False)
                transcripts = [g.outputs[0].text.strip() for g in gens]
                reqs_v2 = []
                for req, messages, tr in zip(reqs, convs, transcripts):
                    base = messages + [{"role": "assistant", "content": [{"type": "text", "text": tr}]}]
                    req["prompt"] = processor.apply_chat_template(
                        base + [{"role": "user", "content": [{"type": "text", "text": fewshot + read2}]}],
                        tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    if args.read2 == "both":  # 같은 판독문 tr 위에 2단계 지시만 v2로 바꾼 요청
                        reqs_v2.append({"prompt": processor.apply_chat_template(
                            base + [{"role": "user", "content": [{"type": "text", "text": READ_2_V2}]}],
                            tokenize=False, add_generation_prompt=True, enable_thinking=False),
                            "multi_modal_data": req["multi_modal_data"]})
                if reqs_v2:
                    outs_v2 = llm.generate(reqs_v2, sp, use_tqdm=False)
                if args.think:
                    # 2026-09-24 H408 오답 분석: 판독은 맞는데 질문이 가리키는 대상·필드를 잘못 고르는 오류가
                    # 실제 오답 19개의 대부분이었다. 답하기 전에 무엇을 묻는지부터 따지게 한다.
                    think_reqs = []
                    for req, messages, tr in zip(reqs, convs, transcripts):
                        base = messages + [{"role": "assistant", "content": [{"type": "text", "text": tr}]}]
                        think_reqs.append({"prompt": processor.apply_chat_template(
                            base + [{"role": "user", "content": [{"type": "text", "text": fewshot + read2 + THINK_NOTE}]}],
                            tokenize=False, add_generation_prompt=True, enable_thinking=True),
                            "multi_modal_data": req["multi_modal_data"]})
                    gens_t = llm.generate(think_reqs, sp_think, use_tqdm=False)
                    reasons = []
                    for req, tq, gt in zip(reqs, think_reqs, gens_t):
                        rt = gt.outputs[0].text
                        reasons.append((rt, len(gt.outputs[0].token_ids)))
                        req["prompt"] = tq["prompt"] + rt + "\n</think>\n\n"
            outs = llm.generate(reqs, sp, use_tqdm=False)

            def to_scores(o):
                lp = o.outputs[0].logprobs[0]
                shown = [lp[t].logprob if t in lp else -1e4 for t in letter_ids]
                # 화면 위치 j의 점수는 원래 보기 (j+shift)%4의 점수다 → 원래 순서로 되돌린다.
                sc = [0.0] * 4
                for j, x in enumerate(shown):
                    sc[(j + args.shift) % 4] = x
                mx = max(sc)
                pr = [math.exp(x - mx) for x in sc]
                b = max(range(4), key=lambda q: sc[q])
                return sc, b, round(pr[b] / sum(pr), 6)

            for k, (r, o) in enumerate(zip(chunk.itertuples(), outs)):
                scores, best, conf = to_scores(o)
                row = {"id": r.id, "pred": LETTERS[best], "gold": r.answer or None,
                       "confidence": conf, "logprobs": [round(x, 6) for x in scores],
                       "prompt_tokens": len(o.prompt_token_ids)}
                if transcripts is not None:
                    row["transcript"] = transcripts[k]
                if reasons is not None:
                    row["reasoning"] = reasons[k][0][:3000]
                    row["think_tokens"] = reasons[k][1]
                if outs_v2 is not None:
                    s2, b2, c2 = to_scores(outs_v2[k])
                    row["v2"] = {"pred": LETTERS[b2], "confidence": c2, "logprobs": [round(x, 6) for x in s2],
                                 "prompt_tokens": len(outs_v2[k].prompt_token_ids)}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                if r.answer:
                    n_seen += 1
                    n_ok += int(LETTERS[best] == r.answer.strip().lower())
            fh.flush()
            i = min(s + args.batch, len(todo))
            rate = (time.time() - t1) / i
            acc = f" acc {n_ok}/{n_seen}={n_ok / n_seen:.4f}" if n_seen else ""
            print(f"[{tag}] {i}/{len(todo)} {rate:.2f}s/건 남은 {rate * (len(todo) - i) / 60:.0f}분{acc}", flush=True)
    meta["sec_per_item"] = round((time.time() - t1) / max(1, len(todo)), 3)
    meta["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1) if torch.cuda.is_available() else None
    (out / f"{tag}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        # vLLM 엔진 오류 후 워커가 남아 프로세스가 멈추는(행) 일이 두 번 있었다(2026-09-22) → 즉시 종료해 GPU 대기 비용을 막는다.
        import os
        import traceback
        traceback.print_exc()
        os._exit(1)
