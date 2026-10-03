# -*- coding: utf-8 -*-
"""README(비공개 저장소)용 VQA 샘플 그림: 쉬움·보통·어려움 문항 한 장씩.

난이도는 val667 에서 모델 8개 중 몇 개가 맞혔는지로 정했다(쉬움 8/8, 보통 4/8, 어려움 2/8).
사람 얼굴·개인정보가 보이지 않는 사진만 골랐다.

⚠️ 대회 사진·문항이 들어 있다 — 공개본에는 넣지 않는다(tools/export_public.py 의 DENY 에 등록).
usage: python tools/plot_vqa_samples.py [--out reports/figures/samples/vqa_samples.png]
"""
import argparse
import csv
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = [('쉬움', 'train_0031.jpg'), ('보통', 'train_3349.jpg'), ('어려움', 'train_2112.jpg')]
MODELS = {
    '9B': 'shared_predictions/runpod_h_20260924/9b_base/val_predictions.jsonl',
    '27B': 'shared_predictions/runpod_27b_R3/R3_27b_tiles/val_predictions.jsonl',
    '27B+셔플': 'shared_predictions/runpod_g_20260924/val_predictions.jsonl',
    'Gemma': 'shared_predictions/runpod_gemma4_31b/gemma4_31b_lora/val_predictions.jsonl',
    '35B': 'shared_predictions/colab_ai2_35b_out/Qwen3.6-35B-A3B_read_val667_tiles2x2x2_predictions.jsonl',
    '122B': 'shared_predictions/runpod_122b_bf16/Qwen3.5-122B-A10B_read_val667_tiles2x2x2_vllm_predictions.jsonl',
    '397B': 'shared_predictions/nebius_397b/Qwen3.5-397B-A17B-FP8_read_val667_tiles2x2x2_vllm_predictions.jsonl',
    'Qwen3.8': 'shared_predictions/runpod_q38_20260924/Qwen3.8-27B_read_val667_tiles2x2x2_vllm_predictions.jsonl',
}
COLOR = {'쉬움': (27, 150, 90), '보통': (230, 150, 0), '어려움': (220, 70, 60)}
INK, INK2, OK, BAD = (30, 30, 30), (110, 110, 110), (27, 130, 80), (200, 60, 50)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='reports/figures/samples/vqa_samples.png')
    out = ROOT / ap.parse_args().out
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = {r['id']: r for r in csv.DictReader(open(ROOT / 'ssafy-16-2-ai/train.csv', encoding='utf-8-sig'))}
    preds = {m: {r['id']: r['pred'].strip().lower() for r in map(json.loads, open(ROOT / f, encoding='utf-8'))}
             for m, f in MODELS.items()}
    f_title = ImageFont.truetype('C:/Windows/Fonts/malgunbd.ttf', 30)
    f_text = ImageFont.truetype('C:/Windows/Fonts/malgun.ttf', 21)
    f_bold = ImageFont.truetype('C:/Windows/Fonts/malgunbd.ttf', 21)
    f_small = ImageFont.truetype('C:/Windows/Fonts/malgun.ttf', 17)

    W, PH, H = 560, 560, 930
    canvas = Image.new('RGB', (W * 3 + 40, H), 'white')
    d = ImageDraw.Draw(canvas)
    for k, (level, sid) in enumerate(SAMPLES):
        r = rows[sid]
        x0 = 20 + k * W
        photo = ImageOps.exif_transpose(Image.open(ROOT / 'ssafy-16-2-ai' / r['path'])).convert('RGB')
        photo.thumbnail((W - 30, PH))
        d.rounded_rectangle((x0, 20, x0 + 120, 62), radius=10, fill=COLOR[level])
        d.text((x0 + 60, 41), level, font=f_title, fill='white', anchor='mm')
        right = [m for m in MODELS if preds[m][sid] == r['answer']]
        d.text((x0 + 135, 41), f'모델 8개 중 {len(right)}개 정답', font=f_bold, fill=INK, anchor='lm')
        canvas.paste(photo, (x0 + (W - 30 - photo.width) // 2, 80))
        y = 80 + PH + 20
        for line in textwrap.wrap('Q. ' + r['question'], 24):
            d.text((x0, y), line, font=f_bold, fill=INK)
            y += 30
        y += 6
        for opt in 'abcd':
            is_ans = opt == r['answer']
            d.text((x0 + 10, y), f'{opt}) {r[opt]}' + ('   ← 정답' if is_ans else ''),
                   font=f_bold if is_ans else f_text, fill=OK if is_ans else INK2)
            y += 30
        y += 10
        d.text((x0, y), '맞힘: ' + (', '.join(right) or '없음'), font=f_small, fill=OK)
        wrong = [m for m in MODELS if m not in right]
        d.text((x0, y + 26), '틀림: ' + (', '.join(wrong) or '없음'), font=f_small, fill=BAD)
    canvas.save(out)
    print(out)


if __name__ == '__main__':
    main()
