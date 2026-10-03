# -*- coding: utf-8 -*-
"""README용 "VQA가 뭔지" 설명 그림. 실제 대회 사진·문항이 아니라 설명을 위해 다시 그린 장면이다.

예시 1: 글자가 너무 작아 원본만 보면 틀리고, 확대 조각(tiles2x2)으로 읽으면 맞힌다.
예시 2: 여러 줄을 읽고 비교해야 맞힌다(받아 적기, read).
장면은 2배 크기로 그린 뒤 줄여서 '원본'으로 쓰고, 확대 조각은 그 고해상도 장면을 잘라 키운다.
공개 저장소에 올려도 되는 합성 그림이다.

usage: python tools/plot_vqa_example.py [--out reports/figures/vqa_example.png]
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONT = 'C:/Windows/Fonts/malgun.ttf'
FONT_B = 'C:/Windows/Fonts/malgunbd.ttf'
INK, MUTED, LINE = (34, 34, 34), (120, 120, 120), (225, 225, 225)
OK, BAD, ACCENT, ORANGE = (27, 130, 80), (210, 70, 30), (60, 90, 220), (240, 140, 20)


def f(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, size)


def cafe_scene():
    """카페 입구(2배 크기). 유리문 오른쪽 아래에 작은 영업시간 스티커가 있다."""
    W, H = 1280, 840
    im = Image.new('RGB', (W, H), (122, 92, 74))
    d = ImageDraw.Draw(im)
    for y in range(0, H, 44):  # 벽돌
        off = 0 if (y // 44) % 2 else 60
        for x in range(-off, W, 120):
            d.rectangle((x + 3, y + 3, x + 117, y + 41), fill=(138, 104, 84))
    d.rectangle((0, 0, W, 150), fill=(40, 52, 48))  # 간판
    d.text((W // 2, 75), 'CAFE  모퉁이', font=f(72, True), fill=(240, 226, 196), anchor='mm')
    d.rectangle((380, 200, 900, H), fill=(70, 70, 70))  # 문틀
    d.rectangle((400, 220, 880, H), fill=(170, 196, 204))  # 유리
    d.polygon([(420, 240), (520, 240), (420, 420)], fill=(205, 224, 230))  # 반사
    d.polygon([(560, 240), (600, 240), (440, 560), (420, 560)], fill=(195, 216, 222))
    d.rectangle((830, 470, 846, 600), fill=(60, 60, 60))  # 손잡이
    d.rectangle((150, 300, 330, 560), fill=(55, 70, 60))  # 왼쪽 메뉴 보드
    d.text((240, 330), 'TODAY', font=f(30, True), fill=(230, 230, 210), anchor='mm')
    for i, s in enumerate(['라떼', '에이드', '케이크']):
        d.text((175, 380 + i * 50), s, font=f(26), fill=(230, 230, 210))
    sx, sy = 690, 660  # 스티커
    d.rectangle((sx, sy, sx + 150, sy + 92), fill=(250, 250, 246), outline=(200, 200, 200))
    d.text((sx + 75, sy + 22), 'OPEN 10:00', font=f(19, True), fill=(40, 40, 40), anchor='mm')
    d.text((sx + 75, sy + 50), 'CLOSE 22:00', font=f(19, True), fill=(40, 40, 40), anchor='mm')
    d.text((sx + 75, sy + 76), '연중무휴', font=f(15), fill=(90, 90, 90), anchor='mm')
    return im.filter(ImageFilter.GaussianBlur(1.1)), (sx - 30, sy - 25, sx + 180, sy + 115)


def bus_scene():
    """버스 정류장 안내판(2배 크기). 노선별 막차 시각이 작은 글씨로 나열돼 있다."""
    W, H = 1280, 840
    im = Image.new('RGB', (W, H), (176, 196, 214))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 560, W, H), fill=(120, 120, 124))  # 도로
    d.rectangle((0, 540, W, 560), fill=(200, 200, 200))
    d.rectangle((140, 80, 180, 560), fill=(70, 74, 80))  # 기둥
    d.rectangle((1100, 80, 1140, 560), fill=(70, 74, 80))
    d.rectangle((120, 60, 1160, 110), fill=(40, 90, 160))  # 지붕
    d.text((640, 85), '정류장  ·  중앙시장', font=f(30, True), fill='white', anchor='mm')
    d.rectangle((560, 170, 1060, 470), fill=(22, 26, 36), outline=(60, 60, 60), width=6)  # 안내판
    amber = (255, 184, 60)
    d.text((585, 195), '노선    행선지        막차', font=f(24, True), fill=(150, 150, 150))
    for i, (no, dest, t) in enumerate([('720', '강남역', '23:40'), ('151', '시청', '24:10'), ('402', '서울역', '23:55'), ('7016', '상명대', '23:20')]):
        y = 245 + i * 52
        d.text((585, y), no, font=f(28, True), fill=amber)
        d.text((700, y), dest, font=f(28), fill=amber)
        d.text((935, y), t, font=f(28, True), fill=amber)
    d.rectangle((230, 300, 470, 540), fill=(90, 110, 130))  # 벤치·광고판
    d.text((350, 420), 'AD', font=f(60, True), fill=(160, 180, 200), anchor='mm')
    return im.filter(ImageFilter.GaussianBlur(1.0))


def card(d, box, title):
    d.rounded_rectangle(box, radius=18, fill='white', outline=LINE, width=2)
    x0, y0 = box[0] + 26, box[1] + 22
    tw = d.textlength(title, font=f(22, True))
    d.rounded_rectangle((x0, y0, x0 + tw + 28, y0 + 38), radius=19, fill=(255, 238, 222))
    d.text((x0 + 14, y0 + 19), title, font=f(22, True), fill=ORANGE, anchor='lm')
    return x0, y0 + 56


def options(d, x, y, items, w):
    for label, kind in items:
        color = OK if kind == 'ok' else BAD if kind == 'bad' else MUTED
        font = f(23, kind != '')
        d.text((x, y), label, font=font, fill=color)
        y += 36
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='reports/figures/vqa_example.png')
    out = ROOT / ap.parse_args().out
    out.parent.mkdir(parents=True, exist_ok=True)

    Wc, Hc = 1900, 900
    canvas = Image.new('RGB', (Wc, Hc), (247, 247, 249))
    d = ImageDraw.Draw(canvas)
    d.text((40, 30), 'VQA = 사진을 보고 4지선다 질문에 답하기', font=f(36, True), fill=INK)
    d.text((40, 82), '사진 + 질문 + 보기 a~d  →  AI가 정답 한 글자를 고른다.  아래는 우리 방법이 왜 필요했는지 보여 주는 예시다.', font=f(22), fill=MUTED)

    # 예시 1
    x, y = card(d, (40, 130, 760, 850), '예시 1 · 글자가 너무 작음')
    hi, sticker = cafe_scene()
    orig = hi.resize((330, 216), Image.LANCZOS)
    canvas.paste(orig, (x, y))
    s = 330 / hi.width
    bx = [int(v * s) for v in sticker]
    d.rectangle((x + bx[0], y + bx[1], x + bx[2], y + bx[3]), outline=ORANGE, width=3)
    zoom = hi.crop(sticker).resize((330, int(330 * (sticker[3] - sticker[1]) / (sticker[2] - sticker[0]))), Image.LANCZOS)
    canvas.paste(zoom, (x + 348, y))
    d.rectangle((x + 348, y, x + 348 + zoom.width, y + zoom.height), outline=ORANGE, width=3)
    d.text((x, y + 226), '원본 — 주황 박스 속 스티커', font=f(19), fill=MUTED)
    d.text((x + 348, y + zoom.height + 10), '확대 조각 — "CLOSE 22:00"이 읽힘', font=f(19), fill=MUTED)
    yy = y + 280
    d.text((x, yy), 'Q. 이 카페가 문을 닫는 시간은?', font=f(27, True), fill=INK)
    yy = options(d, x + 6, yy + 52, [('a) 21:00', ''), ('b) 21:30   ← 원본만 볼 때', 'bad'), ('c) 22:00   ← 정답', 'ok'), ('d) 22:30', '')], 640)
    d.rounded_rectangle((x, yy + 14, x + 668, yy + 70), radius=10, fill=(238, 242, 255))
    d.text((x + 16, yy + 42), '받아 적기 →  OPEN 10:00 · CLOSE 22:00', font=f(22, True), fill=ACCENT, anchor='lm')

    # 예시 2
    x, y = card(d, (790, 130, 1510, 850), '예시 2 · 읽고 비교해야 함')
    bus = bus_scene().resize((440, 289), Image.LANCZOS)
    canvas.paste(bus, (x, y))
    d.text((x, y + 299), '원본 — 정류장 안내판의 노선별 막차 시각', font=f(19), fill=MUTED)
    yy = y + 350
    d.text((x, yy), 'Q. 막차가 가장 늦게 오는 버스는?', font=f(27, True), fill=INK)
    yy = options(d, x + 6, yy + 52, [('a) 720번   ← 원본만 볼 때', 'bad'), ('b) 151번   ← 정답', 'ok'), ('c) 402번', ''), ('d) 7016번', '')], 640)
    d.rounded_rectangle((x, yy + 14, x + 668, yy + 70), radius=10, fill=(238, 242, 255))
    d.text((x + 16, yy + 42), '받아 적기 →  720 23:40 · 151 24:10 · 402 23:55', font=f(22, True), fill=ACCENT, anchor='lm')

    # 풀이
    x0 = 1540
    d.text((x0, 150), '그래서 이렇게 풀었다', font=f(26, True), fill=INK)
    steps = ['원본 1장 + 4분할 × 2배\n확대 조각 4장을 함께 입력', '질문과 관련된 글자를\n먼저 받아 적게 함', '받아 적은 글자로 a~d 중\n선택, 확률만 저장', '여러 모델의 확률을\n평균해 최종 답']
    yy = 200
    for i, t in enumerate(steps):
        d.rounded_rectangle((x0, yy, x0 + 330, yy + 100), radius=14, fill='white', outline=LINE, width=2)
        d.ellipse((x0 + 16, yy + 30, x0 + 56, yy + 70), fill=ACCENT)
        d.text((x0 + 36, yy + 50), str(i + 1), font=f(22, True), fill='white', anchor='mm')
        d.multiline_text((x0 + 70, yy + 50), t, font=f(20), fill=INK, anchor='lm', spacing=6)
        yy += 112
    d.rounded_rectangle((x0, yy + 6, x0 + 330, yy + 194), radius=14, fill=(232, 246, 238))
    d.text((x0 + 20, yy + 26), '검증셋 667문제 · 같은 35B 모델', font=f(18), fill=MUTED)
    d.text((x0 + 20, yy + 62), '0.9445 → 0.9655', font=f(34, True), fill=OK)
    d.text((x0 + 20, yy + 118), '추가 학습 없이 입력만 바꿔서', font=f(19), fill=INK)
    d.text((x0 + 20, yy + 150), '(모델 9B→122B: 0.943→0.958)', font=f(17), fill=MUTED)

    d.text((40, 875), '※ 설명을 위해 다시 그린 장면입니다. 실제 대회 사진·문제가 아닙니다 (대회 데이터는 공개하지 않습니다).', font=f(18), fill=MUTED, anchor='lm')
    canvas.save(out)
    print(out)


if __name__ == '__main__':
    main()
