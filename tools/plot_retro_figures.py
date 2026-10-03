# -*- coding: utf-8 -*-
"""Kaggle Discussion 회고 글용 그래프 4장을 만든다. GPU 미사용.

  retro1_size_vs_input.png   모델 크기 vs 입력 방식 (val667 정확도)
  retro2_error_overlap.png   모델 간 오답 겹침(자카드) 히트맵 + 묶음별 평균
  retro3_public_progress.png Public 점수 추이
  (앙상블 계보 그림은 tools/plot_ensemble_lineage.py 의 SVG 를 쓴다)

수치 출처: val667 예측(shared_predictions/*), 제출 기록(README·HANDOFF). 대회 문항·사진은 쓰지 않는다.
usage: python tools/plot_retro_figures.py [--out reports/figures/retro]
"""
import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
ROOT = Path(__file__).resolve().parents[1]
INK, INK2, GRID = '#1f1f1f', '#5a5a5a', '#e4e3df'
BLUE, ORANGE, GRAY = '#2a78d6', '#eb6834', '#b5b4ad'

VAL667 = {  # 이름: (예측 파일, 학습 여부, 계열)
    '9B LoRA': ('shared_predictions/runpod_h_20260924/9b_base/val_predictions.jsonl', '학습', 'Qwen'),
    '27B LoRA': ('shared_predictions/runpod_27b_R3/R3_27b_tiles/val_predictions.jsonl', '학습', 'Qwen'),
    '27B LoRA+셔플': ('shared_predictions/runpod_g_20260924/val_predictions.jsonl', '학습', 'Qwen'),
    'Gemma 31B LoRA': ('shared_predictions/runpod_gemma4_31b/gemma4_31b_lora/val_predictions.jsonl', '학습', 'Gemma'),
    '35B': ('shared_predictions/colab_ai2_35b_out/Qwen3.6-35B-A3B_read_val667_tiles2x2x2_predictions.jsonl', '제로샷', 'Qwen'),
    '122B': ('shared_predictions/runpod_122b_bf16/Qwen3.5-122B-A10B_read_val667_tiles2x2x2_vllm_predictions.jsonl', '제로샷', 'Qwen'),
    '397B': ('shared_predictions/nebius_397b/Qwen3.5-397B-A17B-FP8_read_val667_tiles2x2x2_vllm_predictions.jsonl', '제로샷', 'Qwen'),
    'Qwen3.8 27B': ('shared_predictions/runpod_q38_20260924/Qwen3.8-27B_read_val667_tiles2x2x2_vllm_predictions.jsonl', '제로샷', 'Qwen'),
}


def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_color(GRID)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig_size_vs_input(out):
    # val667 정확도. 원본 입력: 9B NF4 / 35B / 122B GPTQ-Int4, 확대+받아 적기: 35B / 122B BF16 / 397B FP8
    raw = [('9B', .9430), ('35B', .9445), ('122B', .9490)]
    zoom = [('35B', .9655), ('122B', .9580), ('397B', .9640)]
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    xs1 = [0, 1, 2]
    xs2 = [3.6, 4.6, 5.6]
    b1 = ax.bar(xs1, [v for _, v in raw], width=0.72, color=GRAY)
    b2 = ax.bar(xs2, [v for _, v in zoom], width=0.72, color=BLUE)
    for bars, data in ((b1, raw), (b2, zoom)):
        for b, (_, v) in zip(bars, data):
            ax.text(b.get_x() + b.get_width() / 2, v + .0012, f'{v:.4f}', ha='center', va='bottom', fontsize=10, color=INK)
    ax.set_xticks(xs1 + xs2, [n for n, _ in raw] + [n for n, _ in zoom])
    ax.set_ylim(.93, .975)
    ax.set_ylabel('검증셋 정확도 (val667)', color=INK2)
    ax.text(1, .9715, '사진 그대로 입력', ha='center', fontsize=12, color=INK2, weight='bold')
    ax.text(4.6, .9715, '확대 조각 + 글자 받아 적기', ha='center', fontsize=12, color=BLUE, weight='bold')
    ax.set_title('모델을 키운 효과 < 사진을 잘 보여 준 효과', fontsize=14, color=INK, pad=12, weight='bold')
    style(ax)
    fig.text(.01, .01, '122B: 왼쪽 GPTQ-Int4, 오른쪽 BF16 · 397B는 확대 입력으로만 측정 · 모두 추가 학습 없음', fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, .03, 1, 1))
    fig.savefig(out / 'retro1_size_vs_input.png')
    plt.close(fig)


def fig_error_overlap(out):
    wrong = {}
    for k, (f, _, _) in VAL667.items():
        rows = [json.loads(l) for l in open(ROOT / f, encoding='utf-8') if l.strip()]
        wrong[k] = {r['id'] for r in rows if r['pred'].strip().lower() != r['gold'].strip().lower()}
    names = list(VAL667)
    jac = lambda a, b: len(wrong[a] & wrong[b]) / len(wrong[a] | wrong[b])  # noqa: E731

    fig = plt.figure(figsize=(13.5, 6.2), dpi=150)
    ax = fig.add_axes([.09, .27, .36, .62])
    n = len(names)
    m = [[jac(a, b) if a != b else float('nan') for b in names] for a in names]
    im = ax.imshow(m, cmap='Blues', vmin=.3, vmax=.7)
    for i in range(n):
        for j in range(n):
            if i != j:
                v = m[i][j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8, color='white' if v > .56 else INK)
    ax.set_xticks(range(n), names, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n), names, fontsize=9)
    ax.axhline(3.5, color='white', lw=3)
    ax.axvline(3.5, color='white', lw=3)
    ax.set_title('두 모델이 함께 틀린 비율 (진할수록 같이 틀림)', fontsize=12, color=INK, pad=10, weight='bold')
    fig.colorbar(im, cax=fig.add_axes([.465, .27, .01, .62]))
    fig.text(.09, .03, '위 4개: 대회 데이터로 추가 학습(LoRA) · 아래 4개: 추가 학습 없이 사용', fontsize=9, color=INK2)

    groups = {'학습 ↔ 학습 안 함': [], '학습 ↔ 학습': [], '학습 안 함 ↔ 학습 안 함': [],
              'Qwen ↔ Qwen': [], 'Gemma ↔ Qwen': []}
    for a, b in itertools.combinations(names, 2):
        ta, tb = VAL667[a][1], VAL667[b][1]
        key = '학습 ↔ 학습' if ta == tb == '학습' else '학습 안 함 ↔ 학습 안 함' if ta == tb else '학습 ↔ 학습 안 함'
        groups[key].append(jac(a, b))
        groups['Gemma ↔ Qwen' if 'Gemma 31B LoRA' in (a, b) else 'Qwen ↔ Qwen'].append(jac(a, b))
    ax2 = fig.add_axes([.69, .27, .28, .62])
    labels = list(groups)
    vals = [sum(v) / len(v) for v in groups.values()]
    colors = [BLUE, GRAY, GRAY, ORANGE, ORANGE]
    bars = ax2.barh(range(len(labels))[::-1], vals, color=colors, height=.6)
    for b, v in zip(bars, vals):
        ax2.text(v + .005, b.get_y() + b.get_height() / 2, f'{v:.3f}', va='center', fontsize=10, color=INK)
    ax2.set_yticks(range(len(labels))[::-1], labels, fontsize=10)
    ax2.set_xlim(.4, .54)
    ax2.set_title('묶음별 평균 (낮을수록 서로 다르게 틀림)', fontsize=12, color=INK, pad=10, weight='bold')
    ax2.axhline(1.5, color=GRID, lw=1, ls='--')
    style(ax2)
    ax2.xaxis.grid(True, color=GRID, lw=.8)
    ax2.yaxis.grid(False)
    fig.text(.60, .03, '파랑·회색: 학습 방식으로 묶음 · 주황: 모델 계열로 묶음', fontsize=9, color=INK2)
    fig.savefig(out / 'retro2_error_overlap.png')
    plt.close(fig)


def fig_public_progress(out):
    pts = [('9B', .94221), ('35B', .95144), ('9B+35B', .95621), ('35B 확대', .96455), ('397B', .96693),
           ('35B+397B', .96931), ('4모델', .97110), ('5모델', .97170), ('T4', .97348), ('T6 확률평균', .97438),
           ('T18 +셔플모델', .97497), ('T28 5개 균등평균', .97706)]
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=150)
    xs = range(len(pts))
    ys = [v for _, v in pts]
    ax.plot(xs, ys, color='#1553a8', lw=3.5, marker='o', ms=8, markeredgecolor='white', markeredgewidth=1.5, zorder=3)
    for x, (n, v) in zip(xs, pts):
        if n in ('9B', '35B 확대', '35B+397B', 'T6 확률평균', 'T28 5개 균등평균'):
            ax.annotate(f'{v:.5f}', (x, v), textcoords='offset points', xytext=(0, 9), ha='center', fontsize=9, color=INK)
    ax.set_xticks(list(xs), [n for n, _ in pts], rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Public 정확도', color=INK2)
    ax.set_ylim(.938, .981)
    ax.axvspan(-.4, 2.4, color='#f3f2ee', zorder=0)
    ax.axvspan(2.6, 5.4, color='#eaf2fb', zorder=0)
    ax.axvspan(5.6, 11.4, color='#fdf0ea', zorder=0)
    ax.text(1, .9785, '모델 고르기', ha='center', fontsize=10, color=INK2)
    ax.text(4, .9785, '입력 바꾸기 + 큰 모델', ha='center', fontsize=10, color=INK2)
    ax.text(8.5, .9785, '앙상블', ha='center', fontsize=10, color=INK2)
    ax.set_title('제출할 때마다 오른 Public 점수 (주요 제출만)', fontsize=13, color=INK, pad=12, weight='bold')
    style(ax)
    fig.tight_layout()
    fig.savefig(out / 'retro3_public_progress.png')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='reports/figures/retro')
    out = ROOT / ap.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    fig_size_vs_input(out)
    fig_error_overlap(out)
    fig_public_progress(out)
    print(out)


if __name__ == '__main__':
    main()
