# -*- coding: utf-8 -*-
"""모델 예측 -> 앙상블 제출 계보를 아크 다이어그램(SVG)으로 그린다. 의존성 없음.

가로축 왼쪽은 개별 모델 예측, 오른쪽은 앙상블 제출(제출 순서). 제출마다 그 제출을
만든 구성원(모델 또는 이전 제출)에서 아크가 들어온다. 축 아래 막대는 Public 점수.
최종 선택 T28·T30 으로 이어진 계보만 진하게 칠한다.

구성 출처: 커밋 메시지, reports/all_submissions_20260924.md, tools/export_test_confidence_csv.py.

usage: python tools/plot_ensemble_lineage.py [--out reports/figures/ensemble_lineage.svg]
"""
import argparse
import os
from xml.sax.saxutils import escape

FAMILY_COLOR = {  # dataviz 기본 팔레트 7슬롯 (validate_palette.js light 통과)
    "35B": "#2a78d6",
    "Gemma": "#eb6834",
    "Q38": "#1baf7a",
    "9B": "#eda100",
    "Q35": "#e87ba4",
    "27B": "#008300",
    "397B": "#4a3aa7",
}
FAMILY_NAME = {
    "35B": "Qwen3.6-35B-A3B",
    "397B": "Qwen3.5-397B-A17B",
    "27B": "Qwen3.6-27B (R3)",
    "Gemma": "Gemma 4 31B",
    "Q35": "Qwen3.5-27B (Q35)",
    "9B": "Qwen3.5-9B",
    "Q38": "Qwen3.8-27B",
}
SUB_EDGE = "#8a8983"
INK, INK2, MUTED, SURFACE, RULE = "#0b0b0b", "#52514e", "#8a8983", "#fcfcfb", "#d9d8d3"

# (id, 라벨, 계열, 단독 Public 또는 None)
MODELS = [
    ("9b_zs", "9B 제로샷", "9B", 0.94221),
    ("9b_lora", "9B LoRA", "9B", None),
    ("9b_dora", "9B DoRA", "9B", None),
    ("35b_zs", "35B 제로샷", "35B", 0.95144),
    ("35b_rt", "35B read+tiles", "35B", 0.96455),
    ("35b_shift", "35B 보기순환", "35B", None),
    ("35b_r2v3", "35B r2v3", "35B", None),
    ("35b_fs16", "35B fs16", "35B", None),
    ("35b_r1v3", "35B r1v3", "35B", None),
    ("35b_full", "35B 전체이미지", "35B", None),
    ("27b_zs", "27B 제로샷", "27B", None),
    ("27b_lora", "27B LoRA", "27B", 0.96187),
    ("r3", "R3 (27B tiles)", "27B", None),
    ("397b", "397B read+tiles", "397B", 0.96693),
    ("gemma", "Gemma LoRA", "Gemma", None),
    ("gemma_rot", "Gemma 보기회전", "Gemma", None),
    ("q35", "Q35", "Q35", None),
    ("q38", "Q38 제로샷", "Q38", None),
]

# (id, 라벨, 날짜, Public, 구성원)
SUBS = [
    ("E1", "9B+35B", "09-21", 0.95621, ["9b_zs", "35b_zs"]),
    ("E2", "27B+35B", "09-21", 0.95918, ["27b_zs", "35b_zs"]),
    ("E3", "35B+397B", "09-22", 0.96931, ["35b_rt", "397b"]),
    ("E4", "w47:53", "09-22", 0.96991, ["35b_rt", "397b"]),
    ("S2", "S2", "09-23", 0.96842, ["27b_lora", "397b"]),
    ("S3", "S3", "09-23", 0.96961, ["27b_lora", "35b_rt", "397b"]),
    ("S4", "S4", "09-23", 0.97110, ["27b_lora", "35b_rt", "397b", "9b_lora"]),
    ("T1", "T1", "09-23", 0.97170, ["27b_lora", "35b_rt", "397b", "9b_lora", "gemma"]),
    ("T2", "T2", "09-23", 0.97140, ["27b_lora", "35b_rt", "397b", "9b_lora", "gemma"]),
    ("T3", "T3", "09-23", 0.97319, ["r3", "35b_rt", "397b", "9b_lora", "gemma"]),
    ("T4", "T4", "09-23", 0.97348, ["r3", "35b_rt", "397b", "gemma"]),
    ("T5", "T5", "09-24", 0.97348, ["T4", "35b_shift"]),
    ("T6", "T6", "09-24", 0.97438, ["r3", "35b_rt", "397b", "gemma"]),
    ("T7", "T7", "09-24", 0.97408, ["T6", "35b_r2v3"]),
    ("T8A", "T8A", "09-24", 0.97438, ["T6", "35b_r2v3", "35b_fs16"]),
    ("T9", "T9", "09-24", 0.97378, ["T6", "T7"]),
    ("T10", "T10", "09-24", 0.97408, ["r3", "35b_rt", "35b_r2v3", "35b_fs16", "397b", "gemma"]),
    ("T11", "T11", "09-24", 0.97378, ["r3", "35b_rt", "397b", "gemma"]),
    ("T12", "T12", "09-24", 0.97438, ["T6", "35b_full"]),
    ("T13", "T13", "09-24", 0.97140, ["r3", "35b_r1v3", "397b", "gemma"]),
    ("T14", "T14", "09-24", 0.97378, ["r3", "35b_rt", "397b", "gemma_rot"]),
    ("T15", "T15", "09-24", 0.97467, ["T6", "q38"]),
    ("T16", "T16", "09-24", 0.97378, ["T14", "397b", "gemma", "35b_rt", "r3"]),
    ("T17", "T17", "09-25", 0.97408, ["T6", "9b_dora"]),
    ("T18", "T18", "09-25", 0.97497, ["T6", "q35"]),
    ("T19", "T19", "09-25", 0.97319, ["q35", "35b_rt", "397b", "gemma"]),
    ("T20", "T20", "09-25", 0.97438, ["T14", "q35"]),
    ("T21", "T21", "09-25", 0.97408, ["T16", "T18"]),
    ("T22", "T22", "09-25", 0.97438, ["T21", "T15", "9b_dora"]),
    ("T23", "T23", "09-25", 0.97408, ["T21", "35b_zs"]),
    ("T24", "T24", "09-25", 0.97497, ["T18", "T15"]),
    ("T25", "T25", "09-26", 0.97408, ["T22", "T11", "T19", "q35", "9b_dora"]),
    ("T26", "T26", "09-26", 0.97408, ["T22", "T11", "T19", "q35", "9b_dora"]),
    ("T27", "T27", "09-26", 0.97467, ["T24", "T22", "T19", "q35", "9b_dora"]),
    ("T28", "T28", "09-26", 0.97706, ["T22", "T9", "T14", "q35", "gemma"]),
    ("T29", "T29", "09-26", 0.97616, ["T26", "T27", "T24", "T28"]),
    ("T30", "T30", "09-26", 0.97616, ["T26", "T29"]),
]
FOCUS = ("T28", "T30")   # 최종 선택 2개
MILESTONES = {"E1", "E3", "T4", "T6", "T18", "T28", "T30"}
P_LO, P_HI = 0.94, 0.98


def ancestors(node, parents):
    seen, stack = set(), [node]
    while stack:
        for p in parents.get(stack.pop(), []):
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return seen


def build_svg():
    family = {m[0]: m[2] for m in MODELS}
    parents = {s[0]: s[4] for s in SUBS}
    lineage = set(FOCUS)
    for f in FOCUS:
        lineage |= ancestors(f, parents)

    nodes = [(m[0], m[1], m[3], None) for m in MODELS] + [(s[0], s[1], s[3], s[2]) for s in SUBS]
    gap_after_models = 34
    step = 30
    left = 60
    xs = {}
    x = left
    for i, (nid, *_rest) in enumerate(nodes):
        if i == len(MODELS):
            x += gap_after_models
        xs[nid] = x
        x += step
    width = x - step + left + 40
    base_y = 700
    bar_max = 90
    label_y = base_y + bar_max + 14
    height = label_y + 170

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'font-family="Pretendard, \'Noto Sans KR\', \'Malgun Gothic\', sans-serif">',
        f'<rect width="100%" height="100%" fill="{SURFACE}"/>',
        f'<text x="{left}" y="40" font-size="24" font-weight="700" fill="{INK}">모델 예측이 앙상블 제출로 엮인 계보</text>',
        f'<text x="{left}" y="66" font-size="14" fill="{INK2}">아크 = 구성원 → 제출. 색은 출발 모델 계열, 회색은 이전 제출을 다시 섞은 것. '
        f'진한 선은 최종 선택 T28·T30으로 이어진 계보. 축 아래 막대 = Public 점수({P_LO}~{P_HI}, 09-26 기준).</text>',
    ]

    # 아크: 흐린 것 먼저, 강조 계보를 위에
    edges = []
    for sid, _lab, _d, _p, members in SUBS:
        for m in members:
            edges.append((m, sid))
    edges.sort(key=lambda e: e[1] in lineage)
    for src, dst in edges:
        x1, x2 = xs[src], xs[dst]
        r = (x2 - x1) / 2
        h = min(r * 0.92, base_y - 110)
        color = FAMILY_COLOR[family[src]] if src in family else SUB_EDGE
        focus = dst in lineage
        opacity = 0.9 if focus else 0.16
        sw = 2 if focus else 1.2
        title = f"{src} → {dst}"
        out.append(
            f'<path d="M{x1},{base_y} A{r},{h} 0 0 1 {x2},{base_y}" fill="none" stroke="{color}" '
            f'stroke-width="{sw}" stroke-opacity="{opacity}"><title>{escape(title)}</title></path>'
        )

    # 축
    out.append(f'<line x1="{left - 16}" y1="{base_y}" x2="{width - 40}" y2="{base_y}" stroke="{RULE}" stroke-width="1"/>')

    # 구역 라벨
    split_x = (xs[MODELS[-1][0]] + xs[SUBS[0][0]]) / 2
    out.append(f'<line x1="{split_x}" y1="{base_y - 6}" x2="{split_x}" y2="{height - 20}" stroke="{RULE}" stroke-dasharray="3 4"/>')
    out.append(f'<text x="{xs[MODELS[0][0]]}" y="{height - 24}" font-size="13" fill="{MUTED}">개별 모델 예측 (a~d 확률 저장)</text>')
    out.append(f'<text x="{xs[SUBS[0][0]]}" y="{height - 24}" font-size="13" fill="{MUTED}">앙상블 제출 → 제출 순서</text>')

    # 날짜 구간 표시 (제출 쪽)
    prev = None
    for sid, _lab, day, _p, _m in SUBS:
        if day != prev:
            dx = xs[sid] - step / 2
            if prev is not None:
                out.append(f'<line x1="{dx}" y1="{base_y + 2}" x2="{dx}" y2="{base_y + bar_max}" stroke="{RULE}"/>')
            out.append(f'<text x="{dx + 4}" y="{base_y + bar_max}" font-size="11" fill="{MUTED}">{day}</text>')
            prev = day

    # 노드·막대·라벨
    for nid, label, public, _day in nodes:
        x = xs[nid]
        is_model = nid in family
        color = FAMILY_COLOR[family[nid]] if is_model else INK2
        in_line = nid in lineage
        if public is not None:
            bh = max(2, (public - P_LO) / (P_HI - P_LO) * bar_max)
            fill = color if is_model else (INK if nid in FOCUS else "#b5b4ad")
            out.append(
                f'<rect x="{x - 7}" y="{base_y + 3}" width="14" height="{bh:.1f}" rx="2" fill="{fill}">'
                f'<title>{escape(label)} Public {public:.5f}</title></rect>'
            )
        r = 6 if (in_line or is_model) else 4.5
        stroke = f' stroke="{INK}" stroke-width="2"' if nid in FOCUS else f' stroke="{SURFACE}" stroke-width="2"'
        out.append(f'<circle cx="{x}" cy="{base_y}" r="{r}" fill="{color}"{stroke}/>')
        weight = "700" if (nid in FOCUS or nid in MILESTONES) else "400"
        ink = INK if (in_line or is_model) else MUTED
        out.append(
            f'<text transform="translate({x + 4},{label_y}) rotate(60)" font-size="13" font-weight="{weight}" fill="{ink}">'
            f'{escape(label)}</text>'
        )
        if nid in MILESTONES and public is not None:
            out.append(
                f'<text x="{x}" y="{base_y - 12}" font-size="11" font-weight="700" fill="{INK}" text-anchor="middle">'
                f'{public:.5f}</text>'
            )

    # 범례
    lx, ly = left, 96
    for fam, col in FAMILY_COLOR.items():
        out.append(f'<rect x="{lx}" y="{ly - 10}" width="14" height="4" rx="2" fill="{col}"/>')
        out.append(f'<text x="{lx + 20}" y="{ly - 4}" font-size="13" fill="{INK2}">{escape(FAMILY_NAME[fam])}</text>')
        lx += 20 + 9 * len(FAMILY_NAME[fam]) + 26
    out.append(f'<rect x="{lx}" y="{ly - 10}" width="14" height="4" rx="2" fill="{SUB_EDGE}"/>')
    out.append(f'<text x="{lx + 20}" y="{ly - 4}" font-size="13" fill="{INK2}">이전 제출 재결합</text>')

    out.append("</svg>")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/figures/ensemble_lineage.svg")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(build_svg())
    print(args.out)


if __name__ == "__main__":
    main()
