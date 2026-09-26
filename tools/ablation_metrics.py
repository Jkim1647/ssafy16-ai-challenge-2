# -*- coding: utf-8 -*-
"""
A/AB/ABC 학습 실험 결과 집계 (2026-09-22).
colab_lora_train.py 가 낸 run 디렉토리(val_predictions.jsonl, hard_predictions.jsonl,
meta.json)를 읽어 다음을 계산·기록한다.
  - val667 accuracy / gold_v3 accuracy / gold_v3 H184 accuracy
  - answer별 a/b/c/d accuracy
  - auto_type별 accuracy + OCR/text-heavy vs visual
  - source별 학습 건수 (meta.args.train_csv 기준)
  - best checkpoint / train·val loss / 학습 시간
사용: python tools/ablation_metrics.py --run <RUN_DIR> --label A
동일 --summary(기본 runs/dev_validation_v1/ablation_summary.csv)에 한 줄씩 append.
"""
from __future__ import annotations
import pandas as pd, json, argparse, os
from pathlib import Path

# 로컬(Windows) 기본 경로. Colab 등에서는 --qtypes/--gold-v2/--summary로 덮어쓴다.
ROOT = Path(r'C:\ssafy\AI_2_CHALLENGE')
OUT  = ROOT / 'runs' / 'dev_validation_v1'
TEXT_HEAVY = {'PRICE','TIMEDATE','NUMBER','SPELLING','TABLE','LOGO','COMPOSITE'}

def _first(*cands):
    for c in cands:
        if c and Path(c).exists():
            return str(c)
    return str(cands[-1])

def stem(s): return Path(str(s)).stem

def load_jsonl(p):
    rows = [json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]
    return pd.DataFrame(rows)

def acc(df):
    m = df[df.gold.notna()]
    return round((m.pred == m.gold).mean(), 4), len(m)

def breakdown(df, qt):
    df = df.copy()
    df['cid'] = df['id'].map(stem)
    df['correct'] = (df.pred == df.gold).astype(int)
    per_ans = {a: round(df[df.gold==a]['correct'].mean(), 4) if (df.gold==a).any() else None for a in 'abcd'}
    if qt is None:  # question_types 미탑재 → 유형 분해 생략(예측 jsonl로 사후 계산 가능)
        return per_ans, None, None
    df = df.merge(qt, on='cid', how='left')
    per_type = {t: [round(g['correct'].mean(),4), len(g)] for t, g in df.groupby('auto_type')}
    th = df[df.auto_type.isin(TEXT_HEAVY)]
    vis = df[~df.auto_type.isin(TEXT_HEAVY)]
    grp = {'text_heavy': [round(th['correct'].mean(),4), len(th)] if len(th) else None,
           'visual':     [round(vis['correct'].mean(),4), len(vis)] if len(vis) else None}
    return per_ans, per_type, grp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='colab_lora_train 출력 디렉토리')
    ap.add_argument('--label', required=True, choices=['A','AB','ABC'], help='실험 라벨')
    ap.add_argument('--qtypes', default=None, help='question_types.csv 경로')
    ap.add_argument('--gold-v2', default=None, help='hard_eval_gold_v2.json (H184) 경로')
    ap.add_argument('--summary', default=None)
    args = ap.parse_args()
    run = Path(args.run)
    qtypes = _first(args.qtypes, ROOT/'data_meta'/'question_types.csv', 'question_types.csv')
    goldv2 = _first(args.gold_v2, OUT/'hard_eval_gold_v2.json', 'hard_eval_gold_v2.json')
    summary = args.summary or _first(OUT/'ablation_summary.csv', 'ablation_summary.csv')
    meta = json.loads((run / 'meta.json').read_text(encoding='utf-8'))
    if Path(qtypes).exists():
        qt = pd.read_csv(qtypes, encoding='utf-8-sig').rename(columns={'row_id':'cid'})[['cid','auto_type']]
        qt['cid'] = qt['cid'].map(stem)
    else:
        qt = None
        print('[warn] question_types 없음 → 유형별 정확도는 예측 jsonl로 사후 계산')

    val = load_jsonl(run / 'val_predictions.jsonl')
    val_acc, val_n = acc(val)
    v_ans, v_type, v_grp = breakdown(val, qt)

    gold_acc = gold_n = h184_acc = h184_n = None
    g_ans = g_grp = g_type = None
    hp = run / 'hard_predictions.jsonl'
    if hp.exists():
        gold = load_jsonl(hp)
        gold_acc, gold_n = acc(gold)
        g_ans, g_type, g_grp = breakdown(gold, qt)
        h184_ids = set(json.loads(Path(goldv2).read_text(encoding='utf-8'))) if Path(goldv2).exists() else set()
        gh = gold[gold['id'].isin(h184_ids)]
        if len(gh): h184_acc, h184_n = acc(gh)

    # source 건수
    src = {}
    tcsv = meta.get('args',{}).get('train_csv')
    if tcsv and os.path.exists(tcsv):
        t = pd.read_csv(tcsv, encoding='utf-8-sig')
        if 'source' in t.columns: src = t['source'].value_counts().to_dict()
    # best/loss
    hist = meta.get('history', [])
    losses = [h['loss'] for h in hist if 'loss' in h]
    val_hist = [(h['step'], h['val_acc']) for h in hist if 'val_acc' in h]
    best_ckpt = max(val_hist, key=lambda x: x[1]) if val_hist else ('final', val_acc)

    rec = dict(
        label=args.label, tag=meta.get('tag'), model=meta.get('model'),
        val667_acc=val_acc, val_n=val_n,
        gold_v3_acc=gold_acc, gold_n=gold_n, h184_acc=h184_acc, h184_n=h184_n,
        val_per_answer=v_ans, val_text_vs_visual=v_grp,
        gold_per_answer=g_ans, gold_text_vs_visual=g_grp,
        val_per_type=v_type, gold_per_type=g_type,
        source_counts=src, fit=meta.get('fit'),
        best_ckpt_step=best_ckpt[0], best_ckpt_valacc=best_ckpt[1],
        final_loss=round(losses[-1],4) if losses else None,
        train_sec=meta.get('train_sec'),
        seed=meta.get('args',{}).get('seed'), lr=meta.get('args',{}).get('lr'),
        r=meta.get('args',{}).get('r'), view=meta.get('args',{}).get('view'),
    )
    (run / 'metrics.json').write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding='utf-8')
    # summary CSV append (사람이 볼 산출물 → utf-8-sig)
    flat = {k: (json.dumps(v, ensure_ascii=False) if isinstance(v,(dict,list)) else v) for k,v in rec.items()}
    df = pd.DataFrame([flat])
    sp = Path(summary)
    if sp.exists():
        df = pd.concat([pd.read_csv(sp, encoding='utf-8-sig'), df], ignore_index=True)
    df.to_csv(sp, index=False, encoding='utf-8-sig')
    print(f"[{args.label}] val667 {val_acc} | gold_v3 {gold_acc} | H184 {h184_acc} | fit {meta.get('fit')} | {meta.get('train_sec')}s")
    print('  source:', src)
    print('  -> metrics.json, ', sp)

if __name__ == '__main__':
    main()
