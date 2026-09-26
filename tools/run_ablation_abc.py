# -*- coding: utf-8 -*-
"""
A/AB/ABC 데이터셋 동일조건 비교 러너 (2026-09-22).
데이터셋만 바꾸고 seed/base model/hyperparameter는 고정한다.
  A   = train_A.csv   (train_orig 6002)
  AB  = train_AB.csv  (+ dev_pseudo 692)
  ABC = train_ABC.csv (+ codex_regen 662)
평가셋(val667, gold_v3)은 학습에 들어가지 않는다(colab_lora_train.py의 canonical assert가 강제).

이 스크립트는 GPU 환경(Colab/RunPod)에서 실행한다. 로컬 CPU에서는 --dry로 명령만 출력.
사용:
  python tools/run_ablation_abc.py --data /content/data --out /content/drive/MyDrive/ai2_abc \
         --val-ids val667_ids.json --hard-gold hard_eval_gold_v3.json [--only A AB ABC] [--dry]
"""
from __future__ import annotations
import argparse, subprocess, sys, os
from pathlib import Path

OUT_DIR = Path(r'C:\ssafy\AI_2_CHALLENGE') / 'runs' / 'dev_validation_v1'
DATASETS = {
    'A':   'train_A.csv',
    'AB':  'train_AB.csv',
    'ABC': 'train_ABC.csv',
}
# 고정 하이퍼파라미터 (세 실험 공통)
FIXED = dict(model='Qwen/Qwen3.5-9B', r=8, alpha=16, lr=1e-4, epochs=1.0,
             grad_accum=16, seed=1, view='full')

def build_cmd(label, data, out, val_ids, hard_gold, train_csv, source_weight=None):
    cmd = [sys.executable, 'tools/colab_lora_train.py',
           '--model', FIXED['model'], '--data', data,
           '--val-ids', val_ids, '--hard-gold', hard_gold,
           '--train-csv', train_csv,
           '--out', out, '--tag', f'abc_{label}',
           '--r', str(FIXED['r']), '--alpha', str(FIXED['alpha']),
           '--lr', str(FIXED['lr']), '--epochs', str(FIXED['epochs']),
           '--grad-accum', str(FIXED['grad_accum']), '--seed', str(FIXED['seed']),
           '--view', FIXED['view'], '--skip-test']
    if source_weight:
        cmd += ['--source-weight', source_weight]
    return cmd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='압축 해제된 데이터 루트(train/,dev/,test/,*.csv)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--val-ids', required=True)
    ap.add_argument('--hard-gold', required=True)
    ap.add_argument('--csv-dir', default=str(OUT_DIR), help='train_A/AB/ABC.csv 위치')
    ap.add_argument('--only', nargs='*', default=['A','AB','ABC'])
    ap.add_argument('--source-weight', default=None, help="ABC 비율실험용. 예: 'train_orig:1,dev_pseudo:1,codex_regen:1'")
    ap.add_argument('--dry', action='store_true')
    args = ap.parse_args()

    for label in args.only:
        train_csv = os.path.join(args.csv_dir, DATASETS[label])
        # colab_lora_train.py 는 <out>/<tag> 에 저장한다. tag=abc_{label} → run_dir=<args.out>/abc_{label}
        run_dir = os.path.join(args.out, f'abc_{label}')
        sw = args.source_weight if label == 'ABC' else None
        cmd = build_cmd(label, args.data, args.out, args.val_ids, args.hard_gold, train_csv, sw)
        print('\n$', ' '.join(cmd), flush=True)
        if args.dry:
            continue
        subprocess.run(cmd, check=True)
        mcmd = [sys.executable, 'tools/ablation_metrics.py', '--run', run_dir, '--label', label,
                '--summary', os.path.join(args.csv_dir, 'ablation_summary.csv')]
        qt = os.path.join(args.csv_dir, 'question_types.csv')
        gv2 = os.path.join(args.csv_dir, 'hard_eval_gold_v2.json')
        if os.path.exists(qt):  mcmd += ['--qtypes', qt]
        if os.path.exists(gv2): mcmd += ['--gold-v2', gv2]
        subprocess.run(mcmd, check=True)
    print('\n완료. 결과 비교: runs/dev_validation_v1/ablation_summary.csv')

if __name__ == '__main__':
    main()
