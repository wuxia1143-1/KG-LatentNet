import csv
from collections import defaultdict

results = []
with open('results/tables/tuning/kg_retune_validation_results.csv', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        try:
            row['best_val_mae'] = float(row['best_val_mae'])
            row['candidate_id'] = int(row['candidate_id'])
            row['fold'] = int(row['fold'])
            results.append(row)
        except:
            pass

by_fold = defaultdict(list)
for r in results:
    if r.get('status') == 'ok':
        by_fold[r['fold']].append(r)

for fold in sorted(by_fold.keys()):
    rs = by_fold[fold]
    best = min(rs, key=lambda x: x['best_val_mae'])
    print(f'Fold {fold}: {len(rs)} candidates, best_mae={best["best_val_mae"]:.4f} (c{best["candidate_id"]}, {best["variant_mode"]})')

import os
if os.path.exists('results/tables/tuning/kg_retune_selected_params.csv'):
    print()
    print('=== SELECTED PER FOLD ===')
    with open('results/tables/tuning/kg_retune_selected_params.csv', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            print(f'Fold {row["fold"]}: variant={row["variant_mode"]} mae={float(row["best_val_mae"]):.4f}')

if os.path.exists('configs/locked_kg_retune_config.yaml'):
    print()
    print('=== LOCKED CONFIG EXISTS ===')
    print(open('configs/locked_kg_retune_config.yaml').read()[:500])
