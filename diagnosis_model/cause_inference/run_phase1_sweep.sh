#!/bin/bash
# Phase 1 hyperparameter sweep — find ceiling of zero-training retrieval.
set -euo pipefail

PY=/home/lab603/anaconda3/envs/yj_py/bin/python
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

CASE_DB=diagnosis_model/cause_inference/outputs/case_db
OUT_BASE=diagnosis_model/cause_inference/outputs/phase1_sweep

mkdir -p "$OUT_BASE"

run_one () {
  local tag=$1; shift
  local out="$OUT_BASE/$tag"
  if [[ -f "$out/metrics.json" ]]; then
    echo "[skip] $tag (metrics.json exists)"
    return
  fi
  echo "==== $tag ===="
  "$PY" -m diagnosis_model.cause_inference.phase1_baseline \
    --case_db_dir "$CASE_DB" \
    --output_dir "$out" \
    "$@" 2>&1 | tail -25
}

# 1) baseline reference (already have phase1_full but rerun for parity)
run_one base_K10_a25_d95   --top_k_cases 10 --alpha_global 0.25 --beta_lesion 0.75 --diversify_threshold 0.95
# 2) larger K
run_one K20_a25_d95        --top_k_cases 20 --alpha_global 0.25 --beta_lesion 0.75 --diversify_threshold 0.95
run_one K30_a25_d95        --top_k_cases 30 --alpha_global 0.25 --beta_lesion 0.75 --diversify_threshold 0.95
run_one K50_a25_d95        --top_k_cases 50 --alpha_global 0.25 --beta_lesion 0.75 --diversify_threshold 0.95
# 3) global vs lesion ablation
run_one K10_a00_d95_lesion --top_k_cases 10 --alpha_global 0.0  --beta_lesion 1.0  --diversify_threshold 0.95
run_one K10_a10_d95_global --top_k_cases 10 --alpha_global 1.0  --beta_lesion 0.0  --diversify_threshold 0.95
# 4) diversify threshold
run_one K10_a25_d97        --top_k_cases 10 --alpha_global 0.25 --beta_lesion 0.75 --diversify_threshold 0.97
run_one K10_a25_d99        --top_k_cases 10 --alpha_global 0.25 --beta_lesion 0.75 --diversify_threshold 0.99

echo
echo "==== sweep summary ===="
"$PY" -c "
import json, glob
from pathlib import Path
rows = []
for d in sorted(glob.glob('$OUT_BASE/*/')):
    p = Path(d) / 'metrics.json'
    if not p.exists(): continue
    m = json.load(p.open())
    cfg = m.get('config', {})
    s = m['metrics']['semantic']
    rows.append({
        'tag': Path(d).name,
        'K': cfg.get('top_k_cases'),
        'alpha': cfg.get('alpha_global'),
        'div': cfg.get('diversify_threshold'),
        'pool': m['metrics']['candidate_pool']['mean_size'],
        'cov': s['coverage'],
        'MRR': s['MRR'],
        'R@1': s['R@1'],
        'R@5': s['R@5'],
        'R@10': s['R@10'],
        'R@20': s['R@20'],
    })
print(f'{\"tag\":<28}{\"K\":>4}{\"α\":>6}{\"div\":>5}  pool   cov    MRR    R@1    R@5   R@10   R@20')
for r in rows:
    print(f'{r[\"tag\"]:<28}{r[\"K\"]:>4}{r[\"alpha\"]:>6.2f}{r[\"div\"]:>5.2f}  {r[\"pool\"]:>4.0f}  {r[\"cov\"]:.3f}  {r[\"MRR\"]:.3f}  {r[\"R@1\"]:.3f}  {r[\"R@5\"]:.3f}  {r[\"R@10\"]:.3f}  {r[\"R@20\"]:.3f}')
"
