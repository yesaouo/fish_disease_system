#!/usr/bin/env bash
# Phase 1 aggregation ablation — hungarian vs MaxSim variants on fish.
#
# Run from repo root. All three variants share canonical hyperparams from
# inference.txt [A]; only --lesion_match changes.
#
# A: hungarian              — one-to-one assignment, /max(N,M) penalty   (paper canonical, sem R@10 = 44.4%)
# B: max_mean               — bidirectional MaxSim, no size penalty       (existing, unreported at retrieval time)
# C: max_mean_normalized    — bidirectional MaxSim, /max(N,M) penalty     (new — A vs C isolates the aggregation operator)
set -euo pipefail

PY="${PY:-/home/lab603/anaconda3/envs/SDM/bin/python}"
CASE_DB="diagnosis_model/cause_inference/outputs/case_db"
CLUSTER_JSON="diagnosis_model/cause_inference/outputs/cause_clusters_llm.json"
OUT_ROOT="diagnosis_model/cause_inference/outputs/ablation_aggregation"

mkdir -p "$OUT_ROOT"

for variant in hungarian max_mean max_mean_normalized; do
    out="$OUT_ROOT/$variant"
    echo "=== $variant -> $out ==="
    $PY -m diagnosis_model.cause_inference.phase1_baseline \
        --case_db_dir       "$CASE_DB" \
        --output_dir        "$out" \
        --cluster_json      "$CLUSTER_JSON" \
        --top_k_cases       20 \
        --alpha_global      0.25 \
        --beta_lesion       0.75 \
        --lesion_match      "$variant" \
        --diversify_threshold 0.95 \
        --semantic_threshold  0.95
done

echo "=== done ==="
