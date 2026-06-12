"""Calibrate the demo's cause-folding threshold against the LLM taxonomy.

The demo (demo/app_gradio.py) collapses near-duplicate free-text causes in each
query's candidate pool by agglomerative clustering (pool-centered cosine, average
linkage) cut at a single distance threshold `fold_thresh`. Eyeballing one pool is
not a defensible way to pick the cut, so we calibrate it ONCE, offline, against the
project's own LLM cause taxonomy (cause_clusters_llm.json) and freeze the result —
the LLM is never touched at inference.

Method (route A): sample N train cases as leave-one-out queries against the GROD
soft bank, build each query's candidate pool exactly as the demo does (top-k
neighbours' union of cause indices), then for a grid of cuts compare the
agglomerative grouping to the LLM cluster grouping of the *same* pool causes via
Adjusted Rand Index. Pick the cut with the highest mean ARI over pools with ≥2
distinct LLM clusters (single-cluster pools are uninformative for ARI).

Writes `fold_thresh` (+ the ARI curve) into data/processed/current/thresholds.json,
preserving the existing objectness keys. The demo reads it as its FOLD_THRESH default.

Run from repo root (SDM env):
    $PY -m diagnosis_model.grod.calibrate_fold_threshold
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import adjusted_rand_score

REPO_ROOT = Path(__file__).resolve().parents[2]


def _agglo_labels(cand_embs, cut):
    """Pool-centered cosine, average linkage, cut at `cut` (same as the demo)."""
    P = cand_embs.size(0)
    if P <= 1:
        return np.zeros(P, dtype=int)
    c = F.normalize(cand_embs - cand_embs.mean(0, keepdim=True), dim=-1).cpu().numpy()
    Z = linkage(c, method="average", metric="cosine")
    return fcluster(Z, t=cut, criterion="distance")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_db_dir", default="data/processed/current/artifacts/db/case_db_jointDistRawP")
    ap.add_argument("--bank", default="data/processed/current/artifacts/models/encoder_grod_soft/bank_z_soft.pt")
    ap.add_argument("--clusters_json", default="data/processed/current/artifacts/cause_clusters_llm.json")
    ap.add_argument("--out", default="data/processed/current/thresholds.json")
    ap.add_argument("--n_queries", type=int, default=400)
    ap.add_argument("--top_k_cases", type=int, default=20)
    ap.add_argument("--cut_grid", default="0.30:1.01:0.05",
                    help="start:stop:step (stop exclusive)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    db = Path(args.case_db_dir)
    bank_z = torch.load(args.bank, weights_only=False)["bank_z"].float()
    tc = torch.load(db / "train_cases.pt", weights_only=False)
    cte = torch.load(db / "cause_text_embs.pt", weights_only=False)
    cause_embs = cte["embeddings"].float()
    cause_texts = cte["texts"]
    memb = [c["cause_emb_indices"] for c in tc]

    o2c = json.load(open(args.clusters_json))["original_to_cause_id"]
    # global cause idx -> LLM cluster id; -1 = not in the taxonomy. The taxonomy is
    # built from a (possibly older) cause corpus and may not cover every current
    # case_db cause, so unmapped causes are EXCLUDED from the ARI (not lumped into a
    # spurious single cluster, which would bias the cut toward over-merging).
    cause_cluster = np.array([o2c.get(t, -1) for t in cause_texts])
    cov = (cause_cluster >= 0).mean()
    print(f"[taxonomy] {cov * 100:.1f}% of {len(cause_texts)} causes mapped "
          f"(unmapped excluded from ARI)")

    start, stop, step = (float(x) for x in args.cut_grid.split(":"))
    cuts = np.round(np.arange(start, stop, step), 4)

    bank_n = F.normalize(bank_z, dim=-1)
    rng = np.random.default_rng(args.seed)
    qids = rng.choice(len(tc), size=min(args.n_queries, len(tc)), replace=False)

    per_cut = {float(c): [] for c in cuts}
    n_used = 0
    for q in qids:
        sims = (bank_n[q:q + 1] @ bank_n.t())[0].clone()
        sims[q] = -1
        topi = sims.topk(args.top_k_cases).indices.tolist()
        pool = sorted({i for ci in topi for i in memb[ci]})
        if len(pool) < 2:
            continue
        ref = cause_cluster[pool]
        mapped = ref >= 0                 # drop causes absent from the taxonomy
        if mapped.sum() < 2 or len(set(ref[mapped].tolist())) < 2:
            continue                      # ARI uninformative on a single-cluster pool
        cand_embs = cause_embs[torch.tensor(pool)]
        for c in cuts:
            # cluster the FULL pool (demo-faithful), score ARI only on mapped causes
            lab = _agglo_labels(cand_embs, float(c))
            per_cut[float(c)].append(adjusted_rand_score(ref[mapped], lab[mapped]))
        n_used += 1

    curve = {f"{c:.2f}": float(np.mean(v)) for c, v in per_cut.items()}
    best_cut = max(curve, key=curve.get)
    print(f"使用 {n_used} 個有效 query 池（≥2 LLM clusters）, top_k={args.top_k_cases}")
    print("cut   mean_ARI")
    for c, a in curve.items():
        mark = "  <-- best" if c == best_cut else ""
        print(f"{c}   {a:.4f}{mark}")

    out = Path(args.out)
    payload = json.load(open(out)) if out.exists() else {}
    payload["fold_thresh"] = float(best_cut)
    payload["fold"] = {"mean_ARI": curve[best_cut], "n_pools": n_used,
                       "ARI_curve": curve, "top_k_cases": args.top_k_cases}
    payload["fold_note"] = ("demo cause-folding cut (agglomerative, pool-centered cosine, "
                            "average linkage). Calibrated by max mean-ARI vs cause_clusters_llm "
                            "on LOO train-query pools. LLM used offline only.")
    json.dump(payload, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"\n寫入 {out}: fold_thresh = {best_cut} (mean ARI {curve[best_cut]:.4f})")


if __name__ == "__main__":
    main()
