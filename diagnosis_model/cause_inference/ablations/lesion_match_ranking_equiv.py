"""Phase 1 retrieval-ranking equivalence between Hungarian and max_mean.

Paper claim being validated: although Hungarian (one-to-one LP) and max_mean
(many-to-one soft) differ in per-pair score, the *induced case ranking* — the
only signal Phase 1 retrieval consumes downstream — is preserved. Therefore
choosing max_mean for deployability (vectorized GPU scatter_reduce, linear
scaling in train-bank size) costs nothing methodologically.

For each valid query against the production fish train bank:
  - sims_h, sims_m : full [n_train] similarity vectors under each operator
  - top-K overlap  : | argsort(-sims_h)[:K] ∩ argsort(-sims_m)[:K] | / K
  - Kendall-tau    : rank correlation across the full bank
  - score gap      : (sims_m - sims_h) distribution moments
  - cause-set Jaccard @ K : | causes(top_K_h) ∩ causes(top_K_m) |
                          / | causes(top_K_h) ∪ causes(top_K_m) |
    This is the "absorption" measurement — Phase 1 downstream consumes the
    union of cause indices from retrieved top-K cases (the candidate pool),
    not case identity. High Jaccard explains why case-level top-K overlap
    can be ~0.84 while final R@K stays within 0.25 pp across operators.

Outputs JSON metrics + score-gap histogram PNG.

Usage:
    PY=/home/lab603/anaconda3/envs/SDM/bin/python
    $PY -m diagnosis_model.cause_inference.ablations.lesion_match_ranking_equiv \\
      --case_db_dir diagnosis_model/cause_inference/outputs/case_db_raw \\
      --output_dir diagnosis_model/cause_inference/outputs/ablations/lesion_match_ranking_equiv
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from diagnosis_model.cause_inference.phase1_baseline import (
    MISS_RANK,
    add_recall_at_ks,
    build_candidate_pool,
    compute_case_similarities,
    load_cases,
    load_train_bank,
    offsets_to_case_ids,
    score_candidates,
    select_positive_top_cases,
    summarize_rank_metric,
)


def top_k_overlap(rank_a: np.ndarray, rank_b: np.ndarray, k: int) -> float:
    if k <= 0 or len(rank_a) == 0:
        return 0.0
    k = min(k, len(rank_a))
    return len(set(rank_a[:k].tolist()) & set(rank_b[:k].tolist())) / k


def semantic_ranks_for_gt(
    cand_scores: torch.Tensor,
    candidate_indices: List[int],
    gt_cause_idx: List[int],
    cause_table_embs: torch.Tensor,
    sem_threshold: float,
    device: str,
) -> List[float]:
    """Per GT cause, return semantic rank (1-indexed) within raw candidate-pool
    ranking. Semantic match = cosine to GT cause embedding ≥ threshold. Misses
    → MISS_RANK (mirrors phase1_baseline.main() per-GT semantic rank logic)."""
    if not candidate_indices:
        return [MISS_RANK] * len(gt_cause_idx)
    cand_t = torch.tensor(candidate_indices, device=device, dtype=torch.long)
    cand_embs = cause_table_embs.index_select(0, cand_t)
    raw_sorted_local = torch.argsort(cand_scores, descending=True).detach().cpu().numpy()
    sorted_cand_embs = cand_embs[torch.from_numpy(raw_sorted_local).to(device)]
    gt_t = torch.tensor(gt_cause_idx, device=device, dtype=torch.long)
    gt_embs = cause_table_embs.index_select(0, gt_t)
    cos = gt_embs @ sorted_cand_embs.T  # [G, pool_size]
    sem_match = cos >= sem_threshold
    out: List[float] = []
    for g_i in range(sem_match.size(0)):
        hits = torch.nonzero(sem_match[g_i], as_tuple=False)
        if hits.numel() > 0:
            out.append(float(int(hits[0].item()) + 1))
        else:
            out.append(MISS_RANK)
    return out


def cause_set_jaccard(rank_a: np.ndarray, rank_b: np.ndarray, k: int,
                      train_cases: list) -> float:
    """Jaccard of candidate-pool cause indices between two top-K case lists.
    Mirrors `phase1_baseline.build_candidate_pool` semantics (union of
    `cause_emb_indices` across retrieved cases, no dedup-by-case needed
    since indices are integers)."""
    if k <= 0 or len(rank_a) == 0:
        return float("nan")
    k_a = min(k, len(rank_a))
    k_b = min(k, len(rank_b))
    causes_a: set = set()
    for i in rank_a[:k_a].tolist():
        causes_a.update(int(c) for c in train_cases[int(i)]["cause_emb_indices"])
    causes_b: set = set()
    for i in rank_b[:k_b].tolist():
        causes_b.update(int(c) for c in train_cases[int(i)]["cause_emb_indices"])
    union = causes_a | causes_b
    if not union:
        return float("nan")
    return len(causes_a & causes_b) / len(union)


def kendall_tau_fast(x: np.ndarray, y: np.ndarray, n_pairs: int = 200_000,
                     rng: np.random.Generator | None = None) -> float:
    """Sampled Kendall-tau. Full O(N^2) is 12780^2 = 163M pairs per query;
    sampling 200k random pairs gives stderr ~0.003 which is fine here."""
    n = len(x)
    if n < 2:
        return float("nan")
    if rng is None:
        rng = np.random.default_rng(0)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    dx = x[i] - x[j]
    dy = y[i] - y[j]
    concordant = (dx * dy > 0).sum()
    discordant = (dx * dy < 0).sum()
    total = concordant + discordant
    return float((concordant - discordant) / total) if total > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_db_dir", type=str,
                    default="diagnosis_model/cause_inference/outputs/case_db_raw")
    ap.add_argument("--output_dir", type=str,
                    default="diagnosis_model/cause_inference/outputs/ablations/lesion_match_ranking_equiv")
    ap.add_argument("--alpha_global", type=float, default=0.25)
    ap.add_argument("--beta_lesion", type=float, default=0.75)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 10, 20, 50, 100])
    ap.add_argument("--top_k_cases", type=int, default=20,
                    help="Phase 1 canonical config — top retrieved cases that feed candidate pool.")
    ap.add_argument("--semantic_threshold", type=float, default=0.95,
                    help="Phase 1 canonical config — cosine ≥ this counts as semantic GT match.")
    ap.add_argument("--metric_ks", type=int, nargs="+", default=[1, 5, 10, 20],
                    help="Recall@K levels reported for the end-to-end retrieval metric.")
    ap.add_argument("--max_queries", type=int, default=-1)
    ap.add_argument("--tau_pairs", type=int, default=200_000,
                    help="Random pairs sampled per query for Kendall-tau (full N^2 too slow).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    case_db_dir = Path(args.case_db_dir)
    device = args.device

    print(f"[load] case_db_dir={case_db_dir} device={device}")
    train_cases, train_global_stack, train_lesion_stack, train_offsets = load_train_bank(
        case_db_dir, device,
    )
    valid_cases = load_cases(case_db_dir, "valid")
    cause_pack = torch.load(case_db_dir / "cause_text_embs.pt", weights_only=False)
    cause_table_embs = F.normalize(cause_pack["embeddings"].to(device), dim=-1)
    queries = valid_cases if args.max_queries <= 0 else valid_cases[: args.max_queries]
    n_train = len(train_cases)
    print(f"[load] train={n_train}  valid_queries={len(queries)}  causes={cause_table_embs.size(0)}")

    train_case_ids = offsets_to_case_ids(train_offsets, train_lesion_stack.device)
    rng = np.random.default_rng(args.seed)

    overlap_per_k: Dict[int, List[float]] = {k: [] for k in args.ks}
    jaccard_per_k: Dict[int, List[float]] = {k: [] for k in args.ks}
    tau_per_q: List[float] = []
    gap_means: List[float] = []
    gap_stds: List[float] = []
    gap_abs_max: List[float] = []
    # Subsample raw gap values for histogram (full would be 12780 * 1573 = 20M floats)
    histogram_sample: List[float] = []
    hist_per_query = 1000  # keep 1000 gap samples per query

    # End-to-end Phase 1 metrics (per GT cause occurrence) under each match.
    # Mirrors phase1_baseline.main(): semantic rank (cosine ≥ threshold to GT
    # cause embedding) within raw candidate-pool ranking; miss = MISS_RANK.
    sem_ranks_h: List[float] = []
    sem_ranks_m: List[float] = []
    sem_cov_h: List[int] = []
    sem_cov_m: List[int] = []

    t0 = time.time()
    for qi, q in enumerate(queries):
        q_global = F.normalize(q["global_emb"].to(device), dim=-1)
        q_lesions = F.normalize(q["lesion_embs"].to(device), dim=-1)

        sims_h = compute_case_similarities(
            q_global, q_lesions,
            train_global_stack, train_lesion_stack, train_offsets,
            args.alpha_global, args.beta_lesion,
            lesion_match="hungarian",
        )
        sims_m = compute_case_similarities(
            q_global, q_lesions,
            train_global_stack, train_lesion_stack, train_offsets,
            args.alpha_global, args.beta_lesion,
            lesion_match="max_mean",
            train_case_ids=train_case_ids,
        )

        rank_h = np.argsort(-sims_h)
        rank_m = np.argsort(-sims_m)
        for k in args.ks:
            overlap_per_k[k].append(top_k_overlap(rank_h, rank_m, k))
            jaccard_per_k[k].append(cause_set_jaccard(rank_h, rank_m, k, train_cases))

        # End-to-end Phase 1 (canonical top_k_cases=20, semantic threshold=0.95)
        # under each match. Replicates phase1_baseline.main() per-query scoring.
        gt_cause_idx = [int(g) for g in q["cause_emb_indices"]]
        for sims, sem_ranks_acc, sem_cov_acc in (
            (sims_h, sem_ranks_h, sem_cov_h),
            (sims_m, sem_ranks_m, sem_cov_m),
        ):
            top_k_idx, top_k_w, _ = select_positive_top_cases(sims, args.top_k_cases)
            candidate_indices = build_candidate_pool(top_k_idx, train_cases)
            cand_scores = score_candidates(
                candidate_indices, top_k_idx, top_k_w, train_cases, cause_table_embs,
            )
            sem_ranks = semantic_ranks_for_gt(
                cand_scores, candidate_indices, gt_cause_idx,
                cause_table_embs, args.semantic_threshold, device,
            )
            for r in sem_ranks:
                sem_ranks_acc.append(r)
                sem_cov_acc.append(0 if r == MISS_RANK else 1)

        tau_per_q.append(kendall_tau_fast(sims_h, sims_m, args.tau_pairs, rng))

        gap = sims_m - sims_h
        gap_means.append(float(gap.mean()))
        gap_stds.append(float(gap.std()))
        gap_abs_max.append(float(np.abs(gap).max()))
        sample_idx = rng.choice(n_train, size=min(hist_per_query, n_train), replace=False)
        histogram_sample.extend(gap[sample_idx].tolist())

        if (qi + 1) % 50 == 0 or qi + 1 == len(queries):
            elapsed = time.time() - t0
            rate = (qi + 1) / max(elapsed, 1e-9)
            eta = (len(queries) - qi - 1) / max(rate, 1e-9)
            print(f"[eval] {qi+1}/{len(queries)}  rate={rate:.2f} q/s  ETA={eta/60:.1f} min")

    def summary(arr: List[float]) -> dict:
        a = np.asarray([v for v in arr if np.isfinite(v)], dtype=np.float64)
        if a.size == 0:
            return {"n": 0}
        return {
            "n": int(a.size),
            "mean": float(a.mean()),
            "std": float(a.std()),
            "median": float(np.median(a)),
            "min": float(a.min()),
            "max": float(a.max()),
            "p05": float(np.percentile(a, 5)),
            "p95": float(np.percentile(a, 95)),
        }

    # Recall@K and MRR per match (per-GT-cause aggregation, matches phase1_baseline)
    def phase1_block(ranks_list: List[float], cov_list: List[int]) -> dict:
        arr = np.asarray(ranks_list, dtype=np.float64)
        block = summarize_rank_metric(arr, cov_list)
        add_recall_at_ks(block, arr, args.metric_ks)
        return block

    phase1_h = phase1_block(sem_ranks_h, sem_cov_h)
    phase1_m = phase1_block(sem_ranks_m, sem_cov_m)
    phase1_delta = {}
    for k in [f"R@{k}" for k in args.metric_ks] + ["MRR", "coverage"]:
        if k in phase1_h and k in phase1_m:
            phase1_delta[k] = float(phase1_m[k] - phase1_h[k])

    metrics = {
        "config": vars(args),
        "n_train": n_train,
        "n_queries": len(queries),
        "ranking_equivalence": {
            "top_k_overlap": {f"@{k}": summary(overlap_per_k[k]) for k in args.ks},
            "cause_set_jaccard": {f"@{k}": summary(jaccard_per_k[k]) for k in args.ks},
            "kendall_tau_sampled": summary(tau_per_q),
        },
        "phase1_semantic": {
            "threshold": args.semantic_threshold,
            "top_k_cases": args.top_k_cases,
            "n_gt_occurrences": len(sem_ranks_h),
            "hungarian": phase1_h,
            "max_mean":  phase1_m,
            "delta_max_mean_minus_hungarian": phase1_delta,
        },
        "score_gap_max_mean_minus_hungarian": {
            "per_query_mean":   summary(gap_means),
            "per_query_std":    summary(gap_stds),
            "per_query_absmax": summary(gap_abs_max),
            "pooled_sample": {
                "n": len(histogram_sample),
                "mean": float(np.mean(histogram_sample)),
                "std":  float(np.std(histogram_sample)),
                "median": float(np.median(histogram_sample)),
                "p01":  float(np.percentile(histogram_sample, 1)),
                "p99":  float(np.percentile(histogram_sample, 99)),
            },
        },
    }

    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[save] metrics.json -> {out_dir}")

    # Histogram
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(histogram_sample, bins=80, density=True,
                color="#4C72B0", edgecolor="white", linewidth=0.4)
        ax.axvline(0.0, color="k", linestyle="--", linewidth=0.8)
        ax.set_xlabel("per-pair score gap: max_mean − hungarian")
        ax.set_ylabel("density")
        ax.set_title(
            f"Score gap distribution (n={len(histogram_sample):,} sampled pairs, "
            f"{len(queries):,} queries × {n_train:,} train)"
        )
        fig.tight_layout()
        fig.savefig(out_dir / "score_gap_hist.png", dpi=150)
        plt.close(fig)
        print(f"[save] score_gap_hist.png -> {out_dir}")
    except ImportError:
        print("[skip] matplotlib not available, no histogram saved")

    # Console summary
    print("\n=== Ranking-equivalence summary ===")
    for k in args.ks:
        s = metrics["ranking_equivalence"]["top_k_overlap"][f"@{k}"]
        j = metrics["ranking_equivalence"]["cause_set_jaccard"][f"@{k}"]
        print(f"  K={k:>3}  case-overlap mean={s['mean']:.4f} med={s['median']:.4f} p05={s['p05']:.4f}  "
              f"|  cause-Jaccard mean={j['mean']:.4f} med={j['median']:.4f} p05={j['p05']:.4f}")
    s = metrics["ranking_equivalence"]["kendall_tau_sampled"]
    print(f"  Kendall-tau       mean={s['mean']:.4f}  median={s['median']:.4f}  "
          f"p05={s['p05']:.4f}")
    sg = metrics["score_gap_max_mean_minus_hungarian"]
    print(f"  per-query mean gap   {sg['per_query_mean']['mean']:+.4f}  "
          f"std across queries = {sg['per_query_mean']['std']:.4f}")
    print(f"  per-query std  gap   {sg['per_query_std']['mean']:.4f}")
    print(f"  pooled pairs gap     mean={sg['pooled_sample']['mean']:+.4f}  "
          f"std={sg['pooled_sample']['std']:.4f}  "
          f"[p01, p99] = [{sg['pooled_sample']['p01']:+.4f}, "
          f"{sg['pooled_sample']['p99']:+.4f}]")

    print(f"\n=== Phase 1 semantic retrieval (top_k_cases={args.top_k_cases}, "
          f"threshold={args.semantic_threshold}) ===")
    p1 = metrics["phase1_semantic"]
    print(f"  GT cause occurrences: {p1['n_gt_occurrences']}")
    cols = ["coverage", "MRR"] + [f"R@{k}" for k in args.metric_ks]
    header = f"  {'method':<12} " + "  ".join(f"{c:>8}" for c in cols)
    print(header)
    for name, block in (("hungarian", p1["hungarian"]),
                        ("max_mean",  p1["max_mean"]),
                        ("Δ (mm-hu)", p1["delta_max_mean_minus_hungarian"])):
        row = f"  {name:<12} " + "  ".join(
            f"{block.get(c, float('nan')):>+8.4f}" if name.startswith("Δ")
            else f"{block.get(c, float('nan')):>8.4f}"
            for c in cols
        )
        print(row)


if __name__ == "__main__":
    main()
