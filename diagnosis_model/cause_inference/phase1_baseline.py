"""Phase 1 baseline: zero-training case-based cause retrieval + candidate-restricted cause scoring.

Pipeline per query (valid case):
  1. Combined case similarity vs every train case
       sim(q, c) = α · cos(q.global, c.global)
                 + β · lesion-set cosine score
  2. Top-K retrieved cases, restricted to positive similarity only
  3. Build CANDIDATE POOL: union of deduped cause embedding indices
     appearing in those positive-similarity top-K retrieved cases.
  4. Score each candidate c in the pool using Option A:
       score(c) = Σ_{j in top-K-positive} w_j · max_g cos(c, e_{j,g})
     where w_j is normalized over the retained positive retrieved cases.
  5. Raw ranking is used for metrics.
  6. Diversified ranking is used only for predicted_top_n inspection output.

Metrics:
  - exact: rank of each GT cause within the raw candidate-pool ranking.
           Misses are represented internally as +inf for metrics and as null
           in per_query_results.jsonl, so R@K never counts uncovered GTs.
  - semantic (cosine ≥ threshold): rank of first candidate semantically
           equivalent to GT within the raw candidate-pool ranking
  - cluster (HDBSCAN): rank of first candidate in the same cluster as each GT
                         cause occurrence within the raw candidate-pool ranking
  - coverage_*: per-GT-cause-occurrence fraction of "GT covered by pool"
                under each match type

Notes:
  - Embeddings are explicitly L2-normalized before dot products are used as cosine.
  - Retrieved train cases with similarity <= 0 are excluded from both candidate-pool
    construction and candidate scoring.
  - Cluster metric is computed per GT cause occurrence, matching exact and
    semantic metric denominators. Duplicate GT causes/clusters in a query each
    contribute one occurrence.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_case_db(case_db_dir: Path, query_split: str = "valid"):
    """Load train + query split cases. `query_split` selects which `<split>_cases.pt`
    is returned in the 2nd slot (default `valid` preserves prior behavior)."""
    if query_split not in ("valid", "test"):
        raise ValueError(f"query_split must be 'valid' or 'test', got {query_split!r}")
    train = torch.load(case_db_dir / "train_cases.pt", weights_only=False)
    queries = torch.load(case_db_dir / f"{query_split}_cases.pt", weights_only=False)
    cause = torch.load(case_db_dir / "cause_text_embs.pt", weights_only=False)
    meta = json.load((case_db_dir / "meta.json").open())
    return train, queries, cause, meta


def stack_train_lesions(train_cases) -> Tuple[torch.Tensor, List[int]]:
    """Concatenate all train lesion embs; return (stacked, per-case offsets)."""
    pieces = [c["lesion_embs"] for c in train_cases]
    offsets = [0]
    for p in pieces:
        offsets.append(offsets[-1] + p.size(0))
    return torch.cat(pieces, dim=0), offsets


# ---------------------------------------------------------------------------
# Case similarity
# ---------------------------------------------------------------------------

def hungarian_set_score(sim_block: np.ndarray) -> float:
    """sim_block: [N_q, M_c] cosine matrix → mean of matched cosines / max(N_q, M_c).

    Empty sides return 0. The /max(N,M) normalizer penalizes lesion-count mismatch.
    """
    n, m = sim_block.shape
    if n == 0 or m == 0:
        return 0.0
    if n == 1 or m == 1:
        return float(sim_block.max() / max(n, m))
    row, col = linear_sum_assignment(-sim_block)
    return float(sim_block[row, col].sum() / max(n, m))


def max_mean_set_score(sim_block: np.ndarray) -> float:
    """Symmetric max-mean: 0.5 * (mean_i max_j sim_ij + mean_j max_i sim_ij).

    Differentiable analog of Hungarian (used at training time). Not penalized by
    size mismatch — slight asymmetry vs Hungarian.
    """
    n, m = sim_block.shape
    if n == 0 or m == 0:
        return 0.0
    forward = float(sim_block.max(axis=1).mean())
    backward = float(sim_block.max(axis=0).mean())
    return 0.5 * (forward + backward)


_LESION_MATCH_FNS = {
    "hungarian": hungarian_set_score,
    "max_mean":  max_mean_set_score,
}


def compute_case_similarities(
    q_global: torch.Tensor,           # [D], already normalized
    q_lesions: torch.Tensor,          # [N_q, D], already normalized
    train_global_stack: torch.Tensor, # [n_train, D], already normalized
    train_lesion_stack: torch.Tensor, # [total_lesions, D], already normalized
    train_offsets: List[int],
    alpha: float,
    beta: float,
    lesion_match: str = "hungarian",
) -> np.ndarray:
    """Return [n_train] combined similarity scores.

    Because all embeddings are L2-normalized before this function is called,
    matrix multiplication is cosine similarity.
    """
    match_fn = _LESION_MATCH_FNS[lesion_match]
    g_sim = (q_global.unsqueeze(0) @ train_global_stack.T).squeeze(0)

    if q_lesions.size(0) == 0 or train_lesion_stack.size(0) == 0:
        les_sim = np.empty((q_lesions.size(0), train_lesion_stack.size(0)), dtype=np.float32)
    else:
        les_sim = (q_lesions @ train_lesion_stack.T).detach().cpu().numpy()

    n_train = len(train_offsets) - 1
    l_score = np.zeros(n_train, dtype=np.float32)
    for i in range(n_train):
        s, e = train_offsets[i], train_offsets[i + 1]
        if e > s:
            l_score[i] = match_fn(les_sim[:, s:e])
    return alpha * g_sim.detach().cpu().numpy() + beta * l_score


# ---------------------------------------------------------------------------
# Candidate-restricted cause scoring
# ---------------------------------------------------------------------------

def select_positive_top_cases(
    sims: np.ndarray,
    top_k_cases: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select top-K train cases with strictly positive similarity.

    Returns:
      top_k_idx: selected train-case indices
      top_k_w: normalized positive weights for scoring
      top_k_raw_w: raw positive similarities, kept for inspection output
    """
    ranked_idx = np.argsort(-sims)
    positive_idx = ranked_idx[sims[ranked_idx] > 0]
    top_k_idx = positive_idx[:top_k_cases]

    top_k_raw_w = sims[top_k_idx].astype(np.float32)
    if top_k_raw_w.size > 0:
        top_k_w = top_k_raw_w / (top_k_raw_w.sum() + 1e-8)
    else:
        top_k_w = np.empty(0, dtype=np.float32)

    return top_k_idx, top_k_w.astype(np.float32), top_k_raw_w


def build_candidate_pool(
    top_case_idx: np.ndarray,
    train_cases: list,
) -> List[int]:
    """Unique cause-table indices appearing in the retained retrieved cases.

    The order preserves first-seen order across retrieved cases.
    """
    seen: set = set()
    pool: List[int] = []
    for case_i in top_case_idx.tolist():
        for cidx in train_cases[int(case_i)]["cause_emb_indices"]:
            if cidx not in seen:
                seen.add(cidx)
                pool.append(int(cidx))
    return pool


def diversify(
    sorted_local: np.ndarray,    # local indices into candidate pool, sorted by score desc
    cand_embs: torch.Tensor,     # [pool_size, D]
    threshold: float,
) -> np.ndarray:
    """Greedy MMR-style dedup for output inspection only.

    Keep the highest-scored candidate, then suppress any later candidate whose
    cosine to any already-kept candidate is >= threshold.

    Returns kept local indices in score order. If threshold >= 1.0, returns
    sorted_local unchanged.
    """
    if sorted_local.size == 0 or threshold >= 1.0:
        return sorted_local

    kept_local: List[int] = [int(sorted_local[0])]
    kept_emb_rows: List[int] = [int(sorted_local[0])]

    for li in sorted_local[1:].tolist():
        e = cand_embs[int(li)]
        kept_t = cand_embs[torch.tensor(
            kept_emb_rows,
            device=cand_embs.device,
            dtype=torch.long,
        )]
        max_sim = (kept_t @ e).max().item()
        if max_sim < threshold:
            kept_local.append(int(li))
            kept_emb_rows.append(int(li))

    return np.array(kept_local, dtype=np.int64)


def score_candidates(
    candidate_indices: List[int],
    top_case_idx: np.ndarray,
    top_case_weights: np.ndarray,
    train_cases: list,
    cause_table_embs: torch.Tensor,
) -> torch.Tensor:
    """For each c in candidate_indices:

       score(c) = Σ_j w_j · max_g cos(emb[c], emb_{j,g})

    top_case_weights are expected to be non-negative and normalized.
    Returns [len(candidate_indices)] tensor on cause_table_embs' device.
    """
    device = cause_table_embs.device
    if not candidate_indices:
        return torch.zeros(0, device=device)

    cand_embs = cause_table_embs.index_select(
        0,
        torch.tensor(candidate_indices, device=device, dtype=torch.long),
    )  # [U_cand, D]

    score = torch.zeros(len(candidate_indices), device=device)

    for case_i, w in zip(top_case_idx.tolist(), top_case_weights.tolist()):
        case_cidx = train_cases[int(case_i)]["cause_emb_indices"]
        if not case_cidx:
            continue

        case_embs = cause_table_embs.index_select(
            0,
            torch.tensor(case_cidx, device=device, dtype=torch.long),
        )  # [G, D]

        sims = cand_embs @ case_embs.T  # [U_cand, G]
        score = score + float(w) * sims.max(dim=1).values

    return score


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

MISS_RANK = float("inf")


def summarize_rank_metric(
    ranks: np.ndarray,
    coverage_flags: List[int],
) -> dict:
    """Summarize occurrence-level ranking metrics with correct miss handling.

    Uncovered GT occurrences must not be encoded as pool_size + 1, because that
    makes R@K depend on candidate-pool size and can incorrectly count misses as
    hits for large K. Internally, misses are +inf; therefore:
      - R@K counts only finite ranks <= K.
      - MRR gives misses reciprocal rank 0.
      - mean/median rank are reported over covered occurrences only.
    """
    ranks = np.asarray(ranks, dtype=np.float64)
    finite_mask = np.isfinite(ranks)
    finite_ranks = ranks[finite_mask]

    reciprocal = np.zeros_like(ranks, dtype=np.float64)
    reciprocal[finite_mask] = 1.0 / finite_ranks

    return {
        "coverage": float(np.mean(coverage_flags)) if coverage_flags else 0.0,
        "MRR": float(reciprocal.mean()) if ranks.size else 0.0,
        "mean_rank": float(finite_ranks.mean()) if finite_ranks.size else None,
        "median_rank": float(np.median(finite_ranks)) if finite_ranks.size else None,
        "n_covered": int(finite_ranks.size),
        "n_missed": int(ranks.size - finite_ranks.size),
        "rank_meaning": (
            "mean_rank and median_rank are computed over covered occurrences only; "
            "misses contribute 0 to MRR and R@K."
        ),
    }


def add_recall_at_ks(metric_block: dict, ranks: np.ndarray, ks: List[int]) -> None:
    """Add R@K values. Misses are +inf, so they never satisfy rank <= K."""
    ranks = np.asarray(ranks, dtype=np.float64)
    for k in ks:
        metric_block[f"R@{k}"] = float((ranks <= k).mean()) if ranks.size else 0.0


def fmt_metric(value) -> str:
    """Pretty-print scalar metric values, including None for unavailable ranks."""
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_db_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--query_split", type=str, default="valid",
                    choices=["valid", "test"],
                    help="which split to use as the query set (default valid)")
    ap.add_argument("--top_k_cases", type=int, default=10)
    ap.add_argument("--top_n_causes", type=int, default=20)
    ap.add_argument("--alpha_global", type=float, default=0.25)
    ap.add_argument("--beta_lesion", type=float, default=0.75)
    ap.add_argument("--lesion_match", type=str, default="hungarian",
                    choices=["hungarian", "max_mean"],
                    help="lesion-set matching mode for case similarity")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_queries", type=int, default=-1)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 20, 50, 100])
    ap.add_argument("--semantic_threshold", type=float, default=0.95,
                    help="cosine threshold for treating two cause strings as semantically equivalent")
    ap.add_argument("--cluster_json", type=str,
                    default="diagnosis_model/cause_inference/outputs/cause_clusters_llm.json",
                    help="Cluster taxonomy JSON (raw_string -> cluster_id). "
                         "Paper main: cause_clusters_llm.json (LLM, 466 topics). "
                         "Paper baseline: cause_clusters_hdbscan.json (HDBSCAN, 100). "
                         "Set to '' to disable cluster-level metrics.")
    ap.add_argument("--diversify_threshold", type=float, default=0.95,
                    help="For predicted_top_n output only: suppress any candidate whose cosine "
                         "to a previously kept candidate exceeds this. "
                         "Set to 1.0 to disable diversification.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    train_cases, valid_cases, cause_pack, meta = load_case_db(
        Path(args.case_db_dir), query_split=args.query_split,
    )

    # Explicit normalization: all following dot products are cosine similarities.
    cause_table_embs = F.normalize(cause_pack["embeddings"].to(device), dim=-1)
    cause_texts = cause_pack["texts"]

    print(f"[load] train={len(train_cases)}  {args.query_split}={len(valid_cases)}  "
          f"unique_causes={len(cause_texts)}  dim={cause_table_embs.size(-1)}")

    # Optional: cluster-based evaluation.
    # Cluster metrics are computed per GT cause occurrence, matching exact/semantic.
    cluster_id_array: np.ndarray | None = None
    if args.cluster_json:
        with open(args.cluster_json, encoding="utf-8") as f:
            cl = json.load(f)
        o2c = cl["original_to_cause_id"]
        cluster_id_array = np.array(
            [int(o2c[t]) for t in cause_texts], dtype=np.int64,
        )
        n_clusters = len(set(cluster_id_array.tolist()))
        print(f"[cluster] loaded {n_clusters} clusters from {args.cluster_json}")

    train_global_stack = torch.stack([c["global_emb"] for c in train_cases]).to(device)
    train_global_stack = F.normalize(train_global_stack, dim=-1)

    train_lesion_stack, train_offsets = stack_train_lesions(train_cases)
    train_lesion_stack = F.normalize(train_lesion_stack.to(device), dim=-1)

    print(f"[stack] train_globals={tuple(train_global_stack.shape)}  "
          f"train_lesions={tuple(train_lesion_stack.shape)}")

    queries = valid_cases if args.max_queries <= 0 else valid_cases[: args.max_queries]
    print(f"[eval] queries={len(queries)}  K={args.top_k_cases}  N={args.top_n_causes}  "
          f"alpha={args.alpha_global}  beta={args.beta_lesion}  "
          f"positive_case_only=True")

    per_query_results: list = []
    pool_sizes: List[int] = []
    retained_case_counts: List[int] = []

    all_gt_ranks: List[float] = []          # exact-index rank; MISS_RANK for uncovered GT
    all_gt_sem_ranks: List[float] = []      # semantic-cosine rank; MISS_RANK for uncovered GT
    all_gt_cluster_ranks: List[float] = []  # cluster-level rank; MISS_RANK for uncovered GT

    cov_exact: List[int] = []             # 1 if GT exactly in raw candidate pool
    cov_semantic: List[int] = []          # 1 if any raw pool cand >= threshold to GT
    cov_cluster: List[int] = []           # 1 if any raw pool cand in same cluster

    all_top1_max_cos: List[float] = []    # max cos(raw top-1 pred, any GT) per query

    t0 = time.time()
    for qi, q in enumerate(queries):
        q_global = F.normalize(q["global_emb"].to(device), dim=-1)
        q_lesions = F.normalize(q["lesion_embs"].to(device), dim=-1)

        sims = compute_case_similarities(
            q_global, q_lesions,
            train_global_stack, train_lesion_stack, train_offsets,
            args.alpha_global, args.beta_lesion,
            lesion_match=args.lesion_match,
        )

        # Only positive-similarity retrieved cases are allowed into candidate generation/scoring.
        top_k_idx, top_k_w, top_k_raw_w = select_positive_top_cases(
            sims=sims,
            top_k_cases=args.top_k_cases,
        )
        retained_case_counts.append(int(len(top_k_idx)))

        # Build candidate pool from the retained positive-similarity cases only.
        candidate_indices = build_candidate_pool(top_k_idx, train_cases)
        pool_size = len(candidate_indices)
        pool_sizes.append(pool_size)

        # Score the raw candidate pool only.
        cand_scores = score_candidates(
            candidate_indices, top_k_idx, top_k_w, train_cases, cause_table_embs,
        )  # [pool_size]

        if pool_size == 0:
            raw_sorted_local = np.empty(0, dtype=np.int64)
            raw_sorted_global = np.empty(0, dtype=np.int64)
            div_sorted_local = raw_sorted_local
            div_sorted_global = raw_sorted_global
            cand_embs = torch.empty(0, cause_table_embs.size(-1), device=device)
        else:
            cand_embs = cause_table_embs.index_select(
                0,
                torch.tensor(candidate_indices, device=device, dtype=torch.long),
            )

            # Raw sorted ranking: this is the ranking used for all metrics.
            raw_sorted_local = torch.argsort(cand_scores, descending=True).detach().cpu().numpy()
            raw_sorted_global = np.array(candidate_indices)[raw_sorted_local]

            # Diversified ranking: output inspection only, not used for metrics.
            div_sorted_local = diversify(raw_sorted_local, cand_embs, args.diversify_threshold)
            div_sorted_global = np.array(candidate_indices)[div_sorted_local]

        gt_cause_idx = q["cause_emb_indices"]

        # ---- Exact-index rank within raw pool ranking ----
        # Store misses as null in per_query_results and as +inf internally.
        # This prevents R@K from counting uncovered GTs as hits when K exceeds
        # the candidate-pool size.
        global_to_pool_pos = {int(g): i for i, g in enumerate(raw_sorted_global.tolist())}
        gt_ranks_local: List[int | None] = []
        for g in gt_cause_idx:
            pos = global_to_pool_pos.get(int(g))
            if pos is None:
                gt_ranks_local.append(None)
                all_gt_ranks.append(MISS_RANK)
                cov_exact.append(0)
            else:
                rank = int(pos) + 1
                gt_ranks_local.append(rank)
                all_gt_ranks.append(float(rank))
                cov_exact.append(1)

        # ---- Semantic-cosine rank within raw pool ranking ----
        gt_sem_ranks_local: List[int | None] = []
        if pool_size == 0:
            for _ in gt_cause_idx:
                gt_sem_ranks_local.append(None)
                all_gt_sem_ranks.append(MISS_RANK)
                cov_semantic.append(0)
        else:
            gt_idx_t = torch.tensor(gt_cause_idx, device=device, dtype=torch.long)
            gt_embs = cause_table_embs.index_select(0, gt_idx_t)  # [G, D]

            raw_sorted_cand_embs = cand_embs[
                torch.from_numpy(raw_sorted_local).to(device)
            ]  # [pool_size, D]

            cos_sorted = gt_embs @ raw_sorted_cand_embs.T  # [G, pool_size]
            sem_match = cos_sorted >= args.semantic_threshold

            for g_i in range(sem_match.size(0)):
                hits = torch.nonzero(sem_match[g_i], as_tuple=False)
                if hits.numel() > 0:
                    rank = int(hits[0].item()) + 1
                    gt_sem_ranks_local.append(rank)
                    all_gt_sem_ranks.append(float(rank))
                    cov_semantic.append(1)
                else:
                    gt_sem_ranks_local.append(None)
                    all_gt_sem_ranks.append(MISS_RANK)
                    cov_semantic.append(0)

        # ---- Cluster rank within raw pool ranking ----
        # Per GT cause occurrence, matching exact-index and semantic-cosine metrics.
        # If multiple GT causes map to the same cluster, each occurrence contributes
        # one rank/coverage item. This intentionally does NOT deduplicate clusters
        # within a query.
        gt_cluster_ranks_local: List[int | None] = []
        gt_clusters_local: List[int] = []
        if cluster_id_array is not None:
            raw_sorted_clusters = cluster_id_array[raw_sorted_global] if pool_size > 0 \
                                  else np.empty(0, dtype=np.int64)
            for g in gt_cause_idx:
                cid = int(cluster_id_array[int(g)])
                hits = np.flatnonzero(raw_sorted_clusters == cid) if pool_size > 0 \
                       else np.empty(0, dtype=np.int64)
                if hits.size > 0:
                    rank = int(hits[0]) + 1
                    gt_cluster_ranks_local.append(rank)
                    all_gt_cluster_ranks.append(float(rank))
                    cov_cluster.append(1)
                else:
                    gt_cluster_ranks_local.append(None)
                    all_gt_cluster_ranks.append(MISS_RANK)
                    cov_cluster.append(0)
                gt_clusters_local.append(cid)

        # ---- Raw top-1 max-cos diagnostic ----
        if pool_size > 0:
            top1_global = int(raw_sorted_global[0])
            gt_idx_t = torch.tensor(gt_cause_idx, device=device, dtype=torch.long)
            gt_embs = cause_table_embs.index_select(0, gt_idx_t)
            top1_cos = float((gt_embs @ cause_table_embs[top1_global]).max().item())
        else:
            top1_cos = 0.0
        all_top1_max_cos.append(top1_cos)

        # ---- Top-N predictions for inspection: diversified output ranking ----
        top_n_count = min(args.top_n_causes, len(div_sorted_global))
        top_n_global = div_sorted_global[:top_n_count].tolist()
        top_n_scores = (
            cand_scores[torch.from_numpy(div_sorted_local[:top_n_count]).to(device)]
            .detach().cpu().tolist()
        ) if top_n_count > 0 else []
        top_n_texts = [cause_texts[int(i)] for i in top_n_global]

        per_query_results.append({
            "query_image_id": int(q["image_id"]),
            "query_file_name": q["file_name"],
            "query_lesion_count": int(q["lesion_embs"].size(0)),
            "retained_positive_case_count": int(len(top_k_idx)),
            "candidate_pool_size": pool_size,
            "gt_causes": list(q["causes"]),
            "gt_cause_indices": list(gt_cause_idx),
            "gt_ranks_in_pool": gt_ranks_local,
            "gt_semantic_ranks_in_pool": gt_sem_ranks_local,
            "gt_clusters": gt_clusters_local,
            "gt_cluster_ranks_in_pool": gt_cluster_ranks_local,
            "top1_max_cos_to_gt": top1_cos,
            "retrieved_cases": [
                {
                    "case_id": int(top_k_idx[ki]),
                    "image_id": int(train_cases[int(top_k_idx[ki])]["image_id"]),
                    "similarity_raw": float(top_k_raw_w[ki]),
                    "similarity_weight_normalized": float(top_k_w[ki]),
                    "causes": list(train_cases[int(top_k_idx[ki])]["causes"]),
                }
                for ki in range(len(top_k_idx))
            ],
            "predicted_top_n": [
                {"cause_table_idx": int(i), "score": float(s), "text": t}
                for i, s, t in zip(top_n_global, top_n_scores, top_n_texts)
            ],
        })

        if (qi + 1) % 50 == 0 or qi + 1 == len(queries):
            elapsed = time.time() - t0
            rate = (qi + 1) / elapsed
            eta = (len(queries) - qi - 1) / max(rate, 1e-9)
            print(f"[eval] {qi+1}/{len(queries)}  "
                  f"rate={rate:.2f} q/s  ETA={eta/60:.1f} min")

    # Aggregate metrics
    ranks = np.array(all_gt_ranks, dtype=np.float64)
    sem_ranks = np.array(all_gt_sem_ranks, dtype=np.float64)
    pool_arr = np.array(pool_sizes, dtype=np.float64)
    retained_arr = np.array(retained_case_counts, dtype=np.float64)

    metrics = {
        "n_queries": len(queries),
        "n_gt_cause_occurrences": int(len(ranks)),
        "retrieved_cases": {
            "requested_top_k": int(args.top_k_cases),
            "positive_similarity_only": True,
            "mean_retained": float(retained_arr.mean()) if retained_arr.size else 0.0,
            "median_retained": float(np.median(retained_arr)) if retained_arr.size else 0.0,
            "min_retained": int(retained_arr.min()) if retained_arr.size else 0,
            "max_retained": int(retained_arr.max()) if retained_arr.size else 0,
        },
        "candidate_pool": {
            "mean_size": float(pool_arr.mean()) if pool_arr.size else 0.0,
            "median_size": float(np.median(pool_arr)) if pool_arr.size else 0.0,
            "min_size": int(pool_arr.min()) if pool_arr.size else 0,
            "max_size": int(pool_arr.max()) if pool_arr.size else 0,
        },
        "exact": summarize_rank_metric(ranks, cov_exact),
        "semantic": {
            "threshold": args.semantic_threshold,
            **summarize_rank_metric(sem_ranks, cov_semantic),
            "mean_top1_max_cos_to_gt": float(np.mean(all_top1_max_cos)) if all_top1_max_cos else 0.0,
        },
    }

    add_recall_at_ks(metrics["exact"], ranks, args.ks)
    add_recall_at_ks(metrics["semantic"], sem_ranks, args.ks)

    if all_gt_cluster_ranks:
        cl_ranks = np.array(all_gt_cluster_ranks, dtype=np.float64)
        metrics["cluster"] = {
            "n_gt_cause_occurrences": int(len(cl_ranks)),
            **summarize_rank_metric(cl_ranks, cov_cluster),
        }
        add_recall_at_ks(metrics["cluster"], cl_ranks, args.ks)

    config = {
        **vars(args),
        "case_db_meta": meta,
        "implementation_notes": {
            "embedding_normalization": "L2-normalize global, lesion, and cause embeddings before dot products.",
            "retrieved_case_filter": "Only train cases with combined similarity > 0 are retained for candidate-pool construction and scoring.",
            "case_weighting": "Positive retained similarities are normalized to sum to 1 before candidate scoring.",
            "metrics_ranking": "All metrics use raw candidate-pool score ranking, before diversification.",
            "output_ranking": "predicted_top_n uses diversified ranking for inspection only.",
            "cluster_metric": "Computed per GT cause occurrence; duplicate GT clusters within a query are not deduplicated.",
            "missing_rank_handling": "Misses are +inf internally, null in per_query_results, and never counted by R@K; mean/median rank are covered-only.",
        },
    }

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "config": config}, f,
                  ensure_ascii=False, indent=2)

    with (out_dir / "per_query_results.jsonl").open("w", encoding="utf-8") as f:
        for r in per_query_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== Phase 1 baseline metrics (candidate-restricted, fixed) ===")
    print(f"  n_queries={metrics['n_queries']}  "
          f"n_gt_occ={metrics['n_gt_cause_occurrences']}")
    print(f"  retrieved_cases: positive_only=True  "
          f"requested_K={metrics['retrieved_cases']['requested_top_k']}  "
          f"mean_retained={metrics['retrieved_cases']['mean_retained']:.1f}  "
          f"median_retained={metrics['retrieved_cases']['median_retained']:.0f}  "
          f"min={metrics['retrieved_cases']['min_retained']}  "
          f"max={metrics['retrieved_cases']['max_retained']}")
    print(f"  candidate_pool: mean={metrics['candidate_pool']['mean_size']:.1f}  "
          f"median={metrics['candidate_pool']['median_size']:.0f}  "
          f"min={metrics['candidate_pool']['min_size']}  "
          f"max={metrics['candidate_pool']['max_size']}")

    print(f"\n  -- exact-string match --")
    print(f"    coverage: {metrics['exact']['coverage']:.4f}")
    print(f"    n_covered: {metrics['exact']['n_covered']}  "
          f"n_missed: {metrics['exact']['n_missed']}")
    for k in ["MRR", "median_rank", "mean_rank"]:
        print(f"    {k}: {fmt_metric(metrics['exact'][k])}")
    for k in args.ks:
        print(f"    R@{k}: {metrics['exact'][f'R@{k}']:.4f}")

    print(f"\n  -- semantic match (threshold={args.semantic_threshold}) --")
    print(f"    coverage: {metrics['semantic']['coverage']:.4f}")
    print(f"    n_covered: {metrics['semantic']['n_covered']}  "
          f"n_missed: {metrics['semantic']['n_missed']}")
    for k in ["MRR", "median_rank", "mean_rank", "mean_top1_max_cos_to_gt"]:
        print(f"    {k}: {fmt_metric(metrics['semantic'][k])}")
    for k in args.ks:
        print(f"    R@{k}: {metrics['semantic'][f'R@{k}']:.4f}")

    if "cluster" in metrics:
        print(f"\n  -- cluster-level match (HDBSCAN) --")
        print(f"    coverage: {metrics['cluster']['coverage']:.4f}")
        print(f"    n_covered: {metrics['cluster']['n_covered']}  "
              f"n_missed: {metrics['cluster']['n_missed']}")
        for k in ["MRR", "median_rank", "mean_rank"]:
            print(f"    {k}: {fmt_metric(metrics['cluster'][k])}")
        print(f"    n_gt_cause_occurrences: {metrics['cluster']['n_gt_cause_occurrences']}")
        for k in args.ks:
            print(f"    R@{k}: {metrics['cluster'][f'R@{k}']:.4f}")

    print(f"\n[save] metrics.json + per_query_results.jsonl -> {out_dir}")
    print(f"[done] total time {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
