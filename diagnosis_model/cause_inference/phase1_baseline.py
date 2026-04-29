"""Phase 1 baseline: zero-training C²R + candidate-restricted cause scoring.

Pipeline per query (valid case):
  1. Combined case similarity vs every train case
       sim(q, c) = α · cos(q.global, c.global)
                 + β · Hungarian-matched lesion-set cosine
  2. Top-K retrieved cases (with similarity weights)
  3. Build CANDIDATE POOL: union of (deduped) cause embedding indices
     appearing in those top-K retrieved cases. Pool size ≤ K × max-causes-per-case.
  4. Score each candidate c in the pool using Option A:
       score(c) = Σ_{j in top-K} w_j · max_g cos(c, e_{j,g})
  5. Rank candidates by score → top-N predictions
     (Causes outside the retrieved cases' cause sets are not scored at all —
     this matches the case-based-reasoning intuition: only suggest a cause
     if at least one similar past case had it.)

Metrics:
  - exact: rank of each GT cause within the candidate pool
           (rank = pool_size + 1 if GT not covered by pool)
  - semantic (cosine ≥ threshold): rank of first candidate semantically
           equivalent to GT
  - cluster (HDBSCAN): rank of first candidate in the same cluster as GT
  - coverage_*: per-GT fraction of "GT covered by pool" under each match type
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_case_db(case_db_dir: Path):
    train = torch.load(case_db_dir / "train_cases.pt", weights_only=False)
    valid = torch.load(case_db_dir / "valid_cases.pt", weights_only=False)
    cause = torch.load(case_db_dir / "cause_text_embs.pt", weights_only=False)
    meta = json.load((case_db_dir / "meta.json").open())
    return train, valid, cause, meta


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
    q_global: torch.Tensor,           # [D]
    q_lesions: torch.Tensor,          # [N_q, D]
    train_global_stack: torch.Tensor, # [n_train, D]
    train_lesion_stack: torch.Tensor, # [total_lesions, D]
    train_offsets: List[int],
    alpha: float, beta: float,
    lesion_match: str = "hungarian",
) -> np.ndarray:
    """Return [n_train] combined similarity scores."""
    match_fn = _LESION_MATCH_FNS[lesion_match]
    g_sim = (q_global.unsqueeze(0) @ train_global_stack.T).squeeze(0)
    les_sim = (q_lesions @ train_lesion_stack.T).cpu().numpy()

    n_train = len(train_offsets) - 1
    l_score = np.zeros(n_train, dtype=np.float32)
    for i in range(n_train):
        s, e = train_offsets[i], train_offsets[i + 1]
        if e > s:
            l_score[i] = match_fn(les_sim[:, s:e])
    return alpha * g_sim.cpu().numpy() + beta * l_score


# ---------------------------------------------------------------------------
# Candidate-restricted cause scoring (Option A on retrieved-cases-only pool)
# ---------------------------------------------------------------------------

def build_candidate_pool(
    top_case_idx: np.ndarray,
    train_cases: list,
) -> List[int]:
    """Unique cause-table indices appearing in the K retrieved cases (preserving first-seen order)."""
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
    """Greedy MMR-style dedup: keep the highest-scored candidate, then suppress any
    later candidate whose cosine to ANY already-kept candidate ≥ threshold.

    Returns the kept local indices in score order. If threshold ≥ 1.0, returns
    sorted_local unchanged.
    """
    if sorted_local.size == 0 or threshold >= 1.0:
        return sorted_local
    kept_local: List[int] = [int(sorted_local[0])]
    kept_emb_rows: List[int] = [int(sorted_local[0])]
    for li in sorted_local[1:].tolist():
        e = cand_embs[int(li)]
        kept_t = cand_embs[torch.tensor(kept_emb_rows, device=cand_embs.device,
                                        dtype=torch.long)]
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
    """For each c in candidate_indices, score(c) = Σ_j w_j · max_g cos(emb[c], emb_{j,g}).

    Returns [len(candidate_indices)] tensor on cause_table_embs' device.
    """
    device = cause_table_embs.device
    if not candidate_indices:
        return torch.zeros(0, device=device)
    cand_embs = cause_table_embs.index_select(
        0, torch.tensor(candidate_indices, device=device, dtype=torch.long),
    )  # [U_cand, D]
    score = torch.zeros(len(candidate_indices), device=device)
    for case_i, w in zip(top_case_idx.tolist(), top_case_weights.tolist()):
        case_cidx = train_cases[int(case_i)]["cause_emb_indices"]
        if not case_cidx:
            continue
        case_embs = cause_table_embs.index_select(
            0, torch.tensor(case_cidx, device=device, dtype=torch.long),
        )  # [G, D]
        sims = cand_embs @ case_embs.T  # [U_cand, G]
        score = score + float(w) * sims.max(dim=1).values
    return score


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_db_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
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
                    default="diagnosis_model/cause_inference/outputs/cause_clusters_reassigned.json",
                    help="HDBSCAN clustering JSON (raw_string -> cluster_id). "
                         "Enables cluster-level metrics. Set to '' to disable.")
    ap.add_argument("--diversify_threshold", type=float, default=0.95,
                    help="when ranking candidates for output / metrics, suppress any "
                         "candidate whose cosine to a previously-kept candidate exceeds this. "
                         "Set to 1.0 to disable diversification.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    train_cases, valid_cases, cause_pack, meta = load_case_db(Path(args.case_db_dir))
    cause_table_embs = cause_pack["embeddings"].to(device)
    cause_texts = cause_pack["texts"]
    print(f"[load] train={len(train_cases)}  valid={len(valid_cases)}  "
          f"unique_causes={len(cause_texts)}  dim={cause_table_embs.size(-1)}")

    # Optional: cluster-based evaluation
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
    train_lesion_stack, train_offsets = stack_train_lesions(train_cases)
    train_lesion_stack = train_lesion_stack.to(device)
    print(f"[stack] train_globals={tuple(train_global_stack.shape)}  "
          f"train_lesions={tuple(train_lesion_stack.shape)}")

    queries = valid_cases if args.max_queries <= 0 else valid_cases[: args.max_queries]
    print(f"[eval] queries={len(queries)}  K={args.top_k_cases}  N={args.top_n_causes}  "
          f"alpha={args.alpha_global}  beta={args.beta_lesion}")

    per_query_results: list = []
    pool_sizes: List[int] = []
    all_gt_ranks: List[int] = []          # exact-index rank within candidate pool
    all_gt_sem_ranks: List[int] = []      # semantic-cosine rank within candidate pool
    all_gt_cluster_ranks: List[int] = []  # cluster-level rank within candidate pool
    cov_exact: List[int] = []             # 1 if GT exactly in pool, 0 otherwise
    cov_semantic: List[int] = []          # 1 if any pool cand ≥ threshold to GT
    cov_cluster: List[int] = []           # 1 if any pool cand in same cluster
    all_top1_max_cos: List[float] = []    # max cos(top-1 pred, any GT) per query

    t0 = time.time()
    for qi, q in enumerate(queries):
        q_global = q["global_emb"].to(device)
        q_lesions = q["lesion_embs"].to(device)

        sims = compute_case_similarities(
            q_global, q_lesions,
            train_global_stack, train_lesion_stack, train_offsets,
            args.alpha_global, args.beta_lesion,
            lesion_match=args.lesion_match,
        )
        top_k_idx = np.argsort(-sims)[: args.top_k_cases]
        top_k_w = sims[top_k_idx]

        # Build candidate pool from the top-K cases' causes (deduped)
        candidate_indices = build_candidate_pool(top_k_idx, train_cases)
        pool_size = len(candidate_indices)
        pool_sizes.append(pool_size)
        not_covered_rank = pool_size + 1  # rank assigned to uncovered GT

        # Score the pool only
        cand_scores = score_candidates(
            candidate_indices, top_k_idx, top_k_w, train_cases, cause_table_embs,
        )  # [pool_size]

        if pool_size == 0:
            sorted_local = np.empty(0, dtype=np.int64)
            sorted_global = np.empty(0, dtype=np.int64)
            cand_embs = torch.empty(0, cause_table_embs.size(-1), device=device)
        else:
            cand_embs = cause_table_embs.index_select(
                0, torch.tensor(candidate_indices, device=device, dtype=torch.long),
            )
            score_sorted_local = torch.argsort(cand_scores, descending=True).cpu().numpy()
            sorted_local = diversify(score_sorted_local, cand_embs, args.diversify_threshold)
            sorted_global = np.array(candidate_indices)[sorted_local]

        gt_cause_idx = q["cause_emb_indices"]

        # ---- Exact-index rank within pool ----
        global_to_pool_pos = {int(g): i for i, g in enumerate(sorted_global.tolist())}
        gt_ranks_local: List[int] = []
        for g in gt_cause_idx:
            pos = global_to_pool_pos.get(int(g))
            gt_ranks_local.append((pos + 1) if pos is not None else not_covered_rank)
            cov_exact.append(1 if pos is not None else 0)
        all_gt_ranks.extend(gt_ranks_local)

        # ---- Semantic-cosine rank within pool ----
        gt_sem_ranks_local: List[int] = []
        if pool_size == 0:
            gt_sem_ranks_local = [not_covered_rank] * len(gt_cause_idx)
            for _ in gt_cause_idx:
                cov_semantic.append(0)
        else:
            gt_idx_t = torch.tensor(gt_cause_idx, device=device, dtype=torch.long)
            gt_embs = cause_table_embs.index_select(0, gt_idx_t)        # [G, D]
            # Sorted candidate embeddings (in score-descending order)
            sorted_cand_embs = cand_embs[
                torch.from_numpy(sorted_local).to(device)
            ]  # [pool_size, D]
            cos_sorted = gt_embs @ sorted_cand_embs.T                   # [G, pool_size]
            sem_match = cos_sorted >= args.semantic_threshold
            for g_i in range(sem_match.size(0)):
                hits = torch.nonzero(sem_match[g_i], as_tuple=False)
                if hits.numel() > 0:
                    gt_sem_ranks_local.append(int(hits[0].item()) + 1)
                    cov_semantic.append(1)
                else:
                    gt_sem_ranks_local.append(not_covered_rank)
                    cov_semantic.append(0)
        all_gt_sem_ranks.extend(gt_sem_ranks_local)

        # ---- Cluster rank within pool ----
        gt_cluster_ranks_local: List[int] = []
        gt_clusters_local: List[int] = []
        if cluster_id_array is not None:
            gt_clusters_set = sorted(set(int(cluster_id_array[i]) for i in gt_cause_idx))
            sorted_clusters = cluster_id_array[sorted_global] if pool_size > 0 \
                              else np.empty(0, dtype=np.int64)
            for cid in gt_clusters_set:
                hits = np.flatnonzero(sorted_clusters == cid) if pool_size > 0 \
                       else np.empty(0, dtype=np.int64)
                if hits.size > 0:
                    gt_cluster_ranks_local.append(int(hits[0]) + 1)
                    cov_cluster.append(1)
                else:
                    gt_cluster_ranks_local.append(not_covered_rank)
                    cov_cluster.append(0)
                gt_clusters_local.append(cid)
            all_gt_cluster_ranks.extend(gt_cluster_ranks_local)

        # ---- Top-1 max-cos diagnostic ----
        if pool_size > 0:
            top1_global = int(sorted_global[0])
            gt_idx_t = torch.tensor(gt_cause_idx, device=device, dtype=torch.long)
            gt_embs = cause_table_embs.index_select(0, gt_idx_t)
            top1_cos = float((gt_embs @ cause_table_embs[top1_global]).max().item())
        else:
            top1_cos = 0.0
        all_top1_max_cos.append(top1_cos)

        # ---- Top-N predictions for inspection ----
        top_n_count = min(args.top_n_causes, pool_size)
        top_n_global = sorted_global[:top_n_count].tolist()
        top_n_scores = (
            cand_scores[torch.from_numpy(sorted_local[:top_n_count]).to(device)]
            .cpu().tolist()
        )
        top_n_texts = [cause_texts[int(i)] for i in top_n_global]

        per_query_results.append({
            "query_image_id": int(q["image_id"]),
            "query_file_name": q["file_name"],
            "query_lesion_count": int(q["lesion_embs"].size(0)),
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
                    "similarity": float(top_k_w[ki]),
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
    metrics = {
        "n_queries": len(queries),
        "n_gt_cause_occurrences": int(len(ranks)),
        "candidate_pool": {
            "mean_size": float(pool_arr.mean()) if pool_arr.size else 0.0,
            "median_size": float(np.median(pool_arr)) if pool_arr.size else 0.0,
            "min_size": int(pool_arr.min()) if pool_arr.size else 0,
            "max_size": int(pool_arr.max()) if pool_arr.size else 0,
        },
        "exact": {
            "coverage": float(np.mean(cov_exact)) if cov_exact else 0.0,
            "MRR": float((1.0 / ranks).mean()),
            "mean_rank": float(ranks.mean()),
            "median_rank": float(np.median(ranks)),
        },
        "semantic": {
            "threshold": args.semantic_threshold,
            "coverage": float(np.mean(cov_semantic)) if cov_semantic else 0.0,
            "MRR": float((1.0 / sem_ranks).mean()),
            "mean_rank": float(sem_ranks.mean()),
            "median_rank": float(np.median(sem_ranks)),
            "mean_top1_max_cos_to_gt": float(np.mean(all_top1_max_cos)),
        },
    }
    for k in args.ks:
        metrics["exact"][f"R@{k}"] = float((ranks <= k).mean())
        metrics["semantic"][f"R@{k}"] = float((sem_ranks <= k).mean())

    if all_gt_cluster_ranks:
        cl_ranks = np.array(all_gt_cluster_ranks, dtype=np.float64)
        metrics["cluster"] = {
            "n_gt_clusters_total": int(len(cl_ranks)),
            "coverage": float(np.mean(cov_cluster)) if cov_cluster else 0.0,
            "MRR": float((1.0 / cl_ranks).mean()),
            "mean_rank": float(cl_ranks.mean()),
            "median_rank": float(np.median(cl_ranks)),
        }
        for k in args.ks:
            metrics["cluster"][f"R@{k}"] = float((cl_ranks <= k).mean())

    config = {
        **vars(args),
        "case_db_meta": meta,
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "config": config}, f,
                  ensure_ascii=False, indent=2)

    with (out_dir / "per_query_results.jsonl").open("w", encoding="utf-8") as f:
        for r in per_query_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== Phase 1 baseline metrics (candidate-restricted) ===")
    print(f"  n_queries={metrics['n_queries']}  "
          f"n_gt_occ={metrics['n_gt_cause_occurrences']}")
    print(f"  candidate_pool: mean={metrics['candidate_pool']['mean_size']:.1f}  "
          f"median={metrics['candidate_pool']['median_size']:.0f}  "
          f"min={metrics['candidate_pool']['min_size']}  "
          f"max={metrics['candidate_pool']['max_size']}")
    print(f"\n  -- exact-string match --")
    print(f"    coverage: {metrics['exact']['coverage']:.4f}")
    for k in ["MRR", "median_rank", "mean_rank"]:
        print(f"    {k}: {metrics['exact'][k]:.4f}")
    for k in args.ks:
        print(f"    R@{k}: {metrics['exact'][f'R@{k}']:.4f}")
    print(f"\n  -- semantic match (threshold={args.semantic_threshold}) --")
    print(f"    coverage: {metrics['semantic']['coverage']:.4f}")
    for k in ["MRR", "median_rank", "mean_rank", "mean_top1_max_cos_to_gt"]:
        print(f"    {k}: {metrics['semantic'][k]:.4f}")
    for k in args.ks:
        print(f"    R@{k}: {metrics['semantic'][f'R@{k}']:.4f}")
    if "cluster" in metrics:
        print(f"\n  -- cluster-level match (HDBSCAN) --")
        print(f"    coverage: {metrics['cluster']['coverage']:.4f}")
        for k in ["MRR", "median_rank", "mean_rank"]:
            print(f"    {k}: {metrics['cluster'][k]:.4f}")
        print(f"    n_gt_clusters_total: {metrics['cluster']['n_gt_clusters_total']}")
        for k in args.ks:
            print(f"    R@{k}: {metrics['cluster'][f'R@{k}']:.4f}")
    print(f"\n[save] metrics.json + per_query_results.jsonl -> {out_dir}")
    print(f"[done] total time {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
