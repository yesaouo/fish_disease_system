"""Phase 3 evaluation: hybrid Phase-1 + CEAH scoring on valid set.

For each valid query:
  1. Phase 1 case retrieval → candidate pool
  2. Phase 1 candidate score  s1(c) = Σ_j w_j · max_g cos(emb(c), emb(e_{j,g}))
  3. CEAH forward            → s_ceah(c) ∈ (0, 1)  and  α(c) ∈ [0, 1]^E
  4. Per-query min-max normalize each score to [0, 1]
  5. Hybrid score: hybrid(c, γ) = γ · s1_norm(c) + (1 − γ) · s_ceah_norm(c)
  6. For each γ in --gammas: rank → diversify → semantic R@K / MRR

Outputs:
  - metrics_gammas.json    aggregate metrics per γ
  - per_query.jsonl        one row per query incl. top-N predictions & α dump
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from diagnosis_model.cause_inference.models import CEAH
from diagnosis_model.cause_inference.phase1_baseline import (
    build_candidate_pool, compute_case_similarities, diversify,
    score_candidates, stack_train_lesions,
)


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------

def minmax_norm(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    lo = x.min()
    hi = x.max()
    if hi - lo < 1e-9:
        return torch.zeros_like(x)
    return (x - lo) / (hi - lo)


@torch.no_grad()
def ceah_forward_for_pool(
    ceah, q: dict, candidate_indices: List[int],
    cause_table_embs: torch.Tensor, in_dim: int, device: str,
    use_text_kind: str = "medical",
):
    """Run CEAH for one query against P candidates. Returns (scores [P], alpha [P, max_Ne],
    ev_mask [P, max_Ne]).

    All evidence is replicated across the P candidates.
    """
    P = len(candidate_indices)
    if P == 0:
        return (
            torch.zeros(0, device=device),
            torch.zeros(0, 0, device=device),
            torch.zeros(0, 0, dtype=torch.bool, device=device),
        )
    n_les = q["lesion_embs"].size(0)
    global_emb = q["global_emb"].unsqueeze(0).expand(P, -1).to(device)
    lesion_embs = q["lesion_embs"].unsqueeze(0).expand(P, -1, -1).to(device)
    lesion_mask = torch.ones(P, n_les, dtype=torch.bool, device=device)

    text_key = f"text_{use_text_kind}_emb"
    t = q.get(text_key)
    if t is not None:
        text_emb = t.unsqueeze(0).expand(P, -1).to(device)
        text_present = torch.ones(P, dtype=torch.bool, device=device)
    else:
        text_emb = torch.zeros(P, in_dim, device=device)
        text_present = torch.zeros(P, dtype=torch.bool, device=device)

    cand_embs = cause_table_embs.index_select(
        0, torch.tensor(candidate_indices, device=device, dtype=torch.long),
    )
    scores, alpha, ev_mask = ceah(
        global_emb, text_emb, text_present, lesion_embs, lesion_mask, cand_embs,
    )
    return scores, alpha, ev_mask


# ---------------------------------------------------------------------------
# Per-query evaluation under multiple gammas
# ---------------------------------------------------------------------------

def evaluate_one_query(
    q: dict, train_cases: list,
    train_global_stack: torch.Tensor,
    train_lesion_stack: torch.Tensor,
    train_offsets: List[int],
    cause_table_embs: torch.Tensor,
    cluster_id_array: np.ndarray,
    ceah,
    args,
    device: str,
    in_dim: int,
):
    q_global = q["global_emb"].to(device)
    q_lesions = q["lesion_embs"].to(device)

    sims = compute_case_similarities(
        q_global, q_lesions,
        train_global_stack, train_lesion_stack, train_offsets,
        args.alpha_global, args.beta_lesion, lesion_match=args.lesion_match,
    )
    top_k_idx = np.argsort(-sims)[: args.top_k_cases]
    top_k_w = sims[top_k_idx]

    candidate_indices = build_candidate_pool(top_k_idx, train_cases)
    pool_size = len(candidate_indices)

    if pool_size == 0:
        return {
            "pool_size": 0,
            "ranks_per_gamma": {f"g={g:.2f}": [] for g in args.gammas},
            "alpha": None,
            "predicted_top_n": [],
        }

    cand_idx_t = torch.tensor(candidate_indices, device=device, dtype=torch.long)
    cand_embs = cause_table_embs.index_select(0, cand_idx_t)  # [P, D]

    # Phase 1 score
    s1 = score_candidates(
        candidate_indices, top_k_idx, top_k_w, train_cases, cause_table_embs,
    )  # [P]
    # CEAH score + alpha
    s_ceah, alpha, ev_mask = ceah_forward_for_pool(
        ceah, q, candidate_indices, cause_table_embs, in_dim, device,
        use_text_kind=args.text_kind,
    )

    s1_n = minmax_norm(s1)
    sc_n = minmax_norm(s_ceah)

    # Build rank info per gamma
    gt_idx_t = torch.tensor(q["cause_emb_indices"], device=device, dtype=torch.long)
    gt_embs = cause_table_embs.index_select(0, gt_idx_t)

    out = {
        "pool_size": pool_size,
        "ranks_per_gamma": {},
        "predicted_top_n_per_gamma": {},
    }

    for g in args.gammas:
        hybrid = g * s1_n + (1.0 - g) * sc_n  # [P]
        score_sorted = torch.argsort(hybrid, descending=True).cpu().numpy()
        sorted_local = diversify(score_sorted, cand_embs, args.diversify_threshold)
        sorted_global = np.array(candidate_indices)[sorted_local]

        # Semantic ranks
        sorted_cand_embs = cand_embs[torch.from_numpy(sorted_local).to(device)]
        cos_sorted = gt_embs @ sorted_cand_embs.T
        sem_match = cos_sorted >= args.semantic_threshold

        sem_ranks: List[int] = []
        for g_i in range(sem_match.size(0)):
            hits = torch.nonzero(sem_match[g_i], as_tuple=False)
            if hits.numel() > 0:
                sem_ranks.append(int(hits[0].item()) + 1)
            else:
                sem_ranks.append(pool_size + 1)

        cl_ranks: List[int] = []
        if cluster_id_array is not None:
            gt_clusters_set = sorted(set(int(cluster_id_array[i]) for i in q["cause_emb_indices"]))
            sorted_clusters = cluster_id_array[sorted_global]
            for cid in gt_clusters_set:
                hits = np.flatnonzero(sorted_clusters == cid)
                if hits.size > 0:
                    cl_ranks.append(int(hits[0]) + 1)
                else:
                    cl_ranks.append(pool_size + 1)

        out["ranks_per_gamma"][f"g={g:.2f}"] = {
            "sem_ranks": sem_ranks,
            "cluster_ranks": cl_ranks,
        }

        if abs(g - args.dump_gamma) < 1e-6:
            top_n_count = min(args.top_n_causes, pool_size)
            top_n_global = sorted_global[:top_n_count].tolist()
            top_n_alpha = []
            top_n_score = []
            for li in sorted_local[:top_n_count].tolist():
                top_n_alpha.append(alpha[li].cpu().tolist())
                top_n_score.append(float(hybrid[li].item()))
            out["predicted_top_n_per_gamma"][f"g={g:.2f}"] = [
                {"cause_idx": int(g_idx), "score": s, "alpha": a}
                for g_idx, s, a in zip(top_n_global, top_n_score, top_n_alpha)
            ]

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_db_dir", type=str, required=True)
    ap.add_argument("--ceah_ckpt", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--cluster_json", type=str,
                    default="diagnosis_model/cause_inference/outputs/cause_clusters_reassigned.json")

    ap.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--dump_gamma", type=float, default=0.5,
                    help="γ at which to dump per-query top-N + α (for analysis)")
    ap.add_argument("--top_k_cases", type=int, default=20)
    ap.add_argument("--top_n_causes", type=int, default=20)
    ap.add_argument("--alpha_global", type=float, default=0.25)
    ap.add_argument("--beta_lesion", type=float, default=0.75)
    ap.add_argument("--lesion_match", type=str, default="hungarian")
    ap.add_argument("--semantic_threshold", type=float, default=0.95)
    ap.add_argument("--diversify_threshold", type=float, default=0.95)
    ap.add_argument("--text_kind", type=str, default="medical",
                    choices=["medical", "colloquial"])

    # CEAH model dims
    ap.add_argument("--common_dim", type=int, default=256)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--attribution_mode", type=str, default="sigmoid",
                    choices=["sigmoid", "softmax"])
    ap.add_argument("--scoring_mode", type=str, default="single",
                    choices=["single", "multiplicative"])

    ap.add_argument("--max_queries", type=int, default=-1)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 20])
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    # Load data
    case_db_dir = Path(args.case_db_dir)
    train_cases = torch.load(case_db_dir / "train_cases.pt", weights_only=False)
    valid_cases = torch.load(case_db_dir / "valid_cases.pt", weights_only=False)
    cause_pack = torch.load(case_db_dir / "cause_text_embs.pt", weights_only=False)
    cause_table_embs = cause_pack["embeddings"].to(device)
    cause_texts = cause_pack["texts"]
    print(f"[load] train={len(train_cases)} valid={len(valid_cases)} "
          f"causes={cause_table_embs.size(0)}")

    cluster_id_array = None
    if args.cluster_json:
        with open(args.cluster_json, encoding="utf-8") as f:
            cl = json.load(f)
        o2c = cl["original_to_cause_id"]
        cluster_id_array = np.array(
            [int(o2c[t]) for t in cause_texts], dtype=np.int64,
        )
        print(f"[cluster] {len(set(cluster_id_array.tolist()))} clusters loaded")

    # Stack train embeddings
    train_global_stack = torch.stack([c["global_emb"] for c in train_cases]).to(device)
    train_lesion_stack, train_offsets = stack_train_lesions(train_cases)
    train_lesion_stack = train_lesion_stack.to(device)

    # Load CEAH
    in_dim = cause_table_embs.size(-1)
    ceah = CEAH(
        global_dim=in_dim, text_dim=in_dim, lesion_dim=in_dim, cause_dim=in_dim,
        common_dim=args.common_dim, hidden_dim=args.hidden_dim, dropout=args.dropout,
        attribution_mode=args.attribution_mode, scoring_mode=args.scoring_mode,
    ).to(device)
    ceah.load_state_dict(torch.load(args.ceah_ckpt, map_location=device))
    ceah.eval()
    print(f"[ceah] loaded {args.ceah_ckpt}")

    queries = valid_cases if args.max_queries <= 0 else valid_cases[: args.max_queries]
    print(f"[eval] queries={len(queries)}  K={args.top_k_cases}  "
          f"gammas={args.gammas}  dump_gamma={args.dump_gamma}")

    # Aggregate ranks per gamma
    sem_ranks_all = {f"g={g:.2f}": [] for g in args.gammas}
    cl_ranks_all = {f"g={g:.2f}": [] for g in args.gammas}
    cov_all = {f"g={g:.2f}": [] for g in args.gammas}

    pool_sizes = []
    per_query_results = []

    t0 = time.time()
    for qi, q in enumerate(queries):
        out = evaluate_one_query(
            q, train_cases,
            train_global_stack, train_lesion_stack, train_offsets,
            cause_table_embs, cluster_id_array, ceah, args, device, in_dim,
        )
        pool_sizes.append(out["pool_size"])

        for tag, ranks in out["ranks_per_gamma"].items():
            for r in ranks["sem_ranks"]:
                sem_ranks_all[tag].append(r)
                cov_all[tag].append(1 if r <= out["pool_size"] else 0)
            for r in ranks["cluster_ranks"]:
                cl_ranks_all[tag].append(r)

        # Save per-query info for dump_gamma
        dump_tag = f"g={args.dump_gamma:.2f}"
        per_query_results.append({
            "case_id": int(q["image_id"]),
            "file_name": q["file_name"],
            "lesion_count": int(q["lesion_embs"].size(0)),
            "pool_size": out["pool_size"],
            "gt_causes": list(q["causes"]),
            "gt_cause_indices": list(q["cause_emb_indices"]),
            "ranks_per_gamma": {
                tag: {"sem_ranks": r["sem_ranks"], "cluster_ranks": r["cluster_ranks"]}
                for tag, r in out["ranks_per_gamma"].items()
            },
            "predicted_top_n": out["predicted_top_n_per_gamma"].get(dump_tag, []),
        })

        if (qi + 1) % 100 == 0 or qi + 1 == len(queries):
            elapsed = time.time() - t0
            rate = (qi + 1) / elapsed
            eta = (len(queries) - qi - 1) / max(rate, 1e-9)
            print(f"[eval] {qi+1}/{len(queries)}  "
                  f"rate={rate:.2f} q/s  ETA={eta/60:.1f} min")

    # Aggregate
    metrics: Dict[str, Dict] = {}
    for tag in sem_ranks_all:
        sa = np.array(sem_ranks_all[tag], dtype=np.float64) if sem_ranks_all[tag] else np.array([1.0])
        ca = np.array(cl_ranks_all[tag], dtype=np.float64) if cl_ranks_all[tag] else np.array([1.0])
        m = {
            "sem_MRR": float((1.0 / sa).mean()),
            "sem_median_rank": float(np.median(sa)),
            "sem_coverage": float(np.mean(cov_all[tag])) if cov_all[tag] else 0.0,
        }
        for k in args.ks:
            m[f"sem_R@{k}"] = float((sa <= k).mean())
        if cl_ranks_all[tag]:
            m["cl_MRR"] = float((1.0 / ca).mean())
            for k in args.ks:
                m[f"cl_R@{k}"] = float((ca <= k).mean())
        metrics[tag] = m

    pool_arr = np.array(pool_sizes, dtype=np.float64)
    summary = {
        "n_queries": len(queries),
        "pool_size_mean": float(pool_arr.mean()),
        "pool_size_median": float(np.median(pool_arr)),
        "metrics_per_gamma": metrics,
        "config": vars(args),
    }
    with (out_dir / "metrics_gammas.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with (out_dir / "per_query.jsonl").open("w", encoding="utf-8") as f:
        for r in per_query_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== Hybrid eval summary ===")
    print(f"{'gamma':>6}  {'sem_MRR':>8}  {'sem_R@1':>8}  {'sem_R@5':>8}  "
          f"{'sem_R@10':>9}  {'sem_R@20':>9}  {'sem_cov':>8}")
    for g in args.gammas:
        tag = f"g={g:.2f}"
        m = metrics[tag]
        print(f"{g:>6.2f}  {m['sem_MRR']:>8.4f}  {m['sem_R@1']:>8.4f}  "
              f"{m['sem_R@5']:>8.4f}  {m['sem_R@10']:>9.4f}  "
              f"{m['sem_R@20']:>9.4f}  {m['sem_coverage']:>8.4f}")
    print(f"\n[save] -> {out_dir}")


if __name__ == "__main__":
    main()
