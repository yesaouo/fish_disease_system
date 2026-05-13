"""Apple-to-apple comparison: Phase 1 vs trained encoders, using the SAME
downstream cause-aggregation pipeline (only the case-similarity scorer differs).

Pipeline (per valid query):
    1. score every train case (scorer-specific)
    2. select_positive_top_cases(K) -> normalized weights
    3. build_candidate_pool          -> unique cause indices across the K cases
    4. score_candidates              -> per-candidate weighted max-cos vs GT
                                        causes of the retrieved cases
    5. rank GT causes by semantic cosine match (>= semantic_threshold) within
       the sorted pool -> R@K, MRR

Compared scorers:
    - phase1-hungarian   : the existing best-config Phase 1 baseline
    - mamba/mean/deepsets: trained encoders (single-vector cosine retrieval)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from diagnosis_model.cause_inference.models.case_encoder import (
    EncoderConfig, build_encoder,
)
from diagnosis_model.cause_inference.phase1_baseline import (
    build_candidate_pool,
    compute_case_similarities,
    load_case_db,
    score_candidates,
    select_positive_top_cases,
    stack_train_lesions,
)
from diagnosis_model.cause_inference.train_case_encoder import (
    encode_all,
)


MISS_RANK = float("inf")


def evaluate(
    score_fn: Callable[[int, dict], np.ndarray],   # (q_idx, q_case) -> [N_train] scores
    valid_cases: list,
    train_cases: list,
    cause_table_embs: torch.Tensor,
    top_k_cases: int = 20,
    semantic_threshold: float = 0.95,
    Ks: List[int] = (1, 5, 10, 20, 100),
    device: torch.device = torch.device("cuda"),
) -> dict:
    """Run Phase 1's eval pipeline with a pluggable case-scorer."""
    cause_table_embs = F.normalize(cause_table_embs.to(device), dim=-1)

    all_sem_ranks: List[float] = []
    cov_semantic: List[int] = []
    pool_sizes: List[int] = []
    retained_counts: List[int] = []

    t0 = time.time()
    for qi, q in enumerate(valid_cases):
        sims = score_fn(qi, q)                                          # [N_train]
        top_k_idx, top_k_w, _ = select_positive_top_cases(sims, top_k_cases)
        retained_counts.append(int(len(top_k_idx)))

        candidate_indices = build_candidate_pool(top_k_idx, train_cases)
        pool_size = len(candidate_indices)
        pool_sizes.append(pool_size)

        gt_cause_idx = q["cause_emb_indices"]

        if pool_size == 0:
            for _ in gt_cause_idx:
                all_sem_ranks.append(MISS_RANK)
                cov_semantic.append(0)
            continue

        cand_scores = score_candidates(
            candidate_indices, top_k_idx, top_k_w, train_cases, cause_table_embs,
        )
        raw_sorted_local = torch.argsort(cand_scores, descending=True).cpu().numpy()
        cand_embs = cause_table_embs.index_select(
            0, torch.tensor(candidate_indices, device=device, dtype=torch.long),
        )
        raw_sorted_cand_embs = cand_embs[
            torch.from_numpy(raw_sorted_local).to(device)
        ]

        gt_idx_t = torch.tensor(gt_cause_idx, device=device, dtype=torch.long)
        gt_embs = cause_table_embs.index_select(0, gt_idx_t)             # [G, D]
        cos_sorted = gt_embs @ raw_sorted_cand_embs.T                    # [G, P]
        sem_match = cos_sorted >= semantic_threshold

        for g_i in range(sem_match.size(0)):
            hits = torch.nonzero(sem_match[g_i], as_tuple=False)
            if hits.numel() > 0:
                rank = int(hits[0].item()) + 1
                all_sem_ranks.append(float(rank))
                cov_semantic.append(1)
            else:
                all_sem_ranks.append(MISS_RANK)
                cov_semantic.append(0)

    elapsed = time.time() - t0
    sem_ranks = np.array(all_sem_ranks, dtype=np.float64)
    finite = np.isfinite(sem_ranks)
    reciprocal = np.where(finite, 1.0 / np.where(finite, sem_ranks, 1.0), 0.0)

    out: dict = {}
    for K in Ks:
        out[f"sem_R@{K}"] = float(((sem_ranks <= K)).mean()) if sem_ranks.size else 0.0
    out["sem_MRR"] = float(reciprocal.mean()) if sem_ranks.size else 0.0
    out["coverage"] = float(np.mean(cov_semantic))
    out["mean_pool_size"] = float(np.mean(pool_sizes)) if pool_sizes else 0.0
    out["mean_retained_K"] = float(np.mean(retained_counts)) if retained_counts else 0.0
    out["eval_time_s"] = elapsed
    out["per_query_ms"] = elapsed / max(1, len(valid_cases)) * 1000
    return out


def make_phase1_scorer(train_cases, alpha=0.25, beta=0.75, lesion_match="hungarian",
                       device=torch.device("cuda")):
    G_t = torch.stack([c["global_emb"] for c in train_cases]).to(device)
    G_t = F.normalize(G_t, dim=-1)
    L_t, off = stack_train_lesions(train_cases)
    L_t = L_t.to(device)
    L_t = F.normalize(L_t, dim=-1)

    def score(qi, q):
        g = F.normalize(q["global_emb"].to(device), dim=-1)
        L = q["lesion_embs"].to(device)
        if L.size(0) > 0:
            L = F.normalize(L, dim=-1)
        return compute_case_similarities(g, L, G_t, L_t, off, alpha, beta, lesion_match)

    return score


def make_encoder_scorer(checkpoint_path, train_cases, valid_cases, device):
    pkg = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    cfg_dict = pkg["encoder_config"]
    cfg_dict["dtype"] = torch.bfloat16
    cfg = EncoderConfig(**cfg_dict)
    enc = build_encoder(cfg).to(device)
    enc.load_state_dict(pkg["encoder_state"])
    enc.eval()

    H_train = encode_all(enc, train_cases, device)                       # [Nt, D]
    H_valid = encode_all(enc, valid_cases, device)                       # [Nv, D]
    H_train = F.normalize(H_train.float(), dim=-1).to(device)
    H_valid = F.normalize(H_valid.float(), dim=-1).to(device)
    sim_full = (H_valid @ H_train.T).cpu().numpy()                       # [Nv, Nt]

    def score(qi, q):
        return sim_full[qi]

    return score, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_db_dir", type=str,
                    default="diagnosis_model/cause_inference/outputs/case_db")
    ap.add_argument("--checkpoints", type=str, nargs="+", required=False,
                    default=[],
                    help="space-separated name=path pairs, e.g. "
                         "mamba=outputs/mamba_encoder/mamba_v1/best_encoder.pt")
    ap.add_argument("--include_phase1", action="store_true",
                    help="Also evaluate Phase 1 hungarian baseline.")
    ap.add_argument("--top_k_cases", type=int, default=20)
    ap.add_argument("--semantic_threshold", type=float, default=0.95)
    ap.add_argument("--Ks", type=int, nargs="+", default=[1, 5, 10, 20, 100])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--min_query_lesions", type=int, default=0,
                    help="Restrict valid (query) set to cases with >= this many "
                         "lesions. Train pool is unchanged.")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_cases, valid_cases, cause_pkg, _ = load_case_db(Path(args.case_db_dir))
    cause_embs = cause_pkg["embeddings"]
    if args.min_query_lesions > 0:
        before = len(valid_cases)
        valid_cases = [c for c in valid_cases
                       if c["lesion_embs"].size(0) >= args.min_query_lesions]
        print(f"[filter] valid cases with >= {args.min_query_lesions} lesions: "
              f"{len(valid_cases)} / {before}")
    print(f"train={len(train_cases)} valid={len(valid_cases)}  "
          f"K={args.top_k_cases}  sem_thr={args.semantic_threshold}")

    results: Dict[str, dict] = {}

    if args.include_phase1:
        print("\n[phase1-hungarian] running ...")
        scorer = make_phase1_scorer(train_cases, device=device)
        results["phase1-hungarian"] = evaluate(
            scorer, valid_cases, train_cases, cause_embs,
            top_k_cases=args.top_k_cases,
            semantic_threshold=args.semantic_threshold,
            Ks=tuple(args.Ks),
            device=device,
        )

    for spec in args.checkpoints:
        name, path = spec.split("=", 1)
        print(f"\n[{name}] loading {path} ...")
        scorer, cfg = make_encoder_scorer(Path(path), train_cases, valid_cases, device)
        results[name] = evaluate(
            scorer, valid_cases, train_cases, cause_embs,
            top_k_cases=args.top_k_cases,
            semantic_threshold=args.semantic_threshold,
            Ks=tuple(args.Ks),
            device=device,
        )

    # Pretty-print
    print("\n" + "=" * 78)
    print(f"{'method':<22s} | "
          + " ".join(f"R@{k:<3d}" for k in args.Ks)
          + f" |  MRR  | cov | per-q ms")
    print("-" * 78)
    for name, m in results.items():
        cells = " ".join(f"{m[f'sem_R@{k}']:.3f}" for k in args.Ks)
        print(f"{name:<22s} | {cells} | {m['sem_MRR']:.3f} | "
              f"{m['coverage']:.2f} | {m['per_query_ms']:6.1f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
