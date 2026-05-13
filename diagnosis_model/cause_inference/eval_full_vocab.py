"""Full-vocabulary 1-stage retrieval: query encoding → cosine vs all 56,310
cause text embeddings → rank.

Removes the candidate-pool restriction that bottlenecks Phase 1's official
metric at 93% coverage. The price: ranking is now out of ~56k candidates
instead of ~87.

Two rank flavors per GT cause are reported:
    - exact rank: position of the GT's own cause-text embedding when sorted
                  by score (only sensible because 94.7% of causes are
                  singletons -> exact match is the natural target).
    - semantic rank: position of the first cause-text whose cosine to the
                     GT cause text emb is >= --semantic_threshold.

Compared methods (set via --checkpoints + --include_phase1):
    - phase1-vision      : raw q["global_emb"] scored against cause embs.
                           Sanity baseline; no training, no encoder.
    - <encoder>          : trained case encoder (mamba/mean/deepsets,
                           with or without dual-target InfoNCE).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from diagnosis_model.cause_inference.models.mamba_encoder import (
    EncoderConfig, build_encoder,
)
from diagnosis_model.cause_inference.phase1_baseline import load_case_db
from diagnosis_model.cause_inference.train_mamba_encoder import encode_all


MISS_RANK = float("inf")


@torch.no_grad()
def rank_full_vocab(
    h_query: torch.Tensor,             # [Nq, D] L2-normed
    cause_text_embs: torch.Tensor,     # [V, D] L2-normed
    valid_cases: list,
    semantic_threshold: float = 0.95,
    Ks: List[int] = (1, 5, 10, 20, 100, 1000),
) -> dict:
    """Score every cause for every query, then read off ranks per GT."""
    Nq = h_query.size(0)
    device = h_query.device
    V = cause_text_embs.size(0)

    # Per-query scores against full vocab. [Nq, V] in fp16 = ~177 MB at V=56310.
    # Compute in fp16 then sort indices in fp32 for stability.
    sims = (h_query @ cause_text_embs.T).float()  # [Nq, V]

    # Sort once per query.
    sorted_idx = torch.argsort(sims, dim=1, descending=True)        # [Nq, V]

    # Build a per-cause -> global-rank lookup per query.
    # We use cumulative argwhere via argsort-of-argsort.
    rank_of_cause = torch.argsort(sorted_idx, dim=1)                # [Nq, V]
    # rank_of_cause[q, c] = rank of cause c when sorted by sims[q, :]
    # (0-indexed; we'll +1 below for ranks).

    exact_ranks: List[float] = []
    sem_ranks: List[float] = []
    cov_exact: List[int] = []
    cov_sem: List[int] = []

    for qi in range(Nq):
        gt_idx_list = valid_cases[qi]["cause_emb_indices"]
        if not gt_idx_list:
            continue
        gt_idx_t = torch.tensor(gt_idx_list, device=device, dtype=torch.long)

        # Exact-index rank: where does cause idx land in the sorted list?
        ex_r = (rank_of_cause[qi, gt_idx_t] + 1).float().cpu().numpy()
        for r in ex_r:
            exact_ranks.append(float(r))
            cov_exact.append(1)  # always covered: every cause is in vocab

        # Semantic rank: find first sorted cause whose emb is >= threshold
        # to the GT cause emb.
        gt_embs = cause_text_embs[gt_idx_t]                          # [G, D]
        # Build cumulative-best similarity per sorted position is expensive.
        # Cheap alt: for each GT, find indices of causes with cos>=threshold
        # to it, then take the MIN rank among those.
        cos_gt_to_all = gt_embs @ cause_text_embs.T                  # [G, V]
        sem_mask = cos_gt_to_all >= semantic_threshold               # [G, V]
        # Get ranks via lookup table.
        ranks_all = rank_of_cause[qi].unsqueeze(0).expand(gt_embs.size(0), -1)
        masked_ranks = torch.where(sem_mask, ranks_all,
                                   torch.full_like(ranks_all, V + 1))
        first_match_rank = (masked_ranks.min(dim=1).values + 1).float().cpu().numpy()
        for r in first_match_rank:
            if r > V:
                sem_ranks.append(MISS_RANK)
                cov_sem.append(0)
            else:
                sem_ranks.append(float(r))
                cov_sem.append(1)

    out: dict = {}
    for tag, ranks in [("exact", exact_ranks), ("sem", sem_ranks)]:
        a = np.array(ranks, dtype=np.float64)
        finite = np.isfinite(a)
        for k in Ks:
            out[f"{tag}_R@{k}"] = float(((a <= k) & finite).mean()) if a.size else 0.0
        out[f"{tag}_MRR"] = float(
            np.where(finite, 1.0 / np.where(finite, a, 1.0), 0.0).mean()
        ) if a.size else 0.0
    out["coverage_exact"] = float(np.mean(cov_exact)) if cov_exact else 0.0
    out["coverage_sem"] = float(np.mean(cov_sem)) if cov_sem else 0.0
    return out


def encode_phase1_vision(valid_cases, device):
    """Baseline: just the SigLIP2 global-image emb (no training, no fusion)."""
    H = torch.stack([c["global_emb"] for c in valid_cases]).to(device)
    return F.normalize(H.float(), dim=-1)


def encode_with_checkpoint(checkpoint_path, valid_cases, device):
    pkg = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    cfg_dict = pkg["encoder_config"]
    cfg_dict["dtype"] = torch.bfloat16
    cfg = EncoderConfig(**cfg_dict)
    enc = build_encoder(cfg).to(device)
    enc.load_state_dict(pkg["encoder_state"])
    enc.eval()
    H = encode_all(enc, valid_cases, device)
    return F.normalize(H.float(), dim=-1).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_db_dir", type=str,
                    default="diagnosis_model/cause_inference/outputs/case_db")
    ap.add_argument("--checkpoints", type=str, nargs="+", default=[])
    ap.add_argument("--include_phase1_vision", action="store_true")
    ap.add_argument("--semantic_threshold", type=float, default=0.95)
    ap.add_argument("--Ks", type=int, nargs="+", default=[1, 5, 10, 20, 100, 1000])
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    _train, valid_cases, cause_pkg, _ = load_case_db(Path(args.case_db_dir))
    cause_embs = F.normalize(cause_pkg["embeddings"].float(), dim=-1).to(device)
    V = cause_embs.size(0)
    print(f"valid={len(valid_cases)}  vocab={V}  sem_thr={args.semantic_threshold}")

    results: Dict[str, dict] = {}

    if args.include_phase1_vision:
        print("\n[phase1-vision] global_emb only ...")
        H = encode_phase1_vision(valid_cases, device)
        t0 = time.time()
        results["phase1-vision"] = rank_full_vocab(
            H, cause_embs, valid_cases,
            semantic_threshold=args.semantic_threshold, Ks=tuple(args.Ks),
        )
        results["phase1-vision"]["elapsed_s"] = time.time() - t0

    for spec in args.checkpoints:
        name, path = spec.split("=", 1)
        print(f"\n[{name}] {path}")
        H = encode_with_checkpoint(Path(path), valid_cases, device)
        t0 = time.time()
        results[name] = rank_full_vocab(
            H, cause_embs, valid_cases,
            semantic_threshold=args.semantic_threshold, Ks=tuple(args.Ks),
        )
        results[name]["elapsed_s"] = time.time() - t0

    # Pretty-print: two tables, exact then semantic
    for tag in ["exact", "sem"]:
        print("\n" + "=" * 90)
        print(f"  Full-vocab rank ({tag}-match)   sem_thr={args.semantic_threshold}")
        print("-" * 90)
        hdr = f"{'method':<22s} | " + " ".join(f"R@{k:<4d}" for k in args.Ks) + " |  MRR"
        print(hdr)
        for name, m in results.items():
            cells = " ".join(f"{m[f'{tag}_R@{k}']:.3f}" for k in args.Ks)
            print(f"{name:<22s} | {cells} | {m[f'{tag}_MRR']:.3f}")
        print("=" * 90)
    print()
    print(f"coverage_exact (always 1.0 — every cause is in vocab)")
    for name, m in results.items():
        print(f"  {name:<22s} coverage_sem={m['coverage_sem']:.3f}")


if __name__ == "__main__":
    main()
