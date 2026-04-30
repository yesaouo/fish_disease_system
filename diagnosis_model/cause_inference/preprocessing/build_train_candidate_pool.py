"""Pre-compute Phase 1 candidate pools for every train case (for CEAH training).

For each train case treated as a leave-one-out query:
  1. Compute case similarities vs all OTHER train cases (self excluded).
  2. Take top-K retrieved cases.
  3. Build candidate pool = unique cause indices appearing in those K cases.
  4. Mark each pool entry as positive (semantically equivalent to any of the
     query's GT causes within --semantic_threshold) or negative.

Output (one file):
  train_candidate_pool.pt = {
    "config":          dict of args
    "case_pool":       list[dict], one per train case (case_id is the index)
        each dict:
          "top_k_idx":              LongTensor[K]   train case indices (excludes self)
          "top_k_w":                FloatTensor[K]  similarity weights
          "candidate_cause_indices": LongTensor[P]  cause-table indices
          "positive_mask":          BoolTensor[P]   True if semantically equivalent to a GT
          "gt_exact_mask":          BoolTensor[P]   True if exact GT cause string match
  }

This is a one-time offline computation. ~3-4 minutes for 12780 train cases.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

from diagnosis_model.cause_inference.phase1_baseline import (
    build_candidate_pool,
    compute_case_similarities,
    stack_train_lesions,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_db_dir", type=str, required=True)
    ap.add_argument("--output_path", type=str, required=True)
    ap.add_argument("--top_k_cases", type=int, default=20)
    ap.add_argument("--alpha_global", type=float, default=0.25)
    ap.add_argument("--beta_lesion", type=float, default=0.75)
    ap.add_argument("--lesion_match", type=str, default="hungarian",
                    choices=["hungarian", "max_mean"])
    ap.add_argument("--semantic_threshold", type=float, default=0.95)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = args.device

    case_db_dir = Path(args.case_db_dir)
    train_cases = torch.load(case_db_dir / "train_cases.pt", weights_only=False)
    cause_pack = torch.load(case_db_dir / "cause_text_embs.pt", weights_only=False)
    cause_table_embs = cause_pack["embeddings"].to(device)
    print(f"[load] train={len(train_cases)}  causes={cause_table_embs.size(0)}")

    train_global_stack = torch.stack([c["global_emb"] for c in train_cases]).to(device)
    train_lesion_stack, train_offsets = stack_train_lesions(train_cases)
    train_lesion_stack = train_lesion_stack.to(device)
    print(f"[stack] global={tuple(train_global_stack.shape)}  "
          f"lesions={tuple(train_lesion_stack.shape)}")

    case_pool: List[dict] = []
    n = len(train_cases)
    t0 = time.time()

    for qi, q in enumerate(train_cases):
        q_global = q["global_emb"].to(device)
        q_lesions = q["lesion_embs"].to(device)

        sims = compute_case_similarities(
            q_global, q_lesions,
            train_global_stack, train_lesion_stack, train_offsets,
            args.alpha_global, args.beta_lesion,
            lesion_match=args.lesion_match,
        )
        sims[qi] = -np.inf  # leave-one-out: exclude self

        top_k_idx = np.argsort(-sims)[: args.top_k_cases]
        top_k_w = sims[top_k_idx]

        candidate_indices = build_candidate_pool(top_k_idx, train_cases)
        if not candidate_indices:
            case_pool.append({
                "top_k_idx": torch.tensor(top_k_idx, dtype=torch.long),
                "top_k_w": torch.tensor(top_k_w, dtype=torch.float32),
                "candidate_cause_indices": torch.empty(0, dtype=torch.long),
                "positive_mask": torch.empty(0, dtype=torch.bool),
                "gt_exact_mask": torch.empty(0, dtype=torch.bool),
            })
            continue

        cand_idx_t = torch.tensor(candidate_indices, device=device, dtype=torch.long)
        cand_embs = cause_table_embs.index_select(0, cand_idx_t)         # [P, D]
        gt_idx = q["cause_emb_indices"]
        gt_idx_t = torch.tensor(gt_idx, device=device, dtype=torch.long)
        gt_embs = cause_table_embs.index_select(0, gt_idx_t)             # [G, D]

        # cosine [G, P]; positive if any GT exceeds threshold for this candidate
        cos = gt_embs @ cand_embs.T
        positive_mask = (cos >= args.semantic_threshold).any(dim=0)      # [P]

        gt_set = set(gt_idx)
        gt_exact_mask = torch.tensor(
            [int(c) in gt_set for c in candidate_indices], dtype=torch.bool,
        )

        case_pool.append({
            "top_k_idx": torch.tensor(top_k_idx, dtype=torch.long),
            "top_k_w": torch.tensor(top_k_w, dtype=torch.float32),
            "candidate_cause_indices": cand_idx_t.cpu(),
            "positive_mask": positive_mask.cpu(),
            "gt_exact_mask": gt_exact_mask,
        })

        if (qi + 1) % 500 == 0 or qi + 1 == n:
            elapsed = time.time() - t0
            rate = (qi + 1) / elapsed
            eta = (n - qi - 1) / max(rate, 1e-9)
            print(f"[build] {qi+1}/{n}  rate={rate:.1f} q/s  ETA={eta/60:.1f} min")

    pool_sizes = [int(p["candidate_cause_indices"].numel()) for p in case_pool]
    n_pos = [int(p["positive_mask"].sum().item()) for p in case_pool]
    n_neg = [pool_sizes[i] - n_pos[i] for i in range(n)]

    print()
    print(f"[stats] pool size  mean={np.mean(pool_sizes):.1f}  "
          f"median={int(np.median(pool_sizes))}  "
          f"min={min(pool_sizes)}  max={max(pool_sizes)}")
    print(f"[stats] positives  mean={np.mean(n_pos):.1f}  median={int(np.median(n_pos))}")
    print(f"[stats] negatives  mean={np.mean(n_neg):.1f}  median={int(np.median(n_neg))}")
    print(f"[stats] queries with 0 positives: "
          f"{sum(1 for p in n_pos if p == 0)} / {n}")

    payload = {
        "config": vars(args),
        "case_pool": case_pool,
    }
    torch.save(payload, out_path)
    print(f"[save] {out_path}  ({sum(pool_sizes)} total (case, candidate) pairs)")


if __name__ == "__main__":
    main()
