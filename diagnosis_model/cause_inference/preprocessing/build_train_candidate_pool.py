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
import torch.nn.functional as F

from diagnosis_model.cause_inference.phase1_baseline import (
    build_candidate_pool,
    compute_case_similarities,
    load_train_bank,
    offsets_to_case_ids,
    select_positive_top_cases,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_db_dir", type=str, required=True)
    ap.add_argument("--output_path", type=str, required=True)
    ap.add_argument("--top_k_cases", type=int, default=20)
    ap.add_argument("--alpha_global", type=float, default=0.25)
    ap.add_argument("--beta_lesion", type=float, default=0.75)
    ap.add_argument("--lesion_match", type=str, default="max_mean",
                    choices=["hungarian", "max_mean", "max_mean_normalized"])
    ap.add_argument("--semantic_threshold", type=float, default=0.95)
    ap.add_argument("--bank_dtype", type=str, default="fp32",
                    choices=["fp32", "fp16", "bf16"],
                    help="On-device storage dtype of the train bank. Default "
                         "'fp32' preserves the historic fish workflow; pass "
                         "'bf16' for DDXPlus (1M-case bank in fp32 is ~60 GB "
                         "lesion stack and won't fit on a 32 GB GPU).")
    ap.add_argument("--max_train_cases", type=int, default=0,
                    help="Cap on retained train-bank cases via uniform random "
                         "per-shard subsampling. Applies to BOTH the bank and "
                         "the query set (LOO requires query[i] == bank[i]). "
                         "Default 0 = full bank (correct for fish 12,780-case "
                         "case_dbs). For DDXPlus pass 200000 — the full 1M bank "
                         "requires ~75 GB CPU RAM in load_cases (fp32 upcast of "
                         "all per-case fields) AND ~30 GB GPU bf16 lesion stack, "
                         "neither of which fits on a 62 GB / 32 GB machine. "
                         "200k × top_k=20 = 4M (q, retrieved_case) pairs leaves "
                         "~4k cases per of 49 DDXPlus conditions for CEAH training.")
    ap.add_argument("--sample_seed", type=int, default=42)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    bank_dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    bank_dtype = bank_dtype_map[args.bank_dtype]
    max_train_cases = args.max_train_cases if args.max_train_cases > 0 else None

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = args.device

    case_db_dir = Path(args.case_db_dir)
    cause_pack = torch.load(case_db_dir / "cause_text_embs.pt", weights_only=False)
    # Match phase1_baseline.py: explicitly L2-normalize so dot products are cosine.
    # .float() upcasts DDXPlus bf16/fp16 storage; no-op for fp32 fish builds.
    cause_table_embs = F.normalize(cause_pack["embeddings"].to(device).float(), dim=-1)

    # Load bank with subsample + bf16 storage. ``train_cases`` contains
    # ``cause_emb_indices`` / ``causes`` per case; query embeddings are pulled
    # from train_global_stack / train_lesion_stack directly (already L2-normalized,
    # already on device in bank_dtype) — no second CPU pass needed.
    train_cases, train_global_stack, train_lesion_stack, train_offsets = load_train_bank(
        case_db_dir, device,
        bank_dtype=bank_dtype,
        max_cases=max_train_cases, sample_seed=args.sample_seed,
    )
    n = len(train_cases)
    print(f"[load] train={n}  causes={cause_table_embs.size(0)}  "
          f"bank_dtype={train_global_stack.dtype}")
    print(f"[stack] global={tuple(train_global_stack.shape)}  "
          f"lesions={tuple(train_lesion_stack.shape)}")

    case_pool: List[dict] = []
    train_case_ids = offsets_to_case_ids(train_offsets, train_lesion_stack.device)
    t0 = time.time()

    for qi, q in enumerate(train_cases):
        # Pull query embeddings from the already-normalized bank — avoids
        # holding a parallel CPU copy of every case's emb fields.
        q_global = train_global_stack[qi]
        q_lesions = train_lesion_stack[train_offsets[qi]:train_offsets[qi + 1]]

        sims = compute_case_similarities(
            q_global, q_lesions,
            train_global_stack, train_lesion_stack, train_offsets,
            args.alpha_global, args.beta_lesion,
            lesion_match=args.lesion_match,
            train_case_ids=train_case_ids,
        )
        sims[qi] = -np.inf  # leave-one-out: exclude self (filtered by select_positive_top_cases)

        # Drop train cases with similarity <= 0; positive weights normalized to sum to 1.
        top_k_idx, _, top_k_raw_w = select_positive_top_cases(sims, args.top_k_cases)

        candidate_indices = build_candidate_pool(top_k_idx, train_cases)
        if not candidate_indices:
            case_pool.append({
                "top_k_idx": torch.tensor(top_k_idx, dtype=torch.long),
                "top_k_w": torch.tensor(top_k_raw_w, dtype=torch.float32),
                "candidate_cause_indices": torch.empty(0, dtype=torch.long),
                "positive_mask": torch.empty(0, dtype=torch.bool),
                "gt_exact_mask": torch.empty(0, dtype=torch.bool),
            })
            continue

        cand_idx_t = torch.tensor(candidate_indices, device=device, dtype=torch.long)
        cand_embs = cause_table_embs.index_select(0, cand_idx_t)         # [P, D]
        # CEAH positive_mask uses the STRICT pathology GT (not the expanded
        # cause_emb_indices which now includes DDX alternatives), so CEAH
        # learns to discriminate pathology from DDX-alternatives-and-other-
        # conditions — the hardest decision boundary. DDX alternatives in the
        # pool become hard negatives. Fish builds without pathology_emb_idx
        # fall back to cause_emb_indices (fish has always treated all GT
        # causes as strict positives).
        pidx = q.get("pathology_emb_idx")
        if pidx is not None:
            gt_idx = [int(pidx)]
        else:
            gt_idx = list(q["cause_emb_indices"])
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
            "top_k_w": torch.tensor(top_k_raw_w, dtype=torch.float32),
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
