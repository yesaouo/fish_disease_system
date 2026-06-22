"""Phase 2: train CEAH (Cause-Evidence Attribution Head) with hard negatives.

Each train query brings:
  - evidence: global_emb, optional text_emb (with dropout), lesion_embs
  - candidate cause pool: ~87 candidates from the offline-built pool, each
    labeled positive (semantic-equivalent to a GT cause) or negative

Training (per (query, candidate) pair):
  score = CEAH(evidence, candidate_emb)
  L_cls       = BCE(score, y)
  L_sparsity  = mean(alpha[valid evidence positions])
  L_total     = L_cls + lambda_sparsity * L_sparsity

Validation: re-rank Phase 1 valid pool using CEAH score, track sem_R@K + sem_MRR.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from diagnosis_model.cause_inference.models import CEAH
from diagnosis_model.cause_inference.phase1_baseline import (
    MISS_RANK,
    add_recall_at_ks,
    build_candidate_pool,
    compute_case_similarities,
    load_cases,
    offsets_to_case_ids,
    select_positive_top_cases,
    stack_train_lesions,
    summarize_rank_metric,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CEAHDataset(Dataset):
    """Yields (case_idx, candidate_pool_info) per __getitem__.

    The collate stacks evidence + candidate emb pairs into a flat batch:
      after collate: tensors are shaped to one row per (query, candidate).
    """

    def __init__(self, train_cases: list, train_pool: list, cause_table_embs: torch.Tensor):
        self.cases = train_cases
        self.pool = train_pool
        self.cause_table_embs = cause_table_embs

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> dict:
        c = self.cases[idx]
        p = self.pool[idx]
        # Soft-label GT mirrors positive_mask semantics: strict pathology only
        # for DDXPlus (so high-cosine to DDX alternatives doesn't get treated
        # as positive), fall back to the expanded cause_emb_indices for fish
        # (all GT causes are strict positives).
        pidx = c.get("pathology_emb_idx")
        gt_cause_indices = [int(pidx)] if pidx is not None else c["cause_emb_indices"]
        return {
            "case_idx": idx,
            "global_emb": c["global_emb"],
            "text_colloquial_emb": c.get("text_colloquial_emb"),
            "text_medical_emb": c.get("text_medical_emb"),
            "lesion_embs": c["lesion_embs"],
            "candidate_indices": p["candidate_cause_indices"],
            "positive_mask": p["positive_mask"],
            "gt_cause_indices": gt_cause_indices,
        }


def make_collate(text_dropout: float, in_dim: int, cause_table_embs: torch.Tensor,
                 rng: np.random.Generator,
                 soft_labels: bool = False, soft_lo: float = 0.90, soft_hi: float = 1.0):
    """Pads lesions, samples text token, stacks (query, candidate) pairs.

    When soft_labels, the per-candidate target is a graded relevance
    clamp((max_g cos(cand, gt_g) - soft_lo) / (soft_hi - soft_lo), 0, 1)
    (cosine on L2-normalized cause embeddings, consistent with positive_mask)
    instead of the hard 0/1 positive_mask. Model inputs are unchanged.
    """

    def collate(batch: List[dict]) -> dict:
        B = len(batch)
        max_N = max(b["lesion_embs"].size(0) for b in batch)
        max_P = max(int(b["candidate_indices"].numel()) for b in batch)
        D = in_dim

        global_pad = torch.zeros(B, D)
        lesion_pad = torch.zeros(B, max_N, D)
        lesion_mask = torch.zeros(B, max_N, dtype=torch.bool)

        text_pad = torch.zeros(B, D)
        text_present = torch.zeros(B, dtype=torch.bool)

        cand_pad = torch.zeros(B, max_P, D)
        cand_mask = torch.zeros(B, max_P, dtype=torch.bool)
        target_pad = torch.zeros(B, max_P, dtype=torch.float32)

        for i, b in enumerate(batch):
            global_pad[i] = b["global_emb"]
            n = b["lesion_embs"].size(0)
            lesion_pad[i, :n] = b["lesion_embs"]
            lesion_mask[i, :n] = True

            # Text token: random pick colloquial vs medical, then drop with prob
            tcoll = b.get("text_colloquial_emb")
            tmed  = b.get("text_medical_emb")
            options = [t for t in (tcoll, tmed) if t is not None]
            if options and rng.random() >= text_dropout:
                t = options[int(rng.integers(len(options)))]
                text_pad[i] = t
                text_present[i] = True

            cand_idx = b["candidate_indices"]
            P = int(cand_idx.numel())
            if P > 0:
                cand_pad[i, :P] = cause_table_embs[cand_idx]
            cand_mask[i, :P] = True
            if soft_labels and P > 0:
                gt_idx = torch.as_tensor(b["gt_cause_indices"], dtype=torch.long)
                gt_n = F.normalize(cause_table_embs[gt_idx], dim=-1)        # [G, D]
                cand_n = F.normalize(cause_table_embs[cand_idx], dim=-1)    # [P, D]
                maxcos = (gt_n @ cand_n.T).max(dim=0).values               # [P]
                tgt = ((maxcos - soft_lo) / (soft_hi - soft_lo)).clamp(0.0, 1.0)
                target_pad[i, :P] = tgt
            else:
                target_pad[i, :P] = b["positive_mask"].float()

        return {
            "global_emb": global_pad,
            "text_emb": text_pad,
            "text_present": text_present,
            "lesion_embs": lesion_pad,
            "lesion_mask": lesion_mask,
            "cand_embs": cand_pad,
            "cand_mask": cand_mask,
            "targets": target_pad,
        }

    return collate


# ---------------------------------------------------------------------------
# Validation eval (re-rank Phase 1 pool with CEAH score)
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_valid_with_ceah(
    train_cases: list, valid_cases: list,
    cause_table_embs: torch.Tensor,
    ceah: nn.Module,
    device: str,
    top_k_cases: int = 20,
    semantic_threshold: float = 0.95,
    diversify_threshold: float = 0.95,
    use_text_kind: str = "medical",
    max_queries: int = -1,
    ks=(1, 5, 10, 20),
    alpha_global: float = 0.25,
    beta_lesion: float = 0.75,
    lesion_match: str = "max_mean",
    bank_dtype: torch.dtype = torch.float32,
) -> Dict[str, float]:
    """Phase 1 retrieval → CEAH re-rank → semantic recall metrics.

    Mirrors phase1_baseline.py / eval_ceah.py contract: L2-normalize, drop
    non-positive-sim cases, raw ranking for metrics, miss=+inf.

    ``bank_dtype`` controls the on-device dtype of the validation Phase 1 bank
    (mirrors :func:`load_train_bank`'s parameter). fp32 keeps the historic fish
    path; bf16 is needed for DDXPlus 200k subsample to fit comfortably on a
    32 GB GPU alongside CEAH forward.
    """
    ceah.eval()
    cause_table_embs = F.normalize(cause_table_embs, dim=-1)
    train_global_stack = F.normalize(
        torch.stack([c["global_emb"] for c in train_cases]).to(device, dtype=bank_dtype),
        dim=-1,
    )
    train_lesion_stack, train_offsets = stack_train_lesions(train_cases)
    train_lesion_stack = F.normalize(
        train_lesion_stack.to(device, dtype=bank_dtype), dim=-1,
    )
    train_case_ids = offsets_to_case_ids(train_offsets, train_lesion_stack.device)

    queries = valid_cases if max_queries <= 0 else valid_cases[:max_queries]
    sem_ranks: List[float] = []
    cov: List[int] = []
    in_dim = cause_table_embs.size(-1)
    text_key = f"text_{use_text_kind}_emb"

    for q in queries:
        q_global = F.normalize(q["global_emb"].to(device), dim=-1)
        q_lesions = F.normalize(q["lesion_embs"].to(device), dim=-1)

        sims = compute_case_similarities(
            q_global, q_lesions,
            train_global_stack, train_lesion_stack, train_offsets,
            alpha_global, beta_lesion, lesion_match=lesion_match,
            train_case_ids=train_case_ids,
        )
        top_k_idx, _, _ = select_positive_top_cases(sims, top_k_cases)
        candidate_indices = build_candidate_pool(top_k_idx, train_cases)
        pool_size = len(candidate_indices)
        if pool_size == 0:
            for _ in q["cause_emb_indices"]:
                sem_ranks.append(MISS_RANK)
                cov.append(0)
            continue

        # Build CEAH inputs for this query (B=1, run all candidates as the
        # cause-axis batch by replicating evidence across candidates).
        # CEAH calls .view(B*N, -1) internally, so expanded tensors must be
        # materialized contiguous before being passed in.
        global_emb = q_global.unsqueeze(0).expand(pool_size, -1).contiguous()
        n_les = q_lesions.size(0)
        lesion_embs = q_lesions.unsqueeze(0).expand(pool_size, -1, -1).contiguous()
        lesion_mask = torch.ones(pool_size, n_les, dtype=torch.bool, device=device)

        t = q.get(text_key)
        if t is not None:
            text_emb = t.to(device).unsqueeze(0).expand(pool_size, -1).contiguous()
            text_present = torch.ones(pool_size, dtype=torch.bool, device=device)
        else:
            text_emb = torch.zeros(pool_size, in_dim, device=device)
            text_present = torch.zeros(pool_size, dtype=torch.bool, device=device)

        cand_idx_t = torch.tensor(candidate_indices, device=device, dtype=torch.long)
        cand_embs = cause_table_embs.index_select(0, cand_idx_t)  # [P, D]

        scores, _, _ = ceah(global_emb, text_emb, text_present,
                            lesion_embs, lesion_mask, cand_embs)
        # Raw ranking is used for metrics (no diversification).
        raw_sorted_local = torch.argsort(scores, descending=True).detach().cpu().numpy()

        gt_idx_t = torch.tensor(q["cause_emb_indices"], device=device, dtype=torch.long)
        gt_embs = cause_table_embs.index_select(0, gt_idx_t)
        raw_sorted_cand_embs = cand_embs[torch.from_numpy(raw_sorted_local).to(device)]
        cos_sorted = gt_embs @ raw_sorted_cand_embs.T
        sem_match = cos_sorted >= semantic_threshold
        for g_i in range(sem_match.size(0)):
            hits = torch.nonzero(sem_match[g_i], as_tuple=False)
            if hits.numel() > 0:
                sem_ranks.append(float(hits[0].item()) + 1.0)
                cov.append(1)
            else:
                sem_ranks.append(MISS_RANK)
                cov.append(0)

    arr = np.asarray(sem_ranks, dtype=np.float64)
    block = summarize_rank_metric(arr, cov)
    metrics = {
        "sem_MRR": block["MRR"],
        "sem_median_rank": block["median_rank"],
        "sem_mean_rank": block["mean_rank"],
        "sem_coverage": block["coverage"],
        "sem_n_covered": block["n_covered"],
        "sem_n_missed": block["n_missed"],
    }
    add_recall_at_ks(metrics, arr, list(ks))
    # Rename R@k -> sem_R@k
    for k in ks:
        metrics[f"sem_R@{k}"] = metrics.pop(f"R@{k}")
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_db_dir", type=str, required=True)
    ap.add_argument("--train_pool_path", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)

    # architecture
    ap.add_argument("--common_dim", type=int, default=256)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--attribution_mode", type=str, default="softmax",
                    choices=["sigmoid", "softmax"])
    ap.add_argument("--scoring_mode", type=str, default="multiplicative",
                    choices=["single", "multiplicative"])

    # training
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--lambda_sparsity", type=float, default=0.0)
    ap.add_argument("--text_dropout", type=float, default=0.5)
    # Rung 0: graded soft labels (ramp on max GT cosine) instead of hard positive_mask
    ap.add_argument("--soft_labels", action="store_true")
    ap.add_argument("--soft_lo", type=float, default=0.90)
    ap.add_argument("--soft_hi", type=float, default=1.0)
    # Rung 1: multi-positive listwise CE (cross-candidate competition) on top of BCE
    ap.add_argument("--lambda_rank", type=float, default=0.0)
    ap.add_argument("--rank_temp", type=float, default=0.1)

    # eval
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--eval_top_k_cases", type=int, default=20)
    ap.add_argument("--eval_max_queries", type=int, default=300)
    ap.add_argument("--eval_text_kind", type=str, default="medical")
    ap.add_argument("--eval_lesion_match", type=str, default="max_mean",
                    choices=["hungarian", "max_mean", "max_mean_normalized"],
                    help="Lesion-set match for validation Phase 1 retrieval. "
                         "Default 'max_mean' is the new global default (vectorized GPU, "
                         "≤0.5pp from hungarian on fish per ablations/lesion_match_ranking_equiv; "
                         "hungarian retained for paper-reproduction runs only).")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bank_dtype", type=str, default="fp32",
                    choices=["fp32", "fp16", "bf16"],
                    help="On-device storage dtype of the validation train bank. "
                         "Default 'fp32' preserves the historic fish workflow; "
                         "pass 'bf16' for DDXPlus (200k subsample is ~12 GB fp32 "
                         "/ ~6 GB bf16 — bf16 leaves more headroom for CEAH "
                         "forward and per-query buffers).")
    ap.add_argument("--max_train_cases", type=int, default=0,
                    help="Cap on retained train cases for both the CEAH training "
                         "set (CEAHDataset) and the validation Phase 1 bank. "
                         "Default 0 = full train (correct for fish). For DDXPlus "
                         "pass 200000 — the full 1M train ~64 GB CPU RAM in "
                         "load_cases (fp32 upcast) blows past 62 GB. MUST equal "
                         "the --max_train_cases used when building train_pool, "
                         "and share --sample_seed, so CEAHDataset[idx] case "
                         "aligns with train_pool[idx] candidate set.")
    ap.add_argument("--sample_seed", type=int, default=42,
                    help="Must match build_train_candidate_pool's --sample_seed "
                         "to keep CEAHDataset[idx] case aligned with "
                         "train_pool[idx] candidate set.")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    bank_dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    bank_dtype = bank_dtype_map[args.bank_dtype]
    max_train_cases = args.max_train_cases if args.max_train_cases > 0 else None

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    # Load data. Subsample train cases iff --max_train_cases is set — uses the
    # same per-shard rng.choice as build_train_candidate_pool / load_train_bank
    # with the same sample_seed, so the kept 200k subset is identical across
    # the three scripts (preserves CEAHDataset[idx] ↔ train_pool[idx] alignment).
    case_db_dir = Path(args.case_db_dir)
    train_cases = load_cases(
        case_db_dir, "train",
        max_cases=max_train_cases, sample_seed=args.sample_seed,
    )
    valid_cases = load_cases(case_db_dir, "valid")
    cause_pack = torch.load(case_db_dir / "cause_text_embs.pt", weights_only=False)
    # .float() upcasts DDXPlus bf16/fp16 storage so CEAH's fp32 nn.Linear is
    # dtype-compatible; no-op for fp32 fish builds.
    cause_table_embs_cpu = cause_pack["embeddings"].float()
    cause_table_embs = cause_table_embs_cpu.to(device)
    pool_payload = torch.load(args.train_pool_path, weights_only=False)
    train_pool = pool_payload["case_pool"]
    if len(train_cases) != len(train_pool):
        raise ValueError(
            f"train_cases ({len(train_cases)}) and train_pool ({len(train_pool)}) "
            f"size mismatch — CEAHDataset[idx] assumes positional alignment. "
            f"Pass --max_train_cases / --sample_seed matching the values used "
            f"when running build_train_candidate_pool (default seed=42)."
        )
    print(f"[load] train={len(train_cases)}  valid={len(valid_cases)}  "
          f"causes={cause_table_embs.size(0)}  pool_entries={len(train_pool)}")

    in_dim = cause_table_embs.size(-1)
    ceah = CEAH(
        global_dim=in_dim, text_dim=in_dim, lesion_dim=in_dim, cause_dim=in_dim,
        common_dim=args.common_dim, hidden_dim=args.hidden_dim, dropout=args.dropout,
        attribution_mode=args.attribution_mode, scoring_mode=args.scoring_mode,
    ).to(device)
    n_params = sum(p.numel() for p in ceah.parameters())
    print(f"[model] CEAH params={n_params:,}")

    optimizer = torch.optim.AdamW(
        ceah.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    train_ds = CEAHDataset(train_cases, train_pool, cause_table_embs_cpu)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=make_collate(args.text_dropout, in_dim, cause_table_embs_cpu, rng,
                                soft_labels=args.soft_labels,
                                soft_lo=args.soft_lo, soft_hi=args.soft_hi),
        num_workers=args.num_workers, drop_last=True, pin_memory=False,
    )
    total_steps = len(train_loader) * args.epochs
    print(f"[train] steps/epoch={len(train_loader)}  total_steps={total_steps}")

    def get_lr_mult(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    log_path = out_dir / "train_log.jsonl"
    with (out_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)
    log_f = log_path.open("w", encoding="utf-8")

    best_mrr = -1.0
    global_step = 0
    t_start = time.time()

    for epoch in range(args.epochs):
        ceah.train()
        ep_losses_cls: List[float] = []
        ep_losses_sp: List[float] = []
        ep_losses_rank: List[float] = []
        ep_pos_mean: List[float] = []
        ep_neg_mean: List[float] = []
        ep_alpha_mean: List[float] = []
        t0 = time.time()
        for batch in train_loader:
            global_emb   = batch["global_emb"].to(device, non_blocking=True)        # [B, D]
            text_emb     = batch["text_emb"].to(device, non_blocking=True)          # [B, D]
            text_present = batch["text_present"].to(device, non_blocking=True)
            lesion_embs  = batch["lesion_embs"].to(device, non_blocking=True)       # [B, max_N, D]
            lesion_mask  = batch["lesion_mask"].to(device, non_blocking=True)
            cand_embs    = batch["cand_embs"].to(device, non_blocking=True)         # [B, max_P, D]
            cand_mask    = batch["cand_mask"].to(device, non_blocking=True)
            targets      = batch["targets"].to(device, non_blocking=True)           # [B, max_P]

            B, max_P, _ = cand_embs.shape

            # Replicate evidence across candidates → flatten to (B*max_P) rows
            n_les = lesion_embs.size(1)
            global_rep = global_emb.unsqueeze(1).expand(B, max_P, in_dim).reshape(B * max_P, in_dim)
            text_rep   = text_emb.unsqueeze(1).expand(B, max_P, in_dim).reshape(B * max_P, in_dim)
            tp_rep     = text_present.unsqueeze(1).expand(B, max_P).reshape(B * max_P)
            lesion_rep = lesion_embs.unsqueeze(1).expand(B, max_P, n_les, in_dim).reshape(B * max_P, n_les, in_dim)
            lmask_rep  = lesion_mask.unsqueeze(1).expand(B, max_P, n_les).reshape(B * max_P, n_les)
            cand_flat  = cand_embs.reshape(B * max_P, in_dim)

            scores, alphas, ev_mask = ceah(
                global_rep, text_rep, tp_rep, lesion_rep, lmask_rep, cand_flat,
            )                                                       # scores [B*max_P]
            scores = scores.view(B, max_P)
            alphas = alphas.view(B, max_P, -1)                      # [B, max_P, max_Ne]
            ev_mask = ev_mask.view(B, max_P, -1)

            # BCE on valid candidate positions
            cand_mask_f = cand_mask.float()
            bce = F.binary_cross_entropy(scores, targets, reduction="none")
            loss_cls = (bce * cand_mask_f).sum() / cand_mask_f.sum().clamp_min(1.0)

            # Sparsity: mean of alpha at valid (cand × evidence) positions
            valid_alpha = alphas * ev_mask.float()
            denom = (cand_mask_f.unsqueeze(-1) * ev_mask.float()).sum().clamp_min(1.0)
            loss_sp = (valid_alpha * cand_mask_f.unsqueeze(-1)).sum() / denom

            # Rung 1: multi-positive listwise CE — softmax(score/T) over the pool,
            # target = uniform over the binary positives. Adds cross-candidate
            # competition (sharpens top) on top of the per-candidate BCE.
            loss_rank = scores.new_zeros(())
            if args.lambda_rank > 0:
                logits = (scores / args.rank_temp).masked_fill(~cand_mask, float("-inf"))
                logp = F.log_softmax(logits, dim=1)                  # [B, max_P]
                pos = (targets > 0.5) & cand_mask                    # binary positives
                n_pos = pos.sum(dim=1)                               # [B]
                # torch.where (not logp*pos) to avoid 0*(-inf)=nan at padded positions
                per_q = -torch.where(pos, logp, torch.zeros_like(logp)).sum(dim=1) / n_pos.clamp_min(1)
                valid_q = n_pos > 0
                if valid_q.any():
                    loss_rank = per_q[valid_q].mean()

            loss = loss_cls + args.lambda_rank * loss_rank + args.lambda_sparsity * loss_sp

            lr_mult = get_lr_mult(global_step)
            for g in optimizer.param_groups:
                g["lr"] = args.lr * lr_mult
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ceah.parameters(), 1.0)
            optimizer.step()

            ep_losses_cls.append(float(loss_cls.item()))
            ep_losses_sp.append(float(loss_sp.item()))
            ep_losses_rank.append(float(loss_rank.item()) if torch.is_tensor(loss_rank) else float(loss_rank))

            # Diagnostics
            with torch.no_grad():
                pos_mask = (targets > 0.5) & cand_mask
                neg_mask = (targets < 0.5) & cand_mask
                if pos_mask.any():
                    ep_pos_mean.append(float(scores[pos_mask].mean().item()))
                if neg_mask.any():
                    ep_neg_mean.append(float(scores[neg_mask].mean().item()))
                ep_alpha_mean.append(float(loss_sp.item()))

            global_step += 1

        ep_dur = time.time() - t0
        log_row = {
            "epoch": epoch,
            "loss_cls": float(np.mean(ep_losses_cls)),
            "loss_sp": float(np.mean(ep_losses_sp)),
            "loss_rank": float(np.mean(ep_losses_rank)),
            "pos_score_mean": float(np.mean(ep_pos_mean)) if ep_pos_mean else None,
            "neg_score_mean": float(np.mean(ep_neg_mean)) if ep_neg_mean else None,
            "alpha_mean": float(np.mean(ep_alpha_mean)),
            "epoch_seconds": ep_dur,
            "lr_end": args.lr * lr_mult,
            "global_step": global_step,
        }

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            t_eval = time.time()
            metrics = eval_valid_with_ceah(
                train_cases, valid_cases, cause_table_embs, ceah, device,
                top_k_cases=args.eval_top_k_cases,
                max_queries=args.eval_max_queries,
                use_text_kind=args.eval_text_kind,
                lesion_match=args.eval_lesion_match,
                bank_dtype=bank_dtype,
            )
            log_row.update(metrics)
            log_row["eval_seconds"] = time.time() - t_eval
            if metrics["sem_MRR"] > best_mrr:
                best_mrr = metrics["sem_MRR"]
                torch.save(ceah.state_dict(), out_dir / "best_ceah.pt")
                log_row["saved_best"] = True

        torch.save(ceah.state_dict(), out_dir / "last_ceah.pt")
        log_f.write(json.dumps(log_row) + "\n")
        log_f.flush()

        msg = (
            f"[ep {epoch+1}/{args.epochs}] cls={log_row['loss_cls']:.4f} "
            f"rank={log_row['loss_rank']:.4f} sp={log_row['loss_sp']:.4f}  "
            f"pos={log_row['pos_score_mean']:.3f} neg={log_row['neg_score_mean']:.3f}  "
            f"α̅={log_row['alpha_mean']:.3f}  dur={ep_dur:.1f}s"
        )
        if "sem_MRR" in log_row:
            msg += (f"  | sem_MRR={log_row['sem_MRR']:.3f} "
                    f"R@10={log_row.get('sem_R@10', 0):.3f}")
            if log_row.get("saved_best"):
                msg += " *best*"
        print(msg)

    log_f.close()
    print(f"\n[done] best sem_MRR={best_mrr:.4f}  total {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
