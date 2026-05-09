"""Phase 3: train CEAH (Cause-Evidence Attribution Head) with hard negatives.

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
        return {
            "case_idx": idx,
            "global_emb": c["global_emb"],
            "text_colloquial_emb": c.get("text_colloquial_emb"),
            "text_medical_emb": c.get("text_medical_emb"),
            "lesion_embs": c["lesion_embs"],
            "candidate_indices": p["candidate_cause_indices"],
            "positive_mask": p["positive_mask"],
        }


def make_collate(text_dropout: float, in_dim: int, cause_table_embs: torch.Tensor,
                 rng: np.random.Generator):
    """Pads lesions, samples text token, stacks (query, candidate) pairs."""

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
) -> Dict[str, float]:
    """Phase 1 retrieval → CEAH re-rank → semantic recall metrics.

    Mirrors phase1_baseline.py / eval_ceah.py contract: L2-normalize, drop
    non-positive-sim cases, raw ranking for metrics, miss=+inf.
    """
    ceah.eval()
    cause_table_embs = F.normalize(cause_table_embs, dim=-1)
    train_global_stack = F.normalize(
        torch.stack([c["global_emb"] for c in train_cases]).to(device), dim=-1,
    )
    train_lesion_stack, train_offsets = stack_train_lesions(train_cases)
    train_lesion_stack = F.normalize(train_lesion_stack.to(device), dim=-1)

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
            alpha_global, beta_lesion, lesion_match="hungarian",
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
    ap.add_argument("--attribution_mode", type=str, default="sigmoid",
                    choices=["sigmoid", "softmax"])
    ap.add_argument("--scoring_mode", type=str, default="single",
                    choices=["single", "multiplicative"])

    # training
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--lambda_sparsity", type=float, default=0.05)
    ap.add_argument("--text_dropout", type=float, default=0.5)

    # eval
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--eval_top_k_cases", type=int, default=20)
    ap.add_argument("--eval_max_queries", type=int, default=300)
    ap.add_argument("--eval_text_kind", type=str, default="medical")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    # Load data
    case_db_dir = Path(args.case_db_dir)
    train_cases = torch.load(case_db_dir / "train_cases.pt", weights_only=False)
    valid_cases = torch.load(case_db_dir / "valid_cases.pt", weights_only=False)
    cause_pack = torch.load(case_db_dir / "cause_text_embs.pt", weights_only=False)
    cause_table_embs_cpu = cause_pack["embeddings"]
    cause_table_embs = cause_table_embs_cpu.to(device)
    pool_payload = torch.load(args.train_pool_path, weights_only=False)
    train_pool = pool_payload["case_pool"]
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
        collate_fn=make_collate(args.text_dropout, in_dim, cause_table_embs_cpu, rng),
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

            loss = loss_cls + args.lambda_sparsity * loss_sp

            lr_mult = get_lr_mult(global_step)
            for g in optimizer.param_groups:
                g["lr"] = args.lr * lr_mult
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ceah.parameters(), 1.0)
            optimizer.step()

            ep_losses_cls.append(float(loss_cls.item()))
            ep_losses_sp.append(float(loss_sp.item()))

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
            f"sp={log_row['loss_sp']:.4f}  "
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
