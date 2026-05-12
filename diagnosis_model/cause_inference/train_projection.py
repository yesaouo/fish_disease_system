"""Phase 2: train a lesion projection head with cause-overlap supervision.

Pipeline (per training step):
  1. Sample a batch of B train cases.
  2. For each case: pad lesion embeddings to [max_N, D] with mask.
  3. Forward: projected_lesion = lesion_head(raw_lesion).
  4. Predicted pair-similarity matrix [B, B]:
        pred(i, j) = α · cos(g_i, g_j) + β · max_mean_set_sim(L'_i, L'_j)
     where g is raw global emb (frozen) and L' is projected lesion set.
  5. Target pair-similarity matrix [B, B]:
        target(i, j) = max_mean_set_sim(C_i, C_j)
     using raw cause embeddings (frozen).
  6. Loss = MSE on off-diagonal entries of (pred, target).
  7. Periodically eval on valid via Phase 1 baseline pipeline.

Outputs (under --output_dir):
  - best_lesion_head.pt   best checkpoint by valid semantic MRR
  - last_lesion_head.pt   final checkpoint
  - train_log.jsonl       per-epoch train loss + valid metrics
  - config.json           hyperparams
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from diagnosis_model.cause_inference.models import (
    MLPProjection, pairwise_max_mean_set_sim,
)
from diagnosis_model.cause_inference.phase1_baseline import (
    MISS_RANK,
    add_recall_at_ks,
    build_candidate_pool,
    score_candidates,
    select_positive_top_cases,
    stack_train_lesions,
    compute_case_similarities,
    summarize_rank_metric,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CasePairDataset(Dataset):
    """One sample = one case; collate_fn pads lesions and causes per batch."""

    def __init__(self, cases: list, cause_table_embs: torch.Tensor):
        self.cases = cases
        self.cause_table_embs = cause_table_embs

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> dict:
        c = self.cases[idx]
        cause_embs = self.cause_table_embs[
            torch.tensor(c["cause_emb_indices"], dtype=torch.long)
        ]
        return {
            "global_emb": c["global_emb"],          # [D]
            "lesion_embs": c["lesion_embs"],        # [N, D]
            "cause_embs": cause_embs,               # [G, D]
        }


def make_collate(D: int):
    def collate(batch: List[dict]) -> dict:
        B = len(batch)
        max_N = max(b["lesion_embs"].size(0) for b in batch)
        max_G = max(b["cause_embs"].size(0) for b in batch)

        global_embs = torch.stack([b["global_emb"] for b in batch])  # [B, D]
        lesion_pad = torch.zeros(B, max_N, D)
        lesion_mask = torch.zeros(B, max_N, dtype=torch.bool)
        cause_pad = torch.zeros(B, max_G, D)
        cause_mask = torch.zeros(B, max_G, dtype=torch.bool)

        for i, b in enumerate(batch):
            n = b["lesion_embs"].size(0)
            g = b["cause_embs"].size(0)
            lesion_pad[i, :n] = b["lesion_embs"]
            lesion_mask[i, :n] = True
            cause_pad[i, :g] = b["cause_embs"]
            cause_mask[i, :g] = True

        return {
            "global_emb": global_embs,
            "lesion_emb": lesion_pad,
            "lesion_mask": lesion_mask,
            "cause_emb": cause_pad,
            "cause_mask": cause_mask,
        }

    return collate


# ---------------------------------------------------------------------------
# Pair similarities (batched)
# ---------------------------------------------------------------------------

def compute_pred_sim(
    global_emb: torch.Tensor,    # [B, D]
    proj_lesion: torch.Tensor,   # [B, max_N, D']
    lesion_mask: torch.Tensor,   # [B, max_N]
    alpha: float, beta: float,
) -> torch.Tensor:
    """Predicted case-pair similarity.

    Make the cosine contract explicit: global embeddings and projected lesion
    embeddings are L2-normalized immediately before dot products / set-sim.
    """
    global_emb = F.normalize(global_emb, dim=-1)
    proj_lesion = F.normalize(proj_lesion, dim=-1)
    g_sim = global_emb @ global_emb.T
    l_sim = pairwise_max_mean_set_sim(proj_lesion, lesion_mask, proj_lesion, lesion_mask)
    return alpha * g_sim + beta * l_sim


def compute_target_sim(
    cause_emb: torch.Tensor,    # [B, max_G, D]
    cause_mask: torch.Tensor,   # [B, max_G]
) -> torch.Tensor:
    """Target case-pair similarity from frozen cause embeddings.

    Cause embeddings are explicitly L2-normalized so target sim is cosine set-sim.
    """
    cause_emb = F.normalize(cause_emb, dim=-1)
    return pairwise_max_mean_set_sim(cause_emb, cause_mask, cause_emb, cause_mask)


def offdiag_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    B = pred.size(0)
    eye = torch.eye(B, device=pred.device, dtype=torch.bool)
    diff = (pred - target).pow(2).masked_fill(eye, 0.0)
    return diff.sum() / (B * (B - 1))


# ---------------------------------------------------------------------------
# Inline valid eval (calls Phase 1 baseline-style retrieval with projected embs)
# ---------------------------------------------------------------------------

@torch.no_grad()
def project_all_lesions(
    train_cases: list, lesion_head: nn.Module, device: str, batch_size: int = 4096,
) -> Tuple[torch.Tensor, List[int]]:
    """Project and L2-normalize all train lesion embeddings; return stack + offsets."""
    stacked, offsets = stack_train_lesions(train_cases)
    out = []
    for i in range(0, stacked.size(0), batch_size):
        chunk = stacked[i : i + batch_size].to(device)
        out.append(F.normalize(lesion_head(chunk), dim=-1).cpu())
    return torch.cat(out, dim=0), offsets


@torch.no_grad()
def project_query_lesions(
    cases: list, lesion_head: nn.Module, device: str, batch_size: int = 4096,
) -> List[torch.Tensor]:
    """Project and L2-normalize query lesion embeddings case-by-case."""
    sizes = [c["lesion_embs"].size(0) for c in cases]
    if not sizes:
        return []
    stacked = torch.cat([c["lesion_embs"] for c in cases], dim=0)
    chunks = []
    for i in range(0, stacked.size(0), batch_size):
        chunks.append(F.normalize(lesion_head(stacked[i : i + batch_size].to(device)), dim=-1).cpu())
    flat = torch.cat(chunks, dim=0)
    out, cursor = [], 0
    for n in sizes:
        out.append(flat[cursor : cursor + n])
        cursor += n
    return out


@torch.no_grad()
def eval_valid(
    train_cases: list, valid_cases: list,
    cause_table_embs: torch.Tensor,
    cluster_id_array: np.ndarray | None,
    lesion_head: nn.Module,
    device: str,
    top_k_cases: int = 10,
    top_n_causes: int = 20,
    alpha: float = 0.25, beta: float = 0.75,
    semantic_threshold: float = 0.95,
    diversify_threshold: float = 0.95,
    max_queries: int = -1,
    ks: Tuple[int, ...] = (1, 5, 10, 20),
) -> Dict[str, float]:
    """Run Phase-1-compatible valid eval using projected lesions.

    Contract matched to phase1_baseline.py / Phase 3 eval:
      - L2-normalize embeddings before cosine dot products.
      - Retain only positive-similarity retrieved train cases.
      - Normalize retained positive case similarities before candidate scoring.
      - Use raw candidate ranking for metrics; diversification is not used here.
      - Store misses as +inf so misses contribute 0 to MRR/R@K.
      - Cluster metric is per GT cause occurrence, not per unique cluster.
    """
    del top_n_causes          # kept for CLI/API compatibility; eval returns aggregate metrics only
    del diversify_threshold   # diversification must not affect metrics

    lesion_head.eval()

    cause_table_embs = F.normalize(cause_table_embs.to(device), dim=-1)
    train_global_stack = F.normalize(
        torch.stack([c["global_emb"] for c in train_cases]).to(device), dim=-1,
    )
    train_lesion_proj, train_offsets = project_all_lesions(train_cases, lesion_head, device)
    train_lesion_proj = train_lesion_proj.to(device)

    queries = valid_cases if max_queries <= 0 else valid_cases[:max_queries]
    valid_lesion_proj = project_query_lesions(queries, lesion_head, device)

    sem_ranks: List[float] = []
    cl_ranks: List[float] = []
    cov_sem: List[int] = []
    cov_cl: List[int] = []

    for qi, q in enumerate(queries):
        q_global = F.normalize(q["global_emb"].to(device), dim=-1)
        q_lesions_proj = valid_lesion_proj[qi].to(device)

        sims = compute_case_similarities(
            q_global, q_lesions_proj,
            train_global_stack, train_lesion_proj, train_offsets,
            alpha, beta, lesion_match="hungarian",
        )

        # Phase 1 / Phase 3 contract: positive-similarity cases only, normalized weights.
        top_k_idx, top_k_w, _ = select_positive_top_cases(sims, top_k_cases)
        candidate_indices = build_candidate_pool(top_k_idx, train_cases)
        pool_size = len(candidate_indices)

        if pool_size == 0:
            for _ in q["cause_emb_indices"]:
                sem_ranks.append(MISS_RANK)
                cov_sem.append(0)
                if cluster_id_array is not None:
                    cl_ranks.append(MISS_RANK)
                    cov_cl.append(0)
            continue

        cand_idx_t = torch.tensor(candidate_indices, device=device, dtype=torch.long)
        cand_embs = cause_table_embs.index_select(0, cand_idx_t)
        cand_scores = score_candidates(
            candidate_indices, top_k_idx, top_k_w, train_cases, cause_table_embs,
        )

        # Raw ranking is used for metrics. Do not diversify here.
        raw_sorted_local = torch.argsort(cand_scores, descending=True).detach().cpu().numpy()
        raw_sorted_global = np.array(candidate_indices)[raw_sorted_local]

        # Semantic match: per GT cause occurrence.
        gt_idx_t = torch.tensor(q["cause_emb_indices"], device=device, dtype=torch.long)
        gt_embs = cause_table_embs.index_select(0, gt_idx_t)
        raw_sorted_cand_embs = cand_embs[torch.from_numpy(raw_sorted_local).to(device)]
        cos_sorted = gt_embs @ raw_sorted_cand_embs.T
        sem_match = cos_sorted >= semantic_threshold
        for g_i in range(sem_match.size(0)):
            hits = torch.nonzero(sem_match[g_i], as_tuple=False)
            if hits.numel() > 0:
                sem_ranks.append(float(hits[0].item()) + 1.0)
                cov_sem.append(1)
            else:
                sem_ranks.append(MISS_RANK)
                cov_sem.append(0)

        # Cluster match: per GT cause occurrence, matching Phase 1 denominator.
        if cluster_id_array is not None:
            raw_sorted_clusters = cluster_id_array[raw_sorted_global]
            for g in q["cause_emb_indices"]:
                cid = int(cluster_id_array[int(g)])
                hits = np.flatnonzero(raw_sorted_clusters == cid)
                if hits.size > 0:
                    cl_ranks.append(float(hits[0]) + 1.0)
                    cov_cl.append(1)
                else:
                    cl_ranks.append(MISS_RANK)
                    cov_cl.append(0)

    sem_arr = np.asarray(sem_ranks, dtype=np.float64)
    sem_block = summarize_rank_metric(sem_arr, cov_sem)
    metrics: Dict[str, float] = {
        "sem_MRR": sem_block["MRR"],
        "sem_median_rank": sem_block["median_rank"],
        "sem_mean_rank": sem_block["mean_rank"],
        "sem_coverage": sem_block["coverage"],
        "sem_n_covered": sem_block["n_covered"],
        "sem_n_missed": sem_block["n_missed"],
    }
    add_recall_at_ks(metrics, sem_arr, list(ks))
    for k in ks:
        metrics[f"sem_R@{k}"] = metrics.pop(f"R@{k}")

    if cl_ranks:
        cl_arr = np.asarray(cl_ranks, dtype=np.float64)
        cl_block = summarize_rank_metric(cl_arr, cov_cl)
        metrics.update({
            "cl_MRR": cl_block["MRR"],
            "cl_median_rank": cl_block["median_rank"],
            "cl_mean_rank": cl_block["mean_rank"],
            "cl_coverage": cl_block["coverage"],
            "cl_n_covered": cl_block["n_covered"],
            "cl_n_missed": cl_block["n_missed"],
        })
        add_recall_at_ks(metrics, cl_arr, list(ks))
        for k in ks:
            metrics[f"cl_R@{k}"] = metrics.pop(f"R@{k}")

    return metrics

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_db_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--cluster_json", type=str,
                    default="diagnosis_model/cause_inference/outputs/cause_clusters_reassigned.json")

    # architecture
    ap.add_argument("--projection_dim", type=int, default=256)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)

    # training
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--alpha_global", type=float, default=0.25)
    ap.add_argument("--beta_lesion", type=float, default=0.75)

    # eval
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--eval_top_k_cases", type=int, default=10)
    ap.add_argument("--eval_top_n_causes", type=int, default=20)
    ap.add_argument("--eval_max_queries", type=int, default=-1)
    ap.add_argument("--semantic_threshold", type=float, default=0.95)
    ap.add_argument("--diversify_threshold", type=float, default=0.95)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    # Load case DB
    case_db_dir = Path(args.case_db_dir)
    train_cases = torch.load(case_db_dir / "train_cases.pt", weights_only=False)
    valid_cases = torch.load(case_db_dir / "valid_cases.pt", weights_only=False)
    cause_pack = torch.load(case_db_dir / "cause_text_embs.pt", weights_only=False)
    cause_table_embs = cause_pack["embeddings"].to(device)
    cause_texts = cause_pack["texts"]
    print(f"[load] train={len(train_cases)}  valid={len(valid_cases)}  "
          f"unique_causes={len(cause_texts)}  dim={cause_table_embs.size(-1)}")

    # cluster array for eval
    cluster_id_array = None
    if args.cluster_json:
        with open(args.cluster_json, encoding="utf-8") as f:
            cl = json.load(f)
        o2c = cl["original_to_cause_id"]
        cluster_id_array = np.array(
            [int(o2c[t]) for t in cause_texts], dtype=np.int64,
        )
        print(f"[cluster] {len(set(cluster_id_array.tolist()))} clusters loaded")

    in_dim = cause_table_embs.size(-1)
    lesion_head = MLPProjection(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        out_dim=args.projection_dim,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in lesion_head.parameters())
    print(f"[model] lesion_head params={n_params:,}  in={in_dim}  out={args.projection_dim}")

    # NOTE: training pred-sim mixes raw global (768-d) and projected lesion (256-d).
    # That's allowed because each is a self-cosine (g_sim is [B,B] from raw global
    # cosines, l_sim is [B,B] from projected lesion cosines). The two scalars
    # combine fine with α/β weighting.

    optimizer = torch.optim.AdamW(
        lesion_head.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    train_ds = CasePairDataset(train_cases, cause_table_embs.cpu())
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=make_collate(in_dim), num_workers=args.num_workers,
        drop_last=True, pin_memory=False,
    )
    total_steps = len(train_loader) * args.epochs
    print(f"[train] steps/epoch={len(train_loader)}  total_steps={total_steps}")

    def get_lr_mult(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    log_path = out_dir / "train_log.jsonl"
    config = {
        **vars(args),
        "implementation_notes": {
            "training_objective": "In-batch all-pairs case similarity regression; loss is MSE over off-diagonal pairs only.",
            "embedding_normalization": "Global, projected lesion, and cause embeddings are L2-normalized before cosine/set-sim computations.",
            "eval_case_filter": "Validation retains only positive-similarity retrieved train cases, then normalizes their weights before candidate scoring.",
            "eval_ranking": "Validation metrics use raw candidate-pool ranking; diversification is intentionally not used for metrics.",
            "eval_missing_rank_handling": "Validation misses are +inf internally and therefore contribute 0 to MRR/R@K.",
            "cluster_metric": "Validation cluster metrics are per GT cause occurrence, not deduplicated by cluster within a query.",
        },
    }
    with (out_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)
    log_f = log_path.open("w", encoding="utf-8")

    best_mrr = -1.0
    global_step = 0
    t_train_start = time.time()
    for epoch in range(args.epochs):
        lesion_head.train()
        epoch_losses: List[float] = []
        t0 = time.time()
        for batch in train_loader:
            global_emb  = batch["global_emb"].to(device, non_blocking=True)
            lesion_emb  = batch["lesion_emb"].to(device, non_blocking=True)
            lesion_mask = batch["lesion_mask"].to(device, non_blocking=True)
            cause_emb   = batch["cause_emb"].to(device, non_blocking=True)
            cause_mask  = batch["cause_mask"].to(device, non_blocking=True)

            # Project lesions
            B, max_N, _ = lesion_emb.shape
            flat = lesion_emb.view(B * max_N, in_dim)
            proj = lesion_head(flat).view(B, max_N, args.projection_dim)

            # Predicted similarity
            pred = compute_pred_sim(
                global_emb, proj, lesion_mask,
                alpha=args.alpha_global, beta=args.beta_lesion,
            )

            # Target similarity (frozen, no grad)
            with torch.no_grad():
                target = compute_target_sim(cause_emb, cause_mask)

            loss = offdiag_mse(pred, target)

            lr_mult = get_lr_mult(global_step)
            for g in optimizer.param_groups:
                g["lr"] = args.lr * lr_mult
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lesion_head.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(float(loss.item()))
            global_step += 1

        epoch_loss = float(np.mean(epoch_losses))
        epoch_dur = time.time() - t0
        log_row = {
            "epoch": epoch,
            "train_loss_mean": epoch_loss,
            "train_loss_last": epoch_losses[-1] if epoch_losses else None,
            "epoch_seconds": epoch_dur,
            "lr_end": args.lr * lr_mult,
            "global_step": global_step,
        }
        # Eval
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            t_eval = time.time()
            metrics = eval_valid(
                train_cases, valid_cases, cause_table_embs, cluster_id_array,
                lesion_head, device,
                top_k_cases=args.eval_top_k_cases,
                top_n_causes=args.eval_top_n_causes,
                alpha=args.alpha_global, beta=args.beta_lesion,
                semantic_threshold=args.semantic_threshold,
                diversify_threshold=args.diversify_threshold,
                max_queries=args.eval_max_queries,
            )
            log_row.update(metrics)
            log_row["eval_seconds"] = time.time() - t_eval

            if metrics["sem_MRR"] > best_mrr:
                best_mrr = metrics["sem_MRR"]
                torch.save(lesion_head.state_dict(), out_dir / "best_lesion_head.pt")
                log_row["saved_best"] = True

        torch.save(lesion_head.state_dict(), out_dir / "last_lesion_head.pt")
        log_f.write(json.dumps(log_row) + "\n")
        log_f.flush()

        msg = (
            f"[epoch {epoch+1}/{args.epochs}] loss={epoch_loss:.5f}  "
            f"dur={epoch_dur:.1f}s"
        )
        if "sem_MRR" in log_row:
            msg += (
                f"  sem_MRR={log_row['sem_MRR']:.4f}  "
                f"sem_R@10={log_row.get('sem_R@10', 0):.4f}  "
                f"sem_cov={log_row['sem_coverage']:.4f}"
            )
            if log_row.get("saved_best"):
                msg += "  *best*"
        print(msg)

    log_f.close()
    print(f"\n[done] best sem_MRR={best_mrr:.4f}  "
          f"total time {(time.time()-t_train_start)/60:.1f} min")


if __name__ == "__main__":
    main()
