"""Train the CRR-DeepRVQ residual reranker.

For each training query (a train_case via leave-one-out), score every other
train case by s_first = z_q · ẑ_i, take the top-K_top candidates, and ask
the reranker to predict Δ_i so that s_final = s_first + Δ recovers the
dense ranking s_dense = z_q · z_i.

Loss = listwise KL on top-K rankings  +  λ_mse · MSE on Δ values.

Primary validation metric: sem_R@10 under (top_k_cases=1, sem_thr=0.95),
the Regime B "no-aggregation" setting where RVQ damage is measurable.

CLI (Light, M=2 K=64 — moderate-gap setting, primary validation):
    /home/lab603/anaconda3/envs/SDM/bin/python \\
        -m diagnosis_model.cause_inference.rvq_rerank.train_reranker \\
        --encoder_ckpt diagnosis_model/cause_inference/outputs/encoder_final/best_encoder.pt \\
        --case_db_dir diagnosis_model/cause_inference/outputs/case_db \\
        --rvq_M 2 --rvq_K 64 \\
        --output_dir diagnosis_model/cause_inference/outputs/rvq_rerank/reranker_M2_K64_light \\
        --variant light --K_top 50 --batch_size 64 --epochs 30
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from diagnosis_model.cause_inference.eval_phase1_aligned import evaluate
from diagnosis_model.cause_inference.phase1_baseline import (
    load_case_db, load_cases,
)
from diagnosis_model.cause_inference.train_case_encoder import encode_all
from diagnosis_model.cause_inference.rvq_rerank.fit_rvq import load_encoder
from diagnosis_model.cause_inference.rvq_rerank.rvq import RVQCodebook
from diagnosis_model.cause_inference.rvq_rerank.reranker import (
    Reranker, RerankerConfig, listwise_kl_loss,
)


def _maybe_load_or_fit_rvq(
    case_db_dir: Path,
    encoder,
    z_train: torch.Tensor,
    M: int, K: int,
    device,
    rvq_dir: Path = None,
    kmeans_iters: int = 25,
    seed: int = 42,
):
    """Load codebooks.pt if it exists, otherwise fit fresh."""
    D = z_train.size(1)
    rvq = RVQCodebook(M=M, K=K, D=D).to(device)
    if rvq_dir is not None and (rvq_dir / "codebooks.pt").exists():
        pkg = torch.load(rvq_dir / "codebooks.pt", weights_only=False, map_location=device)
        assert pkg["config"]["M"] == M and pkg["config"]["K"] == K, \
            f"codebook M/K mismatch: have {pkg['config']}, want M={M} K={K}"
        rvq.codebooks.copy_(pkg["codebooks"].to(device))
        rvq.fitted.copy_(pkg["fitted"].to(device))
        print(f"[rvq] loaded existing codebook from {rvq_dir}")
    else:
        print(f"[rvq] fitting fresh codebook M={M} K={K} ...")
        rvq.fit(z_train, n_iters=kmeans_iters, seed=seed, verbose=False)
    return rvq


def _evaluate_reranker(
    reranker: Reranker,
    z_valid: torch.Tensor,
    z_train: torch.Tensor,
    z_hat_train: torch.Tensor,
    e_train: torch.Tensor,
    codes_train: torch.Tensor,
    e_norm_train: torch.Tensor,
    train_cases: list,
    valid_cases: list,
    cause_embs: torch.Tensor,
    K_top: int,
    top_k_cases: int,
    semantic_threshold: float,
    variant: str,
    device: torch.device,
    eval_batch_size: int = 32,
) -> dict:
    """Run the Phase 1-aligned evaluate() with the reranker's sim_final."""
    reranker.eval()
    Nt = z_train.size(0)
    Nv = z_valid.size(0)
    sim_first_v = z_valid @ z_hat_train.T                  # [Nv, Nt] on device
    sim_final_v = sim_first_v.clone()

    with torch.no_grad():
        for vs in range(0, Nv, eval_batch_size):
            ve = min(vs + eval_batch_size, Nv)
            z_q = z_valid[vs:ve]
            s_first_full = z_q @ z_hat_train.T              # [bv, Nt]
            s_first_top, top_idx = s_first_full.topk(K_top, dim=-1)
            z_hat_top = z_hat_train[top_idx]
            codes_top = codes_train[top_idx]
            e_norm_top = e_norm_train[top_idx]
            z_top = z_train[top_idx] if variant == "full" else None
            e_top = e_train[top_idx] if variant == "full" else None
            delta = reranker(
                z_q, z_hat_top, codes_top, s_first_top, e_norm_top,
                z=z_top, e=e_top,
            )
            s_final_top = s_first_top + delta
            # Splice top-K results back into the full sim row
            sim_final_v[vs:ve].scatter_(1, top_idx, s_final_top)

    sim_final_np = sim_final_v.cpu().numpy()

    def score_fn(qi, q):
        return sim_final_np[qi]

    metrics = evaluate(
        score_fn, valid_cases, train_cases, cause_embs,
        top_k_cases=top_k_cases,
        semantic_threshold=semantic_threshold,
        Ks=(1, 5, 10, 20, 100),
        device=device,
    )
    reranker.train()
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder_ckpt", type=str, required=True)
    ap.add_argument("--case_db_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    # RVQ config
    ap.add_argument("--rvq_M", type=int, required=True)
    ap.add_argument("--rvq_K", type=int, required=True)
    ap.add_argument("--rvq_dir", type=str, default="",
                    help="If exists, load codebooks.pt; otherwise fit fresh.")
    ap.add_argument("--kmeans_iters", type=int, default=25)
    # Reranker config
    ap.add_argument("--variant", type=str, default="light",
                    choices=["light", "full"])
    ap.add_argument("--K_top", type=int, default=50)
    ap.add_argument("--d_hidden", type=int, default=512)
    ap.add_argument("--code_emb_dim", type=int, default=32)
    ap.add_argument("--n_attn_heads", type=int, default=8)
    ap.add_argument("--score_head_hidden", type=int, default=256)
    # Training
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--lambda_mse", type=float, default=1.0)
    ap.add_argument("--early_stop_patience", type=int, default=5)
    ap.add_argument("--eval_every_epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    # Eval config (early stop metric)
    ap.add_argument("--eval_top_k_cases", type=int, default=1,
                    help="Regime B (no-aggregation) is the default; "
                         "use 20 to early-stop on Regime A.")
    ap.add_argument("--eval_semantic_threshold", type=float, default=0.95)
    # DDXPlus scale: case-db is sharded and may be too large for full load.
    ap.add_argument("--max_train_cases", type=int, default=-1,
                    help="Cap retained train cases (RVQ + reranker training "
                         "set) via uniform per-shard subsampling. -1 loads "
                         "all (fish). DDXPlus: use 200000 to match the "
                         "Phase 1/2 convention.")
    ap.add_argument("--max_valid_cases", type=int, default=-1,
                    help="Cap valid set used for early-stop sem R@K eval. "
                         "-1 loads all (fish 3k). DDXPlus 130k valid is "
                         "intractable for the [Nv, Nt] sim matrix used here; "
                         "use e.g. 5000.")
    ap.add_argument("--sample_seed", type=int, default=42,
                    help="Subsample RNG seed; match train_case_encoder.py.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out_dir / "config.json", "w"), indent=2)

    # ---- Load encoder + data ----
    encoder, _ = load_encoder(Path(args.encoder_ckpt), device)
    case_db_dir = Path(args.case_db_dir)
    max_train = args.max_train_cases if args.max_train_cases > 0 else None
    max_valid = args.max_valid_cases if args.max_valid_cases > 0 else None
    train_cases = load_cases(
        case_db_dir, "train",
        max_cases=max_train, sample_seed=args.sample_seed,
    )
    valid_cases = load_cases(
        case_db_dir, "valid",
        max_cases=max_valid, sample_seed=args.sample_seed,
    )
    cause_pkg = torch.load(
        case_db_dir / "cause_text_embs.pt", weights_only=False, map_location="cpu",
    )
    # Upcast bf16/fp16 cause embs to fp32 (DDXPlus storage halves disk; the
    # reranker eval consumes them as fp32 via evaluate()'s F.normalize path).
    cause_embs = cause_pkg["embeddings"]
    if cause_embs.dtype in (torch.float16, torch.bfloat16):
        cause_embs = cause_embs.float()
    meta = json.load((case_db_dir / "meta.json").open())
    D = meta["global_dim"]
    print(f"[data] train={len(train_cases)}  valid={len(valid_cases)}  D={D}")

    z_train = encode_all(encoder, train_cases, device).to(device).float()
    z_valid = encode_all(encoder, valid_cases, device).to(device).float()

    # ---- Build / load RVQ + cache encoded train bank ----
    rvq_dir = Path(args.rvq_dir) if args.rvq_dir else None
    rvq = _maybe_load_or_fit_rvq(
        Path(args.case_db_dir), encoder, z_train,
        args.rvq_M, args.rvq_K, device,
        rvq_dir=rvq_dir, kmeans_iters=args.kmeans_iters, seed=args.seed,
    )
    codes_train, z_hat_train, e_train = rvq.encode(z_train)
    e_norm_train = e_train.norm(dim=-1)
    Nt = z_train.size(0)
    print(f"[rvq] M={args.rvq_M} K={args.rvq_K}  "
          f"||e|| mean={e_norm_train.mean().item():.4f}")

    # ---- Baselines for reference (computed once on z_valid) ----
    print("\n[reference] dense vs rvq_only at eval_config ...")
    sim_dense_v = (z_valid @ z_train.T).cpu().numpy()
    sim_first_v = (z_valid @ z_hat_train.T).cpu().numpy()
    m_dense = evaluate(
        lambda qi, q: sim_dense_v[qi],
        valid_cases, train_cases, cause_embs,
        top_k_cases=args.eval_top_k_cases,
        semantic_threshold=args.eval_semantic_threshold,
        Ks=(1, 5, 10, 20, 100), device=device,
    )
    m_rvq_only = evaluate(
        lambda qi, q: sim_first_v[qi],
        valid_cases, train_cases, cause_embs,
        top_k_cases=args.eval_top_k_cases,
        semantic_threshold=args.eval_semantic_threshold,
        Ks=(1, 5, 10, 20, 100), device=device,
    )
    print(f"  dense    sem_R@10={m_dense['sem_R@10']:.4f}  MRR={m_dense['sem_MRR']:.4f}")
    print(f"  rvq_only sem_R@10={m_rvq_only['sem_R@10']:.4f}  MRR={m_rvq_only['sem_MRR']:.4f}")
    print(f"  gap to recover @ R@10: {(m_dense['sem_R@10'] - m_rvq_only['sem_R@10']) * 100:+.2f} pp")

    # ---- Build reranker ----
    cfg = RerankerConfig(
        variant=args.variant, d_model=D, d_hidden=args.d_hidden,
        code_emb_dim=args.code_emb_dim, n_attn_heads=args.n_attn_heads,
        M=args.rvq_M, K=args.rvq_K,
        score_head_hidden=args.score_head_hidden,
    )
    reranker = Reranker(cfg).to(device)
    n_params = sum(p.numel() for p in reranker.parameters() if p.requires_grad)
    print(f"\n[reranker] variant={args.variant}  feat_dim={reranker.feat_dim}  "
          f"params={n_params/1e6:.2f}M")

    # ---- Optimizer + schedule ----
    optim = torch.optim.AdamW(
        reranker.parameters(),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    steps_per_epoch = Nt // args.batch_size
    total_steps = max(1, args.epochs * steps_per_epoch)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        p = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    # ---- Train ----
    log_path = out_dir / "train_log.jsonl"
    log_f = open(log_path, "w")
    best_metric = -1.0
    best_epoch = -1
    patience = 0
    global_step = 0

    for epoch in range(args.epochs):
        reranker.train()
        perm = torch.randperm(Nt, device=device)
        ep_loss = ep_kl = ep_mse = 0.0
        n_batches = 0
        t0 = time.time()

        for b_start in range(0, Nt - args.batch_size + 1, args.batch_size):
            b_idx = perm[b_start:b_start + args.batch_size]
            z_q = z_train[b_idx]                          # [B, D]

            # Full-bank scores
            s_first_full = z_q @ z_hat_train.T            # [B, Nt]
            s_dense_full = z_q @ z_train.T                # [B, Nt]

            # Mask self
            rows = torch.arange(args.batch_size, device=device)
            s_first_full[rows, b_idx] = float("-inf")
            s_dense_full[rows, b_idx] = float("-inf")

            # Top-K_top by RVQ score
            s_first_top, top_idx = s_first_full.topk(args.K_top, dim=-1)

            # Gather candidate features
            z_hat_top = z_hat_train[top_idx]              # [B, K_top, D]
            codes_top = codes_train[top_idx]              # [B, K_top, M]
            e_norm_top = e_norm_train[top_idx]            # [B, K_top]
            s_dense_top = s_dense_full.gather(1, top_idx) # [B, K_top]

            if args.variant == "full":
                z_top = z_train[top_idx]
                e_top = e_train[top_idx]
            else:
                z_top = e_top = None

            # Forward
            delta = reranker(
                z_q, z_hat_top, codes_top, s_first_top, e_norm_top,
                z=z_top, e=e_top,
            )
            s_final = s_first_top + delta
            delta_true = s_dense_top - s_first_top

            loss_kl = listwise_kl_loss(s_final, s_dense_top, temp=args.temp)
            loss_mse = F.mse_loss(delta, delta_true)
            loss = loss_kl + args.lambda_mse * loss_mse

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reranker.parameters(), 1.0)
            optim.step()
            sched.step()

            ep_loss += loss.item()
            ep_kl += loss_kl.item()
            ep_mse += loss_mse.item()
            n_batches += 1
            global_step += 1

        dt = time.time() - t0
        log_row = {
            "epoch": epoch + 1,
            "train_loss": ep_loss / max(1, n_batches),
            "loss_kl": ep_kl / max(1, n_batches),
            "loss_mse": ep_mse / max(1, n_batches),
            "lr": optim.param_groups[0]["lr"],
            "time_s": dt,
        }

        # Validate
        if (epoch + 1) % args.eval_every_epochs == 0:
            t1 = time.time()
            metrics = _evaluate_reranker(
                reranker, z_valid, z_train, z_hat_train, e_train,
                codes_train, e_norm_train,
                train_cases, valid_cases, cause_embs,
                K_top=args.K_top,
                top_k_cases=args.eval_top_k_cases,
                semantic_threshold=args.eval_semantic_threshold,
                variant=args.variant, device=device,
            )
            log_row.update({f"val_{k}": v for k, v in metrics.items()})
            log_row["val_time_s"] = time.time() - t1

            metric = metrics["sem_R@10"]
            improved = metric > best_metric
            if improved:
                best_metric = metric
                best_epoch = epoch + 1
                patience = 0
                torch.save({
                    "reranker_state": reranker.state_dict(),
                    "reranker_config": vars(cfg),
                    "metrics": metrics,
                    "epoch": epoch + 1,
                    "rvq_config": {"M": args.rvq_M, "K": args.rvq_K, "D": D},
                }, out_dir / "best.pt")
            else:
                patience += 1
            log_row["best_so_far_R@10"] = best_metric
            log_row["best_epoch"] = best_epoch
            log_row["delta_vs_dense_pp"] = (metric - m_dense["sem_R@10"]) * 100
            log_row["delta_vs_rvq_only_pp"] = (metric - m_rvq_only["sem_R@10"]) * 100

        print(json.dumps(log_row))
        log_f.write(json.dumps(log_row) + "\n")
        log_f.flush()

        if patience >= args.early_stop_patience:
            print(f"Early stop at epoch {epoch + 1}")
            break

    torch.save({
        "reranker_state": reranker.state_dict(),
        "reranker_config": vars(cfg),
        "rvq_config": {"M": args.rvq_M, "K": args.rvq_K, "D": D},
    }, out_dir / "last.pt")
    log_f.close()

    print(f"\nbest sem_R@10 = {best_metric:.4f}  @ epoch {best_epoch}")
    print(f"  dense baseline : {m_dense['sem_R@10']:.4f}")
    print(f"  rvq_only       : {m_rvq_only['sem_R@10']:.4f}")
    gap_initial = (m_dense["sem_R@10"] - m_rvq_only["sem_R@10"]) * 100
    gap_recovered = (best_metric - m_rvq_only["sem_R@10"]) * 100
    pct = (gap_recovered / gap_initial * 100) if gap_initial > 0 else 0.0
    print(f"  gap recovered  : {gap_recovered:+.2f}pp / {gap_initial:+.2f}pp "
          f"= {pct:.1f}%")


if __name__ == "__main__":
    main()
