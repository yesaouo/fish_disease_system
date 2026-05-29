"""Final evaluation matrix for CRR-DeepRVQ.

For each (M, K) RVQ config, compares four scorers under multiple eval
regimes:

    dense          : q · z_i                          (uncompressed baseline)
    rvq_only       : q · ẑ_i                          (no reranker)
    light_rerank   : s_first + Δ_light(q, ẑ, codes, ‖e‖)   (trained)
    full_analytic  : s_first + Δ = q · e_i           (oracle, no training)

Eval regimes default to:
    Regime A: top_k_cases=20, sem_thr=0.95   (production aggregation buffer)
    Regime B: top_k_cases=1,  sem_thr=0.95   (no-aggregation, ANN-style)

CLI from repo root:
    /home/lab603/anaconda3/envs/SDM/bin/python \\
        -m diagnosis_model.cause_inference.rvq_rerank.eval_final
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch

from diagnosis_model.cause_inference.eval_phase1_aligned import evaluate
from diagnosis_model.cause_inference.phase1_baseline import (
    load_case_db, load_cases,
)
from diagnosis_model.cause_inference.train_case_encoder import encode_all
from diagnosis_model.cause_inference.rvq_rerank.fit_rvq import load_encoder
from diagnosis_model.cause_inference.rvq_rerank.rvq import RVQCodebook
from diagnosis_model.cause_inference.rvq_rerank.reranker import (
    Reranker, RerankerConfig,
)


def _parse_pair(s: str, types=(int, float)) -> Tuple:
    parts = s.split(",")
    return tuple(t(p) for t, p in zip(types, parts))


@torch.no_grad()
def _rerank_sim(
    reranker: Reranker,
    z_q_all: torch.Tensor,
    z_hat_train: torch.Tensor,
    z_train: torch.Tensor,
    e_train: torch.Tensor,
    codes_train: torch.Tensor,
    e_norm_train: torch.Tensor,
    K_top: int,
    variant: str,
    batch_size: int = 32,
) -> torch.Tensor:
    """Build sim_final = s_first with top-K positions replaced by reranked."""
    reranker.eval()
    sim_first = z_q_all @ z_hat_train.T              # [Nq, Nt]
    sim_final = sim_first.clone()
    Nq = z_q_all.size(0)
    for vs in range(0, Nq, batch_size):
        ve = min(vs + batch_size, Nq)
        z_q = z_q_all[vs:ve]
        s_first_full = z_q @ z_hat_train.T
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
        sim_final[vs:ve].scatter_(1, top_idx, s_final_top)
    return sim_final


@torch.no_grad()
def _analytic_full_sim(
    z_q_all: torch.Tensor,
    z_hat_train: torch.Tensor,
    e_train: torch.Tensor,
    K_top: int,
    batch_size: int = 32,
) -> torch.Tensor:
    """sim_final = s_first + q·e on top-K. Equivalent to dense for top-K
    positions, so reduces to 'dense rerank top-K of RVQ first stage'."""
    sim_first = z_q_all @ z_hat_train.T
    sim_final = sim_first.clone()
    Nq = z_q_all.size(0)
    for vs in range(0, Nq, batch_size):
        ve = min(vs + batch_size, Nq)
        z_q = z_q_all[vs:ve]
        s_first_full = z_q @ z_hat_train.T
        s_first_top, top_idx = s_first_full.topk(K_top, dim=-1)
        e_top = e_train[top_idx]                      # [bv, K_top, D]
        delta = (z_q.unsqueeze(1) * e_top).sum(dim=-1)  # [bv, K_top]
        s_final_top = s_first_top + delta
        sim_final[vs:ve].scatter_(1, top_idx, s_final_top)
    return sim_final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder_ckpt", type=str,
                    default="diagnosis_model/cause_inference/outputs/"
                            "encoder_final/best_encoder.pt")
    ap.add_argument("--case_db_dir", type=str,
                    default="diagnosis_model/cause_inference/outputs/case_db")
    ap.add_argument("--rvq_root", type=str,
                    default="diagnosis_model/cause_inference/outputs/rvq_rerank")
    # (M,K,reranker_subdir) tuples
    ap.add_argument("--configs", nargs="+", type=str,
                    default=[
                        "4,256,reranker_M4_K256_light",
                        "2,64,reranker_M2_K64_light",
                        "1,16,reranker_M1_K16_light",
                    ])
    ap.add_argument("--eval_configs", nargs="+", type=str,
                    default=["20,0.95", "1,0.95"],
                    help="(top_k_cases, sem_thr)")
    ap.add_argument("--Ks", type=int, nargs="+", default=[1, 5, 10, 20, 100])
    ap.add_argument("--K_top", type=int, default=50,
                    help="Reranker top-K (must match training)")
    ap.add_argument("--kmeans_iters", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_train_cases", type=int, default=-1,
                    help="Cap retained train cases for the RVQ bank. Must "
                         "match the value used in fit_rvq.py / "
                         "train_reranker.py so codebook indices line up.")
    ap.add_argument("--max_valid_cases", type=int, default=-1,
                    help="Cap valid set used as queries. -1 loads all. "
                         "DDXPlus 130k valid produces [Nv, Nt] sim matrices "
                         "(~100 GB) here; use e.g. 5000 for a tractable scan.")
    ap.add_argument("--sample_seed", type=int, default=42,
                    help="Subsample RNG seed; must match earlier stages.")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rvq_root = Path(args.rvq_root)

    eval_pairs = [_parse_pair(s, (int, float)) for s in args.eval_configs]
    configs = []
    for spec in args.configs:
        parts = spec.split(",")
        M, K, sub = int(parts[0]), int(parts[1]), parts[2]
        configs.append((M, K, sub))

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
    cause_embs = cause_pkg["embeddings"]
    if cause_embs.dtype in (torch.float16, torch.bfloat16):
        cause_embs = cause_embs.float()
    meta = json.load((case_db_dir / "meta.json").open())
    D = meta["global_dim"]
    print(f"[data] train={len(train_cases)}  valid={len(valid_cases)}  D={D}")

    z_train = encode_all(encoder, train_cases, device).to(device).float()
    z_valid = encode_all(encoder, valid_cases, device).to(device).float()
    sim_dense = z_valid @ z_train.T

    Ks = tuple(args.Ks)

    rows = []

    # Dense baseline (independent of RVQ config)
    for top_k, sem_thr in eval_pairs:
        sim_np = sim_dense.cpu().numpy()
        m = evaluate(
            lambda qi, q: sim_np[qi],
            valid_cases, train_cases, cause_embs,
            top_k_cases=top_k, semantic_threshold=sem_thr,
            Ks=Ks, device=device,
        )
        rows.append({"method": "dense", "M": -1, "K": -1,
                     "compression_x": 1.0,
                     "top_k": top_k, "sem_thr": sem_thr, **m})

    for M, K, sub in configs:
        # Fit / encode RVQ
        # Look for existing codebooks.pt under rvq_root/rvq_M{M}_K{K}/
        rvq_dir = rvq_root / f"rvq_M{M}_K{K}"
        rvq = RVQCodebook(M=M, K=K, D=D).to(device)
        if (rvq_dir / "codebooks.pt").exists():
            pkg = torch.load(rvq_dir / "codebooks.pt",
                             weights_only=False, map_location=device)
            rvq.codebooks.copy_(pkg["codebooks"].to(device))
            rvq.fitted.copy_(pkg["fitted"].to(device))
            print(f"[rvq M={M} K={K}] loaded codebook")
        else:
            print(f"[rvq M={M} K={K}] fitting fresh ...")
            rvq.fit(z_train, n_iters=args.kmeans_iters, seed=args.seed,
                    verbose=False)

        codes_train, z_hat_train, e_train = rvq.encode(z_train)
        e_norm_train = e_train.norm(dim=-1)

        import math
        compression_x = (D * 32) / (M * max(1, math.ceil(math.log2(max(K, 2)))))

        # --- rvq_only sim ---
        sim_rvq = z_valid @ z_hat_train.T

        # --- light reranker sim ---
        sub_dir = rvq_root / sub
        light_ckpt = sub_dir / "best.pt"
        sim_light = None
        if light_ckpt.exists():
            ckpt = torch.load(light_ckpt, weights_only=False, map_location=device)
            cfg_dict = dict(ckpt["reranker_config"])
            cfg = RerankerConfig(**cfg_dict)
            reranker = Reranker(cfg).to(device)
            reranker.load_state_dict(ckpt["reranker_state"])
            sim_light = _rerank_sim(
                reranker, z_valid, z_hat_train, z_train, e_train,
                codes_train, e_norm_train, K_top=args.K_top,
                variant=cfg.variant,
            )
            print(f"[rerank M={M} K={K}] loaded {sub_dir} "
                  f"(ep={ckpt.get('epoch','?')}, val_R@10="
                  f"{ckpt['metrics']['sem_R@10']:.4f})")
        else:
            print(f"[rerank M={M} K={K}] no checkpoint at {light_ckpt}, skipping")

        # --- full analytic sim ---
        sim_full_analytic = _analytic_full_sim(
            z_valid, z_hat_train, e_train, K_top=args.K_top,
        )

        for top_k, sem_thr in eval_pairs:
            for name, sim in [
                ("rvq_only", sim_rvq),
                ("light", sim_light),
                ("full_analytic", sim_full_analytic),
            ]:
                if sim is None:
                    continue
                sim_np = sim.cpu().numpy()
                m = evaluate(
                    lambda qi, q, _s=sim_np: _s[qi],
                    valid_cases, train_cases, cause_embs,
                    top_k_cases=top_k, semantic_threshold=sem_thr,
                    Ks=Ks, device=device,
                )
                rows.append({
                    "method": name, "M": M, "K": K,
                    "compression_x": compression_x,
                    "top_k": top_k, "sem_thr": sem_thr, **m,
                })

    # ---- Tabular summary ----
    print("\n\n" + "=" * 104)
    for top_k, sem_thr in eval_pairs:
        regime = "A (top_k=20)" if top_k == 20 else "B (top_k=1)" if top_k == 1 else f"top_k={top_k}"
        print(f"\n--- Regime {regime}  sem_thr={sem_thr} ---")
        sub = [r for r in rows
               if r["top_k"] == top_k and r["sem_thr"] == sem_thr]
        dense_r10 = next(r["sem_R@10"] for r in sub if r["method"] == "dense")
        print(f"{'method':<14s} {'comp×':>7s} | "
              f"{'R@1':>5s} {'R@5':>5s} {'R@10':>5s} {'R@20':>5s} {'R@100':>5s} | "
              f"{'MRR':>5s} | {'Δ@10':>7s}")
        print("-" * 104)
        # group by RVQ config (or "all" for dense)
        for r in sub:
            tag = (f"{r['method']:<14s}")
            comp = "  —" if r["method"] == "dense" else f"{r['compression_x']:>5.0f}×"
            if r["method"] != "dense":
                tag = f"{r['method']}/{r['M']},{r['K']:<3d}"[:14]
                tag = f"{tag:<14s}"
            d10_delta = (r["sem_R@10"] - dense_r10) * 100
            print(f"{tag} {comp:>7s} | "
                  f"{r['sem_R@1']:>5.3f} {r['sem_R@5']:>5.3f} "
                  f"{r['sem_R@10']:>5.3f} {r['sem_R@20']:>5.3f} "
                  f"{r['sem_R@100']:>5.3f} | "
                  f"{r['sem_MRR']:>5.3f} | {d10_delta:>+6.2f}pp")
    print("\n" + "=" * 104)

    # ---- Persist ----
    out_path = rvq_root / "final_eval_report.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "configs": args.configs,
                "eval_configs": args.eval_configs,
                "K_top": args.K_top,
            },
            "rows": rows,
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
