"""Aggregation-buffer absorption surface for CRR-DeepRVQ.

Empirically maps how downstream cause-set aggregation (top_k_cases) absorbs
upstream case-ranking noise introduced by RVQ compression. For each RVQ
config (different reconstruction errors) and each top_k_cases value, runs
the standard Phase 1 eval pipeline with rvq_only scoring and records:

    R@K, MRR, mean_pool_size, mean cos(z, z_hat), mean ||z - z_hat||^2

The question: is R@10^agg_dense - R@10^agg_rvq a predictable function
of (top_k_cases, RVQ_err)? If yes, a method-level "aggregation-aware
quantization" contribution becomes defensible.

CLI from repo root:
    $PY -m diagnosis_model.cause_inference.rvq_rerank.eval_absorption_surface
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch

from diagnosis_model.cause_inference.eval_phase1_aligned import evaluate
from diagnosis_model.cause_inference.phase1_baseline import load_case_db
from diagnosis_model.cause_inference.train_case_encoder import encode_all
from diagnosis_model.cause_inference.rvq_rerank.fit_rvq import load_encoder
from diagnosis_model.cause_inference.rvq_rerank.rvq import RVQCodebook


def _parse_mk(spec: str) -> Tuple[int, int]:
    parts = spec.split(",")
    return int(parts[0]), int(parts[1])


@torch.no_grad()
def _rvq_error_stats(z: torch.Tensor, z_hat: torch.Tensor) -> dict:
    """Per-vector reconstruction quality stats on a tensor [N, D]."""
    diff = z - z_hat
    mse = diff.pow(2).sum(dim=-1).mean().item()          # ||e||^2
    z_norm2 = z.pow(2).sum(dim=-1).mean().item()
    rel_mse = mse / max(z_norm2, 1e-12)
    cos = torch.nn.functional.cosine_similarity(z, z_hat, dim=-1).mean().item()
    return {"abs_mse": mse, "rel_mse": rel_mse, "cos_zz_hat": cos}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder_ckpt", type=str,
                    default="diagnosis_model/cause_inference/outputs/"
                            "encoder_final/best_encoder.pt")
    ap.add_argument("--case_db_dir", type=str,
                    default="diagnosis_model/cause_inference/outputs/case_db")
    ap.add_argument("--rvq_root", type=str,
                    default="diagnosis_model/cause_inference/outputs/rvq_rerank")
    ap.add_argument("--rvq_configs", nargs="+", type=str,
                    default=[
                        "8,256",    # 384x  — low error anchor
                        "4,256",    # 768x  — production
                        "2,64",     # 2048x — mid
                        "1,16",     # 6144x — high
                        "1,4",      # 24576x — extreme
                    ],
                    help="M,K pairs")
    ap.add_argument("--top_ks", type=int, nargs="+",
                    default=[1, 2, 3, 5, 8, 10, 15, 20, 30, 50])
    ap.add_argument("--sem_thr", type=float, default=0.95)
    ap.add_argument("--Ks", type=int, nargs="+", default=[1, 5, 10, 20])
    ap.add_argument("--kmeans_iters", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str,
                    default="diagnosis_model/cause_inference/outputs/"
                            "rvq_rerank/absorption_surface.json")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rvq_root = Path(args.rvq_root)

    # ---- Load encoder + data ----
    encoder, _ = load_encoder(Path(args.encoder_ckpt), device)
    train_cases, valid_cases, cause_pkg, meta = load_case_db(
        Path(args.case_db_dir),
    )
    cause_embs = cause_pkg["embeddings"]
    D = meta["global_dim"]
    print(f"[data] train={len(train_cases)}  valid={len(valid_cases)}  D={D}")

    z_train = encode_all(encoder, train_cases, device).to(device).float()
    z_valid = encode_all(encoder, valid_cases, device).to(device).float()
    sim_dense_np = (z_valid @ z_train.T).cpu().numpy()

    Ks = tuple(args.Ks)
    rows: List[dict] = []

    # --- Dense baseline across all top_ks ---
    for top_k in args.top_ks:
        m = evaluate(
            lambda qi, q: sim_dense_np[qi],
            valid_cases, train_cases, cause_embs,
            top_k_cases=top_k, semantic_threshold=args.sem_thr,
            Ks=Ks, device=device,
        )
        rows.append({
            "method": "dense", "M": -1, "K": -1,
            "compression_x": 1.0,
            "abs_mse": 0.0, "rel_mse": 0.0, "cos_zz_hat": 1.0,
            "top_k": top_k, "sem_thr": args.sem_thr,
            **m,
        })
        print(f"[dense   top_k={top_k:>3d}] R@10={m['sem_R@10']:.4f}  "
              f"pool={m['mean_pool_size']:.1f}")

    # --- RVQ configs ---
    for spec in args.rvq_configs:
        M, K = _parse_mk(spec)
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

        _, z_hat_train, _ = rvq.encode(z_train)
        err_stats = _rvq_error_stats(z_train, z_hat_train)
        import math
        compression_x = (D * 32) / (M * max(1, math.ceil(math.log2(max(K, 2)))))

        sim_rvq_np = (z_valid @ z_hat_train.T).cpu().numpy()

        for top_k in args.top_ks:
            m = evaluate(
                lambda qi, q, _s=sim_rvq_np: _s[qi],
                valid_cases, train_cases, cause_embs,
                top_k_cases=top_k, semantic_threshold=args.sem_thr,
                Ks=Ks, device=device,
            )
            rows.append({
                "method": "rvq_only", "M": M, "K": K,
                "compression_x": compression_x,
                **err_stats,
                "top_k": top_k, "sem_thr": args.sem_thr,
                **m,
            })
            print(f"[M={M} K={K:>3d} top_k={top_k:>3d}] "
                  f"R@10={m['sem_R@10']:.4f}  "
                  f"pool={m['mean_pool_size']:.1f}  "
                  f"relMSE={err_stats['rel_mse']:.4f}  "
                  f"cos={err_stats['cos_zz_hat']:.4f}")

    # ---- Persist ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "rvq_configs": args.rvq_configs,
                "top_ks": args.top_ks,
                "sem_thr": args.sem_thr,
            },
            "rows": rows,
        }, f, indent=2)
    print(f"\nSaved: {out_path}")

    # ---- Quick tabular view: ΔR@10 vs top_k for each RVQ config ----
    print("\n" + "=" * 88)
    print("ΔR@10 vs dense (pp)  —  rows: RVQ config  cols: top_k_cases")
    print("=" * 88)
    dense_by_topk = {
        r["top_k"]: r["sem_R@10"]
        for r in rows if r["method"] == "dense"
    }
    header = "config         relMSE  " + "".join(
        f"{tk:>7d}" for tk in args.top_ks
    )
    print(header)
    for spec in args.rvq_configs:
        M, K = _parse_mk(spec)
        # pull error from first matching row
        any_row = next(r for r in rows
                       if r["method"] == "rvq_only" and r["M"] == M and r["K"] == K)
        rel_mse = any_row["rel_mse"]
        cells = []
        for tk in args.top_ks:
            r = next(r for r in rows
                     if r["method"] == "rvq_only"
                     and r["M"] == M and r["K"] == K and r["top_k"] == tk)
            d = (r["sem_R@10"] - dense_by_topk[tk]) * 100
            cells.append(f"{d:>+6.2f}")
        print(f"M={M} K={K:<4d}     {rel_mse:>5.3f}  " + "".join(
            f"{c:>7s}" for c in cells
        ))


if __name__ == "__main__":
    main()
