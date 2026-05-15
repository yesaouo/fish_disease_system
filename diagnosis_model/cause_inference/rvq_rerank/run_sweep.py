"""CRR-DeepRVQ sweep: (M, K) compression-quality Pareto curve.

Encodes train + valid once, fits a fresh RVQ codebook for each (M, K) cell,
and runs the same Phase 1-aligned evaluation against dense. Reports
compression ratio, recon error, and recall delta vs dense in one table.

Used to find the sweet spot where RVQ starts costing recall — that's
where the reranker has room to recover something.

CLI from repo root (SDM env):
    /home/lab603/anaconda3/envs/SDM/bin/python \\
        -m diagnosis_model.cause_inference.rvq_rerank.run_sweep \\
        --M_list 1 2 4 8 --K_list 16 64 256

Output:
    outputs/rvq_rerank/sweep_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from diagnosis_model.cause_inference.eval_phase1_aligned import evaluate
from diagnosis_model.cause_inference.phase1_baseline import load_case_db
from diagnosis_model.cause_inference.train_case_encoder import encode_all
from diagnosis_model.cause_inference.rvq_rerank.fit_rvq import load_encoder
from diagnosis_model.cause_inference.rvq_rerank.rvq import RVQCodebook


def _compression_x(M: int, K: int, D: int = 768, ref_dtype_bytes: int = 4) -> float:
    """Effective compression vs fp32 dense: bits per case codes vs D*32 bits."""
    bits_per_case = M * max(1, math.ceil(math.log2(max(K, 2))))
    dense_bits = D * ref_dtype_bytes * 8
    return dense_bits / bits_per_case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder_ckpt", type=str,
                    default="diagnosis_model/cause_inference/outputs/"
                            "encoder_final/best_encoder.pt")
    ap.add_argument("--case_db_dir", type=str,
                    default="diagnosis_model/cause_inference/outputs/case_db")
    ap.add_argument("--output_dir", type=str,
                    default="diagnosis_model/cause_inference/outputs/rvq_rerank")
    ap.add_argument("--M_list", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--K_list", type=int, nargs="+", default=[16, 64, 256])
    ap.add_argument("--kmeans_iters", type=int, default=25)
    ap.add_argument("--top_k_cases", type=int, default=20)
    ap.add_argument("--semantic_threshold", type=float, default=0.95)
    ap.add_argument("--Ks", type=int, nargs="+", default=[1, 5, 10, 20, 100])
    ap.add_argument("--save_codebooks", action="store_true",
                    help="Persist codebooks.pt per config (off by default).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # ----- Load once -----
    encoder, _ = load_encoder(Path(args.encoder_ckpt), device)
    train_cases, valid_cases, cause_pkg, meta = load_case_db(Path(args.case_db_dir))
    cause_embs = cause_pkg["embeddings"]
    D = meta["global_dim"]
    print(f"[data] train={len(train_cases)}  valid={len(valid_cases)}  D={D}")

    print("[encode] z_train, z_valid ...")
    z_train = encode_all(encoder, train_cases, device).to(device).float()
    z_valid = encode_all(encoder, valid_cases, device).to(device).float()

    Ks = tuple(args.Ks)

    # ----- Dense baseline (run once) -----
    print("\n[eval dense baseline] ...")
    sim_dense = (z_valid @ z_train.T).cpu().numpy()
    m_dense = evaluate(
        lambda qi, q: sim_dense[qi],
        valid_cases, train_cases, cause_embs,
        top_k_cases=args.top_k_cases,
        semantic_threshold=args.semantic_threshold,
        Ks=Ks, device=device,
    )
    print(f"  dense sem_R@10 = {m_dense['sem_R@10']:.4f}  MRR = {m_dense['sem_MRR']:.4f}")

    results = [{
        "config": "dense_fp32",
        "M": -1, "K": -1,
        "compression_x": 1.0,
        "recon_mse": 0.0,
        "e_norm_mean": 0.0,
        "fit_time_s": 0.0,
        **m_dense,
    }]

    # ----- Sweep -----
    for M in args.M_list:
        for K in args.K_list:
            tag = f"rvq_M{M}_K{K}"
            print(f"\n=== {tag} ===")
            t0 = time.time()
            rvq = RVQCodebook(M=M, K=K, D=D).to(device)
            rvq.fit(z_train, n_iters=args.kmeans_iters, seed=args.seed, verbose=False)
            _, z_hat, e = rvq.encode(z_train)
            fit_time = time.time() - t0
            e_norm = e.norm(dim=-1)
            recon_mse = e.pow(2).mean().item()
            print(f"  fit {fit_time:.1f}s  "
                  f"recon_mse={recon_mse:.3e}  "
                  f"||e||={e_norm.mean().item():.4f}  "
                  f"comp×={_compression_x(M, K, D):.0f}")

            if args.save_codebooks:
                sub_dir = out_root / tag
                sub_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "codebooks": rvq.codebooks.detach().cpu(),
                    "fitted": rvq.fitted.detach().cpu(),
                    "config": {"M": M, "K": K, "D": D,
                               "kmeans_iters": args.kmeans_iters,
                               "seed": args.seed},
                }, sub_dir / "codebooks.pt")

            sim_rvq = (z_valid @ z_hat.T).cpu().numpy()
            m_rvq = evaluate(
                lambda qi, q: sim_rvq[qi],
                valid_cases, train_cases, cause_embs,
                top_k_cases=args.top_k_cases,
                semantic_threshold=args.semantic_threshold,
                Ks=Ks, device=device,
            )

            results.append({
                "config": tag,
                "M": M, "K": K,
                "compression_x": _compression_x(M, K, D),
                "recon_mse": recon_mse,
                "e_norm_mean": e_norm.mean().item(),
                "fit_time_s": fit_time,
                **m_rvq,
            })

    # ----- Print compact summary -----
    print("\n\n" + "=" * 92)
    head = (f"{'config':<13s} | {'comp×':>7s} | {'||e||':>6s} | "
            f"{'R@1':>5s} {'R@5':>5s} {'R@10':>5s} {'R@20':>5s} {'R@100':>5s} | "
            f"{'MRR':>5s} | {'Δ@10':>7s}")
    print(head)
    print("-" * 92)
    r10_base = m_dense["sem_R@10"]
    for r in results:
        delta_pp = (r["sem_R@10"] - r10_base) * 100
        row = (f"{r['config']:<13s} | "
               f"{r['compression_x']:>6.0f}× | "
               f"{r['e_norm_mean']:>6.4f} | "
               f"{r['sem_R@1']:>5.3f} {r['sem_R@5']:>5.3f} "
               f"{r['sem_R@10']:>5.3f} {r['sem_R@20']:>5.3f} "
               f"{r['sem_R@100']:>5.3f} | "
               f"{r['sem_MRR']:>5.3f} | "
               f"{delta_pp:>+6.2f}pp")
        print(row)
    print("=" * 92)

    # ----- Persist -----
    report = {
        "config": {
            "M_list": args.M_list,
            "K_list": args.K_list,
            "kmeans_iters": args.kmeans_iters,
            "top_k_cases": args.top_k_cases,
            "semantic_threshold": args.semantic_threshold,
            "Ks": list(args.Ks),
            "n_train": len(train_cases),
            "n_valid": len(valid_cases),
            "D": D,
            "encoder_ckpt": str(args.encoder_ckpt),
        },
        "dense_baseline": {**m_dense},
        "results": results,
    }
    out_path = out_root / "sweep_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
