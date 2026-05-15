"""Stress-test RVQ retrieval under harder Phase 1-aligned eval configs.

The default sanity check (top_k_cases=20, semantic_threshold=0.95) showed
RVQ is lossless vs dense — because the cause-aggregation pool absorbs
ranking noise. This script sweeps stricter eval configs (smaller top_k,
tighter sem_thr) to find a regime where RVQ damage is measurable and the
reranker has room to recover something.

Grid:
    (M, K) ∈ user-supplied subset (default: aggressive + moderate + light)
    (top_k_cases, semantic_threshold) ∈ user-supplied subset

For each (eval_config, method) we report sem_R@K, MRR, mean pool size,
and Δ@10 (rvq − dense) at the same eval_config.

CLI from repo root:
    /home/lab603/anaconda3/envs/SDM/bin/python \\
        -m diagnosis_model.cause_inference.rvq_rerank.eval_harder
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


def _parse_pair(s: str) -> Tuple[float, float]:
    a, b = s.split(",")
    return float(a), float(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder_ckpt", type=str,
                    default="diagnosis_model/cause_inference/outputs/"
                            "encoder_final/best_encoder.pt")
    ap.add_argument("--case_db_dir", type=str,
                    default="diagnosis_model/cause_inference/outputs/case_db")
    ap.add_argument("--output_dir", type=str,
                    default="diagnosis_model/cause_inference/outputs/rvq_rerank")
    ap.add_argument("--rvq_configs", nargs="+", type=str,
                    default=["4,256", "2,64", "1,16"],
                    help="(M,K) pairs; M=4 K=256 is light, M=1 K=16 most "
                         "aggressive (6144x compression)")
    ap.add_argument("--eval_configs", nargs="+", type=str,
                    default=["20,0.95", "5,0.95", "1,0.95",
                             "20,0.99", "5,0.99"],
                    help="(top_k_cases, semantic_threshold) pairs")
    ap.add_argument("--Ks", type=int, nargs="+", default=[1, 5, 10, 20, 100])
    ap.add_argument("--kmeans_iters", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    rvq_pairs: List[Tuple[int, int]] = [
        tuple(int(x) for x in s.split(",")) for s in args.rvq_configs
    ]
    eval_pairs: List[Tuple[int, float]] = [
        (int(p[0]), float(p[1])) for p in (_parse_pair(s) for s in args.eval_configs)
    ]

    # ----- Load -----
    encoder, _ = load_encoder(Path(args.encoder_ckpt), device)
    train_cases, valid_cases, cause_pkg, meta = load_case_db(
        Path(args.case_db_dir),
    )
    cause_embs = cause_pkg["embeddings"]
    D = meta["global_dim"]
    print(f"[data] train={len(train_cases)}  valid={len(valid_cases)}  D={D}")

    print("[encode] z_train, z_valid ...")
    z_train = encode_all(encoder, train_cases, device).to(device).float()
    z_valid = encode_all(encoder, valid_cases, device).to(device).float()
    sim_dense = (z_valid @ z_train.T).cpu().numpy()

    # ----- Fit each RVQ once, cache sim matrix -----
    sim_rvqs = {}
    for M, K in rvq_pairs:
        rvq = RVQCodebook(M=M, K=K, D=D).to(device)
        rvq.fit(z_train, n_iters=args.kmeans_iters, seed=args.seed, verbose=False)
        _, z_hat, e = rvq.encode(z_train)
        sim_rvqs[(M, K)] = (z_valid @ z_hat.T).cpu().numpy()
        print(f"[rvq] M={M:2d} K={K:3d}  ||e|| mean={e.norm(dim=-1).mean():.4f}")

    Ks = tuple(args.Ks)

    # ----- Evaluate each (eval_config, method) -----
    rows = []
    for top_k, sem_thr in eval_pairs:
        # dense
        m = evaluate(
            lambda qi, q: sim_dense[qi],
            valid_cases, train_cases, cause_embs,
            top_k_cases=top_k, semantic_threshold=sem_thr,
            Ks=Ks, device=device,
        )
        rows.append({"method": "dense", "M": -1, "K": -1,
                     "top_k": top_k, "sem_thr": sem_thr, **m})

        # each rvq
        for (M, K), sim_rvq in sim_rvqs.items():
            m = evaluate(
                lambda qi, q, _s=sim_rvq: _s[qi],
                valid_cases, train_cases, cause_embs,
                top_k_cases=top_k, semantic_threshold=sem_thr,
                Ks=Ks, device=device,
            )
            rows.append({"method": f"rvq_M{M}_K{K}", "M": M, "K": K,
                         "top_k": top_k, "sem_thr": sem_thr, **m})

    # ----- Pretty print, grouped by eval_config -----
    print("\n\n" + "=" * 102)
    for top_k, sem_thr in eval_pairs:
        sub = [r for r in rows
               if r["top_k"] == top_k and r["sem_thr"] == sem_thr]
        dense_row = next(r for r in sub if r["method"] == "dense")
        d10 = dense_row["sem_R@10"]
        d100 = dense_row["sem_R@100"]

        print(f"\n--- top_k_cases={top_k}  sem_thr={sem_thr}  "
              f"mean_pool={dense_row['mean_pool_size']:.1f} ---")
        print(f"{'method':<14s} | "
              f"{'R@1':>5s} {'R@5':>5s} {'R@10':>5s} {'R@20':>5s} {'R@100':>5s} | "
              f"{'MRR':>5s} | {'Δ@10':>7s} {'Δ@100':>7s}")
        print("-" * 102)
        for r in sub:
            d10_delta = (r["sem_R@10"] - d10) * 100
            d100_delta = (r["sem_R@100"] - d100) * 100
            print(f"{r['method']:<14s} | "
                  f"{r['sem_R@1']:>5.3f} {r['sem_R@5']:>5.3f} "
                  f"{r['sem_R@10']:>5.3f} {r['sem_R@20']:>5.3f} "
                  f"{r['sem_R@100']:>5.3f} | "
                  f"{r['sem_MRR']:>5.3f} | "
                  f"{d10_delta:>+6.2f}pp {d100_delta:>+6.2f}pp")
    print("\n" + "=" * 102)

    # ----- Persist -----
    report = {
        "config": {
            "rvq_configs": args.rvq_configs,
            "eval_configs": args.eval_configs,
            "Ks": list(args.Ks),
            "kmeans_iters": args.kmeans_iters,
            "seed": args.seed,
        },
        "rows": rows,
    }
    out_path = out_root / "harder_eval_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
