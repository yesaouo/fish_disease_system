"""Fit vanilla / ranking-aware / aggregation-aware RVQ codebooks at all
(M, K) configs. Saves to outputs/rvq_rerank/variants/{tag}/rvq_M{M}_K{K}/
so eval_absorption_surface can be pointed at each directory.

CLI from repo root:
    $PY -m diagnosis_model.cause_inference.rvq_rerank.fit_rvq_variants
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from diagnosis_model.cause_inference.phase1_baseline import load_case_db
from diagnosis_model.cause_inference.train_case_encoder import encode_all
from diagnosis_model.cause_inference.rvq_rerank.fit_rvq import load_encoder
from diagnosis_model.cause_inference.rvq_rerank.weighted_rvq import (
    WeightedRVQCodebook,
)


VARIANTS = ("vanilla", "ranking", "agg", "agg_inv")
WEIGHT_KEYS = {
    "vanilla": "w_vanilla",
    "ranking": "w_ranking",
    "agg":     "w_agg",       # ranking × isolation (up-weight isolated)
    "agg_inv": "w_agg_inv",   # ranking × density   (up-weight dense)
}


def fit_one(z_train, weights, M, K, D, n_iters, seed, device, out_dir, verbose):
    rvq = WeightedRVQCodebook(M=M, K=K, D=D).to(device)
    stats = rvq.fit_weighted(z_train, weights, n_iters=n_iters,
                             seed=seed, verbose=verbose)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "codebooks": rvq.codebooks.cpu(),
        "fitted": rvq.fitted.cpu(),
        "M": M, "K": K, "D": D,
        "n_iters": n_iters, "seed": seed,
        "stats": stats,
    }, out_dir / "codebooks.pt")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder_ckpt", type=str,
                    default="diagnosis_model/cause_inference/outputs/"
                            "encoder_final/best_encoder.pt")
    ap.add_argument("--case_db_dir", type=str,
                    default="diagnosis_model/cause_inference/outputs/case_db")
    ap.add_argument("--weights_pkg", type=str,
                    default="diagnosis_model/cause_inference/outputs/"
                            "rvq_rerank/case_weights.pt")
    ap.add_argument("--out_root", type=str,
                    default="diagnosis_model/cause_inference/outputs/"
                            "rvq_rerank/variants")
    ap.add_argument("--configs", nargs="+", type=str,
                    default=["8,256", "4,256", "2,64", "1,16", "1,4"])
    ap.add_argument("--variants", nargs="+", type=str,
                    default=list(VARIANTS),
                    choices=list(VARIANTS))
    ap.add_argument("--n_iters", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    encoder, _ = load_encoder(Path(args.encoder_ckpt), device)
    train_cases, _, _, meta = load_case_db(Path(args.case_db_dir))
    D = meta["global_dim"]
    print(f"[data] train={len(train_cases)}  D={D}")

    z_train = encode_all(encoder, train_cases, device).to(device).float()

    wpkg = torch.load(args.weights_pkg, weights_only=False, map_location=device)
    print(f"[weights] loaded {args.weights_pkg}")
    for key in WEIGHT_KEYS.values():
        w = wpkg[key].to(device)
        print(f"  {key:>10s}: min={w.min().item():.3f}  med={w.median().item():.3f}  "
              f"max={w.max().item():.3f}  sum={w.sum().item():.1f}")

    out_root = Path(args.out_root)
    summary = []

    for variant in args.variants:
        wkey = WEIGHT_KEYS[variant]
        w = wpkg[wkey].to(device)
        print(f"\n{'='*72}\n[variant={variant}  weights={wkey}]")
        for spec in args.configs:
            M, K = map(int, spec.split(","))
            out_dir = out_root / variant / f"rvq_M{M}_K{K}"
            print(f"\n--- M={M} K={K} ---")
            stats = fit_one(
                z_train, w, M, K, D,
                args.n_iters, args.seed, device, out_dir, verbose=True,
            )
            summary.append({
                "variant": variant, "M": M, "K": K,
                "final_mse_uniform": stats[-1]["cum_recon_mse_uniform"],
                "final_mse_weighted": stats[-1]["cum_recon_mse_weighted"],
                "usage_rate_last": stats[-1]["usage_rate"],
            })

    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "fit_summary.json", "w") as f:
        json.dump({"summary": summary}, f, indent=2)
    print(f"\nSaved fit summary: {out_root / 'fit_summary.json'}")

    print("\n" + "=" * 88)
    print(f"{'variant':<10s} {'M':>3s} {'K':>4s} {'mse_uniform':>14s} "
          f"{'mse_weighted':>14s} {'usage_last':>12s}")
    print("-" * 88)
    for r in summary:
        print(f"{r['variant']:<10s} {r['M']:>3d} {r['K']:>4d} "
              f"{r['final_mse_uniform']:>14.4e} "
              f"{r['final_mse_weighted']:>14.4e} "
              f"{r['usage_rate_last']:>11.1%}")


if __name__ == "__main__":
    main()
