"""Stage A1 of CRR-DeepRVQ: fit a frozen RVQ codebook on z_train.

Loads the production DeepSets encoder (outputs/encoder_final/best_encoder.pt),
encodes all train cases into z (L2-normed, 768-dim), then sequentially fits
an M-level RVQ codebook (K codes per level) by k-means on residuals.

CLI from repo root (SDM env):
    /home/lab603/anaconda3/envs/SDM/bin/python \\
        -m diagnosis_model.cause_inference.rvq_rerank.fit_rvq \\
        --encoder_ckpt diagnosis_model/cause_inference/outputs/encoder_final/best_encoder.pt \\
        --case_db_dir diagnosis_model/cause_inference/outputs/case_db \\
        --output_dir diagnosis_model/cause_inference/outputs/rvq_rerank \\
        --M 4 --K 256 --kmeans_iters 25 --seed 42

Output:
    outputs/rvq_rerank/rvq_M{M}_K{K}/codebooks.pt
    outputs/rvq_rerank/rvq_M{M}_K{K}/fit_log.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from diagnosis_model.cause_inference.models.case_encoder import (
    EncoderConfig, build_encoder,
)
from diagnosis_model.cause_inference.phase1_baseline import (
    load_case_db, load_cases,
)
from diagnosis_model.cause_inference.train_case_encoder import encode_all
from diagnosis_model.cause_inference.rvq_rerank.rvq import RVQCodebook


def load_encoder(ckpt_path: Path, device: torch.device):
    """Restore DeepSets encoder from best_encoder.pt; returns (encoder, cfg)."""
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    cfg_dict = dict(ckpt["encoder_config"])
    cfg = EncoderConfig(**cfg_dict)
    encoder = build_encoder(cfg).to(device)
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.eval()
    return encoder, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder_ckpt", type=str, required=True)
    ap.add_argument("--case_db_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True,
                    help="Parent dir; script writes rvq_M{M}_K{K}/ underneath")
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--kmeans_iters", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_train_cases", type=int, default=-1,
                    help="Cap retained train cases via uniform per-shard "
                         "subsampling. -1 (default) loads all (fish). "
                         "200000 mirrors the Phase 1/2 DDXPlus convention.")
    ap.add_argument("--sample_seed", type=int, default=42,
                    help="Subsample RNG seed. Should match the seed used in "
                         "train_case_encoder.py so z_train is fitted on the "
                         "same subset of cases that the encoder was trained on.")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir) / f"rvq_M{args.M}_K{args.K}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load frozen DeepSets encoder
    encoder, enc_cfg = load_encoder(Path(args.encoder_ckpt), device)
    print(f"[encoder] type={enc_cfg.encoder_type}  D={enc_cfg.d_model}")

    # 2. Load case database — sharded layout handled by load_cases. Only
    # train cases are needed (RVQ fits codebooks on z_train).
    case_db_dir = Path(args.case_db_dir)
    max_train = args.max_train_cases if args.max_train_cases > 0 else None
    train_cases = load_cases(
        case_db_dir, "train",
        max_cases=max_train, sample_seed=args.sample_seed,
    )
    meta = json.load((case_db_dir / "meta.json").open())
    D = meta["global_dim"]
    print(f"[data] train={len(train_cases)}  D={D}")
    if args.max_train_cases > 0:
        print(f"[data] subsample seed={args.sample_seed} max_train_cases={args.max_train_cases}")

    # 3. Encode train cases -> z [Nt, D]
    z_train = encode_all(encoder, train_cases, device).to(device).float()
    # Sanity: should already be L2-normed by the encoder head
    norms = z_train.norm(dim=-1)
    print(f"[z_train] shape={tuple(z_train.shape)}  "
          f"||z|| mean={norms.mean().item():.4f}  "
          f"min={norms.min().item():.4f}  max={norms.max().item():.4f}")

    # 4. Fit RVQ
    rvq = RVQCodebook(M=args.M, K=args.K, D=D).to(device)
    per_level_stats = rvq.fit(
        z_train, n_iters=args.kmeans_iters, seed=args.seed, verbose=True,
    )

    # 5. Final stats on z_train (encode = re-derive codes from the fitted codebook)
    codes, z_hat, e = rvq.encode(z_train)
    e_norm = e.norm(dim=-1)
    cos_sim = F.cosine_similarity(z_train, z_hat, dim=-1)
    print(f"\n=== Final RVQ stats on z_train (M={args.M}, K={args.K}) ===")
    print(f"  recon MSE       : {e.pow(2).mean().item():.6e}")
    print(f"  ||e|| mean      : {e_norm.mean().item():.4f}")
    print(f"  ||e|| median    : {e_norm.median().item():.4f}")
    print(f"  ||e|| min/max   : {e_norm.min().item():.4f} / {e_norm.max().item():.4f}")
    print(f"  cos(z, z_hat)   : mean={cos_sim.mean().item():.4f}  "
          f"min={cos_sim.min().item():.4f}")

    # 6. Save codebooks + log
    torch.save({
        "codebooks": rvq.codebooks.detach().cpu(),
        "fitted": rvq.fitted.detach().cpu(),
        "config": {
            "M": args.M, "K": args.K, "D": D,
            "kmeans_iters": args.kmeans_iters, "seed": args.seed,
        },
    }, out_dir / "codebooks.pt")

    log = {
        "config": {
            "M": args.M, "K": args.K, "D": D,
            "kmeans_iters": args.kmeans_iters, "seed": args.seed,
            "encoder_ckpt": str(args.encoder_ckpt),
            "case_db_dir": str(args.case_db_dir),
            "n_train": len(train_cases),
        },
        "per_level_stats": per_level_stats,
        "final": {
            "recon_mse": e.pow(2).mean().item(),
            "e_norm_mean": e_norm.mean().item(),
            "e_norm_median": e_norm.median().item(),
            "e_norm_min": e_norm.min().item(),
            "e_norm_max": e_norm.max().item(),
            "cos_z_zhat_mean": cos_sim.mean().item(),
            "cos_z_zhat_median": cos_sim.median().item(),
            "cos_z_zhat_min": cos_sim.min().item(),
        },
    }
    with open(out_dir / "fit_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nSaved: {out_dir / 'codebooks.pt'}")
    print(f"Saved: {out_dir / 'fit_log.json'}")


if __name__ == "__main__":
    main()
