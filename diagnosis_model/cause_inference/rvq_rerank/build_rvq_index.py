"""Stage A2 of CRR-DeepRVQ: build the encoded index for train + valid cases.

Loads the fitted RVQ codebook (Stage A1) and the frozen DeepSets encoder,
encodes every case (train + valid) into (z, z_hat, e, codes), and writes a
single index.pt file. Downstream stages (reranker training, eval, benchmark)
consume this artifact.

CLI from repo root (SDM env):
    /home/lab603/anaconda3/envs/SDM/bin/python \\
        -m diagnosis_model.cause_inference.rvq_rerank.build_rvq_index \\
        --encoder_ckpt diagnosis_model/cause_inference/outputs/encoder_final/best_encoder.pt \\
        --case_db_dir diagnosis_model/cause_inference/outputs/case_db \\
        --rvq_dir diagnosis_model/cause_inference/outputs/rvq_rerank/rvq_M4_K256

Output:
    {rvq_dir}/index.pt   with structure:
        {
          "config":  {M, K, D, ...},
          "train":   {z, z_hat, e, codes, e_norm},   # all aligned by case row
          "valid":   {z, z_hat, e, codes, e_norm},
        }
    z / z_hat / e are saved as fp16; codes are uint8 (K<=256) or int16/32.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from diagnosis_model.cause_inference.phase1_baseline import load_case_db
from diagnosis_model.cause_inference.train_case_encoder import encode_all
from diagnosis_model.cause_inference.rvq_rerank.fit_rvq import load_encoder
from diagnosis_model.cause_inference.rvq_rerank.rvq import (
    RVQCodebook, codes_to_compact,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder_ckpt", type=str, required=True)
    ap.add_argument("--case_db_dir", type=str, required=True)
    ap.add_argument("--rvq_dir", type=str, required=True,
                    help="Dir containing codebooks.pt (from fit_rvq.py)")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rvq_dir = Path(args.rvq_dir)

    # 1. Load codebooks
    pkg = torch.load(rvq_dir / "codebooks.pt", weights_only=False, map_location=device)
    cfg = pkg["config"]
    M, K, D = cfg["M"], cfg["K"], cfg["D"]
    rvq = RVQCodebook(M=M, K=K, D=D).to(device)
    rvq.codebooks.copy_(pkg["codebooks"].to(device))
    rvq.fitted.copy_(pkg["fitted"].to(device))
    print(f"[rvq] M={M}  K={K}  D={D}  loaded from {rvq_dir}")

    # 2. Load encoder
    encoder, enc_cfg = load_encoder(Path(args.encoder_ckpt), device)
    print(f"[encoder] type={enc_cfg.encoder_type}")

    # 3. Load cases
    train_cases, valid_cases, _, meta = load_case_db(Path(args.case_db_dir))
    print(f"[data] train={len(train_cases)}  valid={len(valid_cases)}")

    out = {"config": cfg, "M": M, "K": K, "D": D}

    for split_name, cases in [("train", train_cases), ("valid", valid_cases)]:
        z = encode_all(encoder, cases, device).to(device).float()
        codes, z_hat, e = rvq.encode(z)
        e_norm = e.norm(dim=-1)
        cos_sim = F.cosine_similarity(z, z_hat, dim=-1)
        print(f"\n[{split_name}] N={z.size(0)}")
        print(f"  recon MSE   : {e.pow(2).mean().item():.6e}")
        print(f"  ||e|| mean  : {e_norm.mean().item():.4f}  "
              f"median={e_norm.median().item():.4f}  "
              f"max={e_norm.max().item():.4f}")
        print(f"  cos(z, ẑ)   : mean={cos_sim.mean().item():.4f}  "
              f"min={cos_sim.min().item():.4f}")

        out[split_name] = {
            "z":      z.detach().cpu().half(),                 # [N, D] fp16
            "z_hat":  z_hat.detach().cpu().half(),             # [N, D] fp16
            "e":      e.detach().cpu().half(),                 # [N, D] fp16
            "codes":  codes_to_compact(codes.detach().cpu(), K),
            "e_norm": e_norm.detach().cpu().float(),           # [N]
        }

    torch.save(out, rvq_dir / "index.pt")
    print(f"\nSaved: {rvq_dir / 'index.pt'}")


if __name__ == "__main__":
    main()
