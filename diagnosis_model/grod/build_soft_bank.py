"""Soft-pipeline retrain — step #3: rebuild the retrieval bank with the soft
Aggregator over the soft inputs.

bank_z_soft[i] = soft_encoder(g_i, z_all_i, w_i) for every train case, so the
inference query (also soft-encoded) and the bank live in the same space.
Precomputed to a small file so gpu_infer_soft.py needn't load the 5.6 GB
soft_inputs at startup.

Output: outputs/encoder_grod_soft/bank_z_soft.pt  ([Nt, 768] fp32)

Run from repo root:
    $PY -m diagnosis_model.grod.build_soft_bank
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from diagnosis_model.cause_inference.models.case_encoder import EncoderConfig, build_encoder
from diagnosis_model.grod.train_case_encoder_soft import encode_all_soft, load_soft


def main():
    ap = argparse.ArgumentParser()
    ART = "data/processed/current/artifacts"
    ap.add_argument("--soft_dir", default=f"{ART}/db/soft_inputs")
    ap.add_argument("--encoder_ckpt", default=f"{ART}/models/encoder_grod_soft/best_encoder.pt")
    ap.add_argument("--out", default=f"{ART}/models/encoder_grod_soft/bank_z_soft.pt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    pkg = torch.load(args.encoder_ckpt, weights_only=False, map_location="cpu")
    cfg = EncoderConfig(**pkg["encoder_config"])
    enc = build_encoder(cfg).to(dev).eval()
    enc.load_state_dict(pkg["encoder_state"])

    g, z, w, _ = load_soft(Path(args.soft_dir) / "train.pt")
    bank = encode_all_soft(enc, g, z, w, dev)                  # [Nt, 768] fp32
    torch.save({"bank_z": bank, "encoder_ckpt": args.encoder_ckpt}, args.out)
    print(f"[save] {args.out}  bank_z={tuple(bank.shape)}")


if __name__ == "__main__":
    main()
