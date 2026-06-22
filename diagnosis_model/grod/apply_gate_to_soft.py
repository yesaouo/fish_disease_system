"""Produce a Region-Gate'd copy of soft_inputs + matching bank, so the existing
train_ceah_soft / faithfulness_eval_soft run unchanged on gated weights.

The soft pipeline's gate is the per-lesion weight w. We swap the frozen sigmoid
w for the trained Region Gate (∅-sink softmax_τ) from an L1/L2 checkpoint:

    w_gated = RegionGate(logit(w_sigmoid))            # exact: logit inverts sigmoid

Writes (mirrors soft_inputs layout so --soft_dir just repoints):
    <out_dir>/train.pt, valid.pt    (g, z_all unchanged; w replaced; +cause idx)
    <out_dir>/bank_z_soft.pt        (train bank re-encoded with the L1 encoder + gated w)

Run: $PY -m diagnosis_model.grod.apply_gate_to_soft \
        --gate_ckpt .../models/grace_e2e/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from diagnosis_model.cause_inference.models.case_encoder import EncoderConfig, build_encoder
from diagnosis_model.grod.region_gate import RegionGate
from diagnosis_model.grod.finetune_e2e_grace import recover_logits
from diagnosis_model.grod.train_case_encoder_soft import load_soft, encode_all_soft


def gate_weights(gate, w, device, bs=256):
    """w[N,Q] sigmoid -> RegionGate weights, same shape/dtype."""
    out = []
    with torch.no_grad():
        for s in range(0, w.size(0), bs):
            o = recover_logits(w[s:s + bs]).to(device)
            out.append(gate(o).float().cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ART = "data/processed/current/artifacts"
    ap.add_argument("--gate_ckpt", default=f"{ART}/models/grace_e2e/best.pt")
    ap.add_argument("--soft_dir", default=f"{ART}/db/soft_inputs")
    ap.add_argument("--out_dir", default=f"{ART}/db/soft_inputs_gated")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = args.device
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    pkg = torch.load(args.gate_ckpt, weights_only=False, map_location="cpu")
    gate = RegionGate(init_temp=pkg.get("gate_init_temp", 0.3)).to(dev)
    gate.load_state_dict(pkg["gate_state"]); gate.eval()
    enc = build_encoder(EncoderConfig(**pkg["encoder_config"])).to(dev)
    enc.load_state_dict(pkg["encoder_state"]); enc.eval()
    print(f"[gate] τ={gate.temp.item():.4f} ∅={gate.sink.item():+.4f}  (from {args.gate_ckpt})")

    g_tr = z_tr = None
    for split in ("train", "valid"):
        d = torch.load(Path(args.soft_dir) / f"{split}.pt", weights_only=False)
        wg = gate_weights(gate, d["w"].float(), dev)
        d["w"] = wg
        torch.save(d, out_dir / f"{split}.pt")
        print(f"[save] {out_dir/f'{split}.pt'}  w_gated={tuple(wg.shape)} "
              f"mean Σw/row={wg.sum(1).mean():.3f}")
        if split == "train":
            g_tr, z_tr, w_tr = d["g"], d["z_all"], wg

    bank = encode_all_soft(enc, g_tr, z_tr, w_tr, dev)
    torch.save({"bank_z": bank, "encoder_ckpt": args.gate_ckpt}, out_dir / "bank_z_soft.pt")
    print(f"[save] {out_dir/'bank_z_soft.pt'}  bank={tuple(bank.shape)}")


if __name__ == "__main__":
    main()
