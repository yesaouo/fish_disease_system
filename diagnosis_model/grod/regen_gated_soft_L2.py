"""Regenerate gated soft_inputs + bank from an L1/L2 arm checkpoint, so the
existing train_ceah_soft / faithfulness_eval_soft run unchanged for the L2
faithfulness A/B.

Unlike apply_gate_to_soft (which only re-gates the cached sigmoid w and keeps
the frozen-head z), this recomputes BOTH z and objectness from the arm's
(possibly retrained) semantic + objectness heads on cached hs, then applies the
arm's Region Gate. Global g stays cached (L2 does not touch the global head).

Writes (mirrors soft_inputs layout):
    <out_dir>/train.pt, valid.pt   (g cached; z_all, w recomputed+gated)
    <out_dir>/bank_z_soft.pt       (train bank re-encoded with the arm encoder)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from diagnosis_model.cause_inference.models.case_encoder import EncoderConfig, build_encoder
from diagnosis_model.grod.region_gate import RegionGate
from diagnosis_model.grod import finetune_e2e_L2 as L2
from diagnosis_model.grod.train_case_encoder_soft import encode_all_soft


def main():
    ap = argparse.ArgumentParser()
    ART = "data/processed/current/artifacts"
    ap.add_argument("--arm_ckpt", required=True, help="finetune_e2e_L2 best.pt (enc+gate+sem+cls)")
    ap.add_argument("--soft_dir", default=f"{ART}/db/soft_inputs")
    ap.add_argument("--hs_dir", default=f"{ART}/db/hs_all")
    ap.add_argument("--joint_ckpt", default=f"{ART}/models/joint_rfdetr/checkpoint_best_regular.pth")
    ap.add_argument("--anchors", default=f"{ART}/models/text_anchors.pt")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    pkg = torch.load(args.arm_ckpt, weights_only=False, map_location="cpu")
    enc = build_encoder(EncoderConfig(**pkg["encoder_config"])).to(dev); enc.load_state_dict(pkg["encoder_state"]); enc.eval()
    gate = RegionGate(init_temp=pkg["gate_init_temp"]).to(dev); gate.load_state_dict(pkg["gate_state"]); gate.eval()
    sem, cls = L2.load_heads(args.joint_ckpt, args.anchors, dev)
    if "semantic_embed_state" in pkg:
        sem.load_state_dict(pkg["semantic_embed_state"]); cls.load_state_dict(pkg["class_embed_state"])
    sem.eval(); cls.eval()

    g_keep = {}
    for split in ("train", "valid"):
        d = torch.load(Path(args.soft_dir) / f"{split}.pt", weights_only=False)
        hs, _, _ = L2.load_hs(Path(args.hs_dir) / f"hs_all_{split}.pt")
        assert hs.size(0) == d["g"].size(0), f"{split} hs/g misalign"
        z_list, w_list = [], []
        with torch.no_grad():
            for s in range(0, hs.size(0), 256):
                e = min(s + 256, hs.size(0))
                z, o = L2.heads_fwd(sem, cls, hs[s:e].float().to(dev))
                z_list.append(z.to(torch.bfloat16).cpu())
                w_list.append(gate(o).float().cpu())
        d["z_all"] = torch.cat(z_list)
        d["w"] = torch.cat(w_list)
        torch.save(d, out_dir / f"{split}.pt")
        print(f"[save] {out_dir/f'{split}.pt'} z={tuple(d['z_all'].shape)} "
              f"w mean Σ/row={d['w'].sum(1).mean():.3f}")
        if split == "train":
            g_keep = {"g": d["g"], "z": d["z_all"], "w": d["w"]}

    bank = encode_all_soft(enc, g_keep["g"], g_keep["z"], g_keep["w"], dev)
    torch.save({"bank_z": bank, "encoder_ckpt": args.arm_ckpt}, out_dir / "bank_z_soft.pt")
    print(f"[save] bank_z_soft.pt bank={tuple(bank.shape)}")


if __name__ == "__main__":
    main()
