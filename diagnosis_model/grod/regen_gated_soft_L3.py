"""Regen gated soft_inputs + bank from an L3-lite checkpoint (trained decoder).

Unlike regen_gated_soft_L2 (heads on cached hs), the L3 decoder is retrained so
hs itself changed — z/objectness must come from a LIVE forward of the trained
net over the images. Produces the same layout as soft_inputs so train_ceah_soft
/ faithfulness_eval_soft run unchanged.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diagnosis_model.cause_inference.models.case_encoder import EncoderConfig, build_encoder
from diagnosis_model.grod.region_gate import RegionGate
from diagnosis_model.grod.train_case_encoder_soft import encode_all_soft
from diagnosis_model.grod.finetune_e2e_L3lite import CaseImgDS, freeze_backbone_nograd


def main():
    ap = argparse.ArgumentParser()
    ART = "data/processed/current/artifacts"
    ap.add_argument("--arm_ckpt", required=True)
    ap.add_argument("--soft_dir", default=f"{ART}/db/soft_inputs")
    ap.add_argument("--case_db_dir", default=f"{ART}/db/case_db_jointDistRawP")
    ap.add_argument("--joint_ckpt", default=f"{ART}/models/joint_rfdetr/checkpoint_best_regular.pth")
    ap.add_argument("--anchors", default=f"{ART}/models/text_anchors.pt")
    ap.add_argument("--img_root", default="data/processed/current/detection")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["RFDETR_SEMANTIC_DIM"] = "768"
    os.environ["RFDETR_SEMANTIC_ANCHORS"] = os.path.abspath(args.anchors)
    os.environ["RFDETR_GLOBAL_DIM"] = "768"
    from diagnosis_model.grod.build import build_oavle
    rf = build_oavle(args.joint_ckpt, num_classes=1, freeze_encoder=True)
    net = rf.model.model.to(dev)
    pkg = torch.load(args.arm_ckpt, weights_only=False, map_location="cpu")
    net.load_state_dict(pkg["net_state"]); net.eval()
    freeze_backbone_nograd(net)
    res = int(rf.model.resolution); means = list(rf.means); stds = list(rf.stds)
    Q0 = net.num_queries

    enc = build_encoder(EncoderConfig(**pkg["encoder_config"])).to(dev); enc.load_state_dict(pkg["encoder_state"]); enc.eval()
    gate = RegionGate(init_temp=pkg["gate_init_temp"]).to(dev); gate.load_state_dict(pkg["gate_state"]); gate.eval()

    g_keep = None
    for split, sub in (("train", "train"), ("valid", "valid")):
        d = torch.load(Path(args.soft_dir) / f"{split}.pt", weights_only=False)
        cases = torch.load(Path(args.case_db_dir) / f"{split}_cases.pt", weights_only=False)
        ds = CaseImgDS(cases, f"{args.img_root}/{sub}", res, means, stds)
        ld = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers)
        N = len(cases)
        g_new = torch.zeros(N, 768); z_new = torch.zeros(N, Q0, 768, dtype=torch.bfloat16); w_new = torch.zeros(N, Q0)
        with torch.no_grad():
            for px, idx in ld:
                o = net(px.to(dev))
                g_new[idx] = o["pred_global"][:, :768].float().cpu() if o["pred_global"].dim() == 2 else o["pred_global"].float().cpu()
                z_new[idx] = F.normalize(o["pred_semantic"][:, :Q0].float(), dim=-1).to(torch.bfloat16).cpu()
                w_new[idx] = gate(o["pred_logits"][:, :Q0, 0].float()).cpu()
        d["g"], d["z_all"], d["w"] = g_new, z_new, w_new
        torch.save(d, out_dir / f"{split}.pt")
        print(f"[save] {out_dir/f'{split}.pt'} z={tuple(z_new.shape)} w Σ/row={w_new.sum(1).mean():.3f}")
        if split == "train":
            g_keep = (g_new, z_new, w_new)

    bank = encode_all_soft(enc, *g_keep, dev)
    torch.save({"bank_z": bank, "encoder_ckpt": args.arm_ckpt}, out_dir / "bank_z_soft.pt")
    print(f"[save] bank_z_soft.pt bank={tuple(bank.shape)}")


if __name__ == "__main__":
    main()
