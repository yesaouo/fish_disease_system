"""L3-lite end-to-end probe — unfreeze the decoder (transformer + heads) on top
of a FROZEN backbone, so retrieval gradient reaches the detection representation
itself (decoder query features), not just the heads (L2).

Cost trick: the DINOv3 backbone is frozen and expensive; we run it in a no_grad
context each step (cheap, no activation storage) and let gradients flow only
through the transformer + heads + gate + aggregator. Multi-scale backbone
features are too big to cache (~25GB), so we re-run the frozen backbone live
rather than caching it.

bbox protection WITHOUT reimplementing the Hungarian criterion: a per-query
distillation of pred_boxes / pred_logits toward the FROZEN joint model's outputs
(cached once at init), so the decoder can adapt for retrieval but its boxes stay
near the detector's — a cheap stand-in for L_loc.

Losses:  L_retr (listwise-KL teacher distill + case-cause InfoNCE)
       + lambda_sym * L_sym (matched-query z -> symptom anchor, matched idx fixed)
       + lambda_box * L_box (per-query pred_boxes MSE + pred_logits BCE vs frozen)

Then reuse regen_gated_soft_L2 + train_ceah_soft + faithfulness_eval_soft.

Run (smoke): $PY -m diagnosis_model.grod.finetune_e2e_L3lite --limit 256 --epochs 2
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF

from diagnosis_model.cause_inference.models.case_encoder import (
    EncoderConfig, build_encoder, listwise_kl_loss, case_cause_infonce_loss,
)
from diagnosis_model.grod.region_gate import RegionGate
from diagnosis_model.grod.finetune_e2e_grace import retrieval_eval
from diagnosis_model.grod.train_case_encoder_soft import load_soft


class CaseImgDS(Dataset):
    def __init__(self, cases, img_root, res, means, stds):
        self.paths = [str(Path(img_root) / c["file_name"]) for c in cases]
        self.res, self.means, self.stds = res, means, stds

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = TF.to_tensor(Image.open(self.paths[i]).convert("RGB"))
        img = TF.resize(img, [self.res, self.res])
        return TF.normalize(img, self.means, self.stds), i


def freeze_backbone_nograd(net):
    """Freeze backbone params + run its forward under no_grad (cheap, no grad to it)."""
    for p in net.backbone.parameters():
        p.requires_grad = False
    orig = net.backbone.forward

    def wrapped(*a, **k):
        with torch.no_grad():
            return orig(*a, **k)
    net.backbone.forward = wrapped


def main():
    ap = argparse.ArgumentParser()
    ART = "data/processed/current/artifacts"
    ap.add_argument("--soft_dir", default=f"{ART}/db/soft_inputs")
    ap.add_argument("--hs_dir", default=f"{ART}/db/hs_all")
    ap.add_argument("--case_db_dir", default=f"{ART}/db/case_db_jointDistRawP")
    ap.add_argument("--teacher_path", default=f"{ART}/db/case_db_jointDistRawP/teacher_train_train.pt")
    ap.add_argument("--enc_ckpt", default=f"{ART}/models/encoder_grod_soft/best_encoder.pt")
    ap.add_argument("--joint_ckpt", default=f"{ART}/models/joint_rfdetr/checkpoint_best_regular.pth")
    ap.add_argument("--anchors", default=f"{ART}/models/text_anchors.pt")
    ap.add_argument("--img_root", default="data/processed/current/detection")
    ap.add_argument("--output_dir", default=f"{ART}/models/grace_e2e_L3lite")
    ap.add_argument("--lambda_sym", type=float, default=1.0)
    ap.add_argument("--lambda_box", type=float, default=5.0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gate_lr", type=float, default=1e-3)
    ap.add_argument("--dec_lr", type=float, default=1e-5)
    ap.add_argument("--init_temp", type=float, default=0.3)
    ap.add_argument("--infonce_weight", type=float, default=0.5)
    ap.add_argument("--infonce_temp", type=float, default=0.07)
    ap.add_argument("--top_k_cases", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    if args.seed:
        torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "config.json", "w"), indent=2)

    # --- data ---
    train_cases = torch.load(Path(args.case_db_dir) / "train_cases.pt", weights_only=False)
    valid_cases = torch.load(Path(args.case_db_dir) / "valid_cases.pt", weights_only=False)
    cp = torch.load(Path(args.case_db_dir) / "cause_text_embs.pt", weights_only=False)
    cause_embs = F.normalize(cp["embeddings"].float(), dim=-1).to(dev)
    anchors = F.normalize(torch.load(args.anchors, weights_only=False)["anchor_embs"].float(), dim=-1).to(dev)
    g_tr, _z, _w, cidx_tr = load_soft(Path(args.soft_dir) / "train.pt")
    g_va, _zv, _wv, _ = load_soft(Path(args.soft_dir) / "valid.pt")
    hs_tr = torch.load(Path(args.hs_dir) / "hs_all_train.pt", weights_only=False)
    hs_va = torch.load(Path(args.hs_dir) / "hs_all_valid.pt", weights_only=False)
    mq_tr = [d["matched_qidx"] for d in hs_tr]; mc_tr = [d["matched_cat"] for d in hs_tr]
    teacher = torch.load(args.teacher_path, weights_only=False, map_location="cpu")["scores"]
    if args.limit:
        n = args.limit
        train_cases, valid_cases = train_cases[:n], valid_cases[:n]
        g_tr, cidx_tr, g_va = g_tr[:n], cidx_tr[:n], g_va[:n]
        mq_tr, mc_tr = mq_tr[:n], mc_tr[:n]
    teacher = teacher[:len(train_cases), :len(train_cases)]
    N = len(train_cases)

    # --- model: backbone frozen (no_grad), transformer + heads trainable ---
    os.environ["RFDETR_SEMANTIC_DIM"] = "768"
    os.environ["RFDETR_SEMANTIC_ANCHORS"] = os.path.abspath(args.anchors)
    os.environ["RFDETR_GLOBAL_DIM"] = "768"
    from diagnosis_model.grod.build import build_oavle
    rf = build_oavle(args.joint_ckpt, num_classes=1, freeze_encoder=True)
    net = rf.model.model.to(dev)
    net.global_embed.load_state_dict(
        torch.load(f"{ART}/models/distilled_global_rawP/global_embed_state_dict.pt", map_location=dev))
    freeze_backbone_nograd(net)
    res = int(rf.model.resolution); means = list(rf.means); stds = list(rf.stds)

    # aggregator + gate warm-start
    pkg = torch.load(args.enc_ckpt, weights_only=False, map_location="cpu")
    enc = build_encoder(EncoderConfig(**pkg["encoder_config"])).to(dev); enc.load_state_dict(pkg["encoder_state"])
    gate = RegionGate(init_temp=args.init_temp).to(dev)
    if "gate_state" in pkg:
        gate.load_state_dict(pkg["gate_state"])

    # frozen box/logit reference (per case) for distillation — one no_grad pass
    ds_tr = CaseImgDS(train_cases, f"{args.img_root}/train", res, means, stds)
    ld_ref = DataLoader(ds_tr, batch_size=args.batch_size, num_workers=args.workers)
    net.eval()
    fb = torch.zeros(N, 300, 4); flg = torch.zeros(N, 300)
    with torch.no_grad():
        for px, idx in ld_ref:
            o = net(px.to(dev))
            fb[idx] = o["pred_boxes"].float().cpu()
            flg[idx] = o["pred_logits"][..., 0].float().cpu()
    print(f"[ref] frozen boxes/logits cached {tuple(fb.shape)}")

    # trainable: transformer(enc+dec) + heads + gate + aggregator
    dec_params = [p for n_, p in net.named_parameters() if p.requires_grad]  # backbone already frozen
    opt = torch.optim.AdamW([
        {"params": enc.parameters(), "lr": args.lr},
        {"params": gate.parameters(), "lr": args.gate_lr},
        {"params": dec_params, "lr": args.dec_lr},
    ], weight_decay=1e-2)
    print(f"[trainable] transformer+heads params={sum(p.numel() for p in dec_params)/1e6:.1f}M")

    @torch.no_grad()
    def eval_retr():
        net.eval(); enc.eval()
        def enc_split(cases, g_all, img_sub):
            ds = CaseImgDS(cases, f"{args.img_root}/{img_sub}", res, means, stds)
            ld = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers)
            H = torch.zeros(len(cases), enc.out_dim if hasattr(enc, "out_dim") else 768)
            for px, idx in ld:
                o = net(px.to(dev))
                z = F.normalize(o["pred_semantic"].float(), dim=-1)
                w = gate(o["pred_logits"][..., 0].float())
                lens = torch.full((px.size(0),), z.size(1), device=dev, dtype=torch.long)
                h = enc(o["pred_global"].float(), z, lens, lesion_weights=w)
                H[idx] = h.float().cpu()
            return H
        bank = enc_split(train_cases, g_tr, "train")
        Hva = enc_split(valid_cases, g_va, "valid")
        m = retrieval_eval(bank.to(dev), Hva, train_cases, valid_cases, cause_embs, dev, top_k_cases=args.top_k_cases)
        enc.train(); net.train()
        return m

    m = eval_retr(); print(f"[ep0] sem R@1={m['R@1']:.4f} R@10={m['R@10']:.4f} MRR={m['MRR']:.4f}")

    ld = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    best = -1.0; log = []
    for ep in range(1, args.epochs + 1):
        net.train(); enc.train(); t0 = time.time(); losses = []
        Q0 = net.num_queries
        for px, idx in ld:
            px = px.to(dev); b = idx.size(0)
            o = net(px)
            # train mode emits group_detr*Q queries; keep the primary group (== eval)
            pb = o["pred_boxes"][:, :Q0].float()
            z = F.normalize(o["pred_semantic"][:, :Q0].float(), dim=-1)  # [b,300,768]
            ol = o["pred_logits"][:, :Q0, 0].float()                    # [b,300]
            w = gate(ol)
            lens = torch.full((b,), z.size(1), device=dev, dtype=torch.long)
            h = enc(o["pred_global"].float(), z, lens, lesion_weights=w)
            tb = teacher[idx][:, idx].to(dev).float()
            ld_kl = listwise_kl_loss(h, tb, temp_target=0.1, temp_pred=0.1)
            V = cause_embs.size(0)
            pos = torch.zeros(b, V, dtype=torch.bool, device=dev)
            for i, ci in enumerate(idx.tolist()):
                cc = cidx_tr[ci]
                if cc:
                    pos[i, torch.tensor(cc, dtype=torch.long, device=dev)] = True
            li = case_cause_infonce_loss(h, cause_embs, pos, temp=args.infonce_temp)
            loss = ld_kl + args.infonce_weight * li
            # L_sym on matched queries (fixed idx)
            if args.lambda_sym > 0:
                zs, tg = [], []
                for i, ci in enumerate(idx.tolist()):
                    if len(mq_tr[ci]):
                        zs.append(z[i, mq_tr[ci].to(dev)]); tg.append(mc_tr[ci].to(dev))
                if zs:
                    loss = loss + args.lambda_sym * F.cross_entropy(
                        torch.cat(zs) @ anchors.T / 0.07, torch.cat(tg))
            # L_box distill vs frozen (per query)
            if args.lambda_box > 0:
                loss = loss + args.lambda_box * (
                    F.mse_loss(pb, fb[idx].to(dev))
                    + F.binary_cross_entropy_with_logits(ol, flg[idx].to(dev).sigmoid()))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(dec_params + list(enc.parameters()) + list(gate.parameters()), 1.0)
            opt.step(); losses.append(loss.item())
        m = eval_retr()
        print(f"  [ep{ep}] loss={np.mean(losses):.4f} R@10={m['R@10']:.4f} MRR={m['MRR']:.4f} dur={time.time()-t0:.0f}s")
        log.append({"epoch": ep, "loss": float(np.mean(losses)), **{f"sem_{k}": v for k, v in m.items()}})
        if m["MRR"] > best:
            best = m["MRR"]
            torch.save({"encoder_state": {k: v.cpu() for k, v in enc.state_dict().items()},
                        "encoder_config": vars(EncoderConfig(**pkg["encoder_config"])),
                        "gate_state": {k: v.cpu() for k, v in gate.state_dict().items()},
                        "gate_init_temp": args.init_temp,
                        "net_state": {k: v.cpu() for k, v in net.state_dict().items()}},
                       out / "best.pt")
    json.dump(log, open(out / "train_log.json", "w"), indent=2)
    print(f"[done] best MRR={best:.4f} -> {out/'best.pt'}")


if __name__ == "__main__":
    main()
