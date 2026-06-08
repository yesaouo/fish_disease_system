"""Soft-pipeline retrain — step #1: extract soft GROD inputs per case.

For every case_db case (train + valid), run GROD once and store the *soft
inference distribution* (all 300 queries, no hard threshold):

    g      [768]        pred_global (GROD global head)
    z_all  [300, 768]   L2-normed pred_semantic (every query kept)
    w      [300]        sigmoid(pred_logits[:, 0])  — ABNORMAL objectness (col 0)

plus case_id + cause_emb_indices so the soft Aggregator / CEAH retrain can join
their supervision targets from the existing case_db.

This replaces the hard GT-lesion representation (1-2 clean lesions per case) with
what the soft pipeline actually sees at inference, so downstream retrains on the
matched distribution. Output: outputs/soft_inputs/{train,valid}.pt

Run from repo root:
    $PY -m diagnosis_model.grod.extract_soft_inputs
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ImgDataset(Dataset):
    def __init__(self, paths, res, means, stds):
        self.paths, self.res, self.means, self.stds = paths, res, means, stds

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return TF.normalize(TF.resize(TF.to_tensor(img), [self.res, self.res]),
                            self.means, self.stds)


class Grod:
    def __init__(self, joint_ckpt, global_sd, anchors, device="cuda"):
        self.dev = device
        os.environ["RFDETR_SEMANTIC_DIM"] = "768"
        os.environ["RFDETR_SEMANTIC_ANCHORS"] = os.path.abspath(anchors)
        os.environ["RFDETR_GLOBAL_DIM"] = "768"
        from rfdetr import RFDETRMedium
        rf = RFDETRMedium(pretrain_weights=joint_ckpt, num_classes=1)
        self.net = rf.model.model.to(device).eval()
        self.net.global_embed.load_state_dict(torch.load(global_sd, map_location=device))
        self.res = int(rf.model.resolution)
        self.means, self.stds = list(rf.means), list(rf.stds)

    @torch.no_grad()
    def forward(self, px):
        out = self.net(px.to(self.dev))
        g = out["pred_global"]                                   # [B, 768]
        z = F.normalize(out["pred_semantic"].float(), dim=-1)    # [B, Q, 768]
        w = out["pred_logits"][..., 0].sigmoid()                 # [B, Q] ABNORMAL (col 0)
        return g.float().cpu(), z.cpu(), w.float().cpu()


def extract_split(grod, cases, img_root, coco_split, batch_size, workers):
    paths = [str(Path(img_root) / coco_split / c["file_name"]) for c in cases]
    miss = [p for p in paths if not Path(p).exists()]
    if miss:
        raise FileNotFoundError(f"{len(miss)} images missing, e.g. {miss[0]}")
    loader = DataLoader(ImgDataset(paths, grod.res, grod.means, grod.stds),
                        batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    gs, zs, ws = [], [], []
    for bi, px in enumerate(loader):
        g, z, w = grod.forward(px)
        gs.append(g.to(torch.bfloat16)); zs.append(z.to(torch.bfloat16)); ws.append(w)
        if (bi + 1) % 50 == 0:
            print(f"  ...{(bi+1)*batch_size}/{len(paths)}")
    return {
        "g": torch.cat(gs),                                      # [N,768] bf16
        "z_all": torch.cat(zs),                                  # [N,Q,768] bf16
        "w": torch.cat(ws),                                      # [N,Q] fp32
        "case_id": [c["case_id"] for c in cases],
        "cause_emb_indices": [c["cause_emb_indices"] for c in cases],
    }


def main():
    ap = argparse.ArgumentParser()
    ART = "data/processed/current/artifacts"
    ap.add_argument("--joint_ckpt", default=f"{ART}/models/joint_rfdetr/checkpoint_best_regular.pth")
    ap.add_argument("--global_sd", default=f"{ART}/models/distilled_global_rawP/global_embed_state_dict.pt")
    ap.add_argument("--anchors", default=f"{ART}/models/text_anchors.pt")
    ap.add_argument("--case_db_dir", default=f"{ART}/db/case_db_jointDistRawP")
    ap.add_argument("--img_root", default="data/processed/current/detection")
    ap.add_argument("--out_dir", default=f"{ART}/db/soft_inputs")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: cap cases per split")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    grod = Grod(args.joint_ckpt, args.global_sd, args.anchors, dev)

    for fname, coco_split in [("train_cases.pt", "train"), ("valid_cases.pt", "valid")]:
        cases = torch.load(Path(args.case_db_dir) / fname, weights_only=False)
        if args.limit:
            cases = cases[:args.limit]
        print(f"[extract] {coco_split}: {len(cases)} cases")
        data = extract_split(grod, cases, args.img_root, coco_split,
                             args.batch_size, args.workers)
        suffix = "_smoke" if args.limit else ""
        out = out_dir / f"{coco_split}{suffix}.pt"
        torch.save(data, out)
        print(f"[save] {out}  g={tuple(data['g'].shape)} z_all={tuple(data['z_all'].shape)} "
              f"w={tuple(data['w'].shape)}")


if __name__ == "__main__":
    main()
