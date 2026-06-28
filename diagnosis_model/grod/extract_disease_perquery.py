"""Per-query disease/threshold cache — the gating calibration input.

Reads the versioned `detection` view directly: `isHealthy:true` = healthy image
(no GT box), anything with boxes = diseased (GT boxes from the COCO annotations).
For every image run GROD once and store the FULL soft state aligned per query, plus
a per-query lesion label from GT-box IoU match:

    g  [768]   pred_global
    w  [300]   sigmoid(pred_logits[:, 0])        — per-query objectness
    y  [300]   is_lesion_i ∈ {0,1}               — 1 iff query IoU-matched a GT box
                                                   (healthy images: all 0)

Feeds the production gating defaults: compute_lesion_threshold (lesion-selection τ)
and calibrate_thresholds (abstain / display thresholds.json). Reuses extract_z_joint's
IoU machinery.

Output: {ART}/db/disease_perquery/{train,val}.pt  (val = the `valid` split; the
filename stays `val.pt` for the existing consumers).
Run from repo root (SDM env):
    $PY -m diagnosis_model.grod.extract_disease_perquery
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from diagnosis_model.grod.extract_hs import (
    xywh_to_xyxy, iou_matrix, greedy_match, cxcywh_norm_to_xyxy_abs,
)


class ImgDataset(Dataset):
    """Returns (pixel[3,res,res], orig_W, orig_H, idx) — boxes joined by idx after."""
    def __init__(self, paths, res, means, stds):
        self.paths, self.res, self.means, self.stds = paths, res, means, stds

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        W, H = img.size
        px = TF.normalize(TF.resize(TF.to_tensor(img), [self.res, self.res]),
                          self.means, self.stds)
        return px, W, H, i


class Grod:
    def __init__(self, joint_ckpt, global_sd, anchors, device="cuda"):
        self.dev = device
        os.environ["RFDETR_SEMANTIC_DIM"] = "768"
        os.environ["RFDETR_SEMANTIC_ANCHORS"] = os.path.abspath(anchors)
        os.environ["RFDETR_GLOBAL_DIM"] = "768"
        from diagnosis_model.grod.build import load_oavle
        self.net, self.res, self.means, self.stds = load_oavle(joint_ckpt, device=device)
        self.net.global_embed.load_state_dict(torch.load(global_sd, map_location=device))
        # anchors for centered semantic saliency (anisotropy fix: subtract centroid)
        A = torch.load(anchors, weights_only=False)["anchor_embs"].float().to(device)
        self.mu = A.mean(0, keepdim=True)
        self.A_c = F.normalize(A - self.mu, dim=-1)          # [C,768] centered+normed

    @torch.no_grad()
    def forward(self, px):
        out = self.net(px.to(self.dev))
        g = out["pred_global"].float().cpu()                 # [B,768]
        w = out["pred_logits"][..., 0].sigmoid().float()     # [B,300]
        boxes = out["pred_boxes"].float()                    # [B,300,4] cxcywh norm
        z = F.normalize(out["pred_semantic"].float(), dim=-1)  # [B,300,768]
        # per-query semantic saliency (centered z·anchor): max-cos, margin, softmax peak
        zc = F.normalize(z - self.mu, dim=-1)
        sim = torch.einsum("bqd,cd->bqc", zc, self.A_c)      # [B,300,C]
        top2 = sim.topk(2, dim=-1).values
        qfeat = torch.stack([
            w,                                               # objectness
            top2[..., 0].clamp(0, 1),                        # centered max-cos saliency
            (top2[..., 0] - top2[..., 1]),                   # centered margin (commit to 1 symptom)
            torch.softmax(sim / 0.07, dim=-1).amax(-1),      # centered softmax peak
            boxes[..., 2] * boxes[..., 3],                   # box area (norm)
        ], dim=-1)                                           # [B,300,5]
        return g.cpu(), w.cpu(), boxes.cpu(), qfeat.cpu()


def gather_from_coco(det_root):
    """split -> list of (path, gt_boxes_xywh|None, is_diseased), from detection COCO.

    Output key 'val' = the `valid` folder (kept for the existing consumers).
    """
    det_root = Path(det_root)
    out = {}
    for fold, key in [("train", "train"), ("valid", "val")]:
        coco = json.load(open(det_root / fold / "_annotations.coco.json"))
        by_img = {}
        for a in coco["annotations"]:
            by_img.setdefault(a["image_id"], []).append([float(x) for x in a["bbox"]])  # xywh
        items = []
        for im in coco["images"]:
            path = str(det_root / fold / im["file_name"])
            if im.get("isHealthy"):
                items.append((path, None, 0))
            else:
                items.append((path, by_img.get(im["id"], []), 1))
        out[key] = items
        d = sum(x[2] for x in items)
        print(f"[gather] {fold}: diseased={d} healthy={len(items)-d} total={len(items)}")
    return out


def extract_split(grod, items, iou_thresh, batch_size, workers):
    paths = [it[0] for it in items]
    loader = DataLoader(ImgDataset(paths, grod.res, grod.means, grod.stds),
                        batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    N, Q = len(items), 300
    G = torch.empty(N, 768); Wt = torch.empty(N, Q); Y = torch.zeros(N, Q)
    QF = torch.empty(N, Q, 5)
    n_gt = n_match = 0
    for bi, (px, Ws, Hs, idxs) in enumerate(loader):
        g, w, boxes, qfeat = grod.forward(px)
        for j, idx in enumerate(idxs.tolist()):
            G[idx] = g[j]; Wt[idx] = w[j]; QF[idx] = qfeat[j]
            _, gt, dis = items[idx]
            if not dis or not gt:
                continue
            Wo, Ho = int(Ws[j]), int(Hs[j])
            pred_xyxy = cxcywh_norm_to_xyxy_abs(boxes[j], Wo, Ho)
            gt_xyxy = torch.tensor([xywh_to_xyxy(tuple(b)) for b in gt], dtype=torch.float32)
            assign = greedy_match(iou_matrix(gt_xyxy, pred_xyxy), iou_thresh)
            for p in assign:
                n_gt += 1
                if p >= 0:
                    Y[idx, p] = 1.0; n_match += 1
        if (bi + 1) % 50 == 0:
            print(f"  ...{(bi+1)*batch_size}/{N}  match_rate={n_match/max(1,n_gt):.3f}", flush=True)
    is_dis = torch.tensor([it[2] for it in items], dtype=torch.float32)
    print(f"[split] N={N} GT lesions={n_gt} matched={n_match} ({n_match/max(1,n_gt):.3f}); "
          f"mean lesions/diseased-img={Y[is_dis==1].sum(1).mean():.2f}")
    return {"g": G, "w": Wt, "y": Y, "is_diseased": is_dis, "qfeat": QF,
            "qfeat_names": ["objectness", "sal_maxcos", "sal_margin", "sal_softmax", "box_area"]}


def main():
    ap = argparse.ArgumentParser()
    ART = "data/processed/current/artifacts"
    ap.add_argument("--joint_ckpt", default=f"{ART}/models/joint_rfdetr/checkpoint_best_regular.pth")
    ap.add_argument("--global_sd", default=f"{ART}/models/distilled_global_rawP/global_embed_state_dict.pt")
    ap.add_argument("--anchors", default=f"{ART}/models/text_anchors.pt")
    ap.add_argument("--det_root", default="data/processed/current/detection")
    ap.add_argument("--out_dir", default=f"{ART}/db/disease_perquery")
    ap.add_argument("--iou_thresh", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    items = gather_from_coco(args.det_root)
    if args.limit:
        items = {s: [x for x in v if x[2] == 1][:args.limit] + [x for x in v if x[2] == 0][:args.limit]
                 for s, v in items.items()}
    grod = Grod(args.joint_ckpt, args.global_sd, args.anchors, dev)
    for s in ("train", "val"):
        print(f"[extract] {s}")
        data = extract_split(grod, items[s], args.iou_thresh, args.batch_size, args.workers)
        suffix = "_smoke" if args.limit else ""
        torch.save(data, out_dir / f"{s}{suffix}.pt")
        print(f"[save] {out_dir}/{s}{suffix}.pt")


if __name__ == "__main__":
    main()
