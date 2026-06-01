"""Phase A (frozen) — extract RF-DETR decoder query features (`hs`) for each
case_db lesion box, matched by IoU, and join the per-lesion symptom category.

This is the frozen-detector probe for the detection+vl_classifier merge
(faithfulness-by-construction). We do NOT train RF-DETR here. We:

  1. Load a frozen RF-DETR (fish-finetuned, class-agnostic ABNORMAL detector).
  2. For each case in the existing case_db, run the image once, capture the
     decoder query features `hs` (300 queries x hidden_dim) via a forward hook
     on `class_embed`, and the predicted boxes.
  3. For each GT lesion box (stored in the case), greedily match the highest-IoU
     predicted query (IoU >= --iou_thresh); record that query's `hs` as the
     lesion's region feature.
  4. Join the per-lesion symptom `category_id` from the vl_classifier COCO
     (same file_name + identical bbox; verified 1:1 with the detection COCO).

Output `hs_cache.pt` is aligned 1:1 with the case_db split: a list of dicts,
one per case, each holding the matched `hs` per kept lesion plus diagnostics.
A later script trains a Linear(hidden_dim -> D) semantic head on top.

Run from repo root:
  PY=/home/lab603/anaconda3/envs/SDM/bin/python
  $PY -m diagnosis_model.grod.extract_hs \
      --case_db_dir diagnosis_model/cause_inference/outputs/case_db_raw \
      --split train \
      --det_ckpt diagnosis_model/detection/outputs/rfdetr/checkpoint_best_total.pth \
      --vlc_coco data/coco/_merged/train/_annotations.coco.json \
      --image_root data/detection/coco/_merged/train \
      --output_dir diagnosis_model/grod/outputs/hs_raw \
      --max_cases 20
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image


# ---------------------------------------------------------------------------
# IoU helpers (boxes in absolute xyxy pixel coords)
# ---------------------------------------------------------------------------

def xywh_to_xyxy(b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x, y, w, h = b
    return x, y, x + w, y + h


def iou_matrix(gt_xyxy: torch.Tensor, pred_xyxy: torch.Tensor) -> torch.Tensor:
    """gt: [G,4], pred: [P,4] -> IoU [G,P]. Coords absolute pixels."""
    G, P = gt_xyxy.size(0), pred_xyxy.size(0)
    if G == 0 or P == 0:
        return torch.zeros(G, P)
    gt = gt_xyxy.unsqueeze(1)        # [G,1,4]
    pr = pred_xyxy.unsqueeze(0)      # [1,P,4]
    ix1 = torch.maximum(gt[..., 0], pr[..., 0])
    iy1 = torch.maximum(gt[..., 1], pr[..., 1])
    ix2 = torch.minimum(gt[..., 2], pr[..., 2])
    iy2 = torch.minimum(gt[..., 3], pr[..., 3])
    iw = (ix2 - ix1).clamp(min=0)
    ih = (iy2 - iy1).clamp(min=0)
    inter = iw * ih
    ga = ((gt[..., 2] - gt[..., 0]) * (gt[..., 3] - gt[..., 1])).clamp(min=0)
    pa = ((pr[..., 2] - pr[..., 0]) * (pr[..., 3] - pr[..., 1])).clamp(min=0)
    union = ga + pa - inter
    return inter / union.clamp(min=1e-6)


def greedy_match(iou: torch.Tensor, thresh: float) -> List[int]:
    """For each GT row, assign the best free pred column with IoU>=thresh.

    Greedy by descending IoU; each pred query used at most once. Returns a list
    of length G with pred index (or -1 if unmatched).
    """
    G, P = iou.shape
    assign = [-1] * G
    if G == 0 or P == 0:
        return assign
    flat = [(iou[g, p].item(), g, p) for g in range(G) for p in range(P)]
    flat.sort(reverse=True)
    used_pred = set()
    used_gt = set()
    for v, g, p in flat:
        if v < thresh:
            break
        if g in used_gt or p in used_pred:
            continue
        assign[g] = p
        used_gt.add(g)
        used_pred.add(p)
    return assign


# ---------------------------------------------------------------------------
# vl_classifier COCO join: (file_name, bbox) -> symptom category_id
# ---------------------------------------------------------------------------

def build_category_lookup(vlc_coco_path: Path) -> Dict[Tuple[str, Tuple[int, int, int, int]], int]:
    with vlc_coco_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    id_to_fn = {im["id"]: im["file_name"] for im in coco["images"]}
    lookup: Dict[Tuple[str, Tuple[int, int, int, int]], int] = {}
    for a in coco["annotations"]:
        fn = id_to_fn.get(a["image_id"])
        if fn is None or "bbox" not in a:
            continue
        key = (fn, tuple(int(round(v)) for v in a["bbox"]))
        lookup[key] = int(a["category_id"])
    return lookup


# ---------------------------------------------------------------------------
# RF-DETR frozen forward with hs capture
# ---------------------------------------------------------------------------

def load_detector(det_ckpt: str, device: Optional[str] = None):
    from rfdetr import RFDETRMedium

    rf = RFDETRMedium(pretrain_weights=det_ckpt)
    net = rf.model.model  # the LWDETR nn.Module
    if device is not None:
        net = net.to(device)
    net.eval()
    device = next(net.parameters()).device
    means = list(rf.means)
    stds = list(rf.stds)
    resolution = int(rf.model.resolution)
    return rf, net, device, means, stds, resolution


def find_class_embed(net) -> torch.nn.Linear:
    for name, mod in net.named_modules():
        if name.split(".")[-1] == "class_embed" and isinstance(mod, torch.nn.Linear):
            return mod
    ce = getattr(net, "class_embed", None)
    if isinstance(ce, torch.nn.Linear):
        return ce
    raise RuntimeError("could not locate class_embed Linear on detector")


@torch.no_grad()
def detector_forward(
    net, class_embed, image: Image.Image, device, means, stds, resolution,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (hs[Q, Hd], pred_boxes_cxcywh_norm[Q,4]) for a single image,
    last decoder layer, single group (eval mode)."""
    cap: Dict[str, torch.Tensor] = {}

    def hook(mod, inp, out):
        # inp[0]: [layers, B, Q, Hd] (or [B,Q,Hd]); take last layer, batch 0.
        x = inp[0].detach()
        if x.dim() == 4:
            x = x[-1]
        cap["hs"] = x[0]  # [Q, Hd]

    h = class_embed.register_forward_hook(hook)
    try:
        img_t = TF.to_tensor(image)                       # [3,H,W] in [0,1]
        img_t = TF.resize(img_t, [resolution, resolution])
        img_t = TF.normalize(img_t, means, stds).unsqueeze(0).to(device)
        out = net(img_t)
    finally:
        h.remove()

    pred_boxes = out["pred_boxes"][0].detach().cpu()  # [Q,4] cx,cy,w,h normalized
    return cap["hs"].cpu(), pred_boxes


def cxcywh_norm_to_xyxy_abs(boxes: torch.Tensor, W: int, H: int) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = (cx - w / 2) * W
    y1 = (cy - h / 2) * H
    x2 = (cx + w / 2) * W
    y2 = (cy + h / 2) * H
    return torch.stack([x1, y1, x2, y2], dim=-1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Extract frozen RF-DETR hs per case_db lesion (IoU-matched).")
    ap.add_argument("--case_db_dir", type=str, required=True)
    ap.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"])
    ap.add_argument("--det_ckpt", type=str, required=True)
    ap.add_argument("--vlc_coco", type=str, required=True,
                    help="vl_classifier COCO (has symptom category_id) for the same split")
    ap.add_argument("--image_root", type=str, required=True,
                    help="dir holding the split's images (file_name resolves here)")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--iou_thresh", type=float, default=0.5)
    ap.add_argument("--max_cases", type=int, default=-1)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    case_db_dir = Path(args.case_db_dir)
    cases = torch.load(case_db_dir / f"{args.split}_cases.pt", weights_only=False)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    print(f"[load] {args.split}_cases.pt  n={len(cases)}")

    cat_lookup = build_category_lookup(Path(args.vlc_coco))
    print(f"[load] vlc category lookup: {len(cat_lookup)} (file_name,bbox) keys")

    rf, net, device, means, stds, resolution = load_detector(args.det_ckpt, args.device)
    class_embed = find_class_embed(net)
    print(f"[det] device={device} resolution={resolution} hidden_dim={class_embed.in_features}")

    image_root = Path(args.image_root)
    out_cases: List[Dict] = []

    n_gt = 0
    n_matched = 0
    n_cat_found = 0
    iou_vals: List[float] = []
    t0 = time.time()

    for ci, c in enumerate(cases):
        fn = c["file_name"]
        img_path = image_root / fn
        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        hs, pred_boxes = detector_forward(
            net, class_embed, image, device, means, stds, resolution,
        )
        pred_xyxy = cxcywh_norm_to_xyxy_abs(pred_boxes, W, H)

        gt_boxes_xywh = c["lesion_boxes_xywh"]
        if torch.is_tensor(gt_boxes_xywh):
            gt_boxes_xywh = gt_boxes_xywh.tolist()
        gt_xyxy = torch.tensor(
            [xywh_to_xyxy(tuple(b)) for b in gt_boxes_xywh], dtype=torch.float32,
        ) if gt_boxes_xywh else torch.zeros(0, 4)

        iou = iou_matrix(gt_xyxy, pred_xyxy)            # [G,P]
        assign = greedy_match(iou, args.iou_thresh)     # len G

        Hd = hs.size(-1)
        matched_hs: List[torch.Tensor] = []
        matched_cat: List[int] = []
        matched_iou: List[float] = []
        kept_lesion_idx: List[int] = []

        for g, p in enumerate(assign):
            n_gt += 1
            if p < 0:
                continue
            n_matched += 1
            v = iou[g, p].item()
            iou_vals.append(v)
            matched_hs.append(hs[p])
            matched_iou.append(v)
            kept_lesion_idx.append(g)
            # join symptom category by (file_name, exact bbox)
            key = (fn, tuple(int(round(x)) for x in gt_boxes_xywh[g]))
            cat = cat_lookup.get(key, -1)
            if cat >= 0:
                n_cat_found += 1
            matched_cat.append(cat)

        out_cases.append({
            "case_id": c["case_id"],
            "image_id": c["image_id"],
            "file_name": fn,
            "split": args.split,
            # aligned to kept lesions only:
            "kept_lesion_idx": torch.tensor(kept_lesion_idx, dtype=torch.long),
            "hs": torch.stack(matched_hs) if matched_hs else torch.zeros(0, Hd),
            "lesion_category_id": torch.tensor(matched_cat, dtype=torch.long),
            "match_iou": torch.tensor(matched_iou, dtype=torch.float32),
            "n_gt_lesions": len(assign),
        })
        image.close()

        if ci % 200 == 0:
            print(f"  [{ci}/{len(cases)}] match_rate={n_matched/max(1,n_gt):.3f} "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(out_cases, out_dir / f"hs_{args.split}.pt")

    mean_iou = sum(iou_vals) / max(1, len(iou_vals))
    summary = {
        "split": args.split,
        "n_cases": len(out_cases),
        "n_gt_lesions": n_gt,
        "n_matched": n_matched,
        "match_rate": n_matched / max(1, n_gt),
        "n_category_found": n_cat_found,
        "category_join_rate": n_cat_found / max(1, n_matched),
        "mean_matched_iou": mean_iou,
        "iou_thresh": args.iou_thresh,
        "hidden_dim": int(class_embed.in_features),
        "det_ckpt": args.det_ckpt,
        "case_db_dir": str(case_db_dir),
    }
    with (out_dir / f"summary_{args.split}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"[save] hs_{args.split}.pt  -> {out_dir}")


if __name__ == "__main__":
    main()
