"""Pointing game — pixel-level localization faithfulness, GROD vs baselines.

A faithful saliency method puts its peak on the lesion. For each GT lesion box
and each method, generate a pixel heatmap, take its argmax, and count a hit when
the peak falls inside the GT box. Hit rate = fraction of lesions localized.

This is the pixel-level *corroboration* of GROD's routing faithfulness; the
headline faithfulness number stays the token-level lesion-mask drop in
cause_inference/faithfulness_eval.py (+0.0141 vs −0.0314). Pointing game has no
external-judge coupling (no deletion score, no SigLIP re-encode), so the
"method focuses here vs judge looks there" mismatch that breaks deletion does
not arise.

Methods:
  - grod_attn   : the lesion query's MSDeformAttn sampling points, weight-splatted
  - grad_cam    : Grad-CAM on RF-DETR's backbone feature map (target = z·anchor)
  - grad_cam_pp : Grad-CAM++ (higher-order gradient weighting)
  - rise        : RISE — black-box random-masking saliency (model-agnostic)
  - random      : random peak — chance lower bound (≈ bbox area / image area)

Run from repo root:
  python -m diagnosis_model.grod.pointing_game \
      --joint_ckpt diagnosis_model/grod/outputs/joint_rfdetr/checkpoint_best_regular.pth \
      --anchors diagnosis_model/grod/outputs/text_anchors.pt \
      --coco data/coco/_merged/valid/_annotations.coco.json \
      --image_root data/detection/coco/_merged/valid \
      --methods grod_attn grad_cam random \
      --output diagnosis_model/grod/outputs/pointing_game.json \
      --max_images 200
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


# ---------------------------------------------------------------------------
# Heatmaps (pixel-space importance for one lesion query)
# ---------------------------------------------------------------------------

def grad_cam_heatmap(net, feat_store, grad_store, img_t, query_idx, anchor_vec,
                     H: int, W: int) -> np.ndarray:
    feat_store.clear(); grad_store.clear()
    net.zero_grad(set_to_none=True)
    out = net(img_t)
    z = out["pred_semantic"][0, query_idx]
    target = (F.normalize(z, dim=-1) * anchor_vec).sum()
    target.backward()
    act, grad = feat_store["f"], grad_store["g"]
    cam = F.relu((grad.mean(dim=(2, 3), keepdim=True) * act).sum(dim=1))[0]
    cam = cam / (cam.max() + 1e-9)
    cam = F.interpolate(cam[None, None], size=(H, W), mode="bilinear",
                        align_corners=False)[0, 0]
    return cam.detach().cpu().numpy()


def grad_cam_pp_heatmap(net, feat_store, grad_store, img_t, query_idx, anchor_vec,
                        H: int, W: int) -> np.ndarray:
    """Grad-CAM++: pixel-wise weighting via higher-order gradients (Chattopadhyay
    et al. 2018). Same backward as Grad-CAM, different channel weights."""
    feat_store.clear(); grad_store.clear()
    net.zero_grad(set_to_none=True)
    out = net(img_t)
    z = out["pred_semantic"][0, query_idx]
    target = (F.normalize(z, dim=-1) * anchor_vec).sum()
    target.backward()
    act, grad = feat_store["f"], grad_store["g"]            # [1,C,h,w]
    g2, g3 = grad.pow(2), grad.pow(3)
    denom = 2 * g2 + (act * g3).sum(dim=(2, 3), keepdim=True)
    alpha = g2 / denom.clamp_min(1e-9)
    weights = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(dim=1))[0]
    cam = cam / (cam.max() + 1e-9)
    cam = F.interpolate(cam[None, None], size=(H, W), mode="bilinear",
                        align_corners=False)[0, 0]
    return cam.detach().cpu().numpy()


@torch.no_grad()
def rise_heatmap(net, img_t, query_idx, anchor_vec, H: int, W: int,
                 n_masks: int = 500, grid: int = 8, p_keep: float = 0.5) -> np.ndarray:
    """RISE (Petsiuk et al. 2018): black-box saliency. Randomly mask the input,
    weight each mask by how much the query's z·anchor score survives. No
    gradients, no architecture assumptions — the most model-agnostic baseline."""
    dev = img_t.device
    importance = torch.zeros(H, W, device=dev)
    for i in range(0, n_masks, 50):
        b = min(50, n_masks - i)
        small = (torch.rand(b, 1, grid, grid, device=dev) < p_keep).float()
        masks = F.interpolate(small, size=(img_t.shape[-2], img_t.shape[-1]),
                              mode="bilinear", align_corners=False)
        out = net(img_t * masks)                        # [b,Q,D] via broadcast batch
        z = F.normalize(out["pred_semantic"][:, query_idx], dim=-1)   # [b,D]
        s = (z * anchor_vec).sum(-1).clamp_min(0)        # [b]
        m_full = F.interpolate(small, size=(H, W), mode="bilinear",
                               align_corners=False)[:, 0]  # [b,H,W]
        importance += (s[:, None, None] * m_full).sum(0)
    hm = (importance / importance.max().clamp_min(1e-9)).cpu().numpy()
    return hm.astype(np.float32)


def attn_splat_heatmap(sampling_store, query_idx, H: int, W: int) -> np.ndarray:
    """Splat the lesion query's deformable sampling points (loc × weight) onto a
    pixel grid. Locations are normalized [0,1]; accumulate weights, then blur."""
    locs = sampling_store["loc"]     # [Lq, n_heads, n_levels, n_points, 2] (norm xy)
    wts = sampling_store["w"]        # [Lq, n_heads, n_levels*n_points]
    L = locs[query_idx]              # [n_heads, n_levels, n_points, 2]
    nh, nl, npt, _ = L.shape
    Wt = wts[query_idx].reshape(nh, nl, npt)
    hm = np.zeros((H, W), dtype=np.float32)
    L = L.detach(); Wt = Wt.detach()
    xs = (L[..., 0].reshape(-1).clamp(0, 1) * (W - 1)).round().long().cpu().numpy()
    ys = (L[..., 1].reshape(-1).clamp(0, 1) * (H - 1)).round().long().cpu().numpy()
    ww = Wt.reshape(-1).cpu().numpy()
    for x, y, w in zip(xs, ys, ww):
        hm[y, x] += w
    # light blur so a single peak pixel isn't brittle
    t = torch.from_numpy(hm)[None, None]
    k = 15
    t = F.avg_pool2d(F.pad(t, [k // 2] * 4, mode="reflect"), k, stride=1)
    return t[0, 0].numpy()


def random_heatmap(H: int, W: int) -> np.ndarray:
    return np.random.rand(H, W).astype(np.float32)


def peak_in_bbox(heatmap: np.ndarray, bbox_xyxy) -> bool:
    y, x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    return (x1 <= x < x2) and (y1 <= y < y2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint_ckpt", type=str, required=True)
    ap.add_argument("--anchors", type=str, required=True)
    ap.add_argument("--coco", type=str, required=True)
    ap.add_argument("--image_root", type=str, required=True)
    ap.add_argument("--methods", type=str, nargs="+",
                    default=["grod_attn", "grad_cam", "grad_cam_pp", "rise", "random"],
                    choices=["grod_attn", "grad_cam", "grad_cam_pp", "rise", "random"])
    ap.add_argument("--rise_masks", type=int, default=500,
                    help="number of random masks for RISE")
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--max_images", type=int, default=-1)
    ap.add_argument("--iou_thresh", type=float, default=0.5)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.environ["RFDETR_SEMANTIC_DIM"] = "768"
    os.environ["RFDETR_SEMANTIC_ANCHORS"] = os.path.abspath(args.anchors)
    from diagnosis_model.grod.build import load_oavle
    from rfdetr.models.ops.modules import MSDeformAttn
    from diagnosis_model.grod.extract_hs import (
        iou_matrix, greedy_match, cxcywh_norm_to_xyxy_abs, xywh_to_xyxy,
    )

    net, resolution, means, stds = load_oavle(args.joint_ckpt, device=args.device)
    text_anchors = torch.load(args.anchors, map_location=args.device,
                              weights_only=False)["anchor_embs"].to(args.device)

    need_cam = bool({"grad_cam", "grad_cam_pp"} & set(args.methods))
    need_attn = "grod_attn" in args.methods

    feat_store: Dict[str, torch.Tensor] = {}
    grad_store: Dict[str, torch.Tensor] = {}
    if need_cam:
        proj = dict(net.named_modules())["backbone.0.projector"]
        def fwd_hook(m, i, o):
            if not torch.is_grad_enabled():
                return
            t = o[0] if isinstance(o, (list, tuple)) else o
            if not t.requires_grad:
                t.requires_grad_(True)
            feat_store["f"] = t
            t.register_hook(lambda g: grad_store.__setitem__("g", g))
        proj.register_forward_hook(fwd_hook)

    # hook the LAST decoder layer's deformable cross-attn to grab sampling pts
    sampling_store: Dict[str, torch.Tensor] = {}
    if need_attn:
        deform_layers = [m for m in net.modules() if isinstance(m, MSDeformAttn)]
        last_deform = deform_layers[-1]
        def attn_pre_hook(module, args_in, kwargs_in):
            # forward(query, reference_points, input_flatten, input_spatial_shapes, ...)
            query = args_in[0] if len(args_in) > 0 else kwargs_in["query"]
            ref = args_in[1] if len(args_in) > 1 else kwargs_in["reference_points"]
            iss = args_in[3] if len(args_in) > 3 else kwargs_in["input_spatial_shapes"]
            N, Lq, _ = query.shape
            so = module.sampling_offsets(query).view(
                N, Lq, module.n_heads, module.n_levels, module.n_points, 2)
            aw = F.softmax(module.attention_weights(query).view(
                N, Lq, module.n_heads, module.n_levels * module.n_points), -1)
            # replicate forward's sampling_locations (normalized xy in [0,1])
            if ref.shape[-1] == 2:
                offnorm = torch.stack([iss[..., 1], iss[..., 0]], -1)
                loc = ref[:, :, None, :, None, :] + so / offnorm[None, None, None, :, None, :]
            else:  # 4
                loc = (ref[:, :, None, :, None, :2]
                       + so / module.n_points * ref[:, :, None, :, None, 2:] * 0.5)
            sampling_store["loc"] = loc[0]           # [Lq,nh,nl,npt,2]
            sampling_store["w"] = aw[0]              # [Lq,nh,nl*npt]
        last_deform.register_forward_pre_hook(attn_pre_hook, with_kwargs=True)

    coco = json.load(open(args.coco))
    id2fn = {im["id"]: im["file_name"] for im in coco["images"]}
    anns_by_img: Dict[int, List[dict]] = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    img_ids = list(anns_by_img.keys())
    if args.max_images > 0:
        img_ids = img_ids[: args.max_images]

    hits: Dict[str, int] = {m: 0 for m in args.methods}
    n_lesions = 0
    bbox_area_frac = []

    for ii, img_id in enumerate(img_ids):
        pil = Image.open(Path(args.image_root) / id2fn[img_id]).convert("RGB")
        W, H = pil.size
        img_t = TF.normalize(TF.resize(TF.to_tensor(pil), [resolution, resolution]),
                             means, stds).unsqueeze(0).to(args.device)
        with torch.no_grad():
            out = net(img_t)   # also fills sampling_store via pre-hook
        pred_xyxy = cxcywh_norm_to_xyxy_abs(out["pred_boxes"][0].cpu(), W, H)

        gt = [(a["bbox"], int(a["category_id"])) for a in anns_by_img[img_id]
              if "bbox" in a and "category_id" in a]
        gt_xyxy = torch.tensor([xywh_to_xyxy(tuple(b)) for b, _ in gt],
                               dtype=torch.float32) if gt else torch.zeros(0, 4)
        assign = greedy_match(iou_matrix(gt_xyxy, pred_xyxy), args.iou_thresh)

        for g, qidx in enumerate(assign):
            if qidx < 0:
                continue
            cat = gt[g][1]
            bbox = [int(gt_xyxy[g][0]), int(gt_xyxy[g][1]),
                    int(gt_xyxy[g][2]), int(gt_xyxy[g][3])]
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            n_lesions += 1
            bbox_area_frac.append(((bbox[2]-bbox[0])*(bbox[3]-bbox[1])) / (H*W))
            for m in args.methods:
                if m == "random":
                    hm = random_heatmap(H, W)
                elif m == "grod_attn":
                    hm = attn_splat_heatmap(sampling_store, qidx, H, W)
                elif m == "grad_cam":
                    hm = grad_cam_heatmap(net, feat_store, grad_store, img_t, qidx,
                                          text_anchors[cat], H, W)
                elif m == "grad_cam_pp":
                    hm = grad_cam_pp_heatmap(net, feat_store, grad_store, img_t, qidx,
                                             text_anchors[cat], H, W)
                else:  # rise
                    hm = rise_heatmap(net, img_t, qidx, text_anchors[cat], H, W,
                                      n_masks=args.rise_masks)
                if peak_in_bbox(hm, bbox):
                    hits[m] += 1
        pil.close()
        if (ii + 1) % 50 == 0:
            msg = "  ".join(f"{m}={hits[m]/max(1,n_lesions):.3f}" for m in args.methods)
            print(f"[{ii+1}/{len(img_ids)}] lesions={n_lesions}  hit-rate: {msg}", flush=True)

    summary = {
        "n_lesions": n_lesions,
        "pointing_hit_rate": {m: hits[m] / max(1, n_lesions) for m in args.methods},
        "chance_bbox_area_frac": float(np.mean(bbox_area_frac)) if bbox_area_frac else None,
        "note": "higher hit-rate = more faithful localization; "
                "chance ≈ mean bbox area fraction (random method should match it)",
        "joint_ckpt": args.joint_ckpt,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n=== pointing game (higher hit-rate = more faithful) ===")
    for m in args.methods:
        print(f"  {m:10s}: {hits[m]/max(1,n_lesions):.3f}")
    print(f"  (chance ≈ {summary['chance_bbox_area_frac']:.3f} = mean bbox area frac)")
    print(f"[save] -> {args.output}")


if __name__ == "__main__":
    main()
