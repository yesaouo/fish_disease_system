"""Deletion faithfulness — a common metric to compare attribution methods.

The token-level lesion-mask drop in cause_inference/faithfulness_eval.py only
works for CEAH's per-token α. To put *pixel-space* methods (Grad-CAM, attention)
on the same footing as GROD's routing, we use the standard XAI **deletion**
protocol:

  1. each method produces a pixel-space importance heatmap for a lesion;
  2. progressively mask the top p% most-important pixels (p = 0..50);
  3. score the masked whole image with a FROZEN external judge — raw SigLIP2
     similarity to that lesion's symptom caption;
  4. a faithful heatmap makes the score collapse fast → lower deletion AUC.

The judge (raw SigLIP2 text↔image) is external to every method, so no method
gets a home-field advantage. This file currently implements:
  - grad_cam : Grad-CAM on RF-DETR's backbone feature map
  - uniform  : flat heatmap (no saliency) — a lower-bound control
Attention-based heatmaps (GROD cross-attn via MSDeformAttn sampling, rollout)
are a planned second stage.

Run from repo root:
  python -m diagnosis_model.grod.deletion_faithfulness \
      --joint_ckpt diagnosis_model/grod/outputs/joint_rfdetr/checkpoint_best_regular.pth \
      --anchors diagnosis_model/grod/outputs/text_anchors.pt \
      --coco data/coco/_merged/valid/_annotations.coco.json \
      --image_root data/detection/coco/_merged/valid \
      --symptoms data/raw/symptoms.json \
      --methods grad_cam uniform \
      --output diagnosis_model/grod/outputs/deletion_faithfulness.json \
      --max_images 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

VL_CLASSIFIER_DIR = Path(__file__).resolve().parents[1] / "vl_classifier"
if str(VL_CLASSIFIER_DIR) not in sys.path:
    sys.path.insert(0, str(VL_CLASSIFIER_DIR))
from common import load_flat_caption_bank, get_text_features, get_image_features  # noqa: E402


# ---------------------------------------------------------------------------
# Frozen SigLIP2 judge: image (masked) vs a lesion's symptom captions
# ---------------------------------------------------------------------------

class SigLIPJudge:
    def __init__(self, model_name: str, symptoms_path: str, device: str):
        from transformers import AutoModel, AutoProcessor
        self.device = device
        self.proc = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        # per-category mean text anchor (same construction as build_text_anchors)
        bank = load_flat_caption_bank(symptoms_path, langs=("en", "zh"), text_mode="captions")
        labels = [int(x) for x in bank.label_ids]
        n_cat = max(labels) + 1
        embs = []
        with torch.no_grad():
            for i in range(0, len(bank.texts), 256):
                ti = self.proc(text=bank.texts[i:i+256], return_tensors="pt",
                               padding="max_length", truncation=True, max_length=64)
                ti = {k: v.to(device) for k, v in ti.items()}
                f = get_text_features(self.model, ti["input_ids"], ti.get("attention_mask"))
                embs.append(F.normalize(f.float(), dim=-1).cpu())
        cap = torch.cat(embs)
        D = cap.size(-1)
        anc = torch.zeros(n_cat, D)
        cnt = torch.zeros(n_cat)
        for e, l in zip(cap, labels):
            anc[l] += e; cnt[l] += 1
        self.anchors = F.normalize(anc / cnt.clamp_min(1).unsqueeze(1), dim=-1).to(device)

    @torch.no_grad()
    def score(self, pil_img: Image.Image, category_id: int, bbox_xyxy=None) -> float:
        """cosine(lesion-crop embedding, that symptom's text anchor).

        Masking happens on the whole image (the method must localize the lesion),
        but the judge crops the lesion bbox so the symptom signal is strong enough
        to move — a whole-fish embedding dilutes a single local symptom to ~0.
        """
        img = pil_img
        if bbox_xyxy is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
            if x2 > x1 and y2 > y1:
                img = pil_img.crop((x1, y1, x2, y2))
        px = self.proc(images=[img], return_tensors="pt")["pixel_values"].to(self.device)
        f = F.normalize(get_image_features(self.model, px).float(), dim=-1)
        return float((f @ self.anchors[category_id]).item())


# ---------------------------------------------------------------------------
# Heatmap generators (pixel-space importance for one lesion bbox)
# ---------------------------------------------------------------------------

def heatmap_uniform(H: int, W: int, bbox_xyxy=None) -> np.ndarray:
    """Random importance over the whole image — a no-saliency lower bound (random
    deletion order). A faithful method must beat this by actually pointing at the
    lesion. Uses random noise rather than a constant so argsort deletes pixels in
    random order, not a top-left raster scan."""
    return np.random.rand(H, W).astype(np.float32)


def heatmap_grad_cam(net, feat_store, grad_store, img_t, query_idx, anchor_vec,
                     H: int, W: int) -> np.ndarray:
    """Grad-CAM on the backbone projector feature map.

    Target = the chosen query's semantic z · its symptom anchor (how strongly
    this region reads as that symptom). Backprop to the cached feature map;
    Grad-CAM = ReLU(sum_c grad_c * act_c), upsampled to image size.
    """
    feat_store.clear(); grad_store.clear()
    net.zero_grad(set_to_none=True)
    out = net(img_t)
    z = out["pred_semantic"][0, query_idx]                    # [D]
    target = (F.normalize(z, dim=-1) * anchor_vec).sum()
    target.backward()

    act = feat_store["f"]            # [1, C, h, w]
    grad = grad_store["g"]           # [1, C, h, w]
    weights = grad.mean(dim=(2, 3), keepdim=True)             # GAP over spatial
    cam = F.relu((weights * act).sum(dim=1))[0]               # [h, w]
    cam = cam / (cam.max() + 1e-9)
    cam = F.interpolate(cam[None, None], size=(H, W), mode="bilinear",
                        align_corners=False)[0, 0]
    return cam.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Deletion: progressively mask top-p% pixels, score with the judge
# ---------------------------------------------------------------------------

def deletion_curve(pil_img: Image.Image, heatmap: np.ndarray, bbox_xyxy,
                   judge: SigLIPJudge, category_id: int,
                   fractions=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5)) -> List[float]:
    """Standard whole-image deletion: rank ALL pixels by importance, gray out the
    top-fraction, score the masked image each step. A faithful heatmap points at
    the lesion, so masking it drops the symptom score fast. `bbox_xyxy` unused
    (kept for signature compat with the per-lesion caller)."""
    arr = np.array(pil_img).copy()
    H, W = heatmap.shape
    order = np.argsort(-heatmap.flatten())  # high importance first, whole image
    ys, xs = np.unravel_index(order, heatmap.shape)
    mean_rgb = arr.reshape(-1, arr.shape[-1]).mean(0).astype(arr.dtype)

    n = len(order)
    scores = []
    for frac in fractions:
        work = arr.copy()
        k = int(frac * n)
        if k > 0:
            work[ys[:k], xs[:k]] = mean_rgb
        # mask whole image, but judge the lesion crop (strong local signal)
        scores.append(judge.score(Image.fromarray(work), category_id, bbox_xyxy))
    return scores


def deletion_auc(fractions, scores) -> float:
    """Normalized area under the (fraction, score) deletion curve. Lower = more
    faithful (masking important pixels drops the score fast)."""
    f = np.array(fractions)
    s = np.array(scores)
    return float(np.trapz(s, f) / (f[-1] - f[0] + 1e-9))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint_ckpt", type=str, required=True)
    ap.add_argument("--anchors", type=str, required=True)
    ap.add_argument("--coco", type=str, required=True,
                    help="vl_classifier COCO (has symptom category_id per box)")
    ap.add_argument("--image_root", type=str, required=True)
    ap.add_argument("--symptoms", type=str, default="data/raw/symptoms.json")
    ap.add_argument("--model_name", type=str, default="google/siglip2-base-patch16-224")
    ap.add_argument("--methods", type=str, nargs="+",
                    default=["grad_cam", "uniform"],
                    choices=["grad_cam", "uniform"])
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--max_images", type=int, default=-1)
    ap.add_argument("--iou_thresh", type=float, default=0.5)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.environ["RFDETR_SEMANTIC_DIM"] = "768"
    os.environ["RFDETR_SEMANTIC_ANCHORS"] = os.path.abspath(args.anchors)
    from diagnosis_model.grod.build import load_oavle
    from diagnosis_model.grod.extract_hs import (
        iou_matrix, greedy_match, cxcywh_norm_to_xyxy_abs, xywh_to_xyxy,
    )

    net, resolution, means, stds = load_oavle(args.joint_ckpt, device=args.device)
    anchors_pack = torch.load(args.anchors, map_location=args.device, weights_only=False)
    text_anchors = anchors_pack["anchor_embs"].to(args.device)

    # Grad-CAM hooks on the backbone projector feature map
    feat_store: Dict[str, torch.Tensor] = {}
    grad_store: Dict[str, torch.Tensor] = {}
    proj = dict(net.named_modules())["backbone.0.projector"]
    def fwd_hook(m, i, o):
        # only active during the Grad-CAM forward (grad enabled); the detect-only
        # forward runs under no_grad and is skipped.
        if not torch.is_grad_enabled():
            return
        t = o[0] if isinstance(o, (list, tuple)) else o
        if not t.requires_grad:
            t.requires_grad_(True)
        feat_store["f"] = t
        t.register_hook(lambda g: grad_store.__setitem__("g", g))
    proj.register_forward_hook(fwd_hook)

    judge = SigLIPJudge(args.model_name, args.symptoms, args.device)

    coco = json.load(open(args.coco))
    id2fn = {im["id"]: im["file_name"] for im in coco["images"]}
    anns_by_img: Dict[int, List[dict]] = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    img_ids = list(anns_by_img.keys())
    if args.max_images > 0:
        img_ids = img_ids[: args.max_images]

    fractions = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
    aucs: Dict[str, List[float]] = {m: [] for m in args.methods}
    n_lesions = 0

    for ii, img_id in enumerate(img_ids):
        fn = id2fn[img_id]
        pil = Image.open(Path(args.image_root) / fn).convert("RGB")
        W, H = pil.size
        img_t = TF.normalize(TF.resize(TF.to_tensor(pil), [resolution, resolution]),
                             means, stds).unsqueeze(0).to(args.device)
        with torch.no_grad():
            out = net(img_t)
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
            bbox = [max(0, int(gt_xyxy[g][0])), max(0, int(gt_xyxy[g][1])),
                    min(W, int(gt_xyxy[g][2])), min(H, int(gt_xyxy[g][3]))]
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            n_lesions += 1
            for m in args.methods:
                if m == "uniform":
                    hm = heatmap_uniform(H, W, bbox)
                else:  # grad_cam
                    hm = heatmap_grad_cam(net, feat_store, grad_store, img_t, qidx,
                                          text_anchors[cat], H, W)
                scores = deletion_curve(pil, hm, bbox, judge, cat, fractions)
                aucs[m].append(deletion_auc(fractions, scores))
        pil.close()
        if (ii + 1) % 50 == 0:
            msg = "  ".join(f"{m}={np.mean(aucs[m]):.4f}" for m in args.methods if aucs[m])
            print(f"[{ii+1}/{len(img_ids)}] lesions={n_lesions}  AUC: {msg}", flush=True)

    summary = {
        "n_lesions": n_lesions,
        "fractions": list(fractions),
        "deletion_auc_mean": {m: float(np.mean(v)) if v else None for m, v in aucs.items()},
        "deletion_auc_std": {m: float(np.std(v)) if v else None for m, v in aucs.items()},
        "note": "lower AUC = more faithful (masking important pixels drops the "
                "SigLIP2 symptom score faster)",
        "joint_ckpt": args.joint_ckpt,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n=== deletion faithfulness (lower AUC = more faithful) ===")
    for m in args.methods:
        if aucs[m]:
            print(f"  {m:10s}: AUC={np.mean(aucs[m]):.4f} ± {np.std(aucs[m]):.4f}")
    print(f"[save] -> {args.output}")


if __name__ == "__main__":
    main()
