"""Region faithfulness — GROD's own faithfulness, independent of CEAH.

The token-level lesion-mask drop in cause_inference/faithfulness_eval.py runs
*through CEAH*, so it measures the faithfulness of the cause-attribution head,
not of GROD's region feature itself. This eval removes that coupling: it asks,
for each routing, whether masking the lesion pixels collapses the region
feature's agreement with the lesion's symptom caption.

Protocol (identical mask + identical judge for every routing — only the routing
of the region feature changes):

  baseline = cos( region_feature(image),        symptom_anchor[cat] )
  masked   = cos( region_feature(image w/ lesion bbox grayed), symptom_anchor[cat] )
  drop     = baseline - masked            # larger positive = more faithful

Routings:
  - grod        : RF-DETR decoder query z (semantic head)          [global context]
  - crop        : raw SigLIP2 on the isolated lesion crop          [non-DETR/RoI proxy]
  - fused       : VLM-Lesion LocalGlobalFusionWrapper on the crop  [crop + fusion bolt-on]

The symptom anchor is the same frozen SigLIP2 text bank for grod/crop; for
`fused` the caption is encoded by the fused model's own (frozen) text tower for
a fair within-model cosine.

Run from repo root:
  python -m diagnosis_model.grod.region_faithfulness \
      --joint_ckpt diagnosis_model/grod/outputs/joint_rfdetr/checkpoint_best_regular.pth \
      --anchors diagnosis_model/grod/outputs/text_anchors.pt \
      --coco data/coco/_merged/valid/_annotations.coco.json \
      --image_root data/detection/coco/_merged/valid \
      --symptoms data/raw/symptoms.json \
      --fused_vlm diagnosis_model/vl_classifier/outputs/siglip2_base_patch16_224_multipos_fusion_en_zh \
      --routings grod crop fused \
      --output diagnosis_model/grod/outputs/region_faithfulness.json \
      --max_images 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

VL_DIR = Path(__file__).resolve().parents[1] / "vl_classifier"
if str(VL_DIR) not in sys.path:
    sys.path.insert(0, str(VL_DIR))
from common import (  # noqa: E402
    load_flat_caption_bank, get_text_features, get_image_features,
)


# ---------------------------------------------------------------------------
# Masking: gray out the GT lesion bbox (same definition for every routing)
# ---------------------------------------------------------------------------

def gray_bbox(pil: Image.Image, bbox_xyxy) -> Image.Image:
    arr = np.array(pil).copy()
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    if x2 > x1 and y2 > y1:
        arr[y1:y2, x1:x2] = arr.reshape(-1, arr.shape[-1]).mean(0).astype(arr.dtype)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Per-category symptom text anchors for a given (frozen) model
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_anchors(model, proc, symptoms_path, device) -> torch.Tensor:
    bank = load_flat_caption_bank(symptoms_path, langs=("en", "zh"), text_mode="captions")
    labels = [int(x) for x in bank.label_ids]
    n_cat = max(labels) + 1
    embs = []
    for i in range(0, len(bank.texts), 256):
        ti = proc(text=bank.texts[i:i+256], return_tensors="pt",
                  padding="max_length", truncation=True, max_length=64)
        ti = {k: v.to(device) for k, v in ti.items()}
        f = get_text_features(model, ti["input_ids"], ti.get("attention_mask"))
        embs.append(F.normalize(f.float(), dim=-1).cpu())
    cap = torch.cat(embs); D = cap.size(-1)
    anc = torch.zeros(n_cat, D); cnt = torch.zeros(n_cat)
    for e, l in zip(cap, labels):
        anc[l] += e; cnt[l] += 1
    return F.normalize(anc / cnt.clamp_min(1).unsqueeze(1), dim=-1).to(device)


# ---------------------------------------------------------------------------
# Region feature per routing (returns cos(feature, anchor[cat]))
# ---------------------------------------------------------------------------

class GRODRouting:
    """RF-DETR decoder query z, IoU-matched to the lesion box."""
    def __init__(self, joint_ckpt, anchors_path, device):
        os.environ["RFDETR_SEMANTIC_DIM"] = "768"
        os.environ["RFDETR_SEMANTIC_ANCHORS"] = os.path.abspath(anchors_path)
        from rfdetr import RFDETRMedium
        self.rf = RFDETRMedium(pretrain_weights=joint_ckpt, num_classes=1)
        self.net = self.rf.model.model.to(device).eval()
        self.means, self.stds = list(self.rf.means), list(self.rf.stds)
        self.res = int(self.rf.model.resolution)
        self.device = device
        self.anchors = torch.load(anchors_path, map_location=device,
                                  weights_only=False)["anchor_embs"].to(device)
        from diagnosis_model.grod.extract_hs import (
            iou_matrix, greedy_match, cxcywh_norm_to_xyxy_abs, xywh_to_xyxy,
        )
        self._iou, self._match = iou_matrix, greedy_match
        self._cvt, self._xyxy = cxcywh_norm_to_xyxy_abs, xywh_to_xyxy

    @torch.no_grad()
    def _forward(self, pil):
        W, H = pil.size
        t = TF.normalize(TF.resize(TF.to_tensor(pil), [self.res, self.res]),
                         self.means, self.stds).unsqueeze(0).to(self.device)
        out = self.net(t)
        z = F.normalize(out["pred_semantic"][0].float(), dim=-1)
        return z, self._cvt(out["pred_boxes"][0].cpu(), W, H)

    @torch.no_grad()
    def sims(self, z_row) -> torch.Tensor:
        """cos of one query's z against all category anchors -> [n_cat]."""
        return (z_row @ self.anchors.t()).float().cpu()


class SigLIPCropRouting:
    """Raw SigLIP2 on the isolated lesion crop (non-DETR/RoI proxy)."""
    def __init__(self, model_name, symptoms_path, device):
        from transformers import AutoModel, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.anchors = build_anchors(self.model, self.proc, symptoms_path, device)

    @torch.no_grad()
    def sims(self, pil_crop) -> torch.Tensor:
        px = self.proc(images=[pil_crop], return_tensors="pt")["pixel_values"].to(self.device)
        f = F.normalize(get_image_features(self.model, px).float(), dim=-1)
        return (f @ self.anchors.t())[0].float().cpu()


class FusedRouting:
    """VLM-Lesion LocalGlobalFusionWrapper: crop + whole-image fusion."""
    def __init__(self, fused_path, symptoms_path, device):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                                / "cause_inference" / "preprocessing"))
        from build_case_database import load_vlm
        self.model, self.proc = load_vlm(fused_path, device=device, force_fusion=True)
        self.device = device
        self.anchors = build_anchors(self.model, self.proc, symptoms_path, device)

    @torch.no_grad()
    def sims(self, pil_crop, pil_global) -> torch.Tensor:
        lpx = self.proc(images=[pil_crop], return_tensors="pt")["pixel_values"].to(self.device)
        gpx = self.proc(images=[pil_global], return_tensors="pt")["pixel_values"].to(self.device)
        f = F.normalize(self.model.forward_image(lpx, gpx).float(), dim=-1)
        return (f @ self.anchors.t())[0].float().cpu()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint_ckpt", type=str, required=True)
    ap.add_argument("--anchors", type=str, required=True)
    ap.add_argument("--coco", type=str, required=True)
    ap.add_argument("--image_root", type=str, required=True)
    ap.add_argument("--symptoms", type=str, default="data/raw/symptoms.json")
    ap.add_argument("--model_name", type=str, default="google/siglip2-base-patch16-224")
    ap.add_argument("--fused_vlm", type=str, default=None)
    ap.add_argument("--routings", type=str, nargs="+",
                    default=["grod", "crop", "fused"],
                    choices=["grod", "crop", "fused"])
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--max_images", type=int, default=-1)
    ap.add_argument("--iou_thresh", type=float, default=0.5)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from diagnosis_model.grod.extract_hs import xywh_to_xyxy

    routers: Dict[str, object] = {}
    if "grod" in args.routings:
        routers["grod"] = GRODRouting(args.joint_ckpt, args.anchors, args.device)
    if "crop" in args.routings:
        routers["crop"] = SigLIPCropRouting(args.model_name, args.symptoms, args.device)
    if "fused" in args.routings:
        if not args.fused_vlm:
            raise ValueError("--fused_vlm required for routing 'fused'")
        routers["fused"] = FusedRouting(args.fused_vlm, args.symptoms, args.device)

    coco = json.load(open(args.coco))
    id2fn = {im["id"]: im["file_name"] for im in coco["images"]}
    anns_by_img: Dict[int, List[dict]] = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    img_ids = list(anns_by_img.keys())
    if args.max_images > 0:
        img_ids = img_ids[: args.max_images]

    drops: Dict[str, List[float]] = {r: [] for r in routers}
    base: Dict[str, List[float]] = {r: [] for r in routers}
    n_lesions = 0

    # GROD needs IoU matching to pick the query; do it once per image.
    grod_router: GRODRouting = routers.get("grod")  # type: ignore

    for ii, img_id in enumerate(img_ids):
        pil = Image.open(Path(args.image_root) / id2fn[img_id]).convert("RGB")
        W, H = pil.size
        gt = [(a["bbox"], int(a["category_id"])) for a in anns_by_img[img_id]
              if "bbox" in a and "category_id" in a]
        gt_xyxy = [xywh_to_xyxy(tuple(b)) for b, _ in gt]

        assign = None
        if grod_router is not None:
            import torch as _t
            gtt = _t.tensor(gt_xyxy, dtype=_t.float32) if gt_xyxy else _t.zeros(0, 4)
            _, pred = grod_router._forward(pil)
            assign = grod_router._match(grod_router._iou(gtt, pred), args.iou_thresh)
            _, pred_masked_cache = None, None  # masked forward done per-lesion below

        for g, (bbox_xywh, cat) in enumerate(gt):
            bx = [max(0, int(gt_xyxy[g][0])), max(0, int(gt_xyxy[g][1])),
                  min(W, int(gt_xyxy[g][2])), min(H, int(gt_xyxy[g][3]))]
            if bx[2] <= bx[0] or bx[3] <= bx[1]:
                continue
            masked_img = gray_bbox(pil, bx)
            counted = False

            def prob(sims_vec: torch.Tensor) -> float:
                # rank-aware: softmax over the 19 symptom anchors, take this cat's
                # probability. Cross-space comparable, unaffected by GROD z's low
                # absolute cosine scale.
                return float(F.softmax(sims_vec / 0.07, dim=-1)[cat].item())

            for r, router in routers.items():
                if r == "grod":
                    q = assign[g]
                    if q < 0:
                        continue
                    zb, _ = router._forward(pil)
                    sb = router.sims(zb[q])
                    zm, predm = router._forward(masked_img)
                    am = router._match(router._iou(
                        torch.tensor([gt_xyxy[g]], dtype=torch.float32), predm),
                        args.iou_thresh)
                    qm = q if am[0] < 0 else am[0]
                    sm = router.sims(zm[qm])
                elif r == "crop":
                    from diagnosis_model.cause_inference.preprocessing.build_case_database import scaled_rect_crop
                    sb = router.sims(scaled_rect_crop(pil, bbox_xywh))
                    sm = router.sims(scaled_rect_crop(masked_img, bbox_xywh))
                else:  # fused
                    from diagnosis_model.cause_inference.preprocessing.build_case_database import scaled_rect_crop
                    sb = router.sims(scaled_rect_crop(pil, bbox_xywh), pil)
                    sm = router.sims(scaled_rect_crop(masked_img, bbox_xywh), masked_img)
                pb, pm = prob(sb), prob(sm)
                base[r].append(pb)
                drops[r].append(pb - pm)
                counted = True
            if counted:
                n_lesions += 1
        pil.close()
        if (ii + 1) % 50 == 0:
            msg = "  ".join(f"{r}: drop={np.mean(drops[r]):+.4f}" for r in routers if drops[r])
            print(f"[{ii+1}/{len(img_ids)}] lesions={n_lesions}  {msg}", flush=True)

    summary = {
        "n_lesions": n_lesions,
        "baseline_prob_mean": {r: float(np.mean(base[r])) if base[r] else None for r in routers},
        "lesion_mask_prob_drop_mean": {r: float(np.mean(drops[r])) if drops[r] else None for r in routers},
        "lesion_mask_prob_drop_std": {r: float(np.std(drops[r])) if drops[r] else None for r in routers},
        "note": "drop = P(symptom | region) baseline - lesion-masked, where P is "
                "softmax over the 19 symptom anchors (rank-aware, cross-space "
                "comparable). Larger positive = more faithful. Independent of CEAH.",
        "joint_ckpt": args.joint_ckpt,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n=== region faithfulness (symptom-prob drop, larger+ = more faithful) ===")
    for r in routers:
        if drops[r]:
            print(f"  {r:6s}: prob_drop={np.mean(drops[r]):+.4f} ± {np.std(drops[r]):.4f} "
                  f"(baseline P={np.mean(base[r]):.4f})")
    print(f"[save] -> {args.output}")


if __name__ == "__main__":
    main()
