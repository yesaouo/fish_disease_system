"""Eval — does GROD identify the GT lesion's *symptom type*? (the missing number)

Everything else in the soft pipeline is measured at the disease/healthy (abstain)
level, the box-localization level (detection mAP, extract_z_joint match_rate,
pointing_game), or the whole-case cause-retrieval level (sem R@10). None of it
asks the per-lesion classification question: for a GT lesion box, does GROD's
semantic z classify to the *correct* symptom category?

Protocol (one GROD forward per image):
  z[Q,768], pred_boxes  = joint model
  IoU≥thr match each GT box → query              (greedy, same as extract_z_joint)
  pred_cat = argmax_c cos(z_matched, anchor_c)   over lesion cats (exclude 0=healthy)
  compare to the box's GT symptom_category_id

Reports two denominators so localization and classification are not conflated:
  - acc_matched : correct / matched lesions     (classification quality given a hit)
  - acc_all     : correct / all GT lesions       (end-to-end; unlocalized = miss)
plus top-3, macro-F1, and per-class precision/recall.

Run from repo root:
  $PY -m diagnosis_model.grod.eval_lesion_symptom_cls
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from diagnosis_model.grod.extract_hs import (
    xywh_to_xyxy, iou_matrix, greedy_match, cxcywh_norm_to_xyxy_abs,
)


@torch.no_grad()
def joint_forward(net, image, device, means, stds, res):
    img = TF.normalize(TF.resize(TF.to_tensor(image), [res, res]), means, stds).unsqueeze(0).to(device)
    out = net(img)
    z = F.normalize(out["pred_semantic"][0].float(), dim=-1).cpu()   # [Q,768]
    return z, out["pred_boxes"][0].detach().cpu()                    # [Q,4] cxcywh norm


def main():
    ap = argparse.ArgumentParser()
    ART = "data/processed/current/artifacts"; DET = "data/processed/current/detection"
    ap.add_argument("--joint_ckpt", default=f"{ART}/models/joint_rfdetr/checkpoint_best_regular.pth")
    ap.add_argument("--anchors", default=f"{ART}/models/text_anchors.pt")
    ap.add_argument("--coco", default=f"{DET}/valid/_annotations.coco.json")
    ap.add_argument("--image_root", default=f"{DET}/valid")
    ap.add_argument("--symptoms", default="data/processed/current/symptoms.json")
    ap.add_argument("--iou_thresh", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.environ["RFDETR_SEMANTIC_DIM"] = "768"
    os.environ["RFDETR_SEMANTIC_ANCHORS"] = os.path.abspath(args.anchors)
    from rfdetr import RFDETRMedium
    rf = RFDETRMedium(pretrain_weights=args.joint_ckpt, num_classes=1)
    net = rf.model.model.to(args.device).eval()
    means, stds, res = list(rf.means), list(rf.stds), int(rf.model.resolution)

    anc = torch.load(args.anchors, weights_only=False)
    A = F.normalize(anc["anchor_embs"].float(), dim=-1)              # [C,768]
    C = A.size(0)
    cand = [c for c in range(1, C)]                                  # exclude 0=healthy_region
    cand_t = torch.tensor(cand)
    label_map = json.load(open(args.symptoms))["label_map"]
    name = {int(k): v["zh"] for k, v in label_map.items()}

    coco = json.load(open(args.coco))
    id2fn = {im["id"]: im["file_name"] for im in coco["images"]}
    by_img = defaultdict(list)                                       # image_id -> [(bbox_xywh, sym_cat)]
    for a in coco["annotations"]:
        by_img[a["image_id"]].append((a["bbox"], a["symptom_category_id"]))

    n_gt = n_matched = top1 = top3 = 0
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)  # per-class, over matched
    for iid, items in by_img.items():
        fn_img = id2fn.get(iid)
        if fn_img is None:
            continue
        image = Image.open(Path(args.image_root) / fn_img).convert("RGB")
        W, H = image.size
        z, boxes = joint_forward(net, image, args.device, means, stds, res)
        pred_xyxy = cxcywh_norm_to_xyxy_abs(boxes, W, H)
        gt_xyxy = torch.tensor([xywh_to_xyxy(tuple(b)) for b, _ in items], dtype=torch.float32)
        gt_cat = [c for _, c in items]
        assign = greedy_match(iou_matrix(gt_xyxy, pred_xyxy), args.iou_thresh)
        image.close()

        for g, p in enumerate(assign):
            n_gt += 1
            true_c = gt_cat[g]
            if p < 0:                                               # GT lesion not localized
                fn[true_c] += 1
                continue
            n_matched += 1
            sims = z[p] @ A[cand_t].t()                             # [len(cand)]
            order = sims.argsort(descending=True)
            ranked = [cand[i] for i in order.tolist()]
            pred_c = ranked[0]
            if pred_c == true_c:
                top1 += 1; tp[true_c] += 1
            else:
                fp[pred_c] += 1; fn[true_c] += 1
            if true_c in ranked[:3]:
                top3 += 1

    print(f"\n=== GROD lesion symptom classification (valid, IoU≥{args.iou_thresh}) ===")
    print(f"GT lesions = {n_gt} | localized (matched) = {n_matched} "
          f"(match_rate {n_matched/max(1,n_gt):.3f})")
    print(f"top-1 acc | given match = {top1/max(1,n_matched):.4f}   "
          f"end-to-end (all GT) = {top1/max(1,n_gt):.4f}")
    print(f"top-3 acc | given match = {top3/max(1,n_matched):.4f}")

    f1s = []
    print(f"\n{'cat':>3} {'symptom(zh)':<14} {'GT':>5} {'prec':>6} {'rec':>6} {'f1':>6}")
    for c in cand:
        support = tp[c] + fn[c]
        if support == 0 and fp[c] == 0:
            continue
        prec = tp[c] / max(1, tp[c] + fp[c]); rec = tp[c] / max(1, tp[c] + fn[c])
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        if support > 0:
            f1s.append(f1)
        print(f"{c:>3} {name.get(c,'?')[:13]:<14} {support:>5} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f}")
    print(f"\nmacro-F1 (over present lesion classes) = {sum(f1s)/max(1,len(f1s)):.4f}")


if __name__ == "__main__":
    main()
