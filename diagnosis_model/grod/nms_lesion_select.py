"""Does NMS beat the constant lesion threshold? (the only remaining lever)

probe_lesion_classifier showed neither an image-adaptive τ nor a per-query
semantic classifier beats the constant objectness threshold — because the
queries that break the constant are DUPLICATE detections of the real lesion
(higher objectness AND higher z·anchor saliency than the matched lesion box).
So the residual precision error is a de-duplication problem, not a selection
problem. This tests it directly: NMS the queries (suppress lower-objectness box
when IoU>thr), THEN apply the constant threshold, THEN re-match to GT.

Val diseased cases only (fast, one forward each). Reports micro selection P/R/F1
with and without NMS, overall and on the objectness-overlap subset.

Run from repo root (SDM env):
    $PY -m diagnosis_model.grod.nms_lesion_select
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.ops import nms

from diagnosis_model.grod.extract_hs import (
    xywh_to_xyxy, iou_matrix, greedy_match, cxcywh_norm_to_xyxy_abs,
)


def micro(TP, FP, FN):
    P = TP / (TP + FP + 1e-9); R = TP / (TP + FN + 1e-9)
    return P, R, 2 * P * R / (P + R + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ART = "data/processed/current/artifacts"
    ap.add_argument("--joint_ckpt", default=f"{ART}/models/joint_rfdetr/checkpoint_best_regular.pth")
    ap.add_argument("--global_sd", default=f"{ART}/models/distilled_global_rawP/global_embed_state_dict.pt")
    ap.add_argument("--anchors", default=f"{ART}/models/text_anchors.pt")
    ap.add_argument("--case_db_dir", default=f"{ART}/db/case_db_jointDistRawP")
    ap.add_argument("--img_root", default="data/processed/current/detection")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--iou_gt", type=float, default=0.5, help="GT-match IoU")
    ap.add_argument("--iou_nms", type=float, default=0.5, help="NMS suppression IoU")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    os.environ["RFDETR_SEMANTIC_DIM"] = "768"
    os.environ["RFDETR_SEMANTIC_ANCHORS"] = os.path.abspath(args.anchors)
    os.environ["RFDETR_GLOBAL_DIM"] = "768"
    from rfdetr import RFDETRMedium
    import torch.nn.functional as F
    rf = RFDETRMedium(pretrain_weights=args.joint_ckpt, num_classes=1)
    net = rf.model.model.to(dev).eval()
    res = int(rf.model.resolution); means, stds = list(rf.means), list(rf.stds)
    A = torch.load(args.anchors, weights_only=False)["anchor_embs"].float().to(dev)
    mu = A.mean(0, keepdim=True); A_c = F.normalize(A - mu, dim=-1)
    tp_sal, fp_sal = [], []          # centered saliency of correctly-matched vs false selected boxes

    cases = torch.load(Path(args.case_db_dir) / "valid_cases.pt", weights_only=False)
    agg = {"base": [0, 0, 0], "nms": [0, 0, 0]}
    agg_ov = {"base": [0, 0, 0], "nms": [0, 0, 0]}

    @torch.no_grad()
    def fwd(image):
        px = TF.normalize(TF.resize(TF.to_tensor(image), [res, res]), means, stds).unsqueeze(0).to(dev)
        out = net(px)
        obj = out["pred_logits"][0, :, 0].sigmoid().cpu()
        boxes = out["pred_boxes"][0].cpu()
        z = F.normalize(out["pred_semantic"][0].float(), dim=-1)
        sal = torch.einsum("qd,cd->qc", F.normalize(z - mu, dim=-1), A_c).amax(-1).clamp(0, 1).cpu()
        return obj, boxes, sal

    def score(keep_idx, pred_xyxy, gt_xyxy):
        """keep_idx: selected query indices. Match to GT, return (TP,FP,FN)."""
        if len(keep_idx) == 0:
            return 0, 0, gt_xyxy.size(0)
        sub = pred_xyxy[keep_idx]
        iou = iou_matrix(gt_xyxy, sub)
        assign = greedy_match(iou, args.iou_gt)              # GT -> local idx in sub
        matched = sum(1 for p in assign if p >= 0)
        TP = matched; FN = gt_xyxy.size(0) - matched; FP = len(keep_idx) - matched
        return TP, FP, FN

    for c in cases:
        gt = c["lesion_boxes_xywh"]; gt = gt.tolist() if torch.is_tensor(gt) else gt
        if not gt:
            continue
        image = Image.open(Path(args.img_root) / "valid" / c["file_name"]).convert("RGB")
        W, H = image.size
        obj, boxes, sal = fwd(image); image.close()
        pred_xyxy = cxcywh_norm_to_xyxy_abs(boxes, W, H)
        gt_xyxy = torch.tensor([xywh_to_xyxy(tuple(b)) for b in gt], dtype=torch.float32)

        # saliency of TP vs FP among base-selected boxes (correct GT->selected matching)
        bk = torch.where(obj >= args.tau)[0]
        if bk.numel():
            assign = greedy_match(iou_matrix(gt_xyxy, pred_xyxy[bk]), args.iou_gt)
            matched_local = {p for p in assign if p >= 0}
            for li, qi in enumerate(bk.tolist()):
                (tp_sal if li in matched_local else fp_sal).append(float(sal[qi]))

        # overlap flag: does a non-top box out-fire the weakest GT-matched lesion?
        base_keep = torch.where(obj >= args.tau)[0]
        # base selection
        tb = score(base_keep, pred_xyxy, gt_xyxy)
        # nms then threshold
        if base_keep.numel() > 0:
            k = nms(pred_xyxy[base_keep], obj[base_keep], args.iou_nms)
            nms_keep = base_keep[k]
        else:
            nms_keep = base_keep
        tn = score(nms_keep, pred_xyxy, gt_xyxy)
        for a, t in [("base", tb), ("nms", tn)]:
            for i in range(3): agg[a][i] += t[i]
        # overlap subset: images where NMS actually removed a box that base kept
        if nms_keep.numel() < base_keep.numel():
            for a, t in [("base", tb), ("nms", tn)]:
                for i in range(3): agg_ov[a][i] += t[i]

    print(f"=== lesion selection: constant τ={args.tau}, NMS IoU={args.iou_nms} (val diseased) ===")
    for a in ("base", "nms"):
        P, R, Fa = micro(*agg[a])
        print(f"  {a:<6} F1={Fa:.3f}  P={P:.3f}  R={R:.3f}  (TP={agg[a][0]} FP={agg[a][1]} FN={agg[a][2]})")
    print("  --- subset: images where NMS removed >=1 box ---")
    for a in ("base", "nms"):
        P, R, Fa = micro(*agg_ov[a])
        print(f"  {a:<6} F1={Fa:.3f}  P={P:.3f}  R={R:.3f}")
    tps, fps = np.array(tp_sal), np.array(fp_sal)
    d = (tps.mean() - fps.mean()) / np.sqrt((tps.var() + fps.var()) / 2 + 1e-9)
    print(f"\n  === can semantics filter the genuine FP? (correct matching) ===")
    print(f"  TP selected (n={len(tps)}): saliency mean={tps.mean():.3f}")
    print(f"  FP selected (n={len(fps)}): saliency mean={fps.mean():.3f}")
    print(f"  separation Cohen's d = {d:.2f}  (>~0.5 → a semantic FP-filter could lift precision; ≈0 → no lever)")


if __name__ == "__main__":
    main()
