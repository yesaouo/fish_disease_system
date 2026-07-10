"""Extract ALL decoder queries' hs (aligned to soft_inputs order) for the L2
end-to-end finetune probe.

Rationale: the L1 finetune (``finetune_e2e_grace``) treats objectness logits as
fixed cached inputs — gradient reaches only the Region Gate + aggregator. L2
lets gradient reach the GROD **semantic head** and **objectness (class_embed)
head** as well, by running them LIVE on the decoder query features ``hs`` while
backbone/decoder stay frozen (so bbox is untouched — see BUILD_PIPELINE L2 note).

This script caches the full ``hs`` [Q, Hd] per case (every query kept, matching
the soft pipeline) plus the IoU-matched (query_idx, symptom_category_id) pairs
for the optional L_sym grounding term. Case iteration order mirrors
``extract_soft_inputs.py`` (same ``{split}_cases.pt``), so ``hs_all[i]`` aligns
1:1 with ``soft_inputs`` ``z_all[i]`` / ``w[i]``.

Run:
  $PY -m diagnosis_model.grod.extract_hs_all \
      --case_db_dir  $ART/db/case_db_jointDistRawP \
      --split train \
      --det_ckpt     $ART/models/joint_rfdetr/checkpoint_best_regular.pth \
      --anchors      $ART/models/text_anchors.pt \
      --vlc_coco     data/processed/current/full/train/_annotations.coco.json \
      --image_root   data/processed/current/detection/train \
      --output_dir   $ART/db/hs_all
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image

from diagnosis_model.grod.extract_hs import (
    load_detector, find_class_embed, detector_forward, build_category_lookup,
    iou_matrix, greedy_match, xywh_to_xyxy, cxcywh_norm_to_xyxy_abs,
)


def main():
    ap = argparse.ArgumentParser(
        description="Cache full-Q hs per case (aligned to soft_inputs) for L2 finetune.")
    ap.add_argument("--case_db_dir", type=str, required=True)
    ap.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"])
    ap.add_argument("--det_ckpt", type=str, required=True,
                    help="joint_rfdetr checkpoint_best_regular.pth (same decoder the semantic head trained on)")
    ap.add_argument("--anchors", type=str, required=True,
                    help="text_anchors.pt — sets the semantic env so the joint model builds cleanly")
    ap.add_argument("--vlc_coco", type=str, required=True,
                    help="vl_classifier COCO (has symptom category_id) for the same split")
    ap.add_argument("--image_root", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--iou_thresh", type=float, default=0.5)
    ap.add_argument("--max_cases", type=int, default=-1)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # build the joint model cleanly (semantic + global heads present in ckpt)
    os.environ["RFDETR_SEMANTIC_DIM"] = "768"
    os.environ["RFDETR_SEMANTIC_ANCHORS"] = os.path.abspath(args.anchors)
    os.environ["RFDETR_GLOBAL_DIM"] = "768"

    case_db_dir = Path(args.case_db_dir)
    cases = torch.load(case_db_dir / f"{args.split}_cases.pt", weights_only=False)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    print(f"[load] {args.split}_cases.pt  n={len(cases)}")

    cat_lookup = build_category_lookup(Path(args.vlc_coco))
    print(f"[load] vlc category lookup: {len(cat_lookup)} keys")

    rf, net, device, means, stds, resolution = load_detector(args.det_ckpt, args.device)
    class_embed = find_class_embed(net)
    Hd = int(class_embed.in_features)
    print(f"[det] device={device} resolution={resolution} hidden_dim={Hd}")

    image_root = Path(args.image_root)
    out_cases: List[Dict] = []
    n_gt = n_matched = n_cat_found = 0
    t0 = time.time()

    for ci, c in enumerate(cases):
        fn = c["file_name"]
        image = Image.open(image_root / fn).convert("RGB")
        W, H = image.size
        hs, pred_boxes = detector_forward(
            net, class_embed, image, device, means, stds, resolution)  # hs[Q,Hd], boxes[Q,4]
        pred_xyxy = cxcywh_norm_to_xyxy_abs(pred_boxes, W, H)

        gt_boxes_xywh = c["lesion_boxes_xywh"]
        if torch.is_tensor(gt_boxes_xywh):
            gt_boxes_xywh = gt_boxes_xywh.tolist()
        gt_xyxy = torch.tensor(
            [xywh_to_xyxy(tuple(b)) for b in gt_boxes_xywh], dtype=torch.float32,
        ) if gt_boxes_xywh else torch.zeros(0, 4)
        iou = iou_matrix(gt_xyxy, pred_xyxy)
        assign = greedy_match(iou, args.iou_thresh)  # len G, query idx or -1

        matched_qidx: List[int] = []
        matched_cat: List[int] = []
        for g, p in enumerate(assign):
            n_gt += 1
            if p < 0:
                continue
            n_matched += 1
            matched_qidx.append(int(p))
            key = (fn, tuple(int(round(x)) for x in gt_boxes_xywh[g]))
            cat = cat_lookup.get(key, -1)
            if cat >= 0:
                n_cat_found += 1
            matched_cat.append(cat)

        out_cases.append({
            "case_id": c["case_id"],
            "file_name": fn,
            "hs_all": hs.to(torch.bfloat16),                          # [Q, Hd]
            "matched_qidx": torch.tensor(matched_qidx, dtype=torch.long),
            "matched_cat": torch.tensor(matched_cat, dtype=torch.long),
        })
        image.close()
        if ci % 200 == 0:
            print(f"  [{ci}/{len(cases)}] match_rate={n_matched/max(1,n_gt):.3f} "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(out_cases, out_dir / f"hs_all_{args.split}.pt")
    summary = {
        "split": args.split, "n_cases": len(out_cases), "hidden_dim": Hd,
        "n_gt_lesions": n_gt, "n_matched": n_matched,
        "match_rate": n_matched / max(1, n_gt),
        "category_join_rate": n_cat_found / max(1, n_matched),
        "det_ckpt": args.det_ckpt,
    }
    with (out_dir / f"summary_hs_all_{args.split}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"[save] hs_all_{args.split}.pt -> {out_dir}")


if __name__ == "__main__":
    main()
