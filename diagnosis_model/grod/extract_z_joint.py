"""Joint gate — extract the trained semantic z (pred_semantic) per case_db lesion.

Unlike the frozen probe (extract_hs.py -> train_semantic_head.py -> rebuild),
the joint model's semantic_embed is ALREADY trained, so out["pred_semantic"] is
the final 768-d z. This script runs the joint RF-DETR once per case image, reads
pred_semantic + pred_boxes, IoU-matches each GT lesion box to a query, and writes
the matched z aligned 1:1 to the case_db split — ready for rebuild_case_db.py
(which simply swaps lesion_embs). No separate head training step.

z is L2-normalized to match case_db lesion_emb convention.

Run from repo root:
  PY=/home/lab603/anaconda3/envs/SDM/bin/python
  $PY -m diagnosis_model.grod.extract_z_joint \
      --case_db_dir diagnosis_model/cause_inference/outputs/case_db_raw \
      --joint_ckpt diagnosis_model/grod/outputs/joint_rfdetr/checkpoint_best_regular.pth \
      --anchors diagnosis_model/grod/outputs/text_anchors.pt \
      --image_root data/detection/coco/_merged \
      --output_dir diagnosis_model/grod/outputs/z_joint \
      --splits train valid
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from diagnosis_model.grod.extract_hs import (
    xywh_to_xyxy, iou_matrix, greedy_match, cxcywh_norm_to_xyxy_abs,
)


@torch.no_grad()
def joint_forward(net, image, device, means, stds, resolution):
    """Return (z[Q,768] L2-normalized, pred_boxes_cxcywh_norm[Q,4])."""
    img_t = TF.to_tensor(image)
    img_t = TF.resize(img_t, [resolution, resolution])
    img_t = TF.normalize(img_t, means, stds).unsqueeze(0).to(device)
    out = net(img_t)
    z = F.normalize(out["pred_semantic"][0].float(), dim=-1).cpu()   # [Q, 768]
    boxes = out["pred_boxes"][0].detach().cpu()                       # [Q, 4]
    return z, boxes


def main():
    ap = argparse.ArgumentParser(description="Extract trained semantic z per case_db lesion (joint model).")
    ap.add_argument("--case_db_dir", type=str, required=True)
    ap.add_argument("--joint_ckpt", type=str, required=True)
    ap.add_argument("--anchors", type=str, required=True,
                    help="text_anchors.pt — needed only to enable the semantic head on load")
    ap.add_argument("--semantic_layers", type=int, default=1,
                    help="must match the value the joint ckpt was trained with "
                         "(the head's input is mean-pooled over the last K decoder layers)")
    ap.add_argument("--image_root", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--splits", type=str, nargs="+", default=["train", "valid"])
    ap.add_argument("--iou_thresh", type=float, default=0.5)
    ap.add_argument("--max_cases", type=int, default=-1)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # Enable the semantic head when RFDETRMedium rebuilds the model from the ckpt.
    os.environ["RFDETR_SEMANTIC_DIM"] = "768"
    os.environ["RFDETR_SEMANTIC_LAYERS"] = str(args.semantic_layers)
    os.environ["RFDETR_SEMANTIC_ANCHORS"] = os.path.abspath(args.anchors)

    from diagnosis_model.grod.build import load_oavle

    net, resolution, means, stds = load_oavle(args.joint_ckpt, device=args.device)
    device = next(net.parameters()).device
    assert hasattr(net, "semantic_embed"), "joint ckpt did not load a semantic head"
    print(f"[joint] device={device} resolution={resolution} "
          f"semantic_dim={net.semantic_embed.out_features}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_root = Path(args.image_root)

    for split in args.splits:
        cases = torch.load(Path(args.case_db_dir) / f"{split}_cases.pt", weights_only=False)
        if args.max_cases > 0:
            cases = cases[: args.max_cases]
        print(f"[{split}] {len(cases)} cases")

        out_cases: List[Dict] = []
        n_gt = n_matched = 0
        iou_vals: List[float] = []
        t0 = time.time()

        for ci, c in enumerate(cases):
            image = Image.open(image_root / split / c["file_name"]).convert("RGB")
            W, H = image.size
            z, boxes = joint_forward(net, image, device, means, stds, resolution)
            pred_xyxy = cxcywh_norm_to_xyxy_abs(boxes, W, H)

            gt = c["lesion_boxes_xywh"]
            if torch.is_tensor(gt):
                gt = gt.tolist()
            gt_xyxy = torch.tensor(
                [xywh_to_xyxy(tuple(b)) for b in gt], dtype=torch.float32,
            ) if gt else torch.zeros(0, 4)

            iou = iou_matrix(gt_xyxy, pred_xyxy)
            assign = greedy_match(iou, args.iou_thresh)

            D = z.size(-1)
            matched_z, kept_idx, matched_iou = [], [], []
            for g, p in enumerate(assign):
                n_gt += 1
                if p < 0:
                    continue
                n_matched += 1
                iou_vals.append(iou[g, p].item())
                matched_z.append(z[p])
                kept_idx.append(g)
                matched_iou.append(iou[g, p].item())

            out_cases.append({
                "case_id": c["case_id"],
                "image_id": c["image_id"],
                "file_name": c["file_name"],
                "split": split,
                # keep schema parallel to extract_hs.py so rebuild_case_db works:
                # 'hs' field actually holds the final z (already 768-d, normalized).
                "kept_lesion_idx": torch.tensor(kept_idx, dtype=torch.long),
                "z": torch.stack(matched_z) if matched_z else torch.zeros(0, D),
                "match_iou": torch.tensor(matched_iou, dtype=torch.float32),
                "n_gt_lesions": len(assign),
            })
            image.close()
            if ci % 1000 == 0:
                print(f"  [{ci}/{len(cases)}] match={n_matched/max(1,n_gt):.3f} "
                      f"t={time.time()-t0:.0f}s", flush=True)

        torch.save(out_cases, out_dir / f"z_{split}.pt")
        summary = {
            "split": split, "n_cases": len(out_cases),
            "n_gt_lesions": n_gt, "n_matched": n_matched,
            "match_rate": n_matched / max(1, n_gt),
            "mean_matched_iou": sum(iou_vals) / max(1, len(iou_vals)),
            "joint_ckpt": args.joint_ckpt,
        }
        with (out_dir / f"summary_{split}.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[{split}] match_rate={summary['match_rate']:.3f} "
              f"mIoU={summary['mean_matched_iou']:.3f} -> z_{split}.pt")


if __name__ == "__main__":
    main()
