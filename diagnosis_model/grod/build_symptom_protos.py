"""Per-symptom IMAGE prototypes for Option-C classification (modality-gap safe).

Text anchors can't be compared against the image-trained dense head (SigLIP2
modality gap). So per-symptom maps use IMAGE-space prototypes: average the RAW
SigLIP2 image embedding of GT lesion CROPS, grouped by symptom. Each GT box's
symptom label is GROD's own classification: argmax cos(centered lesion_emb,
centered abnormal text anchor) — lesion_embs (GROD z) live in anchor space, so
this is a clean self-labeling, no external COCO join (the merged_semantic COCO
uses a different file naming and doesn't align with the current case_db).

  proto[s] = mean_{GT box labeled s} SigLIP2_image(crop)        (L2-normed)

Run from repo root (SDM env):
  $PY -m diagnosis_model.grod.build_symptom_protos
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from diagnosis_model.grod.train_dense_head import load_siglip, crop_embs


def main():
    ap = argparse.ArgumentParser()
    ART = "data/processed/current/artifacts"
    ap.add_argument("--anchors", default=f"{ART}/models/text_anchors.pt")
    ap.add_argument("--symptoms", default="data/processed/current/symptoms.json")
    ap.add_argument("--case_db_dir", default=f"{ART}/db/case_db_jointDistRawP")
    ap.add_argument("--img_root", default="data/processed/current/detection")
    ap.add_argument("--siglip", default="google/siglip2-base-patch16-224")
    ap.add_argument("--out", default="outputs/grod/dense_head/symptom_protos.pt")
    ap.add_argument("--min_count", type=int, default=50, help="drop symptoms with fewer GT crops")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    A = torch.load(args.anchors, weights_only=False)["anchor_embs"].float()
    mu = A.mean(0, keepdim=True); A_c = F.normalize(A - mu, dim=-1)
    lm = json.load(open(args.symptoms))["label_map"]
    names = [lm[str(i)]["zh"] for i in range(len(lm))]

    sig, proc = load_siglip(args.siglip, device)
    cases = torch.load(Path(args.case_db_dir) / "train_cases.pt", weights_only=False)

    sums = {}; cnts = {}
    for c in cases:
        gt = c["lesion_boxes_xywh"]; le = c["lesion_embs"]
        if len(gt) == 0:
            continue
        p = Path(args.img_root) / "train" / c["file_name"]
        if not p.exists():
            continue
        image = Image.open(p).convert("RGB")
        gt = torch.as_tensor(gt, dtype=torch.float32)
        xyxy = torch.stack([gt[:, 0], gt[:, 1], gt[:, 0] + gt[:, 2], gt[:, 1] + gt[:, 3]], 1)
        emb = crop_embs(sig, proc, image, xyxy, device)              # [n,768] image space
        lec = F.normalize(le.float() - mu, dim=-1)
        lab = (lec @ A_c.t())[:, 1:].argmax(1) + 1                   # symptom id per box (abnormal)
        for i, s in enumerate(lab.tolist()):
            sums[s] = sums.get(s, 0) + emb[i]
            cnts[s] = cnts.get(s, 0) + 1

    ids, protos, kept_names, kept_cnts = [], [], [], []
    for s in sorted(sums):
        if cnts[s] < args.min_count:
            continue
        ids.append(s); protos.append(F.normalize(sums[s] / cnts[s], dim=0))
        kept_names.append(names[s]); kept_cnts.append(cnts[s])
    protos = torch.stack(protos)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ids": ids, "protos": protos, "names": kept_names}, args.out)
    print(f"[save] {args.out}  kept {len(ids)} symptoms (min_count={args.min_count}):")
    for s, nm, ct in zip(ids, kept_names, kept_cnts):
        print(f"  {s:2} {nm:14} n={ct}")


if __name__ == "__main__":
    main()
