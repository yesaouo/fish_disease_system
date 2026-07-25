"""Per-source-dataset lesion-symptom confusion matrices.

Same protocol as eval_lesion_symptom_cls.py (one GROD/OAVLE forward per image,
IoU-match GT box -> query, pred_cat = argmax_c cos(z, anchor_c) over lesion cats),
but accumulates a *separate* confusion matrix for each of the four official
source datasets (Fish_disease_9911 / fish_disease / fish_disease_detection /
tilapia) and renders one row-normalized PNG per dataset (paper style).

Runs over all splits (train+valid+test) combined so small datasets (tilapia)
are not degenerate. Source is read from the COCO image record's `source_dataset`.

Run from repo root:
  $PY -m diagnosis_model.grod.confusion_by_dataset
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

OFFICIAL = ["Fish_disease_9911", "fish_disease", "fish_disease_detection", "tilapia"]


@torch.no_grad()
def joint_forward(net, image, device, means, stds, res):
    img = TF.normalize(TF.resize(TF.to_tensor(image), [res, res]), means, stds).unsqueeze(0).to(device)
    out = net(img)
    z = F.normalize(out["pred_semantic"][0].float(), dim=-1).cpu()
    return z, out["pred_boxes"][0].detach().cpu()


def render(confusion, ds, labels, title, out_png, font):
    """confusion: dict true_c -> pred_c -> count. Row-normalized % heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    present = sorted({c for c in confusion} | {p for t in confusion for p in confusion[t]})
    if not present:
        print(f"[skip] {name}: no matched lesions")
        return
    idx = {c: i for i, c in enumerate(present)}
    n = len(present)
    counts = [[0] * n for _ in range(n)]
    for t in confusion:
        for p, v in confusion[t].items():
            counts[idx[t]][idx[p]] += v
    support = [sum(row) for row in counts]
    vmax = max((c for row in counts for c in row), default=1)

    fig, ax = plt.subplots(figsize=(max(5.5, 0.62 * n + 2.2), max(5.0, 0.62 * n + 1.8)))
    im = ax.imshow(counts, cmap="Blues", vmin=0, vmax=max(1, vmax), aspect="equal")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    xlabels = [labels[c] for c in present]
    ylabels = [f"{labels[c]} (n={support[i]})" for i, c in enumerate(present)]
    ax.set_xticklabels(xlabels, fontproperties=font, fontsize=9, rotation=45, ha="right")
    ax.set_yticklabels(ylabels, fontproperties=font, fontsize=9)
    ax.set_xlabel("預測外觀表徵", fontproperties=font)
    ax.set_ylabel("真實外觀表徵", fontproperties=font)
    ax.set_title(title, fontproperties=font, fontsize=12)
    for i in range(n):
        for j in range(n):
            v = counts[i][j]
            if v > 0:
                ax.text(j, i, f"{v}", ha="center", va="center", fontsize=8,
                        color="white" if v >= 0.5 * vmax else "#333")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("病灶數量", fontproperties=font, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    total = sum(support)
    diag = sum(counts[i][i] for i in range(n))
    print(f"[dump] {ds}: {out_png}  (matched lesions={total}, top-1 acc={diag/max(1,total):.3f})")


def main():
    ap = argparse.ArgumentParser()
    ART = "data/processed/current/artifacts"; DET = "data/processed/current/detection"
    ap.add_argument("--joint_ckpt", default=f"{ART}/models/joint_rfdetr/checkpoint_best_regular.pth")
    ap.add_argument("--anchors", default=f"{ART}/models/text_anchors.pt")
    ap.add_argument("--det_root", default=DET)
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--symptoms", default="data/processed/current/symptoms.json")
    ap.add_argument("--iou_thresh", type=float, default=0.5)
    ap.add_argument("--out_dir", default="data/processed/current/eval/confusion_by_dataset")
    ap.add_argument("--font", default="paper/cwTeX Q Kai Medium.ttf")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    anc = torch.load(args.anchors, weights_only=False)
    os.environ["RFDETR_SEMANTIC_DIM"] = str(anc.get("dim", anc["anchor_embs"].shape[-1]))
    os.environ["RFDETR_SEMANTIC_ANCHORS"] = os.path.abspath(args.anchors)
    from diagnosis_model.grod.build import load_oavle
    net, res, means, stds = load_oavle(args.joint_ckpt, device=args.device)

    A = F.normalize(anc["anchor_embs"].float(), dim=-1)
    C = A.size(0)
    cand = [c for c in range(1, C)]
    cand_t = torch.tensor(cand)
    label_map = json.load(open(args.symptoms))["label_map"]
    name = {int(k): v["zh"] for k, v in label_map.items()}

    # per-dataset confusion (over matched lesions)
    conf = {d: defaultdict(lambda: defaultdict(int)) for d in OFFICIAL}
    n_gt = defaultdict(int); n_match = defaultdict(int)

    for split in args.splits:
        coco_path = Path(args.det_root) / split / "_annotations.coco.json"
        if not coco_path.exists():
            print(f"[warn] missing {coco_path}, skip"); continue
        coco = json.load(open(coco_path))
        src = {im["id"]: im.get("source_dataset", "?") for im in coco["images"]}
        id2fn = {im["id"]: im["file_name"] for im in coco["images"]}
        by_img = defaultdict(list)
        for a in coco["annotations"]:
            by_img[a["image_id"]].append((a["bbox"], a["symptom_category_id"]))
        img_root = Path(args.det_root) / split

        done = 0
        for iid, items in by_img.items():
            ds = src.get(iid)
            if ds not in conf:                                  # only the 4 official datasets
                continue
            fn_img = id2fn.get(iid)
            if fn_img is None:
                continue
            image = Image.open(img_root / fn_img).convert("RGB")
            W, H = image.size
            z, boxes = joint_forward(net, image, args.device, means, stds, res)
            pred_xyxy = cxcywh_norm_to_xyxy_abs(boxes, W, H)
            gt_xyxy = torch.tensor([xywh_to_xyxy(tuple(b)) for b, _ in items], dtype=torch.float32)
            gt_cat = [c for _, c in items]
            assign = greedy_match(iou_matrix(gt_xyxy, pred_xyxy), args.iou_thresh)
            image.close()
            for g, p in enumerate(assign):
                n_gt[ds] += 1
                if p < 0:
                    continue
                n_match[ds] += 1
                sims = z[p] @ A[cand_t].t()
                pred_c = cand[int(sims.argmax())]
                conf[ds][gt_cat[g]][pred_c] += 1
            done += 1
            if done % 500 == 0:
                print(f"  [{split}] {done} images...", flush=True)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    from matplotlib import font_manager as fm
    fm.fontManager.addfont(args.font)
    font = fm.FontProperties(fname=args.font)

    for d in OFFICIAL:
        mr = n_match[d] / max(1, n_gt[d])
        title = f"{d}（IoU≥{args.iou_thresh}，match {n_match[d]}/{n_gt[d]}={mr:.2f}）"
        render(conf[d], d, name, title, str(out_dir / f"confusion_{d}.png"), font)
        # also dump raw counts JSON for reproducibility
        present = sorted({c for c in conf[d]} | {p for t in conf[d] for p in conf[d][t]})
        mat = [[conf[d][t].get(p, 0) for p in present] for t in present]
        json.dump({"cats": present, "names": [name[c] for c in present], "matrix": mat,
                   "n_gt": n_gt[d], "n_matched": n_match[d]},
                  open(out_dir / f"confusion_{d}.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
