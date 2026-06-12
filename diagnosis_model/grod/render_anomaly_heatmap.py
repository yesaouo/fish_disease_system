"""Option A — sparse-query anomaly heatmap (zero-training report visual).

Splat ALL GROD decoder queries onto a dense field, each as a Gaussian blob sized
to its box and amplitude = objectness w_i = sigmoid(pred_logits[:,0]). Overlapping
confident detections ⟹ hotter ⟹ redder. Pure presentation: this is the SAME
signal as the boxes, so detector-blind lesions (objectness ≈ 0) stay cold — it is
a report visual, not a recall fix (see grod/LESION_GATE.md; the recall fix is the
dense semantic field, Option C). No encoder / CEAH / bank loaded — detector only.

Run from repo root (SDM env):
    $PY -m diagnosis_model.grod.render_anomaly_heatmap --image path/to.jpg --out heat.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from diagnosis_model.grod.extract_hs import cxcywh_norm_to_xyxy_abs


def load_detector(joint_ckpt, anchors, device):
    """GROD joint detector only (box/obj/semantic heads); no global, encoder or CEAH."""
    os.environ["RFDETR_SEMANTIC_DIM"] = "768"
    os.environ["RFDETR_SEMANTIC_ANCHORS"] = os.path.abspath(anchors)
    from rfdetr import RFDETRMedium
    rf = RFDETRMedium(pretrain_weights=joint_ckpt, num_classes=1)
    net = rf.model.model.to(device).eval()
    return net, int(rf.model.resolution), list(rf.means), list(rf.stds)


@torch.no_grad()
def grod_forward(net, image, res, means, stds, device):
    """-> w[Q] objectness, boxes_cxcywh_norm[Q,4]."""
    px = TF.normalize(TF.resize(TF.to_tensor(image), [res, res]),
                      means, stds).unsqueeze(0).to(device)
    out = net(px)
    w = out["pred_logits"][0][:, 0].sigmoid()        # [Q] abnormal objectness
    boxes = out["pred_boxes"][0]                      # [Q,4] cxcywh norm
    return w, boxes


def splat_heatmap(w, boxes, grid=200, sigma_scale=1.0, min_sigma=0.01,
                  agg="sum", min_w=0.0, normalize="absolute"):
    """All queries -> dense [grid,grid] field of Gaussian blobs in [0,1].

    Each query i deposits w_i · exp(-((x-cx)²/2σx² + (y-cy)²/2σy²)) with σ ∝ its
    box size. `sum` rewards agreement (duplicate detections on a real lesion stack
    up); `max` = strongest objectness covering each pixel.

    normalize: `absolute` keeps objectness on a fixed [0,1] scale (clamp) so a
    healthy fish stays cold and images are comparable — use this for reports.
    `per_image` divides by the image max (always produces a red peak, even on a
    healthy fish) — only for "where is the relatively-hottest region".
    """
    dev = w.device
    cx, cy, bw, bh = boxes.unbind(-1)                 # normalized [0,1]
    keep = w >= min_w
    cx, cy, bw, bh, wv = cx[keep], cy[keep], bw[keep], bh[keep], w[keep]
    sx = (bw * 0.5 * sigma_scale).clamp_min(min_sigma)
    sy = (bh * 0.5 * sigma_scale).clamp_min(min_sigma)

    ax = torch.linspace(0, 1, grid, device=dev)
    gy, gx = torch.meshgrid(ax, ax, indexing="ij")    # [G,G]
    dx = gx[..., None] - cx                            # [G,G,Q]
    dy = gy[..., None] - cy
    blob = wv * torch.exp(-(dx**2 / (2 * sx**2) + dy**2 / (2 * sy**2)))
    heat = blob.sum(-1) if agg == "sum" else blob.amax(-1)
    if normalize == "per_image":
        heat = heat / heat.amax().clamp_min(1e-6)
    else:                                              # absolute: objectness scale
        heat = heat.clamp(0, 1)
    return heat.cpu().numpy()                          # [G,G] in [0,1]


def overlay(image, heat, gamma=0.7, max_alpha=0.6, cmap_name="turbo"):
    """Alpha-composite a colormapped heat field over the original image."""
    try:
        from matplotlib import colormaps
        cmap = colormaps[cmap_name]
    except Exception:
        import matplotlib.cm as cm
        cmap = cm.get_cmap(cmap_name)
    W, H = image.size
    h = Image.fromarray((heat * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    hn = np.asarray(h, dtype=np.float32) / 255.0      # [H,W] in [0,1]
    rgb = cmap(hn)[..., :3]                            # [H,W,3] in [0,1]
    alpha = (hn ** gamma * max_alpha)[..., None]       # cold -> transparent
    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    out = base * (1 - alpha) + rgb * alpha
    return Image.fromarray((out * 255).clip(0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ART = "data/processed/current/artifacts"
    ap.add_argument("--joint_ckpt", default=f"{ART}/models/joint_rfdetr/checkpoint_best_regular.pth")
    ap.add_argument("--anchors", default=f"{ART}/models/text_anchors.pt")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="anomaly_heatmap.png")
    ap.add_argument("--grid", type=int, default=200)
    ap.add_argument("--agg", choices=["sum", "max"], default="sum")
    ap.add_argument("--sigma_scale", type=float, default=1.0, help="blob size vs box size")
    ap.add_argument("--gamma", type=float, default=0.7, help="alpha curve; <1 lifts faint areas")
    ap.add_argument("--max_alpha", type=float, default=0.6)
    ap.add_argument("--min_w", type=float, default=0.0, help="drop queries below this objectness")
    ap.add_argument("--normalize", choices=["absolute", "per_image"], default="absolute",
                    help="absolute=fixed objectness scale (healthy stays cold, report-safe); "
                         "per_image=divide by image max (always a red peak)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net, res, means, stds = load_detector(args.joint_ckpt, args.anchors, device)
    image = Image.open(args.image).convert("RGB")
    w, boxes = grod_forward(net, image, res, means, stds, device)
    print(f"[grod] Q={w.numel()} max_w={w.max():.3f} "
          f"#(w>0.5)={int((w > 0.5).sum())} #(0.3<w<0.5)={int(((w > 0.3) & (w < 0.5)).sum())}")

    heat = splat_heatmap(w, boxes, grid=args.grid, sigma_scale=args.sigma_scale,
                         agg=args.agg, min_w=args.min_w, normalize=args.normalize)
    out = overlay(image, heat, gamma=args.gamma, max_alpha=args.max_alpha)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
