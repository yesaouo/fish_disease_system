"""Blind-spot eval for the Tier-1 dense head — does C recover detector-blind lesions?

For each diseased val image: dense lesion-ness field = head(neck)·les_proto −
head(neck)·bg_proto. Each GT box is labeled detected/mid/blind by its best-IoU
query's objectness. We then ask, per group, whether the field is high INSIDE the
GT box vs background — i.e. does C light up lesions the detector head missed.
Healthy images give the false-positive (precision) reference.

Run from repo root (SDM env):
  $PY -m diagnosis_model.grod.eval_dense_head --head_ckpt outputs/grod/dense_head/dense_head.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from diagnosis_model.grod.probe_dense_semantic import load_detector, neck_grid
from diagnosis_model.grod.train_dense_head import DenseHead


def iou(a, b):
    tl = torch.max(a[:, None, :2], b[None, :, :2]); br = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (br - tl).clamp(0).prod(-1)
    aa = (a[:, 2:] - a[:, :2]).clamp(0).prod(-1); ab = (b[:, 2:] - b[:, :2]).clamp(0).prod(-1)
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


@torch.no_grad()
def field(net, head, les, bg, image, res, means, stds, device):
    neck, w = neck_grid(net, image, res, means, stds, device)
    C, Hs, Ws = neck.shape
    z = head(neck.reshape(C, -1).t().float())
    heat = (z @ les.t() - z @ bg.t()).reshape(Hs, Ws)
    return heat, w, Hs, Ws


def main():
    ap = argparse.ArgumentParser()
    ART = "data/processed/current/artifacts"
    ap.add_argument("--head_ckpt", default="outputs/grod/dense_head/dense_head.pt")
    ap.add_argument("--joint_ckpt", default=f"{ART}/models/joint_rfdetr/checkpoint_best_regular.pth")
    ap.add_argument("--anchors", default=f"{ART}/models/text_anchors.pt")
    ap.add_argument("--case_db_dir", default=f"{ART}/db/case_db_jointDistRawP")
    ap.add_argument("--img_root", default="data/processed/current/detection")
    ap.add_argument("--healthy_dir", default="data/healthy_images")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net, res, means, stds = load_detector(args.joint_ckpt, args.anchors, device)
    ck = torch.load(args.head_ckpt, weights_only=False, map_location=device)
    head = DenseHead(ck["d_in"], d_out=ck["d_out"]).to(device).eval()
    head.load_state_dict(ck["head_state"])
    les = ck["les_proto"].to(device); bg = ck["bg_proto"].to(device)

    cases = torch.load(Path(args.case_db_dir) / "valid_cases.pt", weights_only=False)
    groups = {"detected(obj>0.5)": [], "mid(0.1-0.5)": [], "blind(obj<0.1)": []}
    n = 0
    for c in cases:
        p = Path(args.img_root) / "valid" / c["file_name"]
        if not p.exists() or len(c["lesion_boxes_xywh"]) == 0:
            continue
        n += 1
        if n > args.limit:
            break
        image = Image.open(p).convert("RGB"); Wpx, Hpx = image.size
        heat, w, Hs, Ws = field(net, head, les, bg, image, res, means, stds, device)
        gt = torch.as_tensor(c["lesion_boxes_xywh"], dtype=torch.float32)
        gtx = torch.stack([gt[:, 0], gt[:, 1], gt[:, 0] + gt[:, 2], gt[:, 1] + gt[:, 3]], 1)
        gtn = gtx / torch.tensor([Wpx, Hpx, Wpx, Hpx])
        # re-run detector to get per-query boxes (for detected/blind labeling of each GT)
        import torchvision.transforms.functional as TF
        px = TF.normalize(TF.resize(TF.to_tensor(image), [res, res]), means, stds).unsqueeze(0).to(device)
        with torch.no_grad():
            out = net(px)
        qbb = out["pred_boxes"][0]
        qx = torch.stack([qbb[:, 0] - qbb[:, 2] / 2, qbb[:, 1] - qbb[:, 3] / 2,
                          qbb[:, 0] + qbb[:, 2] / 2, qbb[:, 1] + qbb[:, 3] / 2], 1)
        ious = iou(gtn.to(device), qx)
        ys = torch.linspace(0, 1, Hs, device=device); xs = torch.linspace(0, 1, Ws, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        bgmean = heat.mean()
        for gi in range(gtn.size(0)):
            j = ious[gi].argmax(); mi = ious[gi, j].item()
            bo = 0.0 if mi < 0.3 else w[j].item()
            b = gtn[gi]
            inb = (gx >= b[0]) & (gx <= b[2]) & (gy >= b[1]) & (gy <= b[3])
            if inb.sum() < 1:
                continue
            in_max = heat[inb].max().item()                # peak lesion-ness in the GT box
            key = "detected(obj>0.5)" if bo > 0.5 else ("mid(0.1-0.5)" if bo >= 0.1 else "blind(obj<0.1)")
            groups[key].append(in_max)

    # healthy false-positive reference: peak lesion-ness per healthy image
    hpaths = sorted(p for p in Path(args.healthy_dir).iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[:args.limit]
    hmax = []
    import torchvision.transforms.functional as TF
    for p in hpaths:
        image = Image.open(p).convert("RGB")
        heat, _, _, _ = field(net, head, les, bg, image, res, means, stds, device)
        hmax.append(heat.max().item())
    hmax = torch.tensor(hmax)

    import statistics as st
    print(f"\n[blind-spot eval] diseased images used: {n-1}")
    print("  group              n_GT   median in-box PEAK lesion-ness")
    for k, v in groups.items():
        if v:
            print(f"  {k:18} {len(v):4}   {st.median(v):+.3f}   (mean {st.mean(v):+.3f})")
        else:
            print(f"  {k:18}    0")
    # calibrate a fire-threshold at 95th pct of healthy peak; report blind recall at it
    thr = torch.quantile(hmax, 0.95).item()
    print(f"\n  healthy peak lesion-ness: median={hmax.median():+.3f}  95th-pct(τ_fire)={thr:+.3f}")
    for k, v in groups.items():
        if v:
            rec = sum(x > thr for x in v) / len(v)
            print(f"  {k:18} fires (peak>τ): {100*rec:.0f}%")


if __name__ == "__main__":
    main()
