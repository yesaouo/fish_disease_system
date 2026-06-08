"""Extract a single per-image global vector from RF-DETR's backbone, as a
drop-in source for the SigLIP2 whole-image `global_emb` (distillation target in
distill_global_mlp). No SigLIP2 image forward, no learned head: the backbone
feature map is already computed in the detector's forward, so this is a free
byproduct.

Two taps (`--tap`):
  - "B" (PRODUCTION default): raw pre-projector DINOv2 encoder features, a list
    of 4 scales [B,384,H,W]; each masked-mean pooled + L2-normed, then concat
    -> 1536-d. Save raw (--target_dim 0). This is what distilled_global_rawP
    distills from (distill_global_mlp d_in=1536).
  - "A" (ablation probe only): post-projector `srcs` from `net.backbone` (the
    detection neck output, 256-d). Pass --target_dim 768 to zero-pad to a
    768-d case_db drop-in.

Output `dino_global_{split}.pt`: {"global": [N,C] L2-normed, "file_names": [...]}
aligned 1:1 with case_db_raw `{split}_cases.pt` order (C=1536 for tap B).

Run from repo root (production = tap B is the default, no flags needed):
  PY=/home/lab603/anaconda3/envs/SDM/bin/python
  $PY -m diagnosis_model.grod.extract_dino_global \
      --case_db_dir data/processed/current/artifacts/db/case_db_raw \
      --split train \
      --det_ckpt data/processed/current/artifacts/models/rfdetr/checkpoint_best_total.pth \
      --image_root data/processed/current/detection/train \
      --output_dir data/processed/current/artifacts/db/dino_global
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


def load_detector(det_ckpt: str, device: str):
    from rfdetr import RFDETRMedium

    rf = RFDETRMedium(pretrain_weights=det_ckpt)
    net = rf.model.model.to(device).eval()
    resolution = int(rf.model.resolution)
    return net, list(rf.means), list(rf.stds), resolution


@torch.no_grad()
def pool_backbone_global(net, batch_t: torch.Tensor, tap: str = "A") -> torch.Tensor:
    """Run the detector and pool a per-image global vector. batch_t: [B,3,res,res].

    tap="A": post-projector neck features (net.backbone -> (features,poss)),
             1 scale 256-d for Medium, masked-mean pool -> [B,256] L2-normed.
    tap="B": raw pre-projector DINOv2 features (net.backbone[0].encoder), a list
             of 4 scales [B,384,H,W]; each masked-mean pooled + L2-normed, then
             concatenated -> [B, 4*384=1536]. No padding under square resize, so
             plain spatial mean."""
    cap: Dict[str, object] = {}
    if tap == "A":
        h = net.backbone.register_forward_hook(lambda m, i, o: cap.update(out=o))
        try:
            net(batch_t)
        finally:
            h.remove()
        features = cap["out"][0]
        src, mask = features[-1].decompose()
        valid = (~mask).unsqueeze(1).to(src.dtype)
        denom = valid.sum(dim=(2, 3)).clamp_min(1.0)
        pooled = (src * valid).sum(dim=(2, 3)) / denom
        return F.normalize(pooled, dim=-1)
    elif tap == "B":
        h = net.backbone[0].encoder.register_forward_hook(lambda m, i, o: cap.update(enc=o))
        try:
            net(batch_t)
        finally:
            h.remove()
        scales = cap["enc"]                          # list of [B,384,H,W]
        per = [F.normalize(s.mean(dim=(2, 3)), dim=-1) for s in scales]
        return torch.cat(per, dim=-1)                # [B, 4*384]
    else:
        raise ValueError(f"unknown tap {tap!r}")


def main():
    ap = argparse.ArgumentParser(description="Probe A: RF-DETR backbone global vector per case.")
    ap.add_argument("--case_db_dir", type=str, required=True)
    ap.add_argument("--split", type=str, default="train", choices=["train", "valid"])
    ap.add_argument("--det_ckpt", type=str, required=True)
    ap.add_argument("--image_root", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--tap", type=str, default="B", choices=["A", "B"],
                    help="B=raw DINOv2 4-scale concat (1536-d, PRODUCTION default); "
                         "A=post-projector neck (256-d, kept as ablation probe only)")
    ap.add_argument("--target_dim", type=int, default=0,
                    help="zero-pad pooled vector up to this dim for case_db drop-in; "
                         "0 (default) = no pad, save raw (tap B = 1536-d). Pass 768 for the "
                         "tap-A 256->768 drop-in ablation.")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    case_db_dir = Path(args.case_db_dir)
    cases = torch.load(case_db_dir / f"{args.split}_cases.pt", weights_only=False)
    file_names = [c["file_name"] for c in cases]
    print(f"[load] {args.split}_cases.pt  n={len(cases)}")

    net, means, stds, res = load_detector(args.det_ckpt, args.device)
    print(f"[det] device={args.device} resolution={res}")

    image_root = Path(args.image_root)
    out_chunks: List[torch.Tensor] = []
    t0 = time.time()

    for start in range(0, len(file_names), args.batch_size):
        chunk = file_names[start:start + args.batch_size]
        imgs = []
        for fn in chunk:
            im = Image.open(image_root / fn).convert("RGB")
            t = TF.normalize(TF.resize(TF.to_tensor(im), [res, res]), means, stds)
            imgs.append(t)
        batch_t = torch.stack(imgs).to(args.device)
        pooled = pool_backbone_global(net, batch_t, tap=args.tap)   # [B, C] L2-normed
        out_chunks.append(pooled.cpu())
        if (start // args.batch_size) % 20 == 0:
            done = start + len(chunk)
            print(f"  {done}/{len(file_names)}  ({done / max(1e-9, time.time() - t0):.1f} img/s)")

    pooled_all = torch.cat(out_chunks, dim=0)              # [N, C]
    C = pooled_all.size(1)
    if args.target_dim <= 0 or C == args.target_dim:
        global_out = pooled_all
    elif C < args.target_dim:
        pad = torch.zeros(pooled_all.size(0), args.target_dim - C)
        global_out = torch.cat([pooled_all, pad], dim=1)   # [N, target_dim], norm still 1
    elif C > args.target_dim:
        raise ValueError(f"pooled dim {C} > target_dim {args.target_dim}")
    else:
        global_out = pooled_all

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dino_global_{args.split}.pt"
    torch.save({"global": global_out, "file_names": file_names,
                "pooled_dim": C, "target_dim": args.target_dim}, out_path)
    print(f"[done] {tuple(global_out.shape)} (pooled {C}->{args.target_dim}) "
          f"norm={global_out.norm(dim=-1).mean():.3f} -> {out_path}  "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
