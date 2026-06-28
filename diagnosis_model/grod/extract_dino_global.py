"""Extract a single per-image global vector from RF-DETR's backbone, as a
drop-in source for the SigLIP2 whole-image `global_emb` (distillation target in
distill_global_mlp). No SigLIP2 image forward, no learned head: the backbone
feature map is already computed in the detector's forward, so this is a free
byproduct.

Feature: raw pre-projector DINOv2 encoder features (`net.backbone[0].encoder`), a
list of 4 scales [B,384,H,W]; each spatial-mean pooled + L2-normed, then concat
-> 1536-d. This is what distilled_global_rawP distills from (distill_global_mlp
d_in=1536) and what the demo's disease head reads.

Output `dino_global_{split}.pt`: {"global": [N,1536] L2-normed, "file_names": [...]}
aligned 1:1 with case_db_raw `{split}_cases.pt` order.

Run from repo root:
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
    from diagnosis_model.grod.build import load_oavle

    net, resolution, means, stds = load_oavle(det_ckpt, device=device, num_classes=None)
    return net, means, stds, resolution


@torch.no_grad()
def pool_backbone_global(net, batch_t: torch.Tensor) -> torch.Tensor:
    """Run the detector and pool a per-image global vector. batch_t: [B,3,res,res].

    Raw pre-projector DINOv2 features (net.backbone[0].encoder): a list of 4 scales
    [B,384,H,W]; each spatial-mean pooled + L2-normed, then concatenated ->
    [B, 4*384=1536]. Square resize = no padding, so plain spatial mean."""
    cap: Dict[str, object] = {}
    h = net.backbone[0].encoder.register_forward_hook(lambda m, i, o: cap.update(enc=o))
    try:
        net(batch_t)
    finally:
        h.remove()
    scales = cap["enc"]                          # list of [B,C,H,W]
    # Prefer the encoder's own pooling (external backbones share one definition
    # with inference, e.g. DinoV3Backbone.pool_global); fall back to inline
    # spatial-mean for the built-in DinoV2 encoder.
    enc = net.backbone[0].encoder
    if hasattr(enc, "pool_global"):
        return enc.pool_global(scales)           # [B, sum_C]
    per = [F.normalize(s.mean(dim=(2, 3)), dim=-1) for s in scales]
    return torch.cat(per, dim=-1)                # [B, 4*384]


def main():
    ap = argparse.ArgumentParser(description="RF-DETR backbone (DINOv2) 1536-d global vector per case.")
    ap.add_argument("--case_db_dir", type=str, required=True)
    ap.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"])
    ap.add_argument("--det_ckpt", type=str, required=True)
    ap.add_argument("--image_root", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
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
        pooled = pool_backbone_global(net, batch_t)   # [B, 1536] L2-normed
        out_chunks.append(pooled.cpu())
        if (start // args.batch_size) % 20 == 0:
            done = start + len(chunk)
            print(f"  {done}/{len(file_names)}  ({done / max(1e-9, time.time() - t0):.1f} img/s)")

    global_out = torch.cat(out_chunks, dim=0)              # [N, 1536]
    C = global_out.size(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dino_global_{args.split}.pt"
    torch.save({"global": global_out, "file_names": file_names, "pooled_dim": C}, out_path)
    print(f"[done] {tuple(global_out.shape)} "
          f"norm={global_out.norm(dim=-1).mean():.3f} -> {out_path}  "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
