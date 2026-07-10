"""Distill the raw DINOv2 global into the SigLIP2 whole-image global space.

Train a small MLP: DINOv2 backbone global (tap B, 1536-d) -> SigLIP2 whole-image
global (768-d), cosine distillation. The point: distilling toward the
image-text-aligned SigLIP2 target STRIPS the pure-visual signal that let the raw
DINO global become a CEAH attribution shortcut, restoring faithfulness — while
still being producible from one RF-DETR forward (no SigLIP2 image tower).

Inputs:
  - DINOv2 global: dino_global_{split}.pt from extract_dino_global
    (1536-d, fed verbatim; MLP d_in = its dim).
  - SigLIP2 target: <target_db>/{split}_cases.pt global_emb (768, L2-normed).
    Production target = case_db_raw (raw SigLIP2 global) -> "rawP" variant.
Output: distilled_global_{split}.pt + global_embed_state_dict.pt (path P: loads
1:1 into the fork's LWDETR.global_embed).

Run from repo root:
  $PY -m diagnosis_model.grod.distill_global_mlp \
      --dino_dir   data/processed/current/artifacts/db/dino_global \
      --target_db  data/processed/current/artifacts/db/case_db_raw \
      --out_dir    data/processed/current/artifacts/models/distilled_global_rawP
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_split(dino_dir: Path, target_db: Path, split: str):
    dino = torch.load(dino_dir / f"dino_global_{split}.pt", weights_only=False)
    # IMPORTANT: feed exactly what LWDETR._pool_global produces (per-scale L2-norm
    # then concat; NO whole-vector renorm), so the trained MLP state_dict loads
    # 1:1 into the folded global_embed and matches model inputs.
    x = dino["global"].clone()
    cases = torch.load(target_db / f"{split}_cases.pt", weights_only=False)
    fns = dino["file_names"]
    assert [c["file_name"] for c in cases] == fns, f"{split}: order mismatch"
    y = torch.stack([c["global_emb"] for c in cases]).float()   # SigLIP2 global [N,768]
    return x, F.normalize(y, dim=-1), fns


from diagnosis_model.grod.detector.models.math import MLP as _RFMLP  # exact arch of the folded global_embed (vendored)


class MLP(nn.Module):
    """Wraps the fork's MLP (Linear/ReLU FFN) + L2-norm, so the trained
    state_dict loads 1:1 into LWDETR.global_embed (path P, no fork retrain).
    Single Linear (num_layers=1): the pooled-DINO -> SigLIP2-global map is
    near-linear, so 1 layer distills it best (valid_cos 0.945 vs 0.917 for the
    old 3L/256-bottleneck) and is seed-deterministic. Must match region_heads.py's
    global_embed build (num_layers=1). d_hidden unused at num_layers=1."""

    def __init__(self, d_in=1536, d_hidden=768, d_out=768, num_layers=1):
        super().__init__()
        self.net = _RFMLP(d_in, d_hidden, d_out, num_layers)

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

    def export_embed_state_dict(self):
        """state_dict keyed to match LWDETR.global_embed (the inner _RFMLP)."""
        return self.net.state_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dino_dir", type=str, required=True)
    ap.add_argument("--target_db", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dino_dir, target_db = Path(args.dino_dir), Path(args.target_db)
    xtr, ytr, _ = load_split(dino_dir, target_db, "train")
    xva, yva, fva = load_split(dino_dir, target_db, "valid")
    xtr2, ytr2, ftr = load_split(dino_dir, target_db, "train")
    print(f"[data] train={xtr.shape} valid={xva.shape}")

    dev = args.device
    model = MLP(d_in=xtr.size(1), d_out=ytr.size(1)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    xtr, ytr, xva, yva = xtr.to(dev), ytr.to(dev), xva.to(dev), yva.to(dev)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "train_log.jsonl", "w")

    best_va, best_epoch = -1.0, -1
    best_state = None
    for ep in range(args.epochs):
        model.train()
        opt.zero_grad()
        pred = model(xtr)
        loss = (1 - (pred * ytr).sum(-1)).mean()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            cva = (model(xva) * yva).sum(-1).mean().item()
            ctr = (pred * ytr).sum(-1).mean().item()
        if cva > best_va:
            best_va, best_epoch = cva, ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        log_f.write(json.dumps({"epoch": ep + 1, "train_cos": ctr, "valid_cos": cva}) + "\n")
        if (ep + 1) % 25 == 0 or ep == 0:
            print(f"  ep{ep+1:>3} train_cos={ctr:.4f} valid_cos={cva:.4f}")
    log_f.close()

    # save the best-valid checkpoint (not the last epoch)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        gtr = model(xtr2.to(dev)).cpu()
        gva = model(xva).cpu()
    torch.save({"global": gtr, "file_names": ftr}, out_dir / "distilled_global_train.pt")
    torch.save({"global": gva, "file_names": fva}, out_dir / "distilled_global_valid.pt")
    torch.save(model.export_embed_state_dict(), out_dir / "global_embed_state_dict.pt")
    print(f"[done] best valid_cos={best_va:.4f} @ epoch {best_epoch}/{args.epochs}  "
          f"saved train{tuple(gtr.shape)} valid{tuple(gva.shape)} + global_embed_state_dict.pt")


if __name__ == "__main__":
    main()
