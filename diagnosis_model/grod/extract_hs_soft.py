"""L2 prerequisite — cache the full 300-query decoder feature hs[-1] per case.

extract_hs.py only keeps IoU-matched GT lesions (hard, 1-2/case). The soft L2
end-to-end needs ALL 300 queries' hs so the (now trainable) objectness +
semantic heads can be re-run on cached features without re-forwarding RF-DETR.

class_embed and semantic_embed both consume the last decoder layer hs[-1]
(lwdetr.py: pred_logits = class_embed(hs)[-1]; pred_semantic =
normalize(semantic_embed(hs[-1]))), so one [Q, 256] cache feeds both heads.

Correctness gate (printed): re-running the heads on the captured hs must
reproduce the soft_inputs that extract_soft_inputs.py dumped —
  class_embed(hs)[...,0].sigmoid() ≈ w,   normalize(semantic_embed(hs)) ≈ z_all.

Output: <out_dir>/hs_{train,valid}.pt  -> {"hs":[N,Q,256] bf16, "case_id":[...]}
aligned 1:1 to case_db_dir order (same as soft_inputs).

Run:  $PY -m diagnosis_model.grod.extract_hs_soft
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ImgDataset(Dataset):
    def __init__(self, paths, res, means, stds):
        self.paths, self.res, self.means, self.stds = paths, res, means, stds

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return TF.normalize(TF.resize(TF.to_tensor(img), [self.res, self.res]),
                            self.means, self.stds)


class GrodHS:
    """Same model load as extract_soft_inputs.Grod, plus an hs[-1] capture hook."""

    def __init__(self, joint_ckpt, global_sd, anchors, device="cuda"):
        self.dev = device
        os.environ["RFDETR_SEMANTIC_DIM"] = "768"
        os.environ["RFDETR_SEMANTIC_ANCHORS"] = os.path.abspath(anchors)
        os.environ["RFDETR_GLOBAL_DIM"] = "768"
        from diagnosis_model.grod.build import load_oavle
        self.net, self.res, self.means, self.stds = load_oavle(joint_ckpt, device=device)
        self.net.global_embed.load_state_dict(torch.load(global_sd, map_location=device))
        # locate class_embed Linear and hook its input (= full hs [L,B,Q,Hd])
        self.class_embed = self._find_class_embed()
        self.semantic_embed = self.net.semantic_embed
        self._cap = {}
        self.class_embed.register_forward_hook(self._hook)

    def _find_class_embed(self):
        for name, mod in self.net.named_modules():
            if name.split(".")[-1] == "class_embed" and isinstance(mod, torch.nn.Linear):
                return mod
        ce = getattr(self.net, "class_embed", None)
        if isinstance(ce, torch.nn.Linear):
            return ce
        raise RuntimeError("class_embed Linear not found")

    def _hook(self, mod, inp, out):
        x = inp[0]                       # [L,B,Q,Hd] or [B,Q,Hd]
        if x.dim() == 4:
            x = x[-1]                    # last decoder layer -> [B,Q,Hd]
        self._cap["hs"] = x.detach()

    @torch.no_grad()
    def forward(self, px):
        out = self.net(px.to(self.dev))
        hs = self._cap["hs"]                                      # [B,Q,256]
        g = out["pred_global"].float().cpu()
        z = F.normalize(out["pred_semantic"].float(), dim=-1).cpu()
        w = out["pred_logits"][..., 0].sigmoid().float().cpu()
        return g, z, w, hs.float().cpu()


def verify(grod, hs, z, w):
    """Re-run heads on captured hs; must match the model's own outputs."""
    hs_d = hs.to(grod.dev)
    logit = grod.class_embed(hs_d)[..., 0]                        # [B,Q]
    z_hat = F.normalize(grod.semantic_embed(hs_d).float(), dim=-1).cpu()
    w_hat = logit.sigmoid().float().cpu()
    return (w_hat - w).abs().max().item(), (z_hat - z).abs().max().item()


def main():
    ap = argparse.ArgumentParser()
    ART = "data/processed/current/artifacts"
    ap.add_argument("--joint_ckpt", default=f"{ART}/models/joint_rfdetr/checkpoint_best_regular.pth")
    ap.add_argument("--global_sd", default=f"{ART}/models/distilled_global_rawP/global_embed_state_dict.pt")
    ap.add_argument("--anchors", default=f"{ART}/models/text_anchors.pt")
    ap.add_argument("--case_db_dir", default=f"{ART}/db/case_db_jointDistRawP")
    ap.add_argument("--img_root", default="data/processed/current/detection")
    ap.add_argument("--out_dir", default=f"{ART}/db/hs_soft")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    grod = GrodHS(args.joint_ckpt, args.global_sd, args.anchors, dev)

    for fname, split in [("train_cases.pt", "train"), ("valid_cases.pt", "valid")]:
        cases = torch.load(Path(args.case_db_dir) / fname, weights_only=False)
        if args.limit:
            cases = cases[:args.limit]
        paths = [str(Path(args.img_root) / split / c["file_name"]) for c in cases]
        loader = DataLoader(ImgDataset(paths, grod.res, grod.means, grod.stds),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
        print(f"[extract] {split}: {len(cases)} cases")
        hss = []
        checked = False
        for bi, px in enumerate(loader):
            g, z, w, hs = grod.forward(px)
            if not checked:
                dw, dz = verify(grod, hs, z, w)
                print(f"  [verify] max|Δw|={dw:.2e} max|Δz|={dz:.2e} (want ~0)")
                checked = True
            hss.append(hs.to(torch.bfloat16))
            if (bi + 1) % 50 == 0:
                print(f"  ...{(bi+1)*args.batch_size}/{len(paths)}", flush=True)
        data = {"hs": torch.cat(hss), "case_id": [c["case_id"] for c in cases]}
        out = out_dir / f"hs_{split}.pt"
        torch.save(data, out)
        print(f"[save] {out}  hs={tuple(data['hs'].shape)}")


if __name__ == "__main__":
    main()
