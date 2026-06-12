"""Tier-1 dense semantic head for Option C.

Tier-0 (probe_dense_semantic.py) showed the trained GROD semantic head does NOT
transfer to neck tokens (domain gap), but the blind-lesion separability probe
showed neck tokens DO encode lesions (Cohen-d ≈ 2.3 even for objectness<0.1
lesions). So we train a small head to read neck tokens into SigLIP2 space.

  input : neck token (tap A post-projector, 256-d = d_model), one per 36×36 cell
  output: 768-d SigLIP2 vector, L2-normalized
  teacher: RAW SigLIP2 image-tower embedding of each GT lesion CROP (independent
           of the detector → does not inherit the 77.5% blind spot). cells inside
           a GT box learn that box's crop embedding; sampled background cells learn
           the healthy_region anchor (idx 0), down-weighted.

Inference field = cos(centered head(neck), centered symptom anchors)/τ — same op
as everywhere else; render with probe_dense_semantic.py --head_ckpt.

Run from repo root (SDM env):
  $PY -m diagnosis_model.grod.train_dense_head --cache outputs/grod/dense_head/cells.pt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


# --------------------------------------------------------------------------- #
# models / encoders
# --------------------------------------------------------------------------- #
def load_detector(joint_ckpt, anchors, device):
    os.environ["RFDETR_SEMANTIC_DIM"] = "768"
    os.environ["RFDETR_SEMANTIC_ANCHORS"] = os.path.abspath(anchors)
    from rfdetr import RFDETRMedium
    rf = RFDETRMedium(pretrain_weights=joint_ckpt, num_classes=1)
    return rf.model.model.to(device).eval(), int(rf.model.resolution), list(rf.means), list(rf.stds)


def load_siglip(name, device):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vl_classifier"))
    from transformers import AutoModel, AutoProcessor
    return (AutoModel.from_pretrained(name).to(device).eval(),
            AutoProcessor.from_pretrained(name))


def _img_feats(sig, px):
    from common import get_image_features          # vl_classifier helper (unwraps output obj)
    return get_image_features(sig, px)


@torch.no_grad()
def neck_grid(net, image, res, means, stds, device):
    px = TF.normalize(TF.resize(TF.to_tensor(image), [res, res]), means, stds).unsqueeze(0).to(device)
    cap = {}
    h = net.backbone.register_forward_hook(lambda m, i, o: cap.update(o=o))
    net(px)
    h.remove()
    return cap["o"][0][-1].decompose()[0][0]              # [256,Hs,Ws]


@torch.no_grad()
def bg_prototype(sig, proc, healthy_dir, n, device):
    """Mean RAW SigLIP2 image embedding over n healthy fish — image-space 'normal' ref."""
    paths = sorted(p for p in Path(healthy_dir).iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[:n]
    embs = []
    for i in range(0, len(paths), 64):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i + 64]]
        px = proc(images=ims, return_tensors="pt")["pixel_values"].to(device)
        embs.append(F.normalize(_img_feats(sig, px).float(), dim=-1).cpu())
    return F.normalize(torch.cat(embs).mean(0, keepdim=True), dim=-1)   # [1,768]


@torch.no_grad()
def crop_embs(sig, proc, image, boxes_xyxy, device):
    crops = []
    for x1, y1, x2, y2 in boxes_xyxy.tolist():
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = max(x1 + 2, int(x2)), max(y1 + 2, int(y2))
        crops.append(image.crop((x1, y1, x2, y2)))
    px = proc(images=crops, return_tensors="pt")["pixel_values"].to(device)
    return F.normalize(_img_feats(sig, px).float(), dim=-1).cpu()  # [n,768]


# --------------------------------------------------------------------------- #
# extraction: (neck token, teacher target, is_lesion) per labeled cell
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract(cases, img_root, net, res, means, stds, sig, proc, healthy_tgt,
            device, n_bg=16, limit=None):
    toks, tgts, isl = [], [], []
    n = 0
    for c in cases:
        split = "valid" if c["split"] == "val" else c["split"]
        p = Path(img_root) / split / c["file_name"]
        gt = c["lesion_boxes_xywh"]
        if not p.exists() or len(gt) == 0:
            continue
        n += 1
        if limit and n > limit:
            break
        image = Image.open(p).convert("RGB")
        Wpx, Hpx = image.size
        gt = torch.as_tensor(gt, dtype=torch.float32)
        xyxy = torch.stack([gt[:, 0], gt[:, 1], gt[:, 0] + gt[:, 2], gt[:, 1] + gt[:, 3]], 1)
        bemb = crop_embs(sig, proc, image, xyxy, device)             # [nb,768]
        bn = xyxy / torch.tensor([Wpx, Hpx, Wpx, Hpx])               # normalized xyxy

        neck = neck_grid(net, image, res, means, stds, device)       # [256,Hs,Ws]
        C, Hs, Ws = neck.shape
        tok = neck.reshape(C, -1).t().cpu()                          # [HW,256]
        ys = torch.linspace(0, 1, Hs); xs = torch.linspace(0, 1, Ws)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        cx, cy = gx.reshape(-1), gy.reshape(-1)
        areas = (bn[:, 2] - bn[:, 0]).clamp_min(1e-6) * (bn[:, 3] - bn[:, 1]).clamp_min(1e-6)

        best = torch.full((Hs * Ws,), -1, dtype=torch.long)
        bestA = torch.full((Hs * Ws,), 1e9)
        for bi, b in enumerate(bn):
            inb = (cx >= b[0]) & (cx <= b[2]) & (cy >= b[1]) & (cy <= b[3])
            better = inb & (areas[bi] < bestA)
            best[better] = bi
            bestA[better] = areas[bi]

        les = (best >= 0).nonzero().squeeze(1)
        if les.numel():
            toks.append(tok[les]); tgts.append(bemb[best[les]]); isl.append(torch.ones(les.numel()))
        bg = (best < 0).nonzero().squeeze(1)
        if bg.numel():
            sel = bg[torch.randperm(bg.numel())[:n_bg]]
            toks.append(tok[sel]); tgts.append(healthy_tgt.expand(sel.numel(), -1))
            isl.append(torch.zeros(sel.numel()))
    print(f"[extract] images={n} cells={sum(t.size(0) for t in toks)}")
    return (torch.cat(toks).half(), torch.cat(tgts).half(), torch.cat(isl).bool())


# --------------------------------------------------------------------------- #
# head
# --------------------------------------------------------------------------- #
class DenseHead(nn.Module):
    def __init__(self, d_in=256, d_h=512, d_out=768):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_h), nn.GELU(), nn.Linear(d_h, d_out))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def train(tr, va, bg_proto, device, epochs=40, lr=1e-3, bg_w=0.3, bs=4096):
    head = DenseHead().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    Xtr, Ytr, Ltr = (t.to(device) for t in tr)
    Xva, Yva, Lva = (t.to(device) for t in va)
    w_tr = torch.where(Ltr, torch.ones_like(Ltr, dtype=torch.float), torch.full_like(Ltr, bg_w, dtype=torch.float))

    # image-space prototypes (no modality gap): lesion = mean lesion crop emb, bg = healthy ref
    les_proto = F.normalize(Ytr[Ltr].float().mean(0, keepdim=True), dim=-1)   # [1,768]
    bg_p = bg_proto.to(device)

    def proto_sep():
        head.eval()
        with torch.no_grad():
            p = head(Xva.float())
            heat = (p @ les_proto.t()).squeeze(1) - (p @ bg_p.t()).squeeze(1)  # lesion-ness
        return heat[Lva].mean().item(), heat[~Lva].mean().item()

    best = -1e9; best_sd = None
    for ep in range(1, epochs + 1):
        head.train()
        perm = torch.randperm(Xtr.size(0), device=device)
        for i in range(0, Xtr.size(0), bs):
            idx = perm[i:i + bs]
            cos = (head(Xtr[idx].float()) * Ytr[idx].float()).sum(-1)
            loss = ((1 - cos) * w_tr[idx]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        lh, bh = proto_sep()
        score = lh - bh                                  # lesion vs bg lesion-ness separation
        if score > best:
            best = score; best_sd = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            print(f"  ep{ep:>2}  val lesion-ness lesion={lh:+.3f} bg={bh:+.3f}  sep={score:+.3f}")
    head.load_state_dict(best_sd); head.eval()
    print(f"[train] best val proto-sep={best:+.3f}")
    return head, les_proto.cpu()


def main():
    ap = argparse.ArgumentParser()
    ART = "data/processed/current/artifacts"
    ap.add_argument("--joint_ckpt", default=f"{ART}/models/joint_rfdetr/checkpoint_best_regular.pth")
    ap.add_argument("--anchors", default=f"{ART}/models/text_anchors.pt")
    ap.add_argument("--case_db_dir", default=f"{ART}/db/case_db_jointDistRawP")
    ap.add_argument("--img_root", default="data/processed/current/detection")
    ap.add_argument("--healthy_dir", default="data/healthy_images")
    ap.add_argument("--n_healthy", type=int, default=300, help="healthy imgs for the bg prototype")
    ap.add_argument("--siglip", default="google/siglip2-base-patch16-224")
    ap.add_argument("--cache", default="outputs/grod/dense_head/cells.pt")
    ap.add_argument("--out", default="outputs/grod/dense_head/dense_head.pt")
    ap.add_argument("--n_bg", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = Path(args.cache)

    if cache.exists():
        print(f"[cache] load {cache}")
        d = torch.load(cache, weights_only=False)
        tr, va, bg_proto = d["train"], d["val"], d["bg_proto"]
    else:
        net, res, means, stds = load_detector(args.joint_ckpt, args.anchors, device)
        sig, proc = load_siglip(args.siglip, device)
        bg_proto = bg_prototype(sig, proc, args.healthy_dir, args.n_healthy, device)
        print(f"[bg_proto] from {args.n_healthy} healthy imgs")
        trc = torch.load(Path(args.case_db_dir) / "train_cases.pt", weights_only=False)
        vac = torch.load(Path(args.case_db_dir) / "valid_cases.pt", weights_only=False)
        tr = extract(trc, args.img_root, net, res, means, stds, sig, proc, bg_proto,
                     device, args.n_bg, args.limit)
        va = extract(vac, args.img_root, net, res, means, stds, sig, proc, bg_proto,
                     device, args.n_bg, args.limit)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"train": tr, "val": va, "bg_proto": bg_proto}, cache)
        print(f"[cache] saved {cache}")

    print(f"[data] train cells={tr[0].size(0)} (lesion {int(tr[2].sum())})  "
          f"val cells={va[0].size(0)} (lesion {int(va[2].sum())})")
    head, les_proto = train(tr, va, bg_proto, device, epochs=args.epochs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head_state": head.state_dict(), "d_in": 256, "d_out": 768,
                "les_proto": les_proto, "bg_proto": bg_proto}, args.out)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
