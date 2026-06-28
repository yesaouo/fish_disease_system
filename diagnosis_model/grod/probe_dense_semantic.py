"""Tier-0 probe for Option C — dense semantic field, ZERO training.

Reuse the already-trained GROD semantic head (Linear d_model→768) on the dense
NECK feature map (tap A, same 256-d space the decoder queries live in) instead of
on the 300 sparse decoder queries. Project every spatial location into SigLIP2
space, cos vs centered symptom anchors → a per-symptom dense field, NOT gated by
objectness. Tests the make-or-break C hypothesis: do detector-blind lesions
(objectness≈0) still light up in the semantic field?

This validates the projection-head I/O (in 256 neck token / out 768 SigLIP2) and
the core claim before training the Tier-1 dense head. anchors 0.97-collinear, so
center (subtract anchor centroid) before cos — see project_grod_anchor_anisotropy.

Run from repo root (SDM env):
    $PY -m diagnosis_model.grod.probe_dense_semantic --image path/to.jpg --out c.png
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from diagnosis_model.grod.render_anomaly_heatmap import overlay


def load_detector(joint_ckpt, anchors, device):
    os.environ["RFDETR_SEMANTIC_DIM"] = "768"
    os.environ["RFDETR_SEMANTIC_ANCHORS"] = os.path.abspath(anchors)
    from diagnosis_model.grod.build import load_oavle
    return load_oavle(joint_ckpt, device=device)


@torch.no_grad()
def neck_grid(net, image, res, means, stds, device):
    px = TF.normalize(TF.resize(TF.to_tensor(image), [res, res]),
                      means, stds).unsqueeze(0).to(device)
    cap = {}
    h = net.backbone.register_forward_hook(lambda m, i, o: cap.update(out=o))
    out = net(px)
    h.remove()
    src, _ = cap["out"][0][-1].decompose()                 # [1,256,Hs,Ws]
    return src[0], out["pred_logits"][0][:, 0].sigmoid()   # neck [256,Hs,Ws], w[Q]


@torch.no_grad()
def dense_field_tier1(net, image, res, means, stds, head, les_proto, bg_proto, device):
    """Tier-1: trained dense head -> SigLIP image space; lesion-ness vs image protos."""
    neck, w = neck_grid(net, image, res, means, stds, device)
    C, Hs, Ws = neck.shape
    z = head(neck.reshape(C, -1).t().float())              # [HW,768] L2-normed
    heat = (z @ les_proto.to(device).t() - z @ bg_proto.to(device).t()).reshape(Hs, Ws)
    return heat, w


@torch.no_grad()
def per_symptom_field(net, image, res, means, stds, head, protos, bg_proto, device):
    """-> lesion-ness[Hs,Ws], dominant-symptom-idx[Hs,Ws] (into protos rows)."""
    neck, w = neck_grid(net, image, res, means, stds, device)
    C, Hs, Ws = neck.shape
    z = head(neck.reshape(C, -1).t().float())              # [HW,768]
    s = z @ protos.to(device).t()                          # [HW,K] cos to each symptom proto
    bgc = (z @ bg_proto.to(device).t()).squeeze(1)         # [HW]
    return (s.amax(1) - bgc).reshape(Hs, Ws), s.argmax(1).reshape(Hs, Ws), w


def render_per_symptom(image, lesness, dom, names, out, vmax=0.30, max_alpha=0.7):
    """Color each cell by its dominant symptom; alpha = lesion-ness. + legend."""
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.patches import Patch
    try:
        from diagnosis_model.grod.render_anomaly_heatmap import _font  # noqa
    except Exception:
        pass
    K = len(names)
    cmap = colormaps["tab10"]
    W, H = image.size
    a = (lesness.clamp(0, vmax) / vmax).cpu().numpy()       # [Hs,Ws] in [0,1]
    d = dom.cpu().numpy()
    rgb = cmap(d % 10)[..., :3]                             # [Hs,Ws,3] symptom color
    import numpy as np
    from PIL import Image as PILImage
    rgb_full = np.asarray(PILImage.fromarray((rgb * 255).astype(np.uint8)).resize((W, H), PILImage.NEAREST)) / 255.0
    a_full = np.asarray(PILImage.fromarray((a * 255).astype(np.uint8)).resize((W, H), PILImage.BILINEAR)) / 255.0
    alpha = (a_full * max_alpha)[..., None]
    base = np.asarray(image.convert("RGB"), np.float32) / 255.0
    comp = (base * (1 - alpha) + rgb_full * alpha)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(comp); ax.axis("off")
    try:
        fp = _font(11)
    except Exception:
        fp = None
    handles = [Patch(color=cmap(i % 10), label=nm) for i, nm in enumerate(names)]
    ax.legend(handles=handles, loc="lower right", prop=fp, framealpha=0.85, fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


@torch.no_grad()
def dense_semantic_field(net, image, res, means, stds, anchors, device):
    """-> sim[H,W,C] centered cos to each symptom anchor; also w[Q] objectness for A-compare."""
    px = TF.normalize(TF.resize(TF.to_tensor(image), [res, res]),
                      means, stds).unsqueeze(0).to(device)

    cap = {}
    h = net.backbone.register_forward_hook(lambda m, i, o: cap.update(out=o))
    out = net(px)
    h.remove()

    # tap A: post-projector neck memory [B,256,H,W] — same d_model space as hs
    src, _ = cap["out"][0][-1].decompose()                 # [1,256,Hs,Ws]
    _, Cdim, Hs, Ws = src.shape
    tok = src[0].permute(1, 2, 0).reshape(-1, Cdim)        # [Hs*Ws, 256]

    # reuse the TRAINED semantic head (Linear 256->768) + L2-norm — zero training
    z = F.normalize(net.semantic_embed(tok), dim=-1)       # [Hs*Ws, 768]

    A = torch.load(anchors, weights_only=False)["anchor_embs"].float().to(device)  # [C,768]
    mu = A.mean(0, keepdim=True)
    A_c = F.normalize(A - mu, dim=-1)
    z_c = F.normalize(z - mu, dim=-1)
    sim = (z_c @ A_c.t()).reshape(Hs, Ws, -1)              # [Hs,Ws,C] centered cos
    w = out["pred_logits"][0][:, 0].sigmoid()              # [Q] objectness (A's signal)
    return sim, w


def main():
    ap = argparse.ArgumentParser()
    ART = "data/processed/current/artifacts"
    ap.add_argument("--joint_ckpt", default=f"{ART}/models/joint_rfdetr/checkpoint_best_regular.pth")
    ap.add_argument("--anchors", default=f"{ART}/models/text_anchors.pt")
    ap.add_argument("--symptoms", default="data/processed/current/symptoms.json")
    ap.add_argument("--head_ckpt", default=None,
                    help="Tier-1 trained dense head (train_dense_head.py). If set, use the "
                         "image-space lesion-ness operator instead of Tier-0 text-anchor reuse.")
    ap.add_argument("--symptom_protos", default=None,
                    help="per-symptom image protos (build_symptom_protos.py); renders a "
                         "symptom-colored map instead of the binary lesion-ness map.")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="dense_semantic.png")
    ap.add_argument("--normalize", choices=["per_image", "absolute"], default="per_image")
    ap.add_argument("--vmax", type=float, default=0.30,
                    help="absolute mode: lesion-ness mapped [0,vmax]->[0,1] (Tier-1 range ~0.4)")
    ap.add_argument("--gamma", type=float, default=0.7)
    ap.add_argument("--max_alpha", type=float, default=0.6)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net, res, means, stds = load_detector(args.joint_ckpt, args.anchors, device)
    image = Image.open(args.image).convert("RGB")

    if args.head_ckpt and args.symptom_protos:              # Tier-1 per-SYMPTOM map
        from diagnosis_model.grod.train_dense_head import DenseHead
        ck = torch.load(args.head_ckpt, weights_only=False, map_location=device)
        head = DenseHead(ck["d_in"], d_out=ck["d_out"]).to(device).eval()
        head.load_state_dict(ck["head_state"])
        sp = torch.load(args.symptom_protos, weights_only=False)
        lesness, dom, w = per_symptom_field(net, image, res, means, stds, head,
                                            sp["protos"], ck["bg_proto"], device)
        from collections import Counter
        hot = dom[lesness > 0.10]
        top = Counter(sp["names"][i] for i in hot.cpu().tolist())
        print(f"[per-symptom] grid={tuple(lesness.shape)}  hot-cell symptoms: "
              + ", ".join(f"{n}({c})" for n, c in top.most_common(4)))
        render_per_symptom(image, lesness, dom, sp["names"], args.out,
                           vmax=args.vmax, max_alpha=args.max_alpha)
        print(f"[save] {args.out}")
        return
    if args.head_ckpt:                                      # Tier-1: trained head + image protos
        from diagnosis_model.grod.train_dense_head import DenseHead
        ck = torch.load(args.head_ckpt, weights_only=False, map_location=device)
        head = DenseHead(ck["d_in"], d_out=ck["d_out"]).to(device).eval()
        head.load_state_dict(ck["head_state"])
        heat, w = dense_field_tier1(net, image, res, means, stds, head,
                                    ck["les_proto"], ck["bg_proto"], device)
        Hs, Ws = heat.shape
        print(f"[tier1] grid={Hs}x{Ws}  A:max_w={w.max():.3f} #(w>0.5)={int((w>0.5).sum())}  "
              f"C:lesion-ness max={heat.max():.3f} mean={heat.mean():.3f}")
    else:                                                  # Tier-0: reuse semantic head + anchors
        sim, w = dense_semantic_field(net, image, res, means, stds, args.anchors, device)
        heat = sim[..., 1:].amax(-1)                        # max over ABNORMAL anchors
        Hs, Ws = heat.shape
        print(f"[tier0] grid={Hs}x{Ws}  A:max_w={w.max():.3f}  C:max_cos={heat.max():.3f}")

    hn = (heat / args.vmax).clamp(0, 1) if args.normalize == "absolute" else \
        (heat - heat.min()) / (heat.max() - heat.min()).clamp_min(1e-6)
    out = overlay(image, hn.cpu().numpy(), gamma=args.gamma, max_alpha=args.max_alpha)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
