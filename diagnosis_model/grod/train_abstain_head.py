"""Train the GROD-side abstain (健/病) head — the disease head the demo loads.

Reads the versioned dataset directly (`db_pipeline` output): the `detection` view's
COCO already carries the health label per image — `isHealthy:true` = negative
(healthy fish + OOD pools like sashimi / SalmonScan, gathered from folders), anything
with boxes = diseased. Built-in **train / valid / test** split (md5-stable), so the
**test split is a real held-out OOD eval** (sashimi / SalmonScan land there too).

No special class/OOD weighting: with the structured negatives the balance is ~2:1
diseased:healthy and OOD is ~8 % of negatives (not the old 0.3 % drowning case), so a
plain shuffle + BCE suffices (the 2:1 lean toward "diseased" is the safe direction —
fewer missed disease). `valid` selects the epoch + τ*; `test` is reported (incl. a
per-source negative-reject breakdown).

Two input features (``--feat``):
  - **dino_neck** (DEFAULT, production): RF-DETR backbone tap-B DINOv2 4-scale
    patch-mean → 1536. Pre-distillation visual signal; far stronger for health/OOD
    than the distilled global. Free at inference (demo hooks `backbone[0].encoder`).
  - **pooled** (legacy ablation): concat(g[768], max_w, Σw) = 770.

Head = standardize → Linear(dim→1) → sigmoid (linear probe); diseased ⟺ p ≥ τ*.

Add OOD negatives: drop images into `data/healthy_images/<folder>/`, rebuild the
dataset (`db_pipeline`), then rerun this. Run from repo root:
  $PY -m diagnosis_model.grod.train_abstain_head
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader

from diagnosis_model.grod.disease_head import DiseaseHead
from diagnosis_model.grod.extract_dino_global import pool_backbone_global
from diagnosis_model.grod.train_disease_head import GrodFeatures, auroc, extract_features, pick_tau

SPLITS = ["train", "valid", "test"]


def derive_770(X: torch.Tensor) -> torch.Tensor:
    """1068 [g[768] | raw logits[Q]] -> 770 [g | max sigmoid | sum sigmoid]."""
    g, logits = X[:, :768], X[:, 768:]
    w = logits.sigmoid()
    return torch.cat([g, w.amax(1, keepdim=True), w.sum(1, keepdim=True)], dim=-1)


def gather_from_coco(det_root):
    """Return split -> (paths, labels, sources) from the detection COCO.

    label 1 = diseased (has boxes), 0 = healthy (isHealthy). source = source_dataset.
    """
    det_root = Path(det_root)
    out = {}
    for sp in SPLITS:
        coco = json.load(open(det_root / sp / "_annotations.coco.json"))
        paths, labels, sources = [], [], []
        for im in coco["images"]:
            paths.append(str(det_root / sp / im["file_name"]))
            labels.append(0 if im.get("isHealthy") else 1)
            sources.append(im.get("source_dataset", "?"))
        out[sp] = (paths, labels, sources)
        y = torch.tensor(labels)
        print(f"[samples] {sp}: diseased={int((y==1).sum())} healthy={int((y==0).sum())} "
              f"total={len(y)}")
    return out


def extract_neck(grod, paths, batch_size):
    feats = []
    for b in range(0, len(paths), batch_size):
        chunk = paths[b:b + batch_size]
        imgs = [TF.normalize(TF.resize(TF.to_tensor(Image.open(p).convert("RGB")),
                                       [grod.res, grod.res]), grod.means, grod.stds) for p in chunk]
        feats.append(pool_backbone_global(grod.net, torch.stack(imgs).to(grod.dev)).cpu())
        if (b // batch_size) % 50 == 0:
            print(f"    {b}/{len(paths)}")
    return torch.cat(feats, dim=0)


def build_features(grod, samples, cache_path, feat, batch_size, workers, recompute):
    if cache_path.exists() and not recompute:
        d = torch.load(cache_path, weights_only=False)
        if all(s in d and d[s]["y"].numel() == len(samples[s][1]) for s in SPLITS):
            print(f"[cache] loading {cache_path}"); return d
        print("[cache] sample count changed -> recompute")
    data = {}
    for sp in SPLITS:
        paths, labels, sources = samples[sp]
        print(f"[extract:{feat}] {sp}: {len(paths)} images")
        X = extract_neck(grod, paths, batch_size) if feat == "dino_neck" \
            else extract_features(grod, paths, batch_size, workers)
        data[sp] = {"X": X, "y": torch.tensor(labels, dtype=torch.float32), "source": sources}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache_path); print(f"[cache] saved {cache_path}")
    return data


def confusion(p, y, tau):
    pred = p >= tau
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return dict(recall=tp / (tp + fn + 1e-9), reject=tn / (tn + fp + 1e-9),
                tp=tp, fn=fn, fp=fp, tn=tn)


def main():
    ap = argparse.ArgumentParser()
    ART = "data/processed/current/artifacts"
    ap.add_argument("--feat", choices=["dino_neck", "pooled"], default="dino_neck")
    ap.add_argument("--det_root", default="data/processed/current/detection")
    ap.add_argument("--joint_ckpt", default=f"{ART}/models/joint_rfdetr/checkpoint_best_regular.pth")
    ap.add_argument("--global_sd", default=f"{ART}/models/distilled_global_rawP/global_embed_state_dict.pt")
    ap.add_argument("--anchors", default=f"{ART}/models/text_anchors.pt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use_cache", action="store_true",
                    help="reuse cached features if present; default recomputes "
                         "(cache validity only checks sample count, not stale GROD features)")
    args = ap.parse_args()
    md = f"{ART}/models/disease_head"
    out = Path(args.out or (f"{md}/neck_disease_head.pt" if args.feat == "dino_neck"
                            else f"{md}/abstain_head.pt"))
    cache = Path(args.cache or (f"{md}/neck_features.pt" if args.feat == "dino_neck"
                                else f"{md}/abstain_features.pt"))

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    samples = gather_from_coco(args.det_root)
    grod = GrodFeatures(args.joint_ckpt, args.global_sd, args.anchors, dev)
    data = build_features(grod, samples, cache, args.feat, args.batch_size, args.workers,
                          recompute=not args.use_cache)

    def feats(sp):
        X = data[sp]["X"]
        return derive_770(X) if args.feat == "pooled" else X
    Xtr, ytr = feats("train"), data["train"]["y"]
    Xva, yva = feats("valid"), data["valid"]["y"]
    Xte, yte = feats("test"), data["test"]["y"]
    dim = Xtr.size(1)
    layout = ("dino_neck_1536 (RF-DETR tap B 4-scale patch-mean)" if args.feat == "dino_neck"
              else "concat(g[768], max_w, sum_w)")
    print(f"[feat] {args.feat} dim={dim}")

    head = DiseaseHead(dim).to(dev)
    head.feat_mean.copy_(Xtr.mean(0).to(dev)); head.feat_std.copy_(Xtr.std(0).clamp_min(1e-6).to(dev))

    Xtr_d, ytr_d, Xva_d = Xtr.to(dev), ytr.to(dev), Xva.to(dev)
    # plain shuffle — no class/OOD weighting (structured negatives are ~2:1, OOD ~8%)
    idx_loader = DataLoader(torch.arange(len(ytr)), batch_size=256, shuffle=True)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = nn.BCELoss()
    best_auc, best_state = -1.0, None
    for ep in range(1, args.epochs + 1):
        head.train()
        for idx in idx_loader:
            idx = idx.to(dev)
            loss = bce(head(Xtr_d[idx]), ytr_d[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            pv = head(Xva_d).cpu()
        auc = auroc(pv, yva)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            print(f"[ep {ep:>2}] loss={loss.item():.4f} val_auroc={auc:.4f}")

    head.load_state_dict(best_state); head.eval()
    with torch.no_grad():
        pv = head(Xva_d).cpu(); pte = head(Xte.to(dev)).cpu()
    tau, f1 = pick_tau(pv, yva)
    cv, ct = confusion(pv, yva, tau), confusion(pte, yte, tau)
    test_auc = auroc(pte, yte)
    print(f"\n[best] val_auroc={best_auc:.4f}  tau*={tau:.4f} (F1={f1:.4f})")
    print(f"[valid] auroc={best_auc:.4f}  disease recall={cv['recall']:.4f}  healthy reject={cv['reject']:.4f}")
    print(f"[test ] auroc={test_auc:.4f}  disease recall={ct['recall']:.4f}  healthy reject={ct['reject']:.4f}"
          f"  (TP={ct['tp']} FN={ct['fn']} TN={ct['tn']} FP={ct['fp']})")
    # per-source negative reject on the held-out test split (healthy fish vs sashimi vs SalmonScan ...)
    src_te = data["test"]["source"]
    print("[test reject by source (negatives)]")
    negs = sorted({s for s, y in zip(src_te, yte.tolist()) if y == 0})
    for src in negs:
        m = torch.tensor([s == src and y == 0 for s, y in zip(src_te, yte.tolist())])
        if int(m.sum()):
            print(f"    {src:<24} {float((pte[m] < tau).float().mean()):.3f}  "
                  f"({int((pte[m] < tau).sum())}/{int(m.sum())})")

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head_state": best_state, "tau": tau, "dim": dim,
                "val_auroc": best_auc, "test_auroc": test_auc, "feat_layout": layout}, out)
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
