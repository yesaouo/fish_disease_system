"""Train the GROD-side disease (abstain) head — production ``ThresholdHead``.

The head predicts a per-image objectness threshold τ(g) from the whole-fish
global appearance; the verdict is defined through the detector's own objectness:

    w_i = sigmoid(pred_logits[i, 0])         # per-query objectness
    p   = sigmoid(scale · (max_i w_i − τ(g)))  # diseased ⟺ max_w ≥ τ(g)

so it is consistency-by-construction (never 'diseased' with no lesion, nor
'healthy' while a lesion clears the bar). See disease_head.py for the ablation
that selected this form + the sweep-best τ-MLP defaults (hidden=256, n_hidden=2,
lr=3e-4, wd=1e-3, 50 ep). compare_disease_head_feats.py / sweep_disease_head.py
hold the full layout × head-type sweep.

Frozen-probe training (GROD frozen, only the τ-MLP learns):
  positives = case_db diseased images (run through GROD)   label 1
  negatives = data/healthy_images (+ any non-fish dropped in there)  label 0
Both sources go through the *same* GROD path so train == inference distribution.
The slow GROD pass caches the rich ``g | logits[300]`` feature (1068-d, shared
with the ablation script); training derives g[768] + max_w from it and is instant.

Run from repo root:
    $PY -m diagnosis_model.grod.train_disease_head
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from diagnosis_model.grod.disease_head import ThresholdHead

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# GROD feature extractor (frozen)
# ---------------------------------------------------------------------------

class GrodFeatures:
    def __init__(self, joint_ckpt, global_sd, anchors, device="cuda"):
        self.dev = device
        os.environ["RFDETR_SEMANTIC_DIM"] = "768"
        os.environ["RFDETR_SEMANTIC_ANCHORS"] = os.path.abspath(anchors)
        os.environ["RFDETR_GLOBAL_DIM"] = "768"
        from diagnosis_model.grod.build import load_oavle
        self.net, self.res, self.means, self.stds = load_oavle(joint_ckpt, device=device)
        self.net.global_embed.load_state_dict(torch.load(global_sd, map_location=device))

    @torch.no_grad()
    def batch_feat(self, px: torch.Tensor) -> torch.Tensor:
        """px: [B,3,res,res] -> feat [B, 768+Q]."""
        out = self.net(px.to(self.dev))
        logits = out["pred_logits"][..., 0]                  # [B, Q] ABNORMAL objectness (col 0; col 1 unused)
        g = out["pred_global"]                               # [B, 768]
        feat = torch.cat([g, logits], dim=-1)                # [B, 768+Q]
        return feat.float().cpu()


class ImgDataset(Dataset):
    def __init__(self, paths, res, means, stds):
        self.paths, self.res, self.means, self.stds = paths, res, means, stds

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        px = TF.normalize(TF.resize(TF.to_tensor(img), [self.res, self.res]),
                          self.means, self.stds)
        return px


# ---------------------------------------------------------------------------
# Sample gathering
# ---------------------------------------------------------------------------

def gather_samples(case_db_dir, img_root, healthy_dir, val_frac, seed):
    """Return dict split -> (paths[list], labels[list])."""
    img_root = Path(img_root)
    samples = {"train": ([], []), "val": ([], [])}

    # positives: case_db diseased images, reuse the existing train/valid split
    for fname, split in [("train_cases.pt", "train"), ("valid_cases.pt", "val")]:
        cases = torch.load(Path(case_db_dir) / fname, weights_only=False)
        coco_split = "train" if split == "train" else "valid"
        miss = 0
        for c in cases:
            p = img_root / coco_split / c["file_name"]
            if p.exists():
                samples[split][0].append(str(p)); samples[split][1].append(1)
            else:
                miss += 1
        if miss:
            print(f"[warn] {miss} positive images missing for split={split}")

    # negatives: healthy_images, shuffle and split by val_frac
    healthy = sorted(p for p in Path(healthy_dir).iterdir()
                     if p.suffix.lower() in IMG_EXTS)
    random.Random(seed).shuffle(healthy)
    n_val = int(len(healthy) * val_frac)
    for p in healthy[:n_val]:
        samples["val"][0].append(str(p)); samples["val"][1].append(0)
    for p in healthy[n_val:]:
        samples["train"][0].append(str(p)); samples["train"][1].append(0)

    for s in ("train", "val"):
        y = torch.tensor(samples[s][1])
        print(f"[samples] {s}: pos={int((y==1).sum())} neg={int((y==0).sum())} total={len(y)}")
    return samples


def extract_features(grod, paths, batch_size, workers):
    ds = ImgDataset(paths, grod.res, grod.means, grod.stds)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    feats = []
    for bi, px in enumerate(loader):
        feats.append(grod.batch_feat(px))
        if (bi + 1) % 50 == 0:
            print(f"  ...{(bi+1)*batch_size}/{len(paths)}")
    return torch.cat(feats, dim=0)


def build_features(grod, samples, cache_path, batch_size, workers, recompute):
    if cache_path.exists() and not recompute:
        print(f"[cache] loading {cache_path}")
        return torch.load(cache_path, weights_only=False)
    data = {}
    for s in ("train", "val"):
        paths, labels = samples[s]
        print(f"[extract] {s}: {len(paths)} images")
        X = extract_features(grod, paths, batch_size, workers)
        data[s] = {"X": X, "y": torch.tensor(labels, dtype=torch.float32)}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache_path)
    print(f"[cache] saved {cache_path}")
    return data


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def auroc(scores, labels):
    # rank-based AUROC
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float32)
    n_pos = (labels == 1).sum().item()
    n_neg = (labels == 0).sum().item()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_pos = ranks[labels == 1].sum().item()
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def pick_tau(scores, labels):
    """Pick threshold maximizing F1; also report at recall>=0.98."""
    s = scores.numpy(); y = labels.numpy()
    best_f1, best_tau = -1.0, 0.5
    for t in sorted(set(s.tolist())):
        pred = s >= t
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        if f1 > best_f1:
            best_f1, best_tau = f1, float(t)
    return best_tau, best_f1


def main():
    ap = argparse.ArgumentParser()
    ART = "data/processed/current/artifacts"
    ap.add_argument("--joint_ckpt", default=f"{ART}/models/joint_rfdetr/checkpoint_best_regular.pth")
    ap.add_argument("--global_sd", default=f"{ART}/models/distilled_global_rawP/global_embed_state_dict.pt")
    ap.add_argument("--anchors", default=f"{ART}/models/text_anchors.pt")
    ap.add_argument("--case_db_dir", default=f"{ART}/db/case_db_jointDistRawP")
    ap.add_argument("--img_root", default="data/processed/current/detection")
    ap.add_argument("--healthy_dir", default="data/healthy_images")
    ap.add_argument("--out_dir", default=f"{ART}/models/disease_head")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--n_hidden", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recompute", action="store_true", help="force re-extract features")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: cap samples per class")
    args = ap.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    samples = gather_samples(args.case_db_dir, args.img_root, args.healthy_dir,
                             args.val_frac, args.seed)
    if args.limit:
        for s in ("train", "val"):
            P, Y = samples[s]
            pos = [(p, y) for p, y in zip(P, Y) if y == 1][:args.limit]
            neg = [(p, y) for p, y in zip(P, Y) if y == 0][:args.limit]
            mix = pos + neg
            samples[s] = ([p for p, _ in mix], [y for _, y in mix])
        print(f"[limit] capped to {args.limit}/class -> "
              f"train {len(samples['train'][0])} val {len(samples['val'][0])}")

    grod = GrodFeatures(args.joint_ckpt, args.global_sd, args.anchors, dev)
    cache = out_dir / ("features_smoke.pt" if args.limit else "features.pt")
    data = build_features(grod, samples, cache, args.batch_size, args.workers, args.recompute)

    Xtr, ytr = data["train"]["X"], data["train"]["y"]
    Xva, yva = data["val"]["X"], data["val"]["y"]

    # threshold head: τ(g) vs max_w, both derived from the cached g|logits feature
    g_tr, mw_tr = Xtr[:, :768], Xtr[:, 768:].sigmoid().amax(1)
    g_va, mw_va = Xva[:, :768], Xva[:, 768:].sigmoid().amax(1)
    head = ThresholdHead(g_tr.size(1), hidden=args.hidden,
                         n_hidden=args.n_hidden, dropout=args.dropout).to(dev)
    head.g_mean.copy_(g_tr.mean(0).to(dev)); head.g_std.copy_(g_tr.std(0).clamp_min(1e-6).to(dev))

    g_tr_d, mw_tr_d, ytr_d = g_tr.to(dev), mw_tr.to(dev), ytr.to(dev)
    g_va_d, mw_va_d = g_va.to(dev), mw_va.to(dev)

    # balanced sampling: weight inversely to class frequency
    cnt = torch.bincount(ytr.long(), minlength=2).float()
    w_per = (1.0 / cnt.clamp_min(1))[ytr.long()]
    sampler = WeightedRandomSampler(w_per, num_samples=len(ytr), replacement=True)
    idx_loader = DataLoader(torch.arange(len(ytr)), batch_size=256, sampler=sampler)

    opt = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = nn.BCELoss()
    best_auc, best_state = -1.0, None
    for ep in range(1, args.epochs + 1):
        head.train()
        for idx in idx_loader:
            idx = idx.to(dev)
            p, _ = head(g_tr_d[idx], mw_tr_d[idx])
            loss = bce(p, ytr_d[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            pv, _ = head(g_va_d, mw_va_d)
        pv = pv.cpu()
        auc = auroc(pv, yva)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            acc = ((pv >= 0.5).float() == yva).float().mean().item()
            print(f"[ep {ep:>2}] loss={loss.item():.4f} val_auroc={auc:.4f} val_acc@0.5={acc:.4f}")

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        pv, _ = head(g_va_d, mw_va_d)
    pv = pv.cpu()
    tau, f1 = pick_tau(pv, yva)
    pred = pv >= tau
    tp = int(((pred == 1) & (yva == 1)).sum()); fn = int(((pred == 0) & (yva == 1)).sum())
    fp = int(((pred == 1) & (yva == 0)).sum()); tn = int(((pred == 0) & (yva == 0)).sum())
    print(f"\n[best] val_auroc={best_auc:.4f}  tau*={tau:.4f} (F1={f1:.4f})")
    print(f"[confusion @tau*] disease TP={tp} FN={fn} | healthy TN={tn} FP={fp}")
    print(f"  disease recall={tp/(tp+fn+1e-9):.4f}  healthy reject rate={tn/(tn+fp+1e-9):.4f}")

    ckpt = out_dir / "disease_head.pt"
    torch.save({"head_state": best_state, "tau": tau,
                "head_cfg": dict(gdim=int(g_tr.size(1)), hidden=args.hidden,
                                 n_hidden=args.n_hidden, dropout=args.dropout),
                "val_auroc": best_auc,
                "feat_layout": "threshold: p=sigmoid(scale·(max_w − τ(g[768])))"},
               ckpt)
    print(f"[save] {ckpt}")


if __name__ == "__main__":
    main()
