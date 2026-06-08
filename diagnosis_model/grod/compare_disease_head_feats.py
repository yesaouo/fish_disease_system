"""One-off: compare disease-head accuracy across two feat layouts on identical data.

  770  = concat(g[768], max_i sigmoid(logit_i), Σ_i sigmoid(logit_i))   # old pooled
  1068 = concat(g[768], pred_logits[:,0] [Q=300])                       # new full logits

Both are derived from the SAME GROD forward (1068 features extracted/cached, 770
reconstructed from them), and trained with the same loop / split / seed / sampler,
so the only variable is the feature layout. Reports val AUROC + the abstain
operating point (disease recall / healthy reject) for each.

Run from repo root:
    $PY diagnosis_model/grod/compare_disease_head_feats.py
"""

from __future__ import annotations

import argparse
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from diagnosis_model.grod.disease_head import DiseaseHead
from diagnosis_model.grod.train_disease_head import (
    GrodFeatures, auroc, build_features, gather_samples, pick_tau,
)


class MLPHead(nn.Module):
    """Nonlinear free head: standardize -> Linear -> ReLU -> Linear -> sigmoid.

    Unlike DiseaseHead (single Linear, can only do weighted sums), this can
    approximate max over raw logit slots — the test of whether 1068's loss is
    'linear can't compute max' (then MLP recovers) vs 'irreducible noise dims'.
    """

    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.register_buffer("feat_mean", torch.zeros(dim))
        self.register_buffer("feat_std", torch.ones(dim))
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = (feat - self.feat_mean) / self.feat_std
        return torch.sigmoid(self.net(x).squeeze(-1))


def derive_770(X: torch.Tensor) -> torch.Tensor:
    """1068 [g | raw logits[300]] -> 770 [g | max sigmoid | sum sigmoid]."""
    g, logits = X[:, :768], X[:, 768:]
    w = logits.sigmoid()
    return torch.cat([g, w.amax(1, keepdim=True), w.sum(1, keepdim=True)], dim=-1)


def split_g_maxw(X: torch.Tensor):
    """1068 [g | raw logits[300]] -> (g[768], max_i sigmoid(logit_i))."""
    return X[:, :768], X[:, 768:].sigmoid().amax(1)


class ThresholdHead(nn.Module):
    """Predict a per-image objectness threshold τ(g) ∈ (0,1); verdict = max_w ≥ τ.

        p = sigmoid( scale · (max_w − τ(g)) )

    Consistency-by-construction abstain: the verdict is *defined* through the
    detector's own max objectness, so "diseased ⟺ a query clears τ" — it can
    never disagree with the lesion evidence. The same τ(g) can double as the
    adaptive per-image det_thresh downstream.
    """

    def __init__(self, in_dim: int = 768, hidden: int = 256,
                 n_hidden: int = 1, dropout: float = 0.0):
        super().__init__()
        self.register_buffer("x_mean", torch.zeros(in_dim))
        self.register_buffer("x_std", torch.ones(in_dim))
        layers, d = [], in_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.mlp = nn.Sequential(*layers)
        self.log_scale = nn.Parameter(torch.tensor(2.3))   # exp ≈ 10

    def tau(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp((x - self.x_mean) / self.x_std).squeeze(-1))

    def forward(self, x: torch.Tensor, max_w: torch.Tensor):
        tau = self.tau(x)
        return torch.sigmoid(self.log_scale.exp() * (max_w - tau)), tau


def train_eval_threshold(Xtau_tr, mtr, ytr, Xtau_va, mva, yva, epochs, lr, seed, dev,
                         weight_decay=1e-4, head_kwargs=None):
    """Train ThresholdHead: τ(Xtau) vs max_w. Return (best_auroc, tau_op, f1, conf, tau_stats)."""
    torch.manual_seed(seed)
    head = ThresholdHead(Xtau_tr.size(1), **(head_kwargs or {})).to(dev)
    head.x_mean.copy_(Xtau_tr.mean(0).to(dev)); head.x_std.copy_(Xtau_tr.std(0).clamp_min(1e-6).to(dev))

    gtr_d, mtr_d, ytr_d = Xtau_tr.to(dev), mtr.to(dev), ytr.to(dev)
    gva_d, mva_d = Xtau_va.to(dev), mva.to(dev)
    cnt = torch.bincount(ytr.long(), minlength=2).float()
    w_per = (1.0 / cnt.clamp_min(1))[ytr.long()]
    sampler = WeightedRandomSampler(w_per, num_samples=len(ytr), replacement=True)
    idx_loader = DataLoader(torch.arange(len(ytr)), batch_size=256, sampler=sampler)

    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCELoss()
    best_auc, best_state = -1.0, None
    for _ in range(epochs):
        head.train()
        for idx in idx_loader:
            idx = idx.to(dev)
            p, _ = head(gtr_d[idx], mtr_d[idx])
            loss = bce(p, ytr_d[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            pv, _ = head(gva_d, mva_d)
        auc = auroc(pv.cpu(), yva)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    head.load_state_dict(best_state); head.eval()
    with torch.no_grad():
        pv, tva = head(gva_d, mva_d)
    pv, tva = pv.cpu(), tva.cpu()
    tau_op, f1 = pick_tau(pv, yva)
    pred = pv >= tau_op
    tp = int(((pred == 1) & (yva == 1)).sum()); fn = int(((pred == 0) & (yva == 1)).sum())
    fp = int(((pred == 1) & (yva == 0)).sum()); tn = int(((pred == 0) & (yva == 0)).sum())
    conf = dict(tp=tp, fn=fn, fp=fp, tn=tn,
                recall=tp / (tp + fn + 1e-9), reject=tn / (tn + fp + 1e-9))
    tau_stats = (float(tva[yva == 1].mean()), float(tva[yva == 0].mean()),
                 float(mva[yva == 1].mean()), float(mva[yva == 0].mean()))
    return best_auc, tau_op, f1, conf, tau_stats


def train_eval(Xtr, ytr, Xva, yva, epochs, lr, seed, dev, kind="linear"):
    """Train a free head (linear DiseaseHead or MLPHead) on the given layout."""
    torch.manual_seed(seed)
    mean = Xtr.mean(0); std = Xtr.std(0).clamp_min(1e-6)
    head = (MLPHead(Xtr.size(1)) if kind == "mlp" else DiseaseHead(Xtr.size(1))).to(dev)
    head.feat_mean.copy_(mean.to(dev)); head.feat_std.copy_(std.to(dev))

    Xtr_d, ytr_d, Xva_d = Xtr.to(dev), ytr.to(dev), Xva.to(dev)
    cnt = torch.bincount(ytr.long(), minlength=2).float()
    w_per = (1.0 / cnt.clamp_min(1))[ytr.long()]
    sampler = WeightedRandomSampler(w_per, num_samples=len(ytr), replacement=True)
    idx_loader = DataLoader(torch.arange(len(ytr)), batch_size=256, sampler=sampler)

    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCELoss()
    best_auc, best_state = -1.0, None
    for _ in range(epochs):
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

    head.load_state_dict(best_state); head.eval()
    with torch.no_grad():
        pv = head(Xva_d).cpu()
    tau, f1 = pick_tau(pv, yva)
    pred = pv >= tau
    tp = int(((pred == 1) & (yva == 1)).sum()); fn = int(((pred == 0) & (yva == 1)).sum())
    fp = int(((pred == 1) & (yva == 0)).sum()); tn = int(((pred == 0) & (yva == 0)).sum())
    conf = dict(tp=tp, fn=fn, fp=fp, tn=tn,
                recall=tp / (tp + fn + 1e-9), reject=tn / (tn + fp + 1e-9))
    return best_auc, tau, f1, conf


def main():
    ap = argparse.ArgumentParser()
    ART = "data/processed/current/artifacts"
    ap.add_argument("--joint_ckpt", default=f"{ART}/models/joint_rfdetr/checkpoint_best_regular.pth")
    ap.add_argument("--global_sd", default=f"{ART}/models/distilled_global_rawP/global_embed_state_dict.pt")
    ap.add_argument("--anchors", default=f"{ART}/models/text_anchors.pt")
    ap.add_argument("--case_db_dir", default=f"{ART}/db/case_db_jointDistRawP")
    ap.add_argument("--img_root", default="data/processed/current/detection")
    ap.add_argument("--healthy_dir", default="data/healthy_images")
    ap.add_argument("--cache", default=f"{ART}/models/disease_head/features_cmp.pt",
                    help="1068-dim feature cache (g | raw logits).")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()

    from pathlib import Path
    torch.manual_seed(args.seed); random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    samples = gather_samples(args.case_db_dir, args.img_root, args.healthy_dir,
                             args.val_frac, args.seed)
    grod = GrodFeatures(args.joint_ckpt, args.global_sd, args.anchors, dev)
    data = build_features(grod, samples, Path(args.cache),
                          args.batch_size, args.workers, args.recompute)

    Xtr, ytr = data["train"]["X"], data["train"]["y"]
    Xva, yva = data["val"]["X"], data["val"]["y"]
    assert Xtr.size(1) == 1068, f"expected 1068-dim cache, got {Xtr.size(1)} (use --recompute)"

    layouts = {
        "768  (g only)":              (Xtr[:, :768], Xva[:, :768]),
        "770  (g | max,sum sigmoid)": (derive_770(Xtr), derive_770(Xva)),
        "1068 (g | logits[300])":     (Xtr, Xva),
    }
    results = {}
    for kind in ("linear", "mlp"):
        for name, (Xt, Xv) in layouts.items():
            auc, tau, f1, conf = train_eval(Xt, ytr, Xv, yva, args.epochs, args.lr,
                                            args.seed, dev, kind=kind)
            results[f"{name}  [{kind}]"] = (auc, tau, f1, conf)

    # threshold-prediction heads: τ(·) vs max_w (consistency-by-construction)
    g_tr, mw_tr = split_g_maxw(Xtr); g_va, mw_va = split_g_maxw(Xva)
    thresh = {
        "thresh τ(g)    vs max_w":    (g_tr, g_va),
        "thresh τ(g+logits) vs max_w": (Xtr, Xva),
    }
    thresh_res = {
        name: train_eval_threshold(Xt, mw_tr, ytr, Xv, mw_va, yva,
                                   args.epochs, args.lr, args.seed, dev)
        for name, (Xt, Xv) in thresh.items()
    }

    print(f"\n=== disease-head comparison (val: pos={int((yva==1).sum())} "
          f"neg={int((yva==0).sum())}, seed={args.seed}) ===")
    for name, (auc, tau, f1, c) in results.items():
        print(f"\n[{name}]")
        print(f"  val_auroc = {auc:.4f}   tau* = {tau:.4f} (F1={f1:.4f})")
        print(f"  @tau*: disease TP={c['tp']} FN={c['fn']} | healthy TN={c['tn']} FP={c['fp']}")
        print(f"         disease recall = {c['recall']:.4f}   healthy reject = {c['reject']:.4f}")

    for name, (auc, tau, f1, c, st) in thresh_res.items():
        print(f"\n[{name}]")
        print(f"  val_auroc = {auc:.4f}   tau* = {tau:.4f} (F1={f1:.4f})")
        print(f"  @tau*: disease TP={c['tp']} FN={c['fn']} | healthy TN={c['tn']} FP={c['fp']}")
        print(f"         disease recall = {c['recall']:.4f}   healthy reject = {c['reject']:.4f}")
        print(f"  mean τ: disease={st[0]:.3f} healthy={st[1]:.3f} | "
              f"mean max_w: disease={st[2]:.3f} healthy={st[3]:.3f}")


if __name__ == "__main__":
    main()
