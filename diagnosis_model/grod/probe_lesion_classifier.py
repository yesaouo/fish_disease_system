"""Can a per-query lesion CLASSIFIER beat the constant objectness threshold?

The constant τ on objectness caps at val selection F1 ≈ 0.806 because ~22.5% of
diseased images have a background query whose objectness out-fires the weakest
lesion — unfixable by ANY objectness threshold. But those background queries may
be separable on SEMANTICS: high objectness yet z that snaps onto no symptom
anchor. So instead of "threshold on objectness", classify each query as
lesion/background from [objectness + centered z·anchor saliency + box area].

Centered saliency is the anisotropy-corrected signal (raw z·anchor is dead,
d=0.10; centered d=2.65; ρ≈0.1 with objectness — a genuinely independent cue).

Eval = micro selection P/R/F1 on diseased val queries; classifier threshold tuned
on train F1, exactly like the constant. Also reports F1 on the 22.5%
objectness-overlap subset where the constant provably cannot win.

Run from repo root (SDM env):
    $PY -m diagnosis_model.grod.probe_lesion_classifier
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def micro_f1_from_pred(pred, y, dis):
    sel = pred[dis == 1]; yy = y[dis == 1]
    TP = (sel * yy).sum().item(); FP = (sel * (1 - yy)).sum().item(); FN = ((1 - sel) * yy).sum().item()
    P = TP / (TP + FP + 1e-9); R = TP / (TP + FN + 1e-9)
    return P, R, 2 * P * R / (P + R + 1e-9)


def overlap_mask(w, y, dis):
    """Per diseased image: True if a background query out-fires the weakest lesion
    (the subset where a constant objectness threshold provably cannot separate)."""
    m = torch.zeros(len(w), dtype=torch.bool)
    for i in range(len(w)):
        if dis[i] != 1 or y[i].sum() == 0:
            continue
        les = w[i][y[i] == 1]; bg = w[i][y[i] == 0]
        if bg.max() >= les.min():
            m[i] = True
    return m


class Clf(nn.Module):
    def __init__(self, d, hidden=0):
        super().__init__()
        self.register_buffer("mu", torch.zeros(d)); self.register_buffer("sd", torch.ones(d))
        self.net = nn.Linear(d, 1) if hidden == 0 else nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net((x - self.mu) / self.sd).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ART = "data/processed/current/artifacts"
    ap.add_argument("--cache_dir", default=f"{ART}/db/disease_perquery")
    ap.add_argument("--pos_weight", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr = torch.load(Path(args.cache_dir) / "train.pt", weights_only=False)
    va = torch.load(Path(args.cache_dir) / "val.pt", weights_only=False)
    names = tr["qfeat_names"]                      # [objectness, sal_maxcos, sal_margin, sal_softmax, box_area]
    QFtr, ytr_q, distr = tr["qfeat"], tr["y"], tr["is_diseased"]
    QFva, yva_q, disva = va["qfeat"], va["y"], va["is_diseased"]
    wva = va["w"]
    print(f"[data] feats={names}  train imgs={len(distr)} val imgs={len(disva)}")

    ov = overlap_mask(wva, yva_q, disva)
    print(f"[overlap] {int(ov.sum())}/{int((disva==1).sum())} diseased-val imgs have objectness overlap "
          f"(constant ceiling lives here)\n")

    # --- baseline: constant on objectness (tune on train) ---
    obj_idx = names.index("objectness")
    def const_score(split_qf):
        return split_qf[..., obj_idx]
    t = _tune_const(const_score(QFtr), ytr_q, distr)
    P, R, Fc = micro_f1_from_pred((const_score(QFva) >= t).float(), yva_q, disva)
    Po, Ro, Fco = _subset_f1((const_score(QFva) >= t).float(), yva_q, disva, ov)
    print(f"{'BASE const objectness':<34} F1={Fc:.3f} P={P:.3f} R={R:.3f} | overlap-F1={Fco:.3f}")

    # --- classifiers on growing feature sets ---
    feat_sets = {
        "obj only (logreg)":              ["objectness"],
        "obj+saliency (logreg)":          ["objectness", "sal_maxcos", "sal_margin", "sal_softmax"],
        "obj+saliency+box (logreg)":      names,
        "obj+saliency+box (MLP-64)":      names,
    }
    flat = lambda QF, cols: QF[..., [names.index(c) for c in cols]].reshape(-1, len(cols))
    for label, cols in feat_sets.items():
        hidden = 64 if "MLP" in label else 0
        Xtr = flat(QFtr, cols); Xva = flat(QFva, cols)
        torch.manual_seed(args.seed)
        clf = Clf(Xtr.size(1), hidden).to(device)
        clf.mu.copy_(Xtr.mean(0).to(device)); clf.sd.copy_(Xtr.std(0).clamp_min(1e-6).to(device))
        opt = torch.optim.Adam(clf.parameters(), lr=3e-3, weight_decay=1e-4)
        pw = torch.tensor(args.pos_weight, device=device)
        Xtr_d, ytr_d = Xtr.to(device), ytr_q.reshape(-1).to(device)
        n = Xtr_d.size(0); bs = 8192
        for ep in range(120):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                loss = F.binary_cross_entropy_with_logits(clf(Xtr_d[idx]), ytr_d[idx], pos_weight=pw)
                opt.zero_grad(); loss.backward(); opt.step()
        clf.eval()
        with torch.no_grad():
            sc = torch.sigmoid(clf(Xva.to(device))).cpu().reshape(QFva.shape[:2])
            sctr = torch.sigmoid(clf(Xtr.to(device))).cpu().reshape(QFtr.shape[:2])
        t = _tune_const(sctr, ytr_q, distr)
        P, R, Ff = micro_f1_from_pred((sc >= t).float(), yva_q, disva)
        _, _, Ffo = _subset_f1((sc >= t).float(), yva_q, disva, ov)
        print(f"{label:<34} F1={Ff:.3f} P={P:.3f} R={R:.3f} | overlap-F1={Ffo:.3f}")

    print(f"\nREAD: beat BASE const F1 (even slightly) ⇒ semantics adds selection signal "
          f"beyond objectness. Watch overlap-F1 — that is where the constant is provably stuck.")


def _tune_const(score, y, dis):
    grid = np.linspace(0.0, 1.0, 201)[1:-1]      # dense, matches the constant-τ sweep
    best_t, best_f1 = 0.5, -1
    for t in grid:
        _, _, f = micro_f1_from_pred((score >= t).float(), y, dis)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_t


def _subset_f1(pred, y, dis, subset_mask):
    keep = (dis == 1) & subset_mask
    sel = pred[keep]; yy = y[keep]
    TP = (sel * yy).sum().item(); FP = (sel * (1 - yy)).sum().item(); FN = ((1 - sel) * yy).sum().item()
    P = TP / (TP + FP + 1e-9); R = TP / (TP + FN + 1e-9)
    return P, R, 2 * P * R / (P + R + 1e-9)


if __name__ == "__main__":
    main()
