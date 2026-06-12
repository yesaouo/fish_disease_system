"""Calibrate the two decoupled GROD objectness thresholds from the dataset.

Two thresholds with different jobs (see grod/LESION_GATE.md, the selection-vs-
abstain decoupling):

  abstain_thresh : image-level "is this fish diseased?" — pick the cut on per-image
                   max objectness that maximizes Youden's J (sens+spec−1) over
                   healthy vs diseased images.
  display_thresh : per-query "which lesion boxes to show" — recall-leaning, pick
                   the F2-optimal selection cut (β=2 weights recall 2×) on
                   GT-matched vs background queries of diseased images.

Both are dataset-calibrated CONSTANTS, not per-image learned (a learned per-image
τ ties the constant — held-out R²<0; see train_disease_head_perquery.py). Writes
data/processed/current/thresholds.json; the demo reads it for its slider defaults.

Run from repo root (SDM env):
    $PY -m diagnosis_model.grod.calibrate_thresholds
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def micro_prf(tau, w, y, is_dis):
    """Per-query selection P/R/F on diseased images."""
    sel = (w >= tau).float()
    d = is_dis == 1
    TP = (sel[d] * y[d]).sum().item()
    FP = (sel[d] * (1 - y[d])).sum().item()
    FN = ((1 - sel[d]) * y[d]).sum().item()
    P = TP / (TP + FP + 1e-9); R = TP / (TP + FN + 1e-9)
    return P, R


def fbeta(P, R, beta):
    b2 = beta * beta
    return (1 + b2) * P * R / (b2 * P + R + 1e-9)


def image_sens_spec(tau, max_w, is_dis):
    """Image-level: diseased iff max_w >= tau."""
    pred = max_w >= tau
    d = is_dis == 1
    sens = (pred & d).sum().item() / (d.sum().item() + 1e-9)
    spec = (~pred & ~d).sum().item() / ((~d).sum().item() + 1e-9)
    return sens, spec


def main():
    ap = argparse.ArgumentParser()
    ART = "data/processed/current/artifacts"
    ap.add_argument("--cache", default=f"{ART}/db/disease_perquery/train.pt")
    ap.add_argument("--out", default="data/processed/current/thresholds.json")
    ap.add_argument("--beta", type=float, default=2.0, help="display Fβ recall weight")
    ap.add_argument("--grid", type=int, default=199)
    args = ap.parse_args()

    d = torch.load(args.cache, weights_only=False)
    w, y, is_dis = d["w"], d["y"], d["is_diseased"]
    max_w = w.max(1).values
    grid = torch.linspace(0.0, 1.0, args.grid + 1)[1:-1]

    # display_thresh: F2-optimal per-query selection (recall-leaning)
    prf = [micro_prf(float(t), w, y, is_dis) for t in grid]
    f2 = np.array([fbeta(P, R, args.beta) for P, R in prf])
    di = int(f2.argmax()); disp = float(grid[di]); dP, dR = prf[di]

    # abstain_thresh: Youden-optimal image-level (max_w, healthy vs diseased)
    ss = [image_sens_spec(float(t), max_w, is_dis) for t in grid]
    j = np.array([s + sp - 1 for s, sp in ss])
    ai = int(j.argmax()); abst = float(grid[ai]); aSe, aSp = ss[ai]

    out = {
        "abstain_thresh": round(abst, 3),
        "display_thresh": round(disp, 3),
        "abstain": {"sens": round(aSe, 4), "spec": round(aSp, 4), "youden_J": round(float(j[ai]), 4)},
        "display": {"P": round(dP, 4), "R": round(dR, 4), f"F{args.beta:g}": round(float(f2[di]), 4)},
        "note": "GROD objectness thresholds, calibrated on disease_perquery train. "
                "abstain=Youden image-level max_w; display=F{beta} per-query selection. "
                "Dataset-calibrated constants (not per-image learned).".replace("{beta}", f"{args.beta:g}"),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"abstain_thresh = {abst:.3f}  (sens {aSe:.3f} / spec {aSp:.3f})")
    print(f"display_thresh = {disp:.3f}  (P {dP:.3f} / R {dR:.3f} / F{args.beta:g} {f2[di]:.3f})")
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
