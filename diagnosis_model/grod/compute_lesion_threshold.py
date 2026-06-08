"""Compute the manual fixed GROD lesion-selection threshold τ* (the production default).

τ* is the single global objectness cut on w_i = sigmoid(pred_logits[i, 0]) that
best separates GT-matched lesion queries from background, picked by micro
selection-F1 on TRAIN and reported on VAL. The ablation in
train_disease_head_perquery.py shows a learned image-adaptive τ(g) only ties this
constant (per-image optimal τ is unpredictable from the image, held-out R²<0), so
the constant is the principled production gate. This script is its authoritative
derivation; it writes lesion_threshold.json next to the disease head so inference
defaults read one source of truth.

Run from repo root (SDM env):
    $PY -m diagnosis_model.grod.compute_lesion_threshold
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def micro_prf(tau, w, y, is_dis):
    sel = (w >= tau).float()
    d = is_dis == 1
    TP = (sel[d] * y[d]).sum().item()
    FP = (sel[d] * (1 - y[d])).sum().item()
    FN = ((1 - sel[d]) * y[d]).sum().item()
    P = TP / (TP + FP + 1e-9); R = TP / (TP + FN + 1e-9)
    return P, R, 2 * P * R / (P + R + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ART = "data/processed/current/artifacts"
    ap.add_argument("--cache_dir", default=f"{ART}/db/disease_perquery")
    ap.add_argument("--out", default=f"{ART}/models/disease_head/lesion_threshold.json")
    ap.add_argument("--grid", type=int, default=199)
    ap.add_argument("--plateau", type=float, default=0.005, help="F1 tolerance defining the flat band")
    args = ap.parse_args()

    tr = torch.load(Path(args.cache_dir) / "train.pt", weights_only=False)
    va = torch.load(Path(args.cache_dir) / "val.pt", weights_only=False)
    grid = torch.linspace(0.0, 1.0, args.grid + 1)[1:-1]

    f1_tr = np.array([micro_prf(float(t), tr["w"], tr["y"], tr["is_diseased"])[2] for t in grid])
    i_star = int(f1_tr.argmax())
    tau = float(grid[i_star])
    Ptr, Rtr, F1tr = micro_prf(tau, tr["w"], tr["y"], tr["is_diseased"])
    Pva, Rva, F1va = micro_prf(tau, va["w"], va["y"], va["is_diseased"])

    # flat band: contiguous τ range whose train F1 is within `plateau` of the peak
    band = grid[torch.tensor(f1_tr >= f1_tr[i_star] - args.plateau)]
    lo, hi = float(band.min()), float(band.max())

    print(f"τ* (train F1-optimal) = {tau:.3f}")
    print(f"  train: F1={F1tr:.3f} P={Ptr:.3f} R={Rtr:.3f}")
    print(f"  val  : F1={F1va:.3f} P={Pva:.3f} R={Rva:.3f}")
    print(f"  flat band (train F1 within {args.plateau}): [{lo:.3f}, {hi:.3f}] "
          f"— selection is insensitive to τ here")
    print(f"\n  τ      val_F1  val_P  val_R")
    for t in [0.30, 0.40, lo, tau, hi, 0.60, 0.70]:
        p, r, f = micro_prf(float(t), va["w"], va["y"], va["is_diseased"])
        mark = " <- τ*" if abs(t - tau) < 1e-6 else ""
        print(f"  {t:.3f}  {f:.3f}  {p:.3f}  {r:.3f}{mark}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"tau": round(tau, 3),
               "train": {"F1": F1tr, "P": Ptr, "R": Rtr},
               "val": {"F1": F1va, "P": Pva, "R": Rva},
               "flat_band": [round(lo, 3), round(hi, 3)],
               "note": "GROD objectness lesion-selection threshold; "
                       "applies to sigmoid(pred_logits[:,0]). Not the base RF-DETR det_thresh."},
              open(out, "w"), ensure_ascii=False, indent=2)
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
