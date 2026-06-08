"""Hyperparameter sweep for the thresh τ(g) disease head — chase the highest AUROC.

Loads the cached 1068-dim features (g | raw logits[300]) built by
compare_disease_head_feats.py, derives τ-input = g[768] and max_w, then grids over
the τ-MLP capacity / regularization / optimization. Each config is averaged over
N seeds; reports the top configs by mean val AUROC (with std), plus the current
production-candidate baseline (hidden=256, n_hidden=1, lr=1e-3, wd=1e-4) for
reference. Verdict is consistency-by-construction: diseased ⟺ max_w ≥ τ(g).

Run from repo root (features must already be cached — run compare first):
    $PY -m diagnosis_model.grod.sweep_disease_head
"""

from __future__ import annotations

import argparse
import itertools
import statistics
from pathlib import Path

import torch

from diagnosis_model.grod.compare_disease_head_feats import (
    split_g_maxw, train_eval_threshold,
)


def main():
    ap = argparse.ArgumentParser()
    ART = "data/processed/current/artifacts"
    ap.add_argument("--cache", default=f"{ART}/models/disease_head/features_cmp.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.load(Path(args.cache), weights_only=False)
    Xtr, ytr = data["train"]["X"], data["train"]["y"]
    Xva, yva = data["val"]["X"], data["val"]["y"]
    g_tr, mw_tr = split_g_maxw(Xtr); g_va, mw_va = split_g_maxw(Xva)
    print(f"[data] train={len(ytr)} val={len(yva)} (pos={int((yva==1).sum())} "
          f"neg={int((yva==0).sum())})  seeds={args.seeds} epochs={args.epochs}")

    grid = dict(
        hidden=[128, 256, 512],
        n_hidden=[1, 2],
        dropout=[0.0, 0.1],
        lr=[1e-3, 3e-4],
        weight_decay=[1e-4, 1e-3],
    )
    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    variants = {"τ(g) [768]": (g_tr, g_va), "τ(g+logits) [1068]": (Xtr, Xva)}
    print(f"[sweep] {len(combos)} configs × {len(args.seeds)} seeds × {len(variants)} τ-inputs\n")

    def fmt(r):
        auc, std, rec, rej, cfg = r
        c = f"h{cfg['hidden']} L{cfg['n_hidden']} d{cfg['dropout']} lr{cfg['lr']:.0e} wd{cfg['weight_decay']:.0e}"
        return f"  AUROC {auc:.4f} ±{std:.4f}  recall {rec:.4f}  reject {rej:.4f}  | {c}"

    best_per_variant = {}
    for vname, (Xt, Xv) in variants.items():
        rows = []
        for vals in combos:
            cfg = dict(zip(keys, vals))
            hk = dict(hidden=cfg["hidden"], n_hidden=cfg["n_hidden"], dropout=cfg["dropout"])
            aucs, recs, rejs = [], [], []
            for s in args.seeds:
                auc, _, _, conf, _ = train_eval_threshold(
                    Xt, mw_tr, ytr, Xv, mw_va, yva, args.epochs, cfg["lr"], s, dev,
                    weight_decay=cfg["weight_decay"], head_kwargs=hk)
                aucs.append(auc); recs.append(conf["recall"]); rejs.append(conf["reject"])
            rows.append((statistics.mean(aucs), statistics.pstdev(aucs),
                         statistics.mean(recs), statistics.mean(rejs), cfg))
        rows.sort(key=lambda r: r[0], reverse=True)
        best_per_variant[vname] = rows[0]
        print(f"=== {vname} — TOP 5 by mean val AUROC ===")
        for r in rows[:5]:
            print(fmt(r))
        print()

    print("=== best per τ-input ===")
    for vname, r in best_per_variant.items():
        print(f"[{vname}]\n{fmt(r)}")


if __name__ == "__main__":
    main()
