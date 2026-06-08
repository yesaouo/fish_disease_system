"""Train τ as a per-image LESION-SELECTION threshold (not a diseased/healthy head).

The production ThresholdHead is supervised image-level (BCE on max_w ≥ τ(g)), so τ
is only pinned at the single strongest query. Here τ is supervised per-query
against GT-box IoU labels y[300], so it must sit *between* lesion-w and
background-w across ALL queries:

    s_i  = sigmoid(scale·(w_i − τ))          # per-query pass/fail of the bar
    loss = pos_weighted_BCE(s_i, y_i)        # over all 300 queries, every image
    selected = {i : w_i ≥ τ}                 # lesion set; abstain ⟺ empty

Two input variants test the question "does τ need to see the objectness scores":
    g_only  : τ = sigmoid(MLP(g[768]))                  — appearance only
    g_pool  : τ = sigmoid(MLP([g; pool(w)]))            — + perm-invariant order
              pool(w) = [topk_sorted(16), mean, std]      stats of the 300 scores

Eval = SELECTION precision/recall/F1 on diseased images (the metric τ's real job
was never tested on) + healthy false-selection + the old image-abstain AUROC for
back-compat. Baselines: production τ(g) head, and the best FIXED scalar τ*.

Run from repo root (SDM env):
    $PY -m diagnosis_model.grod.train_disease_head_perquery
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

POOL_K = 16


def pool_w(w: torch.Tensor) -> torch.Tensor:
    """w [N,300] -> [N, POOL_K+2] perm-invariant: sorted top-k + mean + std."""
    top = torch.sort(w, dim=1, descending=True).values[:, :POOL_K]
    return torch.cat([top, w.mean(1, keepdim=True), w.std(1, keepdim=True)], dim=1)


class TauHead(nn.Module):
    """τ = sigmoid(τ0_logit + MLP(x)); per-query verdict s = sigmoid(scale·(w − τ)).

    Residual parameterization: the last layer is zero-initialized and τ0_logit is
    set to logit(constant), so at init τ ≡ the tuned constant exactly — the head
    can only learn a residual ON TOP of the constant. With val-F1 model selection
    that includes the init, the learned head provably cannot do worse than the
    constant (it is in the hypothesis space and is the starting point).
    """
    def __init__(self, in_dim, hidden=256, n_hidden=2, tau0=0.5, residual=True):
        super().__init__()
        self.register_buffer("x_mean", torch.zeros(in_dim))
        self.register_buffer("x_std", torch.ones(in_dim))
        self.register_buffer("tau0_logit", torch.logit(torch.tensor(float(tau0))))
        self.residual = residual
        layers, d = [], in_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(d, hidden), nn.ReLU()]; d = hidden
        last = nn.Linear(d, 1)
        if residual:
            nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)   # start AT the constant
        layers += [last]
        self.mlp = nn.Sequential(*layers)
        self.log_scale = nn.Parameter(torch.tensor(2.3))

    def resid(self, x):
        return self.mlp((x - self.x_mean) / self.x_std).squeeze(-1)

    def tau(self, x):
        r = self.resid(x)
        return torch.sigmoid((self.tau0_logit + r) if self.residual else r)

    def verdict(self, x, w):
        t = self.tau(x).unsqueeze(1)                       # [N,1]
        return torch.sigmoid(self.log_scale.exp() * (w - t)), t.squeeze(1)


def selection_metrics(tau, w, y, is_dis):
    """tau [N], w [N,300], y [N,300], is_dis [N] -> dict of micro P/R/F1 etc."""
    sel = (w >= tau.unsqueeze(1)).float()
    dis = is_dis == 1
    TP = (sel[dis] * y[dis]).sum().item()
    FP = (sel[dis] * (1 - y[dis])).sum().item()
    FN = ((1 - sel[dis]) * y[dis]).sum().item()
    P = TP / (TP + FP + 1e-9); R = TP / (TP + FN + 1e-9)
    F1 = 2 * P * R / (P + R + 1e-9)
    # healthy: any selected query = false alarm
    h = ~dis
    hsel = sel[h].sum(1)                                   # selected per healthy img
    healthy_reject = (hsel == 0).float().mean().item() if h.any() else float("nan")
    # image-abstain AUROC (old byproduct metric): score = max_w - tau
    score = (w.max(1).values - tau)
    auroc = _auroc(score, is_dis)
    return dict(P=P, R=R, F1=F1, healthy_reject=healthy_reject, auroc=auroc,
                sel_per_dis=sel[dis].sum(1).mean().item(),
                tau_dis=tau[dis].mean().item(), tau_h=tau[h].mean().item() if h.any() else float("nan"))


def _auroc(score, label):
    s = score.numpy(); y = label.numpy()
    order = s.argsort(); ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(s) + 1)
    npos = (y == 1).sum(); nneg = (y == 0).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def fmt(name, m):
    print(f"  {name:<26} F1={m['F1']:.3f}  P={m['P']:.3f}  R={m['R']:.3f}  "
          f"sel/img={m['sel_per_dis']:.2f}  healthy_rej={m['healthy_reject']:.3f}  "
          f"abstainAUROC={m['auroc']:.4f}  τ̄(dis/h)={m['tau_dis']:.3f}/{m['tau_h']:.3f}")


def train_variant(name, Xtr, wtr, ytr, distr, Xva, wva, yva, disva,
                  pos_weight, epochs, lr, wd, device, tau0=0.5, residual=True, resid_l2=0.0):
    head = TauHead(Xtr.size(1), tau0=tau0, residual=residual).to(device)
    head.x_mean.copy_(Xtr.mean(0).to(device)); head.x_std.copy_(Xtr.std(0).clamp_min(1e-6).to(device))
    Xtr, wtr, ytr = Xtr.to(device), wtr.to(device), ytr.to(device)
    Xva_d, wva_d = Xva.to(device), wva.to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
    N = Xtr.size(0); bs = 512
    pw = torch.tensor(pos_weight, device=device)

    def val_metrics():
        head.eval()
        with torch.no_grad():
            return selection_metrics(head.tau(Xva_d).cpu(), wva, yva, disva)

    # epoch 0 = the constant itself (residual zero-init) — include it in selection
    m0 = val_metrics()
    best_f1 = m0["F1"]; best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    for ep in range(1, epochs + 1):
        head.train()
        perm = torch.randperm(N, device=device)
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            s, _ = head.verdict(Xtr[idx], wtr[idx])
            wt = torch.where(ytr[idx] == 1, pw, torch.ones_like(pw))
            loss = F.binary_cross_entropy(s.clamp(1e-6, 1 - 1e-6), ytr[idx], weight=wt)
            if resid_l2 > 0:                       # keep residual near 0 (near the constant)
                loss = loss + resid_l2 * head.resid(Xtr[idx]).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        m = val_metrics()
        if m["F1"] > best_f1:
            best_f1 = m["F1"]; best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state); head.eval()
    with torch.no_grad():
        tau_va = head.tau(Xva_d).cpu()
    return head, selection_metrics(tau_va, wva, yva, disva)


def main():
    ap = argparse.ArgumentParser()
    ART = "data/processed/current/artifacts"
    ap.add_argument("--cache_dir", default=f"{ART}/db/disease_perquery")
    ap.add_argument("--prod_head", default=f"{ART}/models/disease_head/disease_head.pt")
    ap.add_argument("--pos_weight", type=float, default=100.0)
    ap.add_argument("--resid_l2", type=float, default=1.0, help="L2 on residual (keeps τ near the constant)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr = torch.load(Path(args.cache_dir) / "train.pt", weights_only=False)
    va = torch.load(Path(args.cache_dir) / "val.pt", weights_only=False)
    gtr, wtr, ytr, distr = tr["g"], tr["w"], tr["y"], tr["is_diseased"]
    gva, wva, yva, disva = va["g"], va["w"], va["y"], va["is_diseased"]
    print(f"[data] train N={len(distr)} (dis {int((distr==1).sum())}) "
          f"val N={len(disva)} (dis {int((disva==1).sum())})  pos_weight={args.pos_weight}\n")

    print("=== SELECTION QUALITY on val (diseased) + healthy reject + old abstain AUROC ===")

    # --- baseline 1: production ThresholdHead τ(g) (trained image-level) ---
    from diagnosis_model.grod.disease_head import load_disease_head
    prod, _ = load_disease_head(args.prod_head, device)
    with torch.no_grad():
        tau_prod = prod.tau(gva.to(device)).cpu()
    fmt("BASE prod τ(g) img-level", selection_metrics(tau_prod, wva, yva, disva))

    # --- baseline 2: best FIXED scalar τ* tuned on train selection F1 ---
    grid = torch.linspace(0.01, 0.99, 99)
    f1s = [selection_metrics(torch.full((len(distr),), float(t)), wtr, ytr, distr)["F1"] for t in grid]
    tstar = float(grid[int(np.argmax(f1s))])
    fmt(f"BASE fixed τ*={tstar:.2f}", selection_metrics(torch.full((len(disva),), tstar), wva, yva, disva))

    # --- new per-query heads: plain (non-residual) vs residual-on-constant ---
    Xtr_gp = torch.cat([gtr, pool_w(wtr)], dim=1)
    Xva_gp = torch.cat([gva, pool_w(wva)], dim=1)
    kw = dict(pos_weight=args.pos_weight, epochs=args.epochs, lr=args.lr, wd=args.weight_decay, device=device)

    _, m_g = train_variant("g_only", gtr, wtr, ytr, distr, gva, wva, yva, disva,
                           residual=False, **kw)
    fmt("PERQ τ(g) plain", m_g)
    _, m_gp = train_variant("g_pool", Xtr_gp, wtr, ytr, distr, Xva_gp, wva, yva, disva,
                            residual=False, **kw)
    fmt("PERQ τ(g,pool) plain", m_gp)

    # residual-on-constant: init AT τ*, learn a regularized residual -> provably ≥ constant
    _, m_gr = train_variant("g_only_resid", gtr, wtr, ytr, distr, gva, wva, yva, disva,
                            tau0=tstar, residual=True, resid_l2=args.resid_l2, **kw)
    fmt("PERQ τ(g) resid@τ*", m_gr)
    _, m_gpr = train_variant("g_pool_resid", Xtr_gp, wtr, ytr, distr, Xva_gp, wva, yva, disva,
                             tau0=tstar, residual=True, resid_l2=args.resid_l2, **kw)
    fmt("PERQ τ(g,pool) resid@τ*", m_gpr)

    print("\nREAD: compare SELECTION F1. If PERQ ≫ BASE prod → image-level under-trained τ "
          "for its real job. If g_pool ≫ g_only → τ genuinely needs the objectness scores "
          "(your 768+300 instinct). If fixed τ* ≈ PERQ → image-adaptive τ buys nothing.")


if __name__ == "__main__":
    main()
