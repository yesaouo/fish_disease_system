"""Probe (軸 1 diagnostic): does objectness w decouple from z·anchor saliency?

The soft pipeline gates every region by w_i = sigmoid(objectness_i) — one scalar
doing detection + evidence-weight + abstain. The proposed decoupling reweights
the evidence gate by a *zero-parameter* semantic saliency s_i = "how sharply does
z_i snap onto some symptom anchor". We test whether s carries information beyond
w, and whether it has headroom in the regime that matters (confident detections).

Three findings this script reproduces (valid soft inputs, eval-only):

  1. ANISOTROPY: the 15 SigLIP2 symptom anchors are ~0.97-collinear, so RAW
     max cos(z, anchor) is a dead signal (lesion-vs-bg Cohen's d ≈ 0.10). This is
     why the GROD head needs the /0.07 temperature — it amplifies the tiny margin
     that survives the shared mean direction.
  2. CENTERING REVIVES IT: subtract the anchor centroid (still zero-parameter) and
     max cos(z_c, anchor_c) separates lesion queries from background at d ≈ 2.65,
     while staying ~uncorrelated with objectness (ρ ≈ 0.1) — a genuinely distinct
     signal.
  3. BUT LITTLE HEADROOM: among confident detections (w>0.5, the ones that
     dominate CEAH) centered-s is tight (std ≈ 0.06); only ~4% are
     "confident-but-diffuse" (w>0.5 & s<0.5). Decoupling w·s would reweight that
     4% tail + the weak-w long tail; top-1 evidence flips under w·s in only ~14%
     of cases. Expected effect on headline faithfulness: small / neutral.

Run from repo root (SDM env):
    $PY -m diagnosis_model.grod.probe_objectness_saliency
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def sep(name, smax, w, N):
    """lesion(top-1-w query) vs background mean separation for a saliency map."""
    s_top, s_bg = [], []
    for n in range(N):
        i = int(w[n].argmax())
        if w[n, i] < 0.05:
            continue
        s_top.append(float(smax[n, i]))
        m = torch.ones(smax.size(1), dtype=torch.bool); m[i] = False
        s_bg.append(float(smax[n, m].mean()))
    s_top, s_bg = np.array(s_top), np.array(s_bg)
    dd = s_top - s_bg
    print(f"  {name:<22} lesion={s_top.mean():.3f} bg={s_bg.mean():.3f} "
          f"Δ={dd.mean():+.4f}  d={dd.mean()/dd.std():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ART = "data/processed/current/artifacts"
    ap.add_argument("--soft_valid", default=f"{ART}/db/soft_inputs/valid.pt")
    ap.add_argument("--anchors", default=f"{ART}/models/text_anchors.pt")
    args = ap.parse_args()

    d = torch.load(args.soft_valid, weights_only=False)
    z = d["z_all"].float()                          # [N,Q,768] L2-normed
    w = d["w"].float()                              # [N,Q]
    A = torch.load(args.anchors, weights_only=False)["anchor_embs"].float()
    N, Q, _ = z.shape
    print(f"[probe] N={N} Q={Q} anchors={tuple(A.shape)}")

    # (1) anchor collinearity
    An = F.normalize(A, dim=-1)
    off = (An @ An.T)[~torch.eye(A.size(0), dtype=torch.bool)]
    print(f"\nanchor-anchor cos: mean={off.mean():.3f} min={off.min():.3f} "
          f"max={off.max():.3f}  (high → anisotropic, raw cosine uninformative)")

    # (2) raw vs centered separation
    print("\nlesion-vs-background saliency separation:")
    sep("raw max-cos", torch.einsum("nqd,cd->nqc", z, An).amax(-1), w, N)
    mu = A.mean(0, keepdim=True)
    Ac = F.normalize(A - mu, dim=-1)
    zc = F.normalize(z - mu, dim=-1)
    s = torch.einsum("nqd,cd->nqc", zc, Ac).amax(-1)          # centered max-cos
    sep("centered max-cos", s, w, N)

    # (3) decoupling vs objectness + headroom in the confident regime
    wf, sf = w.flatten(), s.flatten()
    conf = wf > 0.5
    sc, wc = sf[conf].numpy(), wf[conf].numpy()
    n_diffuse = int(((wf > 0.5) & (sf < 0.5)).sum())
    flip = 0
    n_eval = 0
    for n in range(N):
        K = min(32, Q)
        idx = torch.topk(w[n], K).indices
        if w[n, idx[0]] < 0.05:
            continue
        n_eval += 1
        wk, sk = w[n, idx], s[n, idx]
        if int(wk.argmax()) != int((wk * sk).argmax()):
            flip += 1
    print(f"\nconfident detections (w>0.5): {int(conf.sum())} queries")
    print(f"  centered-s spread: mean={sc.mean():.3f} std={sc.std():.3f} "
          f"p10={np.quantile(sc,.1):.3f} p90={np.quantile(sc,.9):.3f}")
    print(f"  corr(w, s) among confident: r={np.corrcoef(wc, sc)[0,1]:+.3f}  (low → distinct signal)")
    print(f"  confident-but-diffuse (w>.5 & s<.5): {n_diffuse} ({100*n_diffuse/conf.sum():.1f}% of confident)")
    print(f"  top-1 evidence flips under w·s: {flip}/{n_eval} ({100*flip/n_eval:.1f}%)")

    print("\nVERDICT: signals are distinct (d=2.65, ρ≈0.1) but the decoupling only "
          "touches a ~4% confident-diffuse tail + weak-w long tail → expect ~neutral "
          "on headline faithfulness. A conclusive A/B needs CEAH RETRAINED on w·s "
          "(eval-only swap is confounded by CEAH's objectness-trained lesion_weights).")


if __name__ == "__main__":
    main()
