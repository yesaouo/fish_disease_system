"""GROD-side disease (abstain) head — standalone, NOT in the inference path.

``ThresholdHead`` predicts a per-image objectness threshold τ(g) from a single
GROD forward; verdict p = sigmoid(scale·(max_i w_i − τ(g))), w_i =
sigmoid(pred_logits[i,0]). Trained by train_disease_head.py.

Kept for experiments / OOD work but **not** loaded by gpu_infer*.py — production
uses a fixed constant threshold. See grod/LESION_GATE.md for why (the head
over-selects as a lesion gate and only ties a constant on abstain).
``DiseaseHead`` + ``build_feat`` are the free-head ablation baseline
(compare_disease_head_feats.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ThresholdHead(nn.Module):
    """Predict per-image objectness threshold τ(g) ∈ (0,1); verdict = max_w ≥ τ(g).

    p = sigmoid( scale · (max_w − τ(g)) ).  Self-contained: standardizes g via
    stored buffers; τ predictor is an n_hidden-layer ReLU MLP.
    """

    def __init__(self, gdim: int = 768, hidden: int = 256,
                 n_hidden: int = 2, dropout: float = 0.0):
        super().__init__()
        self.register_buffer("g_mean", torch.zeros(gdim))
        self.register_buffer("g_std", torch.ones(gdim))
        layers, d = [], gdim
        for _ in range(n_hidden):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.mlp = nn.Sequential(*layers)
        self.log_scale = nn.Parameter(torch.tensor(2.3))   # exp ≈ 10

    def tau(self, g: torch.Tensor) -> torch.Tensor:
        """Per-image objectness threshold τ(g) ∈ (0,1); g: [B,768] -> [B]."""
        return torch.sigmoid(self.mlp((g - self.g_mean) / self.g_std).squeeze(-1))

    def forward(self, g: torch.Tensor, max_w: torch.Tensor):
        """g: [B,768], max_w: [B] -> (p_disease[B], τ[B])."""
        tau = self.tau(g)
        return torch.sigmoid(self.log_scale.exp() * (max_w - tau)), tau


class DiseaseHead(nn.Module):
    """Superseded free-head ablation baseline: standardize -> Linear -> sigmoid.

    Kept only for compare_disease_head_feats.py; not the production abstain gate.
    """

    def __init__(self, dim: int = 1068):
        super().__init__()
        self.register_buffer("feat_mean", torch.zeros(dim))
        self.register_buffer("feat_std", torch.ones(dim))
        self.fc = nn.Linear(dim, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = (feat - self.feat_mean) / self.feat_std
        return torch.sigmoid(self.fc(x).squeeze(-1))


def build_feat(g: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Free-head ablation feat: g[768] + per-query objectness logits[Q] -> [768+Q]."""
    return torch.cat([g, logits])


def load_disease_head(ckpt_path, device="cuda"):
    """Return (ThresholdHead[eval, on device], tau) from a disease_head.pt checkpoint."""
    ck = torch.load(ckpt_path, weights_only=False, map_location=device)
    head = ThresholdHead(**ck["head_cfg"]).to(device).eval()
    head.load_state_dict(ck["head_state"])
    return head, float(ck["tau"])


# ---------------------------------------------------------------------------
# Abstain (health verdict) head — concat(g[768], max_w, Σw) free head.
# This is the head the demo loads as the 健/病 gate (verdict = p ≥ τ); trained by
# train_abstain_head.py. Distinct from the ThresholdHead τ-selector above.
# ---------------------------------------------------------------------------

def abstain_feat(g: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Build the 770-d abstain feature for one image.

    g: [768] global; w: [Q] per-query objectness sigmoids -> [770]
    concat(g, max_i w_i, Σ_i w_i).
    """
    g, w = g.float(), w.float()
    return torch.cat([g, w.amax().view(1), w.sum().view(1)])


def load_abstain_head(ckpt_path, device="cuda"):
    """Return (DiseaseHead[eval, on device], tau) for the 770 concat(g,max_w,Σw) head."""
    ck = torch.load(ckpt_path, weights_only=False, map_location=device)
    head = DiseaseHead(int(ck["dim"])).to(device).eval()
    head.load_state_dict(ck["head_state"])
    return head, float(ck["tau"])
