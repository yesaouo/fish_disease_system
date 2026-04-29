"""Projection heads for FaCE-R Phase 2.

Lightweight MLPs that project frozen VLM embeddings into a space where
case-to-case similarity (computed from projected lesion + raw global) better
correlates with cause-set similarity.

Phase 2 default: train lesion head only, leave global frozen.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPProjection(nn.Module):
    """2-layer MLP + L2 norm. Acts elementwise (per-token projection)."""

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 512,
        out_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_dim]
        y = self.net(x)
        return F.normalize(y, dim=-1)


# ---------------------------------------------------------------------------
# Set-similarity helpers (batched, differentiable)
# ---------------------------------------------------------------------------

def pairwise_max_mean_set_sim(
    a: torch.Tensor,        # [B, max_Na, D]
    a_mask: torch.Tensor,   # [B, max_Na]  bool, True=valid
    b: torch.Tensor,        # [B, max_Nb, D]  (often a == b)
    b_mask: torch.Tensor,   # [B, max_Nb]
) -> torch.Tensor:
    """Symmetric max-mean set similarity for every pair (i, j) in the batch.

    Returns [B_a, B_b] tensor where entry (i, j) =
        0.5 * mean_p max_q cos(a[i, p], b[j, q])
      + 0.5 * mean_q max_p cos(a[i, p], b[j, q]).

    Padded positions are masked out (set to -inf before max, zero before mean).
    Empty rows yield 0 to avoid NaN.
    """
    Ba, Pa, D = a.shape
    Bb, Pb, _ = b.shape
    # All-pairs cosine: [Ba, Bb, Pa, Pb]
    sims = torch.einsum("ipd,jqd->ijpq", a, b)
    # Mask: only valid (i,p,j,q) participate
    mab = a_mask.unsqueeze(1).unsqueeze(3) & b_mask.unsqueeze(0).unsqueeze(2)  # [Ba, Bb, Pa, Pb]
    sims = sims.masked_fill(~mab, float("-inf"))

    # forward: for each p in a, max over q in b → [Ba, Bb, Pa]
    fwd_max = sims.max(dim=3).values
    # zero-out invalid p (and rows with no valid q)
    fwd_valid = a_mask.unsqueeze(1).expand(Ba, Bb, Pa) & b_mask.any(dim=1).view(1, Bb, 1).expand(Ba, Bb, Pa)
    fwd_max = torch.where(fwd_valid, fwd_max, torch.zeros_like(fwd_max))
    fwd_count = fwd_valid.float().sum(dim=2).clamp_min(1.0)
    fwd_mean = fwd_max.sum(dim=2) / fwd_count  # [Ba, Bb]

    # backward: for each q in b, max over p in a → [Ba, Bb, Pb]
    bwd_max = sims.max(dim=2).values
    bwd_valid = b_mask.unsqueeze(0).expand(Ba, Bb, Pb) & a_mask.any(dim=1).view(Ba, 1, 1).expand(Ba, Bb, Pb)
    bwd_max = torch.where(bwd_valid, bwd_max, torch.zeros_like(bwd_max))
    bwd_count = bwd_valid.float().sum(dim=2).clamp_min(1.0)
    bwd_mean = bwd_max.sum(dim=2) / bwd_count

    return 0.5 * (fwd_mean + bwd_mean)
