"""Cross-attention residual reranker for CRR-DeepRVQ.

Predicts Δ_i ≈ qᵀe_i for each top-K candidate, then s_final = s_first + Δ.
Two variants share the same architecture; they differ only in candidate
features:

    Light: ẑ_i, q⊙ẑ_i, |q-ẑ_i|, per-level code embedding, s_first, ‖e_i‖
           → compressed-memory deployment (no z/e access at runtime)

    Full:  Light features + z_i, e_i, q⊙e_i
           → full-memory deployment (top-K fetches dense). With z and e
           in hand, Δ = qᵀe is analytically recoverable, so Full also
           serves as the learned-upper-bound ablation.

Permutation invariance across candidates is preserved: each candidate's
token is built independently, query-to-candidate cross-attention is order-
agnostic, and the set summaries (mean, max) are perm-invariant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class RerankerConfig:
    variant: str = "light"            # "light" | "full"
    d_model: int = 768
    d_hidden: int = 512
    code_emb_dim: int = 32
    n_attn_heads: int = 8
    M: int = 4
    K: int = 256
    score_head_hidden: int = 256


class Reranker(nn.Module):
    """Query-to-candidate cross-attention residual reranker."""

    def __init__(self, cfg: RerankerConfig):
        super().__init__()
        self.cfg = cfg

        # Per-level code embedding lookup
        self.code_embs = nn.ModuleList([
            nn.Embedding(cfg.K, cfg.code_emb_dim) for _ in range(cfg.M)
        ])

        # Candidate feature dimension
        light_feat_dim = (
            cfg.d_model * 3                       # ẑ, q⊙ẑ, |q-ẑ|
            + cfg.M * cfg.code_emb_dim            # M per-level code embs
            + 2                                   # s_first, ‖e‖
        )
        full_extra_dim = cfg.d_model * 3          # z, e, q⊙e
        self.feat_dim = light_feat_dim + (
            full_extra_dim if cfg.variant == "full" else 0
        )

        # Token encoder ψ
        self.token_proj = nn.Sequential(
            nn.Linear(self.feat_dim, cfg.d_hidden),
            nn.GELU(),
            nn.LayerNorm(cfg.d_hidden),
        )

        # Query projection for attention key
        self.q_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_hidden),
            nn.GELU(),
            nn.LayerNorm(cfg.d_hidden),
        )

        # Query-to-candidate cross-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.d_hidden,
            num_heads=cfg.n_attn_heads,
            batch_first=True,
        )

        # Per-candidate score head
        # head_in = [h_i, m_q, h_mean, h_max, s_first, ‖e‖]
        head_in = cfg.d_hidden * 4 + 2
        self.head = nn.Sequential(
            nn.Linear(head_in, cfg.score_head_hidden),
            nn.GELU(),
            nn.LayerNorm(cfg.score_head_hidden),
            nn.Linear(cfg.score_head_hidden, 1),
        )

    def _build_token(
        self,
        q: torch.Tensor,                   # [B, D]
        z_hat: torch.Tensor,               # [B, K_top, D]
        codes: torch.Tensor,               # [B, K_top, M]
        s_first: torch.Tensor,             # [B, K_top]
        e_norm: torch.Tensor,              # [B, K_top]
        z: Optional[torch.Tensor] = None,  # [B, K_top, D]
        e: Optional[torch.Tensor] = None,  # [B, K_top, D]
    ) -> torch.Tensor:                     # [B, K_top, d_hidden]
        B, K_top, D = z_hat.shape
        q_exp = q.unsqueeze(1).expand(B, K_top, D)

        had = q_exp * z_hat                            # [B, K_top, D]
        diff = (q_exp - z_hat).abs()                   # [B, K_top, D]

        codes_long = codes.long()
        code_emb_list = [
            self.code_embs[m](codes_long[:, :, m])     # [B, K_top, code_emb_dim]
            for m in range(self.cfg.M)
        ]
        code_emb_cat = torch.cat(code_emb_list, dim=-1)  # [B, K_top, M*code_emb_dim]
        scalar = torch.stack([s_first, e_norm], dim=-1)  # [B, K_top, 2]

        feats = [z_hat, had, diff, code_emb_cat, scalar]
        if self.cfg.variant == "full":
            assert z is not None and e is not None, \
                "Full variant requires z and e tensors"
            qe = q_exp * e                              # [B, K_top, D]
            feats.extend([z, e, qe])

        x = torch.cat(feats, dim=-1)                    # [B, K_top, feat_dim]
        return self.token_proj(x)

    def forward(
        self,
        q: torch.Tensor,
        z_hat: torch.Tensor,
        codes: torch.Tensor,
        s_first: torch.Tensor,
        e_norm: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        e: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns Δ predictions: [B, K_top]."""
        h = self._build_token(q, z_hat, codes, s_first, e_norm, z=z, e=e)
        # h: [B, K_top, d_hidden]

        # Query token
        q_h = self.q_proj(q).unsqueeze(1)              # [B, 1, d_hidden]

        # Cross-attention: q attends to candidates -> context m_q
        m_q, _ = self.attn(q_h, h, h)                  # [B, 1, d_hidden]
        m_q = m_q.squeeze(1)                           # [B, d_hidden]

        # Permutation-invariant set summaries
        h_mean = h.mean(dim=1)                         # [B, d_hidden]
        h_max = h.max(dim=1).values                    # [B, d_hidden]

        # Per-candidate scoring
        B, K_top, D_h = h.shape
        m_q_exp = m_q.unsqueeze(1).expand(B, K_top, D_h)
        h_mean_exp = h_mean.unsqueeze(1).expand(B, K_top, D_h)
        h_max_exp = h_max.unsqueeze(1).expand(B, K_top, D_h)
        scalar = torch.stack([s_first, e_norm], dim=-1)

        head_in = torch.cat(
            [h, m_q_exp, h_mean_exp, h_max_exp, scalar], dim=-1,
        )
        delta = self.head(head_in).squeeze(-1)          # [B, K_top]
        return delta


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def listwise_kl_loss(
    s_pred: torch.Tensor,         # [B, K_top]
    s_target: torch.Tensor,       # [B, K_top]
    temp: float = 0.1,
) -> torch.Tensor:
    """KL(softmax(s_target/T) || softmax(s_pred/T)) averaged over batch."""
    log_pred = F.log_softmax(s_pred / temp, dim=-1)
    target = F.softmax(s_target / temp, dim=-1)
    return -(target * log_pred).sum(dim=-1).mean()


@torch.no_grad()
def analytic_full_delta(q: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """Oracle Δ = qᵀe (no learning). Used as Full upper-bound baseline.

    q: [B, D];  e: [B, K_top, D]  →  Δ: [B, K_top]
    """
    return (q.unsqueeze(1) * e).sum(dim=-1)
