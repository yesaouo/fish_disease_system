"""CEAH — Cause-Evidence Attribution Head.

Architecture (gated MLP, see Phase 3 design):

  Inputs (per query):
    global_emb   [D_g]      raw VLM-Global vision
    text_emb     [D_g]|None raw VLM-Global text (colloquial or medical)
    lesion_embs  [N, D_l]   raw VLM-Lesion fusion
    cause_emb    [D_g]      raw VLM-Global text (the candidate cause)

  Steps:
    1. Per-modality projection MLP → common dim D_c (with a learned type embedding):
         g'        = global_proj(global_emb)   + type_g
         t'        = text_proj(text_emb)       + type_t  (if present)
         L' = lesion_proj(lesion_embs) + type_l
         c'        = cause_proj(cause_emb)
    2. Stack evidence: E = [g', (t'?), l'_1, …, l'_N]   shape [Ne, D_c]
    3. Per-evidence support score:
         s_i = sigmoid(W_s · [c', e_i, c' ⊙ e_i])
       Concatenated cause-evidence interaction → scalar.
    4. Sparsity: α = s (raw) — sparsity is enforced via L1 in the loss, not
       hard top-k. Optional --hard_topk in the future.
    5. Gated pool (faithfulness by construction):
         p = Σ_i α_i · e_i
       If α_i = 0, evidence i contributes 0 to the score. No back channel.
    6. Cause score:
         score = sigmoid(W_score · [c', p])

  Forward returns: (score [B], alpha [B, max_Ne], evidence_mask [B, max_Ne])
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Token-type ids
TYPE_GLOBAL = 0
TYPE_TEXT   = 1
TYPE_LESION = 2


class _MLPProj(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CEAH(nn.Module):
    """Cause-Evidence Attribution Head.

    Two architectural modes can be combined:

    attribution_mode:
      "sigmoid"  — independent per-evidence support, α_i = sigmoid(s_i).
                   Allows α=0 everywhere (collapse mode).
      "softmax"  — α = softmax(s) over valid positions. Forces Σα = 1,
                   so global cannot monopolize attribution; mass redistributes
                   when global is the wrong choice.

    scoring_mode:
      "single"          — one score head: score = sigmoid(MLP([c, pool]))
                          where pool aggregates ALL evidence.
      "multiplicative"  — split evidence into global vs local (text + lesions);
                          two score heads:
                              s_g = sigmoid(MLP_g([c, gated_global]))
                              s_l = sigmoid(MLP_l([c, gated_local]))
                          final_score = s_g · s_l. Both must be confident →
                          the model is forced to learn local attribution.
    """

    def __init__(
        self,
        global_dim: int = 768,
        text_dim: int = 768,
        lesion_dim: int = 768,
        cause_dim: int = 768,
        common_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        n_token_types: int = 3,
        attribution_mode: str = "sigmoid",
        scoring_mode: str = "single",
    ):
        super().__init__()
        self.common_dim = common_dim
        self.attribution_mode = attribution_mode
        self.scoring_mode = scoring_mode
        assert attribution_mode in ("sigmoid", "softmax")
        assert scoring_mode in ("single", "multiplicative")

        self.global_proj = _MLPProj(global_dim, hidden_dim, common_dim, dropout)
        self.text_proj   = _MLPProj(text_dim,   hidden_dim, common_dim, dropout)
        self.lesion_proj = _MLPProj(lesion_dim, hidden_dim, common_dim, dropout)
        self.cause_proj  = _MLPProj(cause_dim,  hidden_dim, common_dim, dropout)

        self.type_emb = nn.Embedding(n_token_types, common_dim)
        nn.init.normal_(self.type_emb.weight, std=0.02)

        # Per-evidence support scorer: input = [c, e, c⊙e]  → 3·D_c
        self.support = nn.Sequential(
            nn.Linear(3 * common_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        for m in self.support:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        # Score head(s)
        if scoring_mode == "single":
            self.score_head = self._make_score_head(hidden_dim, dropout)
        else:  # multiplicative
            self.score_head_global = self._make_score_head(hidden_dim, dropout)
            self.score_head_local  = self._make_score_head(hidden_dim, dropout)

    def _make_score_head(self, hidden_dim: int, dropout: float) -> nn.Sequential:
        head = nn.Sequential(
            nn.Linear(2 * self.common_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        for m in head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        return head

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def project_evidence(
        self,
        global_emb: torch.Tensor,        # [B, D_g]
        text_emb: Optional[torch.Tensor],  # [B, D_g] or None
        text_present: torch.Tensor,      # [B] bool
        lesion_embs: torch.Tensor,       # [B, max_N, D_l]
        lesion_mask: torch.Tensor,       # [B, max_N]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project to common space, stack, return (E, mask).

        E: [B, max_Ne, D_c]   max_Ne = 1 (global) + 1 (text slot) + max_N
        mask: [B, max_Ne]     True for valid tokens
        """
        B, max_N, _ = lesion_embs.shape
        D_c = self.common_dim
        device = global_emb.device

        type_g = self.type_emb(torch.tensor(TYPE_GLOBAL, device=device))
        type_t = self.type_emb(torch.tensor(TYPE_TEXT, device=device))
        type_l = self.type_emb(torch.tensor(TYPE_LESION, device=device))

        g_proj = self.global_proj(global_emb) + type_g          # [B, D_c]
        if text_emb is None:
            t_proj = torch.zeros(B, D_c, device=device)
        else:
            t_proj = self.text_proj(text_emb) + type_t

        l_flat = lesion_embs.view(B * max_N, -1)
        l_proj = self.lesion_proj(l_flat).view(B, max_N, D_c) + type_l

        # Stack: [global, text_slot, lesions...]
        E = torch.cat([g_proj.unsqueeze(1), t_proj.unsqueeze(1), l_proj], dim=1)
        # Mask
        mask = torch.cat([
            torch.ones(B, 1, dtype=torch.bool, device=device),
            text_present.view(B, 1).to(torch.bool),
            lesion_mask,
        ], dim=1)
        return E, mask

    # -----------------------------------------------------------------------
    # Main forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        global_emb: torch.Tensor,           # [B, D_g]
        text_emb: Optional[torch.Tensor],   # [B, D_g] or None
        text_present: torch.Tensor,         # [B] bool — whether text_emb slot is valid
        lesion_embs: torch.Tensor,          # [B, max_N, D_l]
        lesion_mask: torch.Tensor,          # [B, max_N]
        cause_emb: torch.Tensor,            # [B, D_g]
        force_mask: Optional[torch.Tensor] = None,   # [B, max_Ne]: AND'ed with ev_mask
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (score [B], alpha [B, max_Ne], evidence_mask [B, max_Ne]).

        `force_mask` (if provided) is AND'ed with the internal ev_mask before
        alpha is computed — used at eval time to ablate evidence types
        (e.g., zero out global to verify the score depends on lesion).
        """
        B = global_emb.size(0)
        E, ev_mask = self.project_evidence(
            global_emb, text_emb, text_present, lesion_embs, lesion_mask,
        )
        if force_mask is not None:
            ev_mask = ev_mask & force_mask
        max_Ne = E.size(1)

        c = self.cause_proj(cause_emb)                                      # [B, D_c]
        c_b = c.unsqueeze(1).expand(B, max_Ne, self.common_dim)             # [B, max_Ne, D_c]

        # Per-evidence support: input = [c, e, c⊙e]
        feats = torch.cat([c_b, E, c_b * E], dim=-1)                        # [B, max_Ne, 3D]
        s_logits = self.support(feats).squeeze(-1)                          # [B, max_Ne]
        s_logits = s_logits.masked_fill(~ev_mask, -1e9)

        if self.attribution_mode == "sigmoid":
            alpha = torch.sigmoid(s_logits)                                  # [B, max_Ne]
        else:  # softmax over valid positions
            alpha = torch.softmax(s_logits, dim=-1)                          # [B, max_Ne], Σ=1

        # Gated pool(s)
        gated = alpha.unsqueeze(-1) * E                                     # [B, max_Ne, D_c]

        if self.scoring_mode == "single":
            p = gated.sum(dim=1)                                             # [B, D_c]
            score = torch.sigmoid(
                self.score_head(torch.cat([c, p], dim=-1)).squeeze(-1)
            )
        else:  # multiplicative
            # Token layout: 0=global, 1=text, 2..=lesions
            p_global = gated[:, 0, :]                                        # [B, D_c]
            p_local  = gated[:, 1:, :].sum(dim=1)                            # [B, D_c]
            s_g = torch.sigmoid(
                self.score_head_global(torch.cat([c, p_global], dim=-1)).squeeze(-1)
            )
            s_l = torch.sigmoid(
                self.score_head_local(torch.cat([c, p_local], dim=-1)).squeeze(-1)
            )
            score = s_g * s_l                                                # [B]
        return score, alpha, ev_mask
