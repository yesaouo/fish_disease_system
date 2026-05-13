"""Case encoder for Phase 4 retrieval distillation (DeepSets / MeanPool).

Each case is represented by:
    global_emb         (1 x D, master)
    lesion_embs        (N x D, slaves; sorted by area DESC at dataloader time)

The encoder maps the variable-length set to a single L2-normed vector h_final
(D-dim) so that case-to-case retrieval becomes one cosine dot-product.

Two encoder variants live here:

    'mean'        - mean-pooled (W_g g) + (W_l l_i) baseline
    'deepsets'    - mean+max+sum -> MLP baseline (set-aware, no sequence)

The production choice is 'deepsets' (see README Phase 4). A third variant
('mamba', master-slave Mamba3 stack) lives in
``diagnosis_model.cause_inference.mamba_ablation.mamba_encoder``; it is kept
as an architecture ablation and requires the ``mamba3`` conda env (mamba_ssm
needs ``CC=/usr/bin/gcc-12`` for triton kernel JIT). ``build_encoder``
lazy-imports it only when ``encoder_type='mamba'`` so the SDM env runs the
production path without any extra deps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EncoderConfig:
    encoder_type: str = "deepsets"       # mean | deepsets | mamba (ablation)
    d_model: int = 768                   # input/output feature dim
    # Mamba-specific (only used when encoder_type='mamba'; harmless otherwise)
    n_layers: int = 2
    d_state: int = 128
    headdim: int = 64
    # MIMO bwd kernel currently fails to compile in this env (tilelang CUDA
    # codegen bug). Vanilla Mamba3 backward works -> default to is_mimo=False.
    is_mimo: bool = False
    mimo_rank: int = 4
    chunk_size: int = 16
    # Token construction
    use_role_embeddings: bool = True
    use_input_projection: bool = True    # W_global / W_lesion linear projections
    # Output head
    use_projection_head: bool = True
    head_hidden: int = 768
    # Misc
    dtype: torch.dtype = field(default=torch.bfloat16)


# ---------------------------------------------------------------------------
# Token builder (shared by all variants, including the Mamba ablation)
# ---------------------------------------------------------------------------

class TokenBuilder(nn.Module):
    """Build (B, L_max, D) padded sequence from (global, lesion_set) inputs.

    Output token layout per sample:
        token[0]      = role_global  + W_global(global_emb)
        token[1..N]   = role_lesion  + W_lesion(lesion_embs sorted by area DESC)
        token[N+1:]   = zero-padded; the dataloader provides true seq_len.
    """

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.use_input_projection:
            self.W_global = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            self.W_lesion = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        else:
            self.W_global = nn.Identity()
            self.W_lesion = nn.Identity()
        if cfg.use_role_embeddings:
            self.role_global = nn.Parameter(torch.zeros(cfg.d_model))
            self.role_lesion = nn.Parameter(torch.zeros(cfg.d_model))
        else:
            self.register_buffer("role_global", torch.zeros(cfg.d_model),
                                 persistent=False)
            self.register_buffer("role_lesion", torch.zeros(cfg.d_model),
                                 persistent=False)

    def forward(
        self,
        global_emb: torch.Tensor,        # [B, D]
        lesion_pad: torch.Tensor,        # [B, max_N, D]  (zero-padded)
        lesion_lens: torch.Tensor,       # [B] number of real lesions per sample
    ) -> torch.Tensor:
        B, max_N, D = lesion_pad.shape
        assert global_emb.shape == (B, D)

        g = self.W_global(global_emb) + self.role_global       # [B, D]
        L = self.W_lesion(lesion_pad) + self.role_lesion       # [B, max_N, D]

        # Mask padded positions back to zero so they do not push the SSM state.
        # Shape: [B, max_N, 1].
        mask = (
            torch.arange(max_N, device=lesion_pad.device).unsqueeze(0)
            < lesion_lens.unsqueeze(1)
        ).unsqueeze(-1).to(L.dtype)
        L = L * mask

        seq = torch.cat([g.unsqueeze(1), L], dim=1)            # [B, max_N+1, D]
        return seq


# ---------------------------------------------------------------------------
# Baselines (same I/O contract)
# ---------------------------------------------------------------------------

class MeanPoolEncoder(nn.Module):
    """Trivial baseline: average of (W_g global) and (W_l lesion mean)."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.token_builder = TokenBuilder(cfg)  # reuses W_global/W_lesion/roles
        if cfg.use_projection_head:
            self.head = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.head_hidden),
                nn.GELU(),
                nn.LayerNorm(cfg.head_hidden),
                nn.Linear(cfg.head_hidden, cfg.d_model),
            )
        else:
            self.head = nn.Identity()

    def forward(self, global_emb, lesion_pad, lesion_lens):
        seq = self.token_builder(global_emb, lesion_pad, lesion_lens)  # [B, L, D]
        # mask: token 0 is always real; tokens 1..max_N follow lesion_lens.
        max_N = lesion_pad.size(1)
        device = lesion_pad.device
        lesion_mask = (
            torch.arange(max_N, device=device).unsqueeze(0) < lesion_lens.unsqueeze(1)
        )
        full_mask = torch.cat(
            [torch.ones(seq.size(0), 1, device=device, dtype=torch.bool),
             lesion_mask], dim=1
        ).unsqueeze(-1).to(seq.dtype)
        h = (seq * full_mask).sum(1) / full_mask.sum(1).clamp(min=1.0)
        z = self.head(h)
        z = F.normalize(z, dim=-1)
        return z


class DeepSetsEncoder(nn.Module):
    """mean | max | sum pooled tokens -> MLP -> L2-norm."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.token_builder = TokenBuilder(cfg)
        in_dim = cfg.d_model * 3
        self.head = nn.Sequential(
            nn.Linear(in_dim, cfg.head_hidden),
            nn.GELU(),
            nn.LayerNorm(cfg.head_hidden),
            nn.Linear(cfg.head_hidden, cfg.d_model),
        )

    def forward(self, global_emb, lesion_pad, lesion_lens):
        seq = self.token_builder(global_emb, lesion_pad, lesion_lens)  # [B, L, D]
        max_N = lesion_pad.size(1)
        device = lesion_pad.device
        lesion_mask = (
            torch.arange(max_N, device=device).unsqueeze(0) < lesion_lens.unsqueeze(1)
        )
        full_mask = torch.cat(
            [torch.ones(seq.size(0), 1, device=device, dtype=torch.bool),
             lesion_mask], dim=1
        ).unsqueeze(-1)
        # mean
        m_sum = (seq * full_mask.to(seq.dtype)).sum(1)
        m_cnt = full_mask.sum(1).clamp(min=1.0).to(seq.dtype)
        h_mean = m_sum / m_cnt
        # max (set masked positions to very-neg before max)
        seq_for_max = seq.masked_fill(~full_mask, float("-inf"))
        h_max = seq_for_max.max(dim=1).values
        # sum
        h_sum = m_sum
        h = torch.cat([h_mean, h_max, h_sum], dim=-1)
        z = self.head(h)
        z = F.normalize(z, dim=-1)
        return z


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

ENCODERS = {
    "mean": MeanPoolEncoder,
    "deepsets": DeepSetsEncoder,
}


def build_encoder(cfg: EncoderConfig) -> nn.Module:
    """Build a case encoder by name.

    'mean' and 'deepsets' are pure PyTorch and run in any env. 'mamba' is an
    ablation kept under diagnosis_model.cause_inference.mamba_ablation; it is
    lazy-imported here so callers that never request it don't need mamba_ssm.
    """
    if cfg.encoder_type == "mamba":
        from diagnosis_model.cause_inference.mamba_ablation.mamba_encoder import (
            MambaCaseEncoder,
        )
        return MambaCaseEncoder(cfg)
    if cfg.encoder_type not in ENCODERS:
        raise ValueError(
            f"unknown encoder_type={cfg.encoder_type}; "
            f"choices={list(ENCODERS) + ['mamba (ablation)']}"
        )
    return ENCODERS[cfg.encoder_type](cfg)


# ---------------------------------------------------------------------------
# Listwise KL distillation loss
# ---------------------------------------------------------------------------

def listwise_kl_loss(
    h_student: torch.Tensor,                # [B, D] L2-normed
    teacher_scores: torch.Tensor,           # [B, B] with NaN diagonal
    temp_target: float = 0.1,
    temp_pred: float = 0.1,
) -> torch.Tensor:
    """KL divergence between teacher and student per-anchor distributions.

    For anchor i, the row distribution over the other (B-1) cases in the batch
    is the supervision signal. The diagonal is masked (self).
    """
    B = h_student.size(0)
    device = h_student.device
    s_pred = (h_student @ h_student.T) / temp_pred              # [B, B]
    eye = torch.eye(B, dtype=torch.bool, device=device)

    # Mask self in BOTH distributions with -inf -> softmax(-inf) = 0.
    s_pred_masked = s_pred.masked_fill(eye, float("-inf"))

    s_teacher = teacher_scores.to(device).float()
    nan_mask = torch.isnan(s_teacher) | eye
    s_teacher = s_teacher / temp_target
    s_teacher = s_teacher.masked_fill(nan_mask, float("-inf"))

    log_pred = F.log_softmax(s_pred_masked, dim=-1)             # [B, B]
    target = F.softmax(s_teacher, dim=-1)                       # [B, B]

    # KL(target || pred) = sum target * (log target - log pred)
    # We drop the entropy term (it doesn't depend on student) and just
    # use cross-entropy: -(target * log_pred).sum(-1).mean().
    # Masked positions have target=0 and log_pred=-inf, which produces
    # NaN under direct multiplication; zero them explicitly.
    log_pred = log_pred.masked_fill(nan_mask, 0.0)
    loss = -(target * log_pred).sum(dim=-1).mean()
    return loss


def pairwise_mse_loss(
    h_student: torch.Tensor,
    teacher_scores: torch.Tensor,
) -> torch.Tensor:
    """MSE on case-pair similarity matrix (ablation alternative)."""
    B = h_student.size(0)
    s_pred = h_student @ h_student.T                             # [B, B]
    s_teacher = teacher_scores.to(s_pred.device).float()
    eye = torch.eye(B, dtype=torch.bool, device=s_pred.device)
    valid = (~torch.isnan(s_teacher)) & (~eye)
    return F.mse_loss(s_pred[valid], s_teacher[valid])


def case_cause_infonce_loss(
    h_student: torch.Tensor,                # [B, D] L2-normed
    cause_text_embs: torch.Tensor,          # [V, D] L2-normed (V = 56k)
    pos_mask: torch.Tensor,                 # [B, V] bool, True = positive cause for anchor
    temp: float = 0.07,
) -> torch.Tensor:
    """SupCon (L_out) multi-positive InfoNCE: case h_final aligned to GT causes.

    Negatives are all causes that are NOT GT for the anchor (in-batch + full vocab).
    Gives the encoder direct supervision the Phase-1 teacher never sees.
    """
    B = h_student.size(0)
    device = h_student.device
    logits = (h_student @ cause_text_embs.T) / temp              # [B, V]
    pos_mask = pos_mask.to(device)

    # L_out from SupCon: numerator = sum over positives; denominator = sum over all
    pos_logits = logits.masked_fill(~pos_mask, float("-inf"))
    pos_lse = torch.logsumexp(pos_logits, dim=-1)                # [B]
    all_lse = torch.logsumexp(logits, dim=-1)                    # [B]

    has_pos = pos_mask.any(dim=-1)
    if not has_pos.any():
        return logits.sum() * 0.0  # zero, but keeps graph
    loss = -(pos_lse - all_lse)
    return loss[has_pos].mean()
