"""Master-slave Mamba case encoder (Phase 4 architecture ablation).

This is the original ``MambaCaseEncoder`` that the paper compares against as
an architecture ablation. The production Phase 4 encoder is DeepSets, defined
in ``diagnosis_model.cause_inference.models.case_encoder``; this file is kept
because the Phase 4 ablation table reports Mamba results.

Activating this module imports ``mamba_ssm.Mamba3``, which on this host is
only available in the ``mamba3`` conda env (with ``CC=/usr/bin/gcc-12`` for
triton kernel JIT). To use it, instantiate via
``build_encoder(EncoderConfig(encoder_type='mamba'))`` from the SDM env will
fail at import time — switch envs first.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from diagnosis_model.cause_inference.models.case_encoder import (
    EncoderConfig,
    TokenBuilder,
)


class _MambaBlock(nn.Module):
    """Pre-norm Mamba3 block with residual."""

    def __init__(self, cfg: EncoderConfig, layer_idx: int):
        super().__init__()
        from mamba_ssm import Mamba3
        self.norm = nn.RMSNorm(cfg.d_model)
        self.mamba = Mamba3(
            d_model=cfg.d_model,
            d_state=cfg.d_state,
            headdim=cfg.headdim,
            is_mimo=cfg.is_mimo,
            mimo_rank=cfg.mimo_rank,
            chunk_size=cfg.chunk_size,
            is_outproj_norm=False,
            layer_idx=layer_idx,
            n_layer=cfg.n_layers,
            dtype=cfg.dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mamba3 expects bfloat16 input on CUDA.
        h = self.norm(x)
        h = self.mamba(h.to(self.mamba.in_proj.weight.dtype))
        return x + h.to(x.dtype)


class MambaCaseEncoder(nn.Module):
    """Sequence: [global, lesion_1, ..., lesion_N (area DESC)] -> Mamba -> last.

    The first token (global) seeds the SSM state; subsequent lesion tokens
    update it via Mamba's selective scan. We pool by extracting the output at
    the last real position per sample (true_len - 1).
    """

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.token_builder = TokenBuilder(cfg)
        self.layers = nn.ModuleList(
            [_MambaBlock(cfg, layer_idx=i) for i in range(cfg.n_layers)]
        )
        self.final_norm = nn.RMSNorm(cfg.d_model)
        if cfg.use_projection_head:
            self.head = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.head_hidden),
                nn.GELU(),
                nn.LayerNorm(cfg.head_hidden),
                nn.Linear(cfg.head_hidden, cfg.d_model),
            )
        else:
            self.head = nn.Identity()

    def forward(
        self,
        global_emb: torch.Tensor,        # [B, D]
        lesion_pad: torch.Tensor,        # [B, max_N, D]
        lesion_lens: torch.Tensor,       # [B]
    ) -> torch.Tensor:
        seq = self.token_builder(global_emb, lesion_pad, lesion_lens)  # [B, L, D]
        for layer in self.layers:
            seq = layer(seq)
        seq = self.final_norm(seq)

        # Pool: output at last *real* position. Real-token count = lesion_lens + 1
        # (because the global token sits at index 0).
        true_last = (lesion_lens + 1) - 1                  # [B]   (= lesion_lens)
        idx = true_last.view(-1, 1, 1).expand(-1, 1, seq.size(-1))
        h = seq.gather(dim=1, index=idx).squeeze(1)        # [B, D]

        z = self.head(h)
        z = F.normalize(z, dim=-1)
        return z
