from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DIM, N_CAUSE, N_TREAT


class GeomEmbed(nn.Module):
    """
    將 normalized boxes [x1,y1,x2,y2] 映射成 dim 維幾何 embedding。
    輸入: boxes_norm (B, K, 4)
    輸出: geom_emb (B, K, dim)
    """

    def __init__(self, dim: int = DIM):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(10, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

    def forward(self, boxes_norm: torch.Tensor) -> torch.Tensor:
        # boxes_norm: (B, K, 4) in [0,1]
        if boxes_norm.numel() == 0:
            B, K = boxes_norm.shape[0], boxes_norm.shape[1]
            return boxes_norm.new_zeros(B, K, self.dim)

        x1, y1, x2, y2 = boxes_norm.unbind(dim=-1)  # each (B, K)
        w = (x2 - x1).clamp(min=1e-6)
        h = (y2 - y1).clamp(min=1e-6)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        ar = w / (h + 1e-8)
        area = w * h

        base = torch.stack(
            [x1, y1, x2, y2, w, h, cx, cy, ar, area], dim=-1
        )  # (B, K, 10)

        mean = base.mean(dim=-1, keepdim=True)
        std = base.std(dim=-1, keepdim=True) + 1e-6
        base = (base - mean) / std  # (B, K, 10)

        geom_emb = self.mlp(base)  # (B, K, dim)
        return geom_emb


class CrossAlignFormer(nn.Module):
    """
    完整 Transformer 版本的 CrossAlignFormer。

    輸入 (batch-first):
        global_token: (B, dim)
        roi_feats   : (B, K, dim)
        boxes_norm  : (B, K, 4)
        class_ids   : (B, K)  (0=HEALTHY, 1=ABNORMAL)
        text_emb    : (B, dim)
        roi_mask    : (B, K)  bool，True=有效 ROI, False=padding

    輸出:
        S_final: (B, L, dim)  # L = 1 + K + 1
        q_cause: (B, Nc, dim)
        q_treat: (B, Nt, dim)
    """

    def __init__(
        self,
        dim: int = DIM,
        n_cause: int = N_CAUSE,
        n_treat: int = N_TREAT,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_ff: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_cause = n_cause
        self.n_treat = n_treat

        self.type_global = nn.Parameter(torch.randn(dim))
        self.type_roi = nn.Parameter(torch.randn(dim))
        self.type_text = nn.Parameter(torch.randn(dim))

        # label embedding: 0=HEALTHY, 1=ABNORMAL
        self.label_embed = nn.Embedding(2, dim)

        # 幾何 embedding
        self.geom_embed = GeomEmbed(dim)

        # Transformer encoder（batch_first 語意）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # slot seeds for cause / treat
        self.q_cause_seed = nn.Parameter(torch.randn(n_cause, dim))
        self.q_treat_seed = nn.Parameter(torch.randn(n_treat, dim))

        # cross-attention（slots 作為 query，序列作為 key/value）
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        # type vectors
        nn.init.normal_(self.type_global, std=0.02)
        nn.init.normal_(self.type_roi, std=0.02)
        nn.init.normal_(self.type_text, std=0.02)

        # label embedding
        nn.init.xavier_uniform_(self.label_embed.weight)

        # slot seeds
        nn.init.xavier_uniform_(self.q_cause_seed)
        nn.init.xavier_uniform_(self.q_treat_seed)

    @staticmethod
    def _l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
        return F.normalize(x, p=2.0, dim=dim, eps=eps)

    def _build_tokens(
        self,
        global_token: torch.Tensor,  # (B, dim)
        roi_feats: torch.Tensor,     # (B, K, dim)
        boxes_norm: torch.Tensor,    # (B, K, 4)
        class_ids: torch.Tensor,     # (B, K)
        text_emb: torch.Tensor,      # (B, dim)
        roi_mask: Optional[torch.Tensor],  # (B, K) bool, True=valid
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        組合 global / roi / text 三種 token，輸出:
            S_init: (B, L, dim)
            text_token: (B, 1, dim)
            src_key_padding_mask: (B, L)  True=要 mask 的 padding
        """
        B, dim = global_token.shape
        device = global_token.device

        # ----- global token -----
        # global_token: (B, dim) -> (B, 1, dim)
        global_tok = global_token.unsqueeze(1) + self.type_global.view(1, 1, -1)
        # (B, 1, dim)

        # ----- roi tokens -----
        B, K, _ = roi_feats.shape

        if roi_mask is None:
            roi_mask = torch.ones(B, K, dtype=torch.bool, device=device)

        # 幾何 & label embedding
        geom = self.geom_embed(boxes_norm)  # (B, K, dim)

        clamped_ids = class_ids.clamp(min=0, max=1).long()
        lbl_vec = self.label_embed(clamped_ids)  # (B, K, dim)

        # type_roi: (dim,) -> (1, 1, dim)，broadcast 到 (B, K, dim)
        roi_tok = roi_feats + geom + lbl_vec + self.type_roi.view(1, 1, -1)
        # (B, K, dim)

        # ----- text token -----
        text_token = text_emb.unsqueeze(1) + self.type_text.view(1, 1, -1)
        # (B, 1, dim)

        # ----- concat -----
        # token 排序: [global] + [roi_1 ... roi_K] + [text]
        S_init = torch.cat([global_tok, roi_tok, text_token], dim=1)  # (B, L, dim)
        S_init = self._l2_normalize(S_init, dim=-1)

        # 建立 src_key_padding_mask: True = padding / 不參與 attention
        L = 1 + K + 1
        src_key_padding_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        # 位置 1~K 是 ROI
        src_key_padding_mask[:, 1 : 1 + K] = ~roi_mask

        return S_init, text_token, src_key_padding_mask

    def _slot_cross_attention(
        self,
        seq: torch.Tensor,          # (B, L, dim)
        slots_seed: torch.Tensor,   # (N_slots, dim)
        src_key_padding_mask: Optional[torch.Tensor] = None,  # (B, L)
    ) -> torch.Tensor:
        """
        Multi-head cross-attention:
            Query = slot seeds
            Key/Value = seq
        回傳 (B, N_slots, dim)
        """
        B, L, dim = seq.shape
        N_slots = slots_seed.shape[0]

        # slots_seed: (N_slots, dim) -> (B, N_slots, dim) broadcast
        slots = slots_seed.unsqueeze(0).expand(B, -1, -1).contiguous()

        out, _ = self.cross_attn(
            query=slots,
            key=seq,
            value=seq,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        out = self._l2_normalize(out, dim=-1)
        return out  # (B, N_slots, dim)

    def forward(
        self,
        global_token: torch.Tensor,   # (B, dim)
        roi_feats: torch.Tensor,      # (B, K, dim)
        boxes_norm: torch.Tensor,     # (B, K, 4)
        class_ids: torch.Tensor,      # (B, K)
        text_emb: torch.Tensor,       # (B, dim)
        roi_mask: Optional[torch.Tensor] = None,  # (B, K) bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        回傳:
            S_final: (B, L, dim)
            q_cause: (B, Nc, dim)
            q_treat: (B, Nt, dim)
        """
        # 1) 組合 tokens
        S_init, text_token, src_key_padding_mask = self._build_tokens(
            global_token, roi_feats, boxes_norm, class_ids, text_emb, roi_mask
        )  # S_init: (B, L, dim), text_token: (B, 1, dim)

        # 2) Transformer Encoder
        S_final = self.encoder(
            S_init,
            src_key_padding_mask=src_key_padding_mask,
        )  # (B, L, dim)
        S_final = self._l2_normalize(S_final, dim=-1)

        # 3) cause slots cross-attention
        q_cause = self._slot_cross_attention(
            S_final,
            self.q_cause_seed,
            src_key_padding_mask,
        )  # (B, Nc, dim)

        # 4) treat slots cross-attention（加入 text bias）
        S_treat = S_final + text_token  # (B, L, dim) + (B, 1, dim) broadcast
        S_treat = self._l2_normalize(S_treat, dim=-1)

        q_treat = self._slot_cross_attention(
            S_treat,
            self.q_treat_seed,
            src_key_padding_mask,
        )  # (B, Nt, dim)

        return S_final, q_cause, q_treat
