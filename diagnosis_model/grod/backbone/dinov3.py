"""Owned, swappable DINOv3 backbone for the GROD detector.

This is the paper's "backbone" half of the backbone+decoder split, lifted out of
the rfdetr fork into this repo. It plugs into rfdetr's (still-forked) LWDETR
decoder/projector via the generic external-encoder seam
(`rfdetr.models.backbone.external`): importing this module registers each DINOv3
variant under the name `dinov3_<variant>` (plus bare `dinov3` = base), so that
building an RFDETR model with env `RFDETR_BACKBONE=dinov3_base` (etc.) routes the
backbone here. The decoder stays in rfdetr for now.

Interface contract consumed by rfdetr's Backbone wrapper:
  - forward(pixel_values) -> list[Tensor[B, C, H/p, W/p]]   (one map per tap layer)
  - ._out_feature_channels : list[int]                      (channel width per tap)
DINOv3 is a plain ViT: every tapped transformer layer shares the same spatial
grid and width; the multi-scale pyramid is built downstream by MultiScaleProjector.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import nn

# variant -> HF pretrained id. Only `base` is assumed cached; small/large pull
# from the Hub on first use. Pass a raw HF id as `variant` to use any other.
DINOV3_VARIANTS = {
    "small": "facebook/dinov3-vits16-pretrain-lvd1689m",   # 384-d, 12 layers
    "base": "facebook/dinov3-vitb16-pretrain-lvd1689m",    # 768-d, 12 layers
    "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",   # 1024-d, 24 layers
}


def _default_taps(num_layers: int):
    """Four evenly-spaced tap layers into hidden_states (index 0 = embeddings)."""
    return [num_layers // 4, num_layers // 2, (3 * num_layers) // 4, num_layers]


class DinoV3Backbone(nn.Module):
    def __init__(self, variant="base", out_feature_indexes=None, patch_size=16,
                 interpolate_pos=False):
        super().__init__()
        from transformers import AutoModel

        model_id = DINOV3_VARIANTS.get(variant, variant)
        offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
        self.encoder = AutoModel.from_pretrained(model_id, local_files_only=offline)
        cfg = self.encoder.config
        # Supervised ViTs (e.g. timm/DeiT-style) carry a fixed-resolution pos-embed
        # trained at 224; DINOv3 handles arbitrary resolution natively. Only the
        # former needs pos-embed interpolation at detection resolution.
        self.interpolate_pos = interpolate_pos

        self.patch_size = getattr(cfg, "patch_size", patch_size)
        width = cfg.hidden_size
        # Tap layers: caller-provided (e.g. rfdetr config default [3,6,9,12]) only
        # if valid for this depth, else depth-scaled. hidden_states has L+1 entries.
        n_layers = cfg.num_hidden_layers
        taps = list(out_feature_indexes) if out_feature_indexes else None
        if not taps or max(taps) > n_layers:
            taps = _default_taps(n_layers)
        self.out_feature_indexes = taps
        self._out_feature_channels = [width] * len(taps)
        # Token layout: [CLS] + [register tokens] + [H*W patch tokens].
        self.num_prefix = 1 + int(getattr(cfg, "num_register_tokens", 0))

    def forward(self, pixel_values):
        B, _, H, W = pixel_values.shape
        Hp, Wp = H // self.patch_size, W // self.patch_size
        kwargs = {"output_hidden_states": True}
        if self.interpolate_pos:
            kwargs["interpolate_pos_encoding"] = True
        out = self.encoder(pixel_values, **kwargs)
        hs = out.hidden_states  # tuple, len = num_hidden_layers + 1
        feats = []
        for idx in self.out_feature_indexes:
            t = hs[idx][:, self.num_prefix:, :]                 # [B, Hp*Wp, C]
            t = t.reshape(B, Hp, Wp, -1).permute(0, 3, 1, 2).contiguous()
            feats.append(t)                                     # [B, C, Hp, Wp]
        return feats

    @staticmethod
    def pool_global(feats, tensor_list=None):
        """Single per-image global-pool definition, shared by training
        (extract_dino_global) and inference (rfdetr Backbone defers here). Each
        pre-projector scale [B,C,H,W] -> masked-mean -> L2-norm -> concat
        [B, sum_C]. tensor_list (NestedTensor) carries the padding mask if any;
        with a square unpadded resize this reduces to a plain spatial mean.
        Mirrors the fork's Backbone._pool_global byte-for-byte so the distilled
        MLP sees identical inputs at train and deploy time."""
        m = getattr(tensor_list, "mask", None)
        per = []
        for f in feats:
            if m is not None:
                valid = (~F.interpolate(m[None].float(), size=f.shape[-2:]).to(torch.bool)[0])
                valid = valid.unsqueeze(1).to(f.dtype)              # [B,1,H,W]
                pooled = (f * valid).sum(dim=(2, 3)) / valid.sum(dim=(2, 3)).clamp_min(1.0)
            else:
                pooled = f.mean(dim=(2, 3))
            per.append(F.normalize(pooled, dim=-1))
        return torch.cat(per, dim=-1)
