# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Backbone modules.
"""

import torch
import torch.nn.functional as F
from peft import PeftModel

from diagnosis_model.grod.detector.models.backbone.base import BackboneBase
from diagnosis_model.grod.detector.models.backbone.dinov2 import DinoV2
from diagnosis_model.grod.detector.models.backbone.projector import MultiScaleProjector
from diagnosis_model.grod.detector.utilities.logger import get_logger
from diagnosis_model.grod.detector.utilities.tensors import NestedTensor

logger = get_logger()

__all__ = ["Backbone"]


class Backbone(BackboneBase):
    """backbone."""

    def __init__(
        self,
        name: str,
        pretrained_encoder: str = None,
        window_block_indexes: list = None,
        drop_path=0.0,
        out_channels=256,
        out_feature_indexes: list = None,
        projector_scale: list = None,
        use_cls_token: bool = False,
        freeze_encoder: bool = False,
        layer_norm: bool = False,
        target_shape: tuple[int, int] = (640, 640),
        rms_norm: bool = False,
        backbone_lora: bool = False,
        gradient_checkpointing: bool = False,
        load_dinov2_weights: bool = True,
        patch_size: int = 14,
        num_windows: int = 4,
        positional_encoding_size: int = 0,
    ):
        super().__init__()
        # --- swappable backbone seam -------------------------------------------
        # RFDETR_BACKBONE=<name> selects an externally-registered encoder (the
        # implementation lives in the downstream project, not here — see
        # backbone/external.py). Everything below (projector / global head /
        # decoder) is backbone-agnostic and adapts via encoder._out_feature_channels.
        # Env unset -> stock DinoV2 path, byte-identical.
        import os as _os
        from diagnosis_model.grod.detector.models.backbone.external import build_external_encoder
        _bb = _os.environ.get("RFDETR_BACKBONE", "")
        if _bb:
            self.encoder = build_external_encoder(
                _bb, out_feature_indexes=out_feature_indexes, patch_size=patch_size
            )
        else:
            # an example name here would be "dinov2_base" or "dinov2_registers_windowed_base"
            # if "registers" is in the name, then use_registers is set to True, otherwise it is set to False
            # similarly, if "windowed" is in the name, then use_windowed_attn is set to True, otherwise it is set to False
            # the last part of the name should be the size
            # and the start should be dinov2
            name_parts = name.split("_")
            assert name_parts[0] == "dinov2"
            # name_parts[-1]
            use_registers = False
            if "registers" in name_parts:
                use_registers = True
                name_parts.remove("registers")
            use_windowed_attn = False
            if "windowed" in name_parts:
                use_windowed_attn = True
                name_parts.remove("windowed")
            assert len(name_parts) == 2, (
                "name should be dinov2, then either registers, windowed, both, or none, then the size"
            )
            self.encoder = DinoV2(
                size=name_parts[-1],
                out_feature_indexes=out_feature_indexes,
                shape=target_shape,
                use_registers=use_registers,
                use_windowed_attn=use_windowed_attn,
                gradient_checkpointing=gradient_checkpointing,
                load_dinov2_weights=load_dinov2_weights,
                patch_size=patch_size,
                num_windows=num_windows,
                positional_encoding_size=positional_encoding_size,
                drop_path_rate=drop_path,
            )
        # build encoder + projector as backbone module
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.projector_scale = projector_scale
        assert len(self.projector_scale) > 0
        # x[0]
        assert sorted(self.projector_scale) == self.projector_scale, (
            "only support projector scale P3/P4/P5/P6 in ascending order."
        )
        level2scalefactor = dict(P3=2.0, P4=1.0, P5=0.5, P6=0.25)
        scale_factors = [level2scalefactor[lvl] for lvl in self.projector_scale]

        self.projector = MultiScaleProjector(
            in_channels=self.encoder._out_feature_channels,
            out_channels=out_channels,
            scale_factors=scale_factors,
            layer_norm=layer_norm,
            rms_norm=rms_norm,
        )

        self._export = False

        # Optional global head support: when enabled by the LWDETR global head
        # (global_dim>0), stash a per-image pooled vector of the PRE-projector
        # DINOv2 features (masked-mean per scale, L2-normed, concatenated) so the
        # head can distill it toward a frozen whole-image embedding. Off by
        # default -> byte-identical to the plain detector.
        self.global_pool = False
        self._global_pooled = None

    @staticmethod
    def _pool_global(feats, tensor_list):
        """Masked-mean pool each pre-projector scale [B,C,H,W] -> L2-norm ->
        concat [B, sum_C]. tensor_list carries the padding mask (True=pad)."""
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

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export

        if isinstance(self.encoder, PeftModel):
            logger.info("Merging and unloading LoRA weights")
            self.encoder.merge_and_unload()

    def forward(self, tensor_list: NestedTensor):
        """ """
        # (H, W, B, C)
        feats = self.encoder(tensor_list.tensors)
        if self.global_pool:
            # prefer an encoder-owned pooling definition (external backbones bring
            # their own, shared with their training-side extraction); fall back to
            # the built-in DinoV2 pooling otherwise.
            _pool = getattr(self.encoder, "pool_global", None)
            self._global_pooled = (
                _pool(feats, tensor_list) if _pool is not None
                else self._pool_global(feats, tensor_list)
            )
        feats = self.projector(feats)
        # x: [(B, C, H, W)]
        out = []
        for feat in feats:
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            out.append(NestedTensor(feat, mask))
        return out

    def forward_export(self, tensors: torch.Tensor):
        feats = self.encoder(tensors)
        if self.global_pool:
            _pool = getattr(self.encoder, "pool_global", None)
            self._global_pooled = (
                _pool(feats, None) if _pool is not None
                else self._pool_global(feats, None)
            )
        feats = self.projector(feats)
        out_feats = []
        out_masks = []
        for feat in feats:
            # x: [(B, C, H, W)]
            b, _, h, w = feat.shape
            out_masks.append(torch.zeros((b, h, w), dtype=torch.bool, device=feat.device))
            out_feats.append(feat)
        return out_feats, out_masks

    def get_named_param_lr_pairs(self, args, prefix: str = "backbone.0"):
        num_layers = args.out_feature_indexes[-1] + 1
        backbone_key = "backbone.0.encoder"
        named_param_lr_pairs = {}
        for n, p in self.named_parameters():
            n = prefix + "." + n
            if backbone_key in n and p.requires_grad:
                lr = (
                    args.lr_encoder
                    * get_dinov2_lr_decay_rate(
                        n,
                        lr_decay_rate=args.lr_vit_layer_decay,
                        num_layers=num_layers,
                    )
                    * args.lr_component_decay**2
                )
                wd = args.weight_decay * get_dinov2_weight_decay_rate(n)
                named_param_lr_pairs[n] = {
                    "params": p,
                    "lr": lr,
                    "weight_decay": wd,
                }
        return named_param_lr_pairs


def get_dinov2_lr_decay_rate(name: str, lr_decay_rate: float = 1.0, num_layers: int = 12) -> float:
    """
    Calculate lr decay rate for different ViT blocks.

    Args:
        name: Parameter name.
        lr_decay_rate: Base lr decay rate.
        num_layers: Number of ViT blocks.

    Returns:
        Lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if "embeddings" in name:
            layer_id = 0
        elif ".layer." in name and ".residual." not in name:
            layer_id = int(name[name.find(".layer.") :].split(".")[2]) + 1
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_dinov2_weight_decay_rate(name, weight_decay_rate=1.0):
    if (
        ("gamma" in name)
        or ("pos_embed" in name)
        or ("rel_pos" in name)
        or ("bias" in name)
        or ("norm" in name)
        or ("embeddings" in name)
    ):
        weight_decay_rate = 0.0
    return weight_decay_rate
