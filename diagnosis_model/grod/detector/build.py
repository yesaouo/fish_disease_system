"""Standalone builder for the vendored GROD detector — no rfdetr import.

Replicates rfdetr's `_build_model_context` (build_namespace -> build_model ->
load pretrain) using the vendored config / namespace / model, and registers this
repo's backbone + heads against the vendored seams. So the GROD detector can be
built and run for inference without the rfdetr package.

  RFDETR_BACKBONE=dinov3_base / RFDETR_SEMANTIC_DIM / RFDETR_GLOBAL_DIM env
  switches behave exactly as on the fork path (read by the same factories).
"""

from __future__ import annotations

import torch

# RF-DETR image normalization (ImageNet), matching rfdetr.detr.RFDETR.means/stds.
IMAGENET_MEANS = [0.485, 0.456, 0.406]
IMAGENET_STDS = [0.229, 0.224, 0.225]

from .config import RFDETRMediumConfig, TrainConfig
from ._namespace import build_namespace
from .models import build_model
from .models.backbone.external import register_encoder
from .models.region_heads import register_region_heads
from .models.region_losses import register_region_losses

_REGISTERED = False


def _register_seams():
    """Register this repo's backbone/heads impls against the vendored seams (once)."""
    global _REGISTERED
    if _REGISTERED:
        return
    from diagnosis_model.grod.backbone import register_backbones
    from diagnosis_model.grod.heads import register_heads
    register_backbones(register_encoder)
    register_heads(register_region_heads, register_region_losses)  # no dataset seam (inference)
    _REGISTERED = True


def _load_pretrain(nn_model, ckpt_path, num_queries, group_detr):
    """Minimal port of rfdetr's _load_pretrain_weights_into (local file, no download)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"]

    # align detection-head class count with the checkpoint
    ck_nc = sd["class_embed.bias"].shape[0]
    cur_nc = nn_model.class_embed[0].bias.shape[0] if isinstance(
        nn_model.class_embed, torch.nn.ModuleList) else nn_model.class_embed.bias.shape[0]
    if ck_nc != cur_nc:
        nn_model.reinitialize_detection_head(ck_nc)

    # trim group-detr query embeddings to the configured count
    ndq = num_queries * group_detr
    for name in list(sd.keys()):
        if name.endswith("refpoint_embed.weight") or name.endswith("query_feat.weight"):
            sd[name] = sd[name][:ndq]

    # drop tensors whose shape no longer matches (e.g. backbone-swapped projector)
    msd = nn_model.state_dict()
    for k in [k for k, v in sd.items() if k in msd and msd[k].shape != v.shape]:
        del sd[k]

    nn_model.load_state_dict(sd, strict=False)


def build_grod_detector(num_classes=1, pretrain_weights=None, device="cpu",
                        resolution=None):
    """Build the vendored LWDETR (+ GROD heads via env switches) and optionally
    warm-start from a local checkpoint. Returns the nn.Module in eval mode."""
    _register_seams()
    mc = RFDETRMediumConfig(num_classes=num_classes)
    if resolution is not None:
        mc.resolution = resolution
    args = build_namespace(mc, TrainConfig(dataset_dir=".", output_dir="."))
    args.pretrain_weights = pretrain_weights
    nn_model = build_model(args)
    if pretrain_weights is not None:
        _load_pretrain(nn_model, pretrain_weights, args.num_queries, args.group_detr)
    nn_model = nn_model.to(device).eval()
    # preprocessing meta the consumers need (matches RFDETRMedium's rf.model.resolution / rf.means / rf.stds)
    nn_model.resolution = mc.resolution
    nn_model.means = IMAGENET_MEANS
    nn_model.stds = IMAGENET_STDS
    return nn_model
