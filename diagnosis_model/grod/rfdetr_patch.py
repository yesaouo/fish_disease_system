"""Apply the GROD seams to a STOCK (pip-installed, unmodified) rfdetr at runtime.

The user's intent: keep using the rfdetr package as-is (incl. its training engine
— EMA, LR schedule, mAP eval), without forking/editing its source. This module
re-expresses the former fork diff as idempotent monkeypatches applied on import:

  1. backbone   : external-encoder branch (RFDETR_BACKBONE) + pre-projector global pool
  2. region heads: attach semantic_embed/global_embed; emit pred_semantic/pred_global
                   (decoder query hs captured via a transformer forward-hook)
  3. region losses: criterion dispatch for loss_semantic/loss_global (+ frozen anchors)
  4. coco        : per-box symptom_category_id -> target["symptom_labels"]
  5. weights     : drop shape-mismatched pretrain tensors (backbone-swap safe load)

Seam registries + impls are reused from this repo: the vendored detector's seam
modules act as the registry, and diagnosis_model.grod.{backbone,heads} provide the
factories. Activate with:  import diagnosis_model.grod.rfdetr_patch  (idempotent).
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from diagnosis_model.grod.backbone import register_backbones
from diagnosis_model.grod.heads import register_heads, _symptom_labels

_PATCHED = False

# Lightweight local seam registries (avoid importing the vendored decoder just
# for these). Populated from this repo's backbone/heads factories.
_ENC = {}            # name -> encoder factory(out_feature_indexes, patch_size)
_HEADS = [None]      # [heads factory(args)]
_LOSSES = [None]     # [losses factory(args)]


def build_external_encoder(name, **kw):
    if name not in _ENC:
        raise KeyError(f"no external encoder '{name}' (have {sorted(_ENC)})")
    return _ENC[name](**kw)


def build_region_heads(args):
    return _HEADS[0](args) if _HEADS[0] is not None else None


def build_region_losses(args):
    return _LOSSES[0](args) if _LOSSES[0] is not None else None


def _ensure_registered():
    register_backbones(lambda n, f: _ENC.__setitem__(n, f))
    register_heads(lambda f: _HEADS.__setitem__(0, f),
                   lambda f: _LOSSES.__setitem__(0, f))


# --------------------------------------------------------------------------- #
# 1. Backbone — external encoder + global pool (REPLACE __init__/forward)
# --------------------------------------------------------------------------- #
def _patch_backbone():
    from rfdetr.models.backbone.backbone import Backbone
    from rfdetr.models.backbone.dinov2 import DinoV2
    from rfdetr.models.backbone.projector import MultiScaleProjector
    from rfdetr.utilities.tensors import NestedTensor

    def __init__(self, name, pretrained_encoder=None, window_block_indexes=None,
                 drop_path=0.0, out_channels=256, out_feature_indexes=None,
                 projector_scale=None, use_cls_token=False, freeze_encoder=False,
                 layer_norm=False, target_shape=(640, 640), rms_norm=False,
                 backbone_lora=False, gradient_checkpointing=False,
                 load_dinov2_weights=True, patch_size=14, num_windows=4,
                 positional_encoding_size=0):
        from rfdetr.models.backbone.base import BackboneBase
        BackboneBase.__init__(self)
        _bb = os.environ.get("RFDETR_BACKBONE", "")
        if _bb:
            self.encoder = build_external_encoder(
                _bb, out_feature_indexes=out_feature_indexes, patch_size=patch_size)
        else:
            name_parts = name.split("_")
            assert name_parts[0] == "dinov2"
            use_registers = "registers" in name_parts
            if use_registers:
                name_parts.remove("registers")
            use_windowed_attn = "windowed" in name_parts
            if use_windowed_attn:
                name_parts.remove("windowed")
            assert len(name_parts) == 2
            self.encoder = DinoV2(
                size=name_parts[-1], out_feature_indexes=out_feature_indexes,
                shape=target_shape, use_registers=use_registers,
                use_windowed_attn=use_windowed_attn,
                gradient_checkpointing=gradient_checkpointing,
                load_dinov2_weights=load_dinov2_weights, patch_size=patch_size,
                num_windows=num_windows, positional_encoding_size=positional_encoding_size,
                drop_path_rate=drop_path)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.projector_scale = projector_scale
        assert len(self.projector_scale) > 0
        assert sorted(self.projector_scale) == self.projector_scale
        level2scalefactor = dict(P3=2.0, P4=1.0, P5=0.5, P6=0.25)
        scale_factors = [level2scalefactor[lvl] for lvl in self.projector_scale]
        self.projector = MultiScaleProjector(
            in_channels=self.encoder._out_feature_channels, out_channels=out_channels,
            scale_factors=scale_factors, layer_norm=layer_norm, rms_norm=rms_norm)
        self._export = False
        self.global_pool = False
        self._global_pooled = None

    @staticmethod
    def _pool_global(feats, tensor_list):
        m = getattr(tensor_list, "mask", None)
        per = []
        for f in feats:
            if m is not None:
                valid = (~F.interpolate(m[None].float(), size=f.shape[-2:]).to(torch.bool)[0])
                valid = valid.unsqueeze(1).to(f.dtype)
                pooled = (f * valid).sum(dim=(2, 3)) / valid.sum(dim=(2, 3)).clamp_min(1.0)
            else:
                pooled = f.mean(dim=(2, 3))
            per.append(F.normalize(pooled, dim=-1))
        return torch.cat(per, dim=-1)

    def forward(self, tensor_list):
        feats = self.encoder(tensor_list.tensors)
        if self.global_pool:
            _pool = getattr(self.encoder, "pool_global", None)
            self._global_pooled = (_pool(feats, tensor_list) if _pool is not None
                                   else self._pool_global(feats, tensor_list))
        feats = self.projector(feats)
        out = []
        for feat in feats:
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            out.append(NestedTensor(feat, mask))
        return out

    Backbone.__init__ = __init__
    Backbone._pool_global = _pool_global
    Backbone.forward = forward


# --------------------------------------------------------------------------- #
# 2. LWDETR — attach heads (WRAP __init__) + emit (WRAP forward via hs hook)
# --------------------------------------------------------------------------- #
def _patch_lwdetr():
    from rfdetr.models.lwdetr import LWDETR

    orig_init = LWDETR.__init__
    orig_forward = LWDETR.forward

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        plugin = build_region_heads(None)   # reads env switches
        self.region_heads = plugin
        if plugin is not None:
            hidden_dim = self.bbox_embed.layers[0].in_features
            ch = self.backbone[0].encoder._out_feature_channels
            for name, m in plugin.build(hidden_dim, ch).items():
                setattr(self, name, m)      # semantic_embed / global_embed
            if plugin.needs_global_pool:
                self.backbone[0].global_pool = True
            # capture decoder query features (hs = transformer output[0])
            self._hs_cache = None
            self.transformer.register_forward_hook(
                lambda mod, inp, out: setattr(self, "_hs_cache", out[0]))

    def forward(self, samples, targets=None):
        out = orig_forward(self, samples, targets)
        if getattr(self, "region_heads", None) is not None and self._hs_cache is not None \
                and isinstance(out, dict) and "pred_logits" in out:
            self.region_heads.emit(self, self._hs_cache, out)
        return out

    LWDETR.__init__ = __init__
    LWDETR.forward = forward

    # weight-loader shape filter (covers both detr.py and training loaders, which
    # both call model.load_state_dict) — drop tensors that no longer match shape.
    orig_load = LWDETR.load_state_dict

    def load_state_dict(self, state_dict, strict=True, assign=False):
        msd = self.state_dict()
        dropped = [k for k, v in state_dict.items() if k in msd and msd[k].shape != v.shape]
        if dropped:
            state_dict = {k: v for k, v in state_dict.items() if k not in dropped}
            print(f"[rfdetr_patch] dropped {len(dropped)} shape-mismatched pretrain "
                  f"tensors (backbone swap): e.g. {dropped[:3]}")
        return orig_load(self, state_dict, strict=strict, assign=assign)

    LWDETR.load_state_dict = load_state_dict


# --------------------------------------------------------------------------- #
# 3. SetCriterion — region-loss dispatch (WRAP get_loss) + build wiring
# --------------------------------------------------------------------------- #
def _patch_criterion():
    import rfdetr.models.lwdetr as lw
    from rfdetr.models.criterion import SetCriterion

    orig_get_loss = SetCriterion.get_loss

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kw):
        rl = getattr(self, "region_losses", None)
        if rl is not None and loss in rl.loss_names:
            # aux/enc layers lack pred_semantic/pred_global -> skip (return {})
            need = "pred_semantic" if loss == "semantic" else "pred_global"
            if need not in outputs:
                return {}
            return rl.compute(loss, self, outputs, targets, indices, num_boxes)
        return orig_get_loss(self, loss, outputs, targets, indices, num_boxes, **kw)

    SetCriterion.get_loss = get_loss

    orig_build = lw.build_criterion_and_postprocessors

    def build_criterion_and_postprocessors(args):
        criterion, postprocess = orig_build(args)
        rl = build_region_losses(None)   # reads env switches
        if rl is not None:
            criterion.region_losses = rl.to(next(criterion.parameters()).device) \
                if any(True for _ in criterion.parameters()) else rl
            criterion.losses = list(criterion.losses) + list(rl.loss_names)
            criterion.weight_dict.update(rl.weights)
        return criterion, postprocess

    lw.build_criterion_and_postprocessors = build_criterion_and_postprocessors


# --------------------------------------------------------------------------- #
# 4. ConvertCoco — extra per-box target fields (REPLACE __call__)
# --------------------------------------------------------------------------- #
def _patch_coco():
    from rfdetr.datasets.coco import ConvertCoco

    orig_call = ConvertCoco.__call__

    def __call__(self, image, target):
        anno = [o for o in target.get("annotations", [])
                if "iscrowd" not in o or o["iscrowd"] == 0]
        symptom = _symptom_labels(anno) if anno else torch.zeros(0, dtype=torch.int64)
        image, t = orig_call(self, image, target)
        # re-derive keep from kept boxes count is unreliable; instead recompute the
        # exact keep mask the same way ConvertCoco does, then align symptom_labels.
        import torchvision
        w, h = image.size if hasattr(image, "size") else (None, None)
        # Fallback: align by recomputing boxes/keep from anno (matches ConvertCoco).
        if anno:
            boxes = torch.as_tensor([o["bbox"] for o in anno], dtype=torch.float32).reshape(-1, 4)
            boxes[:, 2:] += boxes[:, :2]
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            t["symptom_labels"] = symptom[keep]
        else:
            t["symptom_labels"] = symptom
        return image, t

    ConvertCoco.__call__ = __call__


def apply_patches():
    global _PATCHED
    if _PATCHED:
        return
    _ensure_registered()
    _patch_backbone()
    _patch_lwdetr()
    _patch_criterion()
    _patch_coco()
    _PATCHED = True


apply_patches()
