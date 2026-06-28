"""OAVLE region/global-head supervision (loss math + frozen text targets),
lifted out of the rfdetr fork's criterion into this repo.

Plugs into the fork's criterion via the generic region-losses seam
(`rfdetr.models.region_losses`). Attached as a submodule of SetCriterion so its
registered buffers (frozen SigLIP2 text anchors) follow criterion.to(device).
The math is verbatim the fork's former loss_semantic / loss_global.

  - semantic : matched-query pred_semantic -> GT symptom-category frozen text
               anchor, multi-positive sigmoid (lesion<->caption). anchor mode
               (1 positive/query, /temp) or bank mode (full caption bank, SigLIP
               native logit_scale/bias) — auto-selected by the anchors pack.
  - global   : image-level pred_global -> frozen whole-image target (e.g. raw
               SigLIP2 global), cosine distillation. (In joint training no
               global_target is supplied, so this is a no-op there; the global
               head is produced by offline distillation, see distill_global_mlp.)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class GrodRegionLosses(nn.Module):
    def __init__(self, semantic_dim=0, semantic_anchors_path=None,
                 semantic_temp=0.07, semantic_loss_coef=1.0,
                 semantic_label_smoothing=0.05,
                 global_dim=0, global_loss_coef=1.0):
        super().__init__()
        self._loss_names = []
        self.weights = {}

        self.semantic_dim = int(semantic_dim)
        self.semantic_temp = semantic_temp
        self.semantic_loss_coef = semantic_loss_coef
        self.semantic_label_smoothing = semantic_label_smoothing
        self.semantic_logit_scale = None
        self.semantic_logit_bias = None
        anchors = None          # frozen text targets [K, D] (None when disabled)
        bank_labels = None      # [K] per-anchor category (bank mode only)
        if self.semantic_dim > 0:
            if semantic_anchors_path is None:
                raise ValueError("semantic_dim>0 requires semantic_anchors_path")
            pack = torch.load(semantic_anchors_path, map_location="cpu", weights_only=False)
            if isinstance(pack, dict) and "bank_embs" in pack:
                # bank mode: full per-caption bank (true multipos)
                anchors = pack["bank_embs"]
                bank_labels = pack["bank_labels"]
            else:
                # anchor mode: category-mean anchors
                anchors = pack["anchor_embs"] if isinstance(pack, dict) else pack
            # calibration orthogonal to target set: native if scale present, else /temp
            if isinstance(pack, dict) and pack.get("logit_scale") is not None:
                self.semantic_logit_scale = float(pack["logit_scale"])
                self.semantic_logit_bias = float(pack["logit_bias"])
            self._loss_names.append("semantic")
            self.weights["loss_semantic"] = semantic_loss_coef
        # register (possibly None) so the tensors follow criterion.to(device)
        self.register_buffer("semantic_text_anchors", anchors)
        self.register_buffer("semantic_bank_labels", bank_labels)

        self.global_dim = int(global_dim)
        if self.global_dim > 0:
            self._loss_names.append("global")
            self.weights["loss_global"] = global_loss_coef

    @property
    def loss_names(self):
        return self._loss_names

    @property
    def last_layer_only(self):
        # semantic/global are supervised on the last decoder layer only
        return set(self._loss_names)

    def compute(self, name, criterion, outputs, targets, indices, num_boxes):
        if name == "semantic":
            return self._loss_semantic(criterion, outputs, targets, indices)
        if name == "global":
            return self._loss_global(outputs, targets)
        raise KeyError(name)

    def _loss_semantic(self, criterion, outputs, targets, indices):
        assert "pred_semantic" in outputs
        assert self.semantic_text_anchors is not None
        idx = criterion._get_src_permutation_idx(indices)
        z = outputs["pred_semantic"][idx]                       # [M, D]
        target_cat = torch.cat(
            [t["symptom_labels"][J] for t, (_, J) in zip(targets, indices)]
        )                                                        # [M]

        valid = target_cat >= 0
        if valid.sum() == 0:
            return {"loss_semantic": z.sum() * 0.0}
        z = F.normalize(z[valid], dim=-1)                       # [Mv, D]
        target_cat = target_cat[valid]

        anchors = F.normalize(self.semantic_text_anchors.to(z.dtype), dim=-1)  # [K, D]
        if self.semantic_bank_labels is not None:
            anchor_cat = self.semantic_bank_labels.to(z.device)
        else:
            anchor_cat = torch.arange(anchors.size(0), device=z.device)
        if self.semantic_logit_scale is not None:
            logits = self.semantic_logit_scale * (z @ anchors.t()) + self.semantic_logit_bias
        else:
            logits = (z @ anchors.t()) / self.semantic_temp        # [Mv, K]

        pos = (target_cat.view(-1, 1) == anchor_cat.view(1, -1))
        ls = self.semantic_label_smoothing
        tgt = pos.float() * (1 - ls) + 0.5 * ls
        element = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none")
        pos_m = pos.float()
        neg_m = (~pos).float()
        pos_loss = (element * pos_m).sum(1) / pos_m.sum(1).clamp_min(1.0)
        neg_loss = (element * neg_m).sum(1) / neg_m.sum(1).clamp_min(1.0)
        loss = (0.5 * (pos_loss + neg_loss)).mean()
        return {"loss_semantic": loss}

    def _loss_global(self, outputs, targets):
        assert "pred_global" in outputs
        pred = F.normalize(outputs["pred_global"], dim=-1)          # [B, D]
        has = torch.tensor(
            ["global_target" in t for t in targets], device=pred.device
        )
        if has.sum() == 0:
            return {"loss_global": pred.sum() * 0.0}
        tgt = torch.stack([
            t["global_target"] if "global_target" in t
            else torch.zeros(pred.size(-1)) for t in targets
        ]).to(pred)                                                 # [B, D]
        tgt = F.normalize(tgt, dim=-1)
        cos = (pred * tgt).sum(-1)                                  # [B]
        loss = (1 - cos)[has].mean()
        return {"loss_global": loss}
