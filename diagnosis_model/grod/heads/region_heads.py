"""OAVLE region + global projection heads (the "heads" the paper attributes to
our model), lifted out of the rfdetr fork into this repo.

They plug into rfdetr's (still-forked) LWDETR decoder via the generic
region-heads seam (`rfdetr.models.region_heads`): importing the package
registers a factory so that building an RFDETR model with the env switches
RFDETR_SEMANTIC_DIM>0 / RFDETR_GLOBAL_DIM>0 routes the heads here.

  - semantic head : Linear(hidden_dim -> semantic_dim) on the decoder query
                    features hs[-1] (or mean of last K layers) -> pred_semantic
                    (open-vocabulary lesion->symptom retrieval).
  - global head   : MLP(sum(backbone_out_channels) -> hidden_dim -> global_dim)
                    on the backbone-pooled pre-projector global vector
                    -> pred_global (case-level retrieval / CEAH).

Modules are attached to LWDETR under canonical names ("semantic_embed" /
"global_embed"), so existing joint checkpoints load unchanged. The loss math
(loss_semantic / loss_global) still lives in the fork's criterion for now and
reads these outputs — only the head architecture + forward emission moved here.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    """DETR-style FFN. Structurally identical to rfdetr's MLP (same `layers`
    ModuleList of Linears), so the global head's state_dict keys
    (global_embed.layers.*) match existing checkpoints either way."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class GrodRegionHeads:
    def __init__(self, semantic_dim=0, semantic_layers=1, global_dim=0):
        self.semantic_dim = int(semantic_dim)
        self.semantic_layers = int(semantic_layers)
        self.global_dim = int(global_dim)

    @property
    def needs_global_pool(self):
        return self.global_dim > 0

    def build(self, hidden_dim, backbone_out_channels):
        """Return the head modules to attach to LWDETR (under canonical names)."""
        mods = {}
        if self.semantic_dim > 0:
            lin = nn.Linear(hidden_dim, self.semantic_dim)
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            mods["semantic_embed"] = lin
        if self.global_dim > 0:
            global_in_dim = int(sum(backbone_out_channels))
            mods["global_embed"] = MLP(global_in_dim, self.global_dim, self.global_dim, 1)
        return mods

    def emit(self, model, hs, out):
        """Write pred_semantic / pred_global into the model output dict.

        L2-normalized to match the bank/aggregator/CEAH convention; idempotent
        w.r.t. loss_semantic which re-normalizes (fork criterion.py).
        """
        if self.semantic_dim > 0:
            k = self.semantic_layers
            hs_sem = hs[-1] if k == 1 else hs[-k:].mean(0)
            out["pred_semantic"] = F.normalize(model.semantic_embed(hs_sem), dim=-1)
        if self.global_dim > 0:
            out["pred_global"] = F.normalize(
                model.global_embed(model.backbone[0]._global_pooled), dim=-1
            )
