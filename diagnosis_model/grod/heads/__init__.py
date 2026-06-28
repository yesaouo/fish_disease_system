"""GROD region/global heads (OAVLE).

Pure impls + factories + a `register_heads()` helper; this package imports no
seam itself. Callers wire it to a seam: diagnosis_model.grod registers it against
the rfdetr-fork seams, the vendored detector builder against its own — so the
same heads serve both paths and the package stays decoder-agnostic.
"""

import os

import torch

from .losses import GrodRegionLosses
from .region_heads import GrodRegionHeads

# Activation switches read straight from env (the same RFDETR_* vars the GROD
# scripts already set), so the rfdetr fork needs no namespace patch to thread
# them onto args — the seam passes `args` but we ignore it.


def _heads_factory(args):
    sd = int(os.environ.get("RFDETR_SEMANTIC_DIM", "0"))
    gd = int(os.environ.get("RFDETR_GLOBAL_DIM", "0"))
    if sd <= 0 and gd <= 0:
        return None
    return GrodRegionHeads(
        semantic_dim=sd,
        semantic_layers=int(os.environ.get("RFDETR_SEMANTIC_LAYERS", "1")),
        global_dim=gd,
    )


def _losses_factory(args):
    sd = int(os.environ.get("RFDETR_SEMANTIC_DIM", "0"))
    gd = int(os.environ.get("RFDETR_GLOBAL_DIM", "0"))
    if sd <= 0 and gd <= 0:
        return None
    return GrodRegionLosses(
        semantic_dim=sd,
        semantic_anchors_path=os.environ.get("RFDETR_SEMANTIC_ANCHORS", None),
        semantic_temp=float(os.environ.get("RFDETR_SEMANTIC_TEMP", "0.07")),
        semantic_loss_coef=float(os.environ.get("RFDETR_SEMANTIC_LOSS_COEF", "1.0")),
        global_dim=gd,
        global_loss_coef=float(os.environ.get("RFDETR_GLOBAL_LOSS_COEF", "1.0")),
    )


def _symptom_labels(anno):
    # per-box symptom category for the semantic head (loss_semantic); -1 = ignore
    return torch.tensor(
        [o.get("symptom_category_id", -1) for o in anno], dtype=torch.int64
    )


def register_heads(reg_heads, reg_losses, reg_target=None):
    """Register the OAVLE heads/losses (+ symptom target) against arbitrary seams,
    so the vendored decoder can reuse the same impls as the rfdetr-fork path."""
    reg_heads(_heads_factory)
    reg_losses(_losses_factory)
    if reg_target is not None:
        reg_target("symptom_labels", _symptom_labels)


__all__ = ["GrodRegionHeads", "GrodRegionLosses", "register_heads"]
