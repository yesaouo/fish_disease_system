"""GROD swappable backbones.

Importing this package registers every owned backbone with rfdetr's external
encoder seam, so that constructing an RFDETR model with
`RFDETR_BACKBONE=dinov3_<variant>` (or bare `dinov3` = base) routes the backbone
to the implementation in this package while the decoder stays in rfdetr.

Import this before building the model:
    import diagnosis_model.grod.backbone   # noqa: F401  (side-effect: registration)
"""

from .dinov3 import DINOV3_VARIANTS, DinoV3Backbone


def _make(variant):
    def factory(out_feature_indexes=None, patch_size=16):
        return DinoV3Backbone(variant, out_feature_indexes, patch_size)
    return factory


# name -> factory, exposed so consumers (e.g. the vendored decoder's external
# seam) can register the same encoders without re-deriving the mapping.
BACKBONE_FACTORIES = {f"dinov3_{v}": _make(v) for v in DINOV3_VARIANTS}
BACKBONE_FACTORIES["dinov3"] = _make("base")   # bare alias -> base


def register_backbones(register_fn):
    """Register all owned backbones against an arbitrary encoder seam's register fn.
    (Called by diagnosis_model.grod for the rfdetr seam, and by the vendored
    detector builder for its own seam — this package itself imports no seam.)"""
    for name, fac in BACKBONE_FACTORIES.items():
        register_fn(name, fac)


__all__ = ["DinoV3Backbone", "DINOV3_VARIANTS", "BACKBONE_FACTORIES", "register_backbones"]
