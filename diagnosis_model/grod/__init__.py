"""GROD package.

Importing it registers this repo's swappable backbone + OAVLE region/global heads
with rfdetr's generic seams (`rfdetr.models.backbone.external` /
`rfdetr.models.region_heads`). So any grod entry point that builds an RFDETR model
with the env switches (RFDETR_BACKBONE / RFDETR_SEMANTIC_DIM / RFDETR_GLOBAL_DIM)
gets the implementations from here rather than from the fork. The factories return
None / are untouched when the switches are unset (plain detector, byte-identical).
"""

from . import backbone  # noqa: F401  swappable backbones (DINOv3 variants)
from . import heads      # noqa: F401  OAVLE region/global heads

# Apply the GROD seams to the STOCK (unmodified) rfdetr package at runtime, so
# `python -m diagnosis_model.grod.<script>` entry points that build via
# RFDETRMedium (training, extraction) get the backbone-swap + region heads/losses
# + coco symptom field without forking rfdetr. Guarded so the vendored-decoder
# inference path still imports in an rfdetr-absent environment.
try:
    from . import rfdetr_patch  # noqa: F401  (side-effect: monkeypatch on import)
except ImportError:
    pass

# Public model surface. Build the OAVLE encoder through these — never construct
# the underlying detector directly — so the stock rfdetr dependency stays behind
# one seam (build.py) and entry points read as building OAVLE, not rfdetr.
from .build import OAVLE, build_oavle, load_oavle  # noqa: E402,F401

__all__ = ["OAVLE", "build_oavle", "load_oavle"]
