"""Single construction seam for the OAVLE detection front-end.

Every OAVLE / GROD entry point builds the encoder through this module, so the
stock ``rfdetr`` dependency is imported in exactly **one** place on the training /
extraction side (inference already runs on the vendored ``detector/`` decoder,
which imports no rfdetr at all). The GROD seams — swappable backbone, region /
global heads, losses — are applied by the ``rfdetr_patch`` monkeypatch on package
import; the env switches (``RFDETR_SEMANTIC_DIM`` / ``_ANCHORS`` /
``RFDETR_GLOBAL_DIM`` / ``RFDETR_BACKBONE``) are set by the caller *before*
calling in here.

Public surface:
  * ``OAVLE``         — the part-1 encoder as a first-class ``nn.Module``; one
                        forward → structured findings (box / objectness /
                        region-z / global) in the paper's vocabulary.
  * ``build_oavle``   — full builder object (use when you need the training
                        engine ``.train(...)`` or ``optimize_for_inference``).
  * ``load_oavle``    — build + unpack to ``(core, resolution, means, stds)``;
                        for probe / extraction scripts that reach into the core
                        submodules (semantic_embed / global_embed / backbone)
                        directly.
"""

from __future__ import annotations

import os

import torch
from torch import nn


def _backbone_from_ckpt(weights) -> str | None:
    """Infer the registered backbone name from a checkpoint's backbone weights.

    The backbone choice is not persisted in checkpoint metadata, yet the detector
    reads it from ``RFDETR_BACKBONE`` at build time — omitting it silently builds
    the stock DINOv2 and drops the mismatched DINOv3 tensors (garbage features, no
    error). We therefore derive it from the weights themselves: DINOv3's HF
    ``mask_token`` is rank-3 ``[1,1,D]`` (stock DINOv2 in rfdetr is rank-2
    ``[1,D]``); ``D`` selects the variant. Returns e.g. ``"dinov3_small"``, or
    ``None`` for the stock DINOv2 path (no env needed).
    """
    if weights is None:
        return None
    try:
        sd = torch.load(weights, map_location="cpu", weights_only=False)
    except Exception:
        return None
    sd = sd.get("model", sd.get("state_dict", sd)) if isinstance(sd, dict) else sd
    if not isinstance(sd, dict):
        return None
    mt = next((v for k, v in sd.items() if k.endswith("embeddings.mask_token")), None)
    if mt is None or mt.dim() != 3:
        return None
    return {384: "dinov3_small", 768: "dinov3_base", 1024: "dinov3_large"}.get(mt.shape[-1])


def _ensure_backbone(weights) -> None:
    """Auto-set ``RFDETR_BACKBONE`` from the checkpoint if the caller didn't.

    Explicit env wins (training sets it deliberately); otherwise we resolve it
    from the checkpoint so loaders never silently build the wrong backbone.
    """
    if os.environ.get("RFDETR_BACKBONE"):
        return
    bb = _backbone_from_ckpt(weights)
    if bb:
        os.environ["RFDETR_BACKBONE"] = bb
        print(f"[build] auto-set RFDETR_BACKBONE={bb} (from checkpoint)")


def build_oavle(weights, *, num_classes: int | None = 1, **kwargs):
    """Construct the OAVLE builder (detection front-end + GROD seams).

    Returns the full builder object so callers that need rfdetr's training engine
    (``.train(...)``) or ``optimize_for_inference`` keep working. For plain
    feature / structured-output extraction use :func:`load_oavle` instead.

    ``num_classes=None`` omits the argument (plain detection checkpoints carry
    their own head shape).
    """
    _ensure_backbone(weights)
    from rfdetr import RFDETRMedium  # the one and only training-side rfdetr import

    if num_classes is None:
        return RFDETRMedium(pretrain_weights=weights, **kwargs)
    return RFDETRMedium(pretrain_weights=weights, num_classes=num_classes, **kwargs)


def load_oavle(weights, *, device=None, num_classes: int | None = 1,
               eval_mode: bool = True, **kwargs):
    """Build and unpack to ``(core, resolution, means, stds)``.

    ``core`` is the underlying LWDETR ``nn.Module`` (box / objectness / region-z /
    global heads); the trailing three are the preprocessing meta the consumers
    need. Probe / extraction scripts reach into ``core`` submodules directly —
    that's intended; they stay at this lower level rather than the OAVLE wrapper.
    """
    rf = build_oavle(weights, num_classes=num_classes, **kwargs)
    core = rf.model.model
    if device is not None:
        core = core.to(device)
    if eval_mode:
        core = core.eval()
    return core, int(rf.model.resolution), list(rf.means), list(rf.stds)


class OAVLE(nn.Module):
    """Object-Aware Vision-Language Encoder — part 1 / Case Encoding front-end.

    Wraps the detection core and exposes one forward over an image batch as the
    structured findings in the paper's vocabulary. Two backends produce the same
    wrapper:

      * :meth:`from_pretrained` — builds via stock rfdetr (training / extraction).
      * :meth:`from_vendored`   — builds via the vendored ``detector/`` decoder
                                  (inference; no rfdetr import).

    ``forward`` is a passthrough to the core (native detector keys, byte-identical
    output) so existing consumers can adopt the wrapper without changing how they
    read the result; :meth:`encode` returns the translated ``{box, obj, z, g}``.
    """

    def __init__(self, core: nn.Module, resolution: int, means, stds):
        super().__init__()
        self.core = core
        self.resolution = int(resolution)
        self.means = list(means)
        self.stds = list(stds)

    @classmethod
    def from_pretrained(cls, weights, *, device=None, num_classes: int | None = 1,
                        eval_mode: bool = True, **kwargs) -> "OAVLE":
        core, res, means, stds = load_oavle(
            weights, device=device, num_classes=num_classes,
            eval_mode=eval_mode, **kwargs)
        return cls(core, res, means, stds)

    @classmethod
    def from_vendored(cls, weights, *, device="cpu", num_classes: int = 1,
                      resolution=None) -> "OAVLE":
        _ensure_backbone(weights)
        from diagnosis_model.grod.detector.build import build_grod_detector
        core = build_grod_detector(num_classes=num_classes, pretrain_weights=weights,
                                   device=device, resolution=resolution)
        return cls(core, core.resolution, core.means, core.stds)

    def forward(self, pixel_values: torch.Tensor) -> dict:
        """Passthrough to the core — native detector keys, byte-identical."""
        return self.core(pixel_values)

    def encode(self, pixel_values: torch.Tensor) -> dict:
        """One forward → structured findings in OAVLE's vocabulary.

        ``box``  [B,Q,4]   region boxes
        ``obj``  [B,Q]     objectness logits (channel 0 = ABNORMAL)
        ``z``    [B,Q,768] region semantic vectors
        ``g``    [B,768]   global case vector (``None`` if the global head is off)
        ``raw``            the native detector dict (passthrough)
        """
        out = self.core(pixel_values)
        return {
            "box": out["pred_boxes"],
            "obj": out["pred_logits"][..., 0],
            "z": out["pred_semantic"],
            "g": out.get("pred_global"),
            "raw": out,
        }
