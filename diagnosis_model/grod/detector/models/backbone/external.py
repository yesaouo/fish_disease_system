# ------------------------------------------------------------------------
# Generic external-encoder seam.
#
# Lets a downstream project supply its OWN backbone encoder (e.g. a DINOv3
# variant) without this package knowing anything about it. The encoder must
# expose the same interface the rfdetr Backbone wrapper consumes:
#
#   - __call__(pixel_values) -> list[Tensor[B, C, H/p, W/p]]   (one per tap)
#   - ._out_feature_channels : list[int]                       (channel per tap)
#
# Selected at build time via env var RFDETR_BACKBONE=<name>; the name must have
# been registered (the registering module must be imported before the model is
# built). With RFDETR_BACKBONE unset, Backbone uses the stock DinoV2 path and
# this registry is never touched (byte-identical default behaviour).
# ------------------------------------------------------------------------

_REGISTRY = {}


def register_encoder(name, factory):
    """Register a callable factory(out_feature_indexes, patch_size) -> nn.Module."""
    _REGISTRY[name] = factory


def is_registered(name):
    return name in _REGISTRY


def build_external_encoder(name, **kwargs):
    if name not in _REGISTRY:
        raise KeyError(
            f"No external encoder registered under '{name}'. Import the module "
            f"that registers it before constructing the model (registered: "
            f"{sorted(_REGISTRY)})."
        )
    return _REGISTRY[name](**kwargs)
