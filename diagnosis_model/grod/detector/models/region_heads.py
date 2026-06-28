# ------------------------------------------------------------------------
# Generic region/global-head seam.
#
# Lets a downstream project supply its OWN extra heads on the decoder query
# features (hs) and the pooled backbone global vector, without this package
# knowing what they are. A registered factory(args) returns a plugin object
# (or None) implementing:
#
#   .needs_global_pool : bool
#   .build(hidden_dim, backbone_out_channels) -> dict[str, nn.Module]
#         modules are attached to LWDETR under their dict keys (so checkpoint
#         state_dict keys stay stable, e.g. "semantic_embed", "global_embed").
#   .emit(model, hs, out) -> None
#         mutate the output dict, e.g. out["pred_semantic"] / out["pred_global"].
#
# Factory must be registered before the model is built (import the registering
# module first). With no factory registered, build_region_heads returns None
# and LWDETR is the plain detector (byte-identical).
# ------------------------------------------------------------------------

_FACTORY = None


def register_region_heads(factory):
    """Register a callable factory(args) -> plugin | None."""
    global _FACTORY
    _FACTORY = factory


def build_region_heads(args):
    if _FACTORY is None:
        return None
    return _FACTORY(args)
