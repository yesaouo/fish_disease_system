# ------------------------------------------------------------------------
# Generic region/global-loss seam (criterion side).
#
# Companion to region_heads.py: lets a downstream project supply the loss math
# (+ frozen target buffers) for its extra heads, without this package knowing
# what they are. A registered factory(args) returns a plugin nn.Module (or None)
# implementing:
#
#   .loss_names      : list[str]   e.g. ["semantic", "global"] (only enabled)
#   .last_layer_only : set[str]    losses to skip on aux/enc decoder layers
#   .weights         : dict[str, float]  weight_dict entries, e.g. {"loss_semantic": 2.0}
#   .compute(name, criterion, outputs, targets, indices, num_boxes) -> dict
#         the loss value(s); may call criterion._get_src_permutation_idx(indices).
#
# Being an nn.Module attached to SetCriterion, its registered buffers (frozen
# text anchors etc.) move with criterion.to(device). With no factory registered,
# build_region_losses returns None and the criterion has no extra losses.
# ------------------------------------------------------------------------

_FACTORY = None


def register_region_losses(factory):
    """Register a callable factory(args) -> plugin nn.Module | None."""
    global _FACTORY
    _FACTORY = factory


def build_region_losses(args):
    if _FACTORY is None:
        return None
    return _FACTORY(args)
