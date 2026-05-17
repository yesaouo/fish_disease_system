"""Shared bbox containment check.

Originally duplicated as `bbox_contains` (xywh) in process_coco_gemini.py and
`is_box_inside` (xyxy) in anotation2coco.py. Same logic, different formats —
folded together with an explicit `fmt` argument so call sites can't silently
disagree.
"""

from __future__ import annotations

from typing import Literal, Sequence

BBoxFormat = Literal["xywh", "xyxy"]


def _to_xyxy(box: Sequence[float], fmt: BBoxFormat) -> tuple[float, float, float, float]:
    if fmt == "xyxy":
        x1, y1, x2, y2 = box
        return float(x1), float(y1), float(x2), float(y2)
    if fmt == "xywh":
        x, y, w, h = box
        return float(x), float(y), float(x) + float(w), float(y) + float(h)
    raise ValueError(f"unsupported bbox format: {fmt!r}")


def bbox_contains(
    inner: Sequence[float],
    outer: Sequence[float],
    *,
    fmt: BBoxFormat,
) -> bool:
    """Return True iff `inner` is fully inside `outer`. Both boxes must use `fmt`."""
    ix1, iy1, ix2, iy2 = _to_xyxy(inner, fmt)
    ox1, oy1, ox2, oy2 = _to_xyxy(outer, fmt)
    return ix1 >= ox1 and iy1 >= oy1 and ix2 <= ox2 and iy2 <= oy2
