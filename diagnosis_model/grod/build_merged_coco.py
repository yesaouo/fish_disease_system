"""Build a merged COCO dataset dir for joint detection+semantic training.

Takes the detection COCO (class-agnostic single class ABNORMAL) as the base and
adds a `symptom_category_id` field to every annotation, joined from the
vl_classifier COCO (19 symptom categories) by (file_name, exact bbox). The two
COCOs share file_names 1:1 with byte-identical boxes (verified), so the join is
exact. category_id stays 0 (ABNORMAL) so the detector remains class-agnostic;
only the new field feeds loss_semantic.

Output dir mirrors the detection COCO layout (train/ valid/ with
_annotations.coco.json) and symlinks images, so RFDETR's build_coco consumes it
unchanged. Annotations whose symptom category cannot be joined get
symptom_category_id = -1 (loss_semantic ignores those).

Run from repo root:
  $PY -m diagnosis_model.grod.build_merged_coco \
      --det_root data/detection/coco/_merged \
      --vlc_root data/coco/_merged \
      --out_root data/detection/coco/_merged_semantic
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def build_lookup(vlc_coco_path: Path):
    with vlc_coco_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    id_to_fn = {im["id"]: im["file_name"] for im in coco["images"]}
    lookup = {}
    for a in coco["annotations"]:
        fn = id_to_fn.get(a["image_id"])
        if fn is None or "bbox" not in a:
            continue
        key = (fn, tuple(int(round(v)) for v in a["bbox"]))
        lookup[key] = int(a["category_id"])
    return lookup


def process_split(det_root: Path, vlc_root: Path, out_root: Path, split: str):
    det_json = det_root / split / "_annotations.coco.json"
    vlc_json = vlc_root / split / "_annotations.coco.json"
    if not det_json.exists():
        print(f"[skip] {split}: {det_json} missing")
        return

    lookup = build_lookup(vlc_json)
    with det_json.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    id_to_fn = {im["id"]: im["file_name"] for im in coco["images"]}

    n_join = n_miss = 0
    for a in coco["annotations"]:
        fn = id_to_fn.get(a["image_id"])
        key = (fn, tuple(int(round(v)) for v in a.get("bbox", [])))
        cat = lookup.get(key, -1)
        a["symptom_category_id"] = cat
        if cat >= 0:
            n_join += 1
        else:
            n_miss += 1

    out_split = out_root / split
    out_split.mkdir(parents=True, exist_ok=True)
    with (out_split / "_annotations.coco.json").open("w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)

    # symlink images (avoid copying ~GBs)
    src_img_dir = det_root / split
    for im in coco["images"]:
        fn = im["file_name"]
        src = (src_img_dir / fn).resolve()
        dst = out_split / fn
        if not dst.exists() and src.exists():
            os.symlink(src, dst)

    print(f"[{split}] anns={len(coco['annotations'])} joined={n_join} "
          f"missed(-1)={n_miss} -> {out_split}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det_root", type=str, required=True)
    ap.add_argument("--vlc_root", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--splits", type=str, nargs="+", default=["train", "valid", "test"])
    args = ap.parse_args()

    det_root = Path(args.det_root)
    vlc_root = Path(args.vlc_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        process_split(det_root, vlc_root, out_root, split)
    print(f"[done] merged COCO -> {out_root}")


if __name__ == "__main__":
    main()
