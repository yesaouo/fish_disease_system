"""data/detection_voc (20 類 COCO) -> data/detection_voc_ca (class-agnostic)。

A1 OAVLE 用:偵測退化成單一 OBJECT 類（class-agnostic，比照魚病 ABNORMAL），每個 box
保留其 VOC 類別作為語意頭目標 `symptom_category_id`（0..19，對應 CLASSES 順序 =
voc_text_anchors 的 anchor 索引）。偵測器只找物件、20 類交給 z·文字錨判定。
"""
import json
import os

SRC = "data/detection_voc"
DST = "data/detection_voc_ca"

for split in ("train", "valid"):
    src, dst = f"{SRC}/{split}", f"{DST}/{split}"
    os.makedirs(dst, exist_ok=True)
    d = json.load(open(f"{src}/_annotations.coco.json"))
    d["categories"] = [{"id": 0, "name": "OBJECT", "supercategory": "OBJECT"}]
    for a in d["annotations"]:
        a["symptom_category_id"] = a["category_id"] - 1   # COCO 1..20 -> VOC 類 0..19
        a["category_id"] = 0                              # class-agnostic 偵測
    json.dump(d, open(f"{dst}/_annotations.coco.json", "w"))
    for im in d["images"]:
        link = f"{dst}/{im['file_name']}"
        if not os.path.islink(link) and not os.path.exists(link):
            os.symlink(os.path.abspath(f"{src}/{im['file_name']}"), link)
    print(f"{split}: {len(d['images'])} imgs, {len(d['annotations'])} boxes -> {dst}")
