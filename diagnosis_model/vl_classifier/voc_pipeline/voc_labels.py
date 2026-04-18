from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

VOC_CLASSES_EN: List[str] = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

VOC_CLASSES_ZH: Dict[str, str] = {
    "aeroplane": "飛機",
    "bicycle": "腳踏車",
    "bird": "鳥",
    "boat": "船",
    "bottle": "瓶子",
    "bus": "公車",
    "car": "汽車",
    "cat": "貓",
    "chair": "椅子",
    "cow": "牛",
    "diningtable": "餐桌",
    "dog": "狗",
    "horse": "馬",
    "motorbike": "機車",
    "person": "人",
    "pottedplant": "盆栽",
    "sheep": "羊",
    "sofa": "沙發",
    "train": "火車",
    "tvmonitor": "螢幕",
}

VOC_CLASS_TO_ID = {name: idx for idx, name in enumerate(VOC_CLASSES_EN)}
VOC_ID_TO_CLASS = {idx: name for idx, name in enumerate(VOC_CLASSES_EN)}


def build_default_voc_label_bank() -> dict:
    label_map = {}
    data = {}
    for idx, en_name in enumerate(VOC_CLASSES_EN):
        zh_name = VOC_CLASSES_ZH.get(en_name, en_name)
        article = "an" if en_name[0].lower() in "aeiou" else "a"
        label_map[str(idx)] = {"en": en_name, "zh": zh_name}
        data[str(idx)] = {
            "captions_en": [
                en_name,
                f"{article} {en_name}",
                f"a photo of {article} {en_name}",
            ]
        }
    return {"label_map": label_map, "data": data}


def save_default_voc_label_bank(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_default_voc_label_bank()
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


if __name__ == "__main__":
    out = Path(__file__).with_name("voc_label_bank.json")
    save_default_voc_label_bank(out)
    print(f"Saved: {out}")
