from __future__ import annotations
import argparse
import json
import os
import shutil
from collections import defaultdict
from typing import Dict, List, Any, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="一次把 COCO 的 train/val/test 轉成 YOLOv8 格式，並建立 data.yaml。"
    )
    p.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="原始 COCO 資料集根目錄，底下應該有 train/、val/、valid/、test/ 等資料夾",
    )
    p.add_argument(
        "--out_root",
        type=str,
        required=True,
        help="YOLO 格式輸出的根目錄，會在這裡建立 images/ 和 labels/ 子資料夾",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="要處理的資料切分名稱 (資料夾名稱)，例如: train valid test",
    )
    p.add_argument(
        "--json_name",
        type=str,
        default="_annotations.coco.json",
        help="各 split 資料夾裡標註檔的檔名，預設為 _annotations.coco.json",
    )
    p.add_argument(
        "--yaml_path",
        type=str,
        default=None,
        help="輸出的 data.yaml 路徑，預設為 out_root/data.yaml",
    )
    p.add_argument(
        "--txt_path",
        type=str,
        default=None,
        help="輸出的 classes.txt 路徑，預設為 out_root/classes.txt",
    )
    return p.parse_args()


def load_coco(coco_path: str) -> Dict[str, Any]:
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    return coco


def build_category_mapping(categories: List[Dict[str, Any]]) -> Dict[int, int]:
    """
    建立 COCO category_id -> YOLO class_id (0-based) 的映射
    """
    sorted_cats = sorted(categories, key=lambda c: c["id"])
    cat_id_to_yolo_id: Dict[int, int] = {}
    for yolo_id, cat in enumerate(sorted_cats):
        cat_id_to_yolo_id[cat["id"]] = yolo_id
    return cat_id_to_yolo_id


def invert_category_mapping(
    categories: List[Dict[str, Any]], cat_id_to_yolo_id: Dict[int, int]
) -> Dict[int, str]:
    """
    回傳 yolo_id -> category_name 的 mapping，方便寫 names
    """
    yolo_to_name: Dict[int, str] = {}
    for cat in categories:
        coco_id = cat["id"]
        name = cat["name"]
        yolo_id = cat_id_to_yolo_id[coco_id]
        yolo_to_name[yolo_id] = name
    return yolo_to_name


def convert_bbox(
    bbox: List[float], img_w: int, img_h: int
) -> List[float]:
    """
    COCO bbox: [x_min, y_min, w, h] (pixel)
    YOLO bbox: [x_center/img_w, y_center/img_h, w/img_w, h/img_h]
    """
    x_min, y_min, w, h = bbox
    x_center = x_min + w / 2.0
    y_center = y_min + h / 2.0

    return [
        x_center / img_w,
        y_center / img_h,
        w / img_w,
        h / img_h,
    ]


def write_data_yaml(
    yaml_path: str, out_root: str, yolo_splits: List[str], yolo_id_to_name: Dict[int, str]
) -> None:
    """
    依照 splits 以及類別名稱，輸出 YOLO data.yaml。
    這裡把 path 設為絕對路徑 out_root，避免工作目錄不同時路徑錯亂。
    """
    # 把 Windows 路徑換成通用的 /，避免 YAML 看起來怪怪的
    norm_out_root = out_root.replace("\\", "/")

    lines: List[str] = []
    lines.append(f"path: {norm_out_root}")

    if "train" in yolo_splits:
        lines.append("train: images/train")
    if "val" in yolo_splits:
        lines.append("val: images/val")
    if "test" in yolo_splits:
        lines.append("test: images/test")

    lines.append("")
    lines.append("names:")
    for yolo_id in sorted(yolo_id_to_name.keys()):
        name = yolo_id_to_name[yolo_id]
        lines.append(f"  {yolo_id}: {name}")

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_classes_txt(path: str, yolo_id_to_name: Dict[int, str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for yolo_id in sorted(yolo_id_to_name.keys()):
            name = yolo_id_to_name[yolo_id]
            f.write(f"{yolo_id}: {name}\n")


def map_split_to_yolo(split: str) -> str:
    """
    把資料夾名稱 map 成 YOLO 內部使用的 split 名稱
    e.g. valid / validation -> val
    """
    s = split.lower()
    if s in ("val", "valid", "validation"):
        return "val"
    if s == "train":
        return "train"
    if s == "test":
        return "test"
    # 其他名稱就照原樣用
    return split


def process_split(
    split_dir_name: str,
    yolo_split_name: str,
    root_dir: str,
    out_root: str,
    json_name: str,
    cat_id_to_yolo_id: Dict[int, int],
) -> None:
    """
    處理單一 split：讀 COCO json -> 寫 YOLO txt -> 複製影像

    split_dir_name: 原始資料夾名稱 (e.g. 'valid')
    yolo_split_name: 要在 YOLO 結構裡使用的名稱 (e.g. 'val')
    """
    coco_json_path = os.path.join(root_dir, split_dir_name, json_name)
    if not os.path.exists(coco_json_path):
        raise FileNotFoundError(f"{split_dir_name} 找不到標註檔: {coco_json_path}")

    labels_dir = os.path.join(out_root, "labels", yolo_split_name)
    images_out_dir = os.path.join(out_root, "images", yolo_split_name)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_out_dir, exist_ok=True)

    coco = load_coco(coco_json_path)
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    if not images:
        raise ValueError(f"{split_dir_name} 的 COCO JSON 裡沒有 images。")
    if not annotations:
        print(f"警告: {split_dir_name} 沒有任何 annotations，仍會複製影像但不會產生 bbox。")

    img_id_to_anns: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        img_id_to_anns[ann["image_id"]].append(ann)

    for img in images:
        img_id = img["id"]
        file_name = img["file_name"]
        img_w = img.get("width", None)
        img_h = img.get("height", None)

        if img_w is None or img_h is None:
            raise ValueError(
                f"影像 {file_name} 沒有 width/height 欄位，"
                "此腳本目前假設 COCO JSON 的 images 內有寬高資訊。"
            )

        anns = img_id_to_anns.get(img_id, [])
        yolo_lines: List[str] = []

        for ann in anns:
            coco_cat_id = ann["category_id"]
            if coco_cat_id not in cat_id_to_yolo_id:
                continue

            yolo_cls_id = cat_id_to_yolo_id[coco_cat_id]
            bbox = ann["bbox"]
            x_c, y_c, w, h = convert_bbox(bbox, img_w, img_h)
            line = f"{yolo_cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
            yolo_lines.append(line)

        base_name = os.path.splitext(os.path.basename(file_name))[0]
        label_path = os.path.join(labels_dir, base_name + ".txt")
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

        # 假設影像就放在 root_dir/split_dir_name 底下
        src_img_path = os.path.join(root_dir, split_dir_name, file_name)

        if not os.path.exists(src_img_path):
            raise FileNotFoundError(f"找不到影像檔: {src_img_path}")

        dst_img_path = os.path.join(images_out_dir, os.path.basename(file_name))
        if not os.path.exists(dst_img_path):
            shutil.copy2(src_img_path, dst_img_path)

    print(f"[{split_dir_name} -> {yolo_split_name}] 已輸出 YOLO labels 至: {labels_dir}")
    print(f"[{split_dir_name} -> {yolo_split_name}] 已複製影像至: {images_out_dir}")


def main() -> None:
    args = parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    # 建立 (原始資料夾名稱, yolo split 名稱) 對應
    split_pairs: List[Tuple[str, str]] = []
    yolo_splits_set = set()
    for s in args.splits:
        y = map_split_to_yolo(s)
        split_pairs.append((s, y))
        yolo_splits_set.add(y)

    # 先用第一個 split 建 category 映射，假設所有 split 類別一致
    first_split, _ = split_pairs[0]
    first_coco_path = os.path.join(args.root_dir, first_split, args.json_name)
    if not os.path.exists(first_coco_path):
        raise FileNotFoundError(f"找不到 {first_split} 的標註檔: {first_coco_path}")

    first_coco = load_coco(first_coco_path)
    categories = first_coco.get("categories", [])
    if not categories:
        raise ValueError("COCO JSON 裡沒有 categories。")

    cat_id_to_yolo_id = build_category_mapping(categories)
    yolo_id_to_name = invert_category_mapping(categories, cat_id_to_yolo_id)

    print("共有類別數:", len(yolo_id_to_name))
    for yolo_id in sorted(yolo_id_to_name.keys()):
        print(f"  {yolo_id}: {yolo_id_to_name[yolo_id]}")

    # 逐個 split 處理
    for split_dir_name, yolo_split_name in split_pairs:
        process_split(
            split_dir_name=split_dir_name,
            yolo_split_name=yolo_split_name,
            root_dir=args.root_dir,
            out_root=args.out_root,
            json_name=args.json_name,
            cat_id_to_yolo_id=cat_id_to_yolo_id,
        )

    # 寫 data.yaml
    yaml_path = args.yaml_path or os.path.join(args.out_root, "data.yaml")
    write_data_yaml(
        yaml_path=yaml_path,
        out_root=args.out_root,
        yolo_splits=sorted(yolo_splits_set),
        yolo_id_to_name=yolo_id_to_name,
    )
    print(f"已輸出 data.yaml: {yaml_path}")

    # 寫 classes.txt
    txt_path = args.txt_path or os.path.join(args.out_root, "classes.txt")
    write_classes_txt(txt_path, yolo_id_to_name)
    print(f"已輸出 classes_txt: {txt_path}")


if __name__ == "__main__":
    main()
