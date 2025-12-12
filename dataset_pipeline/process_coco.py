import os
import json
import random
import shutil
import argparse
from collections import defaultdict
import yaml


def load_coco(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_coco(coco: dict, json_path: str):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)


def process_categories_with_yaml(coco: dict, yaml_path: str):
    """
    同時做：
      1. 移除 id=0 的 category
      2. 用 YAML 更新 supercategory
      3. YAML 多出的 id → 直接報錯
      4. YAML 缺少的 id → 警告 + 移除該類別及其 annotations
      5. 把沒有任何 annotation 的圖片移除
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        mapping = yaml.safe_load(f)

    # YAML key 轉成 int
    yaml_map = {}
    for k, v in mapping.items():
        yaml_map[int(k)] = v

    categories = coco.get("categories", [])
    annotations = coco.get("annotations", [])
    images = coco.get("images", [])

    cat_ids = {c["id"] for c in categories}

    # 多出的 id 仍然視為錯誤
    extra_in_yaml = set(yaml_map.keys()) - cat_ids
    if extra_in_yaml:
        raise ValueError(f"YAML 多出 id: {sorted(extra_in_yaml)}")

    removed_cat_ids = set()
    new_categories = []

    for c in categories:
        cid = c["id"]

        # 先處理 id=0：直接移除
        if cid == 0:
            removed_cat_ids.add(cid)
            print(f"[INFO] 移除 category id=0：{c.get('name')}")
            continue

        # YAML 沒有這個 id：警告 + 移除這個類別及其 annotations
        if cid not in yaml_map:
            removed_cat_ids.add(cid)
            print(f"[WARN] YAML 缺少 id={cid} (name={c.get('name')})，"
                  f"會移除該類別及其所有 annotations。")
            continue

        # 正常情況：套用 supercategory
        c["supercategory"] = yaml_map[cid]
        new_categories.append(c)

    coco["categories"] = new_categories

    # 根據 removed_cat_ids 移除 annotations
    if removed_cat_ids:
        before_ann = len(annotations)
        annotations = [
            a for a in annotations
            if a.get("category_id") not in removed_cat_ids
        ]
        after_ann = len(annotations)
        print(f"[INFO] 因類別移除導致 annotations 數量：{before_ann} -> {after_ann}")
    coco["annotations"] = annotations

    # 重新統計每張圖片剩下多少 annotations，沒有的就移除
    ann_count_by_image = defaultdict(int)
    for ann in annotations:
        ann_count_by_image[ann["image_id"]] += 1

    before_img = len(images)
    images = [
        img for img in images
        if ann_count_by_image.get(img["id"], 0) > 0
    ]
    after_img = len(images)

    if before_img != after_img:
        print(f"[INFO] 因沒有任何 annotation，被移除的圖片數量：{before_img - after_img}")

    coco["images"] = images


# 3. segmentation → bbox，統計並移除 segmentation 欄位
def segmentation_to_bbox(segmentation):
    """
    segmentation 可能是：
      - list[float]  : [x1, y1, x2, y2, ...]
      - list[list]   : [[x1, y1, x2, y2, ...], [x1, y1, ...], ...]
    不處理 RLE dict，如果遇到就丟出例外。
    """
    if not segmentation:
        raise ValueError("Empty segmentation")

    xs = []
    ys = []

    if isinstance(segmentation, list):
        if segmentation and isinstance(segmentation[0], list):
            # 多個 polygon
            for poly in segmentation:
                xs.extend(poly[0::2])
                ys.extend(poly[1::2])
        else:
            # 單一扁平 list
            xs.extend(segmentation[0::2])
            ys.extend(segmentation[1::2])
    else:
        # 例如 RLE dict，不在這邊處理
        raise ValueError(f"Unsupported segmentation type: {type(segmentation)}")

    if not xs or not ys:
        raise ValueError("Invalid segmentation coordinates")

    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


def process_segmentation(coco: dict):
    anns = coco.get("annotations", [])
    seg_count = 0
    conv_count = 0

    for ann in anns:
        if "segmentation" in ann and ann["segmentation"]:
            seg_count += 1
            try:
                bbox = segmentation_to_bbox(ann["segmentation"])
                ann["bbox"] = bbox
                conv_count += 1
            except Exception as e:
                print(f"[WARN] annotation id={ann.get('id')} segmentation 轉 bbox 失敗: {e}")
        # 無論有沒有成功轉換，一律移除 segmentation 欄位
        if "segmentation" in ann:
            del ann["segmentation"]

    print(f"[INFO] 有 {seg_count} 個 annotation 含 segmentation，其中 {conv_count} 個成功轉成 bbox。")


# 4. 依 supercategory 是否為 healthy_region 合併為 2 類，
#    並在 annotations 增加 orig_name。
def merge_to_two_classes(coco: dict):
    categories = coco.get("categories", [])

    # 先記住原本每個 id 的 name
    id_to_name = {c["id"]: c["name"] for c in categories}
    id_to_super = {c["id"]: c.get("supercategory", "") for c in categories}

    healthy_ids = {cid for cid, sc in id_to_super.items() if sc == "healthy_region"}
    abnormal_ids = set(id_to_name.keys()) - healthy_ids

    if not healthy_ids:
        print("[WARN] 沒有任何 supercategory='healthy_region' 的類別，所有類別都會被視為 ABNORMAL。")
    if not abnormal_ids:
        print("[WARN] 所有類別都是 healthy_region？那就全部當 HEALTHY 類。")

    # 更新 annotations：加 orig_name，並把 category_id 合併成 0 或 1
    for ann in coco.get("annotations", []):
        old_id = ann["category_id"]
        orig_name = id_to_name.get(old_id, f"UNKNOWN_{old_id}")
        ann["orig_name"] = orig_name

        if old_id in healthy_ids:
            ann["category_id"] = 0
        else:
            ann["category_id"] = 1

    # 最後 categories 只保留兩個
    coco["categories"] = [
        {"id": 0, "name": "HEALTHY", "supercategory": "HEALTHY"},
        {"id": 1, "name": "ABNORMAL", "supercategory": "ABNORMAL"},
    ]


def bbox_contains(inner, outer):
    """
    檢查 inner bbox 是否「完全在」 outer bbox 之內
    bbox 格式為 [x, y, w, h]
    """
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer

    ix1, iy1, ix2, iy2 = ix, iy, ix + iw, iy + ih
    ox1, oy1, ox2, oy2 = ox, oy, ox + ow, oy + oh

    return (ix1 >= ox1 and iy1 >= oy1 and
            ix2 <= ox2 and iy2 <= oy2)


def remove_healthy_if_abnormal_inside(coco: dict):
    """
    對每張圖：
      - 找出所有 category_id = 0 (HEALTHY) 的框
      - 找出所有 category_id = 1 (ABNORMAL) 的框
      - 如果某個 HEALTHY 框裡「包含」任一 ABNORMAL 框，就把這個 HEALTHY 框移除

    並印出統計資訊。
    """
    anns = coco.get("annotations", [])
    if not anns:
        print("[INFO] 沒有任何 annotation，不需移除 0 的框。")
        return

    anns_by_image = defaultdict(list)
    for ann in anns:
        anns_by_image[ann["image_id"]].append(ann)

    total_healthy = sum(1 for ann in anns if ann.get("category_id") == 0)
    removed_healthy_ids = set()

    for image_id, image_anns in anns_by_image.items():
        healthy_list = [a for a in image_anns if a.get("category_id") == 0]
        abnormal_list = [a for a in image_anns if a.get("category_id") == 1]

        if not healthy_list or not abnormal_list:
            continue  # 這張圖沒有同時有 0 和 1，就跳過

        removed_here = 0

        for h in healthy_list:
            hbbox = h.get("bbox")
            if not hbbox:
                continue  # 沒 bbox 就沒辦法判斷

            # 如果任一 ABNORMAL bbox 完全在這個 HEALTHY bbox 裡，就標記要移除
            has_abnormal_inside = False
            for ab in abnormal_list:
                abbox = ab.get("bbox")
                if not abbox:
                    continue
                if bbox_contains(abbox, hbbox):
                    has_abnormal_inside = True
                    break

            if has_abnormal_inside:
                removed_healthy_ids.add(h["id"])
                removed_here += 1

        if removed_here > 0:
            print(f"[INFO] image_id={image_id} 移除 {removed_here} 個 HEALTHY(0) 框（內含 ABNORMAL(1) 框）。")

    if not removed_healthy_ids:
        print("[INFO] 沒有任何 HEALTHY(0) 框被移除（沒有 1 在 0 內的情況）。")
        return

    before_ann = len(anns)
    coco["annotations"] = [
        a for a in anns if a.get("id") not in removed_healthy_ids
    ]
    after_ann = len(coco["annotations"])

    removed_count = len(removed_healthy_ids)
    remain_healthy = total_healthy - removed_count

    print(f"[INFO] HEALTHY(0) 總數：{total_healthy}，被移除：{removed_count}，剩下：{remain_healthy}")
    print(f"[INFO] annotations 總數：{before_ann} -> {after_ann}")


# 5. 依比例切 train/valid/test，並複製圖片 + 產生新的 json
def split_dataset(coco: dict, input_dir: str, output_dir: str,
                  train_ratio: int, valid_ratio: int, test_ratio: int):
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])
    info = coco.get("info", {})
    licenses = coco.get("licenses", [])

    image_ids = [img["id"] for img in images]
    random.shuffle(image_ids)

    total_ratio = train_ratio + valid_ratio + test_ratio
    n = len(image_ids)

    n_train = int(n * train_ratio / total_ratio) if total_ratio > 0 else 0
    n_valid = int(n * valid_ratio / total_ratio) if total_ratio > 0 else 0
    n_test = n - n_train - n_valid

    print(f"[INFO] 總共 {n} 張圖片，"
          f"train={n_train}, valid={n_valid}, test={n_test}")

    split_ids = {
        "train": set(image_ids[:n_train]),
        "valid": set(image_ids[n_train:n_train + n_valid]),
        "test": set(image_ids[n_train + n_valid:]) if test_ratio > 0 else set(),
    }

    # 建立 id → image 資訊 map
    id_to_image = {img["id"]: img for img in images}

    # 為每個 subset 建立 COCO dict 並輸出
    for subset in ["train", "valid", "test"]:
        if subset == "test" and test_ratio == 0:
            # 比例為 0 就跳過
            continue

        subset_ids = split_ids[subset]
        subset_images = [id_to_image[iid] for iid in subset_ids]
        subset_annos = [ann for ann in annotations if ann["image_id"] in subset_ids]

        subset_coco = {
            "info": info,
            "licenses": licenses,
            "images": subset_images,
            "annotations": subset_annos,
            "categories": categories,
        }

        subset_dir = os.path.join(output_dir, subset)
        os.makedirs(subset_dir, exist_ok=True)

        # 複製圖片（這裡的 input_dir 預期是 images/ 資料夾）
        for img in subset_images:
            src = os.path.join(input_dir, img["file_name"])
            dst = os.path.join(subset_dir, img["file_name"])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.isfile(src):
                print(f"[WARN] 找不到圖片檔案: {src}")
                continue
            shutil.copy2(src, dst)

        # 輸出 annotations
        json_path = os.path.join(subset_dir, "_annotations.coco.json")
        save_coco(subset_coco, json_path)
        print(f"[INFO] 輸出 {subset} 到 {json_path} ，images={len(subset_images)}, annots={len(subset_annos)}")


def main():
    parser = argparse.ArgumentParser(description="COCO JSON 處理與分割")
    parser.add_argument(
        "--input_dir", required=True,
        help="根目錄，底下包含 classes.yaml 以及 images 資料夾（images 底下有 _annotations.coco.json）"
    )
    parser.add_argument("--output_dir", required=True, help="輸出 train/valid/test 的根目錄")
    parser.add_argument("--split", nargs=3, type=int, metavar=("TRAIN", "VALID", "TEST"),
                        default=[8, 1, 1],
                        help="例如: 8 1 1 或 8 2 0")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子")
    args = parser.parse_args()

    random.seed(args.seed)

    # 固定使用 input_dir/images 與 input_dir/classes.yaml
    images_dir = os.path.join(args.input_dir, "images")
    input_json = os.path.join(images_dir, "_annotations.coco.json")
    yaml_path = os.path.join(args.input_dir, "classes.yaml")

    if not os.path.isfile(input_json):
        raise FileNotFoundError(f"找不到 {input_json}")
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"找不到 {yaml_path}（請放在 input_dir 底下）")

    coco = load_coco(input_json)

    # 步驟 1 + 2：用 classes.yaml 處理 categories
    process_categories_with_yaml(coco, yaml_path)

    # 步驟 3
    process_segmentation(coco)

    # 步驟 4（合併類別，並在 annotation 存 orig_name）
    merge_to_two_classes(coco)

    # 新增一步：如果 0 的框內有 1 的框，就移除 0 的框，並印出統計
    remove_healthy_if_abnormal_inside(coco)

    # 步驟 5：依比例分割並輸出各自的 _annotations.coco.json
    train_ratio, valid_ratio, test_ratio = args.split

    # 這裡特別注意：split_dataset 的 input_dir 現在傳入 images_dir
    split_dataset(coco, images_dir, args.output_dir,
                  train_ratio, valid_ratio, test_ratio)


if __name__ == "__main__":
    main()
