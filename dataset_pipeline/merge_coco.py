from __future__ import annotations
import argparse
import json
import shutil
import datetime
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="將 root_dir 底下多個 COCO 格式子資料集合併成一個，輸出到 _merged/"
    )
    p.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="上層資料夾路徑，底下每個子資料夾是一個獨立 DATASET（含 train/、valid/ 等）",
    )
    p.add_argument(
        "--merged_name",
        type=str,
        default="_merged",
        help="合併後輸出資料夾名稱（預設為 _merged）",
    )
    return p.parse_args()


def is_single_dataset_dir(path: Path) -> bool:
    """判斷這個資料夾本身是否就是一個 DATASET（有 train/ 或 valid/）"""
    return (path / "train").is_dir() or (path / "valid").is_dir()


def find_child_datasets(root: Path) -> List[Path]:
    """
    在 root 底下找所有「子資料集目錄」
    條件：子目錄底下至少有 train/ 或 valid/
    """
    datasets: List[Path] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if is_single_dataset_dir(p):
            datasets.append(p)
    return datasets


def merge_coco_splits(
    dataset_dirs: List[Path],
    split: str,
    merged_split_dir: Path,
) -> bool:
    """
    合併多個子資料集某個 split（train / valid / test）。
    回傳：是否有成功合併（沒有任何子資料集有這個 split 則回傳 False）
    """
    merged_split_dir.mkdir(parents=True, exist_ok=True)

    merged_images: List[dict] = []
    merged_annotations: List[dict] = []
    merged_categories: List[dict] = []
    categories_inited = False

    next_image_id = 1
    next_ann_id = 1

    any_found = False

    for ds_dir in dataset_dirs:
        split_dir = ds_dir / split
        if not split_dir.is_dir():
            # 這個子資料集沒有這個 split，就跳過
            continue

        ann_path = split_dir / "_annotations.coco.json"
        if not ann_path.is_file():
            print(f"[警告] {ann_path} 不存在，略過此 split。")
            continue

        any_found = True

        with ann_path.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        # 處理 categories（假設所有子資料集的 categories 相同）
        if not categories_inited:
            merged_categories = coco.get("categories", [])
            categories_inited = True
        else:
            cats = coco.get("categories", [])
            if len(cats) != len(merged_categories):
                raise ValueError(f"{ann_path} 的 categories 與其他資料集不一致")
            for c1, c2 in zip(cats, merged_categories):
                if c1["id"] != c2["id"] or c1["name"] != c2["name"]:
                    raise ValueError(f"{ann_path} 的 categories 與其他資料集不一致")

        # 重新指定 image / annotation id，並 copy 圖片
        id_map: Dict[int, int] = {}
        ds_name = ds_dir.name  # 用子資料夾名稱當前綴，避免檔名衝突

        for img in tqdm(coco.get("images", []), desc=f"  Merging {ds_name}", leave=False):
            old_id = img["id"]
            old_file_name = img["file_name"]

            src_img = split_dir / old_file_name
            if not src_img.is_file():
                print(f"[警告] 找不到圖片檔：{src_img}，略過此圖與其標註")
                continue

            new_id = next_image_id
            next_image_id += 1
            id_map[old_id] = new_id

            # 新的檔名：<子資料夾>_<原本檔名>
            new_file_name = f"{ds_name}_{old_file_name}"

            # 複製圖片到 merged_split_dir
            dst_img = merged_split_dir / new_file_name
            shutil.copy2(src_img, dst_img)

            new_img = dict(img)
            new_img["id"] = new_id
            new_img["file_name"] = new_file_name
            merged_images.append(new_img)

        for ann in coco.get("annotations", []):
            old_img_id = ann["image_id"]
            if old_img_id not in id_map:
                # 對應不到 image，可能因為上面圖片缺檔被略過
                continue

            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            next_ann_id += 1
            new_ann["image_id"] = id_map[old_img_id]
            merged_annotations.append(new_ann)

    if not any_found:
        # 沒有任何子資料集有這個 split
        return False

    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    merged_coco = {
        "info": {
            "year": int(current_date.split("-")[0]),
            "version": "1.0",
            "description": "Merged Dataset",
            "contributor": "Auto-Merger",
            "url": "",
            "date_created": current_date
        },
        "licenses": [{
            "id": 1,
            "name": "CC BY 4.0",
            "url": "https://creativecommons.org/licenses/by/4.0/"
        }],
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": merged_categories,
    }

    out_ann_path = merged_split_dir / "_annotations.coco.json"
    with out_ann_path.open("w", encoding="utf-8") as f:
        json.dump(merged_coco, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 已產生合併後標註：{out_ann_path}")
    print(f"  [統計] split: {split}")
    print(f"    - 來源資料集數量: {len(dataset_dirs)}")
    print(f"    - 合併後圖片總數: {len(merged_images)}")
    print(f"    - 合併後標註總數: {len(merged_annotations)}")
    print(f"    - 類別數量: {len(merged_categories)}")

    return True


def build_merged_dataset(root_dir: Path, merged_name: str = "_merged") -> Path:
    """
    假設 root_dir 底下是多個子資料集：
    root_dir/
      ├─ ds1/
      │   ├─ train/
      │   ├─ valid/
      │   └─ test/
      ├─ ds2/
      │   ├─ train/
      │   ├─ valid/
      │   └─ test/
      └─ ...

    會產生：
    root_dir/_merged/
      ├─ train/
      │   ├─ _annotations.coco.json
      │   └─ *.jpg
      ├─ valid/
      └─ test/（如有）
    """
    root_dir = root_dir.resolve()
    child_ds = find_child_datasets(root_dir)

    if not child_ds:
        raise ValueError(f"{root_dir} 底下沒有找到任何子資料集（含 train/ 或 valid/ 的子目錄）")

    merged_dir = root_dir / merged_name
    if merged_dir.exists():
        print(f"[INFO] 合併資料夾已存在：{merged_dir}，將覆蓋原內容")
        shutil.rmtree(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] 將合併以下子資料集：")
    for ds in child_ds:
        print("  -", ds)

    # 針對 train / valid / test 各自嘗試合併
    has_train = merge_coco_splits(child_ds, "train", merged_dir / "train")
    has_valid = merge_coco_splits(child_ds, "valid", merged_dir / "valid")
    has_test = merge_coco_splits(child_ds, "test", merged_dir / "test")

    if not has_train or not has_valid:
        raise ValueError("合併後至少需要有 train 與 valid split，請確認各子資料集結構。")

    if not has_test:
        # 沒有任何 test split，用不到就刪掉空資料夾
        test_dir = merged_dir / "test"
        if test_dir.exists():
            shutil.rmtree(test_dir)

    print(f"[INFO] 合併完成，輸出至：{merged_dir}")
    return merged_dir


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir)

    if not root_dir.is_dir():
        raise NotADirectoryError(f"{root_dir} 不是一個資料夾")

    merged_dir = build_merged_dataset(root_dir, merged_name=args.merged_name)


if __name__ == "__main__":
    main()
