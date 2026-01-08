import json
import shutil
import argparse
import os
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Convert COCO format to YOLO format.")
    parser.add_argument("--data_root", type=str, required=True, help="Root dir of COCO dataset (should contain train/valid/test subdirs).")
    parser.add_argument("--output_dir", type=str, required=True, help="Output dir for YOLO dataset.")
    return parser.parse_args()

def coco_to_yolo_bbox(bbox, img_w, img_h):
    """
    COCO: [x_min, y_min, width, height]
    YOLO: [x_center, y_center, width, height] (Normalized 0-1)
    """
    x_min, y_min, w, h = bbox
    
    # 計算中心點
    x_center = x_min + w / 2.0
    y_center = y_min + h / 2.0
    
    # 歸一化
    x_center /= img_w
    y_center /= img_h
    w /= img_w
    h /= img_h
    
    # 限制在 0-1 之間 (防止標註超出圖片)
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    
    return x_center, y_center, w, h

def generate_yaml(output_dir, categories):
    """
    產生 data.yaml
    """
    # 確保 categories 依照 ID 排序
    sorted_cats = sorted(categories, key=lambda x: x['id'])
    names = [cat['name'] for cat in sorted_cats]
    
    yaml_content = [
        f"path: {output_dir.absolute()}",  # 使用絕對路徑比較保險
        "train: images/train",
        "val: images/valid",
        "test: images/test",  # Optional
        "",
        f"nc: {len(names)}",
        f"names: {json.dumps(names, ensure_ascii=False)}"
    ]
    
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(yaml_content))
    print(f"[INFO] Generated data.yaml at {yaml_path}")
    
    # 另外產生 classes.txt (方便查閱)
    cls_path = output_dir / "classes.txt"
    with open(cls_path, 'w', encoding='utf-8') as f:
        for name in names:
            f.write(f"{name}\n")

def process_split(split, coco_root, yolo_root):
    """
    處理單個 Split (train/valid/test)
    """
    src_dir = coco_root / split
    if not src_dir.is_dir():
        return None  # Skip if split doesn't exist

    json_file = src_dir / "_annotations.coco.json"
    if not json_file.exists():
        print(f"[WARNING] No JSON found in {src_dir}, skipping.")
        return None

    # 準備輸出資料夾
    images_out = yolo_root / "images" / split
    labels_out = yolo_root / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {json_file}...")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 建立 圖片ID -> 圖片資訊 的映射
    img_map = {img['id']: img for img in data['images']}
    
    # 建立 圖片ID -> 標註列表 的映射
    ann_map = {img['id']: [] for img in data['images']}
    if 'annotations' in data:
        for ann in data['annotations']:
            ann_map[ann['image_id']].append(ann)

    # 開始轉換
    print(f"Converting {split} set ({len(data['images'])} images)...")
    for img_id, img_info in tqdm(img_map.items(), desc=f"  Processing {split}", leave=False):
        file_name = img_info['file_name']
        img_w = img_info['width']
        img_h = img_info['height']
        
        # 1. 複製圖片
        src_img_path = src_dir / file_name
        dst_img_path = images_out / file_name
        
        if src_img_path.exists():
            shutil.copy2(src_img_path, dst_img_path)
        else:
            # 如果來源圖片不見了，跳過產生 label，避免訓練時報錯
            continue

        # 2. 產生 YOLO Label txt
        # 檔名：圖片檔名.jpg -> 圖片檔名.txt
        txt_name = Path(file_name).stem + ".txt"
        dst_txt_path = labels_out / txt_name
        
        yolo_lines = []
        anns = ann_map.get(img_id, [])
        
        for ann in anns:
            cat_id = ann['category_id']
            bbox = ann['bbox']
            
            # 轉換座標
            xc, yc, w, h = coco_to_yolo_bbox(bbox, img_w, img_h)
            
            # YOLO 格式: class_id x_center y_center width height
            yolo_lines.append(f"{cat_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            
        # 寫入 txt (即使沒有標註也要產生空檔案)
        with open(dst_txt_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(yolo_lines))

    return data.get('categories', [])

def main():
    args = parse_args()
    coco_root = Path(args.data_root)
    yolo_root = Path(args.output_dir)
    
    if not coco_root.exists():
        print(f"[Error] COCO root not found: {coco_root}")
        return

    # 清空或是建立輸出目錄
    if yolo_root.exists():
        print(f"[INFO] Cleaning output directory: {yolo_root}")
        shutil.rmtree(yolo_root)
    yolo_root.mkdir(parents=True, exist_ok=True)

    # 處理順序
    splits = ['train', 'valid', 'test']
    categories = None
    
    for split in splits:
        cats = process_split(split, coco_root, yolo_root)
        # 抓取第一份有效的 categories 來產生 data.yaml (假設所有 split 的 categories 都一樣)
        if cats and categories is None:
            categories = cats

    if categories:
        generate_yaml(yolo_root, categories)
        print("\n[Done] Conversion complete.")
        print(f"Dataset saved to: {yolo_root}")
    else:
        print("\n[Error] No valid categories found. Conversion might have failed.")

if __name__ == "__main__":
    main()