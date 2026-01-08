import argparse
import json
import shutil
import os
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Process COCO dataset: Filter, Merge by Name, and Re-index IDs to start from 0.")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of the source COCO dataset.")
    parser.add_argument("--label_file", type=str, required=True, help="Path to labels.txt.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the processed dataset.")
    return parser.parse_args()

def parse_label_file_and_reindex(label_path):
    """
    解析 labels.txt 並重新編號 (Re-index)。
    
    Returns:
        id_map (dict): {原始ID: 新ID} (例如 {0:0, 2:1, 3:1})
        new_categories_list (list): COCO 格式的新類別列表 (ID 從 0 開始)
    """
    print(f"[INFO] Parsing and Re-indexing label file: {label_path}")
    
    # 1. 先讀取所有有效的 (id, name)
    valid_entries = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            try:
                orig_id = int(parts[0])
                name = parts[1].strip()
                valid_entries.append((orig_id, name))
            except ValueError:
                continue
    
    # 2. 依照原始 ID 排序，確保處理順序一致
    valid_entries.sort(key=lambda x: x[0])
    
    # 3. 賦予新 ID (0, 1, 2...)
    name_to_new_id = {}    # { 'HEALTHY': 0, 'ABNORMAL': 1 }
    id_map = {}            # { 0:0, 2:1, 3:1 }
    new_categories_list = []
    
    next_new_id = 0
    
    for orig_id, name in valid_entries:
        if name not in name_to_new_id:
            # 發現新類別名稱，配發一個新的流水號 ID
            current_new_id = next_new_id
            name_to_new_id[name] = current_new_id
            
            new_categories_list.append({
                "id": current_new_id,
                "name": name,
                "supercategory": name
            })
            next_new_id += 1
        else:
            # 名稱已存在，沿用舊的 ID (合併)
            current_new_id = name_to_new_id[name]
            
        # 建立 原始ID -> 新ID 的映射
        id_map[orig_id] = current_new_id
        
        # 顯示對應關係
        arrow = "->" if orig_id != current_new_id else "=="
        print(f"  - Original ID {orig_id} ({name}) {arrow} New ID {current_new_id}")

    print(f"[INFO] Total Unique Categories: {len(new_categories_list)}")
    return id_map, new_categories_list

def process_split(split_name, source_dir, dest_dir, id_map, new_categories):
    src_json_path = source_dir / "_annotations.coco.json"
    if not src_json_path.exists():
        return
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    with open(src_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    stats = {
        "kept": 0,
        "removed": 0,
        "kept_ids": set()
    }
    
    # 1. 替換成全新的 Categories (ID 從 0 開始)
    coco_data['categories'] = new_categories
    
    # 2. 更新 Annotations
    new_annotations = []
    if 'annotations' in coco_data:
        for ann in coco_data['annotations']:
            orig_cat_id = ann.get('category_id')
            
            if orig_cat_id in id_map:
                # 取得新 ID (例如 2 變成 1)
                new_cat_id = id_map[orig_cat_id]
                ann['category_id'] = new_cat_id
                
                new_annotations.append(ann)
                stats['kept'] += 1
                stats['kept_ids'].add(new_cat_id)
            else:
                stats['removed'] += 1
                
        coco_data['annotations'] = new_annotations

    # 3. 複製圖片 (全保留)
    images_list = coco_data.get('images', [])
    for img in tqdm(images_list, desc=f"  Copying {split_name}", leave=False):
        fname = img.get('file_name')
        if fname:
            src = source_dir / fname
            dst = dest_dir / fname
            if src.exists():
                shutil.copy2(src, dst)

    # 4. 存檔
    out_json = dest_dir / "_annotations.coco.json"
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=2)

    print(f"  [{split_name.upper()}] Images: {len(images_list)}, Anns Kept: {stats['kept']}, Removed: {stats['removed']}")

def main():
    args = parse_args()
    root_dir = Path(args.data_root)
    output_dir = Path(args.output_dir)
    label_file = Path(args.label_file)

    if not root_dir.exists() or not label_file.exists():
        print("[Error] Path not found.")
        return

    # 1. 建立 Re-index 映射表
    id_map, new_categories = parse_label_file_and_reindex(label_file)
    if not new_categories:
        print("[Error] No valid categories found.")
        return

    # 2. 處理資料
    for split in ["train", "valid", "test"]:
        src = root_dir / split
        if src.is_dir():
            process_split(split, src, output_dir / split, id_map, new_categories)

    print(f"\n[Done] Output saved to: {output_dir}")

if __name__ == "__main__":
    main()