import os
import json
import csv
import argparse
import itertools
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ===== 核心邏輯：Crop 處理 =====
def get_square_context_crop(img_pil, bbox, square_scale=1.5):
    """
    以 bbox 中心為基準，取 max(w, h) * scale 作為邊長，裁切成正方形。
    若超出邊界則進行 padding (保持圖片置中)。
    """
    x, y, w, h = bbox
    cx, cy = x + w/2, y + h/2
    
    side_length = max(w, h) * square_scale
    half_side = side_length / 2
    
    # 計算原始圖片上的裁切座標
    x1 = cx - half_side
    y1 = cy - half_side
    x2 = cx + half_side
    y2 = cy + half_side
    
    # 原圖尺寸
    orig_w, orig_h = img_pil.size
    
    # 處理 Padding 邏輯
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - orig_w)
    pad_bottom = max(0, y2 - orig_h)
    
    # 實際在原圖上的裁切範圍
    crop_x1 = max(0, x1)
    crop_y1 = max(0, y1)
    crop_x2 = min(orig_w, x2)
    crop_y2 = min(orig_h, y2)
    
    # 從原圖裁切有效區域
    valid_crop = img_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    
    # 如果需要 padding，建立新圖 (黑色背景)
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        new_size = (int(side_length), int(side_length))
        new_img = Image.new("RGB", new_size, (0, 0, 0)) 
        new_img.paste(valid_crop, (int(pad_left), int(pad_top)))
        return new_img
    else:
        return valid_crop

# ===== 核心邏輯：載入 Caption 並建立輪詢器 =====
def create_caption_iterators(json_path):
    """
    讀取 symptoms.json，並為每個 label ID 建立一個無限循環的 iterator (Round-Robin)。
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    data_content = data['data']
    iterators = {}
    
    for label_id, content in data_content.items():
        captions = content.get('captions_en', [])

        # itertools.cycle 會建立一個無限循環的產生器 ['A', 'B'] -> A, B, A, B...
        iterators[str(label_id)] = itertools.cycle(captions)
        
    return iterators

# ===== 處理單一 Split (Train/Valid/Test) =====
def process_single_split(split_name, input_split_dir, output_split_dir, symptoms_json_path, square_scale=1.5):
    """
    處理單一個 split 資料夾 (例如 train)，讀取其中的 _annotations.coco.json 並輸出結果。
    """
    input_split_dir = Path(input_split_dir)
    output_split_dir = Path(output_split_dir)
    
    print(f"\n[{split_name.upper()}] Processing...")
    print(f"  Input: {input_split_dir}")
    print(f"  Output: {output_split_dir}")

    # 1. 檢查輸入資料夾是否存在
    if not input_split_dir.exists():
        print(f"  Warning: Input folder {input_split_dir} does not exist. Skipping.")
        return

    # 2. 建立輸出資料夾
    os.makedirs(output_split_dir, exist_ok=True)
    
    # 3. 準備 Caption Iterators
    caption_iters = create_caption_iterators(symptoms_json_path)
    
    # 4. 載入 COCO JSON
    coco_json_path = input_split_dir / "_annotations.coco.json"
    if not coco_json_path.exists():
        print(f"  Error: {coco_json_path} not found inside {split_name} folder. Skipping.")
        return

    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
        
    # 建立 Annotations Grouping (一張圖可能有多個標註)
    img_id_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)

    csv_rows = []
    total_crops = 0
    
    # 5. 逐張圖片處理
    images_list = coco_data['images']
    print(f"  Found {len(images_list)} images in {split_name}.")
    
    for img_info in tqdm(images_list, desc=f"  Processing {split_name}"):
        img_id = img_info['id']
        file_name = img_info['file_name']
        img_path = input_split_dir / file_name
        
        if not img_path.exists():
            continue
            
        anns = img_id_to_anns.get(img_id, [])
        if not anns:
            continue
            
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  Error reading {file_name}: {e}")
            continue
            
        for ann in anns:
            category_id = str(ann['category_id']) 
            ann_id = ann['id']
            bbox = ann['bbox'] # [x, y, w, h]
            
            # (A) 取得 Caption (輪流取出)
            if category_id in caption_iters:
                caption_text = next(caption_iters[category_id])
            else:
                caption_text = "a photo of a medical symptom" 
            
            # (B) 執行 Square Crop
            crop_img = get_square_context_crop(pil_img, bbox, square_scale=square_scale)
            
            # (C) 存檔 (使用 annotation id 作為檔名)
            save_filename = f"{ann_id}.jpg"
            save_path = output_split_dir / save_filename
            crop_img.save(save_path, quality=95)
            
            # (D) 紀錄 CSV
            csv_rows.append([save_filename, caption_text])
            total_crops += 1

    # 6. 寫入 CSV
    csv_path = output_split_dir / "metadata.csv"
    print(f"  Writing metadata to {csv_path}...")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "caption"]) # Header
        writer.writerows(csv_rows)
        
    print(f"  Done! Generated {total_crops} training pairs for [{split_name}]")

# ===== 遍歷 Train/Valid/Test =====
def main_process(data_root, symptoms_json_path, output_root, square_scale=1.5):
    data_root = Path(data_root)
    output_root = Path(output_root)
    
    # 定義要處理的三個資料夾
    splits = ["train", "valid", "test"]
    
    print(f"Starting Dataset Generation...")
    print(f"Root Input: {data_root}")
    print(f"Root Output: {output_root}")
    
    for split in splits:
        input_split_dir = data_root / split
        output_split_dir = output_root / split
        
        process_single_split(
            split_name=split,
            input_split_dir=input_split_dir,
            output_split_dir=output_split_dir,
            symptoms_json_path=symptoms_json_path,
            square_scale=square_scale
        )
    
    print("\nAll splits processed finished.")

# ===== 參數解析 =====
def parse_args():
    parser = argparse.ArgumentParser(description="Generate Square-Cropped Training Data for Train/Valid/Test splits.")
    
    parser.add_argument(
        "--data_root", 
        type=str, 
        required=True, 
        help="Root path containing train/valid/test subdirectories."
    )
    
    parser.add_argument(
        "--symptoms_json", 
        type=str, 
        required=True, 
        help="Path to the symptoms.json file defining captions"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Root path to save the output train/valid/test subdirectories"
    )
    
    parser.add_argument(
        "--scale", 
        type=float, 
        default=1.5, 
        help="Square crop scaling factor (default: 1.5)"
    )

    return parser.parse_args()

def main():
    args = parse_args()
    
    main_process(
        data_root=args.data_root,
        symptoms_json_path=args.symptoms_json,
        output_root=args.output_dir,
        square_scale=args.scale
    )

if __name__ == "__main__":
    main()