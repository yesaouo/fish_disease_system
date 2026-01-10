import os
import json
import cv2
import random
import numpy as np
import argparse
import shutil
import time
from pathlib import Path

def setup_args():
    parser = argparse.ArgumentParser(description="COCO 資料集 Mosaic 增強工具")
    parser.add_argument('--data_root', type=str, required=True, 
                        help="資料集根目錄 (包含圖片與 _annotations.coco.json)")
    parser.add_argument('--json_name', type=str, default='_annotations.coco.json',
                        help="標註檔案名稱 (預設: _annotations.coco.json)")
    parser.add_argument('--img_size', type=int, default=640, 
                        help="輸出圖片大小 (預設: 640)")
    parser.add_argument('--ratio', type=float, default=0.5, 
                        help="增強比例: 產生相當於原資料集多少比例的新圖片 (預設: 0.5，即增加 50%% 的量)")
    return parser.parse_args()

def load_coco(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_coco(json_path, data):
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def get_img_data(coco_data, img_dir, img_id):
    """讀取單張圖片與其 Bboxes"""
    # 找到圖片資訊
    img_info = next((item for item in coco_data['images'] if item["id"] == img_id), None)
    if not img_info:
        return None, None, None

    img_path = os.path.join(img_dir, img_info['file_name'])
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"警告: 讀取不到圖片 {img_path}，跳過。")
        return None, None, None

    # 找到對應的 annotations
    anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
    
    boxes = []
    labels = []
    
    for ann in anns:
        x, y, w, h = ann['bbox']
        boxes.append([x, y, x + w, y + h]) # xywh -> xyxy
        labels.append(ann['category_id'])
            
    return img, np.array(boxes), np.array(labels)

def generate_mosaic(coco_data, img_dir, img_ids, output_size):
    """生成單張 Mosaic 圖片"""
    selected_ids = random.sample(img_ids, 4)
    w_out, h_out = output_size, output_size
    
    # 建立畫布 (灰色背景 114)
    mosaic_img = np.full((h_out * 2, w_out * 2, 3), 114, dtype=np.uint8)
    
    xc = int(random.uniform(w_out * 0.5, w_out * 1.5))
    yc = int(random.uniform(h_out * 0.5, h_out * 1.5))
    
    mosaic_boxes = []
    mosaic_labels = []

    for i, img_id in enumerate(selected_ids):
        img, boxes, labels = get_img_data(coco_data, img_dir, img_id)
        if img is None: continue
        
        h, w, _ = img.shape

        # 定義貼圖位置
        if i == 0:  # top-left
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:  # top-right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, w_out * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom-left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(yc + h, h_out * 2)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom-right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, w_out * 2), min(yc + h, h_out * 2)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        pad_w = x1a - x1b
        pad_h = y1a - y1b

        if len(boxes) > 0:
            boxes_copy = boxes.copy()
            boxes_copy[:, [0, 2]] += pad_w
            boxes_copy[:, [1, 3]] += pad_h
            mosaic_boxes.append(boxes_copy)
            mosaic_labels.append(labels)

    if len(mosaic_boxes) > 0:
        mosaic_boxes = np.concatenate(mosaic_boxes, 0)
        mosaic_labels = np.concatenate(mosaic_labels, 0)
        
        # Clip
        np.clip(mosaic_boxes[:, 0], 0, 2 * w_out, out=mosaic_boxes[:, 0])
        np.clip(mosaic_boxes[:, 1], 0, 2 * h_out, out=mosaic_boxes[:, 1])
        np.clip(mosaic_boxes[:, 2], 0, 2 * w_out, out=mosaic_boxes[:, 2])
        np.clip(mosaic_boxes[:, 3], 0, 2 * h_out, out=mosaic_boxes[:, 3])

        # Filter small boxes
        w_box = mosaic_boxes[:, 2] - mosaic_boxes[:, 0]
        h_box = mosaic_boxes[:, 3] - mosaic_boxes[:, 1]
        valid_indices = (w_box > 5) & (h_box > 5) # 稍微寬鬆一點，大於 5 pixel
        
        mosaic_boxes = mosaic_boxes[valid_indices]
        mosaic_labels = mosaic_labels[valid_indices]

    # Resize back to output size
    final_img = cv2.resize(mosaic_img, (w_out, h_out))
    
    scale = 0.5 # 因為是從 2x 縮小回 1x
    if len(mosaic_boxes) > 0:
        mosaic_boxes = mosaic_boxes * scale
    
    return final_img, mosaic_boxes, mosaic_labels

def main():
    args = setup_args()
    
    json_path = os.path.join(args.data_root, args.json_name)
    if not os.path.exists(json_path):
        print(f"錯誤: 找不到 JSON 檔案: {json_path}")
        return

    # 1. 備份原始 JSON
    backup_path = json_path + ".bak"
    shutil.copy(json_path, backup_path)
    print(f"已備份原始 JSON 至: {backup_path}")

    # 2. 載入資料
    coco = load_coco(json_path)
    img_ids = [img['id'] for img in coco['images']]
    
    # 初始化 ID 計數器 (從現有最大值開始 + 1)
    max_img_id = max(img_ids) if img_ids else 0
    max_ann_id = max([ann['id'] for ann in coco['annotations']]) if coco['annotations'] else 0
    
    current_img_id = max_img_id + 1
    current_ann_id = max_ann_id + 1

    # 計算要生成的數量
    num_generated = int(len(img_ids) * args.ratio)
    print(f"原始圖片數: {len(img_ids)}, 預計生成 Mosaic 圖片數: {num_generated}")

    new_images = []
    new_annotations = []

    print("開始生成 Mosaic 圖片...")
    for i in range(num_generated):
        # 生成圖片與標註
        try:
            img, boxes, labels = generate_mosaic(coco, args.data_root, img_ids, args.img_size)
        except Exception as e:
            print(f"生成第 {i} 張時發生錯誤: {e}")
            continue

        if img is None: continue

        # 定義新檔名
        file_name = f"mosaic_{int(time.time())}_{i}.jpg"
        save_path = os.path.join(args.data_root, file_name)
        
        # 存檔
        cv2.imwrite(save_path, img)

        # 建立 COCO Image Info
        img_info = {
            "id": current_img_id,
            "file_name": file_name,
            "height": args.img_size,
            "width": args.img_size,
            "date_captured": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        new_images.append(img_info)

        # 建立 COCO Annotations
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            
            ann_info = {
                "id": current_ann_id,
                "image_id": current_img_id,
                "category_id": int(label),
                "bbox": [float(x1), float(y1), float(w), float(h)], # xywh
                "area": float(w * h),
                "iscrowd": 0,
                "segmentation": [] # Mosaic 無法簡單處理 segmentation，留空
            }
            new_annotations.append(ann_info)
            current_ann_id += 1
        
        current_img_id += 1

        if (i + 1) % 10 == 0:
            print(f"進度: {i + 1}/{num_generated}")

    # 3. 更新並儲存 JSON
    coco['images'].extend(new_images)
    coco['annotations'].extend(new_annotations)
    
    save_coco(json_path, coco)
    print(f"\n完成！")
    print(f"新增圖片: {len(new_images)} 張")
    print(f"新增標註: {len(new_annotations)} 個")
    print(f"已更新 JSON 檔案: {json_path}")

if __name__ == "__main__":
    main()