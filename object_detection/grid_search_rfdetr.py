from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd
import torch
import itertools
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from torchvision.ops import nms 

from rfdetr import RFDETRMedium

# =========================
# 全域參數
# =========================
SCORE_RANGE = (0.00, 0.80, 0.02)
NMS_RANGE   = (0.30, 0.90, 0.05)
FIXED_IOU   = 0.5 
TARGET_CLASS_ID = 0
MIN_PRECISION_CONSTRAINT = 0.80 

# =========================
# 工具函式
# =========================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument("--split", type=str, default="valid")
    p.add_argument("--checkpoint_path", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    return p.parse_args()

def load_image_pil(img_path):
    try:
        return Image.open(img_path).convert('RGB')
    except:
        return None

def box_iou_numpy(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clip(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)

# =========================
# 計算邏輯
# =========================
def calculate_metrics(cached_predictions, all_gt_boxes, score_thresh, nms_thresh):
    total_tp = 0
    total_gt = 0
    total_pred = 0 

    for img_idx, raw_pred in enumerate(cached_predictions):
        gt_boxes = all_gt_boxes[img_idx] 
        total_gt += len(gt_boxes)

        mask = (raw_pred['classes'] == TARGET_CLASS_ID) & (raw_pred['scores'] >= score_thresh)
        pred_boxes = raw_pred['boxes'][mask]
        pred_scores = raw_pred['scores'][mask]

        if len(pred_boxes) == 0: continue

        keep = nms(torch.from_numpy(pred_boxes), torch.from_numpy(pred_scores), nms_thresh)
        final_boxes = pred_boxes[keep.numpy()]
        total_pred += len(final_boxes)

        if len(gt_boxes) == 0: continue

        iou_matrix = box_iou_numpy(final_boxes, gt_boxes)
        hits = np.sum(np.max(iou_matrix, axis=0) >= FIXED_IOU)
        total_tp += hits

    recall = total_tp / total_gt if total_gt > 0 else 0.0
    precision = total_tp / total_pred if total_pred > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}

# =========================
# Main
# =========================
def main():
    args = parse_args()
    
    ann_file = os.path.join(args.dataset_dir, args.split, "_annotations.coco.json")
    img_dir = os.path.join(args.dataset_dir, args.split)
    ckpt_path = args.checkpoint_path or os.path.join(args.dataset_dir, "outputs", "rfdetr", "checkpoint_best_total.pth")
    output_dir = args.output_dir or os.path.join(args.dataset_dir, "outputs", "rfdetr_grid_search_final")
    os.makedirs(output_dir, exist_ok=True)

    coco = COCO(ann_file)
    cats = coco.loadCats(coco.getCatIds())
    model = RFDETRMedium(pretrain_weights=ckpt_path, num_classes=len(cats))
    model.optimize_for_inference(compile=False)

    # 1. Cache Inference
    img_ids = coco.getImgIds()
    cached_predictions = [] 
    all_gt_boxes = []      

    print("[INFO] Caching predictions...")
    for img_id in tqdm(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        img = load_image_pil(os.path.join(img_dir, img_info["file_name"]))
        if img is None: continue

        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        gt = []
        for ann in anns:
            cat_idx = next((i for i, c in enumerate(cats) if c['id'] == ann['category_id']), -1)
            if cat_idx == TARGET_CLASS_ID:
                x, y, w, h = ann['bbox']
                gt.append([x, y, x+w, y+h])
        all_gt_boxes.append(np.array(gt) if gt else np.empty((0, 4)))

        pred = model.predict([img], threshold=0.001) 
        det = pred[0] if isinstance(pred, list) else pred
        cached_predictions.append({'boxes': det.xyxy, 'scores': det.confidence, 'classes': det.class_id})

    # 2. Grid Search
    score_vals = np.arange(*SCORE_RANGE)
    nms_vals = np.arange(*NMS_RANGE)
    grid = list(itertools.product(score_vals, nms_vals))
    
    print(f"[INFO] Evaluating {len(grid)} combinations...")
    results = []
    for s, n in tqdm(grid):
        m = calculate_metrics(cached_predictions, all_gt_boxes, s, n)
        results.append({
            "score": round(s, 3), 
            "nms": round(n, 3),
            "precision": m['precision'], 
            "recall": m['recall'], 
            "f1": m['f1']
        })

    df = pd.DataFrame(results)
    
    # 3. 處理排序用的輔助欄位
    # 建立一個 round 過的 recall 欄位，用來分組
    # round(3) 代表小數點後三位相同視為一樣 (0.9123 vs 0.9124 -> 0.912)
    df['recall_sort_key'] = df['recall'].round(3) 

    # 顯示格式
    cols_display = ["score", "nms", "precision", "recall", "f1"]
    fmt = {'precision': '{:.1%}'.format, 'recall': '{:.1%}'.format, 'f1': '{:.1%}'.format}

    # === 策略一：F1 -> Recall ===
    # 這邊因為 F1 是調和平均，通常有明確峰值，比較少發生 float 差異導致的排序問題，但為了保險也可以 round
    df['f1_sort_key'] = df['f1'].round(3)
    df_f1 = df.sort_values(by=["f1_sort_key", "recall"], ascending=[False, False])
    
    print("\n" + "="*65)
    print(" Strategy 1: Best F1-Score")
    print("="*65)
    print(df_f1[cols_display].head(5).to_string(index=False, formatters=fmt))

    # === 策略二：Constraint -> Recall(Rounded) -> Precision ===
    print("\n" + "="*65)
    print(f" Strategy 2: Max Recall (Precision >= {MIN_PRECISION_CONSTRAINT:.0%})")
    print("="*65)
    
    df_constrained = df[df['precision'] >= MIN_PRECISION_CONSTRAINT].copy()
    
    if not df_constrained.empty:
        # 先用 recall_sort_key (模糊化) 排序，如果這一層相同，就會比較 precision
        df_best_recall = df_constrained.sort_values(
            by=["recall_sort_key", "precision"], 
            ascending=[False, False]
        )
        print(df_best_recall[cols_display].head(10).to_string(index=False, formatters=fmt))
        
        best = df_best_recall.iloc[0]
        print(f"\n[RECOMMENDATION] Best Params:")
        print(f"  SCORE = {best['score']}")
        print(f"  NMS   = {best['nms']}")
        print(f"  -> Precision: {best['precision']:.2%}")
        print(f"  -> Recall:    {best['recall']:.2%}")
    else:
        print(f"[WARN] No result found with Precision >= {MIN_PRECISION_CONSTRAINT:.0%}")
        print("Showing Top 5 Precision results:")
        print(df.sort_values("precision", ascending=False)[cols_display].head(5).to_string(index=False, formatters=fmt))

    df.drop(columns=['recall_sort_key', 'f1_sort_key'], inplace=True, errors='ignore')
    df.to_csv(os.path.join(output_dir, "grid_search_final.csv"), index=False)
    print(f"\n[INFO] Saved to {output_dir}")

if __name__ == "__main__":
    main()