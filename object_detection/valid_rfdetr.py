from __future__ import annotations
import os
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from rfdetr import RFDETRMedium

import argparse
from matplotlib import patches
import itertools

# =========================
# 全域變數
# =========================
DATA_ROOT = None
ANN_FILE_PATTERN = None
IMG_DIR_PATTERN = None
CHECKPOINT_PATH = None
OUTPUT_DIR = None

INFER_SCORE_THRESH = 0.001  # 計算 mAP 用 (盡可能包含低分框)
VIS_SCORE_THRESH = 0.3      # 視覺化繪圖用 (過濾雜訊)
DECISION_SCORE_THRESH = 0.25   # 用來產生 confusion matrix / TPR / FPR
IOU_MATCH_THRESH      = 0.45   # 用來判定是否命中 GT


# =========================
# 解析參數
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate RF-DETR with Consolidated Charts")

    p.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="資料集根目錄，需有 train/、valid/、(test/) 及對應的 _annotations.coco.json",
    )
    p.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="模型權重路徑（預設: dataset_dir/outputs/rfdetr/checkpoint_best_total.pth）",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="輸出資料夾（預設: dataset_dir/outputs/rfdetr_eval）",
    )

    return p.parse_args()


# =========================
# 從 COCO JSON 讀取類別數量
# =========================
def get_num_classes_from_coco(dataset_root: str) -> int:
    splits_to_check = ["train", "valid", "test"]
    for split in splits_to_check:
        ann_path = os.path.join(dataset_root, split, "_annotations.coco.json")
        if os.path.exists(ann_path):
            print(f"[INFO] Detecting num_classes from: {ann_path}")
            try:
                with open(ann_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    cats = data.get('categories', [])
                    num_classes = len(cats)
                    print(f"[INFO] Found {num_classes} categories: {[c['name'] for c in cats]}")
                    return num_classes
            except Exception as e:
                print(f"[WARN] Failed to read {ann_path}: {e}")
                continue
    
    raise FileNotFoundError(f"無法在 {dataset_root} 的任何 split 中找到 _annotations.coco.json 來確認 num_classes")


# =========================
# 載入模型
# =========================
def load_model(num_classes: int):
    print(f"[INFO] Loading RFDETRMedium from {CHECKPOINT_PATH} ...")
    print(f"[INFO] Setting model num_classes = {num_classes}")
    
    model = RFDETRMedium(
        pretrain_weights=CHECKPOINT_PATH,
        num_classes=num_classes,
    )
    model.optimize_for_inference(compile=False)
    return model


# =========================
# 輔助函式: 計算 IoU
# =========================
def box_iou_numpy(boxes1, boxes2):
    """
    計算兩個 bbox 陣列的 IoU matrix。
    boxes1: (N, 4) [x1, y1, x2, y2]
    boxes2: (M, 4) [x1, y1, x2, y2]
    Return: (N, M) IoU values
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter
    return inter / union


# =========================
# Confusion Matrix 計算與繪製
# =========================
def compute_confusion_matrix(model, split: str, num_classes: int):
    """
    計算 Confusion Matrix。
    矩陣大小為 (num_classes + 1) x (num_classes + 1)。
    最後一個索引代表 'Background'。
    Rows: Ground Truth
    Cols: Predictions
    """
    ann_file = ANN_FILE_PATTERN.format(split=split)
    img_dir = IMG_DIR_PATTERN.format(split=split)
    
    if not os.path.exists(ann_file):
        print(f"[WARN] No annotation file for CM: {ann_file}")
        return None, None

    coco_gt = COCO(ann_file)
    img_ids = coco_gt.getImgIds()
    
    # 取得類別名稱列表 (確保順序正確，根據 id 排序)
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    cats = sorted(cats, key=lambda x: x['id'])
    class_names = [c['name'] for c in cats]
    class_names.append("Background") # 最後一類為背景

    # 初始化 Confusion Matrix: (GT + Background) x (Pred + Background)
    # confusion_matrix[gt_cls][pred_cls]
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int32)

    for img_id in tqdm(img_ids, desc=f"Calculating CM ({split})"):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info["file_name"])

        img = load_image_pil(img_path)
        if img is None: continue

        # 1. 取得 Ground Truth
        ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        anns = coco_gt.loadAnns(ann_ids)
        
        gt_boxes = []
        gt_classes = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            gt_boxes.append([x, y, x + w, y + h]) # 轉為 xyxy
            # category_id 需要轉回 0-based index
            cat_id = ann['category_id']
            # 尋找 cat_id 在 cats 列表中的 index
            cls_idx = next((i for i, c in enumerate(cats) if c['id'] == cat_id), -1)
            gt_classes.append(cls_idx)
        
        gt_boxes = np.array(gt_boxes) if gt_boxes else np.empty((0, 4))
        gt_classes = np.array(gt_classes, dtype=int)

        # 2. 模型預測
        # 使用 DECISION_SCORE_THRESH 過濾預測
        pred = model.predict([img], threshold=DECISION_SCORE_THRESH)
        detections = pred[0] if isinstance(pred, list) else pred
        
        pred_boxes = detections.xyxy
        pred_scores = detections.confidence
        pred_classes = detections.class_id
        
        if len(pred_boxes) == 0:
            # 只有 FN (所有 GT 都變成 Background)
            for gt_cls in gt_classes:
                cm[gt_cls, num_classes] += 1
            continue

        if len(gt_boxes) == 0:
            # 只有 FP (所有 Pred 都來自 Background)
            for pred_cls in pred_classes:
                cm[num_classes, int(pred_cls)] += 1
            continue

        # 3. 計算 IoU Matrix [Num_Pred, Num_GT]
        # 注意: 這裡習慣用 Pred 對 GT 進行匹配
        iou_mat = box_iou_numpy(pred_boxes, gt_boxes) # Shape: (N_pred, N_gt)

        # 4. 匹配邏輯
        # 標記哪些 GT 已經被匹配過
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)
        
        # 對每個預測框進行檢查 (通常高分優先，但這裡簡化直接依序)
        for i, pred_cls in enumerate(pred_classes):
            pred_cls = int(pred_cls)
            
            # 找出這個 Pred 與所有 GT 的最大 IoU
            max_iou = -1
            best_gt_idx = -1
            
            if iou_mat.shape[1] > 0:
                max_iou = np.max(iou_mat[i])
                best_gt_idx = np.argmax(iou_mat[i])

            if max_iou >= IOU_MATCH_THRESH:
                # 命中某個物體
                if not gt_matched[best_gt_idx]:
                    # 該 GT 尚未被其他 Pred 匹配 -> 這是 TP 或 Classification Error
                    gt_cls = gt_classes[best_gt_idx]
                    cm[gt_cls, pred_cls] += 1
                    gt_matched[best_gt_idx] = True # 鎖定這個 GT
                else:
                    # 該 GT 已經被匹配過了 (Duplicate Detection)
                    # 這個預測框視為 FP (Background -> Pred Class)
                    cm[num_classes, pred_cls] += 1
            else:
                # IoU 不夠高，或是根本沒撞到 -> FP (Background -> Pred Class)
                cm[num_classes, pred_cls] += 1

        # 5. 處理剩下的 FN (未被匹配的 GT)
        for i, gt_cls in enumerate(gt_classes):
            if not gt_matched[i]:
                # GT 存在但沒被檢出 -> FN (GT Class -> Background)
                cm[gt_cls, num_classes] += 1

    return cm, class_names


def plot_confusion_matrix(cm, class_names, save_path, title='Confusion Matrix', normalize=False):
    """
    繪製並儲存 Confusion Matrix 圖像
    """
    if cm is None: return

    # 計算正規化 (Normalized)
    if normalize:
        # 按行 (Row) 正規化：看該 GT 類別被預測成什麼的比例
        # 加上 epsilon 避免除以 0
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        cm_norm = cm.astype('float') / (row_sums + 1e-9)
        # 用於顯示數值的矩陣
        display_cm = cm_norm
        fmt = '.2f'
    else:
        display_cm = cm
        fmt = 'd'

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(display_cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, 
           yticklabels=class_names,
           title=title,
           ylabel='True Label (Ground Truth)',
           xlabel='Predicted Label')

    # 旋轉 X 軸標籤以防重疊
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 在每個格子內標註數值
    thresh = display_cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        val = display_cm[i, j]
        # 如果是 normalized 且數值為 0，就不顯示以保持版面乾淨
        if normalize and val < 0.01:
            continue
        
        color = "white" if val > thresh else "black"
        ax.text(j, i, format(val, fmt),
                ha="center", va="center",
                color=color, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Confusion Matrix saved to {save_path}")
    
    # 額外儲存原始數據 CSV
    csv_path = save_path.replace('.png', '.csv')
    np.savetxt(csv_path, cm, delimiter=",", fmt='%d')
    print(f"[INFO] CM raw data saved to {csv_path}")


# =========================
# 評估相關函式
# =========================
def summarize_coco_eval(coco_eval: COCOeval):
    s = coco_eval.stats
    metrics = {
        "AP_50_95": float(s[0]),
        "AP_50": float(s[1]),
        "AP_75": float(s[2]),
        "AP_small": float(s[3]),
        "AP_medium": float(s[4]),
        "AP_large": float(s[5]),
        "AR_1": float(s[6]),
        "AR_10": float(s[7]),
        "AR_100": float(s[8]),
        "AR_small": float(s[9]),
        "AR_medium": float(s[10]),
        "AR_large": float(s[11]),
    }
    return metrics

def get_zero_metrics():
    return {
        "AP_50_95": 0.0, "AP_50": 0.0, "AP_75": 0.0,
        "AP_small": 0.0, "AP_medium": 0.0, "AP_large": 0.0,
        "AR_1": 0.0, "AR_10": 0.0, "AR_100": 0.0,
        "AR_small": 0.0, "AR_medium": 0.0, "AR_large": 0.0,
    }

def evaluate_coco_split(model, split: str, score_thresh: float = INFER_SCORE_THRESH):
    ann_file = ANN_FILE_PATTERN.format(split=split)
    img_dir = IMG_DIR_PATTERN.format(split=split)
    print(f"\n=== Evaluating split: {split} ===")
    if not os.path.exists(ann_file):
        print(f"[WARN] Annotation file not found: {ann_file}. Skipping {split}.")
        return None

    coco_gt = COCO(ann_file)
    img_ids = coco_gt.getImgIds()
    coco_results = []

    for img_id in tqdm(img_ids, desc=f"Eval Inference {split}"):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info["file_name"])

        img = load_image_pil(img_path)
        if img is None: continue

        pred = model.predict([img], threshold=DECISION_SCORE_THRESH)
        detections = pred[0] if isinstance(pred, list) else pred

        boxes = detections.xyxy
        scores = detections.confidence
        class_ids = detections.class_id

        for bbox, score, cls_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            coco_results.append({
                "image_id": int(img_id),
                "category_id": int(cls_id),
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(score),
            })

    if len(coco_results) == 0:
        return get_zero_metrics()

    det_json_path = os.path.join(OUTPUT_DIR, f"detections_{split}.json")
    with open(det_json_path, "w") as f:
        json.dump(coco_results, f)

    coco_dt = coco_gt.loadRes(det_json_path)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return summarize_coco_eval(coco_eval)

def analyze_size_distribution(split: str):
    ann_file = ANN_FILE_PATTERN.format(split=split)
    if not os.path.exists(ann_file): return None
    with open(ann_file, "r") as f:
        coco_data = json.load(f)
    count_small = count_medium = count_large = 0
    for ann in coco_data["annotations"]:
        w, h = ann["bbox"][2], ann["bbox"][3]
        area = w * h
        if area < 32**2: count_small += 1
        elif area < 96**2: count_medium += 1
        else: count_large += 1
    return {"count_small": count_small, "count_medium": count_medium, "count_large": count_large}


# =========================
# 整合圖表繪製 (Consolidated Plotting)
# =========================

def plot_mAP_summary(all_metrics, save_path):
    splits = [s for s in ['train', 'valid', 'test'] if s in all_metrics and all_metrics[s] is not None]
    if not splits: return
    ap50 = [all_metrics[s]["AP_50"] * 100 for s in splits]
    ap50_95 = [all_metrics[s]["AP_50_95"] * 100 for s in splits]
    x = np.arange(len(splits))
    width = 0.35
    plt.figure(figsize=(8, 6))
    plt.bar(x - width/2, ap50, width, label='mAP@0.5', color='#4e79a7')
    plt.bar(x + width/2, ap50_95, width, label='mAP@0.5:0.95', color='#f28e2b')
    plt.ylabel('Score (%)')
    plt.title('RF-DETR Performance Summary (mAP)')
    plt.xticks(x, splits)
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved mAP summary to {save_path}")

def plot_ap_size_comparison(all_metrics, save_path):
    splits = [s for s in ['train', 'valid', 'test'] if s in all_metrics and all_metrics[s] is not None]
    if not splits: return
    sizes = ['AP_small', 'AP_medium', 'AP_large']
    size_labels = ['Small', 'Medium', 'Large']
    x = np.arange(len(sizes))
    width = 0.25
    plt.figure(figsize=(10, 6))
    colors = {'train': '#a0cbe8', 'valid': '#f28e2b', 'test': '#59a14f'}
    for i, split in enumerate(splits):
        values = [all_metrics[split][k] * 100 for k in sizes]
        offset = (i - len(splits)/2 + 0.5) * width
        plt.bar(x + offset, values, width, label=split, color=colors.get(split, 'gray'))
    plt.ylabel('AP (%)')
    plt.title('Analysis by Object Size: AP Comparison')
    plt.xticks(x, size_labels)
    plt.ylim(0, 100)
    plt.legend(title='Dataset Split')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved AP size comparison to {save_path}")

def plot_data_distribution_comparison(all_size_info, save_path):
    splits = [s for s in ['train', 'valid', 'test'] if s in all_size_info and all_size_info[s] is not None]
    if not splits: return
    small_ratios = []
    medium_ratios = []
    large_ratios = []
    for s in splits:
        d = all_size_info[s]
        total = d['count_small'] + d['count_medium'] + d['count_large']
        if total == 0:
            small_ratios.append(0); medium_ratios.append(0); large_ratios.append(0)
        else:
            small_ratios.append(d['count_small'] / total * 100)
            medium_ratios.append(d['count_medium'] / total * 100)
            large_ratios.append(d['count_large'] / total * 100)
    x = splits
    plt.figure(figsize=(8, 6))
    plt.bar(x, small_ratios, label='Small', color='#76b7b2')
    plt.bar(x, medium_ratios, bottom=small_ratios, label='Medium', color='#edc948')
    bottom_large = [i+j for i,j in zip(small_ratios, medium_ratios)]
    plt.bar(x, large_ratios, bottom=bottom_large, label='Large', color='#e15759')
    plt.ylabel('Percentage (%)')
    plt.title('Data Distribution by Object Size')
    plt.ylim(0, 100)
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    for i, s in enumerate(splits):
        if small_ratios[i] > 5:
            plt.text(i, small_ratios[i]/2, f"{small_ratios[i]:.1f}%", ha='center', va='center', color='black', fontsize=9)
        if medium_ratios[i] > 5:
            plt.text(i, small_ratios[i] + medium_ratios[i]/2, f"{medium_ratios[i]:.1f}%", ha='center', va='center', color='black', fontsize=9)
        if large_ratios[i] > 5:
            plt.text(i, small_ratios[i] + medium_ratios[i] + large_ratios[i]/2, f"{large_ratios[i]:.1f}%", ha='center', va='center', color='black', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved data distribution chart to {save_path}")


# =========================
# 全量視覺化
# =========================
def visualize_all_predictions(model, split: str, score_thresh: float = VIS_SCORE_THRESH):
    ann_file = ANN_FILE_PATTERN.format(split=split)
    img_dir = IMG_DIR_PATTERN.format(split=split)
    if not os.path.exists(ann_file): return
    viz_save_dir = os.path.join(OUTPUT_DIR, "viz_results", split)
    os.makedirs(viz_save_dir, exist_ok=True)
    print(f"[INFO] Visualizing ALL images for split: {split} -> {viz_save_dir}")
    coco_gt = COCO(ann_file)
    img_ids = coco_gt.getImgIds()
    if not img_ids: return
    for img_id in tqdm(img_ids, desc=f"Viz {split}"):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info["file_name"])
        img = load_image_pil(img_path)
        if img is None: continue
        ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        anns = coco_gt.loadAnns(ann_ids)
        pred = model.predict([img], threshold=DECISION_SCORE_THRESH)
        detections = pred[0] if isinstance(pred, list) else pred
        pred_boxes = detections.xyxy
        pred_scores = detections.confidence
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(img)
        for ann in anns:
            x, y, w, h = ann["bbox"]
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="lime", facecolor="none")
            ax.add_patch(rect)
            ax.text(x, y - 2, "GT", fontsize=8, color="lime", weight='bold')
        for bbox, score in zip(pred_boxes, pred_scores):
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor="red", facecolor="none")
            ax.add_patch(rect)
            ax.text(x1, y1 - 2, f"{score:.2f}", fontsize=8, color="red", weight='bold',
                    bbox=dict(facecolor="black", alpha=0.3, pad=1))
        ax.set_axis_off()
        ax.set_title(f"RF-DETR {split} | id={img_id}")
        plt.tight_layout()
        file_basename = os.path.splitext(os.path.basename(img_info["file_name"]))[0]
        out_name = f"{file_basename}_vis.png"
        plt.savefig(os.path.join(viz_save_dir, out_name), dpi=150)
        plt.close(fig)


def load_image_pil(img_path):
    try:
        # 開啟圖片並強制轉為 RGB (避免 RGBA 或 Grayscale 導致錯誤)
        img = Image.open(img_path).convert('RGB')
        return img
    except Exception as e:
        print(f"[WARN] Failed to load image {img_path}: {e}")
        return None


# =========================
# Main
# =========================
def main():
    global DATA_ROOT, ANN_FILE_PATTERN, IMG_DIR_PATTERN, CHECKPOINT_PATH, OUTPUT_DIR
    args = parse_args()
    DATA_ROOT = args.dataset_dir
    ANN_FILE_PATTERN = os.path.join(DATA_ROOT, "{split}", "_annotations.coco.json")
    IMG_DIR_PATTERN = os.path.join(DATA_ROOT, "{split}")
    
    if args.checkpoint_path is None:
        CHECKPOINT_PATH = os.path.join(DATA_ROOT, "outputs", "rfdetr", "checkpoint_best_total.pth")
    else:
        CHECKPOINT_PATH = args.checkpoint_path

    if args.output_dir is None:
        OUTPUT_DIR = os.path.join(DATA_ROOT, "outputs", "rfdetr_eval")
    else:
        OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[INFO] DATA_ROOT:", DATA_ROOT)
    print("[INFO] OUTPUT_DIR:", OUTPUT_DIR)
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[ERROR] Checkpoint not found at {CHECKPOINT_PATH}")
        return
    
    try:
        num_classes = get_num_classes_from_coco(DATA_ROOT)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    model = load_model(num_classes)
    splits = ["train", "valid", "test"]
    all_metrics = {}
    all_size_info = {}

    # 1. 計算所有 Split 的 Metrics 與 Size Info
    for split in splits:
        metrics = evaluate_coco_split(model, split)
        all_metrics[split] = metrics
        size_info = analyze_size_distribution(split)
        all_size_info[split] = size_info

    with open(os.path.join(OUTPUT_DIR, "metrics_all_splits.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    # 2. 繪製整合圖表
    print("\n[INFO] Generating Consolidated Charts...")
    plot_mAP_summary(all_metrics, os.path.join(OUTPUT_DIR, "mAP_Summary.png"))
    plot_ap_size_comparison(all_metrics, os.path.join(OUTPUT_DIR, "Analysis_by_Size_Comparison.png"))
    plot_data_distribution_comparison(all_size_info, os.path.join(OUTPUT_DIR, "Data_Distribution_Comparison.png"))
    
    # 3. 繪製 Confusion Matrix
    target_cm_split = splits[2]
    print(f"\n[INFO] Generating Confusion Matrix for split: {target_cm_split}")
    cm, class_names = compute_confusion_matrix(model, target_cm_split, num_classes)
    if cm is not None:
        cm_save_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{target_cm_split}.png")
        plot_confusion_matrix(cm, class_names, cm_save_path)
        plot_confusion_matrix(cm, class_names, cm_save_path.replace('.png', '_normalized.png'), 
                              title="Confusion Matrix Normalized", normalize=True)

    # 4. 全量視覺化
    target_vis_split = splits[2]
    print(f"\n[INFO] Visualizing all predictions for split: {target_vis_split}")
    visualize_all_predictions(model, split=target_vis_split)

    print("\n=== Done. 評估完成 ===")
    print(f"圖表與結果已儲存至: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()