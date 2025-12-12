from __future__ import annotations
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 根據原本的 import 改為 RF-DETR
from rfdetr import RFDETRMedium

import argparse
from matplotlib import patches

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
# 載入模型
# =========================
def load_model():
    print(f"[INFO] Loading RFDETRMedium from {CHECKPOINT_PATH} ...")
    # 這裡依據原本 valid_rfdetr.py 的寫法設定 num_classes=2
    model = RFDETRMedium(
        pretrain_weights=CHECKPOINT_PATH,
        num_classes=2,
    )
    # RF-DETR 特有的優化
    model.optimize_for_inference(compile=False)
    return model


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

        img = cv2.imread(img_path)
        if img is None: continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # RF-DETR 推論 API: model.predict([img]) -> list[Detections]
        detections_list = model.predict([img_rgb], threshold=score_thresh)
        detections = detections_list[0]

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
    """圖表 1: Train/Valid/Test 整體 mAP 比較"""
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
    """圖表 2: 各 Split 在不同物件大小上的 AP 表現"""
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
    """圖表 3: 各 Split 的物件大小分佈比例"""
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
        
        img = cv2.imread(img_path)
        if img is None: continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        anns = coco_gt.loadAnns(ann_ids)

        # RF-DETR Inference
        detections_list = model.predict([img_rgb], threshold=score_thresh)
        detections = detections_list[0]
        pred_boxes = detections.xyxy
        pred_scores = detections.confidence

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(img_rgb)

        # GT (Lime)
        for ann in anns:
            x, y, w, h = ann["bbox"]
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="lime", facecolor="none")
            ax.add_patch(rect)
            ax.text(x, y - 2, "GT", fontsize=8, color="lime", weight='bold')

        # Pred (Red)
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


# =========================
# Main
# =========================
def main():
    global DATA_ROOT, ANN_FILE_PATTERN, IMG_DIR_PATTERN, CHECKPOINT_PATH, OUTPUT_DIR

    args = parse_args()
    DATA_ROOT = args.dataset_dir
    ANN_FILE_PATTERN = os.path.join(DATA_ROOT, "{split}", "_annotations.coco.json")
    IMG_DIR_PATTERN = os.path.join(DATA_ROOT, "{split}")

    # RF-DETR 預設路徑
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
    model = load_model()

    splits = ["train", "valid", "test"]
    all_metrics = {}
    all_size_info = {}

    # 1. 計算所有 Split 的 Metrics 與 Size Info
    for split in splits:
        metrics = evaluate_coco_split(model, split)
        all_metrics[split] = metrics
        
        size_info = analyze_size_distribution(split)
        all_size_info[split] = size_info

    # 存下完整數據
    with open(os.path.join(OUTPUT_DIR, "metrics_all_splits.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    # 2. 繪製整合圖表
    print("\n[INFO] Generating Consolidated Charts...")
    plot_mAP_summary(all_metrics, os.path.join(OUTPUT_DIR, "mAP_Summary.png"))
    plot_ap_size_comparison(all_metrics, os.path.join(OUTPUT_DIR, "Analysis_by_Size_Comparison.png"))
    plot_data_distribution_comparison(all_size_info, os.path.join(OUTPUT_DIR, "Data_Distribution_Comparison.png"))

    # 3. 全量視覺化 (可視需求註解掉 train)
    # visualize_all_predictions(model, split="train")
    visualize_all_predictions(model, split="valid")
    visualize_all_predictions(model, split="test")

    print("\n=== Done. 評估完成 ===")
    print(f"圖表與結果已儲存至: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
