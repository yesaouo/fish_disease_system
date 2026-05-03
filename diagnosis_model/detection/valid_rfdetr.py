from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from rfdetr import RFDETRMedium
from tqdm import tqdm

SPLITS = ("train", "valid", "test")
EVAL_KEYS = (
    "AP_50_95", "AP_50", "AP_75", "AP_small", "AP_medium", "AP_large",
    "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large",
)


@dataclass(frozen=True)
class EvalConfig:
    dataset_dir: Path
    checkpoint_path: Path = Path("outputs/rfdetr/checkpoint_best_total.pth")
    output_dir: Path = Path("outputs/rfdetr_eval")
    infer_score_thresh: float = 0.001       # COCO mAP：盡量保留低分框
    vis_score_thresh: float = 0.3           # 視覺化：過濾雜訊
    decision_score_thresh: float = 0.25     # Confusion matrix / TPR / FPR
    iou_match_thresh: float = 0.45          # 判定是否命中 GT

    def ann_file(self, split: str) -> Path:
        return self.dataset_dir / split / "_annotations.coco.json"

    def img_dir(self, split: str) -> Path:
        return self.dataset_dir / split


# -------------------------
# 基本工具
# -------------------------
def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="Evaluate RF-DETR with consolidated charts")
    parser.add_argument("--dataset_dir", required=True, type=Path,
                        help="資料集根目錄，需有 train/、valid/、(test/) 與 _annotations.coco.json")
    parser.add_argument("--checkpoint_path", type=Path, default=EvalConfig.checkpoint_path,
                        help="模型權重路徑")
    parser.add_argument("--output_dir", type=Path, default=EvalConfig.output_dir,
                        help="輸出資料夾")
    return EvalConfig(**vars(parser.parse_args()))


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_image(path: Path):
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except Exception as exc:
        print(f"[WARN] Failed to load image {path}: {exc}")
        return None


def existing_splits(cfg: EvalConfig) -> list[str]:
    return [split for split in SPLITS if cfg.ann_file(split).exists()]


def get_categories(cfg: EvalConfig) -> list[dict]:
    for split in SPLITS:
        ann_path = cfg.ann_file(split)
        if ann_path.exists():
            cats = sorted(load_json(ann_path).get("categories", []), key=lambda c: c["id"])
            print(f"[INFO] Found {len(cats)} categories from {ann_path}: {[c['name'] for c in cats]}")
            return cats
    raise FileNotFoundError(f"無法在 {cfg.dataset_dir} 找到任何 _annotations.coco.json")


def load_model(cfg: EvalConfig, num_classes: int):
    print(f"[INFO] Loading RFDETRMedium: {cfg.checkpoint_path}")
    model = RFDETRMedium(pretrain_weights=str(cfg.checkpoint_path), num_classes=num_classes)
    model.optimize_for_inference(compile=False)
    return model


def predict_one(model, img, threshold: float):
    pred = model.predict([img], threshold=threshold)
    return pred[0] if isinstance(pred, list) else pred


def iter_images(coco: COCO, img_dir: Path, desc: str):
    for img_id in tqdm(coco.getImgIds(), desc=desc):
        img_info = coco.loadImgs(img_id)[0]
        img = load_image(img_dir / img_info["file_name"])
        if img is not None:
            yield img_id, img_info, img


def anns_to_arrays(anns: list[dict], cat_id_to_idx: dict[int, int]):
    boxes, classes = [], []
    for ann in anns:
        x, y, w, h = ann["bbox"]
        boxes.append([x, y, x + w, y + h])
        classes.append(cat_id_to_idx[ann["category_id"]])
    return (
        np.asarray(boxes, dtype=float).reshape(-1, 4),
        np.asarray(classes, dtype=int),
    )


def xyxy_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.empty((len(boxes1), len(boxes2)))

    area1 = np.prod(boxes1[:, 2:] - boxes1[:, :2], axis=1)
    area2 = np.prod(boxes2[:, 2:] - boxes2[:, :2], axis=1)
    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2 - inter
    return inter / np.maximum(union, 1e-9)


# -------------------------
# COCO 評估
# -------------------------
def zero_metrics() -> dict[str, float]:
    return {key: 0.0 for key in EVAL_KEYS}


def summarize_coco_eval(coco_eval: COCOeval) -> dict[str, float]:
    return {key: float(value) for key, value in zip(EVAL_KEYS, coco_eval.stats)}


def detections_to_coco(detections, image_id: int, cat_ids: list[int]) -> list[dict]:
    rows = []
    for bbox, score, cls_id in zip(detections.xyxy, detections.confidence, detections.class_id):
        cls_idx = int(cls_id)
        if not 0 <= cls_idx < len(cat_ids):
            continue
        x1, y1, x2, y2 = map(float, bbox)
        rows.append({
            "image_id": int(image_id),
            "category_id": int(cat_ids[cls_idx]),
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": float(score),
        })
    return rows


def evaluate_coco_split(model, cfg: EvalConfig, split: str, cat_ids: list[int]) -> dict[str, float] | None:
    ann_path = cfg.ann_file(split)
    if not ann_path.exists():
        print(f"[WARN] Annotation file not found: {ann_path}. Skip {split}.")
        return None

    print(f"\n=== Evaluating split: {split} ===")
    coco_gt = COCO(str(ann_path))
    coco_results = []

    for img_id, _, img in iter_images(coco_gt, cfg.img_dir(split), f"Eval {split}"):
        det = predict_one(model, img, cfg.infer_score_thresh)
        coco_results.extend(detections_to_coco(det, img_id, cat_ids))

    if not coco_results:
        return zero_metrics()

    det_path = cfg.output_dir / f"detections_{split}.json"
    save_json(coco_results, det_path)

    coco_dt = coco_gt.loadRes(str(det_path))
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return summarize_coco_eval(coco_eval)


def analyze_size_distribution(cfg: EvalConfig, split: str) -> dict[str, int] | None:
    ann_path = cfg.ann_file(split)
    if not ann_path.exists():
        return None

    counts = {"count_small": 0, "count_medium": 0, "count_large": 0}
    for ann in load_json(ann_path).get("annotations", []):
        area = ann["bbox"][2] * ann["bbox"][3]
        if area < 32 ** 2:
            counts["count_small"] += 1
        elif area < 96 ** 2:
            counts["count_medium"] += 1
        else:
            counts["count_large"] += 1
    return counts


# -------------------------
# Confusion matrix
# -------------------------
def compute_confusion_matrix(model, cfg: EvalConfig, split: str, cats: list[dict]):
    ann_path = cfg.ann_file(split)
    if not ann_path.exists():
        print(f"[WARN] No annotation file for CM: {ann_path}")
        return None, None

    coco_gt = COCO(str(ann_path))
    num_classes = len(cats)
    cat_id_to_idx = {cat["id"]: i for i, cat in enumerate(cats)}
    class_names = [cat["name"] for cat in cats] + ["Background"]
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int32)

    for img_id, _, img in iter_images(coco_gt, cfg.img_dir(split), f"CM {split}"):
        anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id]))
        gt_boxes, gt_classes = anns_to_arrays(anns, cat_id_to_idx)
        det = predict_one(model, img, cfg.decision_score_thresh)

        pred_boxes = np.asarray(det.xyxy, dtype=float).reshape(-1, 4)
        pred_scores = np.asarray(det.confidence, dtype=float)
        pred_classes = np.asarray(det.class_id, dtype=int)

        if len(pred_boxes) == 0:
            for gt_cls in gt_classes:
                cm[gt_cls, num_classes] += 1
            continue
        if len(gt_boxes) == 0:
            for pred_cls in pred_classes:
                if 0 <= pred_cls < num_classes:
                    cm[num_classes, pred_cls] += 1
            continue

        iou_mat = xyxy_iou(pred_boxes, gt_boxes)
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)

        # 高分預測優先配對，避免低分框先搶到 GT
        for pred_idx in np.argsort(-pred_scores):
            pred_cls = int(pred_classes[pred_idx])
            if not 0 <= pred_cls < num_classes:
                continue

            best_gt = int(np.argmax(iou_mat[pred_idx]))
            if iou_mat[pred_idx, best_gt] >= cfg.iou_match_thresh and not gt_matched[best_gt]:
                cm[gt_classes[best_gt], pred_cls] += 1
                gt_matched[best_gt] = True
            else:
                cm[num_classes, pred_cls] += 1

        for gt_idx, gt_cls in enumerate(gt_classes):
            if not gt_matched[gt_idx]:
                cm[gt_cls, num_classes] += 1

    return cm, class_names


# -------------------------
# 繪圖
# -------------------------
def save_bar_chart(save_path: Path, title: str, ylabel: str, labels, series: dict[str, list[float]], ylim=(0, 100)):
    x = np.arange(len(labels))
    width = 0.8 / max(1, len(series))
    fig, ax = plt.subplots(figsize=(9, 6))

    for i, (name, values) in enumerate(series.items()):
        offset = (i - (len(series) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=name)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, labels)
    ax.set_ylim(*ylim)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved chart: {save_path}")


def plot_metric_charts(all_metrics: dict, all_size_info: dict, cfg: EvalConfig) -> None:
    splits = [s for s in SPLITS if all_metrics.get(s)]
    if splits:
        save_bar_chart(
            cfg.output_dir / "mAP_Summary.png",
            "RF-DETR Performance Summary (mAP)",
            "Score (%)",
            splits,
            {
                "mAP@0.5": [all_metrics[s]["AP_50"] * 100 for s in splits],
                "mAP@0.5:0.95": [all_metrics[s]["AP_50_95"] * 100 for s in splits],
            },
        )
        save_bar_chart(
            cfg.output_dir / "Analysis_by_Size_Comparison.png",
            "Analysis by Object Size: AP Comparison",
            "AP (%)",
            ["Small", "Medium", "Large"],
            {s: [all_metrics[s][k] * 100 for k in ("AP_small", "AP_medium", "AP_large")] for s in splits},
        )

    size_splits = [s for s in SPLITS if all_size_info.get(s)]
    if not size_splits:
        return

    ratios = {"Small": [], "Medium": [], "Large": []}
    for split in size_splits:
        d = all_size_info[split]
        total = max(1, d["count_small"] + d["count_medium"] + d["count_large"])
        ratios["Small"].append(d["count_small"] / total * 100)
        ratios["Medium"].append(d["count_medium"] / total * 100)
        ratios["Large"].append(d["count_large"] / total * 100)

    fig, ax = plt.subplots(figsize=(8, 6))
    bottom = np.zeros(len(size_splits))
    for name, values in ratios.items():
        values = np.asarray(values)
        ax.bar(size_splits, values, bottom=bottom, label=name)
        for i, value in enumerate(values):
            if value > 5:
                ax.text(i, bottom[i] + value / 2, f"{value:.1f}%", ha="center", va="center", fontsize=9)
        bottom += values

    ax.set_title("Data Distribution by Object Size")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(cfg.output_dir / "Data_Distribution_Comparison.png", dpi=300)
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, save_path: Path, title="Confusion Matrix", normalize=False):
    if cm is None:
        return

    display = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1e-9) if normalize else cm
    fmt = ".2f" if normalize else "d"

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(display, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
        xticklabels=class_names, yticklabels=class_names,
        title=title, ylabel="True Label (Ground Truth)", xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = display.max() / 2 if display.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = display[i, j]
            if normalize and value < 0.01:
                continue
            ax.text(j, i, format(value, fmt), ha="center", va="center",
                    color="white" if value > threshold else "black", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    np.savetxt(save_path.with_suffix(".csv"), cm, delimiter=",", fmt="%d")
    print(f"[INFO] Saved confusion matrix: {save_path}")


# -------------------------
# 視覺化
# -------------------------
def visualize_split(model, cfg: EvalConfig, split: str) -> None:
    ann_path = cfg.ann_file(split)
    if not ann_path.exists():
        return

    save_dir = cfg.output_dir / "viz_results" / split
    save_dir.mkdir(parents=True, exist_ok=True)
    coco_gt = COCO(str(ann_path))
    print(f"[INFO] Visualizing {split} -> {save_dir}")

    for img_id, img_info, img in iter_images(coco_gt, cfg.img_dir(split), f"Viz {split}"):
        anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id]))
        det = predict_one(model, img, cfg.vis_score_thresh)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img)

        for ann in anns:
            x, y, w, h = ann["bbox"]
            ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="lime", facecolor="none"))
            ax.text(x, y - 2, "GT", fontsize=8, color="lime", weight="bold")

        for bbox, score in zip(det.xyxy, det.confidence):
            x1, y1, x2, y2 = bbox
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                           linewidth=2, edgecolor="red", facecolor="none"))
            ax.text(x1, y1 - 2, f"{score:.2f}", fontsize=8, color="red", weight="bold",
                    bbox=dict(facecolor="black", alpha=0.3, pad=1))

        ax.set_axis_off()
        ax.set_title(f"RF-DETR {split} | id={img_id}")
        fig.tight_layout()
        out_name = f"{Path(img_info['file_name']).stem}_vis.png"
        fig.savefig(save_dir / out_name, dpi=150)
        plt.close(fig)


# -------------------------
# Main
# -------------------------
def main() -> None:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] DATA_ROOT: {cfg.dataset_dir}")
    print(f"[INFO] OUTPUT_DIR: {cfg.output_dir}")

    if not cfg.checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {cfg.checkpoint_path}")
        return

    try:
        cats = get_categories(cfg)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return

    model = load_model(cfg, num_classes=len(cats))
    cat_ids = [cat["id"] for cat in cats]

    all_metrics = {split: evaluate_coco_split(model, cfg, split, cat_ids) for split in SPLITS}
    all_size_info = {split: analyze_size_distribution(cfg, split) for split in SPLITS}
    save_json(all_metrics, cfg.output_dir / "metrics_all_splits.json")

    print("\n[INFO] Generating consolidated charts...")
    plot_metric_charts(all_metrics, all_size_info, cfg)

    target_split = "test" if "test" in existing_splits(cfg) else existing_splits(cfg)[-1]
    print(f"\n[INFO] Generating confusion matrix for split: {target_split}")
    cm, class_names = compute_confusion_matrix(model, cfg, target_split, cats)
    if cm is not None:
        cm_path = cfg.output_dir / f"confusion_matrix_{target_split}.png"
        plot_confusion_matrix(cm, class_names, cm_path)
        plot_confusion_matrix(cm, class_names, cm_path.with_name(f"{cm_path.stem}_normalized.png"),
                              title="Confusion Matrix Normalized", normalize=True)

    print(f"\n[INFO] Visualizing predictions for split: {target_split}")
    visualize_split(model, cfg, target_split)

    print("\n=== Done. 評估完成 ===")
    print(f"圖表與結果已儲存至: {cfg.output_dir.resolve()}")


if __name__ == "__main__":
    main()
