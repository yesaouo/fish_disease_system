from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pycocotools.coco import COCO
from rfdetr import RFDETRMedium
from torchvision.ops import nms
from tqdm import tqdm


@dataclass(frozen=True)
class GridConfig:
    dataset_dir: Path
    split: str
    checkpoint_path: Path
    output_dir: Path
    score_range: tuple[float, float, float]
    nms_range: tuple[float, float, float]
    target_class_idx: int = 0
    infer_threshold: float = 0.001
    iou_match_thresh: float = 0.5
    min_precision: float = 0.80

    @property
    def ann_file(self) -> Path:
        return self.dataset_dir / self.split / "_annotations.coco.json"

    @property
    def img_dir(self) -> Path:
        return self.dataset_dir / self.split

    @property
    def csv_path(self) -> Path:
        return self.output_dir / "grid_search_final.csv"

    @property
    def recommendation_path(self) -> Path:
        return self.output_dir / "grid_search_recommendation.txt"


def parse_args() -> GridConfig:
    p = argparse.ArgumentParser(description="Grid search SCORE/NMS thresholds for RF-DETR.")
    p.add_argument("--dataset_dir", type=Path, required=True)
    p.add_argument("--split", type=str, default="valid")
    p.add_argument("--checkpoint_path", type=Path, default=Path("outputs/rfdetr/checkpoint_best_total.pth"))
    p.add_argument("--output_dir", type=Path, default=Path("outputs/rfdetr_grid_search"))

    p.add_argument("--score_range", type=float, nargs=3, default=(0.00, 0.80, 0.02), metavar=("START", "STOP", "STEP"))
    p.add_argument("--nms_range", type=float, nargs=3, default=(0.30, 0.90, 0.05), metavar=("START", "STOP", "STEP"))
    p.add_argument("--target_class_idx", "--target_class_id", type=int, default=0)
    p.add_argument("--infer_threshold", type=float, default=0.001)
    p.add_argument("--iou_match_thresh", type=float, default=0.5)
    p.add_argument("--min_precision", type=float, default=0.80)
    args = p.parse_args()

    return GridConfig(
        dataset_dir=args.dataset_dir,
        split=args.split,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        score_range=tuple(args.score_range),
        nms_range=tuple(args.nms_range),
        target_class_idx=args.target_class_idx,
        infer_threshold=args.infer_threshold,
        iou_match_thresh=args.iou_match_thresh,
        min_precision=args.min_precision,
    )


def frange(spec: tuple[float, float, float]) -> np.ndarray:
    start, stop, step = spec
    # round 避免 0.30000000004 這類浮點數進 CSV/排序。
    return np.round(np.arange(start, stop, step), 6)


def load_image(path: Path) -> Image.Image | None:
    try:
        return Image.open(path).convert("RGB")
    except Exception as exc:
        print(f"[WARN] Failed to load image {path}: {exc}")
        return None


def box_iou_numpy(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    area1 = np.clip(boxes1[:, 2] - boxes1[:, 0], 0, None) * np.clip(boxes1[:, 3] - boxes1[:, 1], 0, None)
    area2 = np.clip(boxes2[:, 2] - boxes2[:, 0], 0, None) * np.clip(boxes2[:, 3] - boxes2[:, 1], 0, None)

    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.clip(union, 1e-9, None)


def xywh_to_xyxy(box: Iterable[float]) -> list[float]:
    x, y, w, h = box
    return [x, y, x + w, y + h]


def as_numpy(x, dtype=None) -> np.ndarray:
    arr = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    return arr.astype(dtype) if dtype is not None else arr


def apply_nms(boxes: np.ndarray, scores: np.ndarray, nms_thresh: float) -> tuple[np.ndarray, np.ndarray]:
    if len(boxes) == 0:
        return boxes, scores

    keep = nms(
        torch.as_tensor(boxes, dtype=torch.float32),
        torch.as_tensor(scores, dtype=torch.float32),
        float(nms_thresh),
    ).cpu().numpy()
    return boxes[keep], scores[keep]


def count_one_to_one_tp(pred_boxes: np.ndarray, pred_scores: np.ndarray, gt_boxes: np.ndarray, iou_thresh: float) -> int:
    """Confidence-descending one-to-one matching; duplicate detections are not counted as extra TP."""
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0

    ious = box_iou_numpy(pred_boxes, gt_boxes)
    gt_used = np.zeros(len(gt_boxes), dtype=bool)
    tp = 0

    for pred_idx in np.argsort(-pred_scores):
        available_gt = np.flatnonzero(~gt_used)
        if len(available_gt) == 0:
            break

        best_local_idx = np.argmax(ious[pred_idx, available_gt])
        best_gt_idx = available_gt[best_local_idx]
        if ious[pred_idx, best_gt_idx] >= iou_thresh:
            gt_used[best_gt_idx] = True
            tp += 1

    return tp


def load_categories(coco: COCO) -> tuple[list[dict], dict[int, int]]:
    cats = sorted(coco.loadCats(coco.getCatIds()), key=lambda c: c["id"])
    return cats, {cat["id"]: idx for idx, cat in enumerate(cats)}


def cache_predictions(model, coco: COCO, cfg: GridConfig, cat_id_to_idx: dict[int, int]) -> list[dict[str, np.ndarray]]:
    items: list[dict[str, np.ndarray]] = []
    img_ids = coco.getImgIds()

    for img_id in tqdm(img_ids, desc="Caching predictions"):
        img_info = coco.loadImgs(img_id)[0]
        img = load_image(cfg.img_dir / img_info["file_name"])
        if img is None:
            continue

        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
        gt_boxes = [
            xywh_to_xyxy(ann["bbox"])
            for ann in anns
            if cat_id_to_idx.get(ann["category_id"]) == cfg.target_class_idx
        ]

        pred = model.predict([img], threshold=cfg.infer_threshold)
        det = pred[0] if isinstance(pred, list) else pred

        items.append({
            "gt_boxes": np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4),
            "boxes": as_numpy(det.xyxy, np.float32).reshape(-1, 4),
            "scores": as_numpy(det.confidence, np.float32),
            "classes": as_numpy(det.class_id, np.int64),
        })

    return items


def calculate_metrics(items: list[dict[str, np.ndarray]], cfg: GridConfig, score_thresh: float, nms_thresh: float) -> dict[str, float]:
    total_tp = total_gt = total_pred = 0

    for item in items:
        gt_boxes = item["gt_boxes"]
        total_gt += len(gt_boxes)

        keep_cls_score = (item["classes"] == cfg.target_class_idx) & (item["scores"] >= score_thresh)
        pred_boxes = item["boxes"][keep_cls_score]
        pred_scores = item["scores"][keep_cls_score]

        pred_boxes, pred_scores = apply_nms(pred_boxes, pred_scores, nms_thresh)
        total_pred += len(pred_boxes)
        total_tp += count_one_to_one_tp(pred_boxes, pred_scores, gt_boxes, cfg.iou_match_thresh)

    precision = total_tp / total_pred if total_pred else 0.0
    recall = total_tp / total_gt if total_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": total_tp, "gt": total_gt, "pred": total_pred}


def run_grid_search(items: list[dict[str, np.ndarray]], cfg: GridConfig) -> pd.DataFrame:
    grid = list(itertools.product(frange(cfg.score_range), frange(cfg.nms_range)))
    rows = []

    for score, nms_thresh in tqdm(grid, desc=f"Evaluating {len(grid)} combinations"):
        m = calculate_metrics(items, cfg, float(score), float(nms_thresh))
        rows.append({"score": score, "nms": nms_thresh, **m})

    return pd.DataFrame(rows)


def print_table(title: str, df: pd.DataFrame, top_k: int = 5) -> None:
    cols = ["score", "nms", "precision", "recall", "f1", "tp", "gt", "pred"]
    fmt = {"precision": "{:.1%}".format, "recall": "{:.1%}".format, "f1": "{:.1%}".format}
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(df[cols].head(top_k).to_string(index=False, formatters=fmt))


def recommend(df: pd.DataFrame, cfg: GridConfig) -> pd.Series:
    constrained = df[df["precision"] >= cfg.min_precision].copy()

    if constrained.empty:
        print(f"[WARN] No result with precision >= {cfg.min_precision:.0%}; fallback to best F1.")
        return df.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0]

    # recall 先 round，可避免浮點誤差讓幾乎相同 recall 的參數排序不穩。
    constrained["recall_key"] = constrained["recall"].round(3)
    return constrained.sort_values(["recall_key", "precision", "f1"], ascending=False).iloc[0]


def save_recommendation(best: pd.Series, cfg: GridConfig) -> None:
    text = (
        "[RECOMMENDATION] Best Params\n"
        f"SCORE = {best['score']:.6g}\n"
        f"NMS   = {best['nms']:.6g}\n"
        f"Precision = {best['precision']:.2%}\n"
        f"Recall    = {best['recall']:.2%}\n"
        f"F1        = {best['f1']:.2%}\n"
        f"TP / GT / Pred = {int(best['tp'])} / {int(best['gt'])} / {int(best['pred'])}\n"
    )
    cfg.recommendation_path.write_text(text, encoding="utf-8")
    print("\n" + text.rstrip())


def main() -> None:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.ann_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {cfg.ann_file}")
    if not cfg.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint_path}")

    coco = COCO(str(cfg.ann_file))
    cats, cat_id_to_idx = load_categories(coco)
    if not 0 <= cfg.target_class_idx < len(cats):
        raise ValueError(f"target_class_idx={cfg.target_class_idx} is out of range; num_classes={len(cats)}")

    print(f"[INFO] Split: {cfg.split}")
    print(f"[INFO] Target class: idx={cfg.target_class_idx}, name={cats[cfg.target_class_idx]['name']!r}")
    print(f"[INFO] Output dir: {cfg.output_dir}")

    model = RFDETRMedium(pretrain_weights=str(cfg.checkpoint_path), num_classes=len(cats))
    model.optimize_for_inference(compile=False)

    items = cache_predictions(model, coco, cfg, cat_id_to_idx)
    if not items:
        raise RuntimeError("No valid images were cached. Check image paths and annotation file.")

    df = run_grid_search(items, cfg)
    df.to_csv(cfg.csv_path, index=False)

    best_f1 = df.sort_values(["f1", "recall", "precision"], ascending=False)
    print_table("Strategy 1: Best F1-Score", best_f1)

    constrained = df[df["precision"] >= cfg.min_precision].copy()
    if not constrained.empty:
        constrained["recall_key"] = constrained["recall"].round(3)
        print_table(f"Strategy 2: Max Recall under Precision >= {cfg.min_precision:.0%}",
                    constrained.sort_values(["recall_key", "precision", "f1"], ascending=False), top_k=10)
    else:
        print_table(f"Fallback: Top Precision because no result >= {cfg.min_precision:.0%}",
                    df.sort_values(["precision", "recall", "f1"], ascending=False))

    best = recommend(df, cfg)
    save_recommendation(best, cfg)
    print(f"[INFO] CSV saved to: {cfg.csv_path}")
    print(f"[INFO] Recommendation saved to: {cfg.recommendation_path}")


if __name__ == "__main__":
    main()
