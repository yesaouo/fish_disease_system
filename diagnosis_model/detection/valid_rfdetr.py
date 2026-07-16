"""RF-DETR baseline 偵測評估入口。

僅負責載入 stock RF-DETR 權重 + 提供 predict 配接器；COCOeval / 圖表 / 混淆矩陣 /
視覺化全部委由 detection_eval_common.run_eval。（OAVLE 走 valid_oavle.py、YOLO 走
valid_yolo.py，三者共用同一引擎、勿混用。）
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 讓直接執行 / runpy 皆能 import 共用引擎
import detection_eval_common as dec  # noqa: E402
from rfdetr import RFDETRMedium  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate RF-DETR (shared engine)")
    p.add_argument("--dataset_dir", required=True, type=Path,
                   help="資料集根目錄，需有 train/、valid/、(test/) 與 _annotations.coco.json")
    p.add_argument("--checkpoint_path", type=Path,
                   default=Path("outputs/rfdetr/checkpoint_best_total.pth"), help="模型權重路徑")
    p.add_argument("--output_dir", type=Path, default=Path("diagnosis_model/detection/outputs/rfdetr_eval"), help="輸出資料夾")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {args.checkpoint_path}")
        return

    cfg = dec.EvalConfig(dataset_dir=args.dataset_dir, output_dir=args.output_dir, model_name="RF-DETR")
    try:
        cats = dec.get_categories(cfg)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return

    print(f"[INFO] Loading RFDETRMedium: {args.checkpoint_path}")
    model = RFDETRMedium(pretrain_weights=str(args.checkpoint_path), num_classes=len(cats))
    model.optimize_for_inference(compile=False)

    def predict_fn(img, score_thresh):
        pred = model.predict([img], threshold=score_thresh)
        det = pred[0] if isinstance(pred, list) else pred
        return dec.Detections(det.xyxy, det.confidence, det.class_id)

    dec.run_eval(cfg, predict_fn, cats)


if __name__ == "__main__":
    main()
