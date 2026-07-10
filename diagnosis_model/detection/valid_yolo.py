"""YOLO 偵測評估入口（Ultralytics）。

僅負責載入 YOLO 權重 + 提供 predict 配接器（COCO cat_id 映射由共用引擎的
detections_to_coco 以 cat_ids[class_id] 處理，故此處輸出 0-indexed class 即可）；
COCOeval / 圖表 / 混淆矩陣 / 視覺化全部委由 detection_eval_common.run_eval。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 讓直接執行 / runpy 皆能 import 共用引擎
import detection_eval_common as dec  # noqa: E402
from ultralytics import YOLO  # noqa: E402

IMGSZ = 640  # YOLO 推論解析度


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate YOLO (shared engine)")
    p.add_argument("--dataset_dir", required=True, type=Path,
                   help="COCO 格式資料集根目錄（需有 _annotations.coco.json）")
    p.add_argument("--checkpoint_path", type=Path, default=None, help="YOLO 權重路徑（.pt）")
    p.add_argument("--output_dir", type=Path, default=None, help="輸出資料夾")
    p.add_argument("--model", type=str, default="yolo11s", help="模型名稱（用於命名輸出 / 顯示）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = args.checkpoint_path or (args.dataset_dir / "outputs" / args.model / "train" / "weights" / "best.pt")
    out_dir = args.output_dir or (args.dataset_dir / "outputs" / f"{args.model}_eval")
    if not Path(ckpt).exists():
        print(f"[ERROR] Checkpoint not found: {ckpt}")
        return

    cfg = dec.EvalConfig(dataset_dir=args.dataset_dir, output_dir=Path(out_dir), model_name=args.model)
    try:
        cats = dec.get_categories(cfg)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return

    print(f"[INFO] Loading YOLO model: {ckpt}")
    model = YOLO(str(ckpt))

    def predict_fn(img, score_thresh):
        r = model.predict(img, imgsz=IMGSZ, conf=score_thresh, verbose=False)[0]
        return dec.Detections(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.conf.cpu().numpy(),
            r.boxes.cls.cpu().numpy().astype(int),
        )

    dec.run_eval(cfg, predict_fn, cats)


if __name__ == "__main__":
    main()
