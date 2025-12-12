from __future__ import annotations
import argparse
import os

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv11 with Ultralytics")
    p.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="資料集根目錄，底下需包含 data.yaml 以及 train/、val/ 等資料夾",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="訓練輸出資料夾（預設為 dataset_dir/outputs/yolov11）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 若未指定 output_dir，則自動用 dataset_dir/outputs/yolov11
    if args.output_dir is None:
        args.output_dir = os.path.join(args.dataset_dir, "outputs", "yolov11")
    os.makedirs(args.output_dir, exist_ok=True)

    # 假設你的 YOLO data.yaml 放在 dataset_dir/data.yaml
    data_cfg = os.path.join(args.dataset_dir, "data.yaml")
    if not os.path.isfile(data_cfg):
        raise FileNotFoundError(f"找不到 data.yaml：{data_cfg}")

    model = YOLO("yolo11s.pt")

    model.train(
        data=data_cfg,
        epochs=100,
        batch=32,
        imgsz=640,
        # 不指定 optimizer / lr0 / lrf / momentum / warmup，讓 YOLO 用預設
        project=args.output_dir,
        name="train",
        exist_ok=True,
    )

    print("訓練完成，檢查輸出：", args.output_dir)


if __name__ == "__main__":
    main()
